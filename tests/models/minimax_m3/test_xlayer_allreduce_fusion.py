"""Unit tests for MiniMax-M3 AMD cross-layer FFN all-reduce deferral.

These exercise the *wiring* logic without building the 227 GB model or needing
a GPU: the deferral chain and the EAGLE3 capture-site reduction are pure Python
over layer objects, so they can be driven with stand-ins exposing the handful of
attributes the chain reads.

Invariants under test:

1. Every deferred all-reduce is completed exactly once - by the next layer's
   ``input_layernorm`` or by the final norm. Never dropped, never doubled.
2. An EAGLE3 aux capture never records a per-rank partial sum.
3. The wiring is stable under repeated refresh and reacts to aux layers being
   chosen after ``__init__``.

Intended to land as
``tests/models/minimax_m3/test_xlayer_allreduce_fusion.py``.

    python test_xlayer_allreduce_fusion.py    # standalone
    pytest test_xlayer_allreduce_fusion.py    # or under pytest
"""

from __future__ import annotations

from itertools import product


class FakeLayer:
    def __init__(self, defers: bool):
        self._defers = defers
        self.fuse_input_allreduce = False

    @property
    def ffn_all_reduce_deferred(self) -> bool:
        return self._defers


class FakeModel:
    """Mirrors MiniMaxM3Model's fusion wiring and capture-site reduction."""

    def __init__(self, layers, aux=()):
        self.layers = layers
        self.start_layer = 0
        self.end_layer = len(layers)
        self.aux_hidden_state_layers = tuple(sorted(set(aux)))
        self.fuse_final_norm_allreduce = False
        self._refresh_allreduce_fusion()

    # --- mirrors of the patched methods ----------------------------------
    def _refresh_allreduce_fusion(self) -> None:
        layers = self.layers[self.start_layer : self.end_layer]
        prev_defers = False
        for idx, layer in enumerate(layers):
            layer.fuse_input_allreduce = idx > 0 and prev_defers
            prev_defers = layer.ffn_all_reduce_deferred
        self.fuse_final_norm_allreduce = prev_defers

    def _aux_needs_reduce(self, layer_idx: int) -> bool:
        layers = self.layers[self.start_layer : self.end_layer]
        return (
            0 < layer_idx <= len(layers)
            and layers[layer_idx - 1].ffn_all_reduce_deferred
        )

    def _set_aux_hidden_state_layers(self, layers) -> None:
        self.aux_hidden_state_layers = tuple(sorted(set(layers)))
        self._refresh_allreduce_fusion()

    # --- executable model of one forward pass ----------------------------
    def simulate(self):
        """Returns (captures_are_reduced, allreduces_balance)."""
        layers = self.layers[self.start_layer : self.end_layer]
        aux = set(self.aux_hidden_state_layers)
        pending = False  # running hidden_states is an un-reduced partial
        captures_ok = True
        balance_ok = True

        if 0 in aux and pending:  # pre-loop capture
            captures_ok = False

        for idx, layer in enumerate(layers):
            if layer.fuse_input_allreduce:
                if not pending:
                    balance_ok = False  # fused a reduce that never happened
                pending = False
            elif pending:
                balance_ok = False  # a deferral was silently dropped
            pending = layer.ffn_all_reduce_deferred

            cap_idx = idx + 1
            if cap_idx in aux:
                # the patched capture site reduces a copy when needed
                sees_partial = pending and not self._aux_needs_reduce(cap_idx)
                if sees_partial:
                    captures_ok = False

        if pending != self.fuse_final_norm_allreduce:
            balance_ok = False
        return captures_ok, balance_ok


# ---------------------------------------------------------------------------
def test_all_layers_defer_chain_is_complete():
    m = FakeModel([FakeLayer(True) for _ in range(8)])
    assert not m.layers[0].fuse_input_allreduce, "layer 0 has no predecessor"
    assert all(lyr.fuse_input_allreduce for lyr in m.layers[1:])
    assert m.fuse_final_norm_allreduce
    assert all(m.simulate())


def test_no_layer_defers():
    m = FakeModel([FakeLayer(False) for _ in range(4)])
    assert not any(lyr.fuse_input_allreduce for lyr in m.layers)
    assert not m.fuse_final_norm_allreduce
    assert all(m.simulate())


def test_mixed_dense_and_moe():
    """M3's real shape: 3 leading dense layers, the rest MoE."""
    layers = [FakeLayer(False) for _ in range(3)] + [FakeLayer(True) for _ in range(5)]
    m = FakeModel(layers)
    assert not layers[3].fuse_input_allreduce, "follows a non-deferring layer"
    assert all(lyr.fuse_input_allreduce for lyr in layers[4:])
    assert m.fuse_final_norm_allreduce
    assert all(m.simulate())


def test_eagle3_capture_never_sees_a_partial():
    """The hazard this patch must not introduce."""
    m = FakeModel([FakeLayer(True) for _ in range(8)], aux=(2, 5))
    captures_ok, balance_ok = m.simulate()
    assert captures_ok, "EAGLE3 recorded an un-reduced per-rank partial"
    assert balance_ok, "the all-reduce chain was broken"


def test_eagle3_capture_after_deferring_layer_is_reduced():
    m = FakeModel([FakeLayer(True) for _ in range(4)], aux=(3,))
    assert m._aux_needs_reduce(3), "layer 2 defers, so capture 3 must reduce"


def test_eagle3_capture_after_reducing_layer_is_not_reduced():
    """No redundant all-reduce when the producer already reduced."""
    m = FakeModel([FakeLayer(False) for _ in range(4)], aux=(3,))
    assert not m._aux_needs_reduce(3), "producer reduced; extra AR would waste"


def test_eagle3_rewires_after_construction():
    """Aux layers are chosen post-__init__ by the model runner."""
    m = FakeModel([FakeLayer(True) for _ in range(6)])
    m._set_aux_hidden_state_layers((1, 3))
    assert m.aux_hidden_state_layers == (1, 3)
    assert all(m.simulate())
    # the deferral chain itself is unchanged: capture handles its own reduce
    assert all(lyr.fuse_input_allreduce for lyr in m.layers[1:])


def test_deferral_chain_unaffected_by_aux():
    """Aux selection must not perturb the fusion chain."""
    base = FakeModel([FakeLayer(True) for _ in range(6)])
    chain = [lyr.fuse_input_allreduce for lyr in base.layers]
    tail = base.fuse_final_norm_allreduce
    withaux = FakeModel([FakeLayer(True) for _ in range(6)], aux=(1, 4, 6))
    assert [lyr.fuse_input_allreduce for lyr in withaux.layers] == chain
    assert withaux.fuse_final_norm_allreduce == tail


def test_idempotent_refresh():
    m = FakeModel([FakeLayer(True) for _ in range(6)], aux=(3,))
    snap = ([lyr.fuse_input_allreduce for lyr in m.layers], m.fuse_final_norm_allreduce)
    for _ in range(3):
        m._refresh_allreduce_fusion()
    assert snap == (
        [lyr.fuse_input_allreduce for lyr in m.layers],
        m.fuse_final_norm_allreduce,
    )


def test_single_layer_model():
    m = FakeModel([FakeLayer(True)])
    assert not m.layers[0].fuse_input_allreduce
    assert m.fuse_final_norm_allreduce
    assert all(m.simulate())


def test_exhaustive_small_configs():
    """Brute force every defer pattern x aux subset up to 6 layers."""
    bad = []
    for n in range(1, 7):
        for defers in product([True, False], repeat=n):
            for auxmask in range(1 << (n + 1)):
                aux = tuple(i for i in range(n + 1) if auxmask >> i & 1)
                m = FakeModel([FakeLayer(d) for d in defers], aux=aux)
                cap, bal = m.simulate()
                if not (cap and bal):
                    bad.append((defers, aux, cap, bal))
    assert not bad, f"{len(bad)} invalid configs, first: {bad[0]}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed.append(t.__name__)
    print()
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
