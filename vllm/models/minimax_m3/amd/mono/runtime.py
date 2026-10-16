# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the MiniMax-M3 FFN MonoKernel: one ``MonoFFN`` per TP rank runs
every sparse layer's FFN launch (``ops.ffn``) of a decode step."""

import hashlib
import os
from dataclasses import dataclass, fields
from pathlib import Path

import flydsl.expr as fx
import torch

from vllm.models.minimax_m3.amd.mono.ops import ffn as ffn_ops

MAX_TOKENS = ffn_ops.MAX_TOKENS
_FP8 = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
_BF16, _F32 = (torch.bfloat16,), (torch.float32,)
_SOURCES = [
    *sorted(Path(ffn_ops.__file__).parent.glob("*.py")),
    Path(__file__).parents[3] / "common/mono/device.py",
]


def _source_digest() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _build_once(cache: dict, key, build):
    k = cache.get(key)
    if k is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"MiniMax-M3 mono: {key} first reached inside graph capture"
            )
        k = cache[key] = build()
    return k


def _stream():
    return fx.Stream(torch.cuda.current_stream())


@dataclass(frozen=True)
class FFNWeights:
    """One sparse layer's FFN tensors as vLLM holds them after loading."""

    g_post: torch.Tensor  # [H] bf16
    gate: torch.Tensor  # [E, H] f32
    bias: torch.Tensor  # [E] f32
    w13: torch.Tensor  # [E, 2I, H] fp8, gate rows then up rows
    s13: torch.Tensor  # [E, 2I(, 1)] f32
    w2: torch.Tensor  # [E, H, I] fp8
    s2: torch.Tensor  # [E, H(, 1)] f32
    sgu: torch.Tensor  # [2I, H] fp8
    sgu_s: torch.Tensor  # [2I(, 1)] f32
    sdn: torch.Tensor  # [H, I] fp8
    sdn_s: torch.Tensor  # [H(, 1)] f32

    def check(self) -> list[str]:
        H, inter, E = ffn_ops.H, ffn_ops.I, ffn_ops.E
        want = {
            "g_post": ((H,), _BF16),
            "gate": ((E, H), _F32),
            "bias": ((E,), _F32),
            "w13": ((E, 2 * inter, H), _FP8),
            "s13": ((E, 2 * inter), _F32),
            "w2": ((E, H, inter), _FP8),
            "s2": ((E, H), _F32),
            "sgu": ((2 * inter, H), _FP8),
            "sgu_s": ((2 * inter,), _F32),
            "sdn": ((H, inter), _FP8),
            "sdn_s": ((H,), _F32),
        }
        return _mismatches(self, want)

    @classmethod
    def from_layer(cls, layer) -> "FFNWeights":
        """Views of a loaded ``MiniMaxM3DecoderLayer``'s FFN tensors. The shared
        expert's linears hold their weight as a ``.t()`` view of [N, K] storage."""
        moe = layer.block_sparse_moe
        e = getattr(moe.experts, "routed_experts", moe.experts)
        sh = moe.shared_experts
        w = cls(
            g_post=layer.post_attention_layernorm.weight,
            gate=moe.gate.weight,
            bias=moe.e_score_correction_bias,
            w13=e.w13_weight,
            s13=e.w13_weight_scale,
            w2=e.w2_weight,
            s2=e.w2_weight_scale,
            sgu=sh.gate_up_proj.weight.t(),
            sgu_s=sh.gate_up_proj.weight_scale,
            sdn=sh.down_proj.weight.t(),
            sdn_s=sh.down_proj.weight_scale,
        )
        return _checked(w, layer)

    def ptrs(self) -> list[int]:
        return [getattr(self, f.name).data_ptr() for f in fields(self)]


class MonoFFN:
    """The FFN MonoKernel of one TP rank (``group``: the TP group's CPU group)."""

    def __init__(
        self, tp: int, rank: int, group, device: torch.device, timeline: bool = False
    ):
        from vllm.models.common.mono.peer import PeerBuffer

        _ensure_writable_flydsl_cache()
        self.tp, self.rank = tp, rank
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        self.scratch = torch.zeros(
            ffn_ops.SCRATCH_BYTES, dtype=torch.uint8, device=device
        )
        self.ctr = torch.zeros(self.grid, dtype=torch.int32, device=device)
        self.peer = PeerBuffer(ffn_ops.PEER_BYTES, group, rank, tp, device)
        self.timeline = timeline
        self.tl = torch.zeros(
            self.grid, ffn_ops.STAMPS, dtype=torch.int64, device=device
        )
        self._key = _source_digest()
        self._kernels: dict[int, object] = {}

    def kernel(self, tokens: int):
        return _build_once(
            self._kernels,
            tokens,
            lambda: ffn_ops.build_ffn(
                tokens, self.grid, self.tp, self._key, self.timeline
            ),
        )

    def run(
        self,
        w: FFNWeights,
        partial: torch.Tensor,
        residual: torch.Tensor,
        seq_lens: torch.Tensor,
        q_len: int = 1,
        out: torch.Tensor | None = None,
        h_mid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (TP-reduced FFN output, new residual) of one sparse layer.

        ``seq_lens``: the step's per-request lengths, ``q_len`` rows a request;
        a 0 marks a CUDA-graph padding request, whose rows' experts are skipped.
        """
        m = partial.shape[0]
        assert (
            1 <= m <= MAX_TOKENS
            and partial.is_contiguous()
            and residual.is_contiguous()
            and seq_lens.dtype == torch.int32
        )
        out = torch.empty_like(partial) if out is None else out
        h_mid = torch.empty_like(residual) if h_mid is None else h_mid
        self.kernel(m)(
            partial.data_ptr(),
            residual.data_ptr(),
            h_mid.data_ptr(),
            out.data_ptr(),
            *w.ptrs(),
            self.scratch.data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.ctr.data_ptr(),
            self.rank,
            self.tl.data_ptr(),
            seq_lens.data_ptr(),
            q_len,
            stream=_stream(),
        )
        return out, h_mid

    def close(self) -> None:
        self.peer.close()


def _checked(w, layer):
    if bad := w.check():
        raise ValueError(f"MiniMax-M3 mono, layer {layer.layer_id}: {bad}")
    return w


def _mismatches(obj, want: dict) -> list[str]:
    bad = []
    for name, (shape, dtypes) in want.items():
        t = getattr(obj, name)
        got = tuple(s for i, s in enumerate(t.shape) if i < len(shape) or s != 1)
        if got != shape or t.dtype not in dtypes or not t.is_contiguous():
            bad.append(
                f"{name}: {tuple(t.shape)} {t.dtype} contiguous={t.is_contiguous()}"
            )
    return bad


def _ensure_writable_flydsl_cache() -> None:
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)
