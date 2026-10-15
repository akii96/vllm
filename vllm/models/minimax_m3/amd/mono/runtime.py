# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the MiniMax-M3 MonoKernels: one ``MonoAttn`` / ``MonoFFN`` per TP
rank runs every sparse layer's attention / FFN launch (``ops``) of a decode step."""

import hashlib
import os
from dataclasses import dataclass, fields
from pathlib import Path

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
            raise RuntimeError(f"MiniMax-M3 mono: {key} first reached inside graph capture")
        k = cache[key] = build()
    return k


def _stream():
    import flydsl.expr as fx

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
        H, I, E = ffn_ops.H, ffn_ops.I, ffn_ops.E
        want = {
            "g_post": ((H,), _BF16),
            "gate": ((E, H), _F32),
            "bias": ((E,), _F32),
            "w13": ((E, 2 * I, H), _FP8),
            "s13": ((E, 2 * I), _F32),
            "w2": ((E, H, I), _FP8),
            "s2": ((E, H), _F32),
            "sgu": ((2 * I, H), _FP8),
            "sgu_s": ((2 * I,), _F32),
            "sdn": ((H, I), _FP8),
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
        self.tp, self.rank, self.device = tp, rank, device
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        self.scratch = torch.zeros(ffn_ops.SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self.ctr = torch.zeros(self.grid, dtype=torch.int32, device=device)
        self.peer = PeerBuffer(ffn_ops.PEER_BYTES, group, rank, tp, device)
        self.timeline = timeline
        self.tl = torch.zeros(self.grid, ffn_ops.STAMPS, dtype=torch.int64, device=device)
        self._key = _source_digest()
        self._kernels: dict[int, object] = {}

    def kernel(self, tokens: int):
        return _build_once(
            self._kernels,
            tokens,
            lambda: ffn_ops.build_ffn(tokens, self.grid, self.tp, self._key, self.timeline),
        )

    def run(
        self,
        w: FFNWeights,
        partial: torch.Tensor,
        residual: torch.Tensor,
        out: torch.Tensor | None = None,
        h_mid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (TP-reduced FFN output, new residual) of one sparse layer."""
        m = partial.shape[0]
        assert 1 <= m <= MAX_TOKENS and partial.is_contiguous() and residual.is_contiguous()
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
            stream=_stream(),
        )
        return out, h_mid

    def close(self) -> None:
        self.peer.close()


@dataclass(frozen=True)
class AttnWeights:
    """One sparse layer's attention tensors as vLLM holds them after loading."""

    g_in: torch.Tensor  # [H] bf16
    w_qkv: torch.Tensor  # [2560, H] fp8: q | k | v | index_q | index_k rows
    s_qkv: torch.Tensor  # [2560(, 1)] f32
    g_q: torch.Tensor  # [128] bf16
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor  # [max_pos, 64] bf16: cos | sin
    w_o: torch.Tensor  # [H, 2048] fp8
    s_o: torch.Tensor  # [H(, 1)] f32
    k_scale: torch.Tensor  # [1] f32
    v_scale: torch.Tensor  # [1] f32
    sm_scale: float
    eps: float

    def check(self) -> list[str]:
        from vllm.models.minimax_m3.amd.mono.ops import attn as a

        want = {
            "g_in": ((a.H,), _BF16),
            "w_qkv": ((a.QKV_ROWS, a.H), _FP8),
            "s_qkv": ((a.QKV_ROWS,), _F32),
            "g_q": ((a.HEAD_DIM,), _BF16),
            "g_k": ((a.HEAD_DIM,), _BF16),
            "g_iq": ((a.HEAD_DIM,), _BF16),
            "g_ik": ((a.HEAD_DIM,), _BF16),
            "w_o": ((a.H, a.O_K), _FP8),
            "s_o": ((a.H,), _F32),
            "k_scale": ((1,), _F32),
            "v_scale": ((1,), _F32),
        }
        bad = _mismatches(self, want)
        cs = self.cos_sin
        if cs.dtype != torch.bfloat16 or cs.shape[-1] != a.ROTARY or not cs.is_contiguous():
            bad.append(f"cos_sin: {tuple(cs.shape)} {cs.dtype}")
        return bad

    @classmethod
    def from_layer(cls, layer) -> "AttnWeights":
        """Views of a loaded ``MiniMaxM3DecoderLayer``'s attention tensors. FP8
        linears hold their weight as a ``.t()`` view of [N, K] storage."""
        a = layer.self_attn
        one = torch.ones(1, dtype=torch.float32, device=a.qkv_proj.weight.device)
        w = cls(
            g_in=layer.input_layernorm.weight,
            w_qkv=a.qkv_proj.weight.t(),
            s_qkv=a.qkv_proj.weight_scale,
            g_q=a.q_norm.weight,
            g_k=a.k_norm.weight,
            g_iq=a.index_q_norm.weight,
            g_ik=a.index_k_norm.weight,
            cos_sin=a.rotary_emb.cos_sin_cache,
            w_o=a.o_proj.weight.t(),
            s_o=a.o_proj.weight_scale,
            k_scale=getattr(a, "_k_scale", one).float().reshape(1),
            v_scale=getattr(a, "_v_scale", one).float().reshape(1),
            sm_scale=a.scaling,
            eps=a.q_norm.variance_epsilon,
        )
        return _checked(w, layer)

    def ptrs(self) -> list[int]:
        return [
            getattr(self, n).data_ptr()
            for n in ("g_in", "w_qkv", "s_qkv", "g_q", "g_k", "g_iq", "g_ik", "cos_sin", "w_o", "s_o")
        ]


class MonoAttn:
    """The attention MonoKernel of one TP rank: no collectives (the FFN launch
    reduces its output)."""

    def __init__(self, device: torch.device, max_blocks: int, timeline: bool = False):
        from vllm.models.minimax_m3.amd.mono.ops import attn as attn_ops

        _ensure_writable_flydsl_cache()
        self.ops = attn_ops
        self.device, self.max_blocks, self.timeline = device, max_blocks, timeline
        self.grid = torch.cuda.get_device_properties(device).multi_processor_count
        self.scratch = torch.zeros(attn_ops.SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self.ctr = torch.zeros(self.grid, dtype=torch.int32, device=device)
        self.tl = torch.zeros(self.grid, attn_ops.STAMPS, dtype=torch.int64, device=device)
        self._key = _source_digest()
        self._kernels: dict = {}

    def kernel(self, tokens: int, q_len: int, w: AttnWeights, fp8_index: bool):
        return _build_once(
            self._kernels,
            (tokens, q_len, fp8_index),
            lambda: self.ops.build_attn(
                tokens, self.grid, self._key, q_len, w.sm_scale, w.eps,
                self.max_blocks, self.timeline, fp8_index,
            ),  # fmt: skip
        )

    def run(
        self,
        w: AttnWeights,
        x: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        q_len: int = 1,
        add_residual: bool = True,
        partial: torch.Tensor | None = None,
        res_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (this rank's unreduced o_proj output, the new residual)."""
        m = x.shape[0]
        assert 1 <= m <= MAX_TOKENS and m % q_len == 0
        assert positions.dtype == slots.dtype == torch.int64
        assert block_table.dtype == seq_lens.dtype == torch.int32
        assert block_table.stride(1) == 1 and kv_cache.stride(-1) == 1
        assert kv_cache.element_size() == 1 and kv_cache.shape[-2:] == (128, 256), (
            f"KV cache {tuple(kv_cache.shape)} {kv_cache.dtype}"
        )
        assert index_cache.dtype in (torch.bfloat16, current_fp8_dtype()), (
            f"index cache {index_cache.dtype}"
        )
        partial = torch.empty_like(x) if partial is None else partial
        res_out = torch.empty_like(x) if res_out is None else res_out
        self.kernel(m, q_len, w, index_cache.dtype != torch.bfloat16)(
            x.data_ptr(), residual.data_ptr(), res_out.data_ptr(), partial.data_ptr(),
            *w.ptrs(),
            kv_cache.data_ptr(), kv_cache.stride(0) * kv_cache.element_size(),
            index_cache.data_ptr(), index_cache.stride(0) * index_cache.element_size(),
            w.k_scale.data_ptr(), w.v_scale.data_ptr(),
            positions.data_ptr(), slots.data_ptr(),
            block_table.data_ptr(), block_table.stride(0), seq_lens.data_ptr(),
            self.scratch.data_ptr(), self.ctr.data_ptr(), int(add_residual),
            self.tl.data_ptr(),
            stream=_stream(),
        )  # fmt: skip
        return partial, res_out


def current_fp8_dtype() -> torch.dtype:
    """The FP8 flavour the device converts natively (e4m3fnuz on gfx942)."""
    from vllm.platforms import current_platform

    return current_platform.fp8_dtype()


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
            bad.append(f"{name}: {tuple(t.shape)} {t.dtype} contiguous={t.is_contiguous()}")
    return bad


def _ensure_writable_flydsl_cache() -> None:
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)
