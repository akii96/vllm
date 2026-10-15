# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MonoKernels in the sparse decoder layers (``--monokernel``).

A pure decode step of at most ``MAX_TOKENS`` rows (speculative verify included)
runs each sparse layer as two launches a rank: the attention launch (input
norm, qkv, head norm / RoPE / cache inserts, indexer scores and top-k, sparse
attention, o_proj) and the FFN launch (o_proj's all-reduce, post-attention norm,
router, routed and shared FP8 experts, the MoE all-reduce). Every other step
runs the layer unchanged. The decision reads only metadata every TP rank holds
alike, since the kernels wait on each other's pushes.
"""

import functools
import os

import torch

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

TP = 4
SPARSE_BLOCK = 128
# Whole layer (attention launch too) or FFN layer mode (vLLM's attention). FFN
# mode is the default: the attention launch only wins at low concurrency.
WHOLE_LAYER = os.environ.get("VLLM_MINIMAX_M3_MONO_WHOLE_LAYER", "0") == "1"


def _strategy(obj, field: str = "weight_quant") -> str | None:
    s = getattr(getattr(obj, field, None), "strategy", None)
    return getattr(s, "value", s)


def refusals(layer, vllm_config) -> list[str]:
    """Every reason the deployment cannot run this layer's MonoKernels."""
    pc = vllm_config.parallel_config
    mode = vllm_config.compilation_config.cudagraph_mode
    why = [
        reason
        for bad, reason in (
            (
                not (current_platform.is_rocm() and current_platform.is_fp8_fnuz()),
                "needs ROCm gfx94x (FP8 fnuz)",
            ),
            (pc.tensor_parallel_size != TP, f"needs tensor parallel size {TP}"),
            (pc.pipeline_parallel_size > 1, "no pipeline parallelism"),
            (pc.data_parallel_size > 1, "no data parallelism"),
            (pc.enable_expert_parallel or pc.enable_eplb, "no expert parallelism / EPLB"),
            (vllm_config.lora_config is not None, "no LoRA"),
            (
                mode == CUDAGraphMode.PIECEWISE or envs.VLLM_USE_BREAKABLE_CUDAGRAPH,
                "no piecewise-only / breakable CUDA graphs",
            ),
        )
        if bad
    ]
    moe = layer.block_sparse_moe
    experts = getattr(moe.experts, "routed_experts", moe.experts)
    qm = getattr(experts, "quant_method", None)
    backend = getattr(getattr(qm, "fp8_backend", None), "name", None)
    if (
        backend != "TRITON"
        or _strategy(qm) != "channel"
        or _strategy(qm, "input_quant") != "token"
    ):
        why.append(
            "needs per-channel FP8 experts on the TRITON MoE backend "
            f"(got {type(qm).__name__}, {backend})"
        )
    sh = moe.shared_experts
    if sh is None:
        why.append("needs the shared expert unfused (shared-expert fusion is on)")
    elif any(
        _strategy(getattr(p, "scheme", None)) != "channel"
        for p in (sh.gate_up_proj, sh.down_proj)
    ):
        why.append("needs a per-channel FP8 shared expert")
    attn = layer.self_attn
    if getattr(attn, "use_aiter_sparse_pa", False) or attn.kv_cache_dtype != "fp8":
        why.append("needs the Triton sparse attention with an fp8 KV cache")
    if getattr(attn, "skip_index_topk", False):
        why.append("no cross-layer index sharing (index_topk_freq)")
    return why


@functools.cache
def _runners(device: torch.device, max_blocks: int):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "--monokernel first reached inside CUDA graph capture, before an eager "
            "decode step"
        )
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    from .runtime import MonoAttn, MonoFFN

    logger.info("MiniMax-M3 MonoKernels active (decode steps)")
    ffn = MonoFFN(
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
        get_tp_group().cpu_group,
        device,
    )
    return MonoAttn(device, max_blocks), ffn


class MonoLayer:
    """One sparse decoder layer's MonoKernel path."""

    def __init__(self, max_tokens: int, max_blocks: int):
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks
        self._ffn_w = None
        self._attn_w = None

    @classmethod
    def create(cls, layer, vllm_config) -> "MonoLayer | None":
        """None when the switch is off or the layer is not a backbone sparse MoE
        layer; raises with every reason when the deployment is unsupported."""
        config = vllm_config.model_config.hf_text_config
        if (
            not vllm_config.kernel_config.monokernel
            or not layer.is_moe_layer
            or layer.layer_id >= config.num_hidden_layers
        ):
            return None
        why = refusals(layer, vllm_config)
        if not why:
            try:
                from .ops.attn import MAX_BLOCKS
                from .runtime import MAX_TOKENS
            except ImportError as err:
                why.append(f"needs FlyDSL and AITER ({err})")
            else:
                max_blocks = -(-vllm_config.model_config.max_model_len // SPARSE_BLOCK)
                if max_blocks > MAX_BLOCKS:
                    why.append(f"max_model_len over {MAX_BLOCKS * SPARSE_BLOCK}")
        if why:
            raise ValueError("--monokernel for MiniMax-M3: " + "; ".join(why) + ".")
        return cls(MAX_TOKENS, max_blocks)

    def _decode(self, layer, rows: int):
        """This step's sparse metadata when the kernels take it, else None."""
        if not 1 <= rows <= self.max_tokens or not is_forward_context_available():
            return None
        fc = get_forward_context()
        md = fc.attn_metadata
        if fc.cudagraph_runtime_mode not in (
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL,
        ) or not isinstance(md, dict):
            return None
        m = md.get(layer.self_attn.layer_name)
        if (
            m is None
            or m.num_prefills != 0
            or m.num_decodes == 0
            or m.decode is None
            or m.num_decode_tokens != rows
        ):
            return None
        return m

    def _weights(self, layer):
        if self._ffn_w is None:
            from .runtime import AttnWeights, FFNWeights

            self._ffn_w = FFNWeights.from_layer(layer)
            self._attn_w = AttnWeights.from_layer(layer) if WHOLE_LAYER else None
        return self._attn_w, self._ffn_w

    def __call__(self, layer, positions, x, residual):
        """The whole layer -> (reduced FFN output, new residual), or None when this
        step takes vLLM's path or (FFN layer mode) runs vLLM's attention first."""
        if not WHOLE_LAYER:
            return None
        m = self._decode(layer, x.shape[0])
        if m is None:
            return None
        attn_r, ffn_r = _runners(x.device, self.max_blocks)
        attn_w, ffn_w = self._weights(layer)
        attn = layer.self_attn
        d = m.decode
        slots = get_forward_context().slot_mapping[attn.layer_name]
        first = residual is None
        partial, res = attn_r.run(
            attn_w,
            x,
            x if first else residual,
            positions,
            slots[: x.shape[0]],
            d.block_table,
            d.seq_lens,
            attn.kv_cache,
            attn.indexer.index_cache.kv_cache,
            q_len=d.decode_query_len,
            add_residual=not first,
        )
        return ffn_r.run(ffn_w, partial, res)

    def ffn(self, layer, partial, residual):
        """FFN layer mode: (reduced FFN output, new residual) from vLLM's
        attention's unreduced o_proj output, or None for vLLM's path."""
        m = None if WHOLE_LAYER else self._decode(layer, partial.shape[0])
        if m is None:
            return None
        _, ffn_r = _runners(partial.device, self.max_blocks)
        return ffn_r.run(self._weights(layer)[1], partial, residual)
