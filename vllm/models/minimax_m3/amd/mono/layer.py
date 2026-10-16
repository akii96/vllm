# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FFN MonoKernel in the sparse decoder layers (``--monokernel``).

A pure decode step of at most ``MAX_TOKENS`` rows (speculative verify included)
runs each sparse layer's FFN half as one launch a rank: o_proj's all-reduce,
the post-attention norm, router, routed and shared FP8 experts and the MoE
all-reduce, after vLLM's attention. Every other step runs the layer unchanged.
The decision reads only metadata every TP rank holds alike, since the kernels
wait on each other's pushes.
"""

import functools
from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.platforms import current_platform

if TYPE_CHECKING:  # the runtime imports FlyDSL, which only a --monokernel run needs
    from .runtime import FFNWeights

logger = init_logger(__name__)

TP = 4


def _strategy(obj, field: str = "weight_quant") -> str | None:
    s = getattr(getattr(obj, field, None), "strategy", None)
    return getattr(s, "value", s)


def refusals(layer, vllm_config) -> list[str]:
    """Every reason the deployment cannot run this layer's MonoKernel."""
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
            (
                pc.enable_expert_parallel or pc.enable_eplb,
                "no expert parallelism / EPLB",
            ),
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
    return why


@functools.cache
def _runner(device: torch.device):
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

    from .runtime import MonoFFN

    logger.info("MiniMax-M3 MonoKernels active (decode steps)")
    return MonoFFN(
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
        get_tp_group().cpu_group,
        device,
    )


class MonoLayer:
    """One sparse decoder layer's MonoKernel path."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self._w: FFNWeights | None = None

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
                from .runtime import MAX_TOKENS
            except ImportError as err:
                why.append(f"needs FlyDSL and AITER ({err})")
        if why:
            raise ValueError("--monokernel for MiniMax-M3: " + "; ".join(why) + ".")
        return cls(MAX_TOKENS)

    def _decode(self, layer, rows: int) -> MiniMaxM3SparseMetadata | None:
        """This step's metadata if it is one the kernel takes: a pure decode of
        at most ``max_tokens`` rows, eager or a full CUDA graph."""
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
            not isinstance(m, MiniMaxM3SparseMetadata)
            or m.num_prefills != 0
            or m.num_decodes == 0
            or m.decode is None
            or m.num_decode_tokens != rows
        ):
            return None
        return m

    def ffn(self, layer, partial, residual):
        """(reduced FFN output, new residual) from vLLM's attention's unreduced
        o_proj output, or None for vLLM's path."""
        m = self._decode(layer, partial.shape[0])
        if m is None:
            return None
        if self._w is None:
            from .runtime import FFNWeights

            self._w = FFNWeights.from_layer(layer)
        d = m.decode
        assert d is not None  # _decode checked it
        return _runner(partial.device).run(
            self._w, partial, residual, d.seq_lens, d.decode_query_len
        )
