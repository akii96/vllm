# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual fusion of tensor-parallel all-reduce with the following GemmaRMSNorm.

Under tensor parallelism a ``RowParallelLinear`` (e.g. attention ``o_proj``)
produces a per-rank partial sum that is all-reduced, and the result is then fed
into a ``GemmaRMSNorm`` that adds the residual and normalizes. flashinfer ships a
kernel that fuses all-reduce + residual-add + RMSNorm into a single launch; this
helper drives it directly (no torch.compile pass) for models that run eager.

On ROCm flashinfer is unavailable, so the same fusion is driven through AITER's
``fused_allreduce_rmsnorm`` kernel, whose ``gemma_norm`` flag applies the Gemma
``(1 + weight)`` scale. Without it the decode path runs
``aiter::cross_device_reduce_2stage`` immediately followed by a separate
``_gemma_fused_add_rmsnorm`` launch.

Scope: attention output only, no quantization. When neither fast path is
applicable (TP==1, no flashinfer/NVSwitch, no AITER custom all-reduce,
unsupported dtype, or an oversize batch) it falls back to ``all_reduce`` +
``GemmaRMSNorm``, which is numerically identical to the unfused model path.
"""

import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

MiB = 1024 * 1024

# flashinfer fused all-reduce + RMSNorm is wired as a registered custom op in
# allreduce_rms_fusion; both that op and the workspace helpers only exist when
# flashinfer.comm.allreduce_fusion is importable.
try:
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        flashinfer_trtllm_fused_allreduce_norm,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        flashinfer_comm,
        get_fi_ar_workspace,
    )

    _AR_RESIDUAL_RMS_NORM = (
        flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm
        if flashinfer_comm is not None
        else None
    )
except (ImportError, AttributeError):
    flashinfer_trtllm_fused_allreduce_norm = None  # type: ignore[assignment]
    get_fi_ar_workspace = None  # type: ignore[assignment]
    _AR_RESIDUAL_RMS_NORM = None


_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _max_token_num(tp_size: int, hidden_size: int, dtype: torch.dtype) -> int | None:
    """Workspace token budget for flashinfer fused all-reduce, or None if the
    current world size / device is unsupported. Mirrors ``FlashInferAllReduce``."""
    from vllm.config.compilation import PassConfig

    max_size_mb = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(tp_size)
    if not max_size_mb:
        return None
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(max_size_mb * MiB) // (hidden_size * element_size)


def _can_use_flashinfer(hidden_states: torch.Tensor, tp_size: int) -> tuple[bool, int]:
    """Whether the flashinfer fused path applies; returns (ok, max_token_num)."""
    if (
        flashinfer_trtllm_fused_allreduce_norm is None
        or get_fi_ar_workspace is None
        or _AR_RESIDUAL_RMS_NORM is None
    ):
        return False, 0
    if (
        not hidden_states.is_cuda
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.dtype not in _FI_SUPPORTED_DTYPES
    ):
        return False, 0

    num_tokens, hidden_size = hidden_states.shape
    max_token_num = _max_token_num(tp_size, hidden_size, hidden_states.dtype)
    if max_token_num is None or num_tokens > max_token_num:
        return False, 0

    # Lazily create / fetch the (globally cached) workspace; returns None on
    # GPUs without NVSwitch, in which case we fall back gracefully.
    workspace = get_fi_ar_workspace(
        world_size=tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=hidden_states.dtype,
        group=get_tp_group().cpu_group,
    )
    if workspace is None:
        return False, 0
    return True, max_token_num


def _can_use_aiter(hidden_states: torch.Tensor, weight: torch.Tensor):
    """Return the AITER custom all-reduce to fuse through, or None.

    Every gate here mirrors one the AITER kernel enforces internally; the check
    has to happen in the caller because ``custom_fused_ar_rms`` returns None on
    an ineligible input and the registered op is not allowed to.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return None
    if (
        hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.dtype not in _FI_SUPPORTED_DTYPES
        or weight.numel() != hidden_states.shape[-1]
    ):
        return None

    from vllm._aiter_ops import (
        fused_allreduce_rmsnorm_is_profitable,
        rocm_aiter_ops,
    )

    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    if aiter_ar is None or aiter_ar.disabled:
        return None
    # Pre-0.1.12 AITER specialized the launcher on HIDDEN_DIM and silently
    # no-op'd outside {512, 1024, 2048, 4096}; M3's 6144 is not in that set.
    if not aiter_ar.supports_dynamic_hidden_dim:
        return None
    if not aiter_ar.should_custom_ar(hidden_states):
        return None
    # Only fuse where fusing is measured to beat the unfused fallback at this
    # rank count. Both fused kernels lose to a plain two-stage all-reduce plus a
    # separate norm above a payload that depends on the world size, and an
    # unmeasured world size declines rather than guessing; falling back leaves
    # the caller at parity instead of regressing.
    if not fused_allreduce_rmsnorm_is_profitable(hidden_states):
        return None
    return aiter_ar


def fused_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce ``hidden_states`` + add ``residual`` + GemmaRMSNorm, fused.

    ``hidden_states`` is the per-rank *partial* (un-reduced) output of a
    row-parallel linear; ``norm`` is the GemmaRMSNorm applied right after.
    Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        # No all-reduce needed; identical to the unfused path.
        return norm(hidden_states, residual)

    ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
    if ok:
        norm_out = torch.empty_like(hidden_states)
        # With norm_out provided, the kernel writes the new residual
        # (all_reduce(hidden_states) + residual) into the hidden_states buffer
        # and the normalized result into norm_out, leaving `residual` untouched.
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=hidden_states,
            residual=residual,
            rms_gamma=norm.weight,
            rms_eps=norm.variance_epsilon,
            world_size=tp_size,
            weight_bias=1.0,  # GemmaRMSNorm-style
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
            pattern_code=_AR_RESIDUAL_RMS_NORM,
            norm_out=norm_out,
        )
        return norm_out, hidden_states

    if _can_use_aiter(hidden_states, norm.weight) is not None:
        from vllm._aiter_ops import rocm_aiter_ops

        # Returns (normed, all_reduce(hidden_states) + residual); `residual` is
        # read-only, so this is safe under CUDA-graph capture.
        return rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
            input_=hidden_states,
            residual=residual,
            weight=norm.weight.to(hidden_states.dtype),
            epsilon=norm.variance_epsilon,
            gemma_norm=True,
        )

    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
