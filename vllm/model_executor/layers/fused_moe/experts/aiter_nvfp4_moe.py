# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 MoE expert on gfx942 via Triton dot_scaled kernel.

Expert weights stay in native NVFP4 (packed e2m1 uint8 + fp8_e4m3fn
per-16 block scales). The kernel uses tl.dot_scaled(e4m3, e2m1) with
fp8 scales, lowering to MFMA bf16 on CDNA3.

This expert is a FusedMoEExpertsMonolithic (routing + GEMM fused) rather
than a FusedMoEExpertsModular. It cannot be folded into rocm_aiter_moe.py
because:
  1. It uses AITER's Triton routing + custom Triton kernel, not the CK/FlyDSL
     fused_moe path that all AiterExperts subclasses share.
  2. It carries its own weight preparation (transpose to K-major layout).
  3. The activation is applied between two separate GEMMs, not fused inside
     a single AITER dispatch.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

_SIGMOID_ROUTING = (RoutingMethodType.DeepSeekV3, RoutingMethodType.MiniMax2)
_SOFTMAX_ROUTING = (RoutingMethodType.Renormalize, RoutingMethodType.RenormalizeNaive)

# AMD gfx942 dot_scaled with e2m1 uses an unnormalized LUT
# {0,1,2,3,4,6,8,12} instead of {0,0.5,1,1.5,2,3,4,6}.
# Compensate by halving the global scale.
_DOT_SCALED_E2M1_CORRECTION = 0.5


def _static_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to fp8 with a single static scale."""
    from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8

    scale = x.abs().amax() / 448.0
    x_fp8 = downcast_to_static_fp8(x, scale)
    return x_fp8, scale


def prepare_nvfp4_moe_weights_for_aiter_nvfp4(
    w: torch.Tensor, w_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose checkpoint [E, N, K//2] -> kernel [E, K//2, N]."""
    w_t = w.transpose(1, 2).contiguous()
    return w_t, w_scale


def aiter_nvfp4_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    routing_data,
    gather_indx: torch.Tensor,
    scatter_indx: torch.Tensor,
    activation: MoEActivation,
    activation_config: ApplyMoEActivationConfig,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_kernel import (
        moe_gemm_a8w4_nvfp4,
    )

    gammas = routing_data.gate_scal
    xq, x_static_scale = _static_fp8_quant(hidden_states)

    w13_global_scale = w13_global_scale * _DOT_SCALED_E2M1_CORRECTION
    w2_global_scale = w2_global_scale * _DOT_SCALED_E2M1_CORRECTION

    # GEMM1: gate+up projection. Activation applied separately after
    # because SWIGLUOAI_UNINTERLEAVE uses [gate|up] layout, not
    # interleaved [g0,u0,g1,u1,...].
    gate_up = moe_gemm_a8w4_nvfp4(
        xq,
        w13,
        w13_scale,
        w13_global_scale,
        x_static_scale=x_static_scale,
        routing_data=routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        out_dtype=hidden_states.dtype,
        apply_swiglu=False,
    )

    intermediate = torch.empty(
        gate_up.shape[0],
        gate_up.shape[1] // 2,
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    apply_moe_activation(
        activation,
        intermediate,
        gate_up,
        activation_config=activation_config,
    )

    aq, a2_static_scale = _static_fp8_quant(intermediate)

    # GEMM2: down projection
    return moe_gemm_a8w4_nvfp4(
        aq,
        w2,
        w2_scale,
        w2_global_scale,
        x_static_scale=a2_static_scale,
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        out_dtype=hidden_states.dtype,
    )


class AiterNvfp4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """Router logits -> AITER fused top-k and sort -> two NVFP4 GEMMs."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not (current_platform.is_rocm() and rocm_aiter_ops.is_enabled()):
            return False
        from vllm.platforms.rocm import get_cdna_version

        return get_cdna_version() == 3  # CDNA3 / gfx942

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if weight_key is None:
            return False
        return weight_key == kNvfp4Static and (
            activation_key is None or activation_key == kNvfp4Dynamic
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in _SIGMOID_ROUTING + _SOFTMAX_ROUTING

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
        routing_replay_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.moe.moe_routing.routing import routing

        routing_method = self.moe_config.routing_method
        if routing_method in _SIGMOID_ROUTING:
            bias = e_score_correction_bias
            routing_data, gather_indx, scatter_indx = routing(
                router_logits.float(),
                self.topk,
                score_mode="sigmoid",
                bias=None if bias is None else bias.float(),
                renorm=True,
                routed_scaling_factor=routed_scaling_factor or 1.0,
                use_grouped_topk=(num_expert_group or 1) > 1,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
            routing_data, gather_indx, scatter_indx = routing(
                router_logits,
                self.topk,
                sm_first=routing_method == RoutingMethodType.RenormalizeNaive,
            )
        return aiter_nvfp4_moe(
            hidden_states,
            w1,
            self.quant_config.w1_scale,
            self.quant_config.g1_alphas,
            w2,
            self.quant_config.w2_scale,
            self.quant_config.g2_alphas,
            routing_data,
            gather_indx,
            scatter_indx,
            activation,
            self.activation_config,
            apply_router_weight_on_input,
        )
