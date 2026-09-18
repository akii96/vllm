# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 MoE oracle backend selection on mi300x/mi325x (GFX942).

The oracle is stubbed rather than run on hardware, so these cover the
selection order itself on any platform.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
)
from vllm.platforms import current_platform


def _make_deepseek_v4_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=384,
        experts_per_token=6,
        hidden_dim=5120,
        intermediate_size=2304,
        num_local_experts=384,
        num_logical_experts=384,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_backend=moe_backend,
        swiglu_limit=10.0,
    )


@pytest.mark.parametrize(
    "unsupported,expected_backend",
    [
        # CK is available: it stays the first choice.
        ((), Mxfp4MoeBackend.AITER_MXFP4_BF16),
        # CK is gfx950-only, so on gfx942 the Triton W4A16 variant must be
        # reached instead of falling through to Triton-unfused.
        (
            (Mxfp4MoeBackend.AITER_MXFP4_BF16,),
            Mxfp4MoeBackend.AITER_TRITON_MXFP4_BF16,
        ),
        # Neither AITER variant available: Triton-unfused is the fallback.
        (
            (
                Mxfp4MoeBackend.AITER_MXFP4_BF16,
                Mxfp4MoeBackend.AITER_TRITON_MXFP4_BF16,
            ),
            Mxfp4MoeBackend.TRITON_UNFUSED,
        ),
    ],
)
def test_rocm_deepseek_v4_backend_priority(unsupported, expected_backend, monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle

    class SupportedExperts:
        @staticmethod
        def is_supported_config(*args, **kwargs):
            return True, None

    class UnsupportedExperts:
        @staticmethod
        def is_supported_config(*args, **kwargs):
            return False, "stubbed unsupported"

    config = _make_deepseek_v4_moe_config()
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(mxfp4_oracle.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [
            UnsupportedExperts if backend in unsupported else SupportedExperts
        ],
    )

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == expected_backend
    assert experts_cls is SupportedExperts
