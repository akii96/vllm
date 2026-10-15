# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the MiniMax-M3 MonoKernel gating: refusals and step routing."""

from types import SimpleNamespace as NS

import pytest

from vllm.config import CUDAGraphMode
from vllm.models.minimax_m3.amd.mono import layer as mono


def _quant(strategy, inp="token"):
    return NS(weight_quant=NS(strategy=strategy), input_quant=NS(strategy=inp))


def _layer(backend="TRITON", strategy="channel", shared=True, moe=True, layer_id=5,
           kv="fp8", aiter_pa=False, skip_topk=False):  # fmt: skip
    lin = NS(scheme=_quant(strategy))
    qm = NS(fp8_backend=NS(name=backend), **vars(_quant(strategy)))
    return NS(
        layer_id=layer_id,
        is_moe_layer=moe,
        self_attn=NS(
            layer_name="model.layers.5.self_attn.attn",
            kv_cache_dtype=kv,
            use_aiter_sparse_pa=aiter_pa,
            skip_index_topk=skip_topk,
        ),
        block_sparse_moe=NS(
            experts=NS(routed_experts=NS(quant_method=qm)),
            shared_experts=NS(gate_up_proj=lin, down_proj=lin) if shared else None,
        ),
    )


def _config(on=True, tp=4, pp=1, dp=1, ep=False, lora=None, mode=CUDAGraphMode.FULL,
            max_len=262144):  # fmt: skip
    return NS(
        kernel_config=NS(monokernel=on),
        model_config=NS(hf_text_config=NS(num_hidden_layers=60), max_model_len=max_len),
        parallel_config=NS(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
            enable_eplb=False,
        ),
        lora_config=lora,
        compilation_config=NS(cudagraph_mode=mode),
    )


@pytest.fixture
def gfx942(monkeypatch):
    plat = NS(is_rocm=lambda: True, is_fp8_fnuz=lambda: True)
    monkeypatch.setattr(mono, "current_platform", plat)
    monkeypatch.setattr(mono.envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False)


def test_supported_deployment_has_no_refusals(gfx942):
    assert mono.refusals(_layer(), _config()) == []


def test_every_reason_is_reported(gfx942):
    why = mono.refusals(
        _layer(backend="AITER", shared=False, kv="auto", skip_topk=True),
        _config(tp=8, pp=2, dp=2, ep=True, lora=object(), mode=CUDAGraphMode.PIECEWISE),
    )
    assert len(why) == 10, why


def test_create(gfx942):
    assert mono.MonoLayer.create(_layer(), _config(on=False)) is None
    assert mono.MonoLayer.create(_layer(moe=False), _config()) is None
    assert mono.MonoLayer.create(_layer(layer_id=60), _config()) is None
    assert mono.MonoLayer.create(_layer(), _config()).max_blocks == 2048
    with pytest.raises(ValueError, match="tensor parallel size 4"):
        mono.MonoLayer.create(_layer(), _config(tp=2))
    with pytest.raises(ValueError, match="max_model_len"):
        mono.MonoLayer.create(_layer(), _config(max_len=2**20))


@pytest.mark.parametrize(
    "rows, mode, prefills, decodes, dec_tokens, md_is_dict, expect",
    [
        (4, CUDAGraphMode.FULL, 0, 4, 4, True, True),
        (8, CUDAGraphMode.NONE, 0, 2, 8, True, True),  # spec verify: 4 tokens a request
        (17, CUDAGraphMode.FULL, 0, 17, 17, True, False),
        (4, CUDAGraphMode.PIECEWISE, 0, 4, 4, True, False),
        (4, CUDAGraphMode.NONE, 1, 3, 3, True, False),
        (4, CUDAGraphMode.NONE, 0, 0, 0, True, False),
        (4, CUDAGraphMode.NONE, 0, 4, 4, False, False),  # profiling / dummy run
    ],
)
def test_decode_routing(monkeypatch, rows, mode, prefills, decodes, dec_tokens,
                        md_is_dict, expect):  # fmt: skip
    layer = _layer()
    md = NS(num_prefills=prefills, num_decodes=decodes, num_decode_tokens=dec_tokens,
            decode=NS() if decodes else None)  # fmt: skip
    fc = NS(
        cudagraph_runtime_mode=mode,
        attn_metadata={layer.self_attn.layer_name: md} if md_is_dict else None,
    )
    monkeypatch.setattr(mono, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mono, "get_forward_context", lambda: fc)
    assert (mono.MonoLayer(16, 64)._decode(layer, rows) is md) is expect


def test_no_forward_context(monkeypatch):
    monkeypatch.setattr(mono, "is_forward_context_available", lambda: False)
    assert mono.MonoLayer(16, 64)._decode(_layer(), 1) is None
