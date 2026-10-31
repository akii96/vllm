# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill index block-score kernel: split-K default vs torch reference and the
legacy kernel (VLLM_MINIMAX_M3_INDEX_SCORE_LEGACY=1)."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MiniMax-M3 AMD index kernels need ROCm", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops import index_topk as IT  # noqa: E402

BLK = 128
FP8 = current_platform.fp8_dtype()
DEV = "cuda"


def _case(q_lens, ctx_lens, heads, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    batch = len(q_lens)
    max_blocks = max((c + BLK - 1) // BLK for c in ctx_lens)
    total_pages = batch * max_blocks + 3
    perm = torch.randperm(total_pages, generator=g, device=DEV)[: batch * max_blocks]
    bt = perm.view(batch, max_blocks).to(torch.int32).contiguous()
    k = (torch.randn(total_pages, BLK, 128, generator=g, device=DEV) * 0.5).to(FP8)
    rows = sum(q_lens)
    q = (torch.randn(rows, heads, 128, generator=g, device=DEV) * 0.5).to(FP8)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32,
                      device=DEV)
    seq = torch.tensor(ctx_lens, dtype=torch.int32, device=DEV)
    pre = torch.tensor([c - ql for c, ql in zip(ctx_lens, q_lens)], dtype=torch.int32,
                       device=DEV)
    return q, k, bt, cu, seq, pre, max(q_lens), max(ctx_lens)


def _reference(q, k, bt, q_lens, ctx_lens):
    """fp32 torch block-max scores and the causal validity mask per row."""
    heads = q.shape[1]
    out, valid, r0 = [], [], 0
    for b, (ql, ctx) in enumerate(zip(q_lens, ctx_lens)):
        nb = (ctx + BLK - 1) // BLK
        kk = k[bt[b, :nb].long()].float().reshape(nb * BLK, 128)
        qq = q[r0 : r0 + ql].float()  # [ql, H, D]
        s = torch.einsum("qhd,kd->hqk", qq, kk)
        qpos = torch.arange(ctx - ql, ctx, device=DEV)
        kpos = torch.arange(nb * BLK, device=DEV)
        s = s.masked_fill(~(qpos[:, None] >= kpos[None, :])[None], float("-inf"))
        s = s.view(heads, ql, nb, BLK).amax(-1)
        out.append(s)
        valid.append(torch.arange(nb, device=DEV)[None, :] <= (qpos // BLK)[:, None])
        r0 += ql
    return out, valid


def _run(monkeypatch, legacy, *args):
    monkeypatch.setenv("VLLM_MINIMAX_M3_INDEX_SCORE_LEGACY", "1" if legacy else "0")
    return IT.minimax_m3_index_score(*args)


CASES = [
    ([16384], [16384], 1),
    ([1000], [70000], 1),
    ([4096], [262144], 2),
    ([257, 3000, 1, 4500], [257, 9000, 20000, 4500], 1),
    ([700, 129, 2048], [5000, 129, 40000], 2),
    ([3, 64, 255], [131, 64, 1000], 4),
]


@pytest.mark.parametrize("q_lens,ctx_lens,heads", CASES)
def test_index_score_split_matches_reference_and_legacy(monkeypatch, q_lens, ctx_lens,
                                                         heads):
    q, k, bt, cu, seq, pre, mq, ms = _case(q_lens, ctx_lens, heads)
    args = (q, k, bt, cu, seq, pre, mq, ms, heads)
    new = _run(monkeypatch, False, *args)
    old = _run(monkeypatch, True, *args)
    refs, valids = _reference(q, k, bt, q_lens, ctx_lens)
    r0 = 0
    for b, ql in enumerate(q_lens):
        ref, valid = refs[b], valids[b]
        nb = ref.shape[-1]
        n = new[:, r0 : r0 + ql, :nb]
        o = old[:, r0 : r0 + ql, :nb]
        vm = valid[None].expand_as(ref)
        # Every causally visible block is finite and bit-identical to legacy.
        assert torch.isfinite(n[vm]).all()
        torch.testing.assert_close(n[vm], o[vm], rtol=0, atol=0)
        scale = ref[vm].abs().max().clamp_min(1.0)
        torch.testing.assert_close(n[vm], ref[vm], rtol=1e-6, atol=1e-5 * scale.item())
        r0 += ql


def test_index_score_split_topk_identical(monkeypatch):
    q_lens, ctx_lens, heads = [2048, 512], [65536, 9000], 2
    q, k, bt, cu, seq, pre, mq, ms = _case(q_lens, ctx_lens, heads, seed=3)
    args = (q, k, bt, cu, seq, pre, mq, ms, heads)
    topk = []
    for legacy in (True, False):
        s = _run(monkeypatch, legacy, *args)
        topk.append(IT.minimax_m3_index_topk(s, cu, pre, mq, 16, 0, 1))
    assert torch.equal(topk[0], topk[1])
