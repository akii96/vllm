# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FFN MonoKernel vs a torch reference of vLLM's FFN chain, 4 GPUs.

Random weights by default; MONO_REAL_LAYER=<n> takes sparse layer n's FFN of
EmbeddedLLM/MiniMax-M3-FP8-dynamic straight from the safetensors in the HF cache.

Run: python -m pytest tests/models/minimax_m3/amd_mono/test_ffn_mono.py
or   python tests/models/minimax_m3/amd_mono/test_ffn_mono.py [widths...]
"""

import glob
import json
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TP = 4
FP8 = torch.float8_e4m3fnuz
FP8_MAX = 224.0  # the routed experts' (vLLM's dynamic per-token quant on fnuz)
SH_FP8_MAX = 240.0  # the shared expert's linears (the format's max)
REAL_LAYER = os.environ.get("MONO_REAL_LAYER")
CHECKPOINT = "models--EmbeddedLLM--MiniMax-M3-FP8-dynamic"


def quant_rows(x: torch.Tensor, shared: bool = False):
    """Per-row FP8 as vLLM quantizes the routed experts' inputs (scaled_fp8_quant:
    amax / 224 correctly rounded, floored, x / scale) or, ``shared``, the shared
    expert's linears' (AITER per-token: amax * f32(1/240), x * (1 / scale))."""
    a = x.abs().amax(-1, keepdim=True).float()
    if shared:
        s = a * torch.tensor(1.0 / SH_FP8_MAX, dtype=torch.float32, device=x.device)
        inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
        return (x.float() * inv).clamp(-SH_FP8_MAX, SH_FP8_MAX).to(FP8), s
    s = (a.double() / FP8_MAX).float().clamp_min(1 / (FP8_MAX * 512))
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8), s


def ref_ffn(w, x):
    """vLLM's chain for one rank's share: (routed + shared) bf16 [M, H]."""
    H = x.shape[1]
    I = w.sdn.shape[1]
    logits = x.float() @ w.gate.T
    sc = logits.sigmoid()
    picks = (sc + w.bias).topk(4, dim=-1).indices
    wts = sc.gather(1, picks)
    wts = wts / wts.sum(-1, keepdim=True)
    xq, xs = quant_rows(x)
    xq_sh, xs_sh = quant_rows(x, shared=True)

    def expert(w13, s13, w2, s2, xq_, xs_, routed=True):
        gu = ((xq_.float() @ w13.float().T) * xs_ * s13.view(1, -1)).bfloat16().float()
        g = gu[:, :I].clamp(max=7.0)
        u = gu[:, I:].clamp(-7.0, 7.0)
        gsig = g * torch.sigmoid(1.702 * g)
        if routed:  # silu_and_mul_with_clamp rounds this product; swiglu_oai_split not
            gsig = gsig.bfloat16().float()
        mid = (gsig * (u + 1.0)).bfloat16()
        mq, ms = quant_rows(mid, shared=not routed)
        return (mq.float() @ w2.float().T) * ms * s2.view(1, -1)

    out = torch.zeros(x.shape[0], H, device=x.device)
    for k in range(x.shape[0]):
        acc = torch.zeros(H, device=x.device)
        for j in range(4):
            e = picks[k, j]
            d = expert(w.w13[e], w.s13[e], w.w2[e], w.s2[e], xq[k : k + 1], xs[k : k + 1])
            acc += (d[0] * wts[k, j]).bfloat16().float()
        routed = (acc.bfloat16().float() * 2.0).bfloat16().float()
        sh = expert(w.sgu, w.sgu_s, w.sdn, w.sdn_s, xq_sh[k : k + 1], xs_sh[k : k + 1], routed=False)
        out[k] = (routed + sh[0].bfloat16().float()).bfloat16().float()
    return out.bfloat16()


def trunc(x):
    """f32 -> bf16 rounding toward zero (AITER's store on gfx94x)."""
    return (x.float().contiguous().view(torch.int32) & -65536).view(torch.float32).bfloat16()


def ar_norm(parts, residual, g):
    """vLLM's AITER fused all-reduce + Gemma norm: (normed, new residual)."""
    h = trunc(sum(p.float() for p in parts)).float() + residual.float()
    rstd = torch.rsqrt(h.pow(2).sum(-1, keepdim=True) / h.shape[-1] + 1e-6)
    return trunc(h * (1.0 + g.float()) * rstd), trunc(h)


def make_weights(dev, seed):
    from vllm.models.minimax_m3.amd.mono.ops.ffn import E, H, I
    from vllm.models.minimax_m3.amd.mono.runtime import FFNWeights

    shared = torch.Generator(device=dev).manual_seed(1)
    g = torch.Generator(device=dev).manual_seed(seed)

    def fp8(*shape):
        return (torch.randn(*shape, device=dev, generator=g) * 0.5).clamp(-200, 200).to(FP8)

    def scale(*shape):
        return torch.rand(*shape, device=dev, generator=g) * 2e-2 + 1e-3

    return FFNWeights(
        g_post=(torch.randn(H, device=dev, generator=shared) * 0.1).bfloat16(),
        gate=torch.randn(E, H, device=dev, generator=shared) * 0.02,
        bias=torch.randn(E, device=dev, generator=shared) * 0.01,
        w13=fp8(E, 2 * I, H),
        s13=scale(E, 2 * I, 1),
        w2=fp8(E, H, I),
        s2=scale(E, H, 1),
        sgu=fp8(2 * I, H),
        sgu_s=scale(2 * I, 1),
        sdn=fp8(H, I),
        sdn_s=scale(H, 1),
    )


def load_weights(dev, rank, layer):
    """This rank's FFN of a checkpoint layer, as vLLM holds it after loading on
    gfx94x: e4m3fn bits read as fnuz with the scales doubled, -0 cleared."""
    from safetensors import safe_open

    from vllm.models.minimax_m3.amd.mono.ops.ffn import E, I
    from vllm.models.minimax_m3.amd.mono.runtime import FFNWeights

    root = glob.glob(f"{os.environ.get('HF_HUB_CACHE', '/data/hf_home')}/{CHECKPOINT}/snapshots/*/")[0]
    wmap = json.load(open(root + "model.safetensors.index.json"))["weight_map"]
    pre = f"language_model.model.layers.{layer}."
    want = {k for k in wmap if k.startswith(pre) and ("block_sparse_moe" in k or "post_attention" in k)}
    t = {}
    for shard in sorted({wmap[k] for k in want}):
        with safe_open(root + shard, "pt", device=str(dev)) as f:
            t.update({k[len(pre) :]: f.get_tensor(k) for k in f.keys() if k in want})
    sl = slice(rank * I, (rank + 1) * I)

    def fnuz(w, s):
        b = w.view(torch.uint8).clone()
        b[b == 0x80] = 0
        return b.view(FP8).contiguous(), (s.float() * 2.0).contiguous()

    def exp(name, x):
        return t[f"block_sparse_moe.experts.{x}.{name}"]

    def shared(name):
        return t[f"block_sparse_moe.shared_experts.{name}"]

    w13, s13 = fnuz(
        torch.stack([torch.cat([exp("w1.weight", x)[sl], exp("w3.weight", x)[sl]]) for x in range(E)]),
        torch.stack([torch.cat([exp("w1.weight_scale", x)[sl], exp("w3.weight_scale", x)[sl]]) for x in range(E)]),
    )  # fmt: skip
    w2, s2 = fnuz(torch.stack([exp("w2.weight", x)[:, sl] for x in range(E)]),
                  torch.stack([exp("w2.weight_scale", x) for x in range(E)]))  # fmt: skip
    sgu, sgu_s = fnuz(torch.cat([shared("gate_proj.weight")[sl], shared("up_proj.weight")[sl]]),
                      torch.cat([shared("gate_proj.weight_scale")[sl], shared("up_proj.weight_scale")[sl]]))  # fmt: skip
    sdn, sdn_s = fnuz(shared("down_proj.weight")[:, sl], shared("down_proj.weight_scale"))
    return FFNWeights(
        g_post=t["post_attention_layernorm.weight"].bfloat16(),
        gate=t["block_sparse_moe.gate.weight"].float().contiguous(),
        bias=t["block_sparse_moe.e_score_correction_bias"].float().contiguous(),
        w13=w13, s13=s13, w2=w2, s2=s2, sgu=sgu, sgu_s=sgu_s, sdn=sdn, sdn_s=sdn_s,
    )  # fmt: skip


def worker(rank, widths, port, result):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=TP)
    from vllm.models.minimax_m3.amd.mono.ops.ffn import H
    from vllm.models.minimax_m3.amd.mono.runtime import MonoFFN

    if REAL_LAYER is None:
        ws_all = [make_weights(dev, 100 + r) for r in range(TP)]
        w = ws_all[rank]
    else:
        w = load_weights(dev, rank, int(REAL_LAYER))
    mono = MonoFFN(TP, rank, dist.group.WORLD, dev)
    errs = {}
    for m in widths:
        gen = torch.Generator(device=dev).manual_seed(7 + m)
        parts = [(torch.randn(m, H, device=dev, generator=gen) * 0.3).bfloat16() for _ in range(TP)]
        residual = (torch.randn(m, H, device=dev, generator=gen) * 3).bfloat16()
        xn, h = ar_norm(parts, residual, w.g_post)
        # the MoE all-reduce as vLLM's (AITER) on gfx94x: rank order, truncating
        if REAL_LAYER is None:
            ref_sum = trunc(sum(ref_ffn(wr, xn).float() for wr in ws_all))
        else:  # each rank holds only its share: reduce the reference over the group
            mine = ref_ffn(w, xn).float().cpu()
            shares = [torch.zeros_like(mine) for _ in range(TP)]
            dist.all_gather(shares, mine)
            ref_sum = trunc(sum(shares)).to(dev)
        for _ in range(3):
            out, h_mid = mono.run(w, parts[rank], residual)
            torch.cuda.synchronize()
        dist.barrier()
        rel = ((out.float() - ref_sum.float()).norm() / ref_sum.float().norm()).item()
        errs[m] = (rel, torch.equal(h_mid, h))
    if rank == 0:
        result.update(errs)
    mono.close()
    dist.barrier()
    dist.destroy_process_group()


def run(widths):
    mgr = mp.Manager()
    result = mgr.dict()
    mp.start_processes(worker, args=(widths, 29611, result), nprocs=TP, start_method="spawn")
    return dict(result)


@pytest.mark.skipif(torch.cuda.device_count() < TP, reason="needs 4 GPUs")
@pytest.mark.parametrize("widths", [[1, 4, 8, 16]])
def test_ffn_mono_matches_reference(widths):
    for m, (rel, res_ok) in run(widths).items():
        assert res_ok, f"width {m}: residual mismatch"
        assert rel < 2e-2, f"width {m}: rel err {rel}"


if __name__ == "__main__":
    ws = [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16]
    for m, (rel, ok) in sorted(run(ws).items()):
        print(f"width {m}: rel={rel:.3e} residual_exact={ok}")
