# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 attention MonoKernel vs vLLM's own decode ops, one GPU.

Reference = the exact op chain vLLM runs for one sparse layer in decode on
gfx942 (Triton/C++ path, fp8 rowwise linears, fp8 KV cache, bf16 index cache):

  gemma_fused_add_rmsnorm -> scaled_fp8_quant(per-token) + _scaled_mm (qkv)
  -> ops.fused_minimax_m3_qknorm_rope_kv_insert -> minimax_m3_index_decode
  -> minimax_m3_sparse_attn_decode -> scaled_fp8_quant + _scaled_mm (o_proj)

Run: python -m pytest tests/models/minimax_m3/amd_mono/test_attn_mono.py
or   python tests/models/minimax_m3/amd_mono/test_attn_mono.py [--ref-only] [--bench]
     [--ctx 300,8192,131072] [--cases 1x1,4x1,8x1,16x1,8x4]   (MxQ_LEN)
"""

import argparse
import os
import sys
from types import SimpleNamespace

import pytest
import torch

FP8 = torch.float8_e4m3fnuz
# MONO_FP8_INDEX=1: an fp8 (fnuz) index cache; the reference's fused writer must
# then be the gfx942-fixed build (run with the fnuz conftest, FNUZ_SO=...)
FP8_INDEX = os.environ.get("MONO_FP8_INDEX", "0") == "1"
IC_DTYPE = FP8 if FP8_INDEX else torch.bfloat16
FP8_MAX = 224.0
H, NH, NKV, HD, ROT = 6144, 16, 1, 128, 64
IDXD = 128
Q_SIZE = NH * HD
QKV_N = Q_SIZE + 2 * NKV * HD + 2 * IDXD  # 2560
TOPK, INIT_B, LOCAL_B, BS = 16, 0, 1, 128
EPS = 1e-6
SM_SCALE = HD**-0.5
MAXPOS = 262144
ROPE_THETA = 5_000_000.0
MAX_TOK = 64

WEIGHT_FIELDS = ("g_in", "w_qkv", "s_qkv", "g_q", "g_k", "g_iq", "g_ik", "cos_sin",
                 "w_o", "s_o", "k_scale", "v_scale", "sm_scale", "eps")

_COS_SIN = {}


def cos_sin_cache(dev):
    if dev not in _COS_SIN:
        inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32, device=dev) / ROT))
        f = torch.outer(torch.arange(MAXPOS, dtype=torch.float32, device=dev), inv)
        _COS_SIN[dev] = torch.cat([f.cos(), f.sin()], dim=-1).to(torch.bfloat16).contiguous()
    return _COS_SIN[dev]


# --- case construction ---
def make_weights(dev, gen):
    def bf(n, s=0.1):
        return (torch.randn(n, device=dev, generator=gen) * s).bfloat16()

    def fp8(*shape):
        # |w| ~ N(0,1) in fp8; with scales 1e-3..2e-2 the GEMM outputs are O(1)
        return torch.randn(*shape, device=dev, generator=gen).clamp(-200, 200).to(FP8)

    def scale(n, lo=1e-3, hi=2e-2):
        return torch.rand(n, device=dev, generator=gen) * (hi - lo) + lo

    one = torch.ones(1, dtype=torch.float32, device=dev)
    return dict(
        g_in=bf(H), w_qkv=fp8(QKV_N, H), s_qkv=scale(QKV_N),
        g_q=bf(HD), g_k=bf(HD), g_iq=bf(IDXD), g_ik=bf(IDXD),
        cos_sin=cos_sin_cache(dev),
        # o_proj input is an attention average (rms ~0.05): larger scales keep partial O(1)
        w_o=fp8(H, Q_SIZE), s_o=scale(H, 5e-2, 3e-1),
        k_scale=one, v_scale=one.clone(), sm_scale=SM_SCALE, eps=EPS,
    )


def make_case(dev, M, q_len, ctx, seed, add_residual=True, weights=None):
    """Random layer + decode step. R = M // q_len requests, each with its own
    shuffled pages; caches pre-filled for positions [0, ctx)."""
    from vllm.triton_utils import triton

    assert M % q_len == 0
    dev = torch.device(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    R = M // q_len
    seq = ctx + q_len
    nbr = triton.cdiv(seq, BS)  # pages per request
    width = triton.cdiv(nbr, 16) * 16  # padded block-table row (stride != nbr)
    nb = R * nbr + 7  # a few spare pages
    w = make_weights(dev, gen) if weights is None else weights

    # random K/V/index_k cast as the insert kernel does (k_scale = v_scale = 1)
    kv = torch.empty(nb, NKV, BS, 2 * HD, dtype=torch.uint8, device=dev)
    ic = torch.empty(nb, BS, IDXD, dtype=IC_DTYPE, device=dev)
    for b0 in range(0, nb, 2048):  # chunked to bound the f32 temporaries
        b1 = min(nb, b0 + 2048)
        k = torch.randn(b1 - b0, NKV, BS, HD, device=dev, generator=gen).bfloat16()
        v = torch.randn(b1 - b0, NKV, BS, HD, device=dev, generator=gen).bfloat16()
        kv[b0:b1] = torch.cat([k, v], -1).to(FP8).view(torch.uint8)
        # log-uniform per-block magnitudes separate the block scores, so the
        # top-k does not hinge on near ties that rounding may flip
        mag = torch.exp(torch.rand(b1 - b0, 1, 1, device=dev, generator=gen) * 3.0 - 1.5)
        keys = (torch.randn(b1 - b0, BS, IDXD, device=dev, generator=gen) * mag).bfloat16()
        ic[b0:b1] = keys.float().clamp(-FP8_MAX, FP8_MAX).to(IC_DTYPE) if FP8_INDEX else keys

    perm = torch.randperm(nb, device=dev, generator=gen).to(torch.int32)
    bt = torch.zeros(R, width, dtype=torch.int32, device=dev)
    bt[:, :nbr] = perm[: R * nbr].view(R, nbr)
    seq_lens = torch.full((R,), seq, dtype=torch.int32, device=dev)
    j = torch.arange(q_len, device=dev)
    positions = (seq - q_len + j).repeat(R).to(torch.int64)  # [M], request-major
    req = torch.arange(R, device=dev).repeat_interleave(q_len)
    slots = (bt[req, positions // BS].to(torch.int64) * BS + positions % BS).contiguous()

    x = torch.randn(M, H, device=dev, generator=gen).bfloat16()
    residual = torch.randn(M, H, device=dev, generator=gen).bfloat16()
    return SimpleNamespace(
        M=M, q_len=q_len, ctx=ctx, R=R, nbr=nbr, w=w, x=x, residual=residual,
        positions=positions, slots=slots, block_table=bt, seq_lens=seq_lens,
        max_seq_len=seq, kv_cache=kv, index_cache=ic, add_residual=add_residual,
    )


# --- vLLM reference chain ---
def fp8_linear(x, w, s, exact=False):
    """vLLM's W8A8 linear on ROCm: AITER's per-token input quant (QuantFP8)."""
    from vllm._aiter_ops import rocm_aiter_ops

    xq, xs = rocm_aiter_ops.per_token_quant(x.contiguous(), FP8, None)
    if exact:  # same quantized operands, fp32 torch matmul (noise-floor variant)
        return ((xq.float() @ w.float().t()) * xs * s.view(1, -1)).bfloat16()
    return torch._scaled_mm(xq, w.t(), out_dtype=torch.bfloat16, scale_a=xs,
                            scale_b=s.view(1, -1))


class RefBuffers:
    def __init__(self, dev):
        self.topk = torch.full((NKV, MAX_TOK, TOPK), -1, dtype=torch.int32, device=dev)
        self.counter = torch.zeros(NKV, MAX_TOK, dtype=torch.int32, device=dev)


_REF_BUFS = {}


def reference_step(w, x, residual, positions, slots, bt, seq_lens, max_seq_len, kv, ic,
                   q_len, add_residual=True, bufs=None, exact_gemm=False):
    """One sparse layer's decode chain as vLLM runs it; mutates kv (uint8 fp8 bytes)
    and ic. -> (partial, res_out, topk, attn_out, q)."""
    from vllm import _custom_ops as ops
    from vllm.models.minimax_m3.amd.ops import gemma_fused_add_rmsnorm, gemma_rmsnorm
    from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_decode
    from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode

    if bufs is None:
        bufs = _REF_BUFS.setdefault(x.device, RefBuffers(x.device))
    M = x.shape[0]
    if add_residual:
        xn, res = gemma_fused_add_rmsnorm(x, residual, w["g_in"], w["eps"])
    else:
        xn, res = gemma_rmsnorm(x, w["g_in"], w["eps"]), x
    qkv = fp8_linear(xn, w["w_qkv"], w["s_qkv"], exact_gemm)
    q = torch.empty(M, Q_SIZE, dtype=torch.bfloat16, device=x.device)
    iq = torch.empty(M, IDXD, dtype=ic.dtype, device=x.device)
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, w["g_q"], w["g_k"], w["cos_sin"], positions, NH, NKV, ROT, w["eps"],
        w["g_iq"], w["g_ik"], 1, slots, slots, kv, ic, BS, q, iq, "fp8")
    minimax_m3_index_decode(
        iq.view(M, 1, IDXD), ic, bt, seq_lens, max_seq_len, TOPK, INIT_B, LOCAL_B,
        NKV, q_len, q_len, out=bufs.topk, completion_counter=bufs.counter)
    topk = bufs.topk[:, :M, :]
    out = torch.empty(M, NH, HD, dtype=torch.bfloat16, device=x.device)
    minimax_m3_sparse_attn_decode(
        q.view(M, NH, HD), kv.view(FP8), topk, bt, seq_lens, NKV, w["sm_scale"], out,
        q_len, k_scale=w["k_scale"], v_scale=w["v_scale"])
    partial = fp8_linear(out.view(M, Q_SIZE), w["w_o"], w["s_o"], exact_gemm)
    return partial, res, topk, out, q


def reference(case, exact_gemm=False):
    kv, ic = case.kv_cache.clone(), case.index_cache.clone()
    partial, res, topk, out, q = reference_step(
        case.w, case.x, case.residual, case.positions, case.slots, case.block_table,
        case.seq_lens, case.max_seq_len, kv, ic, case.q_len, case.add_residual,
        exact_gemm=exact_gemm)
    torch.cuda.synchronize()
    if not exact_gemm:
        case._ref_attn = (out.clone(), q.clone())  # for the torch self-check
    return partial.clone(), res.clone(), kv, ic, topk.clone()


def topk_agree(a, b):
    """Fraction of tokens whose selected block SETS are identical."""
    same = [set(x) - {-1} == set(y) - {-1} for x, y in zip(a[0].tolist(), b[0].tolist())]
    return sum(same) / len(same)


def torch_attn_check(case, kv, topk):
    """Recompute sparse attention in fp32 from the (post-insert) cache and the
    reference top-k -> rel err of vLLM's attention output (sanity of the chain)."""
    out_ref, q = case._ref_attn
    rows = kv.view(FP8).view(-1, 2 * HD).float()
    bt = case.block_table.long().cpu()
    tk = topk[0].long().cpu()
    errs = []
    for t in range(case.M):
        r, j = divmod(t, case.q_len)
        L = case.max_seq_len - (case.q_len - 1 - j)
        blks = sorted({b for b in tk[t].tolist() if b >= 0})
        idx = torch.cat([bt[r, b] * BS + torch.arange(BS) for b in blks])
        pos = torch.cat([b * BS + torch.arange(BS) for b in blks])
        idx = idx[pos < L].to(rows.device)
        K, V = rows[idx, :HD], rows[idx, HD:]
        s = (q[t].view(NH, HD).float() @ K.T) * SM_SCALE
        o = s.softmax(-1) @ V
        errs.append(((o - out_ref[t].float()).norm() / o.norm()).item())
    return max(errs)


# --- MonoKernel ---
_MONO = {}


def mono_obj(dev, max_blocks):
    from vllm.models.minimax_m3.amd.mono.runtime import MonoAttn

    key = (dev, max_blocks)
    if key not in _MONO:
        _MONO[key] = MonoAttn(dev, max_blocks)
    return _MONO[key]


def attn_weights(w):
    from vllm.models.minimax_m3.amd.mono.runtime import AttnWeights

    return AttnWeights(**{k: w[k] for k in WEIGHT_FIELDS})


def mono(case):
    m = mono_obj(case.x.device, case.block_table.shape[1])
    kv, ic = case.kv_cache.clone(), case.index_cache.clone()
    partial, res = m.run(
        attn_weights(case.w), case.x, case.residual, case.positions, case.slots,
        case.block_table, case.seq_lens, kv.view(FP8), ic, q_len=case.q_len,
        add_residual=case.add_residual)
    torch.cuda.synchronize()
    return partial.clone(), res.clone(), kv, ic


# --- compare ---
def compare(case, ref, got):
    p_r, res_r, kv_r, ic_r, _ = ref
    p_m, res_m, kv_m, ic_m = got
    s = case.slots
    kv_rows_r, kv_rows_m = kv_r.view(-1, 2 * HD)[s], kv_m.view(-1, 2 * HD)[s]
    ic_rows_r, ic_rows_m = ic_r.view(-1, IDXD)[s], ic_m.view(-1, IDXD)[s]
    kf_r, kf_m = kv_rows_r.view(FP8).float(), kv_rows_m.view(FP8).float()
    if_r, if_m = ic_rows_r.float(), ic_rows_m.float()
    ic_r, ic_m = ic_r.view(torch.uint8), ic_m.view(torch.uint8)
    ic_rows_r, ic_rows_m = ic_rows_r.view(torch.uint8), ic_rows_m.view(torch.uint8)
    a, b = p_r.float().flatten(), p_m.float().flatten()
    return dict(
        res_exact=torch.equal(res_r, res_m),
        kv_bad_bytes=int((kv_rows_r != kv_rows_m).sum()),
        kv_max_diff=(kf_r - kf_m).abs().max().item(),
        kv_rel=((kf_r - kf_m).norm() / kf_r.norm()).item(),
        ic_bad=int((ic_rows_r != ic_rows_m).sum()),
        ic_max_diff=(if_r - if_m).abs().max().item(),
        ic_rel=((if_r - if_m).norm() / if_r.norm()).item(),
        # anything written outside this step's slots?
        kv_other_bad=int((kv_r != kv_m).sum()) - int((kv_rows_r != kv_rows_m).sum()),
        ic_other_bad=int((ic_r != ic_m).sum()) - int((ic_rows_r != ic_rows_m).sum()),
        rel=((a - b).norm() / a.norm().clamp_min(1e-30)).item(),
        cos=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        finite=bool(torch.isfinite(b).all()),
    )


def tag(M, q_len, ctx):
    return f"M={M:2d} q_len={q_len} ctx={ctx:6d}"


def run_case(dev, M, q_len, ctx, seed=0, ref_only=False):
    case = make_case(dev, M, q_len, ctx, seed)
    ref = reference(case)
    p_r, _, kv_r, _, topk = ref
    nvalid = int((topk[0] >= 0).sum(-1).min())
    line = (f"{tag(M, q_len, ctx)} | ref |partial|rms={p_r.float().pow(2).mean().sqrt():.3f}"
            f" finite={bool(torch.isfinite(p_r).all())} topk_valid_min={nvalid}"
            f" attn_vs_torch={torch_attn_check(case, kv_r, topk):.2e}")
    # noise floor: both GEMMs in fp32 torch on the same fp8 operands
    ref2 = reference(case, exact_gemm=True)
    n = compare(case, ref, ref2[:4])
    line += (f" | floor rel={n['rel']:.2e} kv_bad={n['kv_bad_bytes']} ic_bad={n['ic_bad']}"
             f" topk_same={topk_agree(topk, ref2[4]):.2f}")
    r = None
    if not ref_only:
        r = compare(case, ref, mono(case))
        line += (f" | mono rel={r['rel']:.3e} cos={r['cos']:.6f} res_exact={r['res_exact']}"
                 f" kv_bad={r['kv_bad_bytes']}(max {r['kv_max_diff']:.3g} rel {r['kv_rel']:.1e})"
                 f" ic_bad={r['ic_bad']}(max {r['ic_max_diff']:.3g} rel {r['ic_rel']:.1e})"
                 f" other={r['kv_other_bad']}/{r['ic_other_bad']}")
    del case
    torch.cuda.empty_cache()
    return line, r


# --- bench (HIP graphs, NL distinct layers -> cold weights/caches) ---
def time_graph(fn, nl):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(9):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / nl)
    del g
    return sorted(ts)[len(ts) // 2]


def bench(dev, M, q_len, ctx, nl_cap=32, ref_only=False, mem_cap=60e9):
    from vllm.triton_utils import triton

    nbr = triton.cdiv(ctx + q_len, BS)
    per_layer = 28e6 + (M // q_len) * nbr * BS * (2 * HD + 2 * IDXD)
    nl = max(2, min(nl_cap, int(mem_cap / per_layer)))
    base = make_case(dev, M, q_len, ctx, 0)
    layers = [base] + [make_case(dev, M, q_len, ctx, 1 + i) for i in range(nl - 1)]

    def ref_all():
        for c in layers:
            reference_step(c.w, c.x, c.residual, c.positions, c.slots, c.block_table,
                           c.seq_lens, c.max_seq_len, c.kv_cache, c.index_cache, q_len)

    out = dict(layers=nl, ref_us=time_graph(ref_all, nl))
    if not ref_only:
        m = mono_obj(base.x.device, base.block_table.shape[1])
        ws = [attn_weights(c.w) for c in layers]
        kvs = [c.kv_cache.view(FP8) for c in layers]

        def mono_all():
            for c, w, kv in zip(layers, ws, kvs):
                m.run(w, c.x, c.residual, c.positions, c.slots, c.block_table,
                      c.seq_lens, kv, c.index_cache, q_len=q_len)

        out["mono_us"] = time_graph(mono_all, nl)
    del layers, base
    torch.cuda.empty_cache()
    return out


# --- pytest ---
SMALL = [(1, 1, 300), (4, 1, 300), (8, 1, 300), (16, 1, 300), (8, 4, 300),
         (1, 1, 8192), (16, 1, 8192), (8, 4, 8192)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("M,q_len,ctx", SMALL)
def test_attn_mono_matches_vllm(M, q_len, ctx):
    line, r = run_case(torch.device("cuda"), M, q_len, ctx, seed=M + 31 * q_len)
    print(line)
    assert r["finite"], line
    assert r["res_exact"], line
    # inserted rows: the qkv GEMM's bf16 rounding may differ by an ulp from
    # hipBLASLt's, which flips some fp8/bf16 cache bytes; bound the error instead.
    assert r["kv_rel"] < 3e-2 and r["ic_rel"] < 1e-2, line
    assert r["kv_other_bad"] == 0 and r["ic_other_bad"] == 0, line
    assert r["rel"] < 3e-2, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ctx", default="300,8192,131072")
    ap.add_argument("--cases", default="1x1,4x1,8x1,16x1,8x4")
    ap.add_argument("--layers", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ctxs = [int(c) for c in a.ctx.split(",")]
    cases = [tuple(int(v) for v in c.split("x")) for c in a.cases.split(",")]
    print("device", torch.cuda.get_device_name(dev), flush=True)
    for ctx in ctxs:
        for M, ql in cases:
            try:
                line, _ = run_case(dev, M, ql, ctx, seed=M + 31 * ql, ref_only=a.ref_only)
            except Exception as e:  # keep going so one failure doesn't hide the grid
                line = f"{tag(M, ql, ctx)} | ERROR {type(e).__name__}: {e}"
            print(line, flush=True)
    if a.bench:
        print("bench (HIP graph, distinct layers, us/layer)", flush=True)
        for ctx in ctxs:
            for M, ql in cases:
                try:
                    b = bench(dev, M, ql, ctx, a.layers, a.ref_only)
                    s = f"ref {b['ref_us']:7.1f}"
                    if "mono_us" in b:
                        s += f"  mono {b['mono_us']:7.1f}  speedup {b['ref_us'] / b['mono_us']:.2f}x"
                except Exception as e:
                    s, b = f"ERROR {type(e).__name__}: {e}", {"layers": "?"}
                print(f"{tag(M, ql, ctx)} layers={b['layers']}: {s}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
