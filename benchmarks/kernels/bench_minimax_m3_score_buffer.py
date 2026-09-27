"""Validate the persistent score buffer against per-call allocation.

Checks the two things that could break when a fresh [H, rows, W] tensor is
replaced by a [:, lo:hi, :W] slice of a shared [H, max_tokens, W_max] buffer:

  1. the AITER stride contract (score.stride(2) == 1, checked in
     aiter/ops/msa_block_select.py), and
  2. bit-identical selection, including that decode's [0, nd) and prefill's
     [nd, ntok) regions do not disturb each other.

Also measures the allocation cost the change removes.
"""

import math
import time

import torch

from aiter.ops.msa_block_select import (
    pa_sparse_block_score_decode,
    pa_sparse_block_topk,
)

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16
PAGES_PER_BLOCK = 8
NUM_IDX_HEADS = 1
NUM_KV_HEADS = 1
MAX_MODEL_LEN = 133120
MAX_TOKENS = 32768


def pow2_ceil(n):
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def score_block_width(max_seq_len, block_size):
    max_blk = math.ceil(max(max_seq_len, 1) / block_size)
    return pow2_ceil(math.ceil(max_blk / 64)) * 64


def make_batch(num_reqs, live_ctx, dev, seed=0):
    nblk = math.ceil(live_ctx / BLOCK_SIZE)
    max_blk = math.ceil(MAX_MODEL_LEN / BLOCK_SIZE)
    pages = num_reqs * nblk + 1
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(num_reqs, NUM_IDX_HEADS, HEAD_DIM, generator=g,
                    device=dev, dtype=torch.float32) * 4.0
    kv = torch.randn(pages, BLOCK_SIZE, HEAD_DIM, generator=g,
                     device=dev, dtype=torch.float32) * 4.0
    bt = (torch.arange(num_reqs * max_blk, dtype=torch.int32, device=dev)
          .view(num_reqs, max_blk) % pages).contiguous()
    return dict(
        q=q.to(torch.float8_e4m3fn).contiguous(),
        kv=kv.to(torch.float8_e4m3fn).contiguous(),
        bt=bt,
        sl=torch.full((num_reqs,), live_ctx, dtype=torch.int32, device=dev),
        reqs=num_reqs,
    )


def run(b, score, dev):
    pa_sparse_block_score_decode(
        b["q"], b["kv"], score, b["bt"], b["sl"],
        init_blocks=0, local_blocks=1, query_len=1, max_seq_len=MAX_MODEL_LEN,
    )
    topk_idx = torch.empty((NUM_IDX_HEADS, b["reqs"], TOPK),
                           dtype=torch.int32, device=dev)
    rows = b["reqs"] * NUM_KV_HEADS
    sbt = torch.empty((rows, TOPK * PAGES_PER_BLOCK), dtype=torch.int32, device=dev)
    sctx = torch.empty(rows, dtype=torch.int32, device=dev)
    pa_sparse_block_topk(
        score, topk_idx, b["bt"], b["sl"], max_seq_len=MAX_MODEL_LEN,
        block_size=BLOCK_SIZE, query_len=1, sparse_bt=sbt, sparse_ctx=sctx,
        num_kv_heads=NUM_KV_HEADS, pages_per_block=PAGES_PER_BLOCK,
    )
    torch.cuda.synchronize()
    return topk_idx.clone(), sbt.clone(), sctx.clone()


def main():
    dev = torch.device("cuda")
    width = score_block_width(MAX_MODEL_LEN, BLOCK_SIZE)
    shared = torch.empty((NUM_IDX_HEADS, MAX_TOKENS, width),
                         dtype=torch.float32, device=dev)
    print(f"shared score buffer: {tuple(shared.shape)} "
          f"= {shared.numel() * 4 / 2**20:.1f} MiB\n")

    print("=== 1. stride contract ===")
    for lo, hi in ((0, 64), (64, 192), (7, 71)):
        s = shared[:, lo:hi, :width]
        ok = s.stride(2) == 1 and s.size(0) == NUM_IDX_HEADS
        print(f"  [:, {lo}:{hi}, :{width}] strides={s.stride()} "
              f"contiguous_block_axis={s.stride(2) == 1} {'OK' if ok else 'FAIL'}")
        assert ok

    print("\n=== 2. selection: shared slice vs fresh allocation ===")
    fails = 0
    for live_ctx in (2048, 8192, 60000, 131072):
        for reqs in (4, 64, 128):
            b = make_batch(reqs, live_ctx, dev, seed=live_ctx + reqs)
            fresh = torch.empty((NUM_IDX_HEADS, reqs, width),
                                dtype=torch.float32, device=dev)
            ref = run(b, fresh, dev)
            # Poison the whole shared buffer: a slice that reads outside its
            # own rows would pick the poison up instead of its own scores.
            shared.fill_(float("nan"))
            got = run(b, shared[:, 0:reqs, :width], dev)
            same = all(torch.equal(x, y) for x, y in zip(ref, got))
            fails += not same
            print(f"  ctx={live_ctx:>6} reqs={reqs:>4} "
                  f"{'IDENTICAL' if same else 'MISMATCH'}")

    print("\n=== 3. decode/prefill disjointness ===")
    nd, npre = 64, 128
    bd = make_batch(nd, 8192, dev, seed=1)
    bp = make_batch(npre, 60000, dev, seed=2)
    fd = torch.empty((NUM_IDX_HEADS, nd, width), dtype=torch.float32, device=dev)
    fp = torch.empty((NUM_IDX_HEADS, npre, width), dtype=torch.float32, device=dev)
    ref_d, ref_p = run(bd, fd, dev), run(bp, fp, dev)
    shared.fill_(float("nan"))
    got_d = run(bd, shared[:, 0:nd, :width], dev)
    got_p = run(bp, shared[:, nd:nd + npre, :width], dev)
    # re-read decode AFTER prefill wrote its region: must be untouched
    got_d2 = run(bd, shared[:, 0:nd, :width], dev)
    ok_d = all(torch.equal(x, y) for x, y in zip(ref_d, got_d))
    ok_p = all(torch.equal(x, y) for x, y in zip(ref_p, got_p))
    ok_d2 = all(torch.equal(x, y) for x, y in zip(ref_d, got_d2))
    fails += not (ok_d and ok_p and ok_d2)
    print(f"  decode [0:{nd})        {'IDENTICAL' if ok_d else 'MISMATCH'}")
    print(f"  prefill [{nd}:{nd+npre})   {'IDENTICAL' if ok_p else 'MISMATCH'}")
    print(f"  decode after prefill  {'UNDISTURBED' if ok_d2 else 'CORRUPTED'}")

    print("\n=== 4. allocation cost removed (host time, 57 sparse layers) ===")
    for rows, label in ((64, "decode  nd=64"), (32768, "prefill ntok=32768")):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(57):
            tmp = torch.empty((NUM_IDX_HEADS, rows, width),
                              dtype=torch.float32, device=dev)
            del tmp
        torch.cuda.synchronize()
        alloc_us = (time.perf_counter() - t0) * 1e6
        t0 = time.perf_counter()
        for _ in range(57):
            _ = shared[:, 0:rows, :width]
        torch.cuda.synchronize()
        slice_us = (time.perf_counter() - t0) * 1e6
        mib = NUM_IDX_HEADS * rows * width * 4 / 2**20
        print(f"  {label:<22} {mib:8.1f} MiB/layer  "
              f"alloc={alloc_us:8.1f}us  slice={slice_us:6.1f}us  "
              f"saved={alloc_us - slice_us:8.1f}us/step")

    print("\n" + "=" * 60)
    print("PASS: shared buffer is equivalent" if not fails
          else f"FAIL: {fails} mismatch(es)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
