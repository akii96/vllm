# ISC: fast prefill index block-score kernel (gfx942 / MI300X)

Branch `gfx942-index-score-fast` in worktree `/workspace/isc/vllm` (local only; base
`rocm-minimax-m31-gfx942-mxfp4-kv-cache`). Runtime tree: `PYTHONPATH=/workspace/pkg/isc`
(copy of comb5 with the same `index_topk.py` patch). Dev scratch: `/workspace/isc/dev`.

## What changed
`vllm/models/minimax_m3/amd/ops/index_topk.py`: new `_index_block_score_split_kernel`
is the default of `minimax_m3_index_score`. The old `_index_block_score_kernel` is kept behind
`VLLM_MINIMAX_M3_INDEX_SCORE_LEGACY=1`. The output layout is unchanged, so both the fp8 indexer
prefill and the MXFP4 fp8-copy prefill pick it up.

Why the legacy kernel was slow (measured):
- **Grid.** It launches 256 CTAs (16k/64 rows x H) on 304 CUs, each one a long serial K loop.
  The tail CUs idle, and there is too little parallelism to hide latency.
- **Layout.** In the [64 x 128] dot with 8 warps, warps split N, so the 128-key block max crosses
  warps through LDS. (In the BLOCK_M=256 single-dot variant, WaveScope showed the s_barrier,
  K buffer_load stalls and LDS reduction were hot, with 256 VGPRs and spills.)

The new kernel:
1. **Split-K.** Each CTA does BLOCK_M=256 (row, head) pairs over KB_PER=16 key blocks (2048
   keys). The grid is (m tiles x k splits, batch), and causally empty tiles return
   immediately. Only diagonal blocks are masked.
2. **Head packing.** Heads are packed into M (m = row*H + head), so K is loaded once for all
   heads.
3. **Sub-tiles.** Each 128-key block is processed as 2 x 64-key MFMA sub-tiles with a running
   register max. Triton then picks warpsPerCTA=[4,1] (warps split M), so the block max is
   warp-local. There is no LDS allocation at all (K is streamed from global memory directly in
   dot-operand layout), with 218 VGPRs and no spills.
4. **CTA order.** CTAs are grouped GROUP=4 split columns at a time, for K reuse in L2.
5. **Config.** 4 warps, 3 stages, mfma 16x16x32 fp8, kpack=1. kpack=2 was ~2-3% faster but not
   bit-exact (abs error 7.6e-6), so it was rejected.

Each score is the same MFMA dot (same instruction and k order) as legacy, so the output is
**bit-identical** on every causally visible block. This was verified at all shapes below.

## Kernel results (GPU0, 16k-row chunk, random fp8 q/k, `isc/dev/final_bench.py`)
| case | legacy ms | new ms | speedup | new TFLOP/s | bit-exact |
|---|---|---|---|---|---|
| H=1 16k@64k | 0.590 | 0.296 | 1.99x | 812 | True |
| H=1 16k@128k | 1.266 | 0.540 | 2.34x | 955 | True |
| H=1 16k@256k | 2.343 | 1.078 | 2.17x | 988 | True |
| H=1 16k@512k | 6.403 | 2.284 | 2.80x | 948 | True |
| H=2 16k@64k | 0.929 | 0.551 | 1.69x | 873 | True |
| H=2 16k@128k | 1.845 | 1.043 | 1.77x | 988 | True |
| H=2 16k@256k | 4.137 | 2.139 | 1.93x | 996 | True |
| H=2 16k@512k | 8.252 | 4.568 | 1.81x | 948 | True |
| H=1 ragged 4x4096 rows, ctx 32k/128k/256k/512k | 4.230 | 1.104 | 3.83x | 895 | True |
| H=1 ragged 12000/300/4000/84 rows, ctx up to 512k | 5.915 | 1.859 | 3.18x | 894 | True |
| H=1 ragged 16x1024 rows, ctx 64k-256k | 2.508 | 0.793 | 3.16x | 864 | True |
| H=2 ragged 4x4096 | 5.490 | 2.203 | 2.49x | 897 | True |
| H=2 ragged 12000/300/4000/84 | 7.521 | 3.543 | 2.12x | 938 | True |
| H=2 ragged 16x1024 | 3.481 | 1.564 | 2.23x | 876 | True |

Realistic data (`idx_bench.Workload`: real index_q/k norm weights, RMS-normed, with needles): the
top-16 is identical in **100.000%** of rows for H=1@256k, H=2@256k and H=1@512k. The new kernel
runs 1.11 / 2.16 / 2.32 ms there, against 2.41 / 4.36 / 6.89 ms for legacy.

`harness/fp8copy_e2e.py` (MXFP4 default fp8-copy prefill, score + top-k):
| case | fp8-copy legacy | fp8-copy new | overlap vs exact MXFP4 (both) |
|---|---|---|---|
| H=1 256k | 2.77 ms | 1.38 ms | 0.808 |
| H=1 512k | 5.91 ms | 2.60 ms | 0.802 |
| H=2 256k | 4.29 ms | 2.65 ms | 0.802 |

Things I tried that did not help: a transposed dot (K·Qᵀ) at about the same speed; XCD remapping
(±2%); waves_per_eu=2 with kpack=2 (+2%, not exact); BLOCK_M=128 (slower); 8 warps (slower).
These kernels now run at ~950 TFLOP/s, about 37% of the 2.6 PF dense fp8 peak. The remaining gap
is the per-block max/reduce VALU work and the K load latency at 1 CTA/SIMD. The next step would
be a hand-scheduled FlyDSL/gluon kernel with K in LDS shared across 2 M tiles.

## End-to-end (TP4 GPUs 0-3, real weights, `NOSPEC=1 KV=mxfp4 PKG=isc serve.sh ... mxfp4 524288`)
TTFT is from `harness/smoke.py 8060 <ctx> 2 1`. The second run of each ctx is reported, because
the first includes JIT warmup.
| metric | legacy (`VLLM_MINIMAX_M3_INDEX_SCORE_LEGACY=1`) | new |
|---|---|---|
| TTFT 256k | 13.70 s | 13.48 s |
| TTFT 500k | 29.70 / 29.32 s | 26.35 / 26.59 s |
| GSM8K-400 strict / flexible | 0.9700 / 0.9800 | 0.9650 / 0.9775 |
| needle 120k (12 trials x 3) | 36/36 = 1.000 | 36/36 = 1.000 |

At 500k the TTFT is ~3 s (~10%) lower. 256k is within noise there, because the indexer score is a small share of TTFT at that length. The kernel is bit-exact, so the GSM8K delta (2/400) is run-to-run serving nondeterminism (batching and MoE reduction order), not this change. Raw logs: `isc/dev/e2e_isc_new.txt`,
`isc/dev/e2e_isc_legacy.txt`, and `logs/isc_{new,legacy}.log`.

## Tests
- `tests/models/minimax_m3/test_index_score_split.py` (new, 7 tests). It covers a torch fp32
  reference (rel 1e-6 / abs 1e-5·max), bit-identity with legacy, ragged multi-request batches
  with H=1/2/4 including 1-row and 129-token edges, and top-k identity.
- `tests/models/minimax_m3/test_mxfp4_kv_cache.py`: pass (49 passed together with the new file).
