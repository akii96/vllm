# MiniMax-M3 MonoKernel on gfx942: status and next steps

## Where it stands (2026-10-15, evening)

Branch `akii96/vllm:rocm/minimax-m3-mono-gfx942`, one commit on the image base
`77154ca61`. Opt in with `--monokernel`. The FFN launch is the default;
`VLLM_MINIMAX_M3_MONO_WHOLE_LAYER=1` adds the attention launch.

Separate PR branch: `akii96/vllm:rocm/minimax-m3-indexer-fp8-fnuz` (`9a4b11e82b`),
the gfx942 fp8 (e4m3fnuz) index cache via `--attention-config
'{"indexer_kv_dtype":"fp8"}'`.

### Measured, MI325X TP4, EmbeddedLLM/MiniMax-M3-FP8-dynamic

128K prompt, decode only, fp8 index cache:

| | 4 req TPOT / tok/s | 8 req TPOT / tok/s |
|---|---|---|
| vLLM baseline | 11.47 ms / 349 | 15.15 ms / 529 |
| mono, FFN only | 8.97 / 446 (-22%, +28%) | 12.06 / 656 (-20%, +24%) |

GSM8K, full 1319 questions, chat template, 8-shot: baseline 0.9393, mono 0.9378
(24 vs 22 discordant, p=0.88). 0 corrupted outputs in every run.

DSpark speculative decoding (3 draft tokens), full GSM8K:

| | GSM8K | mean acceptance |
|---|---|---|
| baseline, 5 runs | 0.9378-0.9439 | 2.70-2.81 (mean 2.77) |
| mono before the numerics fixes, 4 runs | 0.9341-0.9378 | 2.71-2.74 (mean 2.72) |
| mono after, 2 runs | 0.9401, 0.9401 | 2.749, 2.747 |

Pairs after the fixes: 0.9401 vs 0.9401 (p=1.0) and 0.9401 vs 0.9378 (p=0.76).

### Numerics alignment with vLLM (found with in-server comparisons, fixed)
The FFN launch now matches vLLM's ops in four places the unit reference had missed:
1. The MoE all-reduce: AITER's custom all-reduce on gfx94x truncates its fp32
   sum to bf16 (was round-to-nearest).
2. The routed SwiGLU: `silu_and_mul_with_clamp` rounds `g * sigmoid(1.702 g)` to
   bf16 before the `* (u + 1)` product; the shared expert's `swiglu_oai_split`
   does not.
3. The shared expert's FP8 linears (`RowWiseTorchFP8ScaledMM`) quantize their
   input and mid to the format's max, 240; the routed MoE uses vLLM's 224.
4. (Earlier) the post-attention all-reduce + norm: truncating stores, the norm on the unrounded sum.

In-server, per layer, the FFN output now agrees with vLLM's to rel ~1e-3 (was
~3e-2 from the shared-expert range alone); the residual is bit-identical.

FFN kernel, checkpoint layers (`MONO_REAL=1 bench_ffn_mono.py`), us per layer:

| tokens | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| start of 10-15 | 41 | 87 | 143 | 255 |
| now | 38 | 82 | 127 | 227 |

The shared expert's second input quant cost ~2 us at 4-8 tokens.

### Where the time goes at 8 tokens (stage stamps, real weights)
route done 22 us -> up/gate done 88 (274 MB at ~4.2 TB/s) -> down done 123 ->
all-reduce. The MI325X reads ~4.75 TB/s; the up/gate stage is at the bandwidth
limit and the down stage is mostly prefetched under it. Lower bound ~117 us vs
~123 measured in the timeline build.

Tried and not kept (code in design/): plain-word mid hand-off with the quant in
the down task (slower: 134-139), down spread over all 304 CTAs (flat), prefetch
depth 2/4/5 (3 best), deferred collect, distinct-expert grouping with global B.

## Next steps, in order

1. **The 0-22 us window before routing** is the only idle HBM time left. Ideas:
   - Prefetch the shared expert's down weights there, not only its up/gate.
   - Speculative routed prefetch from the previous step's routing (decode steps
     route similarly); cancel-free since loads are reads.
   - Shorten norm -> router -> route: router at 18 us, route 22.
2. **Spec decode acceptance**: confirm parity with 2-3 more pairs on the fixed
   numerics (mono 2.75 vs baseline mean 2.77, within the baseline's spread).
   If a gap remains, extend the in-server comparison (`c/cmp_hook.py`) to the
   attention launch inputs.
3. **Attention launch** (only matters if whole-layer is to become the default):
   per-chunk local top-16 with a small merge (gfx950 `rank_blocks`/`select_long`);
   make whole-layer the default only if it beats FFN-only at 4 and 8 requests.
4. **Landing (PRs on akii96/vllm, user opens them):**
   - fp8 index cache: `rocm/minimax-m3-indexer-fp8-fnuz`, ready; description in
     design/fp8_indexer_pr.md. Its server numbers used a test-only overlay, so
     CI needs the C++ rebuild.
   - MonoKernel: `rocm/minimax-m3-mono-gfx942`, single commit; PR body from this
     file and the README.
5. **Tooling:** WaveScope with akii96 PRs #159/#160/#161 applied locally (the
   att.js conflict between #159 and #160 needs resolving upstream); capture
   (`capture_ffn_mono.py`, `c/ws_capture.sh`, `c/ws_brief.sh`); the fast loop
   `c/loop2.sh`; the in-server comparison `c/cmp_run2.sh` + `c/cmp_hook.py`.

## How to resume
- Book an idle hissu MI325X node as `aaknawaz`; recreate the container with
  `~/mono_m3/launch.sh`; run `c/setup_g43.sh` (node, fp8-index overlay, vllm
  overlay copy). `push.ps1` / `run_remote.ps1` target the node named in them.
