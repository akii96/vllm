# MiniMax-M3 MonoKernels (ROCm, FlyDSL)

`--monokernel` runs each sparse MoE layer (3..59) of a pure decode step of at
most 16 rows, speculative verify included, as persistent launches per TP rank
(RFC vllm-project/vllm#60904). Every other step runs the layer unchanged; the paths
share every tensor at the layer boundary, so they interleave freely.

| Launch | `ops/` | From | To |
|---|---|---|---|
| attention | `attn.py` | previous output + residual | this rank's unreduced o_proj output, new residual |
| FFN | `ffn.py` | that partial + residual | TP-reduced FFN output, new residual |

Attention: input Gemma norm, per-token FP8, qkv GEMV, head norm / partial RoPE,
K|V and index-K cache inserts, indexer block scores (bf16 or fp8 MFMA) and a
radix top-16, sparse attention (one selected block per task, log-sum-exp
merge), o_proj GEMV. FFN: o_proj all-reduce (in-kernel, peer memory),
post-attention norm, fp32 router, sigmoid + bias top-4, routed and shared FP8
experts, MoE all-reduce.

The FFN launch is the default; `VLLM_MINIMAX_M3_MONO_WHOLE_LAYER=1` adds the
attention launch.

## Scope

- CDNA3+ (written and validated on MI325X gfx942); grid = CU count, one CTA a CU.
- TP4, per-channel FP8 checkpoints with per-token dynamic activations
  (e.g. `EmbeddedLLM/MiniMax-M3-FP8-dynamic`), TRITON FP8 MoE backend, shared
  expert unfused, fp8 KV cache (Triton sparse attention), bf16 or fp8 index cache.
- Anything else is refused at startup with every reason (`layer.refusals`).

## Numerics

Each stage reproduces vLLM's op, rounding included:

- per-token FP8 as `scaled_fp8_quant` (amax / 224 on fnuz, true division) for
  the routed experts; the shared expert's FP8 linears quantize to 240, the
  format's max, as vLLM's row-wise scaled-mm path;
- the post-attention all-reduce + Gemma norm as AITER's fused kernel on gfx94x:
  rank-ordered fp32 sum, bf16 stores truncate, the norm reads the unrounded sum
  (the new residual is bit-identical; a round-to-nearest residual drifts by an
  ulp a layer, which costs speculative-decoding acceptance);
- the MoE all-reduce as AITER's custom all-reduce: rank order, truncating;
- SwiGLU-OAI (alpha 1.702, limit 7): routed as `silu_and_mul_with_clamp`
  (`g * sigmoid(1.702 g)` rounded to bf16 before `* (u + 1)`), shared as
  `swiglu_oai_split` (unrounded); per-row FP8 of the expert mids, routed
  weights renormalized x 2.0;
- cache inserts byte-identical to `fused_minimax_m3_qknorm_rope_kv_insert`
  (fp8 index keys from fp32, index queries via bf16), block selection identical
  to `minimax_m3_index_decode` (ties to the lower block).

## Tests (`tests/models/minimax_m3/amd_mono/`)

- `test_mono_gating.py`: CPU, refusals and step routing.
- `test_attn_mono.py`: one GPU, the attention launch vs vLLM's own decode ops
  (`MONO_FP8_INDEX=1`: fp8 index cache).
- `test_ffn_mono.py`: four GPUs, the FFN launch vs a torch reference of vLLM's
  chain (`MONO_REAL_LAYER=<n>`: a checkpoint layer's weights).
- `bench_*`, `timeline_*`: per-layer timing and per-stage device-clock stamps.
