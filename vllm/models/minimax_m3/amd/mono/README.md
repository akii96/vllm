# MiniMax-M3 MonoKernels (ROCm, FlyDSL)

`--monokernel` runs each sparse MoE layer (3..59) of a pure decode step of at
most 16 rows, speculative verify included, as two persistent launches per TP
rank (RFC vllm-project/vllm#60904). Every other step runs the layer unchanged;
both paths take and return the same tensors at the layer boundary, so they
interleave freely.

| Launch | `ops/` | From | To |
|---|---|---|---|
| attention | `attn.py` | previous output + residual | this rank's unreduced o_proj output, new residual |
| FFN | `ffn.py` | that partial + residual | TP-reduced FFN output, new residual |

Attention: input Gemma norm, per-token FP8, qkv GEMV, head norm / partial RoPE,
K|V and index-K cache inserts, indexer block scores (bf16 or fp8 index cache),
top-16 block selection (each split ranks its share of the blocks, every
attention task merges the candidates, as the gfx950 mono kernel), sparse
attention (one selected block per task, log-sum-exp merge), o_proj GEMV.

FFN: o_proj all-reduce in-kernel over peer memory, post-attention Gemma norm,
fp32 router, sigmoid + bias top-4, routed and shared FP8 experts, MoE
all-reduce. From 5 rows the MoE runs by distinct expert: each expert's weights
are read once for every token routed to it (the gfx950 mono kernel's wide MoE,
ROCm/aiter#6047). Rows of CUDA-graph padding requests are not run through the
experts, so a padded step costs about its live rows.

## Scope

- CDNA3+ (written and validated on MI325X gfx942); grid = CU count, one CTA a CU.
- TP4, per-channel FP8 checkpoints with per-token dynamic activations
  (e.g. `EmbeddedLLM/MiniMax-M3-FP8-dynamic`), TRITON FP8 MoE backend, shared
  expert unfused, fp8 KV cache (Triton sparse attention), bf16 or fp8 index
  cache, `max_model_len` up to 256K.
- Anything else is refused at startup with every reason (`layer.refusals`).

## Numerics

Each stage reproduces vLLM's op, rounding included:

- cache inserts byte-identical to `fused_minimax_m3_qknorm_rope_kv_insert`,
  block selection identical to `minimax_m3_index_decode` (ties to the lower
  block);
- qkv / o_proj input FP8 as AITER's per-token quant (scale amax * f32(1/240),
  x * (1 / scale));
- the post-attention all-reduce + Gemma norm as AITER's fused kernel on gfx94x:
  rank-ordered fp32 sum, bf16 stores truncate, the norm reads the unrounded sum
  (the new residual is bit-identical; a round-to-nearest residual drifts an ulp
  a layer, which costs speculative decoding acceptance);
- the routed experts' FP8 as `scaled_fp8_quant` (scale amax / 224, correctly
  rounded, floored; x / scale); the shared expert's linears as AITER's
  per-token quant;
- SwiGLU-OAI (alpha 1.702, limit 7): routed as `silu_and_mul_with_clamp`
  (`g * sigmoid(1.702 g)` rounded to bf16 before `* (u + 1)`), shared as
  `swiglu_oai_split`;
- the MoE all-reduce as AITER's custom all-reduce: rank order, truncating.

## Tests (`tests/models/minimax_m3/amd_mono/`)

- `test_mono_gating.py`: CPU, refusals and step routing.
- `test_attn_mono.py`: one GPU, the attention launch vs vLLM's own decode ops
  (`MONO_FP8_INDEX=1`: fp8 index cache; `--bench`: per-layer timing).
- `test_ffn_mono.py`: four GPUs, the FFN launch vs a torch reference of vLLM's
  chain (`MONO_REAL_LAYER=<n>`: a checkpoint layer's weights).
- `bench_ffn_mono.py`, `timeline_*.py`, `capture_ffn_mono.py`: per-layer timing,
  per-stage device-clock stamps, a capture driver for `rocprofv3 --att`.
