# MiniMax-M3 FFN MonoKernel (ROCm, FlyDSL)

`--monokernel` runs the FFN half of each sparse MoE layer (3..59) of a pure
decode step of at most 8 rows, speculative verify included, as one persistent
launch per TP rank (RFC vllm-project/vllm#60904), after vLLM's attention. Every
other step runs the layer unchanged; both paths take and return the same
tensors at the layer boundary, so they interleave freely.

`ops/ffn.py`, from the attention's unreduced o_proj output and the residual to
the TP-reduced FFN output and the new residual:

- o_proj all-reduce in-kernel over peer memory, post-attention Gemma norm;
- fp32 router, sigmoid + bias top-4 (ties to the lower expert);
- routed and shared FP8 experts: up / gate GEMV, SwiGLU-OAI, per-row FP8 of
  the mids, down GEMV, routed weights x 2.0, + shared;
- MoE all-reduce.

From 5 rows the MoE runs by distinct expert: each expert's weights are read
once for every token routed to it (the gfx950 mono kernel's wide MoE,
ROCm/aiter#6047).

## Scope

- CDNA3+ (written and validated on MI325X gfx942); grid = CU count, one CTA a CU.
- TP4, per-channel FP8 checkpoints with per-token dynamic activations
  (e.g. `EmbeddedLLM/MiniMax-M3-FP8-dynamic`), TRITON FP8 MoE backend, shared
  expert unfused.
- Anything else is refused at startup with every reason (`layer.refusals`).

## Numerics

Each stage reproduces vLLM's op, rounding included (in-server, the layer's
output agrees with vLLM's to rel ~1e-4; the residual is bit-identical):

- the post-attention all-reduce + Gemma norm as AITER's fused kernel on gfx94x:
  rank-ordered fp32 sum, bf16 stores truncate, the norm reads the unrounded sum
  (a round-to-nearest residual drifts an ulp a layer, which costs speculative
  decoding acceptance);
- the routed experts' FP8 as `scaled_fp8_quant` (scale amax / 224, correctly
  rounded, floored; x / scale); the shared expert's linears as AITER's per-token
  quant (scale amax * f32(1/240), x * (1 / scale));
- SwiGLU-OAI (alpha 1.702, limit 7): routed as `silu_and_mul_with_clamp`
  (`g * sigmoid(1.702 g)` rounded to bf16 before `* (u + 1)`), shared as
  `swiglu_oai_split`;
- the MoE all-reduce as AITER's custom all-reduce: rank order, truncating.

## Tests (`tests/models/minimax_m3/amd_mono/`)

- `test_mono_gating.py`: CPU, refusals and step routing.
- `test_ffn_mono.py`: four GPUs, the launch vs a torch reference of vLLM's
  chain (`MONO_REAL_LAYER=<n>`: a checkpoint layer's weights).
- `bench_ffn_mono.py`, `timeline_ffn_mono.py`, `capture_ffn_mono.py`: per-layer
  timing, per-stage device-clock stamps, a capture driver for `rocprofv3 --att`
  (`MONO_REAL=1`: checkpoint layers).
