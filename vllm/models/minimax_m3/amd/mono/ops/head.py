# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""Per-head Gemma RMSNorm + partial NeoX RoPE on a half wave, as vLLM's
``fused_minimax_m3_qknorm_rope_kv_insert``: lane l of the 32 holds dims
4l .. 4l + 4; the RoPE partner is ``ROTARY // 2 / 4`` lanes away."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from vllm.models.common.mono.device import bf_hi, bf_lo, butterfly, hw_rsq, xshfl

HEAD_DIM = 128
ROTARY = 64
ELEMS = HEAD_DIM // 32
PARTNER = ROTARY // 2 // ELEMS


def unpack4(words):
    """Two packed bf16 words -> 4 f32."""
    return [f(words[j]) for j in range(2) for f in (bf_lo, bf_hi)]


def norm_rope(x, lane32, gamma, cos, sin, eps):
    """x, gamma, cos, sin: this lane's 4 f32 (cos / sin of its rotary pair
    index). -> 4 f32, normed then roped."""
    ss = fx.Float32(0.0)
    for v in x:
        ss = ss + v * v
    rstd = hw_rsq(butterfly(ss, (16, 8, 4, 2, 1)) / float(HEAD_DIM) + eps)
    y = [x[i] * rstd * (gamma[i] + 1.0) for i in range(ELEMS)]
    partner = [xshfl(v, PARTNER) for v in y]
    first = lane32 * ELEMS < ROTARY // 2
    in_rope = lane32 * ELEMS < ROTARY
    out = []
    for i in range_constexpr(ELEMS):
        rot = first.select(
            y[i] * cos[i] - partner[i] * sin[i], y[i] * cos[i] + partner[i] * sin[i]
        )
        out.append(in_rope.select(rot, y[i]))
    return out
