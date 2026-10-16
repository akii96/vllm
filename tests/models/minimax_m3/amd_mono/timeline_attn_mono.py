# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage timeline of the attention MonoKernel: per stamp, min / median / max of
microseconds since the launch's first CTA started, over the CTAs that wrote it.
Debug tool."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_attn_mono import FP8, attn_weights, make_case  # noqa: E402

NAMES = ["start", "norm", "score_end", "heads_in", "end", "select", "attend", "merge", "s_poll", "s_rank", "s_cands"]


def main(M=8, q_len=1, ctx=8192):
    from vllm.models.minimax_m3.amd.mono.runtime import MonoAttn

    dev = torch.device("cuda")
    c = make_case(dev, M, q_len, ctx, 0)
    m = MonoAttn(dev, c.block_table.shape[1], timeline=True)
    w = attn_weights(c.w)
    for _ in range(4):
        m.tl.zero_()
        m.run(w, c.x, c.residual, c.positions, c.slots, c.block_table, c.seq_lens,
              c.kv_cache.view(FP8), c.index_cache, q_len=q_len)
    torch.cuda.synchronize()
    raw = m.tl.cpu()[:, : len(NAMES)]
    t0 = raw[:, 0].min()
    cols = []
    for i, n in enumerate(NAMES):
        v = raw[:, i][raw[:, i] > 0]
        if len(v):
            v = (v - t0).double() / 100.0
            cols.append(f"{n}: {v.min():.1f}/{v.median():.1f}/{v.max():.1f}")
    print(f"M={M} q_len={q_len} ctx={ctx}: " + " | ".join(cols))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
