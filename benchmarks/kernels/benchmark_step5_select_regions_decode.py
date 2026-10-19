# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused radix select vs sort-based top-k for Step-5 CSA decode.

Each sparse layer scores ``[tokens, groups, regions]`` region summaries and
picks the top regions per row; the sort-based reference launches two full
sorts, the fused kernel one radix select. Shapes follow Step-5 defaults
(region_block_size 8, topk 512), so a 32K context is 4096 regions per row.

Example:
    python benchmarks/kernels/benchmark_step5_select_regions_decode.py \
        --tokens 512 --regions 4096 --topk 512 --groups 1

"""

import argparse
import time

import torch

from vllm.models.step5.common.ops.csa import (
    select_regions,
    select_regions_fused,
)

REGION_BLOCK_SIZE = 8


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--groups", type=int, default=1, help="local groups per rank")
    parser.add_argument("--regions", type=int, default=4096, help="max_seq_len / 8")
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    scores = torch.randn(
        args.tokens, args.groups, args.regions, dtype=torch.float32, device=device
    )
    positions = torch.randint(
        0, args.regions * REGION_BLOCK_SIZE, (args.tokens,), device=device
    ).int()

    def run_reference():
        return select_regions(scores, positions, REGION_BLOCK_SIZE, args.topk)

    def run_fused():
        return select_regions_fused(scores, positions, REGION_BLOCK_SIZE, args.topk)

    torch.testing.assert_close(run_fused().cpu(), run_reference().cpu())

    ref_ms = _time(run_reference, args.iters, args.warmup)
    fused_ms = _time(run_fused, args.iters, args.warmup)
    print(
        f"tokens={args.tokens} groups={args.groups} regions={args.regions} "
        f"topk={args.topk}"
    )
    print(f"reference (two torch.sort) : {ref_ms:8.3f} ms")
    print(f"fused    (radix select)    : {fused_ms:8.3f} ms")
    print(f"speedup                    : {ref_ms / fused_ms:8.2f}x")


if __name__ == "__main__":
    main()
