#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep Triton DiffKV split-KV and launch-config tunables.

Usage:
    python benchmarks/kernels/benchmark_triton_diffkv_attention.py \
        --batches 1 8 64 --windows 128 1024 0

For each (heads, window, batch, kv-len) shape, runs the decode and prefill
attention through ``unified_attention_diffkv`` while overriding the split-KV
segment count and the single-wave 2D launch decision, and reports kernel
time in microseconds so the gate constants can be retuned per arch.
"""

import argparse

import torch

from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

HEAD_SIZES = (192, 128)
BLOCK_SIZE = 16


def make_batch(kv_lens, num_query_heads, num_kv_heads, query_lens=None):
    if query_lens is None:
        query_lens = [1] * len(kv_lens)
    blocks_per_seq = [(n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1 for n in kv_lens]
    num_blocks = sum(blocks_per_seq)
    kv_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        HEAD_SIZES[0] + HEAD_SIZES[1],
        dtype=torch.bfloat16,
    )
    block_tables = torch.zeros(
        len(kv_lens),
        max(blocks_per_seq),
        dtype=torch.int32,
    )
    start = 0
    for i, n in enumerate(blocks_per_seq):
        block_tables[i, :n] = torch.arange(
            start, start + n, dtype=torch.int32, device=kv_cache.device
        )
        start += n
    query = torch.randn(
        sum(query_lens), num_query_heads, HEAD_SIZES[0], dtype=torch.bfloat16
    )
    return query, kv_cache, block_tables, query_lens


def run(query, kv_cache, block_tables, kv_lens, query_lens, window, num_segments):
    num_query_heads = query.shape[1]
    head_size_qk, head_size_v = HEAD_SIZES
    num_seqs = len(kv_lens)
    out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=query.dtype)
    segm_output = torch.empty(
        num_seqs,
        num_query_heads,
        16,
        triton.next_power_of_2(head_size_v),
        dtype=torch.float32,
    )
    segm_max = torch.empty(num_seqs, num_query_heads, 16, dtype=torch.float32)
    segm_expsum = torch.empty(num_seqs, num_query_heads, 16, dtype=torch.float32)

    def kernel():
        unified_attention_diffkv(
            q=query,
            k=kv_cache[..., :head_size_qk],
            v=kv_cache[..., head_size_qk:],
            out=out,
            cu_seqlens_q=torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            ),
            seqused_k=torch.tensor(kv_lens, dtype=torch.int32),
            softmax_scale=head_size_qk**-0.5,
            causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            block_table=block_tables,
            softcap=0,
            max_seqlen_q=max(query_lens),
            seq_threshold_3D=num_seqs,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
        )

    return kernel


def bench(kernel, num_iters=100, warmup=20):
    for _ in range(warmup):
        kernel()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        kernel()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / num_iters * 1000


class Override:
    """Pin the split-KV gate outputs for the duration of a benchmark."""

    def __init__(self, module, num_segments=None, single_wave=None):
        self.module = module
        self.pins = {}
        if num_segments is not None:
            self.pins["_num_kv_segments"] = num_segments
        if single_wave is not None:
            self.pins["_use_single_wave_2d"] = single_wave
        self.saved = {}

    def __enter__(self):
        for attr, value in self.pins.items():
            self.saved[attr] = getattr(self.module, attr)
            setattr(self.module, attr, lambda *_, v=value: v)
        return self

    def __exit__(self, *exc):
        for attr, fn in self.saved.items():
            setattr(self.module, attr, fn)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DiffKV split-KV and single-wave launch configs"
    )
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--windows", type=int, nargs="+", default=[128, 1024, 0])
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--num-segments", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    parser.add_argument("--prefill-query-len", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "requires a GPU"
    torch.set_default_device("cuda")
    set_random_seed(0)

    import vllm.v1.attention.ops.triton_unified_attention_diffkv as diffkv

    hdr = f"{'batch':>6} {'window':>7} {'segments':>9} {'1-wave':>7} {'us':>10}"
    print(hdr)
    print("-" * len(hdr))

    for batch in args.batches:
        qlens = [args.prefill_query_len] * batch if args.prefill_query_len else None
        for window in args.windows:
            query, kv_cache, block_tables, run_qlens = make_batch(
                [args.kv_len] * batch,
                args.num_query_heads,
                args.num_kv_heads,
                qlens,
            )
            kernel = run(
                query,
                kv_cache,
                block_tables,
                [args.kv_len] * batch,
                run_qlens,
                window,
                16,
            )
            kernel()  # compile once before pinning segment count
            for num_segments in args.num_segments:
                for single_wave in (False, True):
                    with Override(
                        diffkv,
                        num_segments=num_segments,
                        single_wave=single_wave,
                    ):
                        us = bench(kernel, args.num_iters)
                    print(
                        f"{batch:>6} {window:>7} {num_segments:>9} "
                        f"{str(single_wave):>7} {us:>10.2f}"
                    )


if __name__ == "__main__":
    main()
