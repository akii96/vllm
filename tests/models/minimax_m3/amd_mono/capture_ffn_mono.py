# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A few warm launches of the FFN MonoKernel on a checkpoint layer, 4 GPUs, for
a profiler (rocprofv3 --att / --kernel-trace) to capture: exits cleanly so the
profiler flushes. Run: python capture_ffn_mono.py [width] [layer]."""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from test_ffn_mono import TP, load_weights  # noqa: E402


def worker(rank, m, layer):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29641")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=TP)
    from vllm.models.minimax_m3.amd.mono.runtime import MonoFFN

    w = load_weights(dev, rank, layer)
    mono = MonoFFN(TP, rank, dist.group.WORLD, dev)
    g = torch.Generator(device=dev).manual_seed(m)
    x = (torch.randn(m, 6144, device=dev, generator=g) * 0.3).bfloat16()
    res = (torch.randn(m, 6144, device=dev, generator=g) * 3).bfloat16()
    for _ in range(4):
        mono.run(w, x, res)
    torch.cuda.synchronize()
    dist.barrier()
    mono.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    mp.start_processes(worker, args=(m, layer), nprocs=TP, start_method="spawn")
