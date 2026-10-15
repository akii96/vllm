# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer time of the FFN MonoKernel: N_LAYERS distinct layers (cold weights)
back to back in one HIP graph, 4 GPUs. Run: python bench_ffn_mono.py [widths]."""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from test_ffn_mono import TP, load_weights, make_weights  # noqa: E402

# MONO_REAL=1: checkpoint layers 20.. instead of random weights (fewer, as each
# rank holds the real experts)
REAL = os.environ.get("MONO_REAL", "0") == "1"
N_LAYERS = 4 if REAL else 8


def worker(rank, widths, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=TP)
    from vllm.models.minimax_m3.amd.mono.ops.ffn import H
    from vllm.models.minimax_m3.amd.mono.runtime import MonoFFN

    if REAL:  # checkpoint layers (real expert sizes and routing statistics)
        ws = [load_weights(dev, rank, 20 + i) for i in range(N_LAYERS)]
    else:
        ws = [make_weights(dev, 1000 * i + rank) for i in range(N_LAYERS)]
    mono = MonoFFN(TP, rank, dist.group.WORLD, dev)
    for m in widths:
        x = torch.randn(m, H, device=dev).bfloat16()
        res = torch.randn(m, H, device=dev).bfloat16()
        out = torch.empty_like(x)
        hm = torch.empty_like(x)

        def step():
            for w in ws:
                mono.run(w, x, res, out, hm)

        step()
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                step()
        torch.cuda.synchronize()
        dist.barrier()
        best = 1e9
        for _ in range(20):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            dist.barrier()
            best = min(best, e0.elapsed_time(e1) * 1000 / N_LAYERS)
        if rank == 0:
            print(f"width {m}: {best:.1f} us/layer", flush=True)
    mono.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ws = [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16]
    mp.start_processes(worker, args=(ws, 29612), nprocs=TP, start_method="spawn")
