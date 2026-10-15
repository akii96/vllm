# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage timeline of the FFN MonoKernel (rank 0): per stamp, the median / max
over CTAs of microseconds since the earliest start. MONO_REAL=1: checkpoint
layers 20, 21 instead of random weights. Debug tool."""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from test_ffn_mono import TP, load_weights, make_weights  # noqa: E402

REAL = os.environ.get("MONO_REAL", "0") == "1"

NAMES = [
    "start",
    "push+norm",
    "router",
    "route",
    "ug done",
    "down done",
    "ar in",
    "x out",
]


def worker(rank, widths, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=TP)
    from vllm.models.minimax_m3.amd.mono.ops.ffn import H
    from vllm.models.minimax_m3.amd.mono.runtime import MonoFFN

    if REAL:
        ws = [load_weights(dev, rank, layer) for layer in (20, 21)]
    else:
        ws = [make_weights(dev, 1000 * i + rank) for i in range(4)]
    mono = MonoFFN(TP, rank, dist.group.WORLD, dev, timeline=True)
    for m in widths:
        g = torch.Generator(device=dev).manual_seed(m)
        x = (torch.randn(m, H, device=dev, generator=g) * 0.3).bfloat16()
        res = (torch.randn(m, H, device=dev, generator=g) * 3).bfloat16()
        for it in range(6):
            mono.tl.zero_()
            mono.run(ws[it % len(ws)], x, res)
        torch.accelerator.synchronize()
        dist.barrier()
        if rank == 0:
            t = mono.tl.cpu().double() / 100.0  # 100 MHz -> us
            t = torch.where(t > 0, t - t[:, 0].min(), torch.full_like(t, float("nan")))
            line = " | ".join(
                f"{n}: {t[:, i].nanmedian():.1f}/{t[:, i].nan_to_num(-1).max():.1f}"
                for i, n in enumerate(NAMES)
            )
            print(f"width {m}: {line}", flush=True)
    mono.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ws = [int(a) for a in sys.argv[1:]] or [1, 4, 8]
    mp.start_processes(worker, args=(ws, 29613), nprocs=TP, start_method="spawn")
