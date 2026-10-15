# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric peer memory for in-kernel TP reductions."""

import torch
import torch.distributed as dist


class MonoUnsupported(Exception):
    """The deployment is outside what a MonoKernel serves."""


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of the CPU ``group`` passed ``ok``. Collective."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class _DeviceBytes:
    """A non-owning ``__cuda_array_interface__`` over raw device bytes."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer per rank; ``addresses`` holds every rank's base (int64,
    indexed by TP rank). Construction is collective over ``group``."""

    def __init__(self, nbytes: int, group, rank: int, npes: int, device):
        from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

        self._heap = UncachedIpcHeap
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        self._opened: list[int] = []
        addresses, failure = [self.local], None
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses = []
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        self._opened.append(UncachedIpcHeap.open_mem_handle(handle))
                        addresses.append(self._opened[-1])
            except RuntimeError as err:
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise MonoUnsupported("peer memory handshake failed") from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.bytes.zero_()

    def close(self) -> None:
        opened, self._opened = self._opened, []
        for base in opened:
            self._heap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            self._heap.free_device_mem(self.local)
            self.local = 0
