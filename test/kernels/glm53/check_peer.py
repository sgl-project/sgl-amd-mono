"""TP4 HIP IPC reads, allocator isolation, sibling lifetime and collective retry."""

import ctypes
import datetime
import os
import time

import pytest
import torch
import torch.distributed as dist

from sgl_amd_mono.common import peer


def check():
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    assert dist.get_world_size() == 4
    hip = peer._hip()
    hip.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    hip.hipMemcpy.restype = ctypes.c_int

    def read(buffer, iteration, slot):
        buffer.bytes.fill_(iteration * 16 + rank * 2 + slot)
        torch.cuda.synchronize()
        dist.barrier()
        for source, address in enumerate(buffer.addresses.cpu().tolist()):
            for offset in (0, buffer.bytes.numel() // 2, buffer.bytes.numel() - 1):
                value = ctypes.c_ubyte()
                peer._check(
                    hip.hipMemcpy(
                        ctypes.byref(value), ctypes.c_void_p(address + offset), 1, 2
                    ),
                    "hipMemcpy",
                )
                assert value.value == iteration * 16 + source * 2 + slot
        dist.barrier()

    for iteration in range(8):
        ordinary = torch.empty(62 << 20, dtype=torch.uint8, device=device)
        del ordinary
        prefix = torch.empty(40498688, dtype=torch.uint8, device=device)
        sizes = (512, 3 << 20) if iteration % 2 == 0 else (1572864, 13369344)
        buffers = [
            peer.PeerBuffer(size, dist.group.WORLD, rank, 4, device) for size in sizes
        ]
        assert len({prefix.data_ptr(), *(b.local for b in buffers)}) == 3
        for slot, buffer in enumerate(buffers):
            read(buffer, iteration, slot)
        if iteration == 0:
            original = peer._close_mapping
            failed = False

            def fail_once(base):
                nonlocal failed
                if rank == 0 and not failed:
                    failed = True
                    raise RuntimeError("injected close failure")
                original(base)

            peer._close_mapping = fail_once
            try:
                with pytest.raises(
                    peer.PeerMemoryError, match="injected close failure"
                ):
                    buffers[-1].close()
                assert buffers[-1].local and buffers[-1].bytes is not None
            finally:
                peer._close_mapping = original
        while buffers:
            buffer = buffers.pop()
            if rank == iteration % 4:
                time.sleep(0.05)
            buffer.close()
            buffer.close()
            assert buffer.local == 0
            del buffer
            if buffers:
                read(buffers[0], iteration, 0)
        del prefix
        torch.cuda.empty_cache()
    if rank == 0:
        print(
            'RESULT {"peer_lifecycle": "passed", "iterations": 8, "tp": 4}', flush=True
        )


if __name__ == "__main__":
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
    try:
        check()
    finally:
        dist.destroy_process_group()
