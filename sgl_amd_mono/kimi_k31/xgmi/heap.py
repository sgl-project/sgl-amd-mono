# SPDX-License-Identifier: Apache-2.0
"""Unified <= 0.4 GiB XGMI symmetric peer heap and comm stream (gfx950 TP8).

Layout inside one natively owned uncached HIP IPC heap (2 MiB granule):
  - Region 0 (decode): 2 parity halves x (RS + AG)             = 31.50 MiB
  - Region 1 (prefill_rs_ring): 4 ring slots x 28 MiB          = 112.00 MiB
  - Region 2 (prefill_ag): 2 parity halves x 35 MiB            = 70.00 MiB
  - Region 3 (flags): 16-byte (epoch, op, rows, shape) flags; RS (32 KiB) + AG (16 KiB) = 48 KiB
  Total: 214 MiB (0.209 GiB) per GPU -- no extra persistent serving weights.

Ownership contract: one XgmiHeap serves one serialized sequence of TP8
eager collectives from a single owning TP worker thread per rank. Multi-thread
sharing and CUDA/HIP graph capture or replay are explicitly rejected so ranks
cannot diverge on collective ordering or reuse torn-down mappings.
Upstream TP-lockstep invariant: SGLang's TP scheduler dispatches identical
``ForwardBatch`` metadata (``num_tokens``, capture state, layer order) to all
TP8 ranks per step; in addition, RS/AG arrival flags encode full 16-byte
(epoch, op_id, logical geometry) tags so mismatched ranks cannot silently
accept incompatible payloads.
"""

from __future__ import annotations

import contextlib
import math
import os
import threading
from typing import Optional

import msgspec
import torch
import torch.distributed as dist

SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
}

TP = 8

DECODE_MAX_ROWS = 192
DECODE_MAX_COLS = 10752  # 3584 latent + 7168 shared
DECODE_BLOCKS = 128

RS_MAX_CHUNK_ROWS = 2048
RS_MAX_COLS = 7168
RS_RING_SLOTS = 4
RS_BLOCKS = 64

PREFILL_MAX_ROWS = 8192
PREFILL_AG_HALF_BYTES = 35 << 20  # >= 8192 * 4224 B (2112-dim bf16 qkv_a)
PREFILL_AG_BLOCKS = 64

# Match wide.py's 900 s watchdog so allocator GC stalls on 32K prefill batches do not trap peers.
DEFAULT_DEADLINE_S = 900.0
MIN_DEADLINE_S = 1e-8
MAX_DEADLINE_S = 86400.0


def _validate_deadline_s(value: float) -> float:
    deadline = float(value)
    if (
        not math.isfinite(deadline)
        or deadline < MIN_DEADLINE_S
        or deadline > MAX_DEADLINE_S
    ):
        raise ValueError(
            f"XgmiHeap deadline_s must be finite and in [{MIN_DEADLINE_S}, {MAX_DEADLINE_S}], got {value!r}"
        )
    return deadline


_ALIGN = 4096


def _align_up(nbytes: int, alignment: int = _ALIGN) -> int:
    return (nbytes + alignment - 1) // alignment * alignment


# Lamport-tagged decode payload: each 16 B packet carries 4 bf16 plus 2 tags, so 4 B per bf16.
DECODE_RS_BYTES = _align_up(DECODE_MAX_ROWS * DECODE_MAX_COLS * 4)
DECODE_AG_BYTES = _align_up(DECODE_MAX_ROWS * DECODE_MAX_COLS * 4)
DECODE_HALF_BYTES = _align_up(DECODE_RS_BYTES + DECODE_AG_BYTES)
DECODE_REGION_OFF = 0
DECODE_REGION_BYTES = 2 * DECODE_HALF_BYTES

RS_SLOT_BYTES = _align_up(RS_MAX_CHUNK_ROWS * RS_MAX_COLS * 2)
RS_REGION_OFF = _align_up(DECODE_REGION_OFF + DECODE_REGION_BYTES)
RS_REGION_BYTES = RS_RING_SLOTS * RS_SLOT_BYTES

AG_REGION_OFF = _align_up(RS_REGION_OFF + RS_REGION_BYTES)
AG_REGION_BYTES = 2 * PREFILL_AG_HALF_BYTES

# 16-byte geometry-tagged arrival flags: RS 32 KiB + AG 16 KiB = 48 KiB
RS_FLAGS_OFF = _align_up(AG_REGION_OFF + AG_REGION_BYTES)
RS_FLAGS_BYTES = _align_up(RS_RING_SLOTS * RS_BLOCKS * TP * 16)
AG_FLAGS_OFF = _align_up(RS_FLAGS_OFF + RS_FLAGS_BYTES)
AG_FLAGS_BYTES = _align_up(2 * PREFILL_AG_BLOCKS * TP * 16)

HEAP_BYTES = _align_up(nbytes=AG_FLAGS_OFF + AG_FLAGS_BYTES, alignment=2 << 20)

# Per-block monotonic epoch table offsets (each block updates only its own word).
EPOCH_DECODE_OFF = 0
EPOCH_RS_OFF = DECODE_BLOCKS
EPOCH_AG_OFF = EPOCH_RS_OFF + RS_BLOCKS
EPOCH_TOTAL_WORDS = EPOCH_AG_OFF + PREFILL_AG_BLOCKS


class XgmiConfig(msgspec.Struct, frozen=True):
    """Immutable runtime settings resolved once at hook installation."""

    prefill_min_rows: int
    prefill_max_rows: int
    use_fp4: bool


class XgmiStreams:
    """Compute stream plus a high-priority XGMI comm stream."""

    def __init__(self, device: torch.device):
        self.device = device
        # No CU masking: ROCm 7.2 hipExtStreamCreateWithCUMask regresses hipBLASLt GEMMs ~2x.
        self.compute = torch.cuda.Stream(device=device)
        self.comm = torch.cuda.Stream(device=device, priority=-1)

    def close(self) -> None:
        pass


def _open_peer_heaps(*, native, heap_id: int, handles: list, rank: int, device) -> None:
    handle_rows: list[list[int]] = []
    for peer_rank, (peer_bytes, peer_handle) in enumerate(handles):
        if peer_bytes != HEAP_BYTES or peer_handle is None:
            raise RuntimeError(
                f"rank {peer_rank} local heap setup failed "
                f"(bytes={peer_bytes}, expected={HEAP_BYTES})"
            )
        handle_rows.append(peer_handle)
    peer_handles_cpu = torch.tensor(
        handle_rows, dtype=torch.uint8, device="cpu"
    ).contiguous()
    native.open_peer_heaps(heap_id, peer_handles_cpu, rank)
    # Synchronize local peer zeroing and epoch init before the TP vote so
    # the all_reduce after it acts as the single collective barrier.
    torch.cuda.synchronize(device)


def _abort_handshake(
    *,
    failure: Optional[Exception],
    native,
    heap_id: Optional[int],
    streams: Optional[XgmiStreams],
) -> None:
    cleanup_err: Optional[Exception] = None
    if heap_id is not None and native is not None:
        try:
            native.unregister_heap(heap_id)
        except Exception as err:
            cleanup_err = err
    if streams is not None and cleanup_err is None:
        streams.close()
    raise RuntimeError(
        f"XgmiHeap peer IPC handshake failed on {'this' if failure else 'another'} rank"
    ) from (failure if failure is not None else cleanup_err)


class XgmiHeap:
    """Single <= 0.4 GiB symmetric XGMI heap validated against TP8 gfx950."""

    def __init__(
        self,
        cpu_group,
        rank: int,
        device: Optional[torch.device] = None,
        *,
        deadline_s: Optional[float] = None,
    ):
        actual_world = dist.get_world_size(group=cpu_group)
        actual_rank = dist.get_rank(group=cpu_group)
        if actual_world != TP:
            raise ValueError(f"XgmiHeap requires TP={TP}, got {actual_world}")
        self.cpu_group = cpu_group
        self.rank = actual_rank
        self.world_size = actual_world
        self.device = device
        self.deadline_s = DEFAULT_DEADLINE_S

        native = None
        heap_id: Optional[int] = None
        local_payload: tuple[int, Optional[list[int]]] = (0, None)
        local_failure: Optional[Exception] = None
        streams: Optional[XgmiStreams] = None
        chunk_events: list[torch.cuda.Event] = []
        try:
            if rank != actual_rank:
                raise ValueError(
                    f"XgmiHeap rank mismatch: passed {rank}, group rank is {actual_rank}"
                )
            self.deadline_s = (
                _validate_deadline_s(
                    os.environ.get(
                        "SGLANG_AMD_MONO_K31_XGMI_DEADLINE_S",
                        str(DEFAULT_DEADLINE_S),
                    )
                )
                if deadline_s is None
                else _validate_deadline_s(deadline_s)
            )
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
                self.device = device
            arch = torch.cuda.get_device_properties(device).gcnArchName
            if not arch.startswith("gfx950"):
                raise RuntimeError(f"XgmiHeap requires gfx950, got {arch!r}")
            self.block_epoch = torch.zeros(
                EPOCH_TOTAL_WORDS, dtype=torch.int32, device=device
            )
            from sgl_amd_mono.kimi_k31.xgmi.kernels import ops

            native = ops()
            raw_id, local_handle = native.create_local_heap(
                self.block_epoch, HEAP_BYTES
            )
            heap_id = int(raw_id)
            streams = XgmiStreams(device)
            chunk_events = [torch.cuda.Event() for _ in range(8)]
            local_payload = (HEAP_BYTES, local_handle.tolist())
        except Exception as err:
            local_failure = err

        # Every rank participates in the handle gather even on local setup failure
        # so peers never hang waiting for a rank that failed JIT or allocation.
        handles: list = [None] * actual_world
        dist.all_gather_object(handles, local_payload, group=cpu_group)
        failure: Optional[Exception] = local_failure
        if failure is None:
            try:
                assert native is not None and heap_id is not None
                _open_peer_heaps(
                    native=native,
                    heap_id=heap_id,
                    handles=handles,
                    rank=actual_rank,
                    device=device,
                )
            except Exception as err:
                failure = err
        ok = torch.tensor([int(failure is None)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=cpu_group)
        if not ok.item():
            _abort_handshake(
                failure=failure, native=native, heap_id=heap_id, streams=streams
            )
        assert streams is not None
        self.streams = streams
        self.chunk_events = chunk_events
        self._state_lock = threading.Lock()
        self._owner_thread: Optional[int] = None
        self._active_scope = False
        self._active_launch = False
        self._closed = False
        self._last_stream: Optional[torch.cuda.Stream] = None
        self._heap_id: Optional[int] = heap_id

    def _check_owner_and_live_locked(self, tid: int) -> int:
        if self._closed:
            raise RuntimeError("XgmiHeap is closed")
        if self._owner_thread is None:
            self._owner_thread = tid
        elif self._owner_thread != tid:
            raise RuntimeError(
                "XgmiHeap must only be used from its owning TP worker thread"
            )
        if self._heap_id is None:
            raise RuntimeError("XgmiHeap native handle is not initialized")
        return self._heap_id

    @contextlib.contextmanager
    def overlap_guard(self):
        """Guard a multi-stream overlapped operation (events + comm stream + launches)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "XgmiHeap does not support CUDA/HIP graph capture or replay"
            )
        tid = threading.get_ident()
        with self._state_lock:
            self._check_owner_and_live_locked(tid)
            if self._active_scope or self._active_launch:
                raise RuntimeError(
                    "Concurrent or nested XgmiHeap overlapped operations are not supported"
                )
            self._active_scope = True
        try:
            yield
        finally:
            with self._state_lock:
                self._active_scope = False

    @contextlib.contextmanager
    def launch_guard(self):
        """Enforce single-thread eager-only serialization and live heap registration."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "XgmiHeap does not support CUDA/HIP graph capture or replay"
            )
        tid = threading.get_ident()
        with self._state_lock:
            heap_id = self._check_owner_and_live_locked(tid)
            if self._active_launch:
                raise RuntimeError(
                    "Concurrent or reentrant XgmiHeap collective launches are not supported"
                )
            cur = torch.cuda.current_stream(self.device)
            if self._last_stream is not None and self._last_stream != cur:
                cur.wait_stream(self._last_stream)
            self._last_stream = cur
            self._active_launch = True
        try:
            yield heap_id
        finally:
            with self._state_lock:
                self._active_launch = False

    @property
    def heap_bytes(self) -> int:
        return HEAP_BYTES

    @property
    def heap_mib(self) -> float:
        return HEAP_BYTES / (1 << 20)

    def close(self) -> None:
        global _GLOBAL_HEAP
        with self._state_lock:
            if self._closed:
                return
            if self._active_launch or self._active_scope:
                raise RuntimeError(
                    "Cannot close XgmiHeap while an XGMI operation is active"
                )
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cannot close XgmiHeap during CUDA/HIP graph capture"
                )
            self._closed = True
            heap_id = self._heap_id
            self._heap_id = None
            self._last_stream = None
            sync_err: Optional[Exception] = None
            try:
                torch.cuda.synchronize(self.device)
            except Exception as err:
                sync_err = err
            if heap_id is not None:
                from sgl_amd_mono.kimi_k31.xgmi.kernels import ops

                try:
                    ops().unregister_heap(heap_id)
                except Exception as err:
                    if sync_err is None:
                        sync_err = err
            if sync_err is None:
                self.streams.close()
            if _GLOBAL_HEAP is self:
                _GLOBAL_HEAP = None
            if sync_err is not None:
                raise sync_err


_GLOBAL_HEAP: Optional[XgmiHeap] = None


def get_global_heap(
    *,
    cpu_group=None,
    rank: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> XgmiHeap:
    """Return the process-wide singleton XgmiHeap (collective on first call)."""
    global _GLOBAL_HEAP
    if cpu_group is None:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        cpu_group = parallel.tp_group.cpu_group
        rank = parallel.tp_rank
    actual_world = dist.get_world_size(group=cpu_group)
    actual_rank = dist.get_rank(group=cpu_group)
    if actual_world != TP:
        raise ValueError(f"XgmiHeap requires tp_size={TP}, got {actual_world}")
    # Common TP preflight so mixed cache state (e.g. after non-collective close)
    # fails cleanly on all ranks instead of stranding ranks in different collectives.
    has_cached = int(_GLOBAL_HEAP is not None and not _GLOBAL_HEAP._closed)
    cache_state = torch.tensor([has_cached, -has_cached], dtype=torch.int32)
    dist.all_reduce(cache_state, op=dist.ReduceOp.MIN, group=cpu_group)
    all_cached = cache_state[0].item() == 1
    none_cached = cache_state[1].item() == 0
    if not all_cached and not none_cached:
        raise RuntimeError(
            "TP ranks have mixed XgmiHeap cache state (some closed, some open); "
            "close XgmiHeap on all ranks before reinitializing"
        )
    if none_cached:
        _GLOBAL_HEAP = XgmiHeap(
            cpu_group=cpu_group,
            rank=actual_rank if rank is None else rank,
            device=device,
        )
    else:
        assert _GLOBAL_HEAP is not None
        local_err: Optional[Exception] = None
        try:
            resolved_dev = (
                torch.device("cuda", torch.cuda.current_device())
                if device is None
                else device
            )
            if (
                _GLOBAL_HEAP.cpu_group is not cpu_group
                or (rank is not None and _GLOBAL_HEAP.rank != rank)
                or _GLOBAL_HEAP.device != resolved_dev
            ):
                raise ValueError(
                    f"XgmiHeap already initialized for rank={_GLOBAL_HEAP.rank}, "
                    f"device={_GLOBAL_HEAP.device}; requested rank={rank}, device={resolved_dev}"
                )
        except Exception as err:
            local_err = err
        ok = torch.tensor([int(local_err is None)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=cpu_group)
        if not ok.item():
            raise ValueError(
                f"XgmiHeap cached-heap validation failed on {'this' if local_err else 'another'} rank"
            ) from local_err
    return _GLOBAL_HEAP
