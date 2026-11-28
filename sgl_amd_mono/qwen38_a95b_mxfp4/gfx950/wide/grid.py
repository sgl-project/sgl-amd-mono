# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#67421 at 28f094565b (Apache-2.0):
# vllm/models/qwen3_5/amd/specialized_decode/wide/grid.py
"""A launch's identity and its synchronization: the grid barrier between
phases and the split-K fold's last-arriver count, both behind a bounded wait.

Grid barrier (striped): a CTA adds one to its
arrival stripe, one of `STRIPES` counters on separate 256-byte lines, four
CTAs a stripe; wave 0's 64 lanes poll every stripe at once, one load each, and
a ballot says when all reached the barrier's count. The counters are a launch
slot's: the k-th barrier of a launch waits for `k * 4` on every stripe, so
each launch must start from zeroed counters. Launches alternate between two
slots, and each zeroes the slot the next one uses (the launch before it is
done by then): `sync` is this launch's slot, `sync_next` the next one's.
A step of an odd launch count runs its last launch on a third slot, so the
next step starts on slot 0 again (`launch_slots`).
Every CTA must take every barrier of a launch: the kernels run one CTA per
CU, all co-resident.

Every wait runs under the plugin guard (`common.guard`, `common.sync._spin`):
past its deadline, or once a peer failed, the launch stops.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.layout import BLOCKS
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.ops import (
    CM_DEV,
    atomic_add,
    drain_stores,
    ld,
    st,
    traced,
    uniform,
)

STRIPES = 64  # arrival counters: wave 0's lanes poll them all at once
STRIPE_WORDS = 64  # one 256-byte line a counter
PER_STRIPE = BLOCKS // STRIPES
SLOT_WORDS = STRIPES * STRIPE_WORDS  # a launch slot's barrier counters
SLOTS = 3
FOLD_BASE = SLOTS * SLOT_WORDS  # the fold counters follow the slots' stripes
FOLD_STRIDE = 16  # a fold counter's words: one 64-byte line
FOLD_COUNTERS = 4096  # pieces from FOLD_PIECES: ~1.5k at 16 rows with K splits


def sync_words() -> int:
    """i32 words of a rank's sync buffer: the launch slots, then the fold
    counters."""
    return FOLD_BASE + FOLD_COUNTERS * FOLD_STRIDE


def launch_slots(n: int) -> list[tuple[int, int]]:
    """(`sync`, `sync_next`) slot indices of a step's `n` launches: the
    same every step (a CUDA graph bakes them), each launch zeroing the next
    one's slot, the last the next step's first's."""
    if n < 2:
        raise ValueError(f"{n} launches a step: the barrier slots need two")
    slot = [i % 2 for i in range(n)]
    if n % 2:
        slot[-1] = 2
    return [(slot[i], slot[(i + 1) % n]) for i in range(n)]


class TraceInvariant:
    """A handle never reassigned inside traced control flow: FlyDSL carries any
    local that is subscripted or has a method called in a dynamic loop or
    branch, and these reference only values defined before it, so they carry
    no IR values and rebuild as themselves."""

    def __extract_to_ir_values__(self):
        return []

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar):
        return exemplar


class Grid(TraceInvariant):
    """The CTA's identity and its synchronization words.

    `sync` / `sync_next`: this launch's barrier slot and the next launch's
    (byte addresses); `folds`: the fold counters; `scratch`: an LDS word the
    CTA's threads share verdicts in."""

    def __init__(self, tid, bid, sync, sync_next, folds, scratch):
        self.tid = fx.Int32(tid)
        self.bid = fx.Int32(bid)
        self.lane = self.tid % fx.Int32(64)
        self.wave = self.tid // fx.Int32(64)
        self.sync = fx.Int64(sync)
        self.sync_next = fx.Int64(sync_next)
        self.folds = fx.Int64(folds)
        self.scratch = scratch
        self.barriers = 0  # trace-time count of the launch's grid barriers
        # the arrival stripe: the XCD the CTA runs on (CTAs are dispatched
        # round-robin over the 8 XCDs), then its rank there
        self.stripe = (self.bid % fx.Int32(8)) * fx.Int32(8) + (
            self.bid // fx.Int32(8)
        ) % fx.Int32(8)
        # a phase of few independent items (a row, a (row, stream), a run of
        # 64 items) takes item i on CTA i % BLOCKS, wave i // BLOCKS: one wave
        # a CU first, so a CU's memory pipe queues one wave's loads where four
        # waves of a quarter of the CUs did. Loop
        # ``range(g.spread, items, BLOCKS * WAVES)``.
        self.spread = self.wave * fx.Int32(BLOCKS) + self.bid


@traced
def prologue(g):
    """The launch's first act: CTA 0 zeroes the next launch's barrier slot."""
    if (g.bid == 0) & (g.tid < STRIPES):
        st(g.sync_next, g.tid * STRIPE_WORDS, fx.Int32(0), cm=CM_DEV)


@traced
def _poll_stripes(g, target):
    """Wave 0: poll every stripe until all reached `target` (the poll loop
    stays one load a lane)."""
    lane_word = g.lane * STRIPE_WORDS  # the descriptor base stays wave-uniform
    sync = g.sync
    _spin(
        lambda: ld(sync, lane_word, fx.Int32, cm=CM_DEV),
        lambda v: fx.Int64(rocdl.ballot(T.i64, (v < target).ir_value())) != fx.Int64(0),
    )


def grid_sync(g, between=None):
    """Every CTA's stores so far visible to every CTA, then all CTAs past this
    point: drain, CTA barrier, one arrival on the CTA's stripe, wave 0 polls
    all stripes for this barrier's count. `between()`, when given, runs
    after the arrival and before the wait: work that reads no other CTA's
    stores and stores nothing another CTA reads, such as loads of data no CTA
    writes (the next phase's weights) or LDS work of the CTA's own; its
    result is returned. Wave 0's poll waits for its loads (`vmcnt` counts
    in issue order)."""
    g.barriers += 1
    _arrive(g)
    out = None if between is None else between()
    _wait(g, fx.Int32(g.barriers * PER_STRIPE))
    return out


@traced
def _arrive(g):
    drain_stores()
    fx.barrier()
    if g.tid == 0:
        atomic_add(g.sync + fx.Int64(g.stripe * (STRIPE_WORDS * 4)), 1)


@traced
def _wait(g, target):
    if uniform(g.wave) == 0:
        _poll_stripes(g, target)
    fx.barrier()
    fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)


@traced
def fold_arrive(g, counter, arrivals):
    """A split-K unit's arrival at fold counter `counter` after its partials:
    True on the CTA that arrives last of `arrivals`, which resets the counter
    for the next launch."""
    drain_stores()
    fx.barrier()
    if g.tid == 0:
        addr = g.folds + fx.Int64(counter * (FOLD_STRIDE * 4))
        old = atomic_add(addr, 1)
        last = old == fx.Int32(arrivals - 1)
        if last:
            st(addr, 0, fx.Int32(0), cm=CM_DEV)
        fx.ptr_store(last.select(fx.Int32(1), fx.Int32(0)), g.scratch)
    fx.barrier()
    verdict = fx.ptr_load(g.scratch) != fx.Int32(0)
    fx.barrier()
    return verdict
