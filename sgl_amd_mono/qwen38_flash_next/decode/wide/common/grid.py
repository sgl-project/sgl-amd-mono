# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# vllm/models/qwen4_exp/amd/specialized_decode/wide/common/grid.py
"""A launch's identity and its synchronization: the grid barrier between
phases, behind a bounded wait, and the split-K fold's last-arriver count.

Grid barrier (striped): a CTA adds one to its
arrival stripe, one of `STRIPES` counters on separate 256-byte lines, four
CTAs a stripe; wave 0's 64 lanes poll every stripe at once, one load each, and
a ballot says when all reached the barrier's count. The counters are a launch
slot's: the k-th barrier of a launch waits for `k * 4` on every stripe, so
each launch must start from zeroed counters. Launches alternate between two
slots, and each zeroes the slot the next one uses (the launch before it is
done by then): `sync` is this launch's slot, `sync_next` the next one's.
Every CTA must take every barrier of a launch: the kernels run one CTA per
CU, all co-resident.

Every wait gives up after the build's deadline (`Grid.ticks`, by default
`WAIT_TICKS`): the CTA records the site in its health record, raises its
rank's flag on the output rank (`alarm`) and sets the rank's FAILED word, so
its later waits, and every other CTA's, exit at once and the launch runs to
its end. The host reads FAILED after the step.

Stop mode (`Grid.stop`, SGLang's setting): a wait checks every `CHECK_POLLS`
polls. Past its deadline it raises its rank's word in every other rank's flags
(`Grid.watch`), drains and faults the launch at its site's page
(`guard.stop_launch`: 0x1000 a grid barrier, 0x2000 a TP exchange line, 0x3000 a
tagged hand-off between CTAs); a peer's word set in its rank's flags faults it at
`SITE_FLAG`'s page, 0x4000, so after a timeout on one rank every other rank
faults at its first wait for the stopped rank's data. A launch never runs on with
words that did not come. `Grid.inject` stalls one wait for the fault injection
(`guard.fault_injection`).

The runtime reports a queue's first fault, so the waits that give up first name
the stall: a rank never raises its own word (its GPU would fault at the flag's
page first), and the deadlines are staggered (`deadlines`) so that the first wait
to give up is of the kind furthest upstream. A hand-off from this GPU (a tagged
hand-off; an exchange line, while only this rank's lines are missing) gives up at
the build's deadline, a grid barrier a grace later, and an exchange wait missing
a peer's line two graces later. A stuck GPU's page then names the hand-off that
never came (0x3000, 0x2000), or a grid barrier only when no hand-off or line of
its own was missing (a CTA that never arrived, as when not every CTA is resident,
or one waiting on a peer's line); a GPU waiting on a stuck peer faults at the
flag's page.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.guard import stop_launch
from sgl_amd_mono.common.sync import CHECK_POLLS
from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import (
    FAILED,
    RECORD_WORDS,
    SITE_FLAG,
    SITE_GRID,
    WAIT_TICKS,
)
from sgl_amd_mono.qwen38_flash_next.decode.wide.common.ops import (
    CM_DEV,
    CM_SYS,
    atomic_add,
    drain_stores,
    ld,
    memrealtime,
    st,
    traced,
    uniform,
)
from sgl_amd_mono.qwen38_flash_next.decode.wide.layout import BLOCKS, WAVES

STRIPES = 64  # arrival counters: wave 0's lanes poll them all at once
STRIPE_WORDS = 64  # one 256-byte line a counter
PER_STRIPE = BLOCKS // STRIPES
SLOT_WORDS = STRIPES * STRIPE_WORDS  # a launch slot's barrier counters
FOLD_BASE = 2 * SLOT_WORDS  # the fold counters follow both slots' stripes
FOLD_STRIDE = 16  # a fold counter's words: one 64-byte line
FOLD_COUNTERS = 512
# stop mode's deadline grace, an eighth of the deadline (`deadlines`): far above a
# wait's check period (`CHECK_POLLS` polls, microseconds) and the start skew between
# a wait and the waits behind it (within a launch, milliseconds), and the latest
# deadline, 1.25 times SGLang's (half the watchdog, at most 60 s), stays below the
# watchdog
GRACES = 8


def sync_words() -> int:
    """i32 words of a rank's sync buffer: two launch slots, then the fold
    counters."""
    return FOLD_BASE + FOLD_COUNTERS * FOLD_STRIDE


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
    (byte addresses); `folds`: the fold counters; `health`: the rank's
    health words; `alarm`: this rank's flag word on the output rank (0:
    none); `scratch`: an LDS word the CTA's threads share verdicts in;
    `ticks`: a wait's deadline in `s_memrealtime` ticks; `stop`: stop mode
    (the module docstring)."""

    def __init__(
        self,
        tid,
        bid,
        sync,
        sync_next,
        folds,
        health,
        alarm,
        scratch,
        *,
        ticks: int = WAIT_TICKS,
        stop: bool = False,
    ):
        self.tid = fx.Int32(tid)
        self.bid = fx.Int32(bid)
        self.lane = self.tid % fx.Int32(64)
        self.wave = self.tid // fx.Int32(64)
        self.sync = fx.Int64(sync)
        self.sync_next = fx.Int64(sync_next)
        self.folds = fx.Int64(folds)
        self.health = fx.Int64(health)
        self.alarm = fx.Int64(alarm)
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
        # waves of a quarter of the CUs did. ``mirror``: CTA BLOCKS - 1 - i %
        # BLOCKS, wave WAVES - 1 - i // BLOCKS, for a phase that shares the
        # CTAs with another. Loop ``range(g.spread, items, BLOCKS * WAVES)``.
        self.spread = self.wave * fx.Int32(BLOCKS) + self.bid
        self.mirror = (fx.Int32(WAVES - 1) - self.wave) * fx.Int32(BLOCKS) + (
            fx.Int32(BLOCKS - 1) - self.bid
        )
        self.ticks, self.stop = ticks, stop
        self.every, self.flags, self.rank = [], None, None  # `watch`
        self.fault = None  # `inject`
        self.waits = 0  # trace-time count of the launch's stop-mode waits

    def watch(self, every, flags, rank):
        """Stop mode across ranks: `every` rank's flag words (an int32 a rank,
        rank order), `flags` this rank's, `rank` its index."""
        assert self.stop and len(every) <= 4, "stop mode, at most 4 ranks"
        self.every = [fx.Int64(f) for f in every]
        self.flags, self.rank = fx.Int64(flags), fx.Int32(rank)

    def inject(self, site, after, epoch):
        """The fault injection: stop-mode wait `site` (1: the launch's first,
        counted in trace order, a wait in a traced loop once) polls on until
        its deadline once `epoch` reaches `after`."""
        assert self.stop and site > 0, "the injected stall ends in stop mode's fault"
        self.fault = (site, fx.Int32(epoch) >= fx.Int32(after))

    def next_wait(self):
        """The next stop-mode wait's stall: the fault injection's armed flag at
        its site, else None."""
        self.waits += 1
        if self.fault is not None and self.fault[0] == self.waits:
            return self.fault[1]
        return None


@traced
def give_up(g, site, seen, target):
    """A wait that timed out: this CTA's record (once), the alarm, FAILED."""
    rec = (g.bid + 1) * RECORD_WORDS
    if ld(g.health, rec, fx.Int32, cm=CM_SYS) == 0:
        at = fx.Int32(memrealtime() & fx.Int64(0x7FFFFFFF))
        words = [fx.Int32(1), fx.Int32(site), fx.Int32(seen), fx.Int32(target)]
        st(g.health, rec, fx.Vector.from_elements(words, fx.Int32), cm=CM_SYS)
        tail = [g.bid, at, fx.Int32(0), fx.Int32(0)]
        st(g.health, rec + 4, fx.Vector.from_elements(tail, fx.Int32), cm=CM_SYS)
    if g.alarm != fx.Int64(0):
        st(g.alarm, 0, fx.Int32(1), cm=CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    st(g.health, FAILED, fx.Int32(1), cm=CM_SYS)


def failed(g):
    return ld(g.health, FAILED, fx.Int32, cm=CM_SYS) != fx.Int32(0)


@traced
def fail_stop(g, site):
    """A stop-mode wait at `site` past its deadline: its rank's word in every
    other rank's flags, drained, then the launch's fault at page `site`."""
    for r in range_constexpr(len(g.every)):
        if g.rank != fx.Int32(r):
            st(g.every[r], g.rank, fx.Int32(1), cm=CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    stop_launch(site)


def peers_failed(g):
    """Whether any peer's word in this rank's flags is set (system scope; two
    to four ranks)."""
    words = ld(g.flags, 0, fx.Int32, 2 if len(g.every) == 2 else 4, cm=CM_SYS)
    acc = words[0]
    for e in range(1, words.numel):
        acc = acc | words[e]
    return acc != fx.Int32(0)


def pending(pend, stall):
    """A stop-mode wait's loop condition: `pend`, or the injected stall."""
    return pend if stall is None else pend | stall


def deadlines(g, site):
    """A stop-mode wait's deadlines in ticks (the module docstring): the wait's
    at `site` (the build's, a grid barrier's a grace later), and an exchange
    wait's while a peer's line is missing (two graces later)."""
    grace = g.ticks // GRACES
    return g.ticks + (grace if site == SITE_GRID else 0), g.ticks + 2 * grace


@traced
def check(g, t0, site, leader=None, peer=None):
    """A pending stop-mode wait's check (every `CHECK_POLLS` polls): past its
    deadline from `t0` (`deadlines`; `peer`: whether a peer's line is missing),
    `fail_stop` at `site` (on the `leader` lanes when given); a set word in this
    rank's flags faults the launch at `SITE_FLAG`. With no peer (one rank)
    nothing writes this rank's flags, so they are not read."""
    own, peers = deadlines(g, site)
    limit = fx.Int64(own)
    if const_expr(peer is not None):
        limit = peer.select(fx.Int64(peers), limit)
    late = memrealtime() - t0 > limit
    if const_expr(leader is not None):
        late = late & leader
    if late:
        fail_stop(g, site)
    if const_expr(len(g.every) > 1):
        if peers_failed(g):
            stop_launch(SITE_FLAG)


@traced
def prologue(g):
    """The launch's first act: CTA 0 zeroes the next launch's barrier slot."""
    if (g.bid == 0) & (g.tid < STRIPES):
        st(g.sync_next, g.tid * STRIPE_WORDS, fx.Int32(0), cm=CM_DEV)


@traced
def _poll_stripes(g, target):
    """Wave 0: poll every stripe until all reached `target`, bounded. A rank
    that failed before this barrier does not wait; one that fails while this
    CTA waits lets it time out on its own (the poll loop stays one load)."""
    lane_word = g.lane * STRIPE_WORDS  # the descriptor base stays wave-uniform
    t0 = memrealtime()
    seen = ld(g.sync, lane_word, fx.Int32, cm=CM_DEV)
    behind = _behind(seen, target)
    why = failed(g).select(fx.Int32(2), fx.Int32(0))
    while (behind != fx.Int64(0)) & (why == 0):
        seen = ld(g.sync, lane_word, fx.Int32, cm=CM_DEV)
        behind = _behind(seen, target)
        late = memrealtime() - t0 > fx.Int64(g.ticks)
        why = late.select(fx.Int32(1), fx.Int32(0))
    if (why == 1) & (g.lane == 0):
        give_up(g, SITE_GRID, seen, target)


@traced
def _poll_stripes_stop(g, target, stall):
    """`_poll_stripes` in stop mode (lane 0 fails a late wait)."""
    lane_word = g.lane * STRIPE_WORDS
    t0 = memrealtime()
    behind = _behind(ld(g.sync, lane_word, fx.Int32, cm=CM_DEV), target)
    polls = fx.Int32(0)
    while pending(behind != fx.Int64(0), stall):
        behind = _behind(ld(g.sync, lane_word, fx.Int32, cm=CM_DEV), target)
        polls = polls + 1
        if (polls & fx.Int32(CHECK_POLLS - 1)) == fx.Int32(0):
            check(g, t0, SITE_GRID, g.lane == 0)


def _behind(seen, target):
    """The lanes whose stripe is short of `target` (a wave ballot)."""
    return fx.Int64(rocdl.ballot(T.i64, (seen < target).ir_value()))


def _poll(g, target):
    if g.stop:
        _poll_stripes_stop(g, target, g.next_wait())
    else:
        _poll_stripes(g, target)


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
        _poll(g, target)
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
