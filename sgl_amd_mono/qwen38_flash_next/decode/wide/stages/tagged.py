# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# vllm/models/qwen4_exp/amd/specialized_decode/wide/stages/tagged.py
"""(value, tag) hand-offs between the CTAs of one launch, without a grid
barrier.

A producer stores each 32-bit value next to a tag, a pair a 64-bit half of
one store (`put`); a consumer re-loads its pairs until every tag is the
hand-off's (`poll`). A hand-off then costs one round trip after the
producer's store lands: no store drain, flag or grid barrier. A tag is unique
to a step, a layer and a hand-off of that layer, so no pair from an earlier
step or layer passes, and a buffer holds one hand-off's pairs at a time.
Producers and consumers must not wait on each other in a cycle; every CTA of
the grid is co-resident.
"""

from __future__ import annotations

import flydsl.expr as fx

from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import SITE_FOLD
from sgl_amd_mono.qwen38_flash_next.decode.wide.common.grid import (
    CHECK_POLLS,
    check,
    failed,
    give_up,
    pending,
)
from sgl_amd_mono.qwen38_flash_next.decode.wide.common.ops import (
    CM_DEV,
    ld,
    memrealtime,
    st,
    traced,
)
from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import tiled


def i32(v):
    return fx.Int32(v)


def put(addr, first, values, tag):
    """`values` (32-bit, one or an even count) as (value, tag) pairs from pair
    index `first` of `addr`, two pairs a 16-byte store."""
    for h in range(-(-len(values) // 2)):
        words = []
        for v in values[2 * h : 2 * h + 2]:
            words += [v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else v, tag]
        st(
            addr,
            (first + i32(2 * h)) * i32(2),
            fx.Vector.from_elements(words, fx.Int32),
            CM_DEV,
        )


def _loads(specs):
    out = []
    for addr, first, n in specs:
        for h in range(-(-n // 2)):
            words = 2 * min(2, n - 2 * h)
            out.append(
                ld(addr, (first + i32(2 * h)) * i32(2), fx.Int32, words, cm=CM_DEV)
            )
    return out


def _stale(seen, tag):
    pend = None
    for w in seen:
        for e in range(1, w.numel, 2):
            bad = w[e] != tag
            pend = bad if pend is None else pend | bad
    return pend


def poll(g, specs, tag):
    """The values of every `(addr, first pair, pairs)` of `specs` (pairs 1, 2
    or 4), all re-loaded together until every tag is `tag`; bounded, a
    timeout recorded at `SITE_FOLD` (`grid`: or in stop mode a fault at its
    page). Returns per spec its value words (Int32)."""
    if g.stop:
        return _values(specs, _poll_stop(g, specs, tag, g.next_wait()))
    return _poll(g, specs, tag)


@traced
def _poll(g, specs, tag):
    seen = _loads(specs)
    t0 = memrealtime()
    pend = _stale(seen, tag)
    why = failed(g).select(i32(2), i32(0))
    while pend & (why == 0):
        seen = _loads(specs)
        pend = _stale(seen, tag)
        late = memrealtime() - t0 > fx.Int64(g.ticks)
        why = late.select(i32(1), i32(0))
    if why == 1:
        give_up(g, SITE_FOLD, seen[0][1], tag)
    return _values(specs, seen)


@traced
def _poll_stop(g, specs, tag, stall):
    seen = _loads(specs)
    t0 = memrealtime()
    pend = _stale(seen, tag)
    polls = i32(0)
    while pending(pend, stall):
        seen = _loads(specs)
        pend = _stale(seen, tag)
        polls = polls + 1
        if (polls & i32(CHECK_POLLS - 1)) == i32(0):
            check(g, t0, SITE_FOLD)
    return seen


def _values(specs, seen):
    """`poll`'s result from the polled pair words."""
    out, i = [], 0
    for _, _, n in specs:
        words = []
        for _h in range(-(-n // 2)):
            words += [seen[i][e] for e in range(0, seen[i].numel, 2)]
            i += 1
        out.append(words)
    return out


class Pairs(tiled.Source):
    """GEMM activations published as (bf16 pair, tag) pairs, a `[tokens, ld /
    2]` pair matrix at `addr`: a resident stage's items of a thread are
    polled together (`tiled.Gemm.fill_resident`)."""

    def __init__(self, g, addr, ld_elems: int, tag):
        self.g, self.addr, self.ld, self.tag = g, fx.Int64(addr), ld_elems, tag

    def issue_batch(self, coords):
        firsts = [(token * i32(self.ld) + k) // i32(2) for token, k in coords]
        got = poll(self.g, [(self.addr, f, 4) for f in firsts], self.tag)
        return [[fx.Vector.from_elements(w, fx.Int32)] for w in got]

    def convert(self, raw):
        return raw[0]
