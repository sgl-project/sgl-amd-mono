# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#67421 at 28f094565b (Apache-2.0):
# vllm/models/qwen3_5/amd/specialized_decode/wide/stages/tagged.py
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

from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.ops import (
    CM_DEV,
    ld,
    st,
    traced,
)


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
    """Every spec's (value, tag) words, in spec order, as one vector."""
    words = []
    for addr, first, n in specs:
        for h in range(-(-n // 2)):
            k = 2 * min(2, n - 2 * h)
            w = ld(addr, (first + i32(2 * h)) * i32(2), fx.Int32, k, cm=CM_DEV)
            words += [w[e] for e in range(k)]
    return fx.Vector.from_elements(words, fx.Int32)


def _stale(seen, tag):
    pend = None
    for e in range(1, seen.numel, 2):
        bad = seen[e] != tag
        pend = bad if pend is None else pend | bad
    return pend


@traced
def poll(g, specs, tag):
    """The values of every `(addr, first pair, pairs)` of `specs` (pairs 1, 2
    or 4), all re-loaded together until every tag is `tag`, under the guard.
    Returns per spec its value words (Int32)."""
    seen = _spin(lambda: _loads(specs), lambda v: _stale(v, tag))
    out, i = [], 0
    for _, _, n in specs:
        out.append([seen[i + 2 * e] for e in range(n)])
        i += 2 * n
    return out
