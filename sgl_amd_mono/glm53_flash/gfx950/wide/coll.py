# SPDX-License-Identifier: Apache-2.0
"""Two-shot all-reduce of the wide launch's o_proj partial [rows, HIDDEN] over ``tp``
ranks, in peer memory (``common.peer.PeerBuffer``, two parity halves).

Shot 1 (RS): the o_proj tile epilogue (Kimi's ``_push_band_flag``) pushes a band of
rb rows x bn columns of this rank's partial into the column owner's ``rs`` region
[src][rows][sl] (sl = HIDDEN / tp), then flags (src, piece) in the owner's ``fl``.
Shot 2 (AG): the owner sums a piece's tp sources in rank order (f32, then bf16, as
the custom all-reduce does) and stores the sum into every rank's ``ag`` region
[rows][HIDDEN], then flags (owner, piece) in every rank's ``agf``. Every rank thus
holds the same bits.

One-shot (``oneshot``): the epilogue pushes the band into every rank's ``os`` region
[src][rows][HIDDEN] and flags (src, piece) in every rank's ``osf``; the reader sums
the tp sources in rank order itself (the same bits as shot 2's). One hop fewer, tp
times shot 1's bytes on every link.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, CM_SYS, bf2_f32, bf16_pair, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN


def pieces(rows, rb, bn, tp):
    """An owner's pieces: row bands of rb x column tiles of bn in its slice."""
    return -(-rows // rb) * ((HIDDEN // tp) // bn)


def full_pieces(rows, rb, bn):
    """A source's one-shot pieces: row bands of rb x column tiles of bn over HIDDEN."""
    return -(-rows // rb) * (HIDDEN // bn)


def layout(rows, rb, bn, tp, oneshot=False, t1=False):
    """Byte offsets of one parity half and the half's size. ``t1``: a second two-shot
    set (``t_`` names) for the launch's entry all-reduce."""
    sizes = {}
    if t1:
        sl = HIDDEN // tp
        sizes["t_rs"] = tp * rows * sl * 2
        sizes["t_ag"] = rows * HIDDEN * 2
        sizes["t_fl"] = tp * BLOCKS * 8
        sizes["t_agf"] = tp * pieces(rows, rb, bn, tp) * 8
    if oneshot:
        sizes["os"] = tp * rows * HIDDEN * 2
        sizes["osf"] = tp * full_pieces(rows, rb, bn) * 8
    else:
        sl = HIDDEN // tp
        sizes["rs"] = tp * rows * sl * 2
        sizes["ag"] = rows * HIDDEN * 2
        sizes["fl"] = tp * BLOCKS * 8
        sizes["agf"] = tp * pieces(rows, rb, bn, tp) * 8
    offs, w = {}, 0
    for k, v in sizes.items():
        offs[k] = w
        w += v
    return offs, (w + 255) // 256 * 256


def peer_bytes(rows, rb, bn, tp, oneshot=False, t1=False):
    return 2 * layout(rows, rb, bn, tp, oneshot, t1)[1] + guard.FLAG_BYTES


def prefixed(c, pre, names=("rs", "ag", "fl", "agf")):
    """A view of region context ``c`` whose ``<name>_of`` / ``<name>_own`` are ``c``'s
    ``<pre><name>`` ones (the shared helpers then work on a second region set)."""
    v = _View({k: c[k] for k in ("tid", "bid", "rank", "put", "put_words", "poll")})
    for n in names:
        v[f"{n}_of"] = c[f"{pre}{n}_of"]
        v.lazy[f"{n}_own"] = lambda n=n: c[f"{pre}{n}_own"]
    return v


class _View(dict):
    """``_own`` entries formed at each use, as in ``region_context``."""

    def __init__(self, *a):
        super().__init__(*a)
        self.lazy = {}

    def __getitem__(self, k):
        if not dict.__contains__(self, k) and k in self.lazy:
            return self.lazy[k]()
        return super().__getitem__(k)

    def __contains__(self, k):
        return dict.__contains__(self, k) or k in self.lazy

    def get(self, k, default=None):
        return self[k] if k in self else default


@flyc.jit
def put_each(c, tp, region, i, v):
    """Lane 0 of wave p puts (i, v) into rank p's ``region`` (p < tp)."""
    assert WAVES >= tp
    tid = c["tid"]
    p = uniform(tid // 64)
    if (p < tp) & (tid % 64 == 0):
        c["put"](c[f"{region}_of"](p), i, v)


@flyc.jit
def reduce_bcast(c, piece, m, rb, bn, tp):
    """Owner side of piece ``piece``: wait for the tp sources, sum in rank order,
    store the bf16 sum into every rank's ``ag``, then flag it there."""
    tid = c["tid"]
    sl = HIDDEN // tp
    q8 = bn // 8
    ct = sl // bn
    if tid < tp:
        c["poll"]([(c["fl_own"], tid * BLOCKS + piece, 1)])
    gpu.barrier()
    band, cj = piece // ct, piece % ct
    own = c["rank"]
    r_rs = rsrc(c["rs_own"].value)
    dsts = [rsrc(c["ag_of"](p).value) for p in range(tp)]
    for u in range(tid, rb * q8, THREADS):
        row = band * rb + u // q8
        col = cj * bn + 8 * (u % q8)
        if row < m:
            got = [
                fx.Vector(
                    bo.buffer_load(
                        r_rs,
                        (src * m * sl + row * sl + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                )
                for src in range(tp)
            ]
            acc = None
            for src in range_constexpr(tp):
                cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
                acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
            words = fx.Vector.from_elements(
                [
                    bf16_pair(acc[2 * j], acc[2 * j + 1]).bitcast(fx.Int32)
                    for j in range(4)
                ],
                fx.Int32,
            )
            for p in range_constexpr(tp):
                bo.buffer_store(
                    words,
                    dsts[p],
                    (row * HIDDEN + own * sl + col) // 2,
                    cache_modifier=CM_SYS,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    put_each(c, tp, "agf", own * pieces(m, rb, bn, tp) + piece, fx.Int32(1))


@flyc.jit
def push_all(c, acc, m, row0, rb, n0, bn, tp, piece):
    """One-shot: rows row0 .. row0 + rb, columns n0 .. n0 + bn of this rank's finished
    bf16 partial ``acc`` [m, HIDDEN] into every rank's ``os`` [rank][m][HIDDEN] (``acc``
    zeroed behind), drained, then flag (rank, piece) in every rank's ``osf``."""
    tid = c["tid"]
    q8 = bn // 8
    r = rsrc(acc)
    dsts = [rsrc(c["os_of"](p).value) for p in range(tp)]
    got = []
    for i in range_constexpr((rb * q8 + THREADS - 1) // THREADS):
        u = fx.min(tid + i * THREADS, rb * q8 - 1)
        row = row0 + u // q8
        col = n0 + 8 * (u % q8)
        w = fx.Vector(
            bo.buffer_load(
                r,
                (fx.min(row, m - 1) * HIDDEN + col) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        got.append((row, col, w, tid + i * THREADS < rb * q8))
    for row, col, w, live in got:
        if (row < m) & live:
            for p in range_constexpr(tp):
                bo.buffer_store(
                    w,
                    dsts[p],
                    ((c["rank"] * m + row) * HIDDEN + col) // 2,
                    cache_modifier=CM_SYS,
                )
            bo.buffer_store(
                fx.Vector.filled(4, 0, fx.Int32),
                r,
                (row * HIDDEN + col) // 2,
                cache_modifier=CM_DEV,
            )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if row0 < m:
        put_each(c, tp, "osf", c["rank"] * full_pieces(m, rb, bn) + piece, fx.Int32(1))
