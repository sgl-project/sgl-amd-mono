# SPDX-License-Identifier: Apache-2.0
"""K1: one Muse Spark layer's attention front, the normed input through the KV
cache write, for S = ``tokens`` rows (a token each; 8 consecutive rows a request)::

    x (bf16: the layer's attention input, the previous boundary's pre-norm)
      -> q | k | v | gate GEMV on the bf16 weight (fp32 WMMA accumulate, one
         bf16 rounding, as the split-K GEMM of ``qkv_proj_qk_norm_rope_cache``)
      -> q, k: per-head RMSNorm without weight (eps 1e-5, bf16); q then scaled by
         qk_scale_factor and rounded; then interleaved RoPE in the layers that
         have one, with bf16 cos / sin and each product rounded, as the fused
         kernel of the ROCm model does
      -> q, k, v, gate out (bf16); K / V -> the layer's page-16 NHD fp8 cache
         (unit KV scale, checked at bind; the sliding layers' slots through the
         SWA pool's full-to-SWA table)

Stage 1 is the GEMV tasks: row groups of ``G_TILES`` WMMA tiles x ``K_PARTS`` K
parts, exactly one a CTA (``layout._k1_split``); a task streams its weights (held
in registers over the token tiles, one token tile at a time) and its K part of
every token's x (straight from memory), its fp32 partials go to slot ``part`` of
the ``k1_part`` mailbox (a pair a (part, token, row)). Stage 2 is a head
task a q, k or v head on CTAs ``HT0`` .. and a gate task a q head after them, each
summing a row's sixteen slots in order (one poll round) and rounding once, 32
tokens a pass on 16 lanes of 4 dims. Dependencies only point
from stage 2 to stage 1 and all CTAs are co-resident, so every wait makes
progress.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    rsrc,
    uniform64,
    wmma_bf16,
    wmma_row_sum,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.muse_spark.gfx1250.config import (
    G_OFF,
    HEAD_DIM,
    HIDDEN,
    K_OFF,
    KV_K,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    V_OFF,
)
from sgl_amd_mono.muse_spark.gfx1250.device import (
    FLAG_STRIDE,
    PF_LINE,
    async_copy16,
    gstore_raw,
    l2_prefetch_gvs,
    lds_addr,
    lds_load16,
    poll_flags,
    stamp,
    wait_lds_dma,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    ACC,
    ELEMS,
    G_ROWS,
    G_TILES,
    GATE_TASKS,
    GEMV_TASKS,
    GT0,
    HEAD_TASKS,
    HT0,
    K_PART,
    K_PARTS,
    K_PER_WAVE,
    K_STEP,
    K_STEPS,
    PARTS,
    QT0,
    RED_FLOATS,
    ROWS,
    SECOND_TASKS,
    TILE,
    TOKEN_LANES,
    TOKENS_PER_PASS,
    token_passes,
    token_tiles,
)

KV5_PAGE = 64  # keys a page of the bf16 vectorized_5d cache (``stage_head``)
KV5_HEAD_BYTES = KV5_PAGE * HEAD_DIM * 2  # a (page, head) of K or of V
assert ELEMS == 4 and KV5_PAGE == 1 << 6


def gemv_k(lane, wave, kpart, kc):
    """The first k of step ``kc`` this lane streams, in K part ``kpart`` (each wave a
    contiguous K_PER_WAVE range of the part): lane half h holds k .. k + 16 of the
    step's 32, on both WMMA operands (any K relabeling shared by A and B is legal)."""
    return kpart * K_PART + wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)


def ld32(r, word, cm=0):
    """The 16 bf16 of a WMMA operand lane at dword ``word`` of buffer ``r``: k .. k + 8
    and k + 16 .. k + 24 (two 16 B loads; the two lane halves' loads are adjacent)."""
    a = fx.Vector(bo.buffer_load(r, word, vec_width=4, dtype=T.i32, cache_modifier=cm))
    b = fx.Vector(
        bo.buffer_load(r, word + 8, vec_width=4, dtype=T.i32, cache_modifier=cm)
    )
    return fx.Vector.from_elements(
        [a[i] for i in range(4)] + [b[i] for i in range(4)], fx.Int32
    )


def gemv_w_loads(c, rg, kpart, w):
    """GEMV task (``rg``, ``kpart``)'s weight loads, this lane's: row tile ``it``'s
    row l16, K_STEPS steps of 32 B, in flight. Row-major bf16, read once a layer
    (nontemporal). ``[tile][step]``. A row group past the last reads past the buffer:
    zeros, no traffic (a CTA without the task)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(w, QKVG_ROWS * HIDDEN * 2)
    return [
        [
            ld32(
                r_w,
                (
                    (rg * G_ROWS + it * ROWS + lane % 16) * HIDDEN
                    + gemv_k(lane, wave, kpart, kc)
                )
                // 2,
                CM_NT,
            )
            for kc in range(K_STEPS)
        ]
        for it in range(G_TILES)
    ]


def x_loads(c, r_x, tok, kpart, cm):
    """Token ``tok``'s x at this lane's k of every step (the WMMA B operand)."""
    lane, wave = c["lane"], c["wave"]
    return [
        ld32(r_x, (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2, cm)
        for kc in range(K_STEPS)
    ]


def tile_acc(w_tile, xb):
    """One row tile x token tile over this wave's K steps (fp32)."""
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    for kc in range_constexpr(len(xb)):
        acc = wmma_bf16(
            w_tile[kc].bitcast(fx.BFloat16), xb[kc].bitcast(fx.BFloat16), acc
        )
    return acc


def red_store(c, red, slot, acc):
    """This wave's accumulators of reduction set ``slot``."""
    fx.ptr_store(acc, red + (slot * RED_FLOATS + (c["wave"] * WAVE + c["lane"]) * ACC))


RING = 2  # a wave's weight tiles in LDS at once (async copies a tile ahead)
ROW_BYTES = K_PER_WAVE * 2  # a wave's bytes of a weight row (its K range)
CHUNKS = ROW_BYTES // 16  # 16 B chunks of it: one a lane per copy
TILE_BYTES = ROWS * ROW_BYTES
WAVE_BYTES = RING * TILE_BYTES  # a wave's LDS: its ring, then its accumulators
assert CHUNKS == WAVE and K_STEP * 2 == 64


def ring_copy(c, buf, row0, k0):
    """Copy weight rows row0 .. row0 + ROWS of this wave's K range (k0 ..) into its
    LDS tile ``buf`` (byte address): a row per instruction, lane l its 16 B chunk l,
    stored at chunk l ^ (row % 16) of the row (``ring_frag`` then reads the 16 rows of
    a chunk from 16 bank groups)."""
    lane = c["lane"]
    w = fx.Int64(c["args"]["w"])
    for j in range_constexpr(ROWS):
        g = w + fx.Int64(((row0 + j) * HIDDEN + k0) * 2) + fx.Int64(lane * 16)
        async_copy16(g, buf + j * ROW_BYTES + (lane ^ j) * 16)


def ring_frag(c, buf, kc):
    """The WMMA A operand of step ``kc`` from LDS tile ``buf``: row l % 16, k 8 h .. + 8
    and 16 + 8 h .. + 8 of the step (h = l / 16), the native operand layout."""
    lane = c["lane"]
    rr = lane % 16
    h = lane // 16
    a = lds_load16(buf + rr * ROW_BYTES + (((kc * 4 + h) ^ rr) * 16))
    b = lds_load16(buf + rr * ROW_BYTES + (((kc * 4 + 2 + h) ^ rr) * 16))
    return fx.Vector.from_elements(
        [a[i] for i in range(4)] + [b[i] for i in range(4)], fx.Int32
    )


# ``SGLANG_AMD_MONO_PF_K1=1``: a merged launch's K1 CTA prefetches the rest of its spread tiles (past the RING
# whose copies go out before the x wait) into GL2 on arrival. It arrives after its K3 work, well before the
# boundary's x; the tiles otherwise stream from HBM only once x is in.
PF_K1 = os.environ.get("SGLANG_AMD_MONO_PF_K1", "0") == "1"
TILE_LINES = (
    ROWS * ROW_BYTES // PF_LINE
)  # a wave's GL2 lines of a tile: ROW_BYTES of each of its ROWS rows
assert ROW_BYTES % PF_LINE == 0 and TILE_LINES % WAVE == 0


@flyc.jit
def ring_prefetch(c, kpart, lo, cnt, tiles):
    """GL2 prefetches of this wave's K range of spread tiles lo + it, it in ``tiles`` below ``cnt``: what
    ``ring_copy`` reads, a line a lane, TILE_LINES / WAVE instructions a tile. The base (tile lo's first row,
    SGPRs) and a lane's line offsets stay fixed; a tile is a constant further (the offset field)."""
    lane = c["lane"]
    k0 = kpart * K_PART + c["wave"] * K_PER_WAVE
    base = uniform64(fx.Int64(c["args"]["w"]) + fx.Int64((lo * ROWS * HIDDEN + k0) * 2))
    per_row = ROW_BYTES // PF_LINE
    offs = [
        ((i * WAVE + lane) // per_row) * (HIDDEN * 2)
        + ((i * WAVE + lane) % per_row) * PF_LINE
        for i in range(TILE_LINES // WAVE)
    ]
    for it in tiles:
        if fx.Int32(it) < cnt:
            for off in offs:
                l2_prefetch_gvs(base, off, it * ROWS * HIDDEN * 2)


def ring_issue(c, kpart, row0s, tiles):
    """Issue the ring copies of tiles ``tiles`` (indices into ``row0s``, at most RING of
    them past the last one consumed): tile ``it`` into this wave's slot ``it % RING``."""
    k0 = kpart * K_PART + c["wave"] * K_PER_WAVE
    mine = lds_addr(c["red"]) + c["wave"] * WAVE_BYTES
    for it in tiles:
        ring_copy(c, mine + (it % RING) * TILE_BYTES, row0s[it], k0)


@flyc.jit
def stage_gemv_ring(c, kpart, row0s, cnt, issued: int = 0):
    """GEMV of K part ``kpart`` over the row tiles starting at ``row0s`` (traced row
    indices; a tile at or past ``cnt`` holds any valid row and writes nothing), every
    token tile at once. Each wave streams its K range (K_PER_WAVE of the part) of the
    tiles' rows into its own LDS ring with async copies (a row's bytes a copy:
    contiguous reads), a tile ahead of the WMMAs (``issued``: the first tiles' copies
    already in flight, ``ring_issue``); lane l of wave w takes row l % 16 of a tile from
    the ring and token 16 t + l % 16 of x (32 B a step straight from memory, the native
    WMMA operand layout on both). The waves' accumulators of each (row tile, token
    tile) are summed in wave order through LDS and the part's fp32 partials go to slot
    ``kpart`` of the mailbox."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    n = len(row0s)
    tiles = token_tiles(S)
    assert n * tiles * WAVE * ACC * 4 <= WAVE_BYTES, (n, tiles)
    r_x = rsrc(a["x"])
    x_cm = c["x_cm"]
    mine = lds_addr(red) + wave * WAVE_BYTES
    ring_issue(c, kpart, row0s, range(issued, min(RING, n)))
    toks = [fx.min(t * TILE + lane % 16, S - 1) for t in range(tiles)]
    xb = [
        [
            ld32(r_x, (toks[t] * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2, x_cm)
            for kc in range(K_STEPS)
        ]
        for t in range(tiles)
    ]
    accs = [
        [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(tiles)] for _ in range(n)
    ]
    for it in range_constexpr(n):
        wait_lds_dma((min(n, RING + it) - it - 1) * ROWS)
        buf = mine + (it % RING) * TILE_BYTES
        for kc in range_constexpr(K_STEPS):
            f = ring_frag(c, buf, kc).bitcast(fx.BFloat16)
            for t in range_constexpr(tiles):
                accs[it][t] = wmma_bf16(f, xb[t][kc].bitcast(fx.BFloat16), accs[it][t])
        if const_expr(it + RING < n):
            rocdl.s_wait_dscnt(0)  # this tile's reads are done before its next copy
            ring_issue(c, kpart, row0s, [it + RING])
    stamp(c, 4)  # the WMMAs issued (thread 0's wave)
    # each wave's accumulators into its own LDS (its ring, done with)
    rocdl.s_wait_dscnt(0)
    for it in range_constexpr(n):
        for t in range_constexpr(tiles):
            fx.ptr_store(
                accs[it][t],
                red
                + (
                    wave * (WAVE_BYTES // 4)
                    + (it * tiles + t) * (WAVE * ACC)
                    + lane * ACC
                ),
            )
    gpu.barrier()
    stamp(c, 5)  # every wave's accumulators in LDS
    # the partials of the part: thread (row of the tiles, token), summed in wave order
    n_out = n * ROWS * S
    for p in range_constexpr(-(-n_out // THREADS)):
        idx = tid + p * THREADS
        r = idx % (n * ROWS)
        it = r // ROWS
        if (idx < n_out) & (it < cnt):
            tk = idx // (n * ROWS)
            rt = r % ROWS
            slot = it * tiles + tk // TILE
            v = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                v = v + fx.ptr_load(
                    red
                    + (
                        w * (WAVE_BYTES // 4)
                        + slot * (WAVE * ACC)
                        + (16 * (rt // 8) + tk % TILE) * ACC
                        + rt % 8
                    )
                )
            row0 = row0s[0]
            for j in range_constexpr(1, n):
                row0 = (it == j).select(row0s[j], row0)
            put(part, (kpart * MAX_TOKENS + tk) * QKVG_ROWS + row0 + rt, v)
    gpu.barrier()  # the LDS is reused by the next task / stage


@flyc.jit
def stage_gemv(c, rg, kpart, first=None):
    """GEMV task: K part ``kpart`` of row group ``rg`` (G_TILES WMMA tiles), through the
    per-wave LDS ring (``stage_gemv_ring``); with ``first`` (weights already in
    registers) or deferred norms, ``stage_gemv_regs``."""
    if const_expr(first is not None or c.get("defer", False)):
        stage_gemv_regs(c, rg, kpart, first)
        return
    stage_gemv_ring(
        c, kpart, [rg * G_ROWS + it * ROWS for it in range(G_TILES)], G_TILES
    )


def spread_rows(lo, cnt, maxt):
    """The row starts of a spread range's ``maxt`` tile slots (a slot past ``cnt``:
    row 0, valid memory that writes nothing)."""
    return [
        (fx.Int32(it) < cnt).select((lo + it) * ROWS, fx.Int32(0)) for it in range(maxt)
    ]


@flyc.jit
def stage_gemv_regs(c, rg, kpart, first=None):
    """``stage_gemv`` with the weights loaded straight into registers (``first``: a
    merged launch's, already in flight; else loaded here), a token tile at a time,
    the waves' accumulators summed in wave order through ``red``."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    x_cm = c["x_cm"]
    w = first if first is not None else gemv_w_loads(c, rg, kpart, a["w"])
    if const_expr(c.get("defer", False)):
        stage_gemv_deferred(c, rg, kpart, w)
        return
    for t in range_constexpr(tiles):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        xb = x_loads(c, r_x, tok, kpart, x_cm)
        rocdl.sched_barrier(0)
        for it in range_constexpr(G_TILES):
            red_store(c, red, it, tile_acc(w[it], xb))
        gpu.barrier()
        nt = min(TILE, S - t * TILE)
        n_out = G_ROWS * nt
        for p in range_constexpr(-(-n_out // THREADS)):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % G_ROWS
                tk = t * TILE + idx // G_ROWS
                v = wmma_row_sum(
                    red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // G_ROWS
                )
                put(part, (kpart * MAX_TOKENS + tk) * QKVG_ROWS + rg * G_ROWS + r, v)
        gpu.barrier()  # ``red`` is reused by the next token tile / task


def spread_tiles(c, bid):
    """A merged launch's GEMV spread (``c["late"]``: [(first CTA, count)] of the CTAs
    that reach K1 late, which take no GEMV work): K part ``bid % K_PARTS``'s row tiles
    in contiguous ranges over the other CTAs of that part, in CTA order. (first tile,
    tiles) of this CTA; (0, 0) for a late CTA."""
    T = QKVG_ROWS // ROWS
    r = bid % K_PARTS
    below = fx.Int32(0)
    total = fx.Int32(0)
    late = fx.Int32(0) == fx.Int32(1)
    for a0, n in c["late"]:
        first = (
            a0 + (r - a0 % K_PARTS + K_PARTS) % K_PARTS
        )  # the group's first CTA of part r
        cnt = (first < a0 + n).select((a0 + n - 1 - first) // K_PARTS + 1, fx.Int32(0))
        d = fx.max(bid - first, fx.Int32(0))
        below = below + fx.min((d + K_PARTS - 1) // K_PARTS, cnt)
        total = total + cnt
        late = late | ((bid >= a0) & (bid < a0 + n))
    n_early = BLOCKS // K_PARTS - total
    j = bid // K_PARTS - below
    lo = j * T // n_early
    hi = (j + 1) * T // n_early
    return lo, late.select(fx.Int32(0), hi - lo)


def spread_max_tiles(late) -> int:
    """The most row tiles a CTA of ``spread_tiles`` takes (its static unroll)."""
    T = QKVG_ROWS // ROWS
    most = 0
    for r in range(K_PARTS):
        n_late = sum(1 for a0, n in late for b in range(a0, a0 + n) if b % K_PARTS == r)
        most = max(most, -(-T // (BLOCKS // K_PARTS - n_late)))
    return most


def spread_w_loads(c, lo, cnt, kpart, w, maxt):
    """The weights of row tiles lo .. lo + cnt of K part ``kpart``, this lane's, ``maxt``
    tiles in flight (a tile past ``cnt`` reads past the buffer: zeros, no traffic)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(w, QKVG_ROWS * HIDDEN * 2)
    out = []
    for it in range(maxt):
        row = (it < cnt).select((lo + it) * ROWS + lane % 16, fx.Int32(QKVG_ROWS))
        out.append(
            [
                ld32(r_w, (row * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2, CM_NT)
                for kc in range(K_STEPS)
            ]
        )
    return out


@flyc.jit
def stage_gemv_spread(c, lo, cnt, kpart, w):
    """``stage_gemv`` on a spread range: row tiles lo .. lo + cnt of K part ``kpart``
    (weights ``w`` in flight, ``spread_w_loads``), the same WMMA, wave-order sums and
    partial slot as the task that held each tile; the reductions go G_TILES tiles a
    round through the reduction sets."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    red, put, part = c["red"], c["put"], c["part"]
    r_x = rsrc(c["args"]["x"])
    x_cm = c["x_cm"]
    maxt = len(w)
    for t in range_constexpr(token_tiles(S)):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        xb = x_loads(c, r_x, tok, kpart, x_cm)
        nt = min(TILE, S - t * TILE)
        for r0 in range_constexpr(0, maxt, G_TILES):
            nr = min(G_TILES, maxt - r0)
            for i in range_constexpr(nr):
                red_store(c, red, i, tile_acc(w[r0 + i], xb))
            gpu.barrier()
            n_out = nr * ROWS * nt
            for p in range_constexpr(-(-n_out // THREADS)):
                idx = tid + p * THREADS
                rr = idx % (nr * ROWS)
                if (idx < n_out) & (r0 + rr // ROWS < cnt):
                    tk = t * TILE + idx // (nr * ROWS)
                    v = wmma_row_sum(
                        red + (rr // ROWS) * RED_FLOATS, rr % ROWS, idx // (nr * ROWS)
                    )
                    put(
                        part,
                        (kpart * MAX_TOKENS + tk) * QKVG_ROWS + (lo + r0) * ROWS + rr,
                        v,
                    )
            gpu.barrier()  # ``red`` is reused by the next round / token tile


@flyc.jit
def stage_gemv_deferred(c, rg, kpart, w):
    """The GEMV task of a merged launch's K1 with the boundary's norms deferred: x =
    bf16(rstd2 npw (alpha st + beta bf16(pp rstd))), pp = bf16(u pfw), is rstd2 (c1 +
    rstd c2) with c1 = bf16(npw alpha st) (written by K2's boundary: ``x``) and c2 =
    bf16(npw beta pp) from the unpooled rows u of this part's columns (one rank's,
    polled from K3's ``unp`` under K3's tag); the c1 half's WMMAs and row sums run
    first (while u lands), then c2's; the token's two scalars (K3's boundary,
    ``nscal``) fold the halves into the partial: so the K1 waits for neither the
    boundary's x nor its flag."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    d_poll, d_unp, d_ns = c["d_poll"], c["d_unp"], c["d_nscal"]
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    r_fw, r_be, r_nw = rsrc(a["pfw"]), rsrc(a["beta2"]), rsrc(a["npw"])
    # this lane's per-k constants (16 values a step)
    fwv, bev, nwv = [], [], []
    for kc in range_constexpr(K_STEPS):
        k = gemv_k(lane, wave, kpart, kc)
        fwv.append(ld32(r_fw, k // 2).bitcast(fx.BFloat16).to(fx.Float32))
        bev.append(
            [
                fx.Vector(bo.buffer_load(r_be, k + 4 * h, vec_width=4, dtype=T.f32))
                for h in range(4)
            ]
        )
        nwv.append(ld32(r_nw, k // 2).bitcast(fx.BFloat16).to(fx.Float32))
    for t in range_constexpr(tiles):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        nt = min(TILE, S - t * TILE)
        n_out = G_ROWS * nt
        rounds = -(-n_out // THREADS)
        # ---- c1: the first half's WMMAs and row sums
        xb = x_loads(c, r_x, tok, kpart, CM_DEV)
        for it in range_constexpr(G_TILES):
            red_store(c, red, it, tile_acc(w[it], xb))
        gpu.barrier()
        r1 = []
        for p in range_constexpr(rounds):
            idx = fx.min(tid + p * THREADS, n_out - 1)
            r1.append(
                wmma_row_sum(
                    red + ((idx % G_ROWS) // ROWS) * RED_FLOATS,
                    (idx % G_ROWS) % ROWS,
                    idx // G_ROWS,
                )
            )
        gpu.barrier()
        # ---- c2 from the unpooled rows of this lane's token and steps
        # (two pairs a spec: a poll load is at most four dwords)
        got = d_poll(
            [
                (d_unp, (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2 + 2 * hf, 2)
                for kc in range(K_STEPS)
                for hf in range(4)
            ]
        )
        ub = []
        for kc in range_constexpr(K_STEPS):
            words = []
            for q in range_constexpr(8):
                lo, hi = bf2_f32(got[4 * kc + q // 2][q % 2])
                vals = []
                for h, u in ((0, lo), (1, hi)):
                    j = 2 * q + h
                    pp = bf16_round(u * fx.Float32(fwv[kc][j]))
                    vals.append(
                        fx.Float32(nwv[kc][j]) * fx.Float32(bev[kc][j // 4][j % 4]) * pp
                    )
                words.append(bf16_pair(vals[0], vals[1]).bitcast(fx.Int32))
            ub.append(fx.Vector.from_elements(words, fx.Int32))
        for it in range_constexpr(G_TILES):
            red_store(c, red, it, tile_acc(w[it], ub))
        gpu.barrier()
        # ---- the token's scalars, the fold, the partials
        for p in range_constexpr(rounds):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % G_ROWS
                tk = t * TILE + idx // G_ROWS
                r2 = wmma_row_sum(
                    red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // G_ROWS
                )
                sc = d_poll([(d_ns, 2 * tk, 2)])
                rstd, rstd2 = sc[0][0].bitcast(fx.Float32), sc[0][1].bitcast(fx.Float32)
                put(
                    part,
                    (kpart * MAX_TOKENS + tk) * QKVG_ROWS + rg * G_ROWS + r,
                    rstd2 * (r1[p] + rstd * r2),
                )
        gpu.barrier()  # ``red`` is reused by the next token tile / task


def half_sum(poll, part, tk, pb):
    """Token ``tk``'s elements ``pb`` .. ``pb + ELEMS`` of the projection: the K
    parts' fp32 partials (``PARTS`` slots a row) from the mailbox, summed in slot
    order, bf16."""
    gq = poll(
        [
            (part, (hf * MAX_TOKENS + tk) * QKVG_ROWS + pb + 2 * p, 2)
            for hf in range(PARTS)
            for p in range(2)
        ],
        batch=2 * PARTS,  # the parts' pairs in one round
    )
    out = []
    for j in range_constexpr(ELEMS):
        v = gq[j // 2][j % 2].bitcast(fx.Float32)
        for hf in range_constexpr(1, PARTS):
            v = v + gq[hf * 2 + j // 2][j % 2].bitcast(fx.Float32)
        out.append(bf16_round(v))
    return out[0], out[1], out[2], out[3]


@flyc.jit
def stage_head(c, h):
    """Head task ``h`` (q heads, then k heads, then v heads): every token's head
    from the halves' partials, token ``tid / 16`` (+ 32 a pass) on 16 lanes of
    ``ELEMS`` dims each. q and k: the per-head norm (q then the query scale) and
    RoPE; k and v: the fp8 cache write."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    poll, part = c["poll"], c["part"]
    a = c["args"]
    is_q = h < LOCAL_Q_HEADS
    is_k = (h >= LOCAL_Q_HEADS) & (h < LOCAL_Q_HEADS + LOCAL_KV_HEADS)
    kv_head = is_k.select(h - LOCAL_Q_HEADS, h - LOCAL_Q_HEADS - LOCAL_KV_HEADS)
    col0 = is_q.select(h * HEAD_DIM, is_k.select(K_OFF, V_OFF) + kv_head * HEAD_DIM)
    normed = is_q | is_k
    d0 = (lane % 16) * ELEMS
    use_rope, use_swa = a["use_rope"], a["use_swa"]
    slot64, swa64 = a["slot64"], a["swa64"]
    r_cs = rsrc(a["cos_sin"])
    KV_ROWS, SWA_ROWS = (
        c["kv_rows"],
        c.get("swa_rows", 0),
    )  # head-major pools' rows (0: NHD)
    KV5 = c.get(
        "kv5d", False
    )  # bf16 caches in SGLang's vectorized_5d layout, 64-key pages (``KV5_*``)
    assert not (KV5 and (KV_ROWS or SWA_ROWS)), "the 5D cache has one layout"
    for p in range_constexpr(token_passes(S)):
        tk = tid // TOKEN_LANES + p * TOKENS_PER_PASS
        if tk < S:
            # positions are int64 and slots int64 or int32: this token's low word
            pos = fx.Int32(
                bo.buffer_load(rsrc(a["positions"]), 2 * tk, vec_width=1, dtype=T.i32)
            )
            slot = fx.Int32(
                bo.buffer_load(
                    rsrc(a["slots"]),
                    (slot64 != 0).select(2 * tk, tk),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            if (use_swa != 0) & (slot >= 0):
                slot = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["swa_map"]),
                        (swa64 != 0).select(2 * slot, slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            # this lane's two RoPE pairs: cos / sin of dims d0 / 2 and d0 / 2 + 1
            cs = fx.Vector(
                bo.buffer_load(r_cs, pos * HEAD_DIM + d0 // 2, vec_width=2, dtype=T.f32)
            )
            sn = fx.Vector(
                bo.buffer_load(
                    r_cs,
                    pos * HEAD_DIM + HEAD_DIM // 2 + d0 // 2,
                    vec_width=2,
                    dtype=T.f32,
                )
            )
            e0, e1, e2, e3 = half_sum(poll, part, tk, col0 + d0)
            e = [e0, e1, e2, e3]
            # the per-head norm: sum of squares over the head's 16 lanes
            ss = fx.Float32(0.0)
            for j in range_constexpr(ELEMS):
                ss = ss + e[j] * e[j]
            ss = butterfly(ss, (8, 4, 2, 1))
            rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
            n = [bf16_round(e[j] * rstd) for j in range(ELEMS)]
            # the query scale after the norm, rounded
            n = [is_q.select(bf16_round(n[j] * QK_SCALE), n[j]) for j in range(ELEMS)]
            # interleaved RoPE on (x, y) = dims (2i, 2i + 1), bf16 cos / sin, each
            # product rounded, then the sum rounded (the fused kernel's order)
            c0, c1 = bf16_round(cs[0]), bf16_round(cs[1])
            s0, s1 = bf16_round(sn[0]), bf16_round(sn[1])
            rot = [
                bf16_round(n[0] * c0) - bf16_round(n[1] * s0),
                bf16_round(n[1] * c0) + bf16_round(n[0] * s0),
                bf16_round(n[2] * c1) - bf16_round(n[3] * s1),
                bf16_round(n[3] * c1) + bf16_round(n[2] * s1),
            ]
            roped = normed & (use_rope != 0)
            out = [
                normed.select(roped.select(bf16_round(rot[j]), n[j]), e[j])
                for j in range(ELEMS)
            ]
            vec = fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16)
            if is_q:
                bo.buffer_store(vec, rsrc(a["q_out"]), tk * Q_K + h * HEAD_DIM + d0)
            else:
                if is_k:
                    bo.buffer_store(
                        vec, rsrc(a["k_out"]), tk * KV_K + kv_head * HEAD_DIM + d0
                    )
                else:
                    bo.buffer_store(
                        vec, rsrc(a["v_out"]), tk * KV_K + kv_head * HEAD_DIM + d0
                    )
                if const_expr(KV5):
                    if slot >= 0:
                        # the bf16 values as written to k_out / v_out, 16 B units: K [page][head][dim / 8][key][8]
                        # (this lane's 4 dims: 8 B), V [page][head][key / 8][dim][8] (a dim a 2 B store)
                        kp = slot & (KV5_PAGE - 1)
                        hb = (
                            fx.Int64(((slot >> 6) * LOCAL_KV_HEADS + kv_head))
                            * KV5_HEAD_BYTES
                        )
                        if is_k:
                            gstore_raw(
                                fx.Int64(a["k_cache"])
                                + hb
                                + fx.Int64(
                                    ((d0 // 8) * KV5_PAGE + kp) * 16 + (d0 % 8) * 2
                                ),
                                vec,
                                8,
                            )
                        else:
                            for j in range_constexpr(ELEMS):
                                gstore_raw(
                                    fx.Int64(a["v_cache"])
                                    + hb
                                    + fx.Int64(
                                        ((kp // 8) * HEAD_DIM + d0 + j) * 16
                                        + (kp % 8) * 2
                                    ),
                                    vec[j],
                                    2,
                                )
                elif slot >= 0:
                    packed = fp8_pack4(
                        clamp_fp8(out[0]),
                        clamp_fp8(out[1]),
                        clamp_fp8(out[2]),
                        clamp_fp8(out[3]),
                    )
                    # NHD page-16 fp8: slot s, head hh, dim d at byte (s KV_HEADS + hh) HEAD_DIM + d;
                    # a full layer's head-major pool (``kv_rows``): byte (hh rows + s) HEAD_DIM + d
                    idx = (slot * LOCAL_KV_HEADS + kv_head) * (HEAD_DIM // 4) + d0 // 4
                    # a sliding layer's head-major SWA pool (``swa_rows``): byte (hh swa_rows + s) HEAD_DIM + d
                    idx_swa = idx
                    if const_expr(SWA_ROWS > 0):
                        idx_swa = (kv_head * SWA_ROWS + slot) * (
                            HEAD_DIM // 4
                        ) + d0 // 4
                    idx_full = idx
                    if const_expr(KV_ROWS > 0):
                        idx_full = (kv_head * KV_ROWS + slot) * (
                            HEAD_DIM // 4
                        ) + d0 // 4
                    if const_expr(KV_ROWS > 0 or SWA_ROWS > 0):
                        idx = (use_swa != 0).select(idx_swa, idx_full)
                    if is_k:
                        bo.buffer_store(packed, rsrc(a["k_cache"]), idx)
                    else:
                        bo.buffer_store(packed, rsrc(a["v_cache"]), idx)


@flyc.jit
def stage_gate(c, g):
    """Gate task ``g``: rows ``G_OFF + 64 g`` .. of every token from the halves'
    partials, rounded once, out as bf16 (the o_gate of q head ``g``)."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    poll, part = c["poll"], c["part"]
    g_out = rsrc(c["args"]["g_out"])
    d0 = (lane % 16) * ELEMS
    for p in range_constexpr(token_passes(S)):
        tk = tid // TOKEN_LANES + p * TOKENS_PER_PASS
        if tk < S:
            e = half_sum(poll, part, tk, G_OFF + g * HEAD_DIM + d0)
            vec = fx.Vector.from_elements(list(e), fx.Float32).to(fx.BFloat16)
            bo.buffer_store(vec, g_out, tk * Q_K + g * HEAD_DIM + d0)


@flyc.jit
def run_k1(c, bid, x_flag=None):
    """K1's stages on this CTA: its GEMV task, then its head or gate task. ``x_flag`` (a mailbox region): x was written earlier in this
    launch (the previous layer's boundary), flagged a token each; the first task's
    weights go out before the wait. In that launch the boundary CTAs and the active
    unpool CTAs reach K1 last (their weights would land after everyone's x): they take
    no GEMV work, which spreads over the other CTAs (``spread_tiles``)."""
    S = c["S"]
    stamp(c, 0)
    first = None
    defer = c.get("defer", False)
    if const_expr(x_flag is not None and defer):
        # deferred norms (the GEMV polls its unpooled rows itself): the boundary CTAs'
        # (0 .. S) tasks run on the last S CTAs, which issue both tasks' weights on arrival
        moved = bid < S
        spare = bid >= BLOCKS - S
        own = moved.select(fx.Int32(GEMV_TASKS), bid)  # past the last task: zero loads
        t2 = spare.select(bid - (BLOCKS - S), fx.Int32(GEMV_TASKS))
        first = gemv_w_loads(c, own // K_PARTS, own % K_PARTS, c["args"]["w"])
        extra = gemv_w_loads(c, t2 // K_PARTS, t2 % K_PARTS, c["args"]["w"])
        rocdl.sched_barrier(0)
    elif const_expr(x_flag is not None):
        # the CTAs that reach K1 late (``c["late"]``: the boundary's and the active unpool
        # CTAs) take no GEMV work: each K part's row tiles spread over the part's other CTAs,
        # whose weights go out on arrival (``spread_tiles``)
        lo, cnt = spread_tiles(c, bid)
        rows = spread_rows(lo, cnt, spread_max_tiles(c["late"]))
        # the first tiles' copies go out before the wait (a CTA without tiles issues none)
        if cnt > 0:
            ring_issue(c, bid % K_PARTS, rows, range(min(RING, len(rows))))
            if const_expr(PF_K1):
                # the later tiles into GL2 meanwhile (a slot past ``cnt`` holds row 0: a valid line)
                ring_prefetch(
                    c, bid % K_PARTS, lo, cnt, range(min(RING, len(rows)), len(rows))
                )
        poll_flags(c, [(x_flag, 0, S, FLAG_STRIDE)])
    stamp(c, 1)  # x arrived (a merged launch) / entry
    if const_expr(x_flag is not None and defer):
        if bid >= S:
            stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
        if bid >= BLOCKS - S:
            t2 = bid - (BLOCKS - S)
            stage_gemv(c, t2 // K_PARTS, t2 % K_PARTS, extra)
    elif const_expr(x_flag is not None):
        if cnt > 0:
            stage_gemv_ring(c, bid % K_PARTS, rows, cnt, min(RING, len(rows)))
    else:
        stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
    if const_expr(SECOND_TASKS > 0):
        if (bid >= QT0) & (bid < QT0 + SECOND_TASKS):
            q = BLOCKS + bid - QT0  # the tasks past one a CTA, on idle CTAs
            stage_gemv(c, q // K_PARTS, q % K_PARTS)
    stamp(c, 2)  # the GEMV task(s) done
    if (bid >= HT0) & (bid < HT0 + HEAD_TASKS):
        stage_head(c, bid - HT0)
    if (bid >= GT0) & (bid < GT0 + GATE_TASKS):
        stage_gate(c, bid - GT0)
    stamp(c, 3)  # the head / gate task done (end)


def k1_lds(tokens: int):
    """K1's LDS: a reduction row set a row tile (a token tile at a time)."""

    @fx.struct
    class Lds:
        red: fx.Array[
            fx.Float32, max(WAVES * WAVE_BYTES // 4, G_TILES * RED_FLOATS), 16
        ]

    return Lds
