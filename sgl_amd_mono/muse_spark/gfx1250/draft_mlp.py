# SPDX-License-Identifier: Apache-2.0
"""D-MLP: one DFlash2 draft layer's MLP half as one launch, for S = ``tokens`` rows
(a block of 8 consecutive rows a request), on a rank of a TP group::

    res' = bf16(res + attn_out)                     (the fused add of the post norm)
    h    = RMSNorm(res') w_post                      (eps 1e-5, bf16)
    delta = h W_kp^T                                 (2048: side x tap x group)
    hc   = conv0(h): (base[0][tap] + delta[token, 0, tap, g]) . h[token - tap] in the block
    g, u = hc W_gu^T (3584 gate rows, 3584 up rows a rank), a = silu(g) u (bf16)
    d    = a W_down^T                                (this rank's partial, 8192)
    d    = sum over the ranks (bf16 pairs through the peer regions, rank order)
    out  = conv1(d) with side 1's coefficients        (bf16)

gfx1250 plan (every CTA, ``BLOCKS`` of them): the hidden splits into ``K_PARTS`` K
parts of ``K_PART`` channels, ``RGS = BLOCKS / K_PARTS`` CTAs a K part. S0 the norm
statistics (the K parts' sums of squares through the mailbox) and the coefficient
GEMV (the part's row tiles split in ``RGS`` contiguous ranges; fp32 partials a K
part); S1 the coefficients' reduce (a contiguous row range a CTA); S2 conv0 on this
K part and the gate / up GEMV (K part x row range, fp32 partials); S3 the reduce +
silu (a range of dim pairs a CTA, bf16 pairs); S4 the down GEMV over the whole K of
the CTA's column block (``col_block``: whole convolution groups) against every
token's activations staged in LDS; S5 the push to every rank, the ranks' sum, conv1
and the output.

The GEMVs (``gemv``) stream each wave's weight rows into its own LDS ring by async
copies (16 rows x 128 K a unit, a row's 256 B contiguous, chunk-swizzled), then
read the WMMA A operand from LDS; a lane's B operand (16 tokens a tile) is read from
the stage's LDS rows. Every fp32 partial is summed in a fixed order (wave order,
K-part order, rank order) and rounded where the gfx950 kernels round.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    peer_bases,
    rsrc,
    sum_partials,
    wmma_bf16,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Addr, Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx950 import compat
from sgl_amd_mono.muse_spark.gfx1250.config import LOG2E, MAX_TOKENS
from sgl_amd_mono.muse_spark.gfx1250.device import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    async_copy16,
    lds_addr,
    lds_load16,
    poll_flags,
    release_stores,
    stamp,
    step_tag,
    wait_lds_dma,
)

BRIDGE = compat.install()
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

# ---------------------------------------------------------------- the draft's shapes (a rank of TP 8)
H = 8192  # hidden
I = 3584  # intermediate a rank (28672 / 8)
GU_ROWS = 2 * I  # the merged gate (first) and up rows
GROUP = 16  # channels a convolution group
GROUPS = H // GROUP  # 512
TAPS = 2
SIDES = 2
COEF_ROWS = (
    SIDES * TAPS * GROUPS
)  # 2048: the kernel projection's rows, (side, tap, group)
BLOCK = 8  # rows a request: the convolution shifts within it
EPS = 1e-5
ROWS = 16  # a WMMA tile's rows
TILE = 16  # tokens a WMMA B operand holds
K_STEP = 32  # the bf16 WMMA's K
ACC = 8  # fp32 accumulator elements a lane of a WMMA tile
K_PARTS = 16  # K parts of the hidden (the norm / conv0 / K-part GEMVs)
K_PART = H // K_PARTS  # 512
RGS = BLOCKS // K_PARTS  # CTAs a K part
CL = K_PART // 8  # channel lanes of a K part: eight channels each
TPP = THREADS // CL  # tokens a pass of the channel lanes
XPAD = 8  # bf16 elements past a token's LDS row (bank spread, 16 B aligned rows)
XP = K_PART + XPAD  # a K part's token row pitch in LDS
MAXG = -(-GROUPS // BLOCKS)  # convolution groups a CTA's column block holds at most
MAXC = MAXG * GROUP  # columns of a column block at most
# the GEMV ring: a wave's unit is ROWS weight rows x KC of K (256 B a row), RING units in LDS
KC = 128
UNIT_BYTES = ROWS * KC * 2
UNIT_CHUNKS = KC * 2 // 16  # 16 B chunks a unit row: 16
COPIES = ROWS * UNIT_CHUNKS // WAVE  # async copies a unit: a 16 B chunk a lane each
RING = 2
RING_BYTES = WAVES * RING * UNIT_BYTES
RED_FLOATS = (
    4096  # a K-split GEMV round's partial tiles (WAVES tiles x 256 x token tiles <= 16)
)
assert K_PARTS * K_PART == H and BLOCKS % K_PARTS == 0 and GROUPS % K_PARTS == 0
assert THREADS % CL == 0 and UNIT_CHUNKS == 16 and COPIES * WAVE == ROWS * UNIT_CHUNKS
# a K part's columns are exactly the column blocks of its RGS CTAs (``col_block``)
assert (
    all((kp * (GROUPS // K_PARTS) * BLOCKS) % GROUPS == 0 for kp in range(K_PARTS))
    and (GROUPS // K_PARTS) * BLOCKS // GROUPS == RGS
)


def token_tiles(s: int) -> int:
    return -(-s // TILE)


def span(i, n: int, parts: int):
    """Range i of ``n`` items split in ``parts`` contiguous ranges: (first, count)."""
    lo = i * n // parts
    return lo, (i + 1) * n // parts - lo


def col_block(bid):
    """CTA ``bid``'s column block of the hidden (the down / o GEMV rows, the TP sum and
    conv1): whole convolution groups, (first group, groups)."""
    return span(bid, GROUPS, BLOCKS)


def gemv_wk(max_tiles: int, klen: int) -> int:
    """The waves a GEMV splits its K over: more when a CTA has few row tiles."""
    for wk in (4, 2, 1):
        if klen % (wk * KC) == 0 and WAVES // wk >= max_tiles:
            return wk
    return 1


def dm_scratch_layout() -> dict:
    """D-MLP's scratch regions (pairs), sized for MAX_TOKENS: the K parts' sums of
    squares, the coefficient partials and sums, the gate / up partials, the
    activations (bf16 pairs), the trace."""
    return pair_layout(
        [
            ("dm_ss", K_PARTS * MAX_TOKENS),
            ("dm_cpart", K_PARTS * MAX_TOKENS * COEF_ROWS),
            ("dm_coef", MAX_TOKENS * COEF_ROWS),
            ("dm_gpart", K_PARTS * MAX_TOKENS * GU_ROWS),
            ("dm_act", MAX_TOKENS * I // 2),
            ("dm_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def dm_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric regions: ``dm`` [source rank][token][H / 2] bf16 pairs,
    ``dmc`` the coefficients [token][COEF_ROWS] (each rank pushes its rows), ``dmo``
    out [token][H / 2] bf16 pairs (each column block's owner pushes it)."""
    return pair_layout(
        [
            ("dm", npes * MAX_TOKENS * H // 2),
            ("dmc", MAX_TOKENS * COEF_ROWS),
            ("dmo", MAX_TOKENS * H // 2),
        ],
        align=256,
    )


def owner_of(c: dict, bid):
    """The rank that sums CTA ``bid``'s column block of a TP sum (``BLOCKS / W`` blocks a
    rank), and whether it is this one."""
    owner = bid // (BLOCKS // c["W"])
    return owner, c["rank"] == owner


def rank_region(regions, rank):
    """``regions[rank]`` (one a rank) for a traced ``rank``."""
    v = regions[0].value
    for pe in range(1, len(regions)):
        v = (rank == pe).select(regions[pe].value, v)
    return Addr(regions[0].region, regions[0].space, v)


def coef_shard(c: dict, npes: int, rank, own=None, peers=None) -> dict:
    """``c`` with its coefficient GEMV's share: with ``peers`` (the ranks'
    ``own``-like regions) this rank computes rows ``COEF_ROWS / npes`` rank .. of
    the kernel projection and pushes them to every rank (the same per-row
    arithmetic as one rank computing all: the weights are replicated); ``own``
    is polled. Else the rows are all its own, in ``c["coef"]`` (scratch)."""
    rows = COEF_ROWS // npes if peers is not None else COEF_ROWS
    assert rows % ROWS == 0
    c.update(c_rows=rows, c_tiles=rows // ROWS)
    if peers is None:
        c.update(coef_row0=0, coef_puts=[c["coef"]])
    else:
        c.update(coef_row0=rank * rows, coef=own, coef_puts=peers)
    return c


def dm_sym_bytes(npes: int) -> int:
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes

    return scratch_bytes(dm_sym_layout(npes))


def dm_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        ring: fx.Array[fx.Int32, RING_BYTES // 4, 16]  # the GEMVs' weight rings
        red: fx.Array[fx.Float32, RED_FLOATS, 16]  # a K-split GEMV round's partials
        xa: fx.Array[fx.BFloat16, S * (I + XPAD), 16]  # h (then hc at hc_off), later a
        ssl: fx.Array[fx.Float32, S * K_PARTS, 16]  # the K parts' sums of squares
        ssw: fx.Array[fx.Float32, S * 2, 16]  # a token's two waves' sums of squares
        nf: fx.Array[fx.Float32, S, 16]  # 1 / rms a token
        dbuf: fx.Array[fx.Float32, S * MAXC, 16]  # the CTA's column block of d
        cf1: fx.Array[fx.Float32, S * TAPS * MAXG, 16]  # side 1 deltas of its groups

    return Lds


LDS_BYTES = 320 * 1024  # a gfx1250 workgroup's LDS


def dm_lds_bytes(tokens: int) -> int:
    S = tokens
    return RING_BYTES + RED_FLOATS * 4 + S * (I + XPAD) * 2 + (S * K_PARTS + 2 * S + S + S * MAXC + S * TAPS * MAXG) * 4 + 8 * 16  # fmt: skip


# ---------------------------------------------------------------- shared pieces
def frag_a(buf, lane, kc):
    """The WMMA A operand of step ``kc`` from a ring unit (byte address ``buf``): row
    l % 16, K 16 (l / 16) .. + 16 of the step (two chunks, swizzled by the row)."""
    rr = lane % 16
    h = lane // 16
    a = lds_load16(buf + rr * (KC * 2) + (((kc * 4 + 2 * h) ^ rr) * 16))
    b = lds_load16(buf + rr * (KC * 2) + (((kc * 4 + 2 * h + 1) ^ rr) * 16))
    return fx.Vector.from_elements(
        [a[i] for i in range(4)] + [b[i] for i in range(4)], fx.Int32
    ).bitcast(fx.BFloat16)


def lds_rows_b(c, base, pitch):
    """B operands from LDS bf16 token rows (``pitch`` elements a row): ``xb_of(tt, k)``
    is token 16 tt + l % 16 (clamped to S - 1), K k + 16 (l / 16) .. + 16."""
    S, lane = c["S"], c["lane"]
    V8B = fx.Vector.make_type(8, fx.BFloat16)

    def xb_of(tt, k):
        tok = fx.min(tt * TILE + lane % 16, S - 1)
        p = base + (tok * pitch + k + 16 * (lane // 16))
        lo = fx.Vector(fx.ptr_load(p, result_type=V8B))
        hi = fx.Vector(fx.ptr_load(p + 8, result_type=V8B))
        return lo.shuffle(hi, list(range(16)))

    return xb_of


@flyc.jit
def gemv(
    c, w, width: int, k0, klen: int, tile_lo, n_tiles, max_tiles: int, xb_of, publish
):
    """Rows ``16 tile_lo`` .. ``16 (tile_lo + n_tiles)`` of the row-major bf16 matrix
    ``w`` [*, width] over K ``k0`` .. ``k0 + klen`` against every token (``xb_of``:
    the B operands, K relative to ``k0``): ``publish(row, token, v)`` a fp32 value of
    every (row, token). The waves split the K in ``WK`` (``gemv_wk``) ranges and the
    row tiles in WAVES / WK strides; each wave streams its units (16 rows x KC) into
    its LDS ring, RING units ahead. With WK > 1 the K ranges' partials are summed in
    wave order through ``red``, a round of tiles at a time (two barriers a round)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    WK = gemv_wk(max_tiles, klen)
    WT = WAVES // WK
    KW = klen // WK
    NKC = KW // KC
    MAXW = -(-max_tiles // WT)
    TT = token_tiles(S)
    assert WK == 1 or WT * WK * TT * 256 <= RED_FLOATS
    wk, wt = wave % WK, wave // WK
    mine = lds_addr(c["ring"]) + wave * (RING * UNIT_BYTES)
    wg = fx.Int64(w)
    last = tile_lo + fx.max(n_tiles - 1, fx.Int32(0))
    units = [(i, j) for i in range(MAXW) for j in range(NKC)]
    NU = len(units)

    def issue(u):
        i, j = units[u]
        row0 = (
            fx.min(tile_lo + wt + WT * i, last) * ROWS
        )  # (a tile past the range: the last, unused)
        kk = k0 + wk * KW + j * KC
        buf = mine + (u % RING) * UNIT_BYTES
        for q in range_constexpr(COPIES):
            row = q * (WAVE // UNIT_CHUNKS) + lane // UNIT_CHUNKS
            ch = lane % UNIT_CHUNKS
            g = (
                wg
                + (fx.Int64(row0 + row) * width + fx.Int64(kk)) * 2
                + fx.Int64(ch * 16)
            )
            async_copy16(g, buf + row * (KC * 2) + ((ch ^ row) * 16))

    for u in range_constexpr(min(RING, NU)):
        issue(u)
    rocdl.sched_barrier(0)
    acc = []
    for u in range_constexpr(NU):
        i, j = units[u]
        issued = min(NU, u + RING)
        wait_lds_dma((issued - u - 1) * COPIES)
        buf = mine + (u % RING) * UNIT_BYTES
        if const_expr(j == 0):
            acc = [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(TT)]
        for kc in range_constexpr(KC // K_STEP):
            f = frag_a(buf, lane, kc)
            for tt in range_constexpr(TT):
                acc[tt] = wmma_bf16(
                    f, xb_of(tt, wk * KW + j * KC + kc * K_STEP), acc[tt]
                )
        if const_expr(u + RING < NU):
            rocdl.s_wait_dscnt(
                0
            )  # this unit's reads are done before its slot's next copy
            issue(u + RING)
            rocdl.sched_barrier(0)
        if const_expr(j == NKC - 1):
            t_i = wt + WT * i  # the wave's tile of round i
            if const_expr(WK == 1):
                if t_i < n_tiles:
                    for tt in range_constexpr(TT):
                        tok = tt * TILE + lane % 16
                        if tok < S:
                            for r in range_constexpr(ACC):
                                publish(
                                    (tile_lo + t_i) * ROWS + r + 8 * (lane // 16),
                                    tok,
                                    fx.Float32(acc[tt][r]),
                                )
            else:
                for tt in range_constexpr(TT):
                    fx.ptr_store(
                        acc[tt], red + (((wt * WK + wk) * TT + tt) * 256 + lane * ACC)
                    )
                gpu.barrier()
                n_out = WT * ROWS * S
                for p in range_constexpr(-(-n_out // THREADS)):
                    idx = tid + p * THREADS
                    tk = idx % S
                    rr = (idx // S) % ROWS
                    wtt = idx // (S * ROWS)
                    if (idx < n_out) & (wtt + WT * i < n_tiles):
                        ln = (rr // 8) * 16 + tk % TILE
                        v = fx.Float32(0.0)
                        for k in range_constexpr(WK):
                            v = v + fx.ptr_load(
                                red
                                + (
                                    ((wtt * WK + k) * TT + tk // TILE) * 256
                                    + ln * ACC
                                    + rr % 8
                                )
                            )
                        publish((tile_lo + wtt + WT * i) * ROWS + rr, tk, v)
                gpu.barrier()  # ``red`` is reused by the next round
    rocdl.s_wait_dscnt(0)
    gpu.barrier()  # the B rows and the rings are free for the next stage


@flyc.jit
def poll_items(c, n, nmax: int, specs_of, use):
    """Items 0 .. n (at most ``nmax``) polled a batch a thread on every thread:
    ``specs_of(idx)`` its mailbox specs, ``use(idx, got)`` its values."""
    tid = c["tid"]
    for q in range_constexpr(-(-nmax // THREADS)):
        idx = tid + q * THREADS
        if idx < n:
            use(idx, c["poll"](specs_of(idx)))


def chan_lanes(c):
    """This thread's channel lane of a K part (eight channels) and its tokens'
    first: a pass covers TPP tokens, thread (token wave pair, lane)."""
    cl = c["tid"] % CL
    return cl, c["tid"] // CL


# ---------------------------------------------------------------- S0: the norm, the coefficient GEMV
@flyc.jit
def stage_norm(c, kp, add_first=None):
    """res' = bf16(res + attn_out) on this K part (``add_first``: a traced flag, res'
    is x itself when set), written out by the part's first CTA, the K part's sums of
    squares published by it, every part's polled; h = RMSNorm(res') w -> ``xh``
    (LDS rows of XP). Arguments: ``c["norm"]`` = (res, x, w, res_out) addresses."""
    S, tid, bid = c["S"], c["tid"], c["bid"]
    put, ssl, ssw, nf, xh = c["put"], c["ssl"], c["ssw"], c["nf"], c["xh"]
    a_res, a_x, a_w, a_out = c["norm"]
    cl, t0 = chan_lanes(c)
    ch = kp * K_PART + 8 * cl
    xcm, ocm = c.get("in_cm", 0), c.get("xcm", 0)
    r_res, r_x, r_out = rsrc(a_res), rsrc(a_x), rsrc(a_out)
    wv = fx.Vector(
        bo.buffer_load(rsrc(a_w), ch // 2, vec_width=4, dtype=T.i32)
    ).bitcast(fx.BFloat16)
    NP = -(-S // TPP)
    rp = []
    for j in range_constexpr(NP):
        t = fx.min(t0 + TPP * j, S - 1)
        xv = fx.Vector(bo.buffer_load(r_x, (t * H + ch) // 2, vec_width=4, dtype=T.i32, cache_modifier=xcm)).bitcast(fx.BFloat16)  # fmt: skip
        rv = fx.Vector(bo.buffer_load(r_res, (t * H + ch) // 2, vec_width=4, dtype=T.i32, cache_modifier=xcm)).bitcast(fx.BFloat16)  # fmt: skip
        if const_expr(add_first is None):
            r8 = [bf16_round(fx.Float32(rv[e]) + fx.Float32(xv[e])) for e in range(8)]
        else:
            r8 = [add_first.select(fx.Float32(xv[e]), bf16_round(fx.Float32(rv[e]) + fx.Float32(xv[e]))) for e in range(8)]  # fmt: skip
        ss = r8[0] * r8[0]
        for e in range_constexpr(1, 8):
            ss = ss + r8[e] * r8[e]
        ss = butterfly(ss, (1, 2, 4, 8, 16))  # a wave's 32 lanes
        live = t0 + TPP * j < S
        if live:
            if bid < K_PARTS:
                bo.buffer_store(fx.Vector.from_elements(r8, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32), r_out, (t * H + ch) // 2, cache_modifier=ocm)  # fmt: skip
            if c["lane"] == 0:
                fx.ptr_store(ss, ssw + (t * 2 + cl // WAVE))
        rp.append(r8)
    gpu.barrier()
    if (bid < K_PARTS) & (tid < S):
        put(
            c["ss"],
            kp * MAX_TOKENS + tid,
            fx.ptr_load(ssw + 2 * tid) + fx.ptr_load(ssw + (2 * tid + 1)),
        )
    # every part's sum of squares of every token
    poll_items(
        c, S * K_PARTS, S * K_PARTS,
        lambda idx: [(c["ss"], (idx % K_PARTS) * MAX_TOKENS + idx // K_PARTS, 1)],
        lambda idx, got: fx.ptr_store(got[0][0].bitcast(fx.Float32), ssl + idx),
    )  # fmt: skip
    gpu.barrier()
    if tid < S:
        tot = fx.ptr_load(ssl + tid * K_PARTS)
        for p in range_constexpr(1, K_PARTS):
            tot = tot + fx.ptr_load(ssl + (tid * K_PARTS + p))
        fx.ptr_store(hw_rsq(tot / float(H) + EPS), nf + tid)
    gpu.barrier()
    for j in range_constexpr(NP):
        t = t0 + TPP * j
        if t < S:
            f = fx.ptr_load(nf + t)
            hv = fx.Vector.from_elements(
                [rp[j][e] * f * fx.Float32(wv[e]) for e in range(8)], fx.Float32
            ).to(fx.BFloat16)
            fx.ptr_store(hv, xh + (t * XP + 8 * cl))
    gpu.barrier()


def kpart_tiles(c, total_tiles: int, bid):
    """This CTA's share of a K-part GEMV of ``total_tiles`` row tiles: (K part, first
    tile, tiles, the most tiles a CTA takes)."""
    kp, j = bid % K_PARTS, bid // K_PARTS
    lo, n = span(j, total_tiles, RGS)
    return kp, lo, n, -(-total_tiles // RGS)


@flyc.jit
def stage_coef(c, bid):
    """The coefficient GEMV task: this CTA's row tiles of this rank's W_kp rows over its
    K part against h; fp32 partials to ``cpart`` [kp][token][row]."""
    put, cpart = c["put"], c["cpart"]
    kp, lo, n, mt = kpart_tiles(c, c["c_tiles"], bid)
    row0 = c["coef_row0"]

    def publish(r, tk, v):
        put(cpart, (kp * MAX_TOKENS + tk) * COEF_ROWS + row0 + r, v)

    gemv(c, fx.Int64(c["w_kp"]) + fx.Int64(row0 * H * 2), H, kp * K_PART, K_PART, lo, n, mt, lds_rows_b(c, c["xh"], XP), publish)  # fmt: skip


# ---------------------------------------------------------------- S1: the coefficients
@flyc.jit
def stage_coef_reduce(c, bid):
    """This CTA's range of this rank's coefficient rows for every token: the K parts'
    partials summed in part order -> every ``coef_puts`` region [token][row] (fp32)
    and, with a ``coef_out`` argument (D-pre: D-post reads it), the plain fp32
    [S, COEF_ROWS] tensor."""
    S = c["S"]
    put, cpart = c["put"], c["cpart"]
    rows = c["c_rows"]
    lo, n = span(bid, rows, BLOCKS)
    row0 = c["coef_row0"] + lo
    cout = c.get("coef_out")

    def specs_of(idx):
        j, t = idx % n, idx // n
        return [
            (cpart, (kp * MAX_TOKENS + t) * COEF_ROWS + row0 + j, 1)
            for kp in range(K_PARTS)
        ]

    def use(idx, got):
        j, t = idx % n, idx // n
        row = row0 + j
        v = got[0][0].bitcast(fx.Float32)
        for kp in range_constexpr(1, K_PARTS):
            v = v + got[kp][0].bitcast(fx.Float32)
        for dst in c["coef_puts"]:
            put(dst, t * COEF_ROWS + row, v)
        if const_expr(cout is not None):
            bo.buffer_store(v, rsrc(cout), t * COEF_ROWS + row, cache_modifier=CM_DEV)

    poll_items(c, n * S, -(-rows // BLOCKS) * S, specs_of, use)


# ---------------------------------------------------------------- S2: conv0, the gate / up GEMV
def conv0_base(c, kp):
    """Side 0's base coefficients of this thread's eight channels of K part ``kp``."""
    r_base = rsrc(c["base"])
    cl, _ = chan_lanes(c)
    ch = kp * K_PART + 8 * cl
    return [
        fx.Vector(bo.buffer_load(r_base, ((0 * TAPS + tap) * H + ch) // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
        for tap in range(TAPS)
    ]  # fmt: skip


@flyc.jit
def stage_conv0(c, kp, b0):
    """``xhc`` <- conv0(``xh``) on this K part: coefficient = base[0][tap][channel]
    (``b0``: ``conv0_base``) + delta[token][0][tap][group], tap 1 reads the block's
    previous row; each thread polls the two deltas it uses."""
    S, poll, xh, xhc = c["S"], c["poll"], c["xh"], c["xhc"]
    GP = K_PART // GROUP
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    cl, t0 = chan_lanes(c)
    g = cl // 2  # the thread's group within the part (eight channels: half a group)
    for j in range_constexpr(-(-S // TPP)):
        t = fx.min(t0 + TPP * j, S - 1)
        tp = fx.max(t - 1, fx.Int32(0))
        got = poll(
            [(c["coef"], t * COEF_ROWS + (0 * TAPS + tap) * GROUPS + kp * GP + g, 1) for tap in range(TAPS)]
        )  # fmt: skip
        d0, d1 = got[0][0].bitcast(fx.Float32), got[1][0].bitcast(fx.Float32)
        hv = fx.Vector(fx.ptr_load(xh + (t * XP + 8 * cl), result_type=V8B))
        pv = fx.Vector(fx.ptr_load(xh + (tp * XP + 8 * cl), result_type=V8B))
        has_prev = (t % BLOCK) >= 1
        vals = []
        for e in range_constexpr(8):
            c0 = fx.Float32(b0[0][e]) + d0
            c1 = fx.Float32(b0[1][e]) + d1
            vals.append(
                c0 * fx.Float32(hv[e])
                + has_prev.select(c1 * fx.Float32(pv[e]), fx.Float32(0.0))
            )
        if t0 + TPP * j < S:
            fx.ptr_store(
                fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16),
                xhc + (t * XP + 8 * cl),
            )
    gpu.barrier()


@flyc.jit
def stage_gu(c, bid):
    """The gate / up GEMV task: this CTA's row tiles of W_gu over its K part against
    hc; fp32 partials to ``gpart`` [kp][token][row]."""
    put, gpart = c["put"], c["gpart"]
    kp, lo, n, mt = kpart_tiles(c, GU_ROWS // ROWS, bid)

    def publish(r, tk, v):
        put(gpart, (kp * MAX_TOKENS + tk) * GU_ROWS + r, v)

    gemv(
        c,
        c["w_gu"],
        H,
        kp * K_PART,
        K_PART,
        lo,
        n,
        mt,
        lds_rows_b(c, c["xhc"], XP),
        publish,
    )


# ---------------------------------------------------------------- S3: the activations
def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


@flyc.jit
def stage_act(c, bid):
    """This CTA's range of intermediate dim pairs for every token: the K parts' gate
    and up partials summed in part order, rounded to bf16, a = silu(g) u -> ``act``
    [token][dim] bf16 pairs (two dims a thread)."""
    S, put_bf = c["S"], c["put_bf"]
    gpart, act = c["gpart"], c["act"]
    lo, n = span(bid, I // 2, BLOCKS)

    def specs_of(idx):
        j, t = idx % n, idx // n
        d0 = 2 * (lo + j)
        return [(gpart, (kp * MAX_TOKENS + t) * GU_ROWS + d0, 2) for kp in range(K_PARTS)] + [
            (gpart, (kp * MAX_TOKENS + t) * GU_ROWS + I + d0, 2) for kp in range(K_PARTS)
        ]  # fmt: skip

    def use(idx, gu):
        j, t = idx % n, idx // n
        d0 = 2 * (lo + j)
        g0, g1, u0, u1 = (
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
        )
        for kp in range_constexpr(K_PARTS):
            g0 = g0 + gu[kp][0].bitcast(fx.Float32)
            g1 = g1 + gu[kp][1].bitcast(fx.Float32)
            u0 = u0 + gu[K_PARTS + kp][0].bitcast(fx.Float32)
            u1 = u1 + gu[K_PARTS + kp][1].bitcast(fx.Float32)
        g0, g1, u0, u1 = bf16_round(g0), bf16_round(g1), bf16_round(u0), bf16_round(u1)
        a0 = bf16_round(g0 * sigmoid(g0)) * u0
        a1 = bf16_round(g1 * sigmoid(g1)) * u1
        put_bf(act, t * I + d0, [a0, a1])

    poll_items(c, n * S, -(-(I // 2) // BLOCKS) * S, specs_of, use)


# ---------------------------------------------------------------- S4: the down GEMV
@flyc.jit
def stage_down(c, bid):
    """The down GEMV task: this CTA's column block of W_down rows over the whole K
    (I) against every token's activations (polled bf16 pairs, staged in LDS rows);
    the rank's fp32 partial rows -> ``dbuf`` [token][MAXC]."""
    S, tid, poll, xa, dbuf = c["S"], c["tid"], c["poll"], c["xa"], c["dbuf"]
    act = c["act"]
    XI = I + XPAD
    Q = I // 8  # 16 B (four pairs) a thread item
    N = S * Q
    for q in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + q * THREADS, N - 1)
        t, k8 = idx // Q, idx % Q
        got = poll([(act, (t * I + 8 * k8) // 2 + 2 * h, 2) for h in range(2)])
        if tid + q * THREADS < N:
            v = fx.Vector.from_elements(
                [fx.Int32(w) for h in range(2) for w in got[h]], fx.Int32
            )
            fx.ptr_store(v.bitcast(fx.BFloat16), xa + (t * XI + 8 * k8))
    gpu.barrier()
    g_lo, n_g = col_block(bid)

    def publish(r, tk, v):
        fx.ptr_store(v, dbuf + (tk * MAXC + r - g_lo * GROUP))

    gemv(c, c["w_down"], I, 0, I, g_lo, n_g, MAXG, lds_rows_b(c, xa, XI), publish)
    gpu.barrier()


# ---------------------------------------------------------------- S5: the TP sum, conv1, out
@flyc.jit
def tp_sum_conv1(c, bid, out, coef_cf1, boxes):
    """The CTA's column block of a rank partial (``dbuf`` [token][MAXC] fp32) for every
    token: pushed as bf16 pairs to every rank's ``own`` region (``boxes``: (own,
    peers, rs_box, peer_box)), the ranks' summed in rank order (two columns a thread),
    conv1 with side 1's coefficients (``coef_cf1(idx)``: the delta of (token, tap,
    group) item ``idx`` of the block) over the block, -> ``out`` (bf16). With a
    ``rs_box`` (TP > 1) the sum is a reduce-scatter and an all-gather: the partials
    go to the block's owner rank only, which sums them, applies conv1 and pushes the
    bf16 pairs to every rank's box; every rank stores its block from its box."""
    S, W, tid, poll, put_bf = c["S"], c["W"], c["tid"], c["poll"], c["put_bf"]
    own, peers, rs_box, peer_box = boxes
    rs = rs_box is not None
    if const_expr(rs):
        owner, mine = owner_of(c, bid)
    dbuf, cf1 = c["dbuf"], c["cf1"]
    rank = c["rank"]
    g_lo, n_g = col_block(bid)
    xcm = c.get("xcm", 0)
    r_out = rsrc(out)
    # side 1's deltas of this block's groups: (token, tap, group) a thread
    N1 = S * TAPS * MAXG
    for p in range_constexpr(-(-N1 // THREADS)):
        idx = tid + p * THREADS
        g = idx % MAXG
        if (idx < N1) & (g < n_g):
            fx.ptr_store(coef_cf1(idx, g_lo), cf1 + idx)
    PAIRS = n_g * (GROUP // 2)
    N = PAIRS * S
    NMAX = MAXG * (GROUP // 2) * S
    for p in range_constexpr(-(-NMAX // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        col = g_lo * GROUP + 2 * j
        v0 = fx.ptr_load(dbuf + (t * MAXC + 2 * j))
        v1 = fx.ptr_load(dbuf + (t * MAXC + 2 * j + 1))
        live = tid + p * THREADS < N
        if live:
            if const_expr(rs):
                put_bf(
                    rank_region(peers, owner),
                    (rank * MAX_TOKENS + t) * H + col,
                    [v0, v1],
                )
            else:
                for pe in range_constexpr(W):
                    put_bf(peers[pe], (rank * MAX_TOKENS + t) * H + col, [v0, v1])

        def pair_at(src):
            return ((src * MAX_TOKENS + t) * H + col) // 2

        if const_expr(rs):
            if mine:
                d0, d1 = sum_partials(poll, own, pair_at, W)
                gpu.barrier()  # every thread's reads of ``dbuf`` are done before it is rewritten
                if live:
                    fx.ptr_store(d0, dbuf + (t * MAXC + 2 * j))
                    fx.ptr_store(d1, dbuf + (t * MAXC + 2 * j + 1))
        else:
            d0, d1 = sum_partials(poll, own, pair_at, W)
            gpu.barrier()  # every thread's reads of ``dbuf`` are done before it is rewritten
            if live:
                fx.ptr_store(d0, dbuf + (t * MAXC + 2 * j))
                fx.ptr_store(d1, dbuf + (t * MAXC + 2 * j + 1))
    gpu.barrier()
    if const_expr(rs):
        if mine:
            conv1_out(
                c,
                bid,
                lambda t, col, vals: [
                    put_bf(box, t * H + col, vals) for box in peer_box
                ],
            )
        # this rank's block of out from its owner
        for p in range_constexpr(-(-NMAX // THREADS)):
            idx = fx.min(tid + p * THREADS, N - 1)
            j, t = idx % PAIRS, idx // PAIRS
            col = g_lo * GROUP + 2 * j
            got = poll([(rs_box, (t * H + col) // 2, 1)])
            if tid + p * THREADS < N:
                bo.buffer_store(
                    got[0][0], r_out, (t * H + col) // 2, cache_modifier=xcm
                )
    else:
        conv1_out(c, bid, lambda t, col, vals: bo.buffer_store(fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_out, (t * H + col) // 2, cache_modifier=xcm))  # fmt: skip


@flyc.jit
def conv1_out(c, bid, emit):
    """conv1 with side 1's coefficients (``cf1``) over the block's summed rows
    (``dbuf``): ``emit(token, column, [two f32])`` a pair of columns a thread."""
    S, tid, dbuf, cf1 = c["S"], c["tid"], c["dbuf"], c["cf1"]
    g_lo, n_g = col_block(bid)
    PAIRS = n_g * (GROUP // 2)
    N = PAIRS * S
    NMAX = MAXG * (GROUP // 2) * S
    r_base = rsrc(c["base"])
    for p in range_constexpr(-(-NMAX // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        col = g_lo * GROUP + 2 * j
        g = (2 * j) // GROUP
        bw = [bf2_f32(fx.Int32(bo.buffer_load(r_base, ((1 * TAPS + tap) * H + col) // 2, vec_width=1, dtype=T.i32))) for tap in range(TAPS)]  # fmt: skip
        has_prev = (t % BLOCK) >= 1
        tp = fx.max(t - 1, fx.Int32(0))
        vals = []
        for e in range_constexpr(2):
            b0e, b1e = bw[0][e], bw[1][e]
            c0 = b0e + fx.ptr_load(cf1 + ((t * TAPS + 0) * MAXG + g))
            c1 = b1e + fx.ptr_load(cf1 + ((t * TAPS + 1) * MAXG + g))
            cur = fx.ptr_load(dbuf + (t * MAXC + 2 * j + e))
            prev = fx.ptr_load(dbuf + (tp * MAXC + 2 * j + e))
            vals.append(c0 * cur + has_prev.select(c1 * prev, fx.Float32(0.0)))
        if tid + p * THREADS < N:
            emit(t, col, vals)


def cf1_item(idx):
    """(token, tap, group in the block) of a ``tp_sum_conv1`` delta item."""
    return idx // (TAPS * MAXG), (idx // MAXG) % TAPS, idx % MAXG


# ---------------------------------------------------------------- the launch
@flyc.jit
def run_dmlp(c):
    """The stages in order; ``c["stages"]`` (a build's, for bisecting) stops after
    that many."""
    bid, n = c["bid"], c["stages"]
    kp = bid % K_PARTS
    stamp(c, 0)
    aflag = c.get("aflag")
    if const_expr(aflag is not None):
        # res and attn_out from this launch (D-pre's and D-post's CTAs of this K part)
        poll_flags(c, [(aflag, kp * RGS, RGS)])
        c["in_cm"] = CM_DEV
    stage_norm(c, kp)
    stamp(c, 1)
    if const_expr(n > 1):
        stage_coef(c, bid)
    b0 = conv0_base(c, kp)
    stamp(c, 2)
    if const_expr(n > 2):
        stage_coef_reduce(c, bid)
    stamp(c, 3)
    if const_expr(n > 3):
        stage_conv0(c, kp, b0)
    stamp(c, 4)
    if const_expr(n > 4):
        stage_gu(c, bid)
    stamp(c, 5)
    if const_expr(n > 5):
        stage_act(c, bid)
    stamp(c, 6)
    if const_expr(n > 6):
        stage_down(c, bid)
    stamp(c, 7)
    if const_expr(n > 7):

        def coef_cf1(idx, g_lo):
            t, tap, g = cf1_item(idx)
            got = c["poll"](
                [(c["coef"], t * COEF_ROWS + (1 * TAPS + tap) * GROUPS + g_lo + g, 1)]
            )
            return got[0][0].bitcast(fx.Float32)

        tp_sum_conv1(
            c,
            bid,
            c["out"],
            coef_cf1,
            (c["own_dm"], c["peer_dm"], c.get("out_box"), c.get("peer_out")),
        )
    xflag, put_next = c.get("xflag"), c.get("put_next")
    if const_expr(xflag is not None):
        # this CTA's out columns are out: its flag under the next layer's tag (its D-pre)
        release_stores()
        gpu.barrier()
        if c["tid"] == 0:
            put_next(xflag, bid, fx.Int32(1))
    stamp(c, 8)


class DMLPBuild(msgspec.Struct, frozen=True):
    tokens: int
    npes: int = 8
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False
    stages: int = 8  # the stages run (fewer: a bisection build)


def dmlp_context(key: DMLPBuild, lds, args: dict, scratch, sym, peers, rank, tag, layout: dict, sym_layout: dict, layer=None) -> dict:  # fmt: skip
    s, npes = key.tokens, key.npes
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    bases = peer_bases(peers, npes)
    dm_off = sym_layout["dm"][0]
    c = {
        "S": s, "W": npes, "tid": tid, "bid": fx.block_idx.x, "lane": tid % WAVE, "wave": tid // WAVE,
        "ring": lds.ring.ptr, "red": lds.red.ptr, "xh": lds.xa.ptr, "xhc": lds.xa.ptr + s * XP,
        "xa": lds.xa.ptr, "ssl": lds.ssl.ptr, "ssw": lds.ssw.ptr, "nf": lds.nf.ptr, "dbuf": lds.dbuf.ptr,
        "cf1": lds.cf1.ptr, "put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll, "rank": rank,
        "trace_on": key.trace, "layer": layer, "stages": key.stages,
        "own_dm": preg(sym, dm_off, "dm"), "peer_dm": [preg(base, dm_off, "dm") for base in bases],
        "norm": (args["res"], args["attn_out"], args["w_post"], args["res_out"]),
        "w_kp": args["w_kp"], "base": args["base"], "w_gu": args["w_gu"], "w_down": args["w_down"],
        "out": args["out"],
    }  # fmt: skip
    for region, (off, _) in layout.items():
        c[region[3:]] = sreg(scratch, off, region)  # "dm_cpart" -> c["cpart"]
    dmc, dmo = sym_layout["dmc"][0], sym_layout["dmo"][0]
    if npes > 1:
        c.update(
            out_box=preg(sym, dmo, "dmo"),
            peer_out=[preg(base, dmo, "dmo") for base in bases],
        )
        return coef_shard(
            c,
            npes,
            rank,
            preg(sym, dmc, "dmc"),
            [preg(base, dmc, "dmc") for base in bases],
        )
    return coef_shard(c, npes, rank)


def build_dmlp(key: DMLPBuild):
    """``@flyc.jit`` launcher of D-MLP for ``key.tokens`` rows on a rank of a
    ``key.npes``-way TP group. Arguments: ``res`` / ``attn_out`` bf16 [S, H],
    ``w_post`` bf16 [H], ``w_kp`` bf16 [2048, H], ``base`` bf16 [2, 2, H], ``w_gu``
    bf16 [7168, H], ``w_down`` bf16 [H, 3584], ``res_out`` / ``out`` bf16 [S, H],
    ``scratch`` (``dm_scratch_layout`` bytes), ``sym`` (this rank's peer buffer),
    ``peers`` (the ranks' peer bases, int64 [npes]), ``rank``, ``epoch``, ``layer``."""
    S, npes = key.tokens, key.npes
    assert S % BLOCK == 0 and 1 <= S <= MAX_TOKENS
    assert dm_lds_bytes(S) <= LDS_BYTES, (
        f"D-MLP at {S} rows: {dm_lds_bytes(S)} B of LDS, over {LDS_BYTES}: draft.DRAFT_WIDTHS"
    )
    name = kernel_symbol(
        "muse_spark_dmlp", s=S, w=npes, f=key.fault_site, t=int(key.trace), n=key.stages
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout, sym_layout = dm_scratch_layout(), dm_sym_layout(npes)
    Lds = dm_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dmlp(
        res: Int64, attn_out: Int64, w_post: Int64, w_kp: Int64, base: Int64, w_gu: Int64, w_down: Int64,
        res_out: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "res": res, "attn_out": attn_out, "w_post": w_post, "w_kp": w_kp, "base": base,
                "w_gu": w_gu, "w_down": w_down, "res_out": res_out, "out": out,
            }  # fmt: skip
            c = dmlp_context(key, lds, args, scratch, sym, peers, rank, step_tag(epoch, layer), layout, sym_layout, layer=layer)  # fmt: skip
            run_dmlp(c)

    @flyc.jit
    def launch(
        res: Int64, attn_out: Int64, w_post: Int64, w_kp: Int64, base: Int64, w_gu: Int64, w_down: Int64,
        res_out: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dmlp(
            res, attn_out, w_post, w_kp, base, w_gu, w_down, res_out, out, scratch, sym, peers, rank, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
