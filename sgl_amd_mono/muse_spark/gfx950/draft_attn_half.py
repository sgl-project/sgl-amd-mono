# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 draft layer's attention half around D-attn, as two launches on a
rank of a TP group, for S = ``tokens`` rows (a block of 8 consecutive rows a
request)::

  D-pre : res' = first ? x : bf16(res + x)         (the input norm's fused add)
          h    = RMSNorm(res') w_in                 (eps 1e-5, bf16)
          delta = h W_kp^T                           (2048: the attention conv's coefficients, both sides -> ``coef``)
          hc   = conv0(h)                            (side 0)
          qkv  = hc W_qkv^T                          (1280 rows a rank: 8 q heads, k, v of 128)
          q, k : per-head RMSNorm (weights, eps 1e-5) -> bf16 -> neox RoPE from the table
          q -> ``q`` bf16 [S, 1024]; k, v -> the layer's fp8 cache at the rows' slots
  D-post: d   = sum over the ranks of o W_o^T         (D-attn's partials, 8192: zero-sentinel
                                                     bf16 pairs through the peer regions, rank order)
          attn_out = conv1(d) with side 1's coefficients (bf16 [S, 8192])

The stages, mailboxes and GEMV machinery are D-MLP's (``draft_mlp.py``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    peer_bases,
    rsrc,
    sum_partials,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx950 import compat
from sgl_amd_mono.muse_spark.gfx950.config import MAX_TOKENS
from sgl_amd_mono.muse_spark.gfx950.device import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    stamp,
    step_tag,
)
from sgl_amd_mono.muse_spark.gfx950.draft_mlp import (
    BLOCK,
    C_ROWS,
    C_TILES,
    COEF_ROWS,
    EPS,
    GROUP,
    GROUPS,
    K_PART,
    K_PARTS,
    K_PER_WAVE,
    K_STEP,
    ROWS,
    SS_COPIES,
    SS_STRIDE,
    TAPS,
    H,
    gemv_tiles,
    lds_b,
    stage_coef,
    w_loads,
    zs_enc,
    zs_poll,
)
from sgl_amd_mono.muse_spark.gfx950.layout import (
    RED_FLOATS,
    TILE,
    scratch_bytes,
)

BRIDGE = compat.install()
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

# the attention geometry (a rank of TP 8)
HEAD_DIM = 128
Q_HEADS = 8
Q_K = Q_HEADS * HEAD_DIM  # 1024
QKV_ROWS = Q_K + 2 * HEAD_DIM  # 1280: q heads, k, v
QKV_HEADS = Q_HEADS + 2  # head-rows a token of the qkv reduce
A_ROWS, A_TILES = 80, 5  # a qkv GEMV task's rows: 16 row groups x 16 K parts
O_ROWS, O_TILES = 32, 2  # an o GEMV task's rows: 256 tasks x full K (1024)
O_KPARTS = Q_K // K_PART  # 2 K parts of 512 a lane's wave split: the task sums them
assert (
    A_TILES * ROWS == A_ROWS
    and 16 * A_ROWS == QKV_ROWS
    and O_ROWS * BLOCKS == H
    and O_TILES * ROWS == O_ROWS
)
assert HEAD_DIM == 2 * 64


def dp_scratch_layout() -> dict:
    """D-pre's scratch regions (pairs): the K parts' sums of squares, the coefficient
    partials, the qkv partials, the trace."""
    return pair_layout(
        [
            ("dp_ss", SS_COPIES * K_PARTS * MAX_TOKENS),
            ("dp_cpart", K_PARTS * MAX_TOKENS * COEF_ROWS),
            ("dp_coef", MAX_TOKENS * COEF_ROWS),
            ("dp_apart", K_PARTS * MAX_TOKENS * QKV_ROWS),
            ("dp_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def do_scratch_layout() -> dict:
    """D-post's scratch regions (pairs): the trace (the o GEMV sums its K parts itself)."""
    return pair_layout([("do_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS)], align=256)


def do_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric region: ``do`` [source rank][token][H / 2] bf16 pairs."""
    return pair_layout([("do", npes * MAX_TOKENS * H // 2)], align=256)


def do_sym_bytes(npes: int) -> int:
    return scratch_bytes(do_sym_layout(npes))


def dp_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, C_TILES * RED_FLOATS, 16]
        hbuf: fx.Array[fx.BFloat16, S * K_PART, 16]
        ssl: fx.Array[fx.Float32, S * K_PARTS, 16]
        nf: fx.Array[fx.Float32, S, 16]
        cf: fx.Array[fx.Float32, S * TAPS * (K_PART // GROUP), 16]
        hrow: fx.Array[fx.Float32, HEAD_DIM, 16]  # a head-row of the qkv reduce
        hss: fx.Array[fx.Float32, WAVES, 16]

    return Lds


def do_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, O_TILES * RED_FLOATS, 16]
        dbuf: fx.Array[fx.Float32, S * O_ROWS, 16]
        cf1: fx.Array[fx.Float32, S * TAPS * (O_ROWS // GROUP), 16]

    return Lds


def _pick(pair, idx):
    """Element ``idx % 2`` of a (lo, hi) bf16 pair for a dynamic ``idx``."""
    return (idx % 2 == 0).select(pair[0], pair[1])


def _cs(r_cs, idx):
    """Element ``idx`` of the rotary table (fp32 [positions, 128]: cos then sin)."""
    return fx.Float32(bo.buffer_load(r_cs, idx, vec_width=1, dtype=T.f32))


# ---------------------------------------------------------------- D-pre stages
def norm_in_loads(c, kp):
    """stage_norm_in's loads (w_in, x / res of the wave's tokens), issued ahead of the
    coefficient task's weights: loads return in issue order, so the K part's sums of
    squares go out before W_kp has streamed in."""
    S, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    ch = kp * K_PART + 8 * lane
    r_res, r_x = rsrc(a["res"]), rsrc(a["x"])
    wv = fx.Vector(
        bo.buffer_load(rsrc(a["w_in"]), ch // 2, vec_width=4, dtype=T.i32)
    ).bitcast(fx.BFloat16)
    xr = []
    for j in range(S // WAVES):
        t = wave + WAVES * j
        xv = fx.Vector(
            bo.buffer_load(r_x, (t * H + ch) // 2, vec_width=4, dtype=T.i32)
        ).bitcast(fx.BFloat16)
        rv = fx.Vector(
            bo.buffer_load(r_res, (t * H + ch) // 2, vec_width=4, dtype=T.i32)
        ).bitcast(fx.BFloat16)
        xr.append((xv, rv))
    return wv, xr


@flyc.jit
def stage_norm_in(c, rg, kp, loaded):
    """res' = first ? x : bf16(res + x) on this K part (row group 0 writes it out), the
    K part's sums of squares published (row group 0), every part's polled; h =
    RMSNorm(res') w_in -> ``hbuf``."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    put, poll, hbuf, ssl, nf = c["put"], c["poll"], c["hbuf"], c["ssl"], c["nf"]
    a = c["args"]
    first = a["first"] != 0
    ch = kp * K_PART + 8 * lane
    r_out = rsrc(a["res_out"])
    wv, xr = loaded
    TPW = S // WAVES
    rp = []
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        xv, rv = xr[j]
        r8 = [
            first.select(
                fx.Float32(xv[e]), bf16_round(fx.Float32(rv[e]) + fx.Float32(xv[e]))
            )
            for e in range(8)
        ]
        ss = r8[0] * r8[0]
        for e in range_constexpr(1, 8):
            ss = ss + r8[e] * r8[e]
        ss = butterfly(ss, (1, 2, 4, 8, 16, 32))
        if rg == 0:
            bo.buffer_store(fx.Vector.from_elements(r8, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32), r_out, (t * H + ch) // 2)  # fmt: skip
            if lane < SS_COPIES:
                put(c["ss"], lane * SS_STRIDE + kp * MAX_TOKENS + t, ss)
        rp.append(r8)
    N = S * K_PARTS
    cp = (c["bid"] % SS_COPIES) * SS_STRIDE  # this CTA's copy
    for p in range_constexpr(-(-N // THREADS)):
        idx = tid + p * THREADS
        if idx < N:
            # thread idx polls (part idx / S, token idx % S): a wave's polls cover whole lines
            kq, tq = idx // S, idx % S
            got = poll([(c["ss"], cp + kq * MAX_TOKENS + tq, 1)])
            fx.ptr_store(got[0][0].bitcast(fx.Float32), ssl + (tq * K_PARTS + kq))
    gpu.barrier()
    if tid < S:
        tot = fx.ptr_load(ssl + tid * K_PARTS)
        for p in range_constexpr(1, K_PARTS):
            tot = tot + fx.ptr_load(ssl + (tid * K_PARTS + p))
        fx.ptr_store(hw_rsq(tot / float(H) + EPS), nf + tid)
    gpu.barrier()
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        f = fx.ptr_load(nf + t)
        hv = fx.Vector.from_elements(
            [rp[j][e] * f * fx.Float32(wv[e]) for e in range(8)], fx.Float32
        ).to(fx.BFloat16)
        fx.ptr_store(hv, hbuf + (t * K_PART + 8 * lane))
    gpu.barrier()


@flyc.jit
def stage_qkv(c, rg, kp, w):
    """The qkv GEMV task (``rg``, ``kp``): rows 80 rg .. of W_qkv over K part kp
    against hc; fp32 partials to ``dp_apart`` [kp][token][row]. ``w``: the task's
    W_qkv tiles, issued by ``run_dpre`` before the coefficient GEMV."""
    S, lane, wave, put = c["S"], c["lane"], c["wave"], c["put"]
    apart = c["apart"]

    def tok_of(t):
        return fx.min(t * TILE + lane % 16, S - 1)

    def k_of(kc):
        return wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)

    def publish(r, tk, v):
        put(apart, (kp * MAX_TOKENS + tk) * QKV_ROWS + rg * A_ROWS + r, v)

    gemv_tiles(c, w, lds_b(c, c["hbuf"], tok_of, k_of), A_TILES, publish)


@flyc.jit
def stage_heads(c, bid):
    """Head-rows (token, head) 10 S in all, one a CTA a pass: the K parts' qkv partials
    summed in part order (a dim a thread, 128 of them), the head's RMSNorm with its
    weight, bf16, neox RoPE from the table at the row's position; q heads -> ``q``
    (bf16), k -> the fp8 cache at the row's slot, v -> the cache unrotated."""
    S, tid, lane, wave, poll = c["S"], c["tid"], c["lane"], c["wave"], c["poll"]
    apart, hrow, hss = c["apart"], c["hrow"], c["hss"]
    a = c["args"]
    r_cs = rsrc(a["cos_sin"])
    r_q, r_k, r_v = rsrc(a["q"]), rsrc(a["k_cache"]), rsrc(a["v_cache"])
    N = QKV_HEADS * S
    for p in range_constexpr(-(-N // BLOCKS)):
        job = bid + p * BLOCKS
        live = job < N
        jb = fx.min(job, N - 1)
        # a CTA without a head-row this pass skips it (CTA-uniform): no idle CTA spins
        # on the last head-row's partials
        if live:
            t, hd = (
                jb // QKV_HEADS,
                jb % QKV_HEADS,
            )  # token, head-row (0..7 q, 8 k, 9 v)
            row0 = hd * HEAD_DIM  # the head's first qkv row
            # the dim a thread (threads 128 .. idle): the K parts' partials summed in order
            d = fx.min(tid, HEAD_DIM - 1)
            is_v = hd == QKV_HEADS - 1
            is_k = hd == QKV_HEADS - 2
            # the row's parameters (position, slot, norm weight, rotary table) load ahead
            # of the partials' polls: their round trips overlap the wait
            pos = fx.Int32(
                bo.buffer_load(
                    rsrc(a["positions"]),
                    (a["pos64"] != 0).select(2 * t, t),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            slot = fx.Int32(
                bo.buffer_load(
                    rsrc(a["slots"]),
                    (a["slot64"] != 0).select(2 * t, t),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            wn = fx.Float32(_pick(bf2_f32(fx.Int32(bo.buffer_load(rsrc(is_k.select(a["k_norm"], a["q_norm"])), d // 2, vec_width=1, dtype=T.i32))), d % 2))  # fmt: skip
            half = d % 64
            cse4 = [_cs(r_cs, pos * HEAD_DIM + (d + e) % 64) for e in range(4)]
            sne4 = [_cs(r_cs, pos * HEAD_DIM + 64 + (d + e) % 64) for e in range(4)]
            # only the dims' threads (waves 0, 1) poll: the other waves would spin on dim 127's words
            v = fx.Float32(0.0)
            if tid < HEAD_DIM:
                got = poll(
                    [
                        (apart, (kp * MAX_TOKENS + t) * QKV_ROWS + row0 + d, 1)
                        for kp in range(K_PARTS)
                    ],
                    batch=K_PARTS,
                )
                v = got[0][0].bitcast(fx.Float32)
                for kp in range_constexpr(1, K_PARTS):
                    v = v + got[kp][0].bitcast(fx.Float32)
            v = bf16_round(v)  # the GEMM's bf16 output
            # the head's sum of squares (q and k): wave partials through LDS
            sq = (tid < HEAD_DIM).select(v * v, fx.Float32(0.0))
            sq = butterfly(sq, (1, 2, 4, 8, 16, 32))
            if lane == 0:
                fx.ptr_store(sq, hss + wave)
            gpu.barrier()
            tot = fx.ptr_load(hss)
            for w in range_constexpr(1, WAVES):
                tot = tot + fx.ptr_load(hss + w)
            inv = hw_rsq(tot / float(HEAD_DIM) + EPS)
            xn = bf16_round(v * inv * wn)  # the normed value, bf16 (q and k); v keeps v
            val = is_v.select(v, xn)
            if tid < HEAD_DIM:
                fx.ptr_store(val, hrow + d)
            gpu.barrier()
            # neox RoPE: dims (d, d + 64) rotate by the position's angle d % 64
            cs, sn = cse4[0], sne4[0]
            x1 = fx.ptr_load(hrow + half)
            x2 = fx.ptr_load(hrow + (half + 64))
            rot = (d < 64).select(x1 * cs - x2 * sn, x2 * cs + x1 * sn)
            out = is_v.select(val, bf16_round(rot))
            if live & (tid < HEAD_DIM):
                if hd < Q_HEADS:
                    # q: bf16 pairs (even dims store the pair)
                    if (d % 2) == 0:
                        nxt = (d < 64).select(
                            bf16_round(
                                fx.ptr_load(hrow + (half + 1)) * cse4[1]
                                - fx.ptr_load(hrow + (half + 65)) * sne4[1]
                            ),  # fmt: skip
                            bf16_round(
                                fx.ptr_load(hrow + (half + 65)) * cse4[1]
                                + fx.ptr_load(hrow + (half + 1)) * sne4[1]
                            ),  # fmt: skip
                        )
                        bo.buffer_store(fx.Vector.from_elements([out, nxt], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_q, (t * Q_K + row0 + d) // 2)  # fmt: skip
                else:
                    # k / v: fp8, four dims a dword, at the row's cache slot
                    if (d % 4) == 0:
                        vals = [out]
                        for e in range_constexpr(1, 4):
                            de = d + e
                            he = de % 64
                            xe1 = fx.ptr_load(hrow + he)
                            xe2 = fx.ptr_load(hrow + (he + 64))
                            cse = cse4[e]
                            sne = sne4[e]
                            rote = (de < 64).select(
                                xe1 * cse - xe2 * sne, xe2 * cse + xe1 * sne
                            )
                            vals.append(
                                is_v.select(fx.ptr_load(hrow + de), bf16_round(rote))
                            )
                        packed = fp8_pack4(
                            clamp_fp8(vals[0]),
                            clamp_fp8(vals[1]),
                            clamp_fp8(vals[2]),
                            clamp_fp8(vals[3]),
                        )
                        bo.buffer_store(
                            packed,
                            rsrc(is_v.select(a["v_cache"], a["k_cache"])),
                            (slot * HEAD_DIM + d) // 4,
                        )
            gpu.barrier()  # ``hrow`` / ``hss`` are reused by the next head-row


@flyc.jit
def stage_conv0_fold(c, kp):
    """``hbuf`` <- conv0(h) on this K part (``draft_mlp.stage_conv0``) with the
    coefficient reduce folded in: each (token, tap, group) delta of this part's side 0
    is the K parts' partials (``dp_cpart``) polled and summed in part order here, no
    reduce CTA and hop between. The base weights load ahead of the polls. The row
    group 0 CTAs also store the side 0 sums to ``coef_out`` (fp32 [S, 2048]; side 1's
    rows come from ``stage_coef_side1``), so the launch's ``coef`` output stays whole."""
    S, tid, lane, wave, poll, hbuf, cf = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["poll"],
        c["hbuf"],
        c["cf"],
    )
    cpart = c["cpart"]
    a = c["args"]
    GP = K_PART // GROUP  # groups a K part: 32
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    ch = kp * K_PART + 8 * lane
    r_base = rsrc(a["base"])
    b0 = [
        fx.Vector(
            bo.buffer_load(
                r_base, ((0 * TAPS + tap) * H + ch) // 2, vec_width=4, dtype=T.i32
            )
        ).bitcast(fx.BFloat16)
        for tap in range(TAPS)
    ]
    N = S * TAPS * GP
    R = -(-N // THREADS)
    specs, outs = [], []
    for p in range_constexpr(R):
        idx = fx.min(tid + p * THREADS, N - 1)
        t, tap, g = idx // (TAPS * GP), (idx // GP) % TAPS, idx % GP
        row = (0 * TAPS + tap) * GROUPS + kp * GP + g
        outs.append(t * COEF_ROWS + row)
        specs += [
            (cpart, (k * MAX_TOKENS + t) * COEF_ROWS + row, 1) for k in range(K_PARTS)
        ]
    got = poll(specs, batch=R * K_PARTS)
    for p in range_constexpr(R):
        v = got[p * K_PARTS][0].bitcast(fx.Float32)
        for k in range_constexpr(1, K_PARTS):
            v = v + got[p * K_PARTS + k][0].bitcast(fx.Float32)
        if tid + p * THREADS < N:
            fx.ptr_store(v, cf + fx.min(tid + p * THREADS, N - 1))
            if c["bid"] < K_PARTS:  # (row group 0: one CTA a K part)
                bo.buffer_store(v, rsrc(a["coef_out"]), outs[p], cache_modifier=CM_DEV)
    gpu.barrier()  # the deltas are in
    TPW = S // WAVES
    outs = []
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        tp = fx.max(t - 1, fx.Int32(0))
        hv = fx.Vector(fx.ptr_load(hbuf + (t * K_PART + 8 * lane), result_type=V8B))
        pv = fx.Vector(fx.ptr_load(hbuf + (tp * K_PART + 8 * lane), result_type=V8B))
        has_prev = (t % BLOCK) >= 1
        g = lane // 2  # the lane's group within the part (8 channels: half a group)
        d0 = fx.ptr_load(cf + ((t * TAPS + 0) * GP + g))
        d1 = fx.ptr_load(cf + ((t * TAPS + 1) * GP + g))
        vals = []
        for e in range_constexpr(8):
            c0 = fx.Float32(b0[0][e]) + d0
            c1 = fx.Float32(b0[1][e]) + d1
            v = c0 * fx.Float32(hv[e]) + has_prev.select(
                c1 * fx.Float32(pv[e]), fx.Float32(0.0)
            )
            vals.append(v)
        outs.append(fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16))
    gpu.barrier()  # every read of h is done
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        fx.ptr_store(outs[j], hbuf + (t * K_PART + 8 * lane))
    gpu.barrier()


@flyc.jit
def stage_coef_side1(c, bid):
    """Side 1's coefficients (D-post's conv1) for every token: the K parts' partials
    summed in part order -> ``coef_out`` (fp32 [S, 2048]). Off the critical path:
    when the head-rows leave CTAs idle (10 S < BLOCKS), only those CTAs do it (a
    contiguous run of the S x 1024 (token, row) items a CTA), so the head-row CTAs
    go straight from their qkv partials to the head-row polls; otherwise rows
    1024 + 4 bid .. on every CTA, after the qkv task's partials are out."""
    S, tid, poll, cpart = c["S"], c["tid"], c["poll"], c["cpart"]
    a = c["args"]
    HALF_ROWS = COEF_ROWS // 2
    NH = QKV_HEADS * S
    if const_expr(NH < BLOCKS):
        NB = BLOCKS - NH  # the idle CTAs
        NI = S * HALF_ROWS
        PER = -(-NI // NB)
        if bid >= NH:
            i0 = (bid - NH) * PER
            for p in range_constexpr(-(-PER // THREADS)):
                k = tid + p * THREADS
                idx = fx.min(i0 + k, NI - 1)
                t, row = idx // HALF_ROWS, HALF_ROWS + idx % HALF_ROWS
                if (k < PER) & (i0 + k < NI):
                    got = poll(
                        [
                            (cpart, (kq * MAX_TOKENS + t) * COEF_ROWS + row, 1)
                            for kq in range(K_PARTS)
                        ],
                        batch=K_PARTS,
                    )
                    v = got[0][0].bitcast(fx.Float32)
                    for kq in range_constexpr(1, K_PARTS):
                        v = v + got[kq][0].bitcast(fx.Float32)
                    bo.buffer_store(
                        v,
                        rsrc(a["coef_out"]),
                        t * COEF_ROWS + row,
                        cache_modifier=CM_DEV,
                    )
    else:
        RPC = HALF_ROWS // BLOCKS  # 4 rows a CTA
        N = RPC * S
        for p in range_constexpr(-(-N // THREADS)):
            idx = fx.min(tid + p * THREADS, N - 1)
            j, t = idx % RPC, idx // RPC
            row = HALF_ROWS + bid * RPC + j
            if tid + p * THREADS < N:
                got = poll(
                    [
                        (cpart, (k * MAX_TOKENS + t) * COEF_ROWS + row, 1)
                        for k in range(K_PARTS)
                    ],
                    batch=K_PARTS,
                )
                v = got[0][0].bitcast(fx.Float32)
                for k in range_constexpr(1, K_PARTS):
                    v = v + got[k][0].bitcast(fx.Float32)
                bo.buffer_store(
                    v, rsrc(a["coef_out"]), t * COEF_ROWS + row, cache_modifier=CM_DEV
                )


@flyc.jit
def run_dpre(c):
    bid = c["bid"]
    rg, kp = bid // K_PARTS, bid % K_PARTS
    stamp(c, 0)
    loaded = norm_in_loads(c, kp)
    w0 = w_loads(c, c["args"]["w_kp"], H, rg * C_ROWS, C_TILES, kp)
    stage_norm_in(c, rg, kp, loaded)
    stamp(c, 1)
    # W_qkv (immutable) streams during the coefficient GEMV and the coefficient hops
    wq = w_loads(c, c["args"]["w_qkv"], H, rg * A_ROWS, A_TILES, kp)
    stage_coef(c, rg, kp, w0)
    stamp(c, 2)
    stamp(c, 3)
    stage_conv0_fold(c, kp)
    stamp(c, 4)
    stage_qkv(c, rg, kp, wq)
    stamp(c, 5)
    stage_coef_side1(c, bid)
    stage_heads(c, bid)
    stamp(c, 6)


# ---------------------------------------------------------------- D-post stages
# ---------------------------------------------------------------- the TP sum's zero-sentinel words
# The ranks' partials of D-post's TP sum are zero-sentinel words, not tagged pairs
# (``draft_mlp.zs_enc``): a (source rank, token, column pair)'s bf16 pair as one word,
# never 0, so an unwritten word reads 0. D-attn's o GEMV pushes them
# (``draft_attn.stage_og``) and D-post polls them; the peer region starts zeroed and
# every word is read and re-armed (0) by its one reader in the D-post after the D-attn
# that pushed it; the next push to it is a later D-attn's, which a peer reaches only
# past this rank's later D-MLP TP sum, behind the re-arm. Half a tagged pair's bytes.
def zs_word(src, t, col):
    """The word of (source rank, token t, column pair at ``col``) in the ``do`` region,
    blocked [rank][32-column block][token][16 pairs]: a CTA's block of every token is
    one contiguous run, so a wave's pushes are whole 256 B runs."""
    return ((src * (H // O_ROWS) + col // O_ROWS) * MAX_TOKENS + t) * (O_ROWS // 2) + (
        col % O_ROWS
    ) // 2


def zs_push(peers, rank, t, col, v0, v1):
    """This rank's (v0, v1) of (token t, columns col, col + 1) to every rank's region."""
    enc = zs_enc(bf16_pair(v0, v1).bitcast(fx.Int32))
    for pe in range_constexpr(len(peers)):
        bo.buffer_store(
            enc, rsrc(peers[pe].value), zs_word(rank, t, col), cache_modifier=CM_SYS
        )


def post_split(s: int) -> bool:
    """``stage_post``'s items fit half the threads (S <= 16); wider steps take
    ``stage_post_wide``."""
    return (O_ROWS // 2) * s <= THREADS // 2


@flyc.jit
def stage_post_wide(c, bid):
    """``stage_post`` for the steps whose items do not fit half the threads
    (``post_split``): side 1's coefficients, then the ranks' words polled and summed
    in passes of THREADS items, then conv1 over the block in passes."""
    S, W, tid = c["S"], c["W"], c["tid"]
    dbuf, cf1 = c["dbuf"], c["cf1"]
    a = c["args"]
    GP = O_ROWS // GROUP  # 2 groups a CTA
    PAIRS = O_ROWS // 2
    r_coef, r_base, r_out = rsrc(a["coef"]), rsrc(a["base"]), rsrc(a["attn_out"])
    N1 = S * TAPS * GP
    N = PAIRS * S
    for p in range_constexpr(-(-N1 // THREADS)):
        idx = fx.min(tid + p * THREADS, N1 - 1)
        t, tap, g = idx // (TAPS * GP), (idx // GP) % TAPS, idx % GP
        v = fx.Float32(bo.buffer_load(r_coef, t * COEF_ROWS + (1 * TAPS + tap) * GROUPS + bid * GP + g, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))  # fmt: skip
        if tid + p * THREADS < N1:
            fx.ptr_store(v, cf1 + idx)
    for p in range_constexpr(-(-N // THREADS)):
        q = tid + p * THREADS
        qq = fx.min(q, N - 1)
        jq, tq = qq % PAIRS, qq // PAIRS
        colq = bid * O_ROWS + 2 * jq
        if q < N:
            d0, d1 = sum_partials(
                zs_poll,
                c["own_do"],
                lambda src, tq=tq, colq=colq: zs_word(src, tq, colq),
                W,
            )
            fx.ptr_store(d0, dbuf + (tq * O_ROWS + 2 * jq))
            fx.ptr_store(d1, dbuf + (tq * O_ROWS + 2 * jq + 1))
    gpu.barrier()
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        col = bid * O_ROWS + 2 * j
        g = (2 * j) // GROUP
        bw = [bf2_f32(fx.Int32(bo.buffer_load(r_base, ((1 * TAPS + tap) * H + col) // 2, vec_width=1, dtype=T.i32))) for tap in range(TAPS)]  # fmt: skip
        has_prev = (t % BLOCK) >= 1
        tp = fx.max(t - 1, fx.Int32(0))
        vals = []
        for e in range_constexpr(2):
            c0 = bw[0][e] + fx.ptr_load(cf1 + ((t * TAPS + 0) * GP + g))
            c1 = bw[1][e] + fx.ptr_load(cf1 + ((t * TAPS + 1) * GP + g))
            cur = fx.ptr_load(dbuf + (t * O_ROWS + 2 * j + e))
            prev = fx.ptr_load(dbuf + (tp * O_ROWS + 2 * j + e))
            vals.append(c0 * cur + has_prev.select(c1 * prev, fx.Float32(0.0)))
        if tid + p * THREADS < N:
            bo.buffer_store(fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_out, (t * H + col) // 2)  # fmt: skip


@flyc.jit
def stage_post(c, bid):
    """Rows 32 bid .. of d for every token: the ranks' zero-sentinel bf16 pair words
    (pushed to every rank's ``do`` region by D-attn's o GEMV, ``draft_attn.stage_og``)
    summed in rank order (two rows a thread) into ``dbuf``, conv1 with side 1's
    coefficients (``coef`` [S, 2048] fp32, D-pre's), attn_out (bf16)."""
    S, W, tid = c["S"], c["W"], c["tid"]
    dbuf, cf1 = c["dbuf"], c["cf1"]
    a = c["args"]
    poll = zs_poll
    GP = O_ROWS // GROUP  # 2 groups a CTA
    PAIRS = O_ROWS // 2
    r_coef, r_base, r_out = rsrc(a["coef"]), rsrc(a["base"]), rsrc(a["attn_out"])
    N1 = S * TAPS * GP
    N = PAIRS * S
    assert post_split(S) and N1 <= N

    def cf_load(idx):
        t, tap, g = idx // (TAPS * GP), (idx // GP) % TAPS, idx % GP
        return fx.Float32(bo.buffer_load(r_coef, t * COEF_ROWS + (1 * TAPS + tap) * GROUPS + bid * GP + g, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))  # fmt: skip

    # conv1's base weights of the output items, loaded ahead of the polls
    idx = fx.min(tid, N - 1)
    j, t = idx % PAIRS, idx // PAIRS
    col = bid * O_ROWS + 2 * j
    bw = [bf2_f32(fx.Int32(bo.buffer_load(r_base, ((1 * TAPS + tap) * H + col) // 2, vec_width=1, dtype=T.i32))) for tap in range(TAPS)]  # fmt: skip
    # threads 0 .. N - 1 poll the ranks' words of the items (D-attn pushed this rank's)
    q = tid
    poller = (q >= 0) & (q < N)
    qq = fx.min(fx.max(q, fx.Int32(0)), N - 1)
    jq, tq = qq % PAIRS, qq // PAIRS
    colq = bid * O_ROWS + 2 * jq

    def pair_at(src):
        return zs_word(src, tq, colq)

    d0, d1 = fx.Float32(0.0), fx.Float32(0.0)
    if poller:
        cfv = cf_load(fx.min(qq, N1 - 1))
        d0, d1 = sum_partials(poll, c["own_do"], pair_at, W)
        if q < N1:
            fx.ptr_store(cfv, cf1 + qq)
    if poller:
        fx.ptr_store(d0, dbuf + (tq * O_ROWS + 2 * jq))
        fx.ptr_store(d1, dbuf + (tq * O_ROWS + 2 * jq + 1))
    gpu.barrier()
    g = (2 * j) // GROUP
    has_prev = (t % BLOCK) >= 1
    tp = fx.max(t - 1, fx.Int32(0))
    vals = []
    for e in range_constexpr(2):
        c0 = bw[0][e] + fx.ptr_load(cf1 + ((t * TAPS + 0) * GP + g))
        c1 = bw[1][e] + fx.ptr_load(cf1 + ((t * TAPS + 1) * GP + g))
        cur = fx.ptr_load(dbuf + (t * O_ROWS + 2 * j + e))
        prev = fx.ptr_load(dbuf + (tp * O_ROWS + 2 * j + e))
        vals.append(c0 * cur + has_prev.select(c1 * prev, fx.Float32(0.0)))
    if tid < N:
        bo.buffer_store(fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_out, (t * H + col) // 2)  # fmt: skip


@flyc.jit
def run_dpost(c):
    stamp(c, 0)
    stamp(c, 1)  # (the o GEMV is D-attn's: ``draft_attn.stage_og``)
    if const_expr(post_split(c["S"])):
        stage_post(c, c["bid"])
    else:
        stage_post_wide(c, c["bid"])
    stamp(c, 2)


# ---------------------------------------------------------------- the launches
class DPreBuild(msgspec.Struct, frozen=True):
    tokens: int
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


class DPostBuild(msgspec.Struct, frozen=True):
    tokens: int
    npes: int = 8
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


def _common(key, lds_members: dict, args: dict, tag, extra: dict, layer=None) -> dict:
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": key.tokens, "tid": tid, "bid": fx.block_idx.x, "lane": tid % 64, "wave": tid // 64,
        "put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll, "args": args, "trace_on": key.trace,
        "layer": layer,
    }  # fmt: skip
    c.update(lds_members)
    c.update(extra)
    return c


def build_dpre(key: DPreBuild):
    """``@flyc.jit`` launcher of D-pre for ``key.tokens`` rows. Arguments: ``x`` /
    ``res`` bf16 [S, H] (``first``: res is x), ``w_in`` bf16 [H], ``w_kp`` bf16 [2048,
    H], ``base`` bf16 [2, 2, H], ``w_qkv`` bf16 [1280, H], ``q_norm`` / ``k_norm`` bf16
    [128], ``cos_sin`` fp32 [*, 128], ``positions`` (int32 / int64 [S]; ``pos64``),
    ``slots`` (int32 / int64 [S]: the rows' cache slots; ``slot64``), ``k_cache`` /
    ``v_cache`` fp8 [slots, 1, 128], ``res_out`` bf16 [S, H], ``q`` bf16 [S, 1024],
    ``coef`` fp32 [S, 2048], ``scratch``, ``epoch``, ``layer``."""
    S = key.tokens
    assert S % WAVES == 0 and 1 <= S <= MAX_TOKENS
    name = kernel_symbol("muse_spark_dpre", s=S, f=key.fault_site, t=int(key.trace))
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = dp_scratch_layout()
    Lds = dp_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpre(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
        cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, k_cache: Int64, v_cache: Int64,
        res_out: Int64, q: Int64, coef: Int64, scratch: Int64, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "x": x, "res": res, "first": first, "w_in": w_in, "w_kp": w_kp, "base": base, "w_qkv": w_qkv,
                "q_norm": q_norm, "k_norm": k_norm, "cos_sin": cos_sin, "positions": positions, "pos64": pos64,
                "slots": slots, "slot64": slot64, "k_cache": k_cache, "v_cache": v_cache, "res_out": res_out,
                "q": q, "coef_out": coef,
            }  # fmt: skip
            members = {"red": lds.red.ptr, "hbuf": lds.hbuf.ptr, "ssl": lds.ssl.ptr, "nf": lds.nf.ptr, "cf": lds.cf.ptr, "hrow": lds.hrow.ptr, "hss": lds.hss.ptr}  # fmt: skip
            c = _common(key, members, args, step_tag(epoch, layer), {}, layer=layer)
            for region, (off, _) in layout.items():
                c[region[3:]] = sreg(scratch, off, region)  # "dp_cpart" -> c["cpart"]
            run_dpre(c)

    @flyc.jit
    def launch(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
        cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, k_cache: Int64, v_cache: Int64,
        res_out: Int64, q: Int64, coef: Int64, scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dpre(
            x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin, positions, pos64, slots, slot64, k_cache,
            v_cache, res_out, q, coef, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


def build_dpost(key: DPostBuild):
    """``@flyc.jit`` launcher of D-post for ``key.tokens`` rows on a rank of a
    ``key.npes``-way TP group. Arguments: ``o`` bf16 [S, 1024], ``w_o`` bf16 [H, 1024],
    ``coef`` fp32 [S, 2048] (D-pre's), ``base`` bf16 [2, 2, H], ``attn_out`` bf16 [S,
    H], ``scratch``, ``sym``, ``peers``, ``rank``, ``epoch``, ``layer``."""
    S, npes = key.tokens, key.npes
    assert S % WAVES == 0 and 1 <= S <= MAX_TOKENS
    name = kernel_symbol(
        "muse_spark_dpost", s=S, w=npes, f=key.fault_site, t=int(key.trace)
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout, sym_layout = do_scratch_layout(), do_sym_layout(npes)
    Lds = do_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpost(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64, scratch: Int64, sym: Int64, peers: Int64,
        rank: Int32, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "o": o,
                "w_o": w_o,
                "coef": coef,
                "base": base,
                "attn_out": attn_out,
            }
            bases = peer_bases(peers, npes)
            do_off = sym_layout["do"][0]
            extra = {
                "W": npes,
                "rank": rank,
                "own_do": preg(sym, do_off, "do"),
                "peer_do": [preg(b, do_off, "do") for b in bases],
            }
            members = {"red": lds.red.ptr, "dbuf": lds.dbuf.ptr, "cf1": lds.cf1.ptr}
            c = _common(key, members, args, step_tag(epoch, layer), extra, layer=layer)
            for region, (off, _) in layout.items():
                c[region[3:]] = sreg(scratch, off, region)
            run_dpost(c)

    @flyc.jit
    def launch(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64, scratch: Int64, sym: Int64, peers: Int64,
        rank: Int32, epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dpost(
            o, w_o, coef, base, attn_out, scratch, sym, peers, rank, epoch, layer
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
