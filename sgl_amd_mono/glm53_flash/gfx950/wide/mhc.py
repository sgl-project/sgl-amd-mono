# SPDX-License-Identifier: Apache-2.0
# Slice / gate structure adapted from sgl_amd_mono.deepseek_v41.gfx950.stages.seam
# (itself from ROCm/ATOM atom/models/deepseek_v41/mono/kernels/attn_pre.py, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.).
"""GLM-5.3-Flash mHC pre (and the hc_post folded in front of it) for 16-128 rows,
on a residual every TP rank holds in full.

Two stages, handing off through ``c["put"]`` / ``c["getn"]`` (plain stores and loads in
a standalone launch, flagged mailboxes in the layer launch):

- ``slice_task(c, task)``: hidden columns 32 task .. + 32 of every row. With ``x_in``
  it first applies the owed hc_post, R' = post x + comb^T R (bf16), and stores R';
  then the slice's partial mix projections (fp32 MFMA 16x16x4 on fp32 ``hc_fn``) and
  sum of squares -> ``pmix[task][row][25]``.
- ``gate_row(c, t)``: row t's slices summed in slice order (so every rank gets the
  same bits) -> pre / post / comb (Sinkhorn on one wave), the layer input
  bf16(sum_n pre_n R'_n), its RMSNorm -> ``x_out``; post / comb -> ``post_out`` /
  ``comb_out``.

The math is SGLang's ``_mhc_pre_torch`` / ``_mhc_post_torch``
(``sglang/kernels/ops/layernorm/mhc.py``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    batched_rounds,
    bf2_f32,
    bf16_round,
    butterfly,
    div_rn,
    hw_exp,
    hw_rcp,
    hw_rsq,
    rsrc,
)
from sgl_amd_mono.common.plan import THREADS, WAVES, kernel_sources, key_tuple
from sgl_amd_mono.glm53_flash.gfx950.config import (
    EPS,
    HC_EPS,
    HC_MULT,
    HC_POST_MULT,
    HIDDEN,
    SINKHORN_ITERS,
)

HC = HC_MULT
K_MIX = HC * HIDDEN
MIX = HC * (HC + 2)  # pre 4 | post 4 | comb 16
PART = MIX + 1  # a slice's partials a row: the mixes, then the sum of squares
COLS = 32
SLICES = HIDDEN // COLS
KT = HC * COLS
GATE_BLOCK = 8
TILE = 16
CPT = HIDDEN // THREADS
COMB0 = 16  # comb[i][j] on lane COMB0 + 4 i + j
# LDS row stride of R' and fn rows: 16 B of padding puts an MFMA operand's 16 rows
# (and its 4 K columns) on distinct banks
KS = KT + 4


def lds_words(s, cap=96):
    """f32 LDS words the stages use: R' slice rows, the slice's fn, the reduce scratch."""
    br = s // bands(s, cap)
    red = max(br * UNITS, -(-br // TILE) * WAVES * 2 * 64 * 4, WAVES * 2 * 64 * 4)
    return {"rl": br * KS, "fl": MIX * KS, "red": red}


def pmix_words(s):
    return SLICES * s * PART


def ld_f32(ptr, i, cm=0):
    return fx.Float32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32, cache_modifier=cm)
    )


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def div(a, b):
    return div_rn(a, b, hw_rcp(b))


def sigmoid(a):
    return div(fx.Float32(1.0), 1.0 + hw_exp(-a))


def _plus(t0, v):
    return v if isinstance(t0, int) and t0 == 0 else v + t0


# ---------------------------------------------------------------- slice
VEC = 8  # columns a slice unit: one 16 B load of a bf16 row
UNITS = KT // VEC  # units a row


def bands(s, cap=96):
    """Row bands a slice is split into (tasks = SLICES * bands): 2 past 48 rows, and
    at most ``cap`` rows a band (a band's mix tiles are live across its K loop)."""
    b = max(-(-s // cap), 2 if s > 48 else 1)
    assert s % b == 0, s
    return b


def tasks(s, cap=96):
    return SLICES * bands(s, cap)


def _ld_row8(ptr, elem, cm=0, nbytes=None):
    """bf16 elements elem .. elem + 7 -> 8 f32."""
    w = fx.Vector(
        bo.buffer_load(
            rsrc(ptr, nbytes), elem // 2, vec_width=4, dtype=T.i32, cache_modifier=cm
        )
    )
    return [v for e in range(4) for v in bf2_f32(fx.Int32(w[e]))]


def _slice_at(c0, tb, u):
    """Unit u of a slice task at column c0, rows from tb: (row, local row, stream, column)."""
    tl = u // UNITS
    return tb + tl, tl, (u // (COLS // VEC)) % HC, c0 + (u % (COLS // VEC)) * VEC


def _slice_loads(c, c0, tb, u):
    a = c["args"]
    t, _, o, col = _slice_at(c0, tb, u)
    ccm = c.get("cm", 0)
    nb = a.get("res_bytes")  # rows past it read 0 (a caller's residual of fewer rows)
    if const_expr(a.get("x_in") is None):
        return (_ld_row8(a["res_in"], (t * HC + o) * HIDDEN + col, ccm, nb),)
    r = [_ld_row8(a["res_in"], (t * HC + i) * HIDDEN + col, ccm, nb) for i in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + i) * HC + o, ccm) for i in range(HC)]
    if const_expr(a.get("x_parts") is None):
        x = _ld_row8(a["x_in"], t * HIDDEN + col, c.get("x_cm", ccm))
    else:
        # x is the bf16 sum, in rank order, of tp bf16 planes of plane elements
        tp, plane = a["x_parts"]
        xs = [
            _ld_row8(a["x_in"], p * plane + t * HIDDEN + col, c.get("x_cm", ccm))
            for p in range(tp)
        ]
        x = xs[0]
        for p in range_constexpr(1, tp):
            x = [x[q] + xs[p][q] for q in range(VEC)]
        x = [bf16_round(v) for v in x]
    return r, cm, ld_f32(a["post_in"], t * HC + o, ccm), x


def _slice_row(c, c0, tb, u, got):
    a = c["args"]
    t, tl, o, col = _slice_at(c0, tb, u)
    if const_expr(a.get("x_in") is None):
        v = got[0]
    else:
        r, cm, post, x = got
        v = []
        for q in range_constexpr(VEC):
            # torch: post x + (sum over i of comb[i][o] r_i), then bf16
            acc = cm[0] * r[0][q]
            for i in range_constexpr(1, HC):
                acc = acc + cm[i] * r[i][q]
            v.append(bf16_round(post * x[q] + acc))
        bo.buffer_store(
            fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32),
            rsrc(a["res_out"]),
            ((t * HC + o) * HIDDEN + col) // 2,
            cache_modifier=c.get("cm", 0),
        )
    dst = c["rl"] + (tl * KS + o * COLS + col - c0)
    for h in range_constexpr(2):
        fx.ptr_store(
            fx.Vector.from_elements(v[4 * h : 4 * h + 4], fx.Float32), dst + 4 * h
        )
    sq = v[0] * v[0]
    for q in range_constexpr(1, VEC):
        sq = sq + v[q] * v[q]
    fx.ptr_store(sq, c["red"] + u)


def _slice_fn_loads(c, c0):
    tid, a = c["tid"], c["args"]
    nq = MIX * KT // 4
    out = []
    for i in range_constexpr((nq + THREADS - 1) // THREADS):
        q = fx.min(tid + THREADS * i, nq - 1)
        j = q // (KT // 4)
        o = (q % (KT // 4)) // (COLS // 4)
        cc = (q % (COLS // 4)) * 4
        out.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["hc_fn"]),
                    j * K_MIX + o * HIDDEN + c0 + cc,
                    vec_width=4,
                    dtype=T.f32,
                )
            )
        )
    return out


@flyc.jit
def slice_task(c, task):
    """R' (with ``x_in``), the mix partials and the sum of squares of one slice task:
    columns 32 (task % SLICES) .. + 32 of the rows in band task // SLICES."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl = c["rl"], c["fl"]
    br = c.get("BR", s // bands(s))
    c0 = (task % SLICES) * COLS
    tb = (task // SLICES) * br
    fn = _slice_fn_loads(c, c0)
    batched_rounds(
        tid,
        br * UNITS,
        lambda u: _slice_loads(c, c0, tb, u),
        lambda u, got: _slice_row(c, c0, tb, u, got),
    )
    if const_expr(c["args"].get("x_in") is not None):
        # R' out before any partial is published: the gate reads it after the partials
        rocdl.s_waitcnt(vmcnt=0)
    for i in range_constexpr(len(fn)):
        q = tid + THREADS * i
        if q < MIX * KT // 4:
            fx.ptr_store(fn[i], fl + (q // (KT // 4)) * KS + 4 * (q % (KT // 4)))
    gpu.barrier()
    # sum of squares: a row's unit partials in unit order
    if tid < br:
        sq = fx.ptr_load(c["red"] + tid * UNITS)
        for k in range_constexpr(1, UNITS):
            sq = sq + fx.ptr_load(c["red"] + tid * UNITS + k)
        c["put"](c["args"]["pmix"], ((task % SLICES) * s + tb + tid) * PART + MIX, sq)
    gpu.barrier()
    _slice_mix(c, task % SLICES, tb, br)


@flyc.jit
def _slice_mix(c, sl, tb, br):
    """Mix partials of the band's br rows (rows tb ..): fp32 MFMA 16x16x4, A = fn rows,
    B = R'^T, every 16-row tile in one pass; each wave a K slice, summed in wave order."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    nt = -(-br // TILE)
    toks = [fx.min(t0 * TILE + lane % 16, br - 1) for t0 in range(nt)]
    kw = KT // WAVES
    cs = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)] for _ in range(nt)]
    for kk in range_constexpr(kw // 4):
        colk = wave * kw + lane // 16 + 4 * kk
        av = []
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            v = fx.ptr_load(fl + (fx.min(row, MIX - 1) * KS + colk))
            av.append((row < MIX).select(v, fx.Float32(0.0)))
        for ti in range_constexpr(nt):
            b = fx.ptr_load(rl + (toks[ti] * KS + colk))
            for h in range_constexpr(2):
                ops = rocdl._split_mfma_operands([av[h], b, cs[ti][h], 0, 0, 0])
                cs[ti][h] = fx.Vector(
                    rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result
                )
    for ti in range_constexpr(nt):
        for h in range_constexpr(2):
            fx.ptr_store(
                cs[ti][h], red + (((ti * WAVES + wave) * 2 + h) * 64 + lane) * 4
            )
    gpu.barrier()
    for p0 in range_constexpr(0, MIX * br, THREADS):
        e = p0 + tid
        if e < MIX * br:
            tl = e // MIX
            j = e % MIX
            ti = tl // TILE
            h = j // 16
            ln = 16 * ((j % 16) // 4) + tl % TILE
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(
                    red + ((((ti * WAVES + w) * 2 + h) * 64 + ln) * 4 + j % 4)
                )
            c["put"](c["args"]["pmix"], (sl * s + tb + tl) * PART + j, tot)
    gpu.barrier()


# ---------------------------------------------------------------- gate
def _row_sum(v):
    return butterfly(v, (1, 2))


def _col_sum(v):
    return butterfly(v, (4, 8))


def _row_max(v):
    return butterfly(v, (1, 2), fx.max)


@flyc.jit
def gate_row(c, t):
    """Row t: mixes -> pre / post / comb, the layer input and its RMSNorm."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    r_src = a["res_out"] if const_expr(a.get("x_in") is not None) else a["res_in"]
    r_nb = None if const_expr(a.get("x_in") is not None) else a.get("res_bytes")
    ccm = c.get("cm", 0)
    groups = SLICES // GATE_BLOCK
    col = tid * CPT

    def load_rows():
        return [
            fx.Vector(
                bo.buffer_load(
                    rsrc(r_src, r_nb),
                    ((t * HC + i) * HIDDEN + col) // 2,
                    vec_width=CPT // 2,
                    dtype=T.i32,
                    cache_modifier=ccm,
                )
            )
            for i in range(HC)
        ]

    # R_in and the norm weights do not wait on the mixes; R' (the slices' output in
    # this launch) only once its partials are in
    post_in = a.get("x_in") is not None
    rw = None if post_in else load_rows()
    gw = fx.Vector(
        bo.buffer_load(rsrc(a["norm_w"]), col // 2, vec_width=CPT // 2, dtype=T.i32)
    )
    # the gate constants of wave 0's lanes
    jl = (lane < COMB0).select(
        lane % (2 * HC), fx.min(lane - (COMB0 - 2 * HC), MIX - 1)
    )
    sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
    base = ld_f32(a["hc_base"], jl)
    if tid < groups * PART:
        g = tid // PART
        j = tid % PART
        vals = c["getn"](
            a["pmix"],
            [((g * GATE_BLOCK + k) * s + t) * PART + j for k in range(GATE_BLOCK)],
        )
        acc = vals[0]
        for k in range_constexpr(1, GATE_BLOCK):
            acc = acc + vals[k]
        fx.ptr_store(acc, red + (g * PART + j))
    gpu.barrier()
    if const_expr(post_in):
        rw = load_rows()
    sv = red + groups * PART  # pre [HC] | ss [WAVES] | v [64]
    if wave == 0:
        # lanes 0..7 pre and post, COMB0.. comb (the lanes between repeat 0..7)
        mix = fx.ptr_load(red + jl)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + jl))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        v = mix * hw_rsq(sq * (1.0 / K_MIX) + EPS)
        fx.ptr_store(v, sv + HC + WAVES + lane)
        if lane < HC:
            fx.ptr_store(sigmoid(v * sc[0] + base) + HC_EPS, sv + lane)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(v * sc[1] + base) * HC_POST_MULT
            bo.buffer_store(
                post, rsrc(a["post_out"]), t * HC + lane - HC, cache_modifier=ccm
            )
    gpu.barrier()
    # the layer input needs only pre: x goes out before Sinkhorn
    pre = [fx.ptr_load(sv + i) for i in range(HC)]
    li = []
    for e in range_constexpr(CPT // 2):
        for h in range_constexpr(2):
            v = fx.Float32(0.0)
            for i in range_constexpr(HC):
                v = v + pre[i] * bf2_f32(fx.Int32(rw[i][e]))[h]
            li.append(bf16_round(v))
    ss = fx.Float32(0.0)
    for v in li:
        ss = ss + v * v
    ss = butterfly(ss, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ss, sv + HC + wave)
    gpu.barrier()
    tot = fx.ptr_load(sv + HC)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(sv + HC + w)
    rstd = hw_rsq(tot * (1.0 / HIDDEN) + EPS)
    gam = [g for e in range(CPT // 2) for g in bf2_f32(fx.Int32(gw[e]))]
    ys = fx.Vector.from_elements(
        [li[q] * rstd * gam[q] for q in range(CPT)], fx.Float32
    )
    c["store_x"](t, col, ys.to(fx.BFloat16).bitcast(fx.Int32))
    if const_expr("x_done" in c):
        c["x_done"](t)
    if wave == 0:
        cv = fx.ptr_load(sv + HC + WAVES + lane) * sc[2] + base
        cv = hw_exp(cv - _row_max(cv))
        cv = div(cv, _row_sum(cv)) + HC_EPS
        cv = div(cv, _col_sum(cv) + HC_EPS)
        for _ in range_constexpr(SINKHORN_ITERS - 1):
            cv = div(cv, _row_sum(cv) + HC_EPS)
            cv = div(cv, _col_sum(cv) + HC_EPS)
        if (lane >= COMB0) & (lane < COMB0 + HC * HC):
            bo.buffer_store(
                cv, rsrc(a["comb_out"]), t * HC * HC + lane - COMB0, cache_modifier=ccm
            )
    gpu.barrier()


# ---------------------------------------------------------------- standalone
SOURCES = kernel_sources(__name__)
_BUILDS: dict = {}


def standalone(s, post, which="both"):
    """Two launches: slices (grid SLICES), then gates (grid s). ``post``: fold the
    owed hc_post (x_in, post_in, comb_in -> res_out) in front. ``which``: "both",
    "slice" or "gate" (timing one stage)."""
    keyed = key_tuple(locals(), SOURCES)
    key = (s, post, which)
    if key in _BUILDS:
        return _BUILDS[key]
    lw = lds_words(s)

    lw_rl, lw_fl, lw_red = lw["rl"], lw["fl"], lw["red"]

    @fx.struct
    class Smem:
        rl: fx.Array[fx.Float32, lw_rl, 16]
        fl: fx.Array[fx.Float32, lw_fl, 16]
        red: fx.Array[fx.Float32, lw_red, 16]

    def ctx(args):
        tid = fx.thread_idx.x
        st = fx.SharedAllocator().allocate(Smem)
        sm = st.peek()

        def put(base, i, v):
            bo.buffer_store(v, rsrc(base), i)

        def getn(base, idxs):
            return [ld_f32(base, i) for i in idxs]

        def store_x(t, col, words):
            bo.buffer_store(words, rsrc(args["x_out"]), (t * HIDDEN + col) // 2)

        return dict(
            S=s,
            tid=tid,
            lane=tid % 64,
            wave=tid // 64,
            rl=sm.rl.ptr,
            fl=sm.fl.ptr,
            red=sm.red.ptr,
            args=args,
            put=put,
            getn=getn,
            store_x=store_x,
        )

    def unpack(
        res_in,
        res_out,
        x_in,
        post_in,
        comb_in,
        hc_fn,
        hc_scale,
        hc_base,
        norm_w,
        pmix,
        x_out,
        post_out,
        comb_out,
    ):
        a = dict(
            res_in=res_in,
            hc_fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            norm_w=norm_w,
            pmix=pmix,
            x_out=x_out,
            post_out=post_out,
            comb_out=comb_out,
        )
        if post:
            a.update(res_out=res_out, x_in=x_in, post_in=post_in, comb_in=comb_in)
        return a

    I64 = fx.Int64

    @flyc.kernel(
        name=f"g53w_mhc_slice_s{s}_p{int(post)}", known_block_size=[THREADS, 1, 1]
    )
    def slice_kernel(
        res_in: I64,
        res_out: I64,
        x_in: I64,
        post_in: I64,
        comb_in: I64,
        hc_fn: I64,
        hc_scale: I64,
        hc_base: I64,
        norm_w: I64,
        pmix: I64,
        x_out: I64,
        post_out: I64,
        comb_out: I64,
    ):
        args = unpack(
            res_in,
            res_out,
            x_in,
            post_in,
            comb_in,
            hc_fn,
            hc_scale,
            hc_base,
            norm_w,
            pmix,
            x_out,
            post_out,
            comb_out,
        )
        slice_task(ctx(args), fx.block_idx.x)

    @flyc.kernel(
        name=f"g53w_mhc_gate_s{s}_p{int(post)}", known_block_size=[THREADS, 1, 1]
    )
    def gate_kernel(
        res_in: I64,
        res_out: I64,
        x_in: I64,
        post_in: I64,
        comb_in: I64,
        hc_fn: I64,
        hc_scale: I64,
        hc_base: I64,
        norm_w: I64,
        pmix: I64,
        x_out: I64,
        post_out: I64,
        comb_out: I64,
    ):
        args = unpack(
            res_in,
            res_out,
            x_in,
            post_in,
            comb_in,
            hc_fn,
            hc_scale,
            hc_base,
            norm_w,
            pmix,
            x_out,
            post_out,
            comb_out,
        )
        gate_row(ctx(args), fx.block_idx.x)

    @flyc.jit
    def launch(
        res_in: I64,
        res_out: I64,
        x_in: I64,
        post_in: I64,
        comb_in: I64,
        hc_fn: I64,
        hc_scale: I64,
        hc_base: I64,
        norm_w: I64,
        pmix: I64,
        x_out: I64,
        post_out: I64,
        comb_out: I64,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        ptrs = (
            res_in,
            res_out,
            x_in,
            post_in,
            comb_in,
            hc_fn,
            hc_scale,
            hc_base,
            norm_w,
            pmix,
            x_out,
            post_out,
            comb_out,
        )
        if const_expr(which != "gate"):
            slice_kernel(*ptrs).launch(
                grid=(tasks(s), 1, 1), block=(THREADS,), stream=stream
            )
        if const_expr(which != "slice"):
            gate_kernel(*ptrs).launch(grid=(s, 1, 1), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch
