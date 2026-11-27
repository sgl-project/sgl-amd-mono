# SPDX-License-Identifier: Apache-2.0
# Conv / gate / recurrence adapted from SGLang's gfx950 FlyDSL KDA verify
# (sglang/kernels/ops/attention/linear/kda_verify_gfx950_flydsl/kernel.py,
# sgl-project/sglang#51126, Apache-2.0).
"""GLM-5.3-Flash KDA verify on the ReplaySSM ring for 16-128 rows: one task a
(request, head), 512 threads.

The task first computes the head's f_b and g_b rows for its L tokens (bf16 MFMA, K =
128, the low-rank gates SGLang gets from ``fused_fg_b_proj``), then each 256-thread
half runs SGLang's FlyDSL verify body on one 64-wide V tile: width-4 conv + SiLU of
q / k / v, the safe gate, beta, the delta rule on the committed state (read only), and
the ring writes (raw k / v, gate, beta; the conv windows with ``has_iw``) bit for bit
as that kernel. Last, wave t applies the gated RMSNorm (o_norm, sigmoid(g)) to token
t's 128 outputs -> ``core_out`` [rows, 2048] bf16, o_proj's input.

The in_proj output ``proj`` [rows, 6416] (q | k | v | beta | f_a | g_a) is read with
plain loads after ``c["wait_proj"](req, head)``.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import butterfly, mfma_bf16, rsrc
from sgl_amd_mono.common.plan import THREADS, kernel_sources, key_tuple

KD = 128
NH = 16
P = NH * KD
NPROJ = 3 * P + NH + 2 * KD
BETA_0 = 3 * P
FA_0 = BETA_0 + NH
GA_0 = FA_0 + KD
HALF = 256
KSPLIT = 8
KG = KD // (4 * KSPLIT)
NCG = HALF // KSPLIT
BVC = 64
VPT = BVC // NCG
LAG = 2
LOG2E = struct.unpack("f", struct.pack("I", 0x3FB8AA3B))[0]
assert THREADS == 2 * HALF and KD // BVC == 2


def _load(addr, off, dtype, vec=1, cm=0):
    return bo.buffer_load(
        rsrc(addr), off, vec_width=vec, dtype=dtype, cache_modifier=cm
    )


def _load_uniform(addr, i):
    return fx.Int32(
        bo.buffer_load(rsrc(addr), i, vec_width=1, dtype=T.i32, is_scalar=True)
    )


def _uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))


def _dpp(v, ctrl):
    x = fx.Float32(v).bitcast(fx.Int32)
    return fx.Int32(update_dpp_i32(x, x, ctrl, 0xF, 0xF, False)).bitcast(fx.Float32)


def _sum8(v):
    """Sum over aligned groups of 8 lanes: quad xor 1, quad xor 2, row_half_mirror."""
    v = v + _dpp(v, 0xB1)
    v = v + _dpp(v, 0x4E)
    return v + _dpp(v, 0x141)


def _readlane(v, lane):
    x = fx.Float32(v).bitcast(fx.Int32)
    return fx.Int32(rocdl.readlane(T.i32, x, fx.Int32(lane))).bitcast(fx.Float32)


def _wave_sum(v):
    v = _sum8(v)
    v = v + _dpp(v, 0x140)
    return _readlane(v, 0) + _readlane(v, 16) + _readlane(v, 32) + _readlane(v, 48)


def _after(x, deps):
    """x through an opaque move that also reads deps, so its users wait for them."""
    args = [fx.Float32(x).bitcast(fx.Int32).ir_value()]
    args += [fx.Float32(d).bitcast(fx.Int32).ir_value() for d in deps]
    y = llvm.inline_asm(
        T.i32, args, "v_mov_b32 $0, $1", "=v" + ",v" * len(args), has_side_effects=False
    )
    return fx.Int32(y).bitcast(fx.Float32)


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _rsq(x):
    return fx.Float32(rocdl.rsq(T.f32, fx.Float32(x).ir_value()))


def _tl_exp(x):
    """tl.exp on gfx950: v_exp_f32(fma(x, log2e, off)) * 2^-off, off = 64 below 2^-126."""
    small = (x * fx.Float32(LOG2E)) < fx.Float32(-126.0)
    off = small.select(fx.Float32(64.0), fx.Float32(0.0))
    r = _exp2(fmath.fma(x, fx.Float32(LOG2E), off))
    return r * small.select(fx.Float32(2.0**-64), fx.Float32(1.0))


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + _tl_exp(-x))


def _conv_silu(xs, w):
    """Width-4 conv tap sum (every product rounded) then SiLU, rounded to bf16."""
    acc = fx.Float32(0.0) + fx.Float32(xs[0]) * w[0]
    for i in range(1, 4):
        acc = acc + fx.Float32(xs[i]) * w[i]
    return (acc / (fx.Float32(1.0) + _tl_exp(-acc))).to(fx.BFloat16)


def lds_floats(tt):
    """f32 LDS words a task uses: f / g gates (shared), then per half q|k, decay, v,
    beta, sums of squares, and the head's outputs (shared)."""
    half = 2 * tt * KD + tt * KD + tt * BVC + 16 + 4 * tt
    return 2 * tt * KD + 2 * half + tt * KD


@flyc.jit
def kda_head(c, n, hv, lds):
    """Request n's head hv; ``lds`` = ``lds_floats(TT)`` f32 words."""
    TT, tid = c["TT"], c["tid"]
    a = c["args"]
    lane, wave = tid % 64, tid // 64
    vt = tid // HALF
    th = tid % HALF
    hbase = 2 * TT * KD + vt * (3 * TT * KD + TT * BVC + 16 + 4 * TT)
    s_fa, s_ga = lds, lds + TT * KD
    s_qk = lds + hbase
    s_dec = s_qk + 2 * TT * KD
    s_v = s_dec + TT * KD
    s_beta = s_v + TT * BVC
    s_ss = s_beta + 16
    s_o = lds + 2 * TT * KD + 2 * (3 * TT * KD + TT * BVC + 16 + 4 * TT)
    x, xs_ = a["proj"], NPROJ
    pcm = c.get("proj_cm", 0)
    v0 = vt * BVC
    bos = n * TT
    L = a["ring_len"]
    cs_slot = _load_uniform(a["cs_idx"], n)
    h0 = _load_uniform(a["h0_idx"], n)
    ring = _load_uniform(a["ring_idx"], n)
    live = cs_slot >= 0
    ring_ok = live & (ring >= 0)
    qk_owner = vt == 0
    ring_c = fx.Int64(fx.max(ring, 0))
    cg = th // KSPLIT
    ks = th % KSPLIT

    # loads that wait on nothing: f_b / g_b rows, conv weights and state, the state tile
    gr = hv * KD + wave * 16 + lane % 16
    gw = [
        [
            fx.Vector(
                _load(
                    a["w_fgb"],
                    ((which * P + gr) * KD + (lane // 16) * 32) // 2 + 4 * q,
                    T.i32,
                    4,
                )
            )
            for q in range(4)
        ]
        for which in range(2)
    ]
    ch = th % KD
    is_k = _uniform(th) >= KD
    qk_ch = is_k.select(fx.Int32(P), fx.Int32(0)) + hv * KD + ch
    cs_base = a["conv_st"] + fx.Int64(fx.max(cs_slot, 0)) * fx.Int64(a["cs_line"]) * 2
    cs_tok = a["cs_tok"]
    xw = [fx.BFloat16(_load(cs_base, qk_ch + j * cs_tok, T.bf16)) for j in range(3)]
    wq = fx.Vector(_load(a["conv_w"], qk_ch * 4, T.f32, 4))
    has_v = is_k & (ch < BVC)
    v_ch = 2 * P + hv * KD + v0 + fx.min(ch, BVC - 1)
    vw = [fx.BFloat16(_load(cs_base, v_ch + j * cs_tok, T.bf16)) for j in range(3)]
    wv = fx.Vector(_load(a["conv_w"], v_ch * 4, T.f32, 4))
    dtb = fx.Float32(_load(a["dt_bias"], hv * KD + ch, T.f32))
    a_log = fx.Float32(_load(a["a_log"], hv, T.f32))
    state_bf16 = c["state_bf16"]
    st_base = a["state"] + fx.Int64(fx.max(h0, 0)) * fx.Int64(
        NH * KD * KD * (2 if state_bf16 else 4)
    )
    state_t = T.bf16 if state_bf16 else T.f32
    hs = []
    for j in range_constexpr(VPT):
        row = (hv * KD + v0 + cg + NCG * j) * KD
        hs.append(
            [
                fx.Vector(_load(st_base, row + (i * KSPLIT + ks) * 4, state_t, 4))
                for i in range(KG)
            ]
        )

    c["wait_proj"](n, hv)
    if const_expr("stamp" in c):
        c["stamp"](8)
    # every proj load goes out together: one round trip
    t = fx.min(lane % 16, TT - 1)
    xbs = [
        [
            fx.Vector(
                _load(
                    x,
                    ((bos + t) * xs_ + col0 + (lane // 16) * 32 + 8 * q) // 2,
                    T.i32,
                    4,
                    pcm,
                )
            )
            for q in range(4)
        ]
        for col0 in (FA_0, GA_0)
    ]
    xw += [
        fx.BFloat16(_load(x, (bos + tt) * xs_ + qk_ch, T.bf16, 1, pcm))
        for tt in range(TT)
    ]
    vw += [
        fx.BFloat16(_load(x, (bos + tt) * xs_ + v_ch, T.bf16, 1, pcm))
        for tt in range(TT)
    ]
    b_row = bos + fx.min(ch, TT - 1)
    bb = fx.Float32(fx.BFloat16(_load(x, b_row * xs_ + BETA_0 + hv, T.bf16, 1, pcm)))
    # f_b / g_b: wave w's MFMA gives rows 16 w .. of both for the TT tokens (bf16 out)
    for which in range_constexpr(2):
        gacc = fx.Vector.filled(4, 0.0, fx.Float32)
        for q in range_constexpr(4):
            gacc = mfma_bf16(
                gw[which][q].bitcast(fx.BFloat16),
                xbs[which][q].bitcast(fx.BFloat16),
                gacc,
            )
        dst = s_fa if which == 0 else s_ga
        for i in range_constexpr(4):
            kk = wave * 16 + 4 * (lane // 16) + i
            if lane % 16 < TT:
                fx.ptr_store(
                    fx.Float32(fx.Float32(gacc[i]).to(fx.BFloat16)), dst + (t * KD + kk)
                )
    gpu.barrier()
    if const_expr("stamp" in c):
        c["stamp"](9)

    # conv + silu, rounded to bf16 as the unfused path stores it
    qk, rk_bf = [], []
    wql = [wq[i] for i in range(4)]
    qk_row = is_k.select(fx.Int32(TT * KD), fx.Int32(0))
    rawk_base = a["rawk"] + ring_c * fx.Int64(a["rawk_slot"]) * 2
    for tt in range_constexpr(TT):
        r = _conv_silu(xw[tt : tt + 4], wql)
        qk.append(fx.Float32(r))
        rk_bf.append(r)
        fx.ptr_store(fx.Float32(r), s_qk + (qk_row + tt * KD + ch))
    if const_expr("stamp" in c):
        c["stamp"](13)
    wvl = [wv[i] for i in range(4)]
    rawv_base = a["rawv"] + ring_c * fx.Int64(a["rawv_slot"]) * 2
    if has_v:
        for tt in range_constexpr(TT):
            r = _conv_silu(vw[tt : tt + 4], wvl)
            fx.ptr_store(fx.Float32(r), s_v + (tt * BVC + ch))
    if const_expr("stamp" in c):
        c["stamp"](14)
    g_base = a["ring_g"] + ring_c * fx.Int64(a["g_slot"]) * 4
    lb = fx.Float32(c["lower_bound"])
    if _uniform(th) < KD:
        ea = _tl_exp(a_log)
        for tt in range_constexpr(TT):
            y = ea * (fx.ptr_load(s_fa + (tt * KD + ch)) + dtb)
            g = lb * (fx.Float32(1.0) / (fx.Float32(1.0) + _tl_exp(-y)))
            fx.ptr_store(_tl_exp(g), s_dec + (tt * KD + ch))
    beta_base = a["ring_beta"] + ring_c * fx.Int64(a["beta_slot"]) * 4
    if is_k & (ch < TT):
        beta = fx.Float32(1.0) / (fx.Float32(1.0) + _tl_exp(-bb))
        fx.ptr_store(beta, s_beta + ch)
    if const_expr("stamp" in c):
        c["stamp"](15)
    # l2 norm sums of squares: waves 0-1 of a half hold the q rows, waves 2-3 the k rows
    ss = [_wave_sum(qk[tt] * qk[tt]) for tt in range(TT)]
    if th % 64 == 0:
        for tt in range_constexpr(TT):
            fx.ptr_store(ss[tt], s_ss + ((th // 64) * TT + tt))
    gpu.barrier()
    if const_expr("stamp" in c):
        c["stamp"](10)

    alive = h0 >= 0
    hsf = [
        [
            alive.select(fx.Float32(hs[j][i // 4][i % 4]), fx.Float32(0.0))
            for i in range(4 * KG)
        ]
        for j in range(VPT)
    ]
    v4 = fx.Vector.make_type(4, fx.Float32)
    beta_t = [fx.ptr_load(s_beta + tt) for tt in range(TT)]
    out = [[None] * TT for _ in range(VPT)]
    for tt in range_constexpr(TT):
        rk = _rsq(
            fx.ptr_load(s_ss + (2 * TT + tt))
            + fx.ptr_load(s_ss + (3 * TT + tt))
            + fx.Float32(1e-6)
        )
        rq = _rsq(
            fx.ptr_load(s_ss + tt) + fx.ptr_load(s_ss + (TT + tt)) + fx.Float32(1e-6)
        ) * fx.Float32(KD**-0.5)
        d, k, q = [], [], []
        for i in range_constexpr(KG):
            off = tt * KD + (i * KSPLIT + ks) * 4
            fd = fx.Vector(fx.ptr_load(s_dec + off, result_type=v4))
            fk = fx.Vector(fx.ptr_load(s_qk + (TT * KD + off), result_type=v4))
            fq = fx.Vector(fx.ptr_load(s_qk + off, result_type=v4))
            d += [fd[e] for e in range(4)]
            k += [fk[e] for e in range(4)]
            q += [fq[e] for e in range(4)]
        if const_expr(tt >= LAG):
            d[0] = _after(d[0], [out[j][tt - LAG] for j in range(VPT)])
        pk = []
        for j in range_constexpr(VPT):
            acc = [fx.Float32(0.0)] * 4
            for i in range_constexpr(4 * KG):
                hsf[j][i] = hsf[j][i] * d[i]
                acc[i % 4] = fmath.fma(hsf[j][i], k[i], acc[i % 4])
            pk.append((acc[0] + acc[2]) + (acc[1] + acc[3]))
        pk = [_sum8(p) for p in pk]
        for j in range_constexpr(VPT):
            vv = fx.ptr_load(s_v + (tt * BVC + cg + NCG * j))
            u = fmath.fma(-pk[j], rk, vv) * beta_t[tt] * rk
            acc = [fx.Float32(0.0)] * 4
            for i in range_constexpr(4 * KG):
                hsf[j][i] = fmath.fma(k[i], u, hsf[j][i])
                acc[i % 4] = fmath.fma(hsf[j][i], q[i], acc[i % 4])
            out[j][tt] = (acc[0] + acc[2]) + (acc[1] + acc[3])
        for j in range_constexpr(VPT):
            out[j][tt] = _sum8(out[j][tt]) * rq
    # every lane of a column group holds the sums; lane ks keeps token ks
    for j in range_constexpr(VPT):
        val = out[j][0]
        for tt in range_constexpr(1, TT):
            val = (ks == tt).select(out[j][tt], val)
        if ks < TT:
            fx.ptr_store(
                fx.Float32(val.to(fx.BFloat16)), s_o + (ks * KD + v0 + cg + NCG * j)
            )
    gpu.barrier()
    if const_expr("stamp" in c):
        c["stamp"](11)

    # gated RMSNorm: wave tt = token tt
    core_out, out_cm, eps = rsrc(a["core_out"]), c.get("out_cm", 0), c["onorm_eps"]
    for tt0 in range_constexpr(0, TT, THREADS // 64):
        tt = fx.min(wave + tt0, TT - 1)
        e0 = 2 * lane
        o0 = fx.ptr_load(s_o + (tt * KD + e0))
        o1 = fx.ptr_load(s_o + (tt * KD + e0 + 1))
        rstd = _rsq(
            butterfly(o0 * o0 + o1 * o1, (32, 16, 8, 4, 2, 1)) * (1.0 / KD) + eps
        )
        won = fx.Vector(_load(a["on_w"], e0, T.f32, 2))
        y = [
            o * rstd * won[e] * _sigmoid(fx.ptr_load(s_ga + (tt * KD + e0 + e)))
            for e, o in enumerate((o0, o1))
        ]
        y = [live.select(v, fx.Float32(0.0)) for v in y]
        if wave + tt0 < TT:
            bo.buffer_store(
                fx.Vector.from_elements(y, fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0],
                core_out,
                ((bos + tt) * P + hv * KD + e0) // 2,
                cache_modifier=out_cm,
            )
    if const_expr("head_done" in c):
        c["head_done"](n, hv)

    # the ring and conv window, off core_out's path (the accept step reads them)
    for tt in range_constexpr(TT):
        if is_k & qk_owner & ring_ok & (fx.Int32(tt) < L):
            bo.buffer_store(rk_bf[tt], rsrc(rawk_base), (hv * L + tt) * KD + ch)
    if has_v & ring_ok:
        for tt in range_constexpr(TT):
            if fx.Int32(tt) < L:
                r = fx.ptr_load(s_v + (tt * BVC + ch)).to(fx.BFloat16)
                bo.buffer_store(r, rsrc(rawv_base), (hv * L + tt) * KD + v0 + ch)
    if (_uniform(th) < KD) & qk_owner & ring_ok:
        ea2 = _tl_exp(a_log)
        for tt in range_constexpr(TT):
            if fx.Int32(tt) < L:
                y = ea2 * (fx.ptr_load(s_fa + (tt * KD + ch)) + dtb)
                g = lb * (fx.Float32(1.0) / (fx.Float32(1.0) + _tl_exp(-y)))
                bo.buffer_store(g, rsrc(g_base), (hv * L + tt) * KD + ch)
    if is_k & (ch < TT) & qk_owner & ring_ok & (ch < L):
        bo.buffer_store(fx.ptr_load(s_beta + ch), rsrc(beta_base), hv * L + ch)
    if const_expr(c["has_iw"]):
        iw_slot = _load_uniform(a["iw_idx"], n)
        iw_base = a["iw"] + fx.Int64(fx.max(iw_slot, 0)) * fx.Int64(a["iw_line"]) * 2
        iw_ok = live & (iw_slot >= 0)
        iw_step, iw_win = a["iw_step"], a["iw_win"]
        if iw_ok & qk_owner:
            for tt in range_constexpr(TT):
                for j in range_constexpr(3):
                    bo.buffer_store(
                        xw[tt + 1 + j], rsrc(iw_base), tt * iw_step + qk_ch + j * iw_win
                    )
        if iw_ok & has_v:
            for tt in range_constexpr(TT):
                for j in range_constexpr(3):
                    bo.buffer_store(
                        vw[tt + 1 + j], rsrc(iw_base), tt * iw_step + v_ch + j * iw_win
                    )
    gpu.barrier()
    if const_expr("stamp" in c):
        c["stamp"](12)


# ---------------------------------------------------------------- standalone
SOURCES = kernel_sources(__name__)
_BUILDS: dict = {}

ARGS = (
    "proj",
    "conv_w",
    "conv_st",
    "cs_idx",
    "iw",
    "iw_idx",
    "w_fgb",
    "a_log",
    "dt_bias",
    "state",
    "h0_idx",
    "rawv",
    "rawk",
    "ring_g",
    "ring_beta",
    "ring_idx",
    "on_w",
    "core_out",
)
IARGS = (
    "cs_line",
    "cs_tok",
    "iw_line",
    "iw_step",
    "iw_win",
    "rawv_slot",
    "rawk_slot",
    "g_slot",
    "beta_slot",
    "ring_len",
)


def standalone(reqs, tt, lower_bound, onorm_eps, state_bf16, has_iw):
    """Grid (reqs * NH): one task a CTA."""
    keyed = key_tuple(locals(), SOURCES)
    key = (reqs, tt, lower_bound, onorm_eps, state_bf16, has_iw)
    if key in _BUILDS:
        return _BUILDS[key]
    nw = lds_floats(tt)

    @fx.struct
    class Smem:
        f: fx.Array[fx.Float32, nw, 16]

    I64, I32 = fx.Int64, fx.Int32

    def body(ptrs, ints):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek().f.ptr
        a = dict(zip(ARGS, ptrs))
        a.update(zip(IARGS, ints))
        c = dict(
            TT=tt,
            tid=tid,
            args=a,
            lower_bound=lower_bound,
            onorm_eps=onorm_eps,
            state_bf16=state_bf16,
            has_iw=has_iw,
            wait_proj=lambda n, hv: None,
        )
        b = fx.block_idx.x
        kda_head(c, b // NH, b % NH, lds)

    @flyc.kernel(
        name=f"g53w_kda_r{reqs}_t{tt}_s{int(state_bf16)}_iw{int(has_iw)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        proj: I64,
        conv_w: I64,
        conv_st: I64,
        cs_idx: I64,
        iw: I64,
        iw_idx: I64,
        w_fgb: I64,
        a_log: I64,
        dt_bias: I64,
        state: I64,
        h0_idx: I64,
        rawv: I64,
        rawk: I64,
        ring_g: I64,
        ring_beta: I64,
        ring_idx: I64,
        on_w: I64,
        core_out: I64,
        cs_line: I32,
        cs_tok: I32,
        iw_line: I32,
        iw_step: I32,
        iw_win: I32,
        rawv_slot: I32,
        rawk_slot: I32,
        g_slot: I32,
        beta_slot: I32,
        ring_len: I32,
    ):
        body(
            (
                proj,
                conv_w,
                conv_st,
                cs_idx,
                iw,
                iw_idx,
                w_fgb,
                a_log,
                dt_bias,
                state,
                h0_idx,
                rawv,
                rawk,
                ring_g,
                ring_beta,
                ring_idx,
                on_w,
                core_out,
            ),
            (
                cs_line,
                cs_tok,
                iw_line,
                iw_step,
                iw_win,
                rawv_slot,
                rawk_slot,
                g_slot,
                beta_slot,
                ring_len,
            ),
        )

    @flyc.jit
    def launch(
        proj: I64,
        conv_w: I64,
        conv_st: I64,
        cs_idx: I64,
        iw: I64,
        iw_idx: I64,
        w_fgb: I64,
        a_log: I64,
        dt_bias: I64,
        state: I64,
        h0_idx: I64,
        rawv: I64,
        rawk: I64,
        ring_g: I64,
        ring_beta: I64,
        ring_idx: I64,
        on_w: I64,
        core_out: I64,
        cs_line: I32,
        cs_tok: I32,
        iw_line: I32,
        iw_step: I32,
        iw_win: I32,
        rawv_slot: I32,
        rawk_slot: I32,
        g_slot: I32,
        beta_slot: I32,
        ring_len: I32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        kernel(
            proj,
            conv_w,
            conv_st,
            cs_idx,
            iw,
            iw_idx,
            w_fgb,
            a_log,
            dt_bias,
            state,
            h0_idx,
            rawv,
            rawk,
            ring_g,
            ring_beta,
            ring_idx,
            on_w,
            core_out,
            cs_line,
            cs_tok,
            iw_line,
            iw_step,
            iw_win,
            rawv_slot,
            rawk_slot,
            g_slot,
            beta_slot,
            ring_len,
        ).launch(grid=(reqs * NH, 1, 1), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch
