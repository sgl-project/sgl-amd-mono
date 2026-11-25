# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 attention front: one launch from a sparse layer's input to its
attention's inputs, for the layers whose input is the previous FFN launch's output.

SGLang runs it as three kernels: the input norm's residual add, Gemma RMSNorm and
per-token FP8 quant (``gemma_add_rmsnorm_fp8``), the fused qkv + index projection on
its rowwise-FP8 weight (``gemm_a8w8_bpreshuffle``), and the per-head Gemma norms,
NeoX RoPE over 64 of 128 columns and the NVFP4 K / V / index-K pool store
(``sparse_qk_index_gemma_rmsnorm_rope_cache``). This launch is ``HEADS x CHUNKS``
CTAs, CTA (h, c) the 128 projection columns of head h over hidden columns
``CHUNK c ..``:

    x       the chunk of every row: residual + MoE output in f32 (bf16 -> the new
            residual), its sum of squares, x (1 + w) quantized to FP8 a row's chunk
    gemm    a wave a 16-column tile of the head: the chunk's FP8 MFMA on the
            rowwise-FP8 weight, scaled by the row's chunk scale -> partial (f32)
    head    the head's last CTA to arrive: the partials summed in chunk order,
            x rstd (the norm's scale factors out of the GEMM) x the weight scale
            -> bf16, then by head: Q and index Q normed and rotated (bf16 out),
            K and index K normed, rotated and stored to the pool, V stored

Per rank (TP4): 16 Q heads, then K, V, index Q and index K, one 128-column head
each, as the fused weight lays them out. X is quantized per 512-column chunk rather
than per row, and the input norm's scale is applied after the GEMM: the bits differ
from SGLang's, the error is no larger.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from sgl_amd_mono.common.ops import (
    CM_NT,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    clock,
    fp8_pack4,
    hw_rcp,
    hw_rsq,
    ld_i32,
    rsrc,
    sload64,
    xshfl,
)
from sgl_amd_mono.common.plan import THREADS, kernel_sources, key_tuple

from .ffn import EPS, HIDDEN, INV_FP8_MAX, MAX_TOKENS
from .nvfp4 import nvfp4_row
from .sync import (
    _ptr,
    cta_sync,
    load_dev,
    store32,
    store_dev,
    sync_acquire,
    sync_release,
)

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

HEAD = 128
Q_HEADS = 16
K_HEAD, V_HEAD, IQ_HEAD, IK_HEAD = 16, 17, 18, 19
HEADS = 20
COLS = HEADS * HEAD  # the fused projection's columns on a TP4 rank
CHUNK = 512
CHUNKS = HIDDEN // CHUNK
BLOCKS = HEADS * CHUNKS
STEPS = CHUNK // 64  # the chunk's 64-wide K steps, two an MFMA
ROPE_HALF = 32  # NeoX pairs columns d and d + 32 of the 64 rotary ones
XCDS = 8
assert BLOCKS == XCDS * (2 * CHUNKS + (HEADS - 2 * XCDS) * CHUNKS // XCDS)

# scratch dwords: every CTA's partial, its rows' sums of squares, a counter a head
PART = 0
SSQ = PART + BLOCKS * MAX_TOKENS * HEAD
CNT = SSQ + BLOCKS * MAX_TOKENS
# timeline stamps (i64): start, X in LDS, partial stored, counter read, head done
TL_POINTS = 5
CNT_STRIDE = 32  # dwords: a counter a 128-byte line, so arrivals on others do not queue
TL = CNT + HEADS * CNT_STRIDE
SCRATCH_BYTES = 4 * (TL + 2 * BLOCKS * TL_POINTS)


@dataclass(frozen=True)
class FrontBuild:
    """The pool strides are (page, row) in bytes: main K / V payload and scales,
    then index K's."""

    tokens: int
    page: int  # rows of a pool page
    kv_pay_page: int
    kv_pay_row: int
    kv_sc_page: int
    kv_sc_row: int
    ik_pay_page: int
    ik_pay_row: int
    ik_sc_page: int
    ik_sc_row: int
    cos_bf16: bool = False  # the cos / sin cache's dtype (else f32)
    timeline: bool = False  # tid 0 of every CTA stamps the realtime clock at TL_POINTS

    def kv_strides(self):
        return (self.kv_pay_page, self.kv_pay_row), (self.kv_sc_page, self.kv_sc_row)

    def ik_strides(self):
        return (self.ik_pay_page, self.ik_pay_row), (self.ik_sc_page, self.ik_sc_row)


def head_chunk(bid):
    """CTA ``bid``'s (head, chunk). CTAs go to the 8 XCDs round-robin: a Q head's
    chunks share one XCD's L2 (heads x and x + 8 on XCD x), the K, V and index heads
    two (6 chunks on each), so a head's partials and counter stay near its last CTA."""
    x, j = bid % XCDS, bid // XCDS
    lo = j < 2 * CHUNKS
    h = lo.select(x + XCDS * (j // CHUNKS), K_HEAD + x // 2)
    c = lo.select(j % CHUNKS, (x % 2) * (CHUNKS // 2) + j - 2 * CHUNKS)
    return h, c


def kernel_name(key: FrontBuild) -> str:
    return f"minimax_m31_front_s{key.tokens}" + ("_tl" if key.timeline else "")


@flyc.jit
def stamp(a, k):
    if const_expr(a["timeline"]):  # noqa: SIM102
        if fx.thread_idx.x == 0:
            bo.buffer_store(
                clock(), rsrc(a["scratch"]), TL // 2 + fx.block_idx.x * TL_POINTS + k
            )


@flyc.jit
def store_nvfp4(pay, sc, strides, page, loc, lo, hi, lane, skip_pad):
    """A row's two columns a lane -> NVFP4 at pool row ``loc``; ``skip_pad``: not
    the CUDA-graph padding slot 0 (padding rows may hold NaN)."""
    pw, sw = nvfp4_row(lo, hi)
    (pp, pr), (sp, sr) = strides
    pg, row = loc // page, loc % page
    keep = (loc != 0) if const_expr(skip_pad) else (loc >= 0)
    if keep & ((lane & 3) == 0):
        bo.buffer_store(pw, rsrc(pay), (pg * pp + row * pr) // 4 + (lane >> 2))
    if keep & ((lane & 31) == 0):
        bo.buffer_store(sw, rsrc(sc), (pg * sp + row * sr) // 4 + (lane >> 5))


@flyc.jit
def cos_sin(a, pos, lane):
    """The cos and sin of lane's columns 2 lane, + 1 (modulo the 32 a rotary half)
    at ``pos``."""
    j = (2 * lane) % ROPE_HALF
    if const_expr(a["cos_bf16"]):
        cw = ld_i32(a["cs"], (pos * 64 + j) // 2)
        sw = ld_i32(a["cs"], (pos * 64 + ROPE_HALF + j) // 2)
        return bf_lo(cw), bf_hi(cw), bf_lo(sw), bf_hi(sw)
    cv = fx.Vector(
        bo.buffer_load(rsrc(a["cs"]), pos * 64 + j, vec_width=2, dtype=T.f32)
    )
    sv = fx.Vector(
        bo.buffer_load(
            rsrc(a["cs"]), pos * 64 + ROPE_HALF + j, vec_width=2, dtype=T.f32
        )
    )
    return cv[0], cv[1], sv[0], sv[1]


@flyc.jit
def head_out(a, h, row, lane, yb0, yb1, loc, cs4, nw):
    """Head h of ``row`` from the projection (bf16 as f32, columns 2 lane, + 1):
    V to the pool; the others Gemma-normed and NeoX-rotated as SGLang's tiled
    kernel does, then out (Q, index Q) or to the pool (K, index K). ``loc``,
    ``cs4`` (the row's cos / sin) and ``nw`` (the head norm's weights) are loaded
    ahead."""
    if h == V_HEAD:
        store_nvfp4(a["vp"], a["vsc"], a["kv"], a["page"], loc, yb0, yb1, lane, True)
    else:
        ss = butterfly(yb0 * yb0 + yb1 * yb1, (32, 16, 8, 4, 2, 1))
        hr = hw_rsq(ss * fx.Float32(1.0 / HEAD) + fx.Float32(EPS))
        n0 = bf16_round(yb0 * hr * (bf_lo(nw) + 1.0))
        n1 = bf16_round(yb1 * hr * (bf_hi(nw) + 1.0))
        # lane l ^ 16 holds the partner columns of l's (lanes below 32 rotate)
        p0, p1 = xshfl(n0, 16), xshfl(n1, 16)
        c0, c1, s0, s1 = cs4
        first = lane < 16
        r0 = first.select(n0 * c0 - p0 * s0, n0 * c0 + p0 * s0)
        r1 = first.select(n1 * c1 - p1 * s1, n1 * c1 + p1 * s1)
        rot = lane < ROPE_HALF
        o0 = bf16_round(rot.select(r0, n0))
        o1 = bf16_round(rot.select(r1, n1))
        if h < Q_HEADS:
            bo.buffer_store(
                bf16_pair(o0, o1).bitcast(fx.Int32),
                rsrc(a["q_out"]),
                row * (Q_HEADS * HEAD // 2) + h * (HEAD // 2) + lane,
            )
        if h == IQ_HEAD:
            bo.buffer_store(
                bf16_pair(o0, o1).bitcast(fx.Int32),
                rsrc(a["iq_out"]),
                row * (HEAD // 2) + lane,
            )
        if h == K_HEAD:
            store_nvfp4(a["kp"], a["ksc"], a["kv"], a["page"], loc, o0, o1, lane, True)
        if h == IK_HEAD:
            store_nvfp4(a["ip"], a["isc"], a["ik"], a["page"], loc, o0, o1, lane, False)


@flyc.jit
def finish_head(a, h, wave, lane, ws2, nw, locs, css):
    """The head's last CTA: every row's projection from the CTAs' partials; the
    wave's rows are wave + 8 k, their slots ``locs`` and cos / sin ``css``."""
    s, scr = a["S"], a["scratch"]
    for k in range_constexpr(len(locs)):
        row = wave + 8 * k
        if row < s:
            y0, y1, ss = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
            for cc in range_constexpr(CHUNKS):
                b = h * CHUNKS + cc
                v = load_dev(scr, PART + (b * MAX_TOKENS + row) * HEAD + 2 * lane, 2)
                y0 = y0 + fx.Int32(v[0]).bitcast(fx.Float32)
                y1 = y1 + fx.Int32(v[1]).bitcast(fx.Float32)
                ss = ss + load_dev(scr, SSQ + b * MAX_TOKENS + row).bitcast(fx.Float32)
            var = ss / fx.Float32(HIDDEN)
            rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS)))
            yb0 = bf16_round(y0 * rstd * ws2[0])
            yb1 = bf16_round(y1 * rstd * ws2[1])
            head_out(a, h, row, lane, yb0, yb1, locs[k], css[k], nw)


def build_front(key: FrontBuild):
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    keyed = key_tuple(key, SOURCES)

    @fx.struct
    class Lds:
        xa: fx.Array[fx.Int32, (STEPS // 2) * 64 * 8, 16]  # the MFMAs' A operands
        xs: fx.Array[fx.Float32, MAX_TOKENS, 16]  # a row's chunk scale
        last: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kfront(
        x: Int64,
        res: Int64,
        res_new: Int64,
        norm_w: Int64,
        w: Int64,
        ws: Int64,
        qn: Int64,
        kn: Int64,
        iqn: Int64,
        ikn: Int64,
        pos: Int64,
        cs: Int64,
        loc: Int64,
        kp: Int64,
        ksc: Int64,
        vp: Int64,
        vsc: Int64,
        ip: Int64,
        isc: Int64,
        q_out: Int64,
        iq_out: Int64,
        scratch: Int64,
    ):
        _ = keyed
        a = {
            "S": s,
            "page": key.page,
            "kv": key.kv_strides(),
            "ik": key.ik_strides(),
            "cos_bf16": key.cos_bf16,
            "timeline": key.timeline,
            "ws": ws,
            "qn": qn,
            "kn": kn,
            "iqn": iqn,
            "ikn": ikn,
            "pos": pos,
            "cs": cs,
            "loc": loc,
            "kp": kp,
            "ksc": ksc,
            "vp": vp,
            "vsc": vsc,
            "ip": ip,
            "isc": isc,
            "q_out": q_out,
            "iq_out": iq_out,
            "scratch": scratch,
        }
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        h, c = head_chunk(bid)
        slot = h * CHUNKS + c
        lds = fx.SharedAllocator().allocate(Lds).peek()
        xa, xsl, lastp = lds.xa.ptr, lds.xs.ptr, lds.last.ptr
        stamp(a, 0)
        k0 = c * CHUNK
        # row t's 16 columns 16 q .. of the chunk; rows past s read 0. Out first: a
        # wait on a load waits on every earlier one
        t, q = tid // 32, tid % 32
        col = k0 + q * 16
        nbytes = s * HIDDEN * 2
        rx, rr, rg = rsrc(x, nbytes), rsrc(res, nbytes), rsrc(norm_w)
        xw = [
            fx.Vector(
                bo.buffer_load(
                    rx, (t * HIDDEN + col) // 2 + 4 * j, vec_width=4, dtype=T.i32
                )
            )
            for j in range(2)
        ]
        rv = [
            fx.Vector(
                bo.buffer_load(
                    rr, (t * HIDDEN + col) // 2 + 4 * j, vec_width=4, dtype=T.i32
                )
            )
            for j in range(2)
        ]
        gv = [
            fx.Vector(bo.buffer_load(rg, col // 2 + 4 * j, vec_width=4, dtype=T.i32))
            for j in range(2)
        ]
        rocdl.sched_barrier(0)
        # the head's last CTA's inputs (every CTA loads them; only it reads them):
        # the slots and positions of the wave's rows wave + 8 k, the head's weight
        # scales and norm weights
        # (scalar loads: a uniform vector load's value would wait on the X loads)
        ks = [k for k in range(2) if 8 * k < s]
        rows = [fx.min(wave + 8 * k, s - 1) for k in ks]
        locs = [fx.Int32(sload64(loc + fx.Int64(8 * r))) for r in rows]
        poss = [fx.Int32(sload64(pos + fx.Int64(8 * r))) for r in rows]
        ws2 = fx.Vector(
            bo.buffer_load(rsrc(ws), h * HEAD + 2 * lane, vec_width=2, dtype=T.f32)
        )
        nwp = (h < Q_HEADS).select(
            qn, (h == K_HEAD).select(kn, (h == IQ_HEAD).select(iqn, ikn))
        )
        nw = ld_i32(nwp, lane)
        # the wave's tile over the chunk, AITER's (16, 16) shuffle as ``ffn.oproj``
        # reads it; the scheduler keeps the X loads ahead
        rocdl.sched_barrier(0)
        tile = h * (HEAD // 16) + wave
        rw = rsrc(w)
        wv = [
            fx.Vector(
                bo.buffer_load(
                    rw,
                    (tile * 16 * HIDDEN + k0 * 16 + lane * 16 + i * 1024) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
            for i in range(STEPS)
        ]
        rocdl.sched_barrier(0)
        sv, xg = [], []
        for j in range_constexpr(2):
            for d in range_constexpr(4):
                xd, rd, gd = fx.Int32(xw[j][d]), fx.Int32(rv[j][d]), fx.Int32(gv[j][d])
                lo, hi = bf_lo(xd) + bf_lo(rd), bf_hi(xd) + bf_hi(rd)
                sv += [lo, hi]
                xg += [lo * (bf_lo(gd) + 1.0), hi * (bf_hi(gd) + 1.0)]
        # past the X wait, so the positions are in: the rows' cos / sin
        css = [cos_sin(a, p_, lane) for p_ in poss]
        if (h == 0) & (t < s):
            words = [
                bf16_pair(sv[2 * i], sv[2 * i + 1]).bitcast(fx.Int32) for i in range(8)
            ]
            for j in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(words[4 * j : 4 * j + 4], fx.Int32),
                    rsrc(res_new),
                    (t * HIDDEN + col) // 2 + 4 * j,
                )
        sq = fx.Float32(0.0)
        mx = fx.Float32(0.0)
        for i in range_constexpr(16):
            sq = sq + sv[i] * sv[i]
            mx = fx.max(mx, fmath.absf(xg[i]))
        # a row is the 32 lanes of a half wave
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        mx = butterfly(mx, (16, 8, 4, 2, 1), fx.max)
        scale = mx * fx.Float32(INV_FP8_MAX)
        inv = (mx > 0.0).select(hw_rcp(scale), fx.Float32(0.0))
        qw = []
        for j in range_constexpr(4):
            vals = [
                fx.min(
                    fx.max(xg[4 * j + e] * inv, fx.Float32(-448.0)), fx.Float32(448.0)
                )
                for e in range(4)
            ]
            qw.append(fp8_pack4(*vals))
        # A operand of MFMA p = step / 2: lane 16 (q % 4) + t, dwords 4 (step % 2) ..
        step = q // 4
        at = ((step // 2) * 64 + (q % 4) * 16 + t) * 8 + (step % 2) * 4
        for j in range_constexpr(4):
            fx.ptr_store(qw[j], xa + (at + j))
        if q == 0:
            fx.ptr_store(scale, xsl + t)
            if t < s:
                store_dev(scratch, SSQ + slot * MAX_TOKENS + t, [sq.bitcast(fx.Int32)])
        gpu.barrier()
        stamp(a, 1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for p in range_constexpr(STEPS // 2):
            a8 = fx.Vector.from_elements(
                [fx.ptr_load(xa + ((p * 64 + lane) * 8 + e)) for e in range(8)],
                fx.Int32,
            )
            w8 = fx.Vector.from_elements(
                [wv[2 * p + i][e] for i in range(2) for e in range(4)], fx.Int32
            )
            acc = fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.vec(4, T.f32), [a8, w8, acc, 0, 0, 0, 127, 0, 127]
                )
            )
        # MFMA C: element r of lane l is row 4 (l / 16) + r, column l % 16
        for r in range_constexpr(4):
            m = 4 * (lane // 16) + r
            if m < s:
                v = acc[r] * fx.ptr_load(xsl + m)
                store_dev(
                    scratch,
                    PART + (slot * MAX_TOKENS + m) * HEAD + wave * 16 + lane % 16,
                    [v.bitcast(fx.Int32)],
                )
        cta_sync(drain=True)
        stamp(a, 2)
        if tid == 0:
            sync_release("agent")
            old = fx.llvm.atomic_add(
                _ptr(scratch, CNT + h * CNT_STRIDE, 32), fx.Int32(1), syncscope="agent"
            )
            fx.ptr_store(old, lastp)
        gpu.barrier()
        stamp(a, 3)
        if fx.ptr_load(lastp) == CHUNKS - 1:
            sync_acquire("agent")
            if tid == 0:
                store32(scratch, CNT + h * CNT_STRIDE, 0, "agent")
            finish_head(a, h, wave, lane, ws2, nw, locs, css)
            cta_sync(drain=True)
            stamp(a, 4)

    @flyc.jit
    def launch(
        x: Int64,
        res: Int64,
        res_new: Int64,
        norm_w: Int64,
        w: Int64,
        ws: Int64,
        qn: Int64,
        kn: Int64,
        iqn: Int64,
        ikn: Int64,
        pos: Int64,
        cs: Int64,
        loc: Int64,
        kp: Int64,
        ksc: Int64,
        vp: Int64,
        vsc: Int64,
        ip: Int64,
        isc: Int64,
        q_out: Int64,
        iq_out: Int64,
        scratch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kfront(
            x,
            res,
            res_new,
            norm_w,
            w,
            ws,
            qn,
            kn,
            iqn,
            ikn,
            pos,
            cs,
            loc,
            kp,
            ksc,
            vp,
            vsc,
            ip,
            isc,
            q_out,
            iq_out,
            scratch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
