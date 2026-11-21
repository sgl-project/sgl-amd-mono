# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 verify sparse attention: one launch from a verify step's Q and every
row's top-k context blocks to the attention output, over SGLang's NVFP4 K / V pool.

SGLang runs it as ``_nvfp4_sparse_decode_kernel`` (a program a row's top-k chunk, an
online softmax over its blocks, K and V dequantized to fp16) and
``_merge_topk_attn_out_kernel``. Here a CTA of 4 waves attends one row's one block, a
wave 32 tokens with the 16 Q heads as the MFMA tile: S^T = K Q^T puts a token's 4 rows
and a head on a lane, which are the P^T operand of O^T = V^T P^T as they are (the PV
K index 8 g + j is token 4 g + j of the first 16, then of the second). The waves'
softmax states merge in LDS, and the row's last CTA to arrive merges its blocks' states
into the output, rounded to bf16 once.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import (
    bf16_pair,
    clock,
    hw_exp2,
    hw_rcp,
    ld_i32,
    rsrc,
    sload64,
    xshfl,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple

from .index import e4m3_f16x2, fp4x2_f16, note_bad
from .sync import (
    _ptr,
    cta_sync,
    load_dev,
    store32,
    store128,
    store_dev,
    sync_acquire,
    sync_release,
)

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

HEADS = 16
HEAD = 128
BLOCK = 128
TOPK = 16
WAVES = 4
THREADS = 64 * WAVES
TOK_W = BLOCK // WAVES  # tokens a wave
LOG2E = 1.4426950408889634
SM_SCALE = HEAD**-0.5
NEG_INF = float("-inf")

# scratch dwords: a block's state (M, L of each head, then O = ACC / L [head][dim] in
# bf16, as SGLang's partials) for every row's top-k entry, then a counter a row
STATE = 2 * HEADS + HEADS * HEAD // 2
PART = 0
CNT = PART + 16 * TOPK * STATE
# timeline stamps (i64): start, K / V loads out, state in LDS, state stored, counter
# read, row merged
TL_POINTS = 6
CNT_STRIDE = 32  # dwords: a counter a 128-byte line, so the rows' arrivals do not queue
# inputs-out-of-range words (an OR of 1 slot, 2 length, 4 block; the largest value)
DBG = CNT + 16 * CNT_STRIDE
TL = DBG + CNT_STRIDE
SCRATCH_BYTES = 4 * (TL + 2 * 16 * TOPK * TL_POINTS)


@dataclass(frozen=True)
class SparseBuild:
    tokens: int  # rows
    pay_page: int  # main K / V pool strides in bytes: payload page, row
    pay_row: int
    sc_page: int
    sc_row: int
    seq_i64: bool = True
    timeline: bool = False


def kernel_name(key: SparseBuild) -> str:
    return f"minimax_m31_sparse_s{key.tokens}" + ("_tl" if key.timeline else "")


@flyc.jit
def stamp(a, k):
    if const_expr(a["timeline"]):  # noqa: SIM102
        if fx.thread_idx.x == 0:
            bo.buffer_store(
                clock(), rsrc(a["scratch"]), TL // 2 + fx.block_idx.x * TL_POINTS + k
            )


def bf16x8_f16(w4):
    """Four dwords of 8 bf16 -> four dwords of the same 8 values as f16."""
    out = []
    for d in range_constexpr(4):
        w = fx.Int32(w4[d])
        lo, hi = (w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)
        out.append(
            fx.Vector.from_elements([lo, hi], fx.Float32)
            .to(fx.Float16)
            .bitcast(fx.Int32)[0]
        )
    return fx.Vector.from_elements(out, fx.Int32)


def mfma16(a, b, c):
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_f16(
            T.vec(4, T.f32), [a.bitcast(fx.Float16), b.bitcast(fx.Float16), c]
        )
    )


def f16x2(lo, hi):
    return (
        fx.Vector.from_elements([lo, hi], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
    )


@flyc.jit
def merge_row(a, row, tid):
    """The row's last CTA: its blocks' states -> out (bf16). A thread 8 of the
    row's 2048 outputs: head tid / 16, dims 8 (tid % 16) .."""
    h, d0 = tid // 16, 8 * (tid % 16)
    base = PART + row * TOPK * STATE
    mls = [load_dev(a["scratch"], base + j * STATE + 2 * h, 2) for j in range(TOPK)]
    os_ = [
        load_dev(a["scratch"], base + j * STATE + 2 * HEADS + (h * HEAD + d0) // 2, 4)
        for j in range(TOPK)
    ]
    ms = [fx.Int32(mls[j][0]).bitcast(fx.Float32) for j in range(TOPK)]
    ls = [fx.Int32(mls[j][1]).bitcast(fx.Float32) for j in range(TOPK)]
    mx = ms[0]
    for j in range_constexpr(1, TOPK):
        mx = fx.max(mx, ms[j])
    lsum = fx.Float32(0.0)
    o = [fx.Float32(0.0) for _ in range(8)]
    for j in range_constexpr(TOPK):
        # an empty block's L is 0: its weight is 0 (its M is -inf)
        f = (ls[j] > 0.0).select(ls[j] * hw_exp2(ms[j] - mx), fx.Float32(0.0))
        lsum = lsum + f
        for q in range_constexpr(4):
            w = fx.Int32(os_[j][q])
            o[2 * q] = o[2 * q] + (w << 16).bitcast(fx.Float32) * f
            o[2 * q + 1] = o[2 * q + 1] + (w & -65536).bitcast(fx.Float32) * f
    inv = (lsum > 0.0).select(hw_rcp(lsum), fx.Float32(0.0))
    words = [
        bf16_pair(o[2 * k] * inv, o[2 * k + 1] * inv).bitcast(fx.Int32)
        for k in range(4)
    ]
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(a["out"]),
        (row * HEADS * HEAD + h * HEAD + d0) // 2,
    )


def build_sparse(key: SparseBuild):
    s = key.tokens
    assert 1 <= s <= 16
    keyed = key_tuple(key, SOURCES)

    @fx.struct
    class Lds:
        ml: fx.Array[fx.Float32, WAVES * 2 * HEADS, 16]
        acc: fx.Array[fx.Float32, WAVES * HEADS * HEAD, 16]
        last: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def ksparse(
        q: Int64,
        kp: Int64,
        ksc: Int64,
        vp: Int64,
        vsc: Int64,
        r2t: Int64,
        seq: Int64,
        slot: Int64,
        topk: Int64,
        out: Int64,
        scratch: Int64,
        r2t_stride: Int32,
        max_pages: Int32,
        r2t_rows: Int32,
        r2t_len: Int32,
    ):
        _ = keyed
        a = {"out": out, "scratch": scratch, "timeline": key.timeline}
        bid, tid = fx.block_idx.x, fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        g, n16 = lane // 16, lane % 16
        row, j = bid // TOPK, bid % TOPK
        lds = fx.SharedAllocator().allocate(Lds).peek()
        stamp(a, 0)
        blk = fx.Int32(rocdl.readfirstlane(T.i32, ld_i32(topk, row * TOPK + j)))
        # Q^T, the S^T MFMAs' B: head n16 at dims 32 g + 8 kk .. (a lane's 32
        # contiguous dims, as its K payload bytes are)
        qw = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(q),
                    (row * HEADS + n16) * (HEAD // 2) + 16 * g + 4 * kk,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for kk in range(4)
        ]
        if const_expr(key.seq_i64):
            seq_len = fx.Int32(sload64(seq + fx.Int64(8 * row)))
        else:
            seq_len = fx.Int32(rocdl.readfirstlane(T.i32, ld_i32(seq, row)))
        req = fx.Int32(sload64(slot + fx.Int64(8 * row)))
        # inputs past the slot table (a padded or stale row) read inside it only
        bad_req = (req < 0) | (req >= r2t_rows)
        bad_len = seq_len > r2t_len
        seq_len = fx.min(seq_len, r2t_len)
        bad_blk = blk * BLOCK >= seq_len
        if (tid == 0) & (bad_req | bad_len | bad_blk):
            bits = bad_req.select(fx.Int32(1), fx.Int32(0)) | bad_len.select(
                fx.Int32(2), fx.Int32(0)
            )
            bits = bits | bad_blk.select(fx.Int32(4), fx.Int32(0))
            note_bad(scratch, DBG, bits, fx.max(blk, seq_len))
        req = bad_req.select(fx.Int32(0), req)
        valid = (blk >= 0) & (~bad_blk)
        bsafe = valid.select(blk, fx.Int32(0))
        s0 = fx.Int32(
            rocdl.readfirstlane(T.i32, ld_i32(r2t, req * r2t_stride + bsafe * BLOCK))
        )
        page = ((s0 // BLOCK) % max_pages + max_pages) % max_pages
        nvalid = valid.select(seq_len - bsafe * BLOCK, fx.Int32(0))
        t0 = wave * TOK_W
        rk, rks, rv, rvs = rsrc(kp), rsrc(ksc), rsrc(vp), rsrc(vsc)
        # K: S^T's A, token t0 + 16 u + n16 at dims 32 g .. (16 payload bytes)
        kw = [
            fx.Vector(
                bo.buffer_load(
                    rk,
                    (page * key.pay_page + (t0 + 16 * u + n16) * key.pay_row + 16 * g)
                    // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for u in range(2)
        ]
        ks = [
            ld_i32(
                ksc,
                (page * key.sc_page + (t0 + 16 * u + n16) * key.sc_row + 4 * (g // 2))
                // 4,
            )
            for u in range(2)
        ]
        # V: O^T's A, dims 8 n16 .. (payload dword n16) of the tokens of PV K index
        # 8 g + jj: t0 + 4 g + jj, then t0 + 16 + 4 g + jj - 4
        vtok = [
            t0 + 4 * g + jj if jj < 4 else t0 + 16 + 4 * g + jj - 4 for jj in range(8)
        ]
        vw = [
            ld_i32(vp, (page * key.pay_page + tk * key.pay_row) // 4 + n16)
            for tk in vtok
        ]
        vs = [
            ld_i32(vsc, (page * key.sc_page + tk * key.sc_row) // 4 + n16 // 8)
            for tk in vtok
        ]
        stamp(a, 1)
        qb = [bf16x8_f16(qw[kk]) for kk in range(4)]
        sacc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        for u in range_constexpr(2):
            bits = ks[u] >> (16 * (g % 2))
            sc0, sc1 = e4m3_f16x2(bits & 0xFF), e4m3_f16x2((bits >> 8) & 0xFF)
            for kk in range_constexpr(4):
                word = fx.Int32(kw[u][kk])
                scl = sc0 if kk < 2 else sc1
                ka = fx.Vector.from_elements(
                    [fp4x2_f16(word, scl, jb) for jb in range(4)], fx.Int32
                )
                sacc[u] = mfma16(ka, qb[kk], sacc[u])
        # S^T element i of tile u: token t0 + 16 u + 4 g + i, head n16
        qk = fx.Float32(SM_SCALE * LOG2E)
        sv = []
        for u in range_constexpr(2):
            for i in range_constexpr(4):
                tok = t0 + 16 * u + 4 * g + i
                sv.append((tok < nvalid).select(sacc[u][i] * qk, fx.Float32(NEG_INF)))
        m = sv[0]
        for k in range_constexpr(1, 8):
            m = fx.max(m, sv[k])
        for off in (16, 32):
            m = fx.max(m, xshfl(m, off))
        ps = [(m > NEG_INF).select(hw_exp2(v - m), fx.Float32(0.0)) for v in sv]
        lsum = ps[0]
        for k in range_constexpr(1, 8):
            lsum = lsum + ps[k]
        for off in (16, 32):
            lsum = lsum + xshfl(lsum, off)
        # P^T: K index 8 g + jj is S^T tile jj // 4, element jj % 4
        pb = fx.Vector.from_elements(
            [f16x2(ps[2 * k], ps[2 * k + 1]) for k in range(4)], fx.Int32
        )
        oacc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(8)]
        vscl = []
        for jj in range_constexpr(8):
            # a token past the row's length reads scale 0: its codes add 0 to P V
            byte = (vs[jj] >> (8 * ((n16 // 2) % 4))) & 0xFF
            vscl.append(e4m3_f16x2((vtok[jj] < nvalid).select(byte, fx.Int32(0))))
        for e in range_constexpr(8):
            pairs = []
            for k in range_constexpr(4):
                w0, w1 = vw[2 * k], vw[2 * k + 1]
                nib = ((w0 >> (4 * e)) & 0xF) | (((w1 >> (4 * e)) & 0xF) << 4)
                sc = fx.Vector.from_elements(
                    [vscl[2 * k][0], vscl[2 * k + 1][0]], fx.Float16
                )
                pairs.append(fp4x2_f16(nib, sc, 0))
            va = fx.Vector.from_elements(pairs, fx.Int32)
            oacc[e] = mfma16(va, pb, oacc[e])
        # the wave's state -> LDS: M, L of head n16 (lanes 0 .. 15), ACC element
        # i of tile e is dim 8 (4 g + i) + e of head n16
        if g == 0:
            fx.ptr_store(m, lds.ml.ptr + (wave * 2 * HEADS + n16))
            fx.ptr_store(lsum, lds.ml.ptr + (wave * 2 * HEADS + HEADS + n16))
        for i in range_constexpr(4):
            at = (wave * HEADS + n16) * HEAD + 8 * (4 * g + i)
            for e in range_constexpr(8):
                fx.ptr_store(oacc[e][i], lds.acc.ptr + (at + e))
        gpu.barrier()
        stamp(a, 2)
        # the block's state: the waves' merged, a thread 8 dims of a head
        h, d0 = tid // 16, 8 * (tid % 16)
        mw = [fx.ptr_load(lds.ml.ptr + (w * 2 * HEADS + h)) for w in range(WAVES)]
        lw = [
            fx.ptr_load(lds.ml.ptr + (w * 2 * HEADS + HEADS + h)) for w in range(WAVES)
        ]
        mb = mw[0]
        for w in range_constexpr(1, WAVES):
            mb = fx.max(mb, mw[w])
        fw = [
            (lw[w] > 0.0).select(hw_exp2(mw[w] - mb), fx.Float32(0.0))
            for w in range(WAVES)
        ]
        lb = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            lb = lb + lw[w] * fw[w]
        base = PART + (row * TOPK + j) * STATE
        inv = (lb > 0.0).select(hw_rcp(lb), fx.Float32(0.0))
        vals = []
        for dd in range_constexpr(8):
            v = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                v = (
                    v
                    + fx.ptr_load(lds.acc.ptr + ((w * HEADS + h) * HEAD + d0 + dd))
                    * fw[w]
                )
            vals.append(v * inv)
        words = [
            bf16_pair(vals[2 * k], vals[2 * k + 1]).bitcast(fx.Int32) for k in range(4)
        ]
        store128(
            scratch,
            base + 2 * HEADS + (h * HEAD + d0) // 2,
            fx.Vector.from_elements(words, fx.Int32),
            "agent",
        )
        if tid % 16 == 0:
            store_dev(
                scratch, base + 2 * h, [mb.bitcast(fx.Int32), lb.bitcast(fx.Int32)]
            )
        cta_sync(drain=True)
        stamp(a, 3)
        if tid == 0:
            sync_release("agent")
            old = fx.llvm.atomic_add(
                _ptr(scratch, CNT + row * CNT_STRIDE, 32),
                fx.Int32(1),
                syncscope="agent",
            )
            fx.ptr_store(old, lds.last.ptr)
        gpu.barrier()
        stamp(a, 4)
        if fx.ptr_load(lds.last.ptr) == TOPK - 1:
            sync_acquire("agent")
            if tid == 0:
                store32(scratch, CNT + row * CNT_STRIDE, 0, "agent")
            merge_row(a, row, tid)
            cta_sync(drain=True)
            stamp(a, 5)

    @flyc.jit
    def launch(
        q: Int64,
        kp: Int64,
        ksc: Int64,
        vp: Int64,
        vsc: Int64,
        r2t: Int64,
        seq: Int64,
        slot: Int64,
        topk: Int64,
        out: Int64,
        scratch: Int64,
        r2t_stride: Int32,
        max_pages: Int32,
        r2t_rows: Int32,
        r2t_len: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        ksparse(
            q,
            kp,
            ksc,
            vp,
            vsc,
            r2t,
            seq,
            slot,
            topk,
            out,
            scratch,
            r2t_stride,
            max_pages,
            r2t_rows,
            r2t_len,
        ).launch(grid=(s * TOPK,), block=(THREADS,), stream=stream)

    return launch
