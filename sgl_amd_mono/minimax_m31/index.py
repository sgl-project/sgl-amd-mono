# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 verify indexer: one launch from the index Q of a verify step's rows to
every row's top-k context blocks, over SGLang's NVFP4 index-K pool.

SGLang runs it as ``_verify_score_nvfp4_kernel`` (a row's score of a 128-token block:
the max over its causal tokens of q . k, scaled) and the radix block top-k. Here a
request's ``DRAFT`` rows are one MFMA tile: a wave scores one block at a time (K
dequantized to fp16 as SGLang's kernel does, fp16 MFMA, f32 sums), its row maxima go
to scratch, and the request's last CTA to arrive picks each row's top ``TOPK`` blocks,
a wave a row: higher score first, then the lower block id, the row's last block forced,
written in ascending block order with -1 past the row's blocks.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.ops import (
    ballot,
    clock,
    lanes_below,
    ld_i32,
    popcount,
    rsrc,
    sload64,
    xshfl,
)
from sgl_amd_mono.common.plan import THREADS, WAVES, kernel_sources, key_tuple

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
BLOCK = 128  # tokens a block (= the pool's page)
TOPK = 16
DRAFT = 8
BLOCKS = 256
MAX_BLOCKS = 16384  # SGLang's radix top-k bound: 2M tokens
LOG2E = 1.4426950408889634
SM_SCALE = HEAD**-0.5
FORCED = 1e29  # SGLang's score of a row's local block
I32_MIN = -(1 << 31)
I64_MIN = -(1 << 63)
PER_LANE = 16  # a lane's scores per pass of the top-k

# scratch dwords: every row's block scores, then a counter a request
SCORE = 0
CNT = SCORE + 2 * DRAFT * MAX_BLOCKS
# then the inputs-out-of-range words (DBG: an OR of 1 slot, 2 length; then the
# largest length seen)
# timeline stamps (i64): start, set up, scored, counter read, picked
TL_POINTS = 5
CNT_STRIDE = 32  # dwords: a counter a 128-byte line
DBG = CNT + 2 * CNT_STRIDE
TL = DBG + CNT_STRIDE
SCRATCH_BYTES = 4 * (TL + 2 * BLOCKS * TL_POINTS)


@dataclass(frozen=True)
class IndexBuild:
    tokens: int  # rows: requests x DRAFT
    pay_page: int  # the index-K pool's page strides in bytes
    sc_page: int
    seq_i64: bool = True  # the verify seq_lens dtype
    timeline: bool = False


def kernel_name(key: IndexBuild) -> str:
    return f"minimax_m31_index_s{key.tokens}" + ("_tl" if key.timeline else "")


@flyc.jit
def stamp(a, k):
    if const_expr(a["timeline"]):  # noqa: SIM102
        if fx.thread_idx.x == 0:
            bo.buffer_store(
                clock(), rsrc(a["scratch"]), TL // 2 + fx.block_idx.x * TL_POINTS + k
            )


def order_key(score):
    """An f32 as an int32 that orders like it (signed compare)."""
    b = score.bitcast(fx.Int32)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


def rank_key(score, blk):
    """A block's rank as one int64: higher score first, then the lower block id
    (signed compare; unique a block)."""
    return (fx.Int64(order_key(score)) << 32) | fx.Int64(MAX_BLOCKS - 1 - blk)


def key_block(k):
    return MAX_BLOCKS - 1 - fx.Int32(k & 0xFFFFFFFF)


def xshfl64(v, off):
    lo, hi = xshfl(fx.Int32(v), off), xshfl(fx.Int32(v >> 32), off)
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


def lane64(v, j):
    lo, hi = xshfl_lane(fx.Int32(v), j), xshfl_lane(fx.Int32(v >> 32), j)
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


def sort_desc(k, lane):
    """Bitonic sort of the wave's 64 distinct rank keys, best first."""
    s = 2
    while s <= 64:
        j = s // 2
        while j >= 1:
            p = xshfl64(k, j)
            desc = (lane & s) == 0 if s < 64 else lane >= 0
            lower = (lane & j) == 0
            # keys are distinct: take the partner's iff it belongs on this side
            take = (p > k) == (lower == desc)
            k = take.select(p, k)
            j //= 2
        s *= 2
    return k


@flyc.jit
def pick_row(a, r, m, lane, slots, nblk):
    """Row m of request r (a wave, its ``nblk`` blocks): its top TOPK blocks of the
    scores -> out. Per 1024 blocks (16 a lane): the threshold T is the worst of 16
    lane groups' bests (and at least the running top-k's last), so at least TOPK
    blocks reach it; those that do are packed into LDS ``slots`` and sorted 48 at a
    time with the running top TOPK."""
    row = r * DRAFT + m
    none = fx.Int64(I64_MIN)
    run = none  # lanes below TOPK
    for c0 in range(0, nblk, 64 * PER_LANE):
        ks = []
        for i in range_constexpr(PER_LANE):
            blk = c0 + lane + 64 * i
            ok = blk < nblk
            v = load_dev(a["scratch"], SCORE + row * MAX_BLOCKS + fx.min(blk, nblk - 1))
            ks.append(ok.select(rank_key(v.bitcast(fx.Float32), blk), none))
        best = ks[0]
        for i in range_constexpr(1, PER_LANE):
            best = fx.max(best, ks[i])
        # T: the worst of 16 groups' bests (4 lanes a group): TOPK blocks reach it
        for off in (1, 2):
            best = fx.max(best, xshfl64(best, off))
        t = best
        for off in (4, 8, 16, 32):
            t = fx.min(t, xshfl64(t, off))
        t = fx.max(t, lane64(run, TOPK - 1))
        # pack the blocks at or above T, in (i, lane) order
        base = fx.Int32(0)
        for i in range_constexpr(PER_LANE):
            keep = (ks[i] >= t) & (ks[i] != none)
            bal = ballot(keep)
            at = base + lanes_below(bal, lane)
            if keep:
                fx.ptr_store(fx.Int32(ks[i]), slots + 2 * at)
                fx.ptr_store(fx.Int32(ks[i] >> 32), slots + 2 * at + 1)
            base = base + popcount(bal)
        lds_wait()
        for b0 in range(0, base, 64 - TOPK):
            q = b0 + lane - TOPK
            inb = (lane >= TOPK) & (q < base)
            qa = 2 * fx.max(fx.min(q, base - 1), 0)
            qk = (fx.Int64(fx.ptr_load(slots + qa + 1)) << 32) | fx.Int64(
                fx.Uint32(fx.ptr_load(slots + qa))
            )
            c = (lane < TOPK).select(run, inb.select(qk, none))
            run = sort_desc(c, lane)
        lds_wait()
    win = ((lane < TOPK) & (run != none)).select(key_block(run), fx.Int32(-1))
    # ascending block order, -1 last: rank among lanes 0 .. TOPK - 1
    key = (win < 0).select(fx.Int32(MAX_BLOCKS), win)
    rank = fx.Int32(0)
    for j in range_constexpr(TOPK):
        other = xshfl_lane(key, j)
        before = (other < key) | ((other == key) & (lane > j))
        rank = rank + before.select(fx.Int32(1), fx.Int32(0))
    if lane < TOPK:
        bo.buffer_store(win, rsrc(a["out"]), row * TOPK + rank)


def lds_wait():
    """The wave's LDS accesses so far are done: its lanes' writes reach later reads
    (a wave issues LDS operations in order)."""
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.sched_barrier(0)


def note_bad(scratch, at, bit, value):
    """Record an out-of-range input: OR ``bit`` into word ``at``, max ``value``
    into word ``at + 1`` (host-side debugging reads them)."""
    fx.llvm.atomic_or(_ptr(scratch, at, 32), fx.Int32(bit), syncscope="agent")
    fx.llvm.atomic_max(_ptr(scratch, at + 1, 32), fx.Int32(value), syncscope="agent")


def xshfl_lane(v, j):
    """Lane j's v, every lane."""
    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(v), fx.Int32(j)))


def build_index(key: IndexBuild):
    s = key.tokens
    assert s % DRAFT == 0 and 1 <= s // DRAFT <= 2
    reqs = s // DRAFT
    ctas = BLOCKS // reqs  # a request's CTAs
    keyed = key_tuple(key, SOURCES)

    @fx.struct
    class Lds:
        last: fx.Array[fx.Int32, 4, 16]
        slots: fx.Array[fx.Int32, WAVES * 2 * 64 * PER_LANE, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kindex(
        iq: Int64,
        pay: Int64,
        sc: Int64,
        r2t: Int64,
        seq: Int64,
        slot: Int64,
        out: Int64,
        scratch: Int64,
        r2t_stride: Int32,
        max_pages: Int32,
        r2t_rows: Int32,
        r2t_len: Int32,
    ):
        _ = keyed
        a = {
            "seq": seq,
            "out": out,
            "scratch": scratch,
            "seq_i64": key.seq_i64,
            "timeline": key.timeline,
        }
        bid, tid = fx.block_idx.x, fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        lds = fx.SharedAllocator().allocate(Lds).peek()
        stamp(a, 0)
        r = bid % reqs
        wi = (bid // reqs) * WAVES + wave  # the wave's index among the request's
        g, n16 = lane // 16, lane % 16
        # the MFMA tile's rows m = n16 (past DRAFT zero) at dims 32 g + 8 kk ..:
        # a lane's 32 dims are contiguous, as its K payload bytes are
        qrow = r * DRAFT + fx.min(n16, DRAFT - 1)
        qv = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(iq),
                    qrow * (HEAD // 2) + 16 * g + 4 * j,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for j in range(4)
        ]
        qa = []
        for kk in range_constexpr(4):
            halves = []
            for d in range_constexpr(4):
                w = fx.Int32(qv[kk][d])
                lo, hi = (w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)
                zero = n16 >= DRAFT
                lo, hi = (
                    zero.select(fx.Float32(0.0), lo),
                    zero.select(fx.Float32(0.0), hi),
                )
                halves.append(
                    fx.Vector.from_elements([lo, hi], fx.Float32)
                    .to(fx.Float16)
                    .bitcast(fx.Int32)[0]
                )
            qa.append(fx.Vector.from_elements(halves, fx.Int32))
        # the C rows a lane holds: 4 g + i
        seqs, nbs = [], []
        for i in range_constexpr(4):
            m0 = 4 * g + i
            row = r * DRAFT + fx.min(m0, DRAFT - 1)
            if const_expr(key.seq_i64):
                sl = ld_i32(seq, 2 * row)
            else:
                sl = ld_i32(seq, row)
            sl = (m0 < DRAFT).select(sl, fx.Int32(0))
            # a length past the slot row (a padded or stale row) reads the row only
            if sl > r2t_len:
                note_bad(scratch, DBG, 2, sl)
            sl = fx.min(sl, r2t_len)
            seqs.append(sl)
            nbs.append((sl + BLOCK - 1) // BLOCK)
        nblk = fx.Int32(0)
        for i in range_constexpr(4):
            nblk = fx.max(nblk, nbs[i])
        for off in (16, 32):
            nblk = fx.max(nblk, xshfl(nblk, off))
        req_slot = fx.Int32(sload64(slot + fx.Int64(8 * r * DRAFT)))
        if (req_slot < 0) | (req_slot >= r2t_rows):
            if tid == 0:
                note_bad(scratch, DBG, 1, req_slot)
            req_slot = fx.Int32(0)
        rp, rs = rsrc(pay), rsrc(sc)
        # the request's working CTAs: a block a wave while they last
        working = fx.min(fx.Int32(ctas), (nblk + WAVES - 1) // WAVES)
        # CTAs past the request's working ones skip to the end (a return inside a
        # traced branch does not end the kernel)
        if bid // reqs < working:
            stamp(a, 1)
            for blk in range(wi, nblk, working * WAVES):
                s0 = ld_i32(r2t, req_slot * r2t_stride + blk * BLOCK)
                page = ((s0 // BLOCK) % max_pages + max_pages) % max_pages
                acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(8)]
                kv = [
                    fx.Vector(
                        bo.buffer_load(
                            rp,
                            (page * key.pay_page + (t * 16 + n16) * 64 + 16 * g) // 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    for t in range(8)
                ]
                sw = [
                    ld_i32(
                        sc,
                        (page * key.sc_page + (t * 16 + n16) * 8 + 4 * (g // 2)) // 4,
                    )
                    for t in range(8)
                ]
                for t in range_constexpr(8):
                    # scale bytes 2 g, 2 g + 1 of the token: dims 32 g .., 32 g + 16 ..
                    bits = sw[t] >> (16 * (g % 2))
                    sc0 = e4m3_f16x2(bits & 0xFF)
                    sc1 = e4m3_f16x2((bits >> 8) & 0xFF)
                    for kk in range_constexpr(4):
                        word = fx.Int32(kv[t][kk])
                        scale = sc0 if kk < 2 else sc1
                        b8 = fx.Vector.from_elements(
                            [fp4x2_f16(word, scale, j) for j in range(4)], fx.Int32
                        )
                        acc[t] = fx.Vector(
                            rocdl.mfma_f32_16x16x32_f16(
                                T.vec(4, T.f32),
                                [
                                    qa[kk].bitcast(fx.Float16),
                                    b8.bitcast(fx.Float16),
                                    acc[t],
                                ],
                            )
                        )
                # C element i of lane l: row 4 g + i, token 16 t + n16
                for i in range_constexpr(4):
                    mx = fx.Float32(float("-inf"))
                    for t in range_constexpr(8):
                        pos = blk * BLOCK + t * 16 + n16
                        v = acc[t][i]
                        mx = fx.max(
                            mx, (pos < seqs[i]).select(v, fx.Float32(float("-inf")))
                        )
                    for off in (8, 4, 2, 1):
                        mx = fx.max(mx, xshfl(mx, off))
                    mm = 4 * g + i
                    score = (blk >= nbs[i] - 1).select(
                        fx.Float32(FORCED), mx * fx.Float32(SM_SCALE * LOG2E)
                    )
                    if (n16 == 0) & (mm < DRAFT) & (blk < nbs[i]):
                        store_dev(
                            scratch,
                            SCORE + (r * DRAFT + mm) * MAX_BLOCKS + blk,
                            [score.bitcast(fx.Int32)],
                        )
            cta_sync(drain=True)
            stamp(a, 2)
            if tid == 0:
                old = fx.Int32(0)
                if working > 1:
                    sync_release("agent")
                    old = fx.llvm.atomic_add(
                        _ptr(scratch, CNT + r * CNT_STRIDE, 32),
                        fx.Int32(1),
                        syncscope="agent",
                    )
                fx.ptr_store(old, lds.last.ptr)
            gpu.barrier()
            stamp(a, 3)
            if fx.ptr_load(lds.last.ptr) == working - 1:
                sync_acquire("agent")
                if (tid == 0) & (working > 1):
                    store32(scratch, CNT + r * CNT_STRIDE, 0, "agent")
                mine = nbs[0]
                for i in range_constexpr(1, 4):
                    mine = (wave % 4 == i).select(nbs[i], mine)
                row_nblk = fx.Int32(rocdl.readlane(T.i32, mine, 16 * (wave // 4)))
                pick_row(
                    a, r, wave, lane, lds.slots.ptr + wave * 2 * 64 * PER_LANE, row_nblk
                )
                cta_sync(drain=True)
                stamp(a, 4)

    @flyc.jit
    def launch(
        iq: Int64,
        pay: Int64,
        sc: Int64,
        r2t: Int64,
        seq: Int64,
        slot: Int64,
        out: Int64,
        scratch: Int64,
        r2t_stride: Int32,
        max_pages: Int32,
        r2t_rows: Int32,
        r2t_len: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kindex(
            iq,
            pay,
            sc,
            r2t,
            seq,
            slot,
            out,
            scratch,
            r2t_stride,
            max_pages,
            r2t_rows,
            r2t_len,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def e4m3_f16x2(byte):
    """An E4M3 byte (low 8 bits) as two f16 (exact)."""
    pair = fx.Vector(
        rocdl.cvt_pk_f32_fp8(
            res=fx.Vector.make_type(2, fx.Float32), src=byte, word_sel=False
        )
    )
    return fx.Vector.from_elements([pair[0], pair[0]], fx.Float32).to(fx.Float16)


def fp4x2_f16(word, scale, j):
    """Byte j of ``word`` (two e2m1 codes, the even column low) x ``scale`` (two f16)
    -> two f16 as one int32. The converter's f32 scale applies its exponent only (an MX
    scale), so the E4M3 scale multiplies after: e2m1 x E4M3 is exact in f16."""
    pair = fx.Vector(
        rocdl.cvt_scalef32_pk_f16_fp4(
            T.vec(2, T.f16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Float32(1.0)),
            j,
        )
    )
    return (pair * scale).bitcast(fx.Int32)[0]
