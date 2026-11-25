# SPDX-License-Identifier: Apache-2.0
"""One wide-row MLA/DSA projection-front launch for ``m <= 64`` rows on gfx950.

Fuses all local DSA attention projections and norms into one single-rank launch
(zero cross-GPU communication):

    Stage 1 (K=7168, 31 tiles of bn=128, ks1=8 -> 248 CTAs):
        fused_qkv_a (2112: 1536 q_lat | 512 kv_lat | 64 k_rope),
        g_proj (1536), indexer.wk (128), indexer.weights_proj (32)
    Mid (m CTAs q_a_layernorm + m CTAs kv_a_layernorm):
        RMSNorm(q_lat) -> q_lora [m, 1536], RMSNorm(kv_lat) -> k_nope [m, 512]
    Stage 2 (K=1536, reads q_lora):
        indexer.wq_b (4096: 32 tiles) + q_b_proj (2304 DCP1: 18 tiles, ks2=4 -> 200 CTAs;
        or 18432 DCP8 q_replicate: 144 tiles, ks2=1 -> 176 CTAs)

Outputs match SGLang's ``_mono_pre`` contract: ``q``, ``latent`` (raw kv+rope),
``k_nope`` (normed), ``q_lora`` (normed), ``gate``, ``idx_q``, ``idx_k``, ``idx_w``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import (
    CM_DEV,
    KernArgs,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    rsrc,
    wave_sum,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k3.gfx950.tag import TAG_SLOTS, step_tag
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import DEPTH, RW, mma_tile

SOURCES = kernel_sources(__name__)

H = 7168
QL = 1536
KV = 512
ROPE = 64
LATW = QL + KV + ROPE  # 2112
GATE = 1536
IK = 128
IW = 32
IQ = 4096
QB_LOCAL = 12 * 192  # 2304 (DCP1)
QB_QREP = 96 * 192  # 18432 (DCP8 replicated Q)
MAX_ROWS = 64
MAX_ROWS_LOCAL = 32
MAX_ROWS_QREP = 32

BN = 128
T_Q = QL // BN  # 12
T_KV = KV // BN  # 4
T_QKV = (LATW + BN - 1) // BN  # 17 (tile 16 holds 64 k_rope cols)
T_G = GATE // BN  # 12
T_K = IK // BN  # 1
T_P = 1  # 32 cols
S1_TILES = T_QKV + T_G + T_K + T_P  # 31
S1_N = S1_TILES * BN  # 3968
KS1 = 8
KT1 = H // KS1  # 896
S1_TASKS = S1_TILES * KS1  # 248

T_IQ = IQ // BN  # 32
CNT_STRIDE = 32

MARKS = ("start", "s1", "norm", "s2")
TL_POINTS = len(MARKS)

_BUILDS: dict = {}


def tile_shape(m: int) -> tuple[int, int, int, int]:
    """(bm, bn, wm, wn) for ``m <= 64`` so every launch uses one M-tile."""
    assert 1 <= m <= MAX_ROWS, m
    if m <= 16:
        return 16, BN, 1, 8
    if m <= 32:
        return 32, BN, 2, 4
    return 64, BN, 2, 4


def stage2_spec(qb_rows: int) -> tuple[int, int, int]:
    """(n_tiles, ks2, kt2) for Stage 2 given ``qb_rows`` (2304 or 18432)."""
    assert qb_rows % BN == 0 and qb_rows > 0, qb_rows
    n_qb = qb_rows // BN
    n2 = T_IQ + n_qb
    ks2 = 4 if n2 <= 64 else 1
    assert QL % ks2 == 0 and (QL // ks2) % 64 == 0
    return n2, ks2, QL // ks2


def scratch_layout(m: int, qb_rows: int, timeline: bool = False) -> dict:
    """Byte offsets and total size of the single-rank uncached scratch buffer."""
    n2, ks2, _ = stage2_spec(qb_rows)
    sizes = [
        ("fbuf1", m * S1_N * 4),
        ("q_lat", m * QL * 2),
        ("t_done", (T_Q + T_KV) * 8),
        ("qnorm_done", m * 8),
    ]
    if ks2 > 1:
        sizes.append(("fbuf2", m * n2 * BN * 4))
    if timeline:
        sizes.append(("tl", BLOCKS * TL_POINTS * 8))
    offs, at = {}, 0
    for k, n in sizes:
        offs[k] = at
        at += (n + 255) // 256 * 256
    return {"offs": offs, "nbytes": at}


def counters(qb_rows: int) -> int:
    """int32 counter words needed for one ``mla_pre_wide`` bucket."""
    n2, ks2, _ = stage2_spec(qb_rows)
    return (S1_TILES + (n2 if ks2 > 1 else 0)) * CNT_STRIDE


@flyc.jit
def _arrive_last(tid, cnt, tile, ks, slot):
    """Drain global writes, count this K-slice, return True on the last slice."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        ptr = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            cnt + fx.Int64(tile * (4 * CNT_STRIDE)),
        )
        fx.ptr_store(fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent"), slot)
    gpu.barrier()
    return (fx.ptr_load(slot) + 1) % ks == 0


@flyc.jit
def _finish_s1_tile(
    tid,
    tile,
    m,
    fbuf1,
    q_lat,
    latent,
    gate,
    idx_k,
    idx_w,
    t_done,
    put,
):
    """Last slice of Stage 1 tile: read fbuf1, zero it, emit bf16, and flag q/kv tiles."""
    rf = rsrc(fbuf1, m * S1_N * 4)
    rq = rsrc(q_lat, m * QL * 2)
    rl = rsrc(latent, m * (KV + ROPE) * 2)
    rg = rsrc(gate, m * GATE * 2)
    rk = rsrc(idx_k, m * IK * 2)
    rp = rsrc(idx_w, m * IW * 2)
    zero2 = fx.Vector.filled(2, 0.0, fx.Float32)
    pairs = m * (BN // 2)
    col0 = tile * BN
    is_q = tile < T_Q
    is_kv = (tile >= T_Q) & (tile < T_Q + T_KV)
    is_rope = tile == T_Q + T_KV
    is_g = (tile >= T_QKV) & (tile < T_QKV + T_G)
    is_k = tile == T_QKV + T_G
    is_p = tile == T_QKV + T_G + T_K
    for idx in range(tid, pairs, THREADS):
        row = idx // (BN // 2)
        c2 = (idx % (BN // 2)) * 2
        f_off = row * S1_N + col0 + c2
        v = fx.Vector(
            bo.buffer_load(rf, f_off, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
        )
        bo.buffer_store(zero2, rf, f_off, cache_modifier=CM_DEV)
        pair = bf16_pair(v[0], v[1]).bitcast(fx.Int32)
        if is_q:
            bo.buffer_store(
                pair, rq, (row * QL + tile * BN + c2) // 2, cache_modifier=CM_DEV
            )
        if is_kv:
            bo.buffer_store(
                pair,
                rl,
                (row * (KV + ROPE) + (tile - T_Q) * BN + c2) // 2,
                cache_modifier=CM_DEV,
            )
        if is_rope & (c2 < ROPE):
            bo.buffer_store(pair, rl, (row * (KV + ROPE) + KV + c2) // 2)
        if is_g:
            bo.buffer_store(pair, rg, (row * GATE + (tile - T_QKV) * BN + c2) // 2)
        if is_k:
            bo.buffer_store(pair, rk, (row * IK + c2) // 2)
        if is_p & (c2 < IW):
            bo.buffer_store(pair, rp, (row * IW + c2) // 2)
    if is_q | is_kv:
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            put(t_done, tile, fx.Int32(1))


@flyc.jit
def _rmsnorm_row(tid, row, cols, src, src_ld, nw, dst, dst_ld, eps, red, dst_cm):
    """RMSNorm one bf16 row of length ``cols`` (multiple of 8, <= 4096) in one CTA."""
    nvec = cols // 8
    r_src = rsrc(src)
    r_nw = rsrc(nw, cols * 2)
    r_dst = rsrc(dst)
    e = fx.min(tid, nvec - 1)
    v = fx.Vector(
        bo.buffer_load(
            r_src,
            (row * src_ld + e * 8) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    w = fx.Vector(bo.buffer_load(r_nw, (e * 8) // 2, vec_width=4, dtype=T.i32))
    if tid < nvec:
        ss = fx.Float32(0.0)
        for d in range_constexpr(4):
            wd = fx.Int32(v[d])
            x0, x1 = bf_lo(wd), bf_hi(wd)
            ss = ss + x0 * x0 + x1 * x1
        fx.ptr_store(ss, red + tid)
    gpu.barrier()
    # reduce nvec floats in wave 0 via wave_sum (64-lane butterfly with permlane32_swap)
    if tid < 64:
        acc = fx.Float32(0.0)
        for j in range_constexpr((nvec + 63) // 64):
            idx = tid + 64 * j
            if idx < nvec:
                acc = acc + fx.ptr_load(red + idx)
        acc = wave_sum(acc)
        if tid == 0:
            fx.ptr_store(hw_rsq(acc / fx.Float32(cols) + fx.Float32(eps)), red)
    gpu.barrier()
    if tid < nvec:
        rs = fx.ptr_load(red)
        out_w = []
        for d in range_constexpr(4):
            vd, wd = fx.Int32(v[d]), fx.Int32(w[d])
            y0 = bf16_round((bf_lo(vd) * rs) * bf_lo(wd))
            y1 = bf16_round((bf_hi(vd) * rs) * bf_hi(wd))
            out_w.append(bf16_pair(y0, y1).bitcast(fx.Int32))
        bo.buffer_store(
            fx.Vector.from_elements(out_w, fx.Int32),
            r_dst,
            (row * dst_ld + tid * 8) // 2,
            cache_modifier=dst_cm,
        )


def _bf16_emit_dyn(out, ldo, n0, m, n):
    """bf16_emit supporting dynamic ``ldo``, ``n0``, ``n`` (row-major [m, ldo])."""
    r_o = rsrc(out, fx.Int64(m) * fx.Int64(ldo) * fx.Int64(2))

    def emit(row, col, v, live):
        words = fx.Vector.from_elements(
            [
                bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                bf16_pair(v[2], v[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        bo.buffer_store(words, r_o, (row * ldo + n0 + col) // 2, mask=ok)

    return emit


@flyc.jit
def _finish_s2_tile(tid, tile, m, n2, qb_rows, fbuf2, idx_q, q):
    """Last slice of Stage 2 tile (ks2 > 1): read fbuf2, zero it, emit bf16."""
    s2_n = n2 * BN
    rf = rsrc(fbuf2, m * s2_n * 4)
    ri = rsrc(idx_q, m * IQ * 2)
    rq = rsrc(q, m * qb_rows * 2)
    zero2 = fx.Vector.filled(2, 0.0, fx.Float32)
    pairs = m * (BN // 2)
    col0 = tile * BN
    is_iq = tile < T_IQ
    for idx in range(tid, pairs, THREADS):
        row = idx // (BN // 2)
        c2 = (idx % (BN // 2)) * 2
        f_off = row * s2_n + col0 + c2
        v = fx.Vector(
            bo.buffer_load(rf, f_off, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
        )
        bo.buffer_store(zero2, rf, f_off, cache_modifier=CM_DEV)
        pair = bf16_pair(v[0], v[1]).bitcast(fx.Int32)
        if is_iq:
            bo.buffer_store(pair, ri, (row * IQ + tile * BN + c2) // 2)
        else:
            bo.buffer_store(pair, rq, (row * qb_rows + (tile - T_IQ) * BN + c2) // 2)


def mla_pre_wide(
    m: int,
    qb_rows: int = QB_LOCAL,
    launches: int = 1,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    timeline: bool = False,
):
    """Build the wide MLA projection-front launch; returns ``(launch, layout)``.

    ``launches`` must be in ``1..TAG_SLOTS`` (256), with each launch invoked at
    a distinct ``slot_i`` in ``0..launches-1`` per epoch (see ``step_tag``).
    """
    assert 1 <= launches <= TAG_SLOTS, (launches, TAG_SLOTS)
    bm, bn, wm, wn = tile_shape(m)
    n2, ks2, kt2 = stage2_spec(qb_rows)
    n2_tasks = n2 * ks2
    lay = scratch_layout(m, qb_rows, timeline)
    soffs = lay["offs"]
    keyed = key_tuple(
        {
            "m": m,
            "qb_rows": qb_rows,
            "launches": launches,
            "q_eps": q_eps,
            "kv_eps": kv_eps,
            "timeline": timeline,
        },
        SOURCES,
    )
    key = (m, qb_rows, launches, q_eps, kv_eps, timeline)
    if key in _BUILDS:
        return _BUILDS[key]
    cwords = bm * (bn + 4)

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bm * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bn * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, cwords, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    @fx.struct
    class Small:
        slot: fx.Array[fx.Int32, 4, 16]
        red: fx.Array[fx.Float32, QL // 8, 16]

    @fx.struct
    class Tls:
        v: fx.Array[fx.Int64, TL_POINTS, 16]

    name = f"k31_mla_pre_wide_m{m}_qb{qb_rows}_l{launches}" + (
        "_tl" if timeline else ""
    )

    def body(
        x: Int64,
        w_qkv: Int64,
        w_g: Int64,
        w_k: Int64,
        w_p: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        w_iq: Int64,
        w_qb: Int64,
        q: Int64,
        latent: Int64,
        k_nope: Int64,
        q_lora: Int64,
        gate: Int64,
        idx_q: Int64,
        idx_k: Int64,
        idx_w: Int64,
        cnt: Int64,
        scratch: Int64,
        epoch: Int64,
        slot_i: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator()
        st = alloc.allocate(Smem)
        small = alloc.allocate(Small).peek()
        slot, red = small.slot.ptr, small.red.ptr
        tls = alloc.allocate(Tls).peek().v.ptr if timeline else None
        stamp_begin(timeline, tls, tid, TL_POINTS)
        lds, cl = st.ab.peek(), st.c.peek().c.ptr
        with guard.scope(guard.Guard(epoch)):
            tag = step_tag(epoch, slot_i)
            mb = Mailbox(tag)
            put, poll = mb.put, mb.poll

            def sp(a, region):
                return a.scratch + fx.Int64(soffs[region])

            def sr(a, region):
                return sreg(a.scratch, soffs[region], region)

            # ---- Stage 1: fused_qkv_a | g_proj | wk | weights_proj (K=7168, 248 CTAs)
            for task in range(bid, S1_TASKS, BLOCKS):
                tile = task // KS1
                sk = task % KS1
                a1 = ka(
                    "x",
                    "w_qkv",
                    "w_g",
                    "w_k",
                    "w_p",
                    "latent",
                    "gate",
                    "idx_k",
                    "idx_w",
                    "cnt",
                    "scratch",
                )
                w_ptr = (tile < T_QKV).select(
                    a1.w_qkv,
                    (tile < T_QKV + T_G).select(
                        a1.w_g, (tile == T_QKV + T_G).select(a1.w_k, a1.w_p)
                    ),
                )
                n_mat = (tile < T_QKV).select(
                    fx.Int32(LATW),
                    (tile < T_QKV + T_G).select(
                        fx.Int32(GATE),
                        (tile == T_QKV + T_G).select(fx.Int32(IK), fx.Int32(IW)),
                    ),
                )
                n0_mat = (tile < T_QKV).select(
                    tile * BN,
                    (tile < T_QKV + T_G).select((tile - T_QKV) * BN, fx.Int32(0)),
                )
                fbuf1 = sp(a1, "fbuf1")
                mma_tile(
                    tid,
                    a1.x,
                    H,
                    m,
                    0,
                    w_ptr,
                    H,
                    n_mat,
                    n0_mat,
                    sk * KT1,
                    KT1,
                    bm,
                    bn,
                    wm,
                    wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    None,
                    DEPTH,
                    atomic_lin(fbuf1, S1_N, tile * BN, m, S1_N, False),
                    cl,
                )
                if _arrive_last(tid, a1.cnt, tile, KS1, slot):
                    _finish_s1_tile(
                        tid,
                        tile,
                        m,
                        fbuf1,
                        sp(a1, "q_lat"),
                        a1.latent,
                        a1.gate,
                        a1.idx_k,
                        a1.idx_w,
                        sr(a1, "t_done"),
                        put,
                    )
            stamp(timeline, tls, tid, 1)

            # ---- Mid: q_a_layernorm (CTAs 0..m-1) and kv_a_layernorm (last m CTAs)
            for row in range(bid, m, BLOCKS):
                an = ka("q_nw", "q_lora", "scratch")
                t_done = sr(an, "t_done")
                if tid < T_Q:
                    poll([(t_done, tid, 1)])
                gpu.barrier()
                _rmsnorm_row(
                    tid,
                    row,
                    QL,
                    sp(an, "q_lat"),
                    QL,
                    an.q_nw,
                    an.q_lora,
                    QL,
                    q_eps,
                    red,
                    CM_DEV,
                )
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tid == 0:
                    put(sr(an, "qnorm_done"), row, fx.Int32(1))
            for row in range(first_task(bid, max(BLOCKS - m, 0)), m, BLOCKS):
                ak = ka("kv_nw", "latent", "k_nope", "scratch")
                t_done = sr(ak, "t_done")
                if tid < T_KV:
                    poll([(t_done, T_Q + tid, 1)])
                gpu.barrier()
                _rmsnorm_row(
                    tid,
                    row,
                    KV,
                    ak.latent,
                    KV + ROPE,
                    ak.kv_nw,
                    ak.k_nope,
                    KV,
                    kv_eps,
                    red,
                    0,
                )
            stamp(timeline, tls, tid, 2)

            # ---- Stage 2: indexer.wq_b | q_b_proj (K=1536, reads q_lora at CM_DEV)
            if bid < n2_tasks:
                aw = ka("scratch")
                qnorm_done = sr(aw, "qnorm_done")
                if tid < m:
                    poll([(qnorm_done, tid, 1)])
                gpu.barrier()
            for task in range(bid, n2_tasks, BLOCKS):
                tile = task // ks2
                sk = task % ks2
                a2 = ka("q_lora", "w_iq", "w_qb", "idx_q", "q", "cnt", "scratch")
                is_iq = tile < T_IQ
                w_ptr = is_iq.select(a2.w_iq, a2.w_qb)
                n_mat = is_iq.select(fx.Int32(IQ), fx.Int32(qb_rows))
                n0_mat = is_iq.select(tile * BN, (tile - T_IQ) * BN)
                if const_expr(ks2 == 1):
                    out_ptr = is_iq.select(a2.idx_q, a2.q)
                    mma_tile(
                        tid,
                        a2.q_lora,
                        QL,
                        m,
                        0,
                        w_ptr,
                        QL,
                        n_mat,
                        n0_mat,
                        0,
                        QL,
                        bm,
                        bn,
                        wm,
                        wn,
                        lds.xs.ptr,
                        lds.ws.ptr,
                        _bf16_emit_dyn(out_ptr, n_mat, n0_mat, m, n_mat),
                        DEPTH,
                        None,
                        cl,
                        CM_DEV,
                    )
                else:
                    fbuf2 = sp(a2, "fbuf2")
                    s2_n = n2 * BN
                    mma_tile(
                        tid,
                        a2.q_lora,
                        QL,
                        m,
                        0,
                        w_ptr,
                        QL,
                        n_mat,
                        n0_mat,
                        sk * kt2,
                        kt2,
                        bm,
                        bn,
                        wm,
                        wn,
                        lds.xs.ptr,
                        lds.ws.ptr,
                        None,
                        DEPTH,
                        atomic_lin(fbuf2, s2_n, tile * BN, m, s2_n, False),
                        cl,
                        CM_DEV,
                    )
                    if _arrive_last(tid, a2.cnt, S1_TILES + tile, ks2, slot):
                        _finish_s2_tile(
                            tid, tile, m, n2, qb_rows, fbuf2, a2.idx_q, a2.q
                        )
            stamp(timeline, tls, tid, 3)
            if const_expr(timeline):
                at = ka("scratch")
                stamp_flush(timeline, tls, sp(at, "tl"), tid, bid, TL_POINTS)

    ka = KernArgs(
        [
            (n, 8 if t is Int64 else 4)
            for n, t in body.__annotations__.items()
            if n != "return"
        ]
    )
    kernel = flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])(body)

    @flyc.jit
    def launch(
        x: Int64,
        w_qkv: Int64,
        w_g: Int64,
        w_k: Int64,
        w_p: Int64,
        q_nw: Int64,
        kv_nw: Int64,
        w_iq: Int64,
        w_qb: Int64,
        q: Int64,
        latent: Int64,
        k_nope: Int64,
        q_lora: Int64,
        gate: Int64,
        idx_q: Int64,
        idx_k: Int64,
        idx_w: Int64,
        cnt: Int64,
        scratch: Int64,
        epoch: Int64,
        slot_i: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        kernel(
            x,
            w_qkv,
            w_g,
            w_k,
            w_p,
            q_nw,
            kv_nw,
            w_iq,
            w_qb,
            q,
            latent,
            k_nope,
            q_lora,
            gate,
            idx_q,
            idx_k,
            idx_w,
            cnt,
            scratch,
            epoch,
            slot_i,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch, lay)
    return launch, lay
