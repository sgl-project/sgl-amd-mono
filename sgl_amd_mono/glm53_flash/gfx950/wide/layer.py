# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash wide layer launch (milestone 1): one persistent launch (BLOCKS
CTAs, one task loop a stage, no grid barrier) a KDA layer on each of ``tp`` ranks:

    (``t1``: the previous layer's FFN exit first)
    tpush   this rank's MoE partial, band by band, to the column owners (RS)
    treduce the owner's pieces summed in rank order, stored on every rank (AG)
    aseam   (with t1) R_in = post_p moe + comb_p R_mid_p (hc_post) as it reads R
    aseam   attention mHC slices of the replicated residual R_in (and zero ``proj``)
    agate   a row's mixes -> pre / post / comb, the attention input x (RMSNorm)
    inproj  proj = x @ W_in^T (this rank's heads): direct-to-LDS tile GEMM, split-K adds
    kda     (request, head): f_b / g_b, conv, ring verify, gated RMSNorm -> core
    oproj   this rank's partial core @ W_o^T, K slices merged in a local tile, each
            slice's band of rows pushed to the column owner (RS)
    reduce  the owner sums its pieces in rank order and stores them on every rank (AG);
            with ``oneshot`` none: oproj pushes to every rank, fseam sums the sources
    fseam   FFN mHC slices: R_mid = post o + comb R (hc_post) and their mixes
    fgate   a row's FFN mixes -> post_f / comb_f, the FFN input x_f
    router  split-K f32 partials of x_f @ W_r^T (``route``)
    route   a row: logits [rows, 288] f32, sigmoid, top-8 + the shared slot -> ids /
            weights [rows, 9]

``stop`` ends the launch after a stage (``STOPS``). Without ``body`` (a DSA layer's
seams) the launch runs [t1] aseam agate, then (``stop="route"``) router route [sort]
on agate's x: before the attention, the previous MoE's exit and the attention input
(``stop="agate"``); after it, the attention's exit and the FFN input with its route. Hand-offs inside a rank: tagged
mailbox pairs (tag = epoch + 1) for the slice partials, flag words (:= epoch + 1) for
x rows, KDA heads and x_f rows, monotonic counters on their own lines for the GEMM
tiles; across ranks: Kimi's flag format in the parity halves of the peer buffer.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.epoch import EPOCH_MARKS, epoch_begin, epoch_end
from sgl_amd_mono.common.ops import CM_DEV, CM_SYS, rsrc, uniform
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN, N_EXPERTS
from sgl_amd_mono.glm53_flash.gfx950.wide import coll, kda, mhc, route, sort, tile_dma
from sgl_amd_mono.glm53_flash.gfx950.wide.m1a import (
    CNT_STRIDE,
    TRACE_POINTS,
    Gemm,
    _count,
    _flag,
    _head_tiles,
    _stamp,
    _wait_words,
    inproj_cfg,
)
from sgl_amd_mono.kimi_k31.gfx950.stages import coll as kcoll
from sgl_amd_mono.kimi_k31.gfx950.stages.gemm_coll import _arrive_all, _push_band_flag
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin

SOURCES = kernel_sources(__name__)

STOPS = ("kda", "reduce", "fgate", "route")
SEAM_STOPS = ("agate", "route")

# o_proj [rows, 2048] x [4096, 2048]^T, the fastest direct-to-LDS cfg a width (dma_bench)
OPROJ = {
    32: Gemm(32, 64, 2, 4, 4, 64, 6),
    64: Gemm(64, 64, 2, 4, 4, 64, 6),
    128: Gemm(128, 64, 4, 2, 4, 64, 4),
    144: Gemm(96, 64, 2, 4, 2, 64, 4),
    192: Gemm(128, 64, 4, 2, 2, 64, 4),
}


def oproj_cfg(rows):
    return OPROJ[min(w for w in OPROJ if w >= rows)]


def scratch_layout(rows, tt, trace=False):
    """Word offsets in the (local) scratch buffer."""
    g, go, gr = inproj_cfg(rows), oproj_cfg(rows), route.rgate_cfg(rows)
    offs, w = {}, 0

    def take(name, words, align=32):
        nonlocal w
        w = -(-w // align) * align
        offs[name] = w
        w += words

    take("xflag", rows)
    take("cnt_p", CNT_STRIDE * -(-kda.NPROJ // g.bn))
    take("kflag", (rows // tt) * kda.NH)
    take("cnt_o", CNT_STRIDE * (HIDDEN // go.bn) * -(-rows // go.bm))
    take("xfflag", rows)
    take("pmix_a", 2 * mhc.pmix_words(rows))
    take("pmix_f", 2 * mhc.pmix_words(rows))
    take("cnt_r", CNT_STRIDE * route.n_tiles(gr) * -(-rows // gr.bm))
    take("parts", gr.ks * rows * N_EXPERTS)
    take("sflag", rows)
    if trace:
        take("trace", BLOCKS * TRACE_POINTS * 2)
    offs["end"] = w
    return offs


@flyc.jit
def _poll_peer(c, n, region, idx):
    """Threads below ``n`` poll their own flag ``idx`` of peer region ``region``."""
    if c["tid"] < n:
        c["poll"]([(c[region], idx, 1)])
    gpu.barrier()


def _poll_ag(c, task, rows, br, rb, bn, sl, n_pieces, nbw):
    """Before slice task ``task``: the AG flags of the pieces its rows and columns touch."""
    c0 = (task % mhc.SLICES) * mhc.COLS
    tb = (task // mhc.SLICES) * br
    last = (fx.min(tb + br, rows) - 1) // rb
    band = fx.min(tb // rb + c["tid"], last)
    idx = (c0 // sl) * n_pieces + band * (sl // bn) + (c0 % sl) // bn
    _poll_peer(c, fx.min(nbw, last - tb // rb + 1), "agf_own", idx)


_BUILDS: dict = {}


def build(
    rows,
    tt,
    tp,
    lower_bound,
    onorm_eps,
    state_bf16,
    has_iw,
    stop="route",
    trace=False,
    shared_w=1.0,
    oneshot=False,
    t1=False,
    moe_sort=False,
    body=True,
):
    keyed = key_tuple(locals(), SOURCES)
    key = (
        rows,
        tt,
        tp,
        lower_bound,
        onorm_eps,
        state_bf16,
        has_iw,
        stop,
        trace,
        shared_w,
        oneshot,
        t1,
        moe_sort,
        body,
    )
    if key in _BUILDS:
        return _BUILDS[key]
    assert rows % tt == 0 and stop in (STOPS if body else SEAM_STOPS)
    level = STOPS.index(stop) if body else -1
    do_route = stop == "route"
    reqs = rows // tt
    g = inproj_cfg(rows)
    tn, tm = -(-kda.NPROJ // g.bn), -(-rows // g.bm)
    n_inproj = tn * g.ks * tm
    go = oproj_cfg(rows)
    tno, tmo = HIDDEN // go.bn, -(-rows // go.bm)
    n_oproj = tno * tmo * go.ks
    kto = kda.P // go.ks
    rb_o = go.bm // go.ks
    sl = HIDDEN // tp
    ct_o = sl // go.bn
    n_pieces = coll.pieces(rows, rb_o, go.bn, tp)
    n_reduce = 0 if oneshot else n_pieces
    pf = coll.full_pieces(rows, rb_o, go.bn)
    # a body launch spills past 48-row bands; a seam launch keeps one round of tasks
    band_cap = 48 if body else 96
    n_seam = mhc.tasks(rows, band_cap)
    gr = route.rgate_cfg(rows)
    ntr, tmr = route.n_tiles(gr), -(-rows // gr.bm)
    n_router = ntr * gr.ks * tmr
    br = rows // mhc.bands(rows, band_cap)
    nbw = (br - 1) // rb_o + 2  # AG bands a slice band can touch
    sc = scratch_layout(rows, tt, trace)
    poffs, half = coll.layout(rows, rb_o, go.bn, tp, oneshot, t1)
    n_tpush = -(-rows // rb_o) * (HIDDEN // go.bn) if t1 else 0
    n_treduce = n_pieces if t1 else 0
    lw = mhc.lds_words(rows, band_cap)
    proj_words = rows * kda.NPROJ // 2
    zero_per = -(-proj_words // (n_seam * 4)) * 4

    lw_rl, lw_fl, lw_red = lw["rl"], lw["fl"], lw["red"]

    @fx.struct
    class MhcLds:
        rl: fx.Array[fx.Float32, lw_rl, 16]
        fl: fx.Array[fx.Float32, lw_fl, 16]
        red: fx.Array[fx.Float32, lw_red, 16]

    gemm_words = max(c.stages * (c.bm + c.bn) * c.bk // 2 for c in (g, go, gr))
    c_words = max([c.bm * (c.bn + 4) for c in (g, go)] + [route.RED_WORDS])

    @fx.struct
    class GemmLds:
        ab: fx.Array[fx.Int32, gemm_words, 16]

    @fx.struct
    class GemmC:
        c: fx.Array[fx.Float32, c_words, 16]

    @fx.struct
    class KdaLds:
        f: fx.Array[fx.Float32, kda.lds_floats(tt), 16]

    swl = sort.lds_words(rows)

    swl_masks, swl_bbl, swl_scan = swl["masks"], swl["bbl"], swl["scan"]

    @fx.struct
    class SortLds:
        masks: fx.Array[fx.Int32, swl_masks, 16]
        bbl: fx.Array[fx.Int32, swl_bbl, 16]
        scan: fx.Array[fx.Int32, swl_scan, 16]

    @fx.union
    class Smem:
        mhc: MhcLds
        gemm: GemmLds
        gc: GemmC
        kda: KdaLds
        srt: SortLds

    sly = sort.layout(rows)
    # the CTA the sort task lands on: first past the route rows (stage bases are static)
    body_tasks = (
        n_inproj + reqs * kda.NH + n_oproj + n_reduce + n_seam + rows if body else 0
    )
    sort_cta = (
        n_seam + rows + body_tasks + n_router + rows + (n_tpush + n_treduce)
    ) % BLOCKS

    I64, I32 = fx.Int64, fx.Int32

    @flyc.kernel(
        name=f"g53w_layer_r{rows}_t{tt}_tp{tp}_s{int(state_bf16)}_iw{int(has_iw)}_{stop}"
        f"_tr{int(trace)}_os{int(oneshot)}_t1{int(t1)}_b{int(body)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        res_in: I64,
        hc_fn_a: I64,
        hc_scale_a: I64,
        hc_base_a: I64,
        norm_w_a: I64,
        x_a: I64,
        post_a: I64,
        comb_a: I64,
        w_in: I64,
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
        core: I64,
        w_o: I64,
        acc_o: I64,
        hc_fn_f: I64,
        hc_scale_f: I64,
        hc_base_f: I64,
        norm_w_f: I64,
        res_mid: I64,
        x_f: I64,
        post_f: I64,
        comb_f: I64,
        w_r: I64,
        bias_r: I64,
        ids: I64,
        wts: I64,
        logits: I64,
        moe_in: I64,
        post_p: I64,
        comb_p: I64,
        res_cur: I64,
        epoch: I64,
        scratch: I64,
        peers: I64,
        rank: I32,
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
        sbuf: I64,
        moe_out: I64,
        m_tok: I32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem)
        ml = lds.mhc.peek()
        ab = lds.gemm.peek().ab.ptr
        cl = lds.gc.peek().c.ptr
        kl = lds.kda.peek().f.ptr
        ep = uniform(
            fx.Int32(
                bo.buffer_load(
                    rsrc(epoch), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
        )
        cb = dict(tid=tid, bid=bid)
        epoch_begin(cb, epoch, ep)
        box = Mailbox(ep + 1)

        def stamp(point):
            if const_expr(trace):
                _stamp(tid, bid, scratch, sc["trace"], point)

        stamp(0)
        if const_expr(moe_sort):
            if bid == sort_cta:
                sort.sort_reset(tid, sly, sbuf)

        def seam_ctx(pmix_name, args, store_x, x_done, **kw):
            pa = sreg(scratch, sc[pmix_name] * 4, pmix_name)

            def put(_base, i, v):
                box.put(pa, i, v)

            def getn(_base, idxs):
                got = box.poll([(pa, i, 1) for i in idxs])
                return [w[0].bitcast(fx.Float32) for w in got]

            return dict(
                S=rows,
                BR=br,
                tid=tid,
                lane=tid % 64,
                wave=tid // 64,
                rl=ml.rl.ptr,
                fl=ml.fl.ptr,
                red=ml.red.ptr,
                args=dict(args, pmix=None),
                put=put,
                getn=getn,
                store_x=store_x,
                x_done=x_done,
                cm=CM_DEV,
                **kw,
            )

        def x_store(buf):
            def store(t, col, words):
                bo.buffer_store(
                    words, rsrc(buf), (t * HIDDEN + col) // 2, cache_modifier=CM_DEV
                )

            return store

        base = 0
        # R_in may be the caller's own residual of m_tok rows: pad rows read 0
        res_bytes = fx.Int32(m_tok) * (mhc.HC * HIDDEN * 2)
        if const_expr(t1):
            ct = coll.prefixed(
                kcoll.region_context(peers, rank, epoch, 0, poffs, half, tid, bid), "t_"
            )
            # ---- tpush: the MoE partial's (band, column tile) pieces to their owners
            for task in range(first_task(bid, base), n_tpush, BLOCKS):
                band = task // (HIDDEN // go.bn)
                n0 = (task % (HIDDEN // go.bn)) * go.bn
                # the partial may be the caller's of m_tok rows (a DSA layer's attention)
                _push_band_flag(
                    ct,
                    moe_in,
                    rows,
                    HIDDEN,
                    band * rb_o,
                    rb_o,
                    n0,
                    go.bn,
                    1,
                    "linbf",
                    sl,
                    band * ct_o + (n0 % sl) // go.bn,
                    nbytes=fx.Int32(m_tok) * (HIDDEN * 2),
                )
            base += n_tpush
            # ---- treduce
            for piece in range(first_task(bid, base), n_treduce, BLOCKS):
                coll.reduce_bcast(ct, piece, rows, rb_o, go.bn, tp)
            base += n_treduce
            r_attn = res_cur
            a_in = dict(
                res_in=res_in,
                res_bytes=res_bytes,
                x_in=ct["ag_own"].value,
                post_in=post_p,
                comb_in=comb_p,
                res_out=res_cur,
            )
            a_kw = dict(x_cm=CM_SYS)
        else:
            r_attn = res_in
            a_in = dict(res_in=res_in, res_bytes=res_bytes)
            a_kw = {}
        # ---- aseam: zero this task's share of proj, then the attention mHC slice
        mca = seam_ctx(
            "pmix_a",
            dict(
                hc_fn=hc_fn_a,
                hc_scale=hc_scale_a,
                hc_base=hc_base_a,
                norm_w=norm_w_a,
                post_out=post_a,
                comb_out=comb_a,
                **a_in,
            ),
            x_store(x_a),
            lambda t: _flag(tid, scratch, sc["xflag"] + t, ep + 1),
            **a_kw,
        )
        r_proj = rsrc(proj, proj_words * 4)
        zero4 = fx.Vector.filled(4, 0, fx.Int32)
        for task in range(first_task(bid, base), n_seam, BLOCKS):
            if const_expr(body):
                for z0 in range(0, zero_per, THREADS * 4):
                    wi = task * zero_per + z0 + tid * 4
                    bo.buffer_store(
                        zero4, r_proj, wi, mask=wi < proj_words, cache_modifier=CM_DEV
                    )
                rocdl.s_waitcnt(vmcnt=0)
            if const_expr(t1):
                _poll_ag(ct, task, rows, br, rb_o, go.bn, sl, n_pieces, nbw)
            mhc.slice_task(mca, task)
        stamp(1)
        # ---- agate
        base += n_seam
        for t in range(first_task(bid, base), rows, BLOCKS):
            mhc.gate_row(mca, t)
        stamp(2)
        # ---- inproj
        base += rows
        if const_expr(body):
            kt = HIDDEN // g.ks
            cnt_p = scratch + fx.Int64(sc["cnt_p"] * 4)
            for task in range(first_task(bid, base), n_inproj, BLOCKS):
                n0 = (task % tn) * g.bn
                sk = (task // tn) % g.ks
                m0 = (task // (tn * g.ks)) * g.bm
                tile_dma.dma_tile(
                    tid,
                    x_a,
                    HIDDEN,
                    rows,
                    m0,
                    w_in,
                    HIDDEN,
                    kda.NPROJ,
                    n0,
                    sk * kt,
                    kt,
                    g.bm,
                    g.bn,
                    g.wm,
                    g.wn,
                    ab,
                    None,
                    g.bk,
                    g.stages,
                    atomic_lin(proj, kda.NPROJ, n0, rows, kda.NPROJ, True),
                    cl,
                    CM_DEV,
                    wait_x=lambda: _wait_words(
                        tid,
                        fx.min(g.bm, rows - m0),
                        scratch,
                        sc["xflag"] + m0 + tid,
                        ep + 1,
                    ),
                )
                _count(tid, cnt_p + fx.Int64((task % tn) * CNT_STRIDE * 4))
            stamp(3)
            # ---- kda
            base += n_inproj

            def wait_proj(n, hv):
                tile, nt = _head_tiles(hv, g.bn, tid)
                _wait_words(
                    tid,
                    nt,
                    scratch,
                    sc["cnt_p"] + tile * CNT_STRIDE,
                    (ep + 1) * g.ks * tm,
                )

            kc = dict(
                TT=tt,
                tid=tid,
                lower_bound=lower_bound,
                onorm_eps=onorm_eps,
                state_bf16=state_bf16,
                has_iw=has_iw,
                proj_cm=CM_DEV,
                out_cm=CM_DEV,
                wait_proj=wait_proj,
                head_done=lambda n, hv: _flag(
                    tid, scratch, sc["kflag"] + n * kda.NH + hv, ep + 1
                ),
                args=dict(
                    proj=proj,
                    conv_w=conv_w,
                    conv_st=conv_st,
                    cs_idx=cs_idx,
                    iw=iw,
                    iw_idx=iw_idx,
                    w_fgb=w_fgb,
                    a_log=a_log,
                    dt_bias=dt_bias,
                    state=state,
                    h0_idx=h0_idx,
                    rawv=rawv,
                    rawk=rawk,
                    ring_g=ring_g,
                    ring_beta=ring_beta,
                    ring_idx=ring_idx,
                    on_w=on_w,
                    core_out=core,
                    cs_line=cs_line,
                    cs_tok=cs_tok,
                    iw_line=iw_line,
                    iw_step=iw_step,
                    iw_win=iw_win,
                    rawv_slot=rawv_slot,
                    rawk_slot=rawk_slot,
                    g_slot=g_slot,
                    beta_slot=beta_slot,
                    ring_len=ring_len,
                ),
            )
            for task in range(first_task(bid, base), reqs * kda.NH, BLOCKS):
                kda.kda_head(kc, task // kda.NH, task % kda.NH, kl)
            stamp(4)
        if const_expr(body and level >= 1):
            co = kcoll.region_context(peers, rank, epoch, 0, poffs, half, tid, bid)
            # ---- oproj: (tile, K slice) tasks, a tile's slices consecutive
            base += reqs * kda.NH
            nh = kto // kda.KD
            cnt_o = scratch + fx.Int64(sc["cnt_o"] * 4)
            for task in range(first_task(bid, base), n_oproj, BLOCKS):
                sk = task % go.ks
                tile = task // go.ks
                m0 = (tile // tno) * go.bm
                n0 = (tile % tno) * go.bn
                r0 = m0 // tt
                nr = (fx.min(m0 + go.bm, rows) - 1) // tt - r0 + 1
                h0 = sk * nh
                tile_dma.dma_tile(
                    tid,
                    core,
                    kda.P,
                    rows,
                    m0,
                    w_o,
                    kda.P,
                    HIDDEN,
                    n0,
                    sk * kto,
                    kto,
                    go.bm,
                    go.bn,
                    go.wm,
                    go.wn,
                    ab,
                    None,
                    go.bk,
                    go.stages,
                    atomic_lin(acc_o, HIDDEN, n0, rows, HIDDEN, True),
                    cl,
                    CM_DEV,
                    wait_x=lambda: _wait_words(
                        tid,
                        nr * nh,
                        scratch,
                        sc["kflag"] + (r0 + tid // nh) * kda.NH + h0 + tid % nh,
                        ep + 1,
                    ),
                )
                _arrive_all(tid, cnt_o, tile, go.ks, kl, CNT_STRIDE)
                ob = (m0 + sk * rb_o) // rb_o
                if const_expr(oneshot):
                    coll.push_all(
                        co,
                        acc_o,
                        rows,
                        m0 + sk * rb_o,
                        rb_o,
                        n0,
                        go.bn,
                        tp,
                        ob * (HIDDEN // go.bn) + n0 // go.bn,
                    )
                else:
                    _push_band_flag(
                        co,
                        acc_o,
                        rows,
                        HIDDEN,
                        m0 + sk * rb_o,
                        rb_o,
                        n0,
                        go.bn,
                        go.ks,
                        "linbf",
                        sl,
                        ob * ct_o + (n0 % sl) // go.bn,
                    )
            stamp(5)
            # ---- reduce + broadcast: this rank's column slice
            base += n_oproj
            if const_expr(not oneshot):
                for piece in range(first_task(bid, base), n_reduce, BLOCKS):
                    coll.reduce_bcast(co, piece, rows, rb_o, go.bn, tp)
            stamp(6)
        if const_expr(body and level >= 2):
            # ---- fseam: R_mid = post o + comb R, the FFN mixes
            base += n_reduce
            if const_expr(oneshot):
                o_args = dict(x_in=co["os_own"].value, x_parts=(tp, rows * HIDDEN))
            else:
                o_args = dict(x_in=co["ag_own"].value)
            mcf = seam_ctx(
                "pmix_f",
                dict(
                    res_in=r_attn,
                    res_bytes=None if t1 else res_bytes,
                    post_in=post_a,
                    comb_in=comb_a,
                    res_out=res_mid,
                    hc_fn=hc_fn_f,
                    hc_scale=hc_scale_f,
                    hc_base=hc_base_f,
                    norm_w=norm_w_f,
                    post_out=post_f,
                    comb_out=comb_f,
                    **o_args,
                ),
                x_store(x_f),
                lambda t: _flag(tid, scratch, sc["xfflag"] + t, ep + 1),
                x_cm=CM_SYS,
            )
            for task in range(first_task(bid, base), n_seam, BLOCKS):
                if const_expr(oneshot):
                    c0 = (task % mhc.SLICES) * mhc.COLS
                    tb = (task // mhc.SLICES) * br
                    last = (fx.min(tb + br, rows) - 1) // rb_o
                    band = fx.min(tb // rb_o + tid % nbw, last)
                    idx = (tid // nbw) * pf + band * (HIDDEN // go.bn) + c0 // go.bn
                    _poll_peer(co, tp * nbw, "osf_own", idx)
                else:
                    _poll_ag(co, task, rows, br, rb_o, go.bn, sl, n_pieces, nbw)
                mhc.slice_task(mcf, task)
            stamp(7)
            # ---- fgate
            base += n_seam
            for t in range(first_task(bid, base), rows, BLOCKS):
                mhc.gate_row(mcf, t)
            stamp(8)
        if const_expr(do_route):
            # ---- router: (column tile, K slice, row tile) tasks, on fgate's x_f (agate's
            # x without a body)
            if const_expr(body):
                base += rows
            x_r, xr_flag = (x_f, sc["xfflag"]) if body else (x_a, sc["xflag"])
            ktr = HIDDEN // gr.ks
            parts = scratch + fx.Int64(sc["parts"] * 4)
            cnt_r = scratch + fx.Int64(sc["cnt_r"] * 4)
            for task in range(first_task(bid, base), n_router, BLOCKS):
                nt = task % ntr
                sk = (task // ntr) % gr.ks
                mt = task // (ntr * gr.ks)
                m0 = mt * gr.bm
                tile_dma.dma_tile(
                    tid,
                    x_r,
                    HIDDEN,
                    rows,
                    m0,
                    w_r,
                    HIDDEN,
                    N_EXPERTS,
                    nt * gr.bn,
                    sk * ktr,
                    ktr,
                    gr.bm,
                    gr.bn,
                    gr.wm,
                    gr.wn,
                    ab,
                    route.part_emit(parts, rows, sk, nt * gr.bn),
                    gr.bk,
                    gr.stages,
                    None,
                    cl,
                    CM_DEV,
                    wait_x=lambda: _wait_words(
                        tid,
                        fx.min(gr.bm, rows - m0),
                        scratch,
                        xr_flag + m0 + tid,
                        ep + 1,
                    ),
                )
                _count(tid, cnt_r + fx.Int64((mt * ntr + nt) * CNT_STRIDE * 4))
            stamp(10)
            # ---- route
            base += n_router
            for t in range(first_task(bid, base), rows, BLOCKS):
                _wait_words(
                    tid,
                    ntr,
                    scratch,
                    sc["cnt_r"] + ((t // gr.bm) * ntr + tid) * CNT_STRIDE,
                    (ep + 1) * gr.ks,
                )
                stamp(12)
                route.route_row(
                    tid, t, rows, gr.ks, parts, bias_r, ids, wts, logits, cl, shared_w
                )
                if const_expr(moe_sort):
                    _flag(tid, scratch, sc["sflag"] + t, ep + 1)
            stamp(11)
            if const_expr(moe_sort):
                # ---- sort: one CTA past the route rows; the rest zero the experts' output
                base += rows
                assert base % BLOCKS == sort_cta, (base, sort_cta)
                if first_task(bid, base) == 0:
                    sl_ = lds.srt.peek()
                    sort.sort_rows(
                        tid,
                        m_tok,
                        sly,
                        scratch,
                        sc["sflag"],
                        ep + 1,
                        ids,
                        wts,
                        sbuf,
                        sl_.masks.ptr,
                        sl_.bbl.ptr,
                        sl_.scan.ptr,
                    )
                base += 1
                r_moe = rsrc(moe_out, rows * HIDDEN * 2)
                for r in range(first_task(bid, base), rows, BLOCKS):
                    bo.buffer_store(zero4, r_moe, r * (HIDDEN // 2) + tid * 4)
                stamp(13)
        epoch_end(cb, epoch, ep)
        stamp(9)

    @flyc.jit
    def launch(
        res_in: I64,
        hc_fn_a: I64,
        hc_scale_a: I64,
        hc_base_a: I64,
        norm_w_a: I64,
        x_a: I64,
        post_a: I64,
        comb_a: I64,
        w_in: I64,
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
        core: I64,
        w_o: I64,
        acc_o: I64,
        hc_fn_f: I64,
        hc_scale_f: I64,
        hc_base_f: I64,
        norm_w_f: I64,
        res_mid: I64,
        x_f: I64,
        post_f: I64,
        comb_f: I64,
        w_r: I64,
        bias_r: I64,
        ids: I64,
        wts: I64,
        logits: I64,
        moe_in: I64,
        post_p: I64,
        comb_p: I64,
        res_cur: I64,
        epoch: I64,
        scratch: I64,
        peers: I64,
        rank: I32,
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
        sbuf: I64,
        moe_out: I64,
        m_tok: I32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = keyed
        kernel(
            res_in,
            hc_fn_a,
            hc_scale_a,
            hc_base_a,
            norm_w_a,
            x_a,
            post_a,
            comb_a,
            w_in,
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
            core,
            w_o,
            acc_o,
            hc_fn_f,
            hc_scale_f,
            hc_base_f,
            norm_w_f,
            res_mid,
            x_f,
            post_f,
            comb_f,
            w_r,
            bias_r,
            ids,
            wts,
            logits,
            moe_in,
            post_p,
            comb_p,
            res_cur,
            epoch,
            scratch,
            peers,
            rank,
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
            sbuf,
            moe_out,
            m_tok,
        ).launch(grid=(BLOCKS, 1, 1), block=(THREADS,), stream=stream)

    launch.plan = dict(
        rows=rows,
        gemm=g,
        oproj=go,
        rb_o=rb_o,
        n_seam=n_seam,
        n_inproj=n_inproj,
        n_kda=reqs * kda.NH,
        n_oproj=n_oproj,
        n_pieces=n_pieces,
        rgate=gr,
        n_router=n_router,
        scratch_words=sc["end"],
        epoch_words=EPOCH_MARKS + BLOCKS,
        peer_bytes=coll.peer_bytes(rows, rb_o, go.bn, tp, oneshot, t1),
        oneshot=oneshot,
        t1=t1,
        n_tpush=n_tpush,
        moe_sort=moe_sort,
        sort_layout=sly,
    )
    _BUILDS[key] = launch
    return launch
