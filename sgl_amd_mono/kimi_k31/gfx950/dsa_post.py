# SPDX-License-Identifier: Apache-2.0
"""One wide-row DSA post-attention launch: T4 .. T6 + MoE front tail.

    o_proj -> RS + residual -> MLP AttnRes seam -> MoE front
    -> route (+ sort) / shared SiTU -> shared down

Reuses the post-attention half of ``gfx950/layer.py`` without T1..T3 or KDA for
DSA layers (whose attention stops at ``forward_pre_o_proj``). Uses its own peer
ring and scratch (``peers_dsa``, ``scratch_dsa``) while sharing the epoch ring,
sort buffer, and activation buffers with the KDA layer launches.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, KernArgs, load_ptr64
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.sync import sreg
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k31.gfx950 import chain as post
from sgl_amd_mono.kimi_k31.gfx950.stages import coll
from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf
from sgl_amd_mono.kimi_k31.gfx950.stages.attnres import red_words
from sgl_amd_mono.kimi_k31.gfx950.stages.gemm_coll import (
    _arrive_all,
    _push_band_flag,
    dev_emit,
)
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin, bf16_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import BK, DEPTH, RW, mma_tile
from sgl_amd_mono.kimi_k31.plan import Plan

SOURCES = kernel_sources(__name__)

TP, H, SL, KO = coll.TP, post.H, post.H // coll.TP, post.KO
CNT_STRIDE = 32
MARKS = (
    "start",
    "t4 oproj",
    "t5 rs",
    "t5 seam",
    "t6 front",
    "route",
    "situ",
    "sdown",
)
TL_POINTS = len(MARKS)

_BUILDS: dict = {}


def plan(m, go, gf, gs, nf, nvb_m, timeline=False, nbk=8):
    """Combined plan for ``dsa_post`` (post-attention chain + MoE front tail).

    Reserves ``o_xs`` for ``nbk`` blocks so every ``nvb_m`` sharing one peer ring
    places all mailbox regions at identical byte offsets.
    """
    assert 0 <= nvb_m <= nbk, (nvb_m, nbk)
    assert go.ks > 0 and go.bm % go.ks == 0 and SL % go.bn == 0, (go,)
    po = post.plan(m, go, gf, nf, nbk)
    p = Plan()
    for r in po.regions:
        p.region("o_" + r.name, r.nbytes, r.peer)
    for s in po.stages:
        p.stage(
            s.name if s.name != "seam" else "mseam",
            s.tasks,
            tuple(w if w != "seam" else "mseam" for w in s.waits_on),
            s.all_ranks,
        )
    assert gf.ks > 0 and gf.bm % gf.ks == 0, (gf.bm, gf.ks)
    nbf = (m + gf.bm // gf.ks - 1) // (gf.bm // gf.ks)
    tnf = (nf + gf.bn - 1) // gf.bn
    p.region("rgat", m * mf.E * 4)
    p.region("lgat", m * mf.LAT * 2)
    p.region("fgf", TP * nbf * mf.gathered_tiles(gf.bn) * 8)
    p.region("fbuf", m * nf * 4, peer=False)
    p.region("sgu", m * 2 * mf.SI * 2, peer=False)
    p.region("smark", nbf * tnf * 8, peer=False)
    p.region("hbuf", m * mf.SI * 2, peer=False)
    p.region("hmark", m * 8, peer=False)
    p.region("srtmk", 8, peer=False)
    p.stage("route", m, waits_on=("front",), all_ranks=True)
    p.stage("situ", m, waits_on=("front",))
    p.stage(
        "sdown",
        ((m + gs.bm - 1) // gs.bm) * ((H + gs.bn - 1) // gs.bn) * gs.ks,
        waits_on=("situ",),
    )
    if timeline:
        p.region("tl", BLOCKS * TL_POINTS * 8, peer=False)
    return p.check(), po


def counters(m, go, gf, nf):
    """int32 tile counters a ``dsa_post`` build uses (o_proj + front + sort)."""
    assert go.ks > 0 and go.bm % go.ks == 0 and SL % go.bn == 0, (go,)
    return (
        ((m + go.bm - 1) // go.bm) * (H // go.bn)
        + ((m + gf.bm - 1) // gf.bm) * ((nf + gf.bn - 1) // gf.bn)
        + 2
    ) * CNT_STRIDE


def dsa_post(
    m,
    go,
    gf,
    gs,
    nf,
    nvb_m=0,
    nbk=8,
    launches=1,
    eps=1e-5,
    half_bytes=0,
    timeline=False,
    front_rows=None,
    sort=False,
):
    """Build the DSA post-attention launch; returns ``(launch, plan)``.

    ``prefix`` and ``prefix_out`` are this rank's columns of full-width ``[rows, H]``
    buffers (row stride ``H``, pointer at ``col0 = 2 * rank * SL``); ``pfx_bytes``
    bounds readable bytes from ``prefix`` so padded bucket rows read 0.
    """
    sort = ("sgl" if sort is True else sort) or None
    assert sort is None or sort in mf.SORT_KINDS, sort
    keyed = key_tuple({**locals(), "front_rows": str(front_rows)}, SOURCES)
    key = (
        m,
        go.key(),
        gf.key(),
        gs.key(),
        nf,
        nvb_m,
        nbk,
        launches,
        eps,
        half_bytes,
        timeline,
        front_rows,
        sort,
    )
    if key in _BUILDS:
        return _BUILDS[key]
    assert nf == mf.NF and gf.ks > 1 and gs.ks == 1 and KO % go.ks == 0
    assert go.ks > 0 and go.bm % go.ks == 0 and SL % go.bn == 0, (go,)
    assert (KO // go.ks) % BK == 0
    assert 0 <= nvb_m <= nbk, (nvb_m, nbk)
    assert front_rows is None or len(front_rows) == 3, front_rows
    assert not sort or m <= mf.SORT_MAX_ROWS, m
    pl, po = plan(m, go, gf, gs, nf, nvb_m, timeline, nbk=nbk)
    offs, half = pl.peer_offsets
    assert half_bytes == 0 or half <= half_bytes, (half, half_bytes)
    half = max(half, half_bytes)
    soffs, _ = pl.scratch_offsets
    post_offs = {r.name: offs["o_" + r.name] for r in po.regions if r.peer}
    fo = 2 * half
    rb_o, ct_o = go.bm // go.ks, SL // go.bn
    assert H % gf.ks == 0, (H, gf.ks)
    kto, ktf = KO // go.ks, H // gf.ks
    assert ktf % BK == 0 and (ktf % SL == 0 or SL % ktf == 0), (gf.ks, ktf, SL)
    nown_f = max(ktf // SL, 1)
    tno, tmo = H // go.bn, (m + go.bm - 1) // go.bm
    tnf, tmf = (nf + gf.bn - 1) // gf.bn, (m + gf.bm - 1) // gf.bm
    n_o, n_f = tmo * tno * go.ks, tmf * tnf * gf.ks
    cnt_f = tmo * tno
    rbf = gf.bm // gf.ks
    nbf = (m + rbf - 1) // rbf
    cnt_end = cnt_f + tmf * tnf
    w_rt = cnt_end * CNT_STRIDE
    w_lc = (cnt_end + 1) * CNT_STRIDE
    lay_s = mf.sort_layout(m, sort) if sort else None
    sort_q = sort == "sgl"
    sort_cta = BLOCKS - m - 1
    help0 = sort_cta - mf.SORT_HELPERS
    wrow_f = None
    if front_rows is not None:
        n_r, n_l = mf.E_R, mf.L_R
        o_r, o_l, o_s = front_rows

        def wrow_f(row):
            return row + (row < n_r).select(
                fx.Int32(o_r),
                (row < n_r + n_l).select(
                    fx.Int32(o_l - n_r), fx.Int32(o_s - n_r - n_l)
                ),
            )

    tns, tms = (H + gs.bn - 1) // gs.bn, (m + gs.bm - 1) // gs.bm
    n_s = tms * tns
    assert not sort or BLOCKS - m - 1 - mf.SORT_HELPERS >= max(m, n_s), (m, n_s)
    gems = (go, gf, gs)
    bmx, bnx = max(g.bm for g in gems), max(g.bn for g in gems)
    cwords = max(g.bm * (g.bn + 4) for g in gems)

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bmx * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bnx * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, cwords, 16]

    @fx.struct
    class SortLds:
        masks: fx.Array[fx.Int32, 2 * mf.E, 16]
        bb: fx.Array[fx.Int32, mf.E, 16]
        sa: fx.Array[fx.Int32, WAVES, 16]
        sb: fx.Array[fx.Int32, WAVES, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds
        srt: SortLds

    @fx.struct
    class Small:
        slot: fx.Array[fx.Int32, 4, 16]
        red: fx.Array[fx.Float32, red_words(nvb_m), 16]

    @fx.struct
    class Tls:
        v: fx.Array[fx.Int64, TL_POINTS, 16]

    cfg_tag = "_".join("x".join(map(str, g.key())) for g in gems)
    name = (
        f"k31_dsa_post_m{m}_{cfg_tag}_n{nf}_m{nvb_m}of{nbk}_l{launches}_hb{half}"
        + ("_tl" if timeline else "")
        + ("_fr" + "x".join(map(str, front_rows)) if front_rows else "")
        + ({"sgl": "_srt", "aux16": "_srt16"}.get(sort, ""))
        + "_pH"
    )

    def dsa_post_body(
        core: Int64,
        wo: Int64,
        prefix: Int64,
        prefix_out: Int64,
        bank: Int64,
        cw_m: Int64,
        w_ln: Int64,
        wf: Int64,
        acc_o: Int64,
        bias: Int64,
        lat_out: Int64,
        topk_ids: Int64,
        topk_w: Int64,
        w_sd: Int64,
        sh_out: Int64,
        moe_out: Int64,
        logits_out: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        pfx_bytes: Int64,
        sort_buf: Int64,
    ):
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(
                epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo
            )
        ):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            alloc = fx.SharedAllocator()
            st = alloc.allocate(Smem)
            small = alloc.allocate(Small).peek()
            slot, red = small.slot.ptr, small.red.ptr
            tls = alloc.allocate(Tls).peek().v.ptr if timeline else None
            stamp_begin(timeline, tls, tid, TL_POINTS)
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            co = coll.region_context(
                peers, rank, epoch, slot_i, post_offs, half, tid, bid, launches
            )
            tail_offs = {k: offs[k] for k in ("rgat", "lgat", "fgf")}
            cf = coll.region_context(
                peers, rank, epoch, slot_i, tail_offs, half, tid, bid, launches
            )

            def sr(a, region):
                return sreg(a.scratch, soffs[region], region)

            def sp(a, region):
                return a.scratch + fx.Int64(soffs[region])

            # ---- T4: o_proj (core is already ready from forward_pre_o_proj)
            for task in range(bid, n_o, BLOCKS):
                sk = task % go.ks
                tile = task // go.ks
                m0 = (tile // tno) * go.bm
                n0 = (tile % tno) * go.bn
                a4 = ka("core", "wo", "acc_o", "cnt")
                mma_tile(
                    tid,
                    a4.core,
                    KO,
                    m,
                    m0,
                    a4.wo,
                    KO,
                    H,
                    n0,
                    sk * kto,
                    kto,
                    go.bm,
                    go.bn,
                    go.wm,
                    go.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    dev_emit(a4.acc_o, H, n0, m, H, False) if go.ks == 1 else None,
                    DEPTH,
                    None if go.ks == 1 else atomic_lin(a4.acc_o, H, n0, m, H, True),
                    cl,
                    CM_DEV,
                )
                _arrive_all(
                    tid,
                    a4.cnt,
                    tile,
                    go.ks,
                    slot,
                    CNT_STRIDE,
                )
                piece = ((m0 + sk * rb_o) // rb_o) * ct_o + (n0 % SL) // go.bn
                _push_band_flag(
                    co,
                    a4.acc_o,
                    m,
                    H,
                    m0 + sk * rb_o,
                    rb_o,
                    n0,
                    go.bn,
                    go.ks,
                    "linbf",
                    SL,
                    piece,
                )
            stamp(timeline, tls, tid, 1)
            # ---- T5: RS + residual, MLP AttnRes seam
            for piece in range(bid, ((m + rb_o - 1) // rb_o) * ct_o, BLOCKS):
                a5 = ka("prefix", "prefix_out", "scratch", "pfx_bytes")
                post._rs_piece(
                    co,
                    piece,
                    m,
                    SL,
                    rb_o,
                    go.bn,
                    a5.prefix,
                    a5.prefix_out,
                    sr(a5, "o_pdone"),
                    False,
                    None,
                    False,
                    0,
                    H,
                    H,
                    a5.pfx_bytes,
                )
            stamp(timeline, tls, tid, 2)
            for row in range(bid, m, BLOCKS):
                a5r = ka("prefix_out", "bank", "cw_m", "w_ln", "scratch")
                post._seam_row(
                    co,
                    row,
                    m,
                    SL,
                    rb_o,
                    ct_o,
                    nvb_m,
                    nbk,
                    a5r.prefix_out,
                    a5r.bank,
                    a5r.cw_m,
                    a5r.w_ln,
                    eps,
                    sr(a5r, "o_pdone"),
                    red,
                    False,
                    False,
                    None,
                    False,
                    0,
                    H,
                )
            stamp(timeline, tls, tid, 3)
            # ---- T6: MoE front
            xm = co["ag_own"].value
            g0 = first_task(bid, max(BLOCKS - n_f, 0))
            for task in range(g0, n_f, BLOCKS):
                sk = task % gf.ks
                tile = task // gf.ks
                m0 = (tile // tnf) * gf.bm
                n0 = (tile % tnf) * gf.bn
                post._front_wait(
                    co, m0, gf.bm, m, (sk * ktf) // SL, nown_f, False, None, False
                )
                a6 = ka("wf", "cnt", "scratch")
                fbuf = sp(a6, "fbuf")
                mma_tile(
                    tid,
                    xm,
                    H,
                    m,
                    m0,
                    a6.wf,
                    H,
                    nf,
                    n0,
                    sk * ktf,
                    ktf,
                    gf.bm,
                    gf.bn,
                    gf.wm,
                    gf.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    None,
                    DEPTH,
                    atomic_lin(fbuf, nf, n0, m, nf, False),
                    cl,
                    wrow=wrow_f,
                )
                _arrive_all(
                    tid,
                    a6.cnt + fx.Int64(4 * CNT_STRIDE * cnt_f),
                    tile,
                    gf.ks,
                    slot,
                    CNT_STRIDE,
                )
                r0 = m0 + sk * rbf
                mf.finish_front(
                    cf,
                    fbuf,
                    sp(a6, "sgu"),
                    m,
                    r0,
                    rbf,
                    n0,
                    gf.bn,
                    r0 // rbf,
                    n0 // gf.bn,
                    nbf,
                    sr(a6, "smark"),
                    tnf,
                )
            stamp(timeline, tls, tid, 4)
            # ---- T6 tail: routing + latent for the experts, shared SiTU + down
            for row in range(bid, m, BLOCKS):
                if const_expr(sort):
                    a7 = ka(
                        "bias",
                        "lat_out",
                        "topk_ids",
                        "topk_w",
                        "logits_out",
                        "moe_out",
                        "sort_buf",
                        "cnt",
                    )
                    mf.route_row(
                        cf,
                        row,
                        m,
                        rbf,
                        nbf,
                        gf.bn,
                        a7.bias,
                        a7.lat_out,
                        a7.topk_ids,
                        a7.topk_w,
                        a7.logits_out,
                        qout=(
                            a7.sort_buf + fx.Int64(lay_s["offs"]["q"])
                            if sort_q
                            else None
                        ),
                        tscale=(
                            a7.sort_buf + fx.Int64(lay_s["offs"]["ts"])
                            if sort_q
                            else None
                        ),
                        moe_out=a7.moe_out,
                        cnt=a7.cnt,
                        cnt_word=w_rt,
                    )
                else:
                    a7 = ka("bias", "lat_out", "topk_ids", "topk_w", "logits_out")
                    mf.route_row(
                        cf,
                        row,
                        m,
                        rbf,
                        nbf,
                        gf.bn,
                        a7.bias,
                        a7.lat_out,
                        a7.topk_ids,
                        a7.topk_w,
                        a7.logits_out,
                    )
            if const_expr(sort):
                sl = st.srt.peek()
                if bid == sort_cta:
                    a7s = ka("cnt", "topk_ids", "topk_w", "sort_buf", "scratch")
                    mf.sort_rows(
                        cf,
                        tid,
                        m,
                        a7s.cnt,
                        w_rt,
                        w_lc,
                        a7s.topk_ids,
                        a7s.topk_w,
                        a7s.sort_buf,
                        lay_s,
                        sr(a7s, "srtmk"),
                        sl.masks.ptr,
                        sl.bb.ptr,
                        sl.sa.ptr,
                        sl.sb.ptr,
                    )
                if const_expr(sort_q):
                    if (bid >= help0) & (bid < sort_cta):
                        a7h = ka("sort_buf", "scratch")
                        mf.place_codes(
                            cf,
                            tid,
                            bid - help0,
                            m,
                            a7h.sort_buf,
                            lay_s,
                            sr(a7h, "srtmk"),
                        )
            stamp(timeline, tls, tid, 5)
            for row in range(first_task(bid, max(BLOCKS - m, 0)), m, BLOCKS):
                a8 = ka("scratch")
                mf.situ_row(
                    cf,
                    row,
                    m,
                    rbf,
                    tnf,
                    gf.bn,
                    sp(a8, "sgu"),
                    sr(a8, "smark"),
                    sp(a8, "hbuf"),
                    sr(a8, "hmark"),
                )
            stamp(timeline, tls, tid, 6)
            for task in range(bid, n_s, BLOCKS):
                a9 = ka("w_sd", "sh_out", "scratch")
                hbuf, hmark = sp(a9, "hbuf"), sr(a9, "hmark")
                m0 = (task // tns) * gs.bm
                n0 = (task % tns) * gs.bn
                rows = fx.min(m0 + gs.bm, m) - m0
                if tid < rows:
                    cf["poll"]([(hmark, m0 + tid, 1)])
                gpu.barrier()
                mma_tile(
                    tid,
                    hbuf,
                    mf.SI,
                    m,
                    m0,
                    a9.w_sd,
                    mf.SI,
                    H,
                    n0,
                    0,
                    mf.SI,
                    gs.bm,
                    gs.bn,
                    gs.wm,
                    gs.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    bf16_emit(a9.sh_out, H, n0, m, H),
                    DEPTH,
                    None,
                    cl,
                    CM_DEV,
                )
            stamp(timeline, tls, tid, 7)
            if const_expr(timeline):
                a = ka("scratch")
                stamp_flush(timeline, tls, sp(a, "tl"), tid, bid, TL_POINTS)

    ka = KernArgs(
        [
            (n, 8 if t is Int64 else 4)
            for n, t in dsa_post_body.__annotations__.items()
            if n != "return"
        ]
    )
    post_kernel = flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])(
        dsa_post_body
    )

    @flyc.jit
    def launch(
        core: Int64,
        wo: Int64,
        prefix: Int64,
        prefix_out: Int64,
        bank: Int64,
        cw_m: Int64,
        w_ln: Int64,
        wf: Int64,
        acc_o: Int64,
        bias: Int64,
        lat_out: Int64,
        topk_ids: Int64,
        topk_w: Int64,
        w_sd: Int64,
        sh_out: Int64,
        moe_out: Int64,
        logits_out: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        pfx_bytes: Int64,
        sort_buf: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        post_kernel(
            core,
            wo,
            prefix,
            prefix_out,
            bank,
            cw_m,
            w_ln,
            wf,
            acc_o,
            bias,
            lat_out,
            topk_ids,
            topk_w,
            w_sd,
            sh_out,
            moe_out,
            logits_out,
            cnt,
            scratch,
            peers,
            rank,
            epoch,
            slot_i,
            pfx_bytes,
            sort_buf,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    from sgl_amd_mono.common.compiled_launch import compiled_launch

    launch = compiled_launch(launch)
    _BUILDS[key] = (launch, pl)
    return launch, pl
