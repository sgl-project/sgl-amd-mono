# SPDX-License-Identifier: Apache-2.0
"""Milestone 1a of the GLM-5.3-Flash wide launch: one persistent launch (BLOCKS CTAs,
one task loop a stage, no grid barrier) running

    aseam   mHC slices of the replicated residual R_in (and zero ``proj``)
    agate   a row's mixes -> pre / post / comb, the attention input x (RMSNorm)
    inproj  proj [rows, 6416] = x @ W_in^T: direct-to-LDS tile GEMM, split-K bf16 adds
    kda     (request, head): f_b / g_b, conv, ring verify, gated RMSNorm -> core_out

on one GPU (the attention side up to o_proj; no collective yet). Hand-offs: the
slice partials go through tagged mailbox pairs (tag = epoch + 1), each x row through
a flag word (:= epoch + 1), each in_proj column tile through a monotonic counter
(target = (epoch + 1) * arrivals) on its own line, so nothing is reset between
launches; a KDA head waits for the six column tiles it reads only. CTA 0 moves the epoch on once every CTA has read it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.epoch import EPOCH_MARKS, epoch_begin, epoch_end
from sgl_amd_mono.common.ops import CM_DEV, clock, ld_i32, rsrc, uniform
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.sync import Mailbox, _spin, sreg
from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN
from sgl_amd_mono.glm53_flash.gfx950.wide import kda, mhc, tile_dma
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin

SOURCES = kernel_sources(__name__)


class Gemm(msgspec.Struct, frozen=True):
    bm: int
    bn: int
    wm: int
    wn: int
    ks: int
    bk: int
    stages: int


# in_proj [rows, 4096] x [6416, 4096]^T, the fastest direct-to-LDS cfg a width (dma_bench)
INPROJ = {
    18: Gemm(32, 128, 2, 4, 4, 128, 4),
    36: Gemm(64, 128, 2, 4, 4, 64, 6),
    72: Gemm(128, 128, 2, 4, 4, 64, 4),
    96: Gemm(128, 128, 2, 4, 4, 64, 4),
    144: Gemm(96, 128, 2, 4, 2, 64, 4),
    192: Gemm(96, 128, 2, 4, 2, 64, 4),
}


def inproj_cfg(rows):
    return INPROJ[min(w for w in INPROJ if w >= rows)]


def _tiles(rows, g):
    return -(-kda.NPROJ // g.bn), -(-rows // g.bm)


TRACE_POINTS = 16  # timestamps a CTA records with ``trace``
CNT_STRIDE = 32  # words between counters: one 128 B line each


def scratch_layout(rows, trace=False):
    """Word offsets in the scratch buffer: counters, the slice partials (pairs), then
    with ``trace`` [BLOCKS, TRACE_POINTS] i64 timestamps."""
    xflag = 0
    cnt_p = -(-rows // 32) * 32
    pmix = cnt_p + CNT_STRIDE * -(-kda.NPROJ // 64)
    pmix_end = pmix + 2 * mhc.pmix_words(rows)
    end = pmix_end + (BLOCKS * TRACE_POINTS * 2 if trace else 0)
    return {"xflag": xflag, "cnt_p": cnt_p, "pmix": pmix, "trace": pmix_end, "end": end}


@flyc.jit
def _stamp(tid, bid, scratch, word0, point):
    if tid == 0:
        t = clock()
        w = fx.Vector.from_elements([fx.Int32(t), fx.Int32(t >> 32)], fx.Int32)
        bo.buffer_store(w, rsrc(scratch), word0 + (bid * TRACE_POINTS + point) * 2)


@flyc.jit
def _count(tid, cnt):
    """Drain this CTA's stores, then add one to the counter at ``cnt``."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), cnt)
        fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
    gpu.barrier()


@flyc.jit
def _flag(tid, base, i, v):
    """Drain this CTA's stores, then word ``i`` of ``base`` := ``v``."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        bo.buffer_store(v, rsrc(base), i, cache_modifier=CM_DEV)


@flyc.jit
def _wait_words(tid, n, base, idx, target):
    """Threads below ``n`` spin until word ``idx`` (their own) of ``base`` reaches
    ``target`` (flags and counters only grow); then the CTA goes on together."""
    if tid < n:
        _spin(lambda: ld_i32(base, idx, CM_DEV), lambda v: (v - target) < 0)
    gpu.barrier()


def _head_tiles(hv, bn, i):
    """Column tile ``i`` of the ones KDA head ``hv`` reads: its q, k, v columns, then
    the shared tail (beta, f_a, g_a); and how many there are."""
    per = kda.KD // bn
    nq = 3 * per
    tail0 = kda.BETA_0 // bn
    nt = nq + (kda.NPROJ - 1) // bn - tail0 + 1
    r, j = i // per, i % per
    qkv = (r * kda.P + hv * kda.KD) // bn + j
    return (i < nq).select(qkv, tail0 + i - nq), nt


ARGS = (
    # mHC
    "res_in",
    "hc_fn",
    "hc_scale",
    "hc_base",
    "norm_w",
    "x",
    "post_out",
    "comb_out",
    # in_proj
    "w_in",
    "proj",
    # KDA
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
    # launch
    "epoch",
    "scratch",
)

_BUILDS: dict = {}


def build(rows, tt, lower_bound, onorm_eps, state_bf16, has_iw, trace=False):
    keyed = key_tuple(locals(), SOURCES)
    key = (rows, tt, lower_bound, onorm_eps, state_bf16, has_iw, trace)
    if key in _BUILDS:
        return _BUILDS[key]
    assert rows % tt == 0
    reqs = rows // tt
    g = inproj_cfg(rows)
    tn, tm = _tiles(rows, g)
    n_inproj = tn * g.ks * tm
    n_aseam = mhc.tasks(rows)
    sc = scratch_layout(rows, trace)
    lw = mhc.lds_words(rows)
    proj_words = rows * kda.NPROJ // 2
    zero_per = -(-proj_words // (n_aseam * 4)) * 4  # i32 words a slice task zeroes

    lw_rl, lw_fl, lw_red = lw["rl"], lw["fl"], lw["red"]

    @fx.struct
    class MhcLds:
        rl: fx.Array[fx.Float32, lw_rl, 16]
        fl: fx.Array[fx.Float32, lw_fl, 16]
        red: fx.Array[fx.Float32, lw_red, 16]

    @fx.struct
    class GemmLds:
        ab: fx.Array[fx.Int32, g.stages * (g.bm + g.bn) * g.bk // 2, 16]

    @fx.struct
    class GemmC:
        c: fx.Array[fx.Float32, g.bm * (g.bn + 4), 16]

    @fx.struct
    class KdaLds:
        f: fx.Array[fx.Float32, kda.lds_floats(tt), 16]

    @fx.union
    class Smem:
        mhc: MhcLds
        gemm: GemmLds
        gc: GemmC
        kda: KdaLds

    I64, I32 = fx.Int64, fx.Int32

    @flyc.kernel(
        name=f"g53w_m1a_r{rows}_t{tt}_s{int(state_bf16)}_iw{int(has_iw)}_tr{int(trace)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        res_in: I64,
        hc_fn: I64,
        hc_scale: I64,
        hc_base: I64,
        norm_w: I64,
        x: I64,
        post_out: I64,
        comb_out: I64,
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
        core_out: I64,
        epoch: I64,
        scratch: I64,
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
        pmix_a = sreg(scratch, sc["pmix"] * 4, "pmix")
        cnt_p = scratch + fx.Int64(sc["cnt_p"] * 4)

        def stamp(point):
            if const_expr(trace):
                _stamp(tid, bid, scratch, sc["trace"], point)

        stamp(0)

        # ---- aseam: zero this task's share of proj, then the mHC slice
        def put(_base, i, v):
            box.put(pmix_a, i, v)

        def getn(_base, idxs):
            got = box.poll([(pmix_a, i, 1) for i in idxs])
            return [w[0].bitcast(fx.Float32) for w in got]

        def store_x(t, col, words):
            bo.buffer_store(
                words, rsrc(x), (t * HIDDEN + col) // 2, cache_modifier=CM_DEV
            )

        def x_done(t):
            _flag(tid, scratch, sc["xflag"] + t, ep + 1)

        mc = dict(
            S=rows,
            tid=tid,
            lane=tid % 64,
            wave=tid // 64,
            rl=ml.rl.ptr,
            fl=ml.fl.ptr,
            red=ml.red.ptr,
            args=dict(
                res_in=res_in,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                norm_w=norm_w,
                pmix=None,
                post_out=post_out,
                comb_out=comb_out,
            ),
            put=put,
            getn=getn,
            store_x=store_x,
            x_done=x_done,
        )
        r_proj = rsrc(proj, proj_words * 4)
        zero4 = fx.Vector.filled(4, 0, fx.Int32)
        for task in range(first_task(bid, 0), n_aseam, BLOCKS):
            for z0 in range(0, zero_per, THREADS * 4):
                wi = task * zero_per + z0 + tid * 4
                bo.buffer_store(
                    zero4, r_proj, wi, mask=wi < proj_words, cache_modifier=CM_DEV
                )
            rocdl.s_waitcnt(vmcnt=0)
            mhc.slice_task(mc, task)
        stamp(1)
        # ---- agate: one row a task
        base = n_aseam
        for t in range(first_task(bid, base), rows, BLOCKS):
            mhc.gate_row(mc, t)
        # ---- inproj: (n tile, K slice, m tile) tasks, after every x row
        stamp(2)
        base += rows
        stamp(3)
        kt = HIDDEN // g.ks
        for task in range(first_task(bid, base), n_inproj, BLOCKS):
            n0 = (task % tn) * g.bn
            sk = (task // tn) % g.ks
            m0 = (task // (tn * g.ks)) * g.bm
            lin = atomic_lin(proj, kda.NPROJ, n0, rows, kda.NPROJ, True)
            tile_dma.dma_tile(
                tid,
                x,
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
                lin,
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
        stamp(4)
        # ---- kda: (request, head) tasks, after every in_proj tile
        base += n_inproj

        def wait_proj(n, hv):
            tile, nt = _head_tiles(hv, g.bn, tid)
            _wait_words(
                tid, nt, scratch, sc["cnt_p"] + tile * CNT_STRIDE, (ep + 1) * g.ks * tm
            )
            stamp(7)

        kc = dict(
            TT=tt,
            tid=tid,
            lower_bound=lower_bound,
            onorm_eps=onorm_eps,
            state_bf16=state_bf16,
            has_iw=has_iw,
            proj_cm=CM_DEV,
            **({"stamp": stamp} if trace else {}),
            wait_proj=wait_proj,
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
                core_out=core_out,
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
        stamp(5)
        epoch_end(cb, epoch, ep)
        stamp(6)

    @flyc.jit
    def launch(
        res_in: I64,
        hc_fn: I64,
        hc_scale: I64,
        hc_base: I64,
        norm_w: I64,
        x: I64,
        post_out: I64,
        comb_out: I64,
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
        core_out: I64,
        epoch: I64,
        scratch: I64,
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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = keyed
        kernel(
            res_in,
            hc_fn,
            hc_scale,
            hc_base,
            norm_w,
            x,
            post_out,
            comb_out,
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
            core_out,
            epoch,
            scratch,
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
        ).launch(grid=(BLOCKS, 1, 1), block=(THREADS,), stream=stream)

    launch.plan = dict(
        rows=rows,
        gemm=g,
        n_aseam=n_aseam,
        n_inproj=n_inproj,
        n_kda=reqs * kda.NH,
        scratch_words=sc["end"],
        epoch_words=EPOCH_MARKS + BLOCKS,
    )
    _BUILDS[key] = launch
    return launch
