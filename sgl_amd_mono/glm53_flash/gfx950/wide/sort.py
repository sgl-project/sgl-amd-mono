# SPDX-License-Identifier: Apache-2.0
"""The routed experts' sort at the wide launch's tail: AITER's BM = 16 aux sort
(``aiter.presorted_moe.register_presorted_moe``), so ``fused_moe`` launches only its two
GEMMs. Follows kimi_k31's ``sort_rows`` "aux16" kind, with GLM-5.3-Flash's 289 experts
(the fused shared one last), 9 slots, up to 192 tokens, and the token count ``m`` a
runtime value (the launch is built for its bucket, ``b`` rows; a step routes ``m <= b``).

The buffer of bucket ``b`` (``layout``): ids (slot << 24) | token, pad ``b``; weights f32,
pad 0; an expert a block of 16; num_valid [blocks x 16, m]; m_indices token, pad ``b``;
reverse_sorted, pair p's sorted row. It rests all pad (``init``); a launch first turns the
previous launch's pair rows (num_valid[1] tokens of them) back to pad (``sort_reset``), then
writes its own,
so the rows below num_valid are this launch's pairs or pads with no pad pass. A pad token
of ``b`` is at least every step's ``m``, as AITER's own pad ``M`` is.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import CM_DEV, block_excl_scan, ld_i32, popcount, rsrc
from sgl_amd_mono.common.plan import THREADS, WAVES
from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.glm53_flash.gfx950.config import MOE_SLOTS, N_EXPERTS

BM = 16
E = N_EXPERTS + 1  # the fused shared expert is id N_EXPERTS
TOPK = MOE_SLOTS
assert E <= THREADS


def mask_words(b):
    """32-bit words of an expert's token mask."""
    return -(-b // 32)


def lds_words(b):
    return {"masks": E * mask_words(b), "bbl": E, "scan": WAVES}


def layout(b):
    """Byte offsets of the sort buffer of bucket ``b`` and its size: AITER's
    ``aux_sort_capacity`` for M = b (it holds every M <= b)."""
    p = b * TOPK
    maxp = (p + min(E, p) * (BM - 1) + BM - 1) // BM * BM
    sizes = dict(
        ids=4 * maxp, w=4 * maxp, eids=4 * (maxp // BM), nv=8, mi=4 * maxp, dest=4 * p
    )
    offs, at = {}, 0
    for k, n in sizes.items():
        offs[k] = at
        at += (n + 255) // 256 * 256
    return dict(offs=offs, maxp=maxp, nblk=maxp // BM, nbytes=at, b=b)


def init(buf, b):
    """The resting state (host, once): every row pad, weight 0, num_valid [0, 0]."""
    import torch

    ly = layout(b)
    o = ly["offs"]
    buf.zero_()
    for k in ("ids", "mi"):
        buf[o[k] : o[k] + 4 * ly["maxp"]].view(torch.int32).fill_(b)


def views(buf, ly, m):
    """The six tensors ``register_presorted_moe`` takes, for a step of ``m`` tokens."""
    import torch

    o, maxp = ly["offs"], ly["maxp"]

    def at(k, n, dt=torch.int32):
        return buf[o[k] : o[k] + 4 * n].view(dt)

    return dict(
        sorted_token_ids=at("ids", maxp),
        sorted_weights=at("w", maxp, torch.float32),
        sorted_expert_ids=at("eids", ly["nblk"]),
        num_valid_ids=at("nv", 2),
        m_indices=at("mi", maxp),
        reverse_sorted=at("dest", m * TOPK),
    )


@flyc.jit
def sort_reset(tid, ly, sbuf):
    """The previous launch's pair rows (num_valid[1] tokens of them) back to pad: at the
    sort CTA's start, off the routing's path."""
    b = ly["b"]
    o = ly["offs"]
    a_ids = sbuf + fx.Int64(o["ids"])
    a_w = sbuf + fx.Int64(o["w"])
    a_mi = sbuf + fx.Int64(o["mi"])
    a_dest = sbuf + fx.Int64(o["dest"])
    pad = fx.Int32(b)
    pr = -(-b * TOPK // THREADS)
    p_old = ld_i32(sbuf + fx.Int64(o["nv"]), 1) * TOPK
    old_d = [ld_i32(a_dest, fx.min(tid + k * THREADS, b * TOPK - 1)) for k in range(pr)]
    for k in range_constexpr(pr):
        if tid + k * THREADS < p_old:
            bo.buffer_store(pad, rsrc(a_ids), old_d[k])
            bo.buffer_store(fx.Float32(0.0), rsrc(a_w), old_d[k])
            bo.buffer_store(pad, rsrc(a_mi), old_d[k])
    # the resets land before any of this launch's rows
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


@flyc.jit
def sort_rows(tid, m, ly, flags, fidx, target, ids, wts, sbuf, masks, bbl, scan):
    """One CTA (``sort_reset`` earlier): once routing flags ``fidx`` .. + b of ``flags``
    reach ``target``, the sort of the first ``m`` tokens' pairs of ``ids`` / ``wts``
    [b, TOPK] into ``sbuf``."""
    lane, wave = tid % 64, tid // 64
    b = ly["b"]
    mw = mask_words(b)
    o = ly["offs"]
    a_ids = sbuf + fx.Int64(o["ids"])
    a_w = sbuf + fx.Int64(o["w"])
    a_eids = sbuf + fx.Int64(o["eids"])
    a_nv = sbuf + fx.Int64(o["nv"])
    a_mi = sbuf + fx.Int64(o["mi"])
    a_dest = sbuf + fx.Int64(o["dest"])
    pr = -(-b * TOPK // THREADS)
    p_n = m * TOPK
    for i in range(tid, E * mw, THREADS):
        fx.ptr_store(fx.Int32(0), masks + i)
    # 1. every row routed
    if tid < b:
        _spin(lambda: ld_i32(flags, fidx + tid, CM_DEV), lambda v: (v - target) < 0)
    gpu.barrier()
    # 2. the pairs, an expert's tokens as an mw-word mask (a token picks an expert once)
    es, ws = [], []
    for k in range_constexpr(pr):
        p = fx.min(tid + k * THREADS, b * TOPK - 1)
        es.append(fx.min(fx.max(ld_i32(ids, p, CM_DEV), 0), E - 1))
        ws.append(
            fx.Float32(
                bo.buffer_load(
                    rsrc(wts), p, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV
                )
            )
        )
    for k in range_constexpr(pr):
        p = tid + k * THREADS
        if p < p_n:
            t = p // TOPK
            fx.llvm.atomic_or(
                masks + es[k] * mw + t // 32,
                fx.Int32(1) << (t % 32),
                syncscope="workgroup",
            )
    gpu.barrier()
    # 3. blocks an expert (expert tid), blocks before it
    e = fx.min(tid, E - 1)
    cnt = fx.Int32(0)
    for w in range_constexpr(mw):
        cnt = cnt + popcount(fx.Uint32(fx.ptr_load(masks + e * mw + w)))
    nb = (tid < E).select((cnt + BM - 1) // BM, fx.Int32(0))
    x, tot = block_excl_scan(nb, lane, wave, scan)
    if tid < E:
        fx.ptr_store(x, bbl + tid)
        for k in range(nb):
            bo.buffer_store(fx.Int32(tid), rsrc(a_eids), x + k)
    if tid == 0:
        bo.buffer_store(tot * BM, rsrc(a_nv), 0)
        bo.buffer_store(m, rsrc(a_nv), 1)
    gpu.barrier()
    # 4. a pair's row: its expert's first row + the expert's tokens below t
    for k in range_constexpr(pr):
        p = tid + k * THREADS
        if p < p_n:
            ek = es[k]
            t = p // TOPK
            s = p % TOPK
            rank = fx.Int32(0)
            for w in range_constexpr(mw):
                word = fx.ptr_load(masks + ek * mw + w)
                lo = t - 32 * w
                keep = (lo >= 32).select(
                    fx.Int32(-1),
                    (lo <= 0).select(
                        fx.Int32(0), (fx.Int32(1) << fx.max(lo, 0)) - fx.Int32(1)
                    ),
                )
                rank = rank + popcount(fx.Uint32(word & keep))
            dest = fx.ptr_load(bbl + ek) * BM + rank
            bo.buffer_store((s << 24) | t, rsrc(a_ids), dest)
            bo.buffer_store(ws[k], rsrc(a_w), dest)
            bo.buffer_store(t, rsrc(a_mi), dest)
            bo.buffer_store(dest, rsrc(a_dest), p)
