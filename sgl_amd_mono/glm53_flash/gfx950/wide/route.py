# SPDX-License-Identifier: Apache-2.0
"""The wide launch's router and expert routing (GLM-5.3-Flash, 288 routed experts and
the fused shared expert).

    router  split-K tiles of logits = x_f @ W_r^T, each K slice's f32 partials stored
            to its own [rows, N_EXPERTS] plane (no atomics: every rank must route alike)
    route   a row: the slices summed in a fixed order, sigmoid scores, top-8 of score +
            the f32 bias rounded to bf16 (AITER's tie order), weights score / sum *
            ROUTE_SCALE; slot TOP_K is the shared expert

The logits stay f32 (SGLang's split-K decode router above 16 rows; the narrow kernel's
small-M router rounds them to bf16).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    bpermute_i32,
    hw_exp,
    hw_rcp,
    rsrc,
    wave_umax,
    write_lane_i32,
    xred,
)
from sgl_amd_mono.common.plan import WAVES
from sgl_amd_mono.glm53_flash.gfx950.config import (
    MOE_SLOTS,
    N_EXPERTS,
    ROUTE_SCALE,
    ROUTER_CANDIDATES,
    SHARED_EXPERT,
    TOP_K,
)
from sgl_amd_mono.glm53_flash.gfx950.kernel import ID_BITS, RANK_BASE, SORT5, tie_rank
from sgl_amd_mono.glm53_flash.gfx950.wide.m1a import Gemm

# router [rows, 4096] x [288, 4096]^T: 64-column tiles (the last half empty), 16 K slices
RGATE = {
    32: Gemm(32, 64, 2, 4, 16, 64, 4),
    64: Gemm(64, 64, 2, 4, 16, 64, 4),
    128: Gemm(128, 64, 4, 2, 16, 64, 4),
    192: Gemm(96, 64, 2, 4, 16, 64, 4),
}
CAND = ROUTER_CANDIDATES
RED_WORDS = WAVES * CAND * 64


def rgate_cfg(rows):
    return RGATE[min(w for w in RGATE if w >= rows)]


def n_tiles(g):
    return -(-N_EXPERTS // g.bn)


def part_emit(parts, rows, sk, n0):
    """``emit`` storing K slice ``sk``'s f32 partials (device-coherent) to its plane."""
    plane = rows * N_EXPERTS
    r_o = rsrc(parts, (sk + 1) * plane * 4)

    def emit(row, col, v, live):
        ok = (row < rows) & (n0 + col < N_EXPERTS)
        bo.buffer_store(
            v,
            r_o,
            sk * plane + row * N_EXPERTS + n0 + col,
            mask=ok,
            cache_modifier=CM_DEV,
        )

    return emit


def _top8(lane, raws, bs):
    """``route_top8`` of the narrow kernel on this lane's candidate scores ``raws``
    (experts lane + 64 i) and bf16 biases ``bs``: (expert id, weight) of pick ``lane``,
    valid in lanes < TOP_K."""
    ks, ids, rks = [], [], []
    for i in range_constexpr(CAND):
        e = lane + 64 * i
        kb = (raws[i] + bs[i]).bitcast(fx.Int32)
        ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
        valid = e < N_EXPERTS
        ks.append(valid.select(fx.Uint32(ok), fx.Uint32(0)))
        ids.append(valid.select(e, fx.Int32(N_EXPERTS)))
        rks.append(valid.select(tie_rank(e), fx.Int32(RANK_BASE)))
    for a, b in SORT5:
        first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (rks[a] < rks[b]))
        ka, kb, ia, ib, ra, rb = ks[a], ks[b], ids[a], ids[b], rks[a], rks[b]
        ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
        ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
        rks[a], rks[b] = first.select(ra, rb), first.select(rb, ra)
    ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
    ids = ids + [fx.Int32(N_EXPERTS)]
    rks = rks + [fx.Int32(RANK_BASE)]
    mv = fx.Int32(0)
    for k in range_constexpr(TOP_K):
        m = wave_umax(ks[0])
        winner = wave_umax(
            (ks[0] == m).select(((RANK_BASE - rks[0]) << ID_BITS) | ids[0], fx.Int32(0))
        )
        expert = winner & fx.Int32((1 << ID_BITS) - 1)
        hit = (ks[0] == m) & (ids[0] == expert)
        ks = [hit.select(ks[i + 1], ks[i]) for i in range(CAND)] + [ks[CAND]]
        ids = [hit.select(ids[i + 1], ids[i]) for i in range(CAND)] + [ids[CAND]]
        rks = [hit.select(rks[i + 1], rks[i]) for i in range(CAND)] + [rks[CAND]]
        mv = write_lane_i32(expert, k, mv)
    e = mv
    src = (e % 64) * 4
    got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
    raw = got[0]
    for i in range_constexpr(1, CAND):
        raw = (e // 64 == i).select(got[i], raw)
    raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
    tot = raw
    for off in (1, 2, 4):
        tot = xred(tot, off, lambda a, b: a + b)
    return e, raw * (hw_rcp(tot) * ROUTE_SCALE)


@flyc.jit
def route_row(tid, t, rows, ks, parts, bias, ids_out, w_out, logits, red, shared_w):
    """Row ``t`` (its partials complete): wave w sums K slices w, w + WAVES, .. of its
    lanes' candidates, wave 0 the waves in order (stored to ``logits`` [rows,
    N_EXPERTS] f32, for SGLang's own top-k), then the top-8."""
    lane, wave = tid % 64, tid // 64
    plane = rows * N_EXPERTS
    r_p = rsrc(parts, ks * plane * 4)
    per = ks // WAVES
    cand = [fx.min(lane + 64 * i, N_EXPERTS - 1) for i in range(CAND)]
    got = [
        [
            fx.Float32(
                bo.buffer_load(
                    r_p,
                    (wave + j * WAVES) * plane + t * N_EXPERTS + cand[i],
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            for j in range(per)
        ]
        for i in range(CAND)
    ]
    for i in range_constexpr(CAND):
        s = got[i][0]
        for j in range_constexpr(1, per):
            s = s + got[i][j]
        fx.ptr_store(s, red + ((wave * CAND + i) * 64 + lane))
    gpu.barrier()
    if wave == 0:
        r_b = rsrc(bias)
        # the f32 bias as SGLang's grouped top-k holds it: rounded to bf16
        bs = [
            fx.Float32(
                fx.Float32(bo.buffer_load(r_b, cand[i], vec_width=1, dtype=T.f32)).to(
                    fx.BFloat16
                )
            )
            for i in range(CAND)
        ]
        raws = []
        r_l = rsrc(logits, rows * N_EXPERTS * 4)
        for i in range_constexpr(CAND):
            logit = fx.ptr_load(red + i * 64 + lane)
            for w in range_constexpr(1, WAVES):
                logit = logit + fx.ptr_load(red + ((w * CAND + i) * 64 + lane))
            bo.buffer_store(
                logit,
                r_l,
                t * N_EXPERTS + lane + 64 * i,
                mask=lane + 64 * i < N_EXPERTS,
            )
            raws.append(hw_rcp(1.0 + hw_exp(-logit)))
        e, wt = _top8(lane, raws, bs)
        sh = lane == TOP_K
        ok = lane < MOE_SLOTS
        # device-coherent: the launch's sort CTA reads them
        bo.buffer_store(
            sh.select(fx.Int32(SHARED_EXPERT), e),
            rsrc(ids_out, rows * MOE_SLOTS * 4),
            t * MOE_SLOTS + lane,
            mask=ok,
            cache_modifier=CM_DEV,
        )
        bo.buffer_store(
            sh.select(fx.Float32(shared_w), wt),
            rsrc(w_out, rows * MOE_SLOTS * 4),
            t * MOE_SLOTS + lane,
            mask=ok,
            cache_modifier=CM_DEV,
        )
    gpu.barrier()
