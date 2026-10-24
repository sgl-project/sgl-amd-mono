# SPDX-License-Identifier: Apache-2.0
"""Per-channel FP8 GEMV for the AttnFP8 projections.

Weight rows are contiguous E4M3 (the ``[K, N]`` stride ``(1, K)`` view).
Quark's HIP path stores that view as ``shuffle_weight(w_nk, (16, 16)).t()``:
logical ``(n, k)`` is at byte

    ((n // 16) * (K // 32) + k // 32) * 512
    + ((k // 16) % 2) * 256 + (n % 16) * 16 + (k % 16)

of the underlying buffer. Activations are dynamic per-token FP8
(``scale = amax / 448``). The MFMA runs at unit E8M0 scale and the fp32
scales are applied after the split-K sum:

    y[m, n] = bf16(dot(x_fp8[m], w_fp8[n]) * x_scale[m] * w_scale[n])
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.mx import UNIT_SCALE, mfma_scaled
from sgl_amd_mono.common.ops import bf_hi, bf_lo, fp8_pack4, rsrc
from sgl_amd_mono.common.plan import THREADS, WAVES
from sgl_amd_mono.qwen35_a17b_attnfp8.gfx950._k3 import LDS_PAD

FP8_MAX = 448.0


def x8_row(k):
    """Dwords of one token's packed E4M3 row, plus the MFMA tail pad."""
    return k // 4 + 4


def bf_row(k):
    """Bf16 elements of the pre-quant row, matching ``gemv``'s LDS pad."""
    return k + LDS_PAD


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def _load4(r_w, dword):
    return fx.Vector(
        bo.buffer_load(r_w, dword, vec_width=4, dtype=T.i32, cache_modifier=0)
    )


def fp8_loads(lane, wave, w, row0, k, shuffled=False):
    """This wave's E4M3 tiles for output rows ``row0`` .. + 16, K = ``k``.

    A step is 128 K. Wave ``w`` owns steps ``w, w + WAVES, ...``. Each step is
    two 64-byte chunks (``lo`` at the step base, ``hi`` 64 bytes later), which
    is the operand layout of ``mfma_scale_f32_16x16x128_f8f6f4``. ``shuffled``
    gathers those bytes through Quark's ``(16, 16)`` preshuffle.
    """
    steps = k // 128
    per = steps // WAVES
    r_w = rsrc(w)
    j = lane // 16
    out = []
    if shuffled:
        bn = lane % 16
        n0 = row0 // 16
        groups = k // 32
        for i in range(per):
            st = wave + WAVES * i
            # lo: K offset st*128 + j*16; hi: 64 elements later. Each is one
            # 16-byte chunk, contiguous in the preshuffle.
            k_lo = st * 128 + j * 16
            k_hi = k_lo + 64
            k0_lo = k_lo // 32
            k0_hi = k_hi // 32
            sub = (k_lo // 16) % 2
            byte_lo = ((n0 * groups + k0_lo) * 2 + sub) * 256 + bn * 16
            byte_hi = ((n0 * groups + k0_hi) * 2 + sub) * 256 + bn * 16
            out.append((_load4(r_w, byte_lo // 4), _load4(r_w, byte_hi // 4)))
        return out
    row = row0 + lane % 16
    for i in range(per):
        st = wave + WAVES * i
        base = row * (k // 4) + st * 32
        out.append((_load4(r_w, base + j * 4), _load4(r_w, base + 16 + j * 4)))
    return out


def fp8_mfmas(lane, wave, xl, xrow, k, s, ops):
    """``fp8_loads`` against the packed E4M3 rows at ``xl`` (dword stride ``xrow``)."""
    steps = k // 128
    per = steps // WAVES
    t = fx.min(lane % 16, s - 1)
    j = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    unit = fx.Int32(UNIT_SCALE)
    for i in range(per):
        st = wave + WAVES * i
        lo, hi = ops[i]
        av = fx.Vector.from_elements(
            [lo[d] for d in range(4)] + [hi[d] for d in range(4)], fx.Int32
        )
        at = t * xrow + st * 32
        xlo = fx.Vector(
            fx.ptr_load(xl + at + j * 4, result_type=fx.Vector.make_type(4, fx.Int32))
        )
        xhi = fx.Vector(
            fx.ptr_load(
                xl + at + 16 + j * 4, result_type=fx.Vector.make_type(4, fx.Int32)
            )
        )
        xv = fx.Vector.from_elements(
            [xlo[d] for d in range(4)] + [xhi[d] for d in range(4)], fx.Int32
        )
        acc = mfma_scaled(av, xv, acc, unit, unit)
    return acc


@flyc.jit
def quant_pc(c, k):
    """Bf16 rows in ``c['xl']`` -> per-token E4M3 in ``c['x8']`` and the token
    scale ``amax / 448`` in ``c['xsc']``. A zero row stores scale 0 and codes 0.
    Every thread participates: the reduction barriers are grid-uniform per CTA.
    """
    s, tid = c["S"], c["tid"]
    xrow = bf_row(k)
    dst_row = x8_row(k)
    nwords = k // 4
    iters = (nwords + THREADS - 1) // THREADS
    for t in range_constexpr(s):
        amax = fx.Float32(0.0)
        for i in range_constexpr(iters):
            dword = tid + THREADS * i
            if dword < nwords:
                base = dword * 4
                w0 = fx.Int32(fx.ptr_load(c["xl"] + (t * xrow + base) // 2))
                w1 = fx.Int32(fx.ptr_load(c["xl"] + (t * xrow + base + 2) // 2))
                for v in (bf_lo(w0), bf_hi(w0), bf_lo(w1), bf_hi(w1)):
                    amax = fx.max(amax, fx.max(v, fx.Float32(0.0) - v))
        fx.ptr_store(amax, c["red"] + tid)
        gpu.barrier()
        for stride in (256, 128, 64, 32, 16, 8, 4, 2, 1):
            if tid < stride:
                a = fx.ptr_load(c["red"] + tid)
                b = fx.ptr_load(c["red"] + (tid + stride))
                fx.ptr_store(fx.max(a, b), c["red"] + tid)
            gpu.barrier()
        amax = fx.ptr_load(c["red"])
        positive = amax > fx.Float32(0.0)
        scale = positive.select(amax / fx.Float32(FP8_MAX), fx.Float32(0.0))
        inv = positive.select(fx.Float32(FP8_MAX) / amax, fx.Float32(0.0))
        if tid == 0:
            fx.ptr_store(scale, c["xsc"] + t)
        gpu.barrier()
        for i in range_constexpr(iters):
            dword = tid + THREADS * i
            if dword < nwords:
                base = dword * 4
                w0 = fx.Int32(fx.ptr_load(c["xl"] + (t * xrow + base) // 2))
                w1 = fx.Int32(fx.ptr_load(c["xl"] + (t * xrow + base + 2) // 2))
                packed = fp8_pack4(
                    bf_lo(w0) * inv,
                    bf_hi(w0) * inv,
                    bf_lo(w1) * inv,
                    bf_hi(w1) * inv,
                )
                fx.ptr_store(packed, c["x8"] + (t * dst_row + dword))
        gpu.barrier()
