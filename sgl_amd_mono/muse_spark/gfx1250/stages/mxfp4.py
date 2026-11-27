# SPDX-License-Identifier: Apache-2.0
"""MX activation quantization on gfx1250, as the normal gfx1250 expert path (aiter's
A8W4 TDM grouped MoE) rounds it: FP8 E4M3 values with one E8M0 scale a 32-value
block, the scale by the RoundUp rule (``emit_mx_e8m0_scale(mode=RoundUp,
dtype=FP8_E4M3)``, the rule of ``moe_fused_route_quant_scatter`` for the experts'
input and of the stage-1 epilogue for the SwiGLU mids).

RoundUp: the block's max |x| times fp32(1 / 448) (bits 0x3B124925); its biased
exponent, plus one when any mantissa bit is set, clamped to [0, 255], is the E8M0
code e; each value times 2^(127 - e) (exact) is rounded to E4M3 half to even
(``v_cvt_pk_fp8_f32``; the scaled values stay within +-448 by construction).

``lanes_before`` counts the lanes of a ballot below a lane (wave32: the low 32 bits).
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import popcount

INV_448_BITS = 0x3B124925  # fp32(1 / 448), the E4M3 max's reciprocal as aiter rounds it


def e8m0_f32(byte):
    """An E8M0 byte (as loaded, any width) -> the f32 power of two it encodes."""
    return ((fx.Int32(byte) & 0xFF) << 23).bitcast(fx.Float32)


def mx8_scale(amax):
    """The MX FP8 (RoundUp) block scale of a block whose max |x| is ``amax`` (fp32,
    >= 0): (the E8M0 code as an Int32, the quantizing factor 2^(127 - e) as fp32)."""
    working = fx.Float32(amax) * fx.Int32(INV_448_BITS).bitcast(fx.Float32)
    wb = working.bitcast(fx.Int32)
    e = ((wb >> 23) & 0xFF) + ((wb & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    e = fx.min(fx.max(e, fx.Int32(0)), fx.Int32(254))
    return e, ((fx.Int32(254) - e) << 23).bitcast(fx.Float32)


def fp8_byte(q):
    """The E4M3 code (an Int32 holding 8 bits) of the scaled value ``q``."""
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False)) & 0xFF


def fp8_word(a, b, c, d):
    """Four scaled values -> one word of four E4M3 codes (a in bits 7:0)."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def lanes_before(bal, lane):
    """How many lanes of ballot ``bal`` (Int64, wave32: the low 32 bits) lie below ``lane``."""
    lo = fx.Int32(bal & fx.Int64(0xFFFFFFFF))
    return popcount(lo & ((fx.Int32(1) << fx.min(lane, 31)) - 1))
