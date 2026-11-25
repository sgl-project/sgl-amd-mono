# SPDX-License-Identifier: Apache-2.0
"""NVFP4 rows as SGLang's MiniMax-M3 pool stores them (``MiniMaxM3Nvfp4KVPool``, MSA-v1):
a 128-column row is 64 payload bytes, two e2m1 codes a byte with the even column in the
low nibble, and 8 E4M3 scales, one a 16-column group, under a unit outer scale. A wave
holds a row, two columns a lane. The arithmetic is ``_nvfp4_quarter`` of
``sglang/kernels/ops/attention/minimax_m3_qk_norm_rope.py``: amax / 6.0 rounded to E4M3,
the codes of x / scale rounded half to even and saturated at 6."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import Int64, T

from sgl_amd_mono.common.ops import bf_hi, bf_lo, fp8_roundtrip, ld_i32, rsrc, xshfl
from sgl_amd_mono.common.plan import THREADS, WAVES, kernel_sources, key_tuple

SOURCES = kernel_sources(__name__)

ROW = 128
PAYLOAD_WORDS = ROW // 8  # 4 bytes a word, 2 columns a byte
SCALE_WORDS = ROW // 64  # one E4M3 byte a 16-column group
# e2m1 magnitudes 0.5, 1, 1.5, 2, 3, 4, 6: each edge counts one code up; a tie goes to the
# even code, so the edges below an even code are inclusive
_EDGES = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)


def e2m1_code(y):
    """The 4-bit e2m1 code of f32 ``y`` (sign in bit 3; -0.0 and NaN code 0)."""
    a = fmath.absf(y)
    mag = fx.Int32(0)
    for edge, inclusive in _EDGES:
        hit = (a >= edge) if inclusive else (a > edge)
        mag = mag + hit.select(fx.Int32(1), fx.Int32(0))
    return mag | (y < 0.0).select(fx.Int32(8), fx.Int32(0))


def nvfp4_row(lo, hi):
    """A lane's columns 2 * lane and 2 * lane + 1 of a row -> (payload word, scale word).
    Lane 4 k holds payload word k; lanes 0 and 32 hold scale words 0 and 1."""
    amax = fx.max(fmath.absf(lo), fmath.absf(hi))
    for off in (1, 2, 4):  # the 8 lanes of a 16-column group
        amax = fx.max(amax, xshfl(amax, off))
    scale = amax / 6.0
    scale_bits = fx.Int32(
        rocdl.cvt_pk_fp8_f32(T.i32, scale, fx.Float32(0.0), fx.Int32(0), False)
    )
    scale_bits = scale_bits & 0xFF
    scale_f32, _ = fp8_roundtrip(scale, fx.Float32(0.0))
    inv = (amax != 0.0).select(1.0 / scale_f32, fx.Float32(0.0))
    byte = e2m1_code(lo * inv) | (e2m1_code(hi * inv) << 4)
    b1 = xshfl(byte, 1)
    payload = byte | (b1 << 8) | (xshfl(byte, 2) << 16) | (xshfl(b1, 2) << 24)
    s8 = xshfl(scale_bits, 8)
    scales = (
        scale_bits | (s8 << 8) | (xshfl(scale_bits, 16) << 16) | (xshfl(s8, 16) << 24)
    )
    return payload, scales


def build_rows(rows: int):
    """A launch quantizing ``rows`` contiguous bf16 rows of 128 columns (tests, probes)."""
    keyed = key_tuple(locals(), SOURCES)
    blocks = -(-rows // WAVES)

    @flyc.kernel(name=f"mono_m31_nvfp4_rows_{rows}", known_block_size=[THREADS, 1, 1])
    def k(x: Int64, payload: Int64, scales: Int64):
        _ = keyed
        lane = fx.thread_idx.x & 63
        row = fx.block_idx.x * WAVES + (fx.thread_idx.x >> 6)
        if row < rows:
            w = ld_i32(x, row * (ROW // 2) + lane)
            pw, sw = nvfp4_row(bf_lo(w), bf_hi(w))
            if (lane & 3) == 0:
                bo.buffer_store(pw, rsrc(payload), row * PAYLOAD_WORDS + (lane >> 2))
            if (lane & 31) == 0:
                bo.buffer_store(sw, rsrc(scales), row * SCALE_WORDS + (lane >> 5))

    @flyc.jit
    def launch(
        x: Int64, payload: Int64, scales: Int64, stream: fx.Stream = fx.Stream(None)
    ):
        _ = keyed
        k(x, payload, scales).launch(grid=(blocks,), block=(THREADS,), stream=stream)

    return launch
