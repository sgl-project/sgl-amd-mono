# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm at 910ca119ca (vllm-project/vllm#63032, Apache-2.0):
# vllm/models/qwen4_exp/amd/specialized_decode/mono/common/ops.py, itself adapted from
# ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
"""The device primitives these kernels use that ``sgl_amd_mono.common.ops`` does not
have: the wave sum and maximum (``common.ops.wave_max`` is
``fx.coop.warp_reduce``, other code), the scans and ``traced``.

Ported from the FlyDSL GLM-5 layer kernel (ROCm/FlyDSL branch
``glm5-mla-moe-layer-monokernel`` @ 1e28fd8a,
``kernels/mla_moe_layer/shared_reuse_moe_kernel.py``)."""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr

from sgl_amd_mono.common.ops import butterfly, permlane_swap, xshfl
from sgl_amd_mono.common.plan import WAVES


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


def traced(fn):
    """Apply flydsl's AST rewriting to a helper, so its ``while`` / ``if`` over
    traced values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


def _xpartner(v, off, lane):
    """Lane ``lane ^ off``'s v: DPP within a row, v_permlane*_swap across rows
    (``xshfl`` takes ``shuffle_xor`` there)."""
    if off < 16:
        return xshfl(v, off)
    a, b = permlane_swap(off, v, v)
    return ((lane & off) != 0).select(a, b)


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16, 32):
        o = _xpartner(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 63:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot
