# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's device helpers that ``gfx950/common`` does not hold."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.ops import butterfly, rsrc, uniform, wave_max
from sgl_amd_mono.common.plan import WAVES
from sgl_amd_mono.minimax_m3.gfx950.config import LAYER_SLOTS


def readlane(v, src_lane):
    """Lane ``src_lane``'s value of v (wave-uniform)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def unlikely(cond):
    """``cond`` marked cold (llvm.expect), so LLVM lays the branch out of the hot path:
    a rare path's code inline among the hot stages costs them even when it never runs."""
    return fx.Boolean(
        llvm.call_intrinsic(
            fx.Boolean.ir_type,
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


@flyc.jit
def block_max(v, lane, wave, red):
    """Max of v over the CTA's WAVES waves; ``red`` needs WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t


def step_tag(step_ptr, layer):
    """The layer's mailbox tag this step, ``step * LAYER_SLOTS + layer + 1`` with
    ``step`` the runner's device counter: graph-safe, and no mailbox is cleared
    between steps."""
    step = uniform(bo.buffer_load(rsrc(step_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + layer + 1


def release_stores():
    """Order this wave's earlier device-scope (SC1) stores before a flag other CTAs poll.

    A workgroup-scope release fence does not wait for global stores, so the flag could
    land first and a consumer read stale data; the stores write through to L2, so their
    completion is all it must wait for."""
    rocdl.s_waitcnt(vmcnt=0)
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
    )


def ar_pack_sumsq(vals):
    """One logical thread of the custom all-reduce epilogue: its 8 elements'
    squares accumulated in order (hipcc contracts ``acc += v * v`` into fma)."""
    acc = fx.Float32(0.0)
    for v in vals:
        acc = fmath.fma(v, v, acc)
    return acc
