# SPDX-License-Identifier: Apache-2.0
"""Muse Spark's gfx1250 device helpers that ``sgl_amd_mono.common`` does not hold:
the step's mailbox tag, the fences of a plain-words hand-off and the async
global -> LDS copies (gfx1250 counts stores apart from loads, and its LDS copies on
their own counter, ``s_wait_asynccnt``)."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.ops import CM_DEV, clock, rsrc, uniform, wait_stores
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.muse_spark.gfx1250.config import HIDDEN, LAYER_SLOTS


def step_tag(epoch_ptr, layer):
    """The layer's mailbox tag this step, ``step * LAYER_SLOTS + layer + 1``: the
    step is word 0 of the epoch buffer, which the host moves on after a step's last
    launch. Graph-safe, and no mailbox is cleared between steps."""
    step = uniform(bo.buffer_load(rsrc(epoch_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + layer + 1


def release_stores():
    """Order this wave's earlier device-scope stores before a flag that other CTAs
    poll: wait for them (a workgroup release fence alone does not wait for global
    stores), then the fence."""
    wait_stores()
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="workgroup")


def acquire_loads():
    """Order this thread's later loads after a flag it polled."""
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")


# ---------------------------------------------------------------- stage trace
WTRACE0 = (
    32  # the first per-wave stamp slot (``wstamp``): WTRACE0 + point * WAVES + wave
)
WTRACE_POINTS = 4  # per-wave stamp points a kernel
# the CTAs whose waves stamp (a clock read costs tens of ns and the reads queue); SGLANG_AMD_MONO_WTRACE_BIDS=""
# turns the per-wave stamps off (they delay their CTAs by a few us)
WTRACE_BIDS = tuple(
    int(x)
    for x in os.environ.get("SGLANG_AMD_MONO_WTRACE_BIDS", "8,40,72").split(",")
    if x
)
TRACE_SLOTS = 96  # stamps a CTA: int64 clock ticks (100 MHz) at trace[((layer % TRACE_LAYERS) BLOCKS + bid) TRACE_SLOTS + idx]
TRACE_LAYERS = 64  # a layer's own stamp slots (``c["layer"]``): a server snapshot keeps every layer's last launch
TRACE_BLOCKS = BLOCKS  # CTAs a launch


@flyc.jit
def stamp(c, idx: int):
    """Trace stamp ``idx`` of this CTA, compiled in by a build with ``trace``: the
    100 MHz realtime clock, an int64 in the launch's ``*_trace`` scratch region
    (``c["trace"]``, a pair a stamp), written by thread 0."""
    if const_expr(not c.get("trace_on", False)):
        return
    layer = c.get("layer")
    if const_expr(layer is None):
        base = c["bid"] * TRACE_SLOTS
    else:
        base = (layer % TRACE_LAYERS) * (TRACE_BLOCKS * TRACE_SLOTS) + c[
            "bid"
        ] * TRACE_SLOTS
    if c["tid"] == 0:
        now = clock()
        addr = c["trace"].value + fx.Int64((base + idx) * 8)
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
        llvm.StoreOp(as_ir_value(now), ptr, alignment=8)


@flyc.jit
def wstamp(c, point: int):
    """Per-wave trace stamp ``point`` (0 .. WTRACE_POINTS - 1) of the ``WTRACE_BIDS`` CTAs: every wave's lane 0
    stores the clock at slot WTRACE0 + point * WAVES + wave (``stamp``'s region and build switch)."""
    if const_expr(not c.get("trace_on", False) or not WTRACE_BIDS):
        return
    assert 0 <= point < WTRACE_POINTS
    layer = c.get("layer")
    if const_expr(layer is None):
        base = c["bid"] * TRACE_SLOTS
    else:
        base = (layer % TRACE_LAYERS) * (TRACE_BLOCKS * TRACE_SLOTS) + c[
            "bid"
        ] * TRACE_SLOTS
    bid = c["bid"]
    hit = bid == WTRACE_BIDS[0]
    for b in WTRACE_BIDS[1:]:
        hit = hit | (bid == b)
    if hit & (c["lane"] == 0):
        now = clock()
        addr = c["trace"].value + fx.Int64(
            (base + WTRACE0 + point * WAVES + c["wave"]) * 8
        )
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
        llvm.StoreOp(as_ir_value(now), ptr, alignment=8)


# SGLANG_AMD_MONO_OG_ONCE (default 1; a merged launch): the attention's merge writes the normed, gated o rows (the
# o projection's operand) where it would write o; the o tasks copy them into LDS instead of every CTA rebuilding
# them from o and the gate, a round a token pair. The same values (the expressions and their order are the og
# rows'). 0: the first form.
OG_ONCE = os.environ.get("SGLANG_AMD_MONO_OG_ONCE", "1") == "1"


def _push_set():
    """SGLANG_AMD_MONO_PUSH_ALL: the exchanges whose pushes go out from every wave of the CTA, a wave to one
    rank (``wave_rank``): "o" (the attention all-reduce, the default), "pred" (the pooled slices), "fin" (the
    experts' all-reduce), "unp" (the unpooled rows); "1" / "all": every one, "0": none (the first forms: a thread
    stores its values to the ranks in turn)."""
    v = os.environ.get("SGLANG_AMD_MONO_PUSH_ALL", "o")
    if v in ("1", "all"):
        return frozenset(("o", "pred", "fin", "unp"))
    if v in ("0", ""):
        return frozenset()
    out = frozenset(x for x in v.split(",") if x)
    assert out <= {"o", "pred", "fin", "unp"}, v
    return out


PUSH_SET = _push_set()
# SGLANG_AMD_MONO_ATTN_CM (default 1; 0: the first layout): the attention all-reduce's region CTA-major,
# [source][o CTA][token][the CTA's task slot][16 columns], instead of [source][token][column]. A remote store
# instruction is fast when its lanes fill whole 128 B blocks and several times slower when it writes a block in
# part. An o task's rows of a token are 64 B and their column neighbours are other CTAs' tasks; CTA-major, a CTA's
# task slots of a token are 384 contiguous bytes (a round's four slots 256 B, its last two 128 B).
ATTN_CM = os.environ.get("SGLANG_AMD_MONO_ATTN_CM", "1") == "1"
ATTN_SLOTS = -(
    -(HIDDEN // 16) // BLOCKS
)  # o tasks (16 columns) a CTA: post_attn.O_PER_CTA
ATTN_COLS = (
    BLOCKS * ATTN_SLOTS * 16 if ATTN_CM else HIDDEN
)  # columns a (source, token) of ``attn``


def wave_rank(c, npes: int):
    """The rank this thread's wave stores to in an all-waves push (THREADS / npes threads a rank: whole waves),
    as a wave-uniform scalar, and the thread's index among the rank's threads."""
    tpd = THREADS // npes
    assert THREADS % npes == 0 and tpd % WAVE == 0, npes
    return uniform(c["tid"] // tpd), c["tid"] % tpd, tpd


def rank_base(bases, dest):
    """``bases[dest]`` (Int64 addresses) for the wave-uniform scalar ``dest`` by scalar selects: the store's
    resource stays in SGPRs (a per-lane select makes LLVM loop over the lanes' resources)."""
    b = fx.Int64(bases[0])
    for p in range(1, len(bases)):
        b = (dest == p).select(fx.Int64(bases[p]), b)
    return b


FLAG_STRIDE = 8  # pairs between a broadcast flag and the next: a 64 B line each (every CTA polls them)


@flyc.jit
def poll_flags(c, regions, first=0, sync=True):
    """Wait for the flags ``regions`` = [(region, n) | (region, base, n) | (region,
    base, n, stride), ...] (flags base, base + stride, .. of each, n of them), one
    flag a thread, then a barrier and an acquire: one round trip whatever the count
    (a thread polling every flag takes a round trip per ``POLL_MAX`` specs). Every
    thread of the CTA must call it (the barrier)."""
    tid = c["tid"]
    off = first  # the first polling thread (``first`` > 0: off wave 0, which works meanwhile)
    for spec in regions:
        region, base, n, stride = (
            spec
            if len(spec) == 4
            else ((*spec, 1) if len(spec) == 3 else (spec[0], 0, spec[1], 1))
        )
        if (tid >= off) & (tid < off + n):
            c["poll"]([(region, base + (tid - off) * stride, 1)])
        off += n
    if isinstance(off, int):
        assert off <= THREADS, off
    if const_expr(sync):
        # (``sync`` False: the caller's next barrier and ``acquire_loads`` close the wait)
        gpu.barrier()
        acquire_loads()


def lds_dma_16(addr, lds_ptr, voffset, policy: int = 0):
    """``global_load_async_to_lds_b128``: 16 B a lane from global address ``addr`` +
    byte ``voffset`` (per lane) into LDS at ``lds_ptr`` (wave-uniform) + 16 lane. The
    compiler does not count these copies: the caller waits ``wait_lds_dma`` (they
    complete in issue order) before anything reads the LDS. ``policy``: a cache policy
    (``CM_DEV`` for data another CTA stored in this launch)."""
    _async_copy(addr, lds_ptr, voffset, 16, policy)


def lds_dma_4(addr, lds_ptr, voffset, policy: int = 0):
    """``global_load_async_to_lds_b32``: 4 B a lane, ``lds_dma_16``'s rules."""
    _async_copy(addr, lds_ptr, voffset, 4, policy)


def _async_copy(addr, lds_ptr, voffset, nbytes, policy):
    lane = fx.Int32(fx.thread_idx.x) % 32
    g = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"),
        as_ir_value(fx.Int64(addr) + fx.Int64(fx.Int32(voffset))),
    ).result
    base = fx.Int32(llvm.ptrtoint(T.i32, lds_ptr.llvm_ptr))
    lp = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<3>"), as_ir_value(base + lane * nbytes)
    ).result
    op = (
        rocdl.global_load_async_to_lds_b128
        if nbytes == 16
        else rocdl.global_load_async_to_lds_b32
    )
    op(g, lp, 0, aux=_aux(policy))


def _aux(policy: int):
    """The async copy's cache policy as the op's attribute (None: the default)."""
    return ir.IntegerAttr.get(T.i32, policy) if policy else None


def async_copy16(gaddr, lds_addr, policy: int = 0):
    """``global_load_async_to_lds_b128`` with this lane's own global address (Int64)
    and LDS byte address (Int32): any per-lane placement, e.g. a swizzle. Counted by
    ``wait_lds_dma``."""
    g = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(gaddr))
    ).result
    lp = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<3>"), as_ir_value(fx.Int32(lds_addr))
    ).result
    rocdl.global_load_async_to_lds_b128(g, lp, 0, aux=_aux(policy))


def async_copy4(gaddr, lds_addr, policy: int = 0):
    """``global_load_async_to_lds_b32`` with this lane's own global address (Int64) and LDS byte
    address (Int32), ``async_copy16``'s rules."""
    g = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(gaddr))
    ).result
    lp = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<3>"), as_ir_value(fx.Int32(lds_addr))
    ).result
    rocdl.global_load_async_to_lds_b32(g, lp, 0, aux=_aux(policy))


def async_copy16_at(lds_reg, g64, off: int):
    """``global_load_async_to_lds_b128`` as written: 16 B a lane, LDS[``lds_reg`` + ``off``] <- memory[``g64`` +
    ``off``] (the instruction adds its one offset field to both addresses; ``lds_reg`` an i32 and ``g64`` an i64
    lane value, taken as they are: nothing is rebuilt around the copy). Counted by ``wait_lds_dma``."""
    assert 0 <= off < (1 << 20), off
    llvm.inline_asm(
        None,
        [as_ir_value(fx.Int32(lds_reg)), as_ir_value(fx.Int64(g64))],
        f"global_load_async_to_lds_b128 $0, $1, off offset:{off}",
        "v,v",
        has_side_effects=True,
    )


def async_copy4_at(lds_reg, g64, off: int):
    """``global_load_async_to_lds_b32`` as written: 4 B a lane, ``async_copy16_at``'s rules."""
    assert 0 <= off < (1 << 20), off
    llvm.inline_asm(
        None,
        [as_ir_value(fx.Int32(lds_reg)), as_ir_value(fx.Int64(g64))],
        f"global_load_async_to_lds_b32 $0, $1, off offset:{off}",
        "v,v",
        has_side_effects=True,
    )


def wait_xcnt():
    """``s_wait_xcnt 0``: every memory access this wave issued has had its address taken. The compiler puts this
    wait before it overwrites a register an access it issued still reads (a replayed access reads it again); it
    does not see the raw copies (``async_copy16_at``), so the code that renews or drops their operand registers
    waits here."""
    llvm.inline_asm(None, [], "s_wait_xcnt 0x0", "", has_side_effects=True)


def pin_add(v, const: int):
    """``v`` + ``const`` as one opaque instruction (an i32 lane value in a register of its own)."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [as_ir_value(fx.Int32(v))],
            f"v_add_nc_u32 $0, {const & 0xFFFFFFFF:#x}, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def pin64(v):
    """An i64 lane value pinned in a register pair (an opaque move a half: the compiler cannot re-derive it from a
    shared base, so a copy that names it rebuilds nothing)."""
    v = fx.Int64(v)
    lo = llvm.inline_asm(
        T.i32,
        [as_ir_value(fx.Int32(v))],
        "v_mov_b32 $0, $1",
        "=v,v",
        has_side_effects=True,
    )
    hi = llvm.inline_asm(
        T.i32,
        [as_ir_value(fx.Int32(v >> 32))],
        "v_mov_b32 $0, $1",
        "=v,v",
        has_side_effects=True,
    )
    return (fx.Int64(fx.Int32(hi)) << 32) | fx.Int64(fx.Uint32(fx.Int32(lo)))


def gstore_raw(addr, value, align: int):
    """A plain store of ``value`` (a scalar or a vector of any element type: 2 B for a bf16) at the 64-bit global
    address ``addr`` (Int64), ``align`` bytes aligned."""
    ptr = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))
    ).result
    llvm.StoreOp(as_ir_value(value), ptr, alignment=align)


PF_LINE = 128  # bytes one prefetch lane covers (a GL2 line)


def l2_prefetch(gaddr):
    """``global_prefetch_b8`` (``llvm.prefetch``, locality 1 = SCOPE_DEV): the
    ``PF_LINE`` line at this lane's global address ``gaddr`` (Int64) into GL2 (and the
    memory-side cache on its way). No data comes back and no counter moves, so no
    later wait sees it; keep the address inside the buffer."""
    g = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(gaddr))
    ).result
    llvm.intr_prefetch(g, 0, 1, 1)


def l2_prefetch_gvs(base, lane_off, imm: int = 0):
    """``l2_prefetch`` of ``base`` + ``lane_off`` + ``imm`` in the instruction's SGPR-base form: ``base`` wave-uniform
    (pass it through ``uniform64``: SGPRs), ``lane_off`` this lane's unsigned Int32, computed once and kept unchanged
    over a run of prefetches, ``imm`` a constant (the offset field). Such a run rewrites no address register, so
    the XNACK-replay rule (``s_wait_xcnt`` before a register an untranslated access reads is overwritten) puts no
    wait between them; a per-prefetch 64-bit VGPR address gets one wait each."""
    l2_prefetch(base + fx.Int64(fx.Uint32(lane_off)) + fx.Int64(imm))


def wave_prefetch(gaddr, nbytes: int, lane):
    """This wave's lanes prefetch the ``nbytes`` (a Python int) at ``gaddr`` into GL2, a
    ``PF_LINE`` line a lane and instruction; lanes past the end take the last line
    again (no traced branch)."""
    lines = -(-nbytes // PF_LINE)
    for i in range(-(-lines // WAVE)):
        line = i * WAVE + lane
        if (i + 1) * WAVE > lines:
            line = fx.min(line, fx.Int32(lines - 1))
        l2_prefetch(fx.Int64(gaddr) + fx.Int64(line * PF_LINE))


def lds_load4(lds_addr):
    """The i32 at LDS byte address ``lds_addr`` (volatile, as ``lds_load16``)."""
    p = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<3>"), as_ir_value(fx.Int32(lds_addr))
    ).result
    return fx.Int32(llvm.LoadOp(T.i32, p, alignment=4, volatile_=True).result)


def lds_load16(lds_addr):
    """16 B (4 x i32) at LDS byte address ``lds_addr`` (volatile: it reads what async
    copies wrote, which the compiler does not see)."""
    p = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<3>"), as_ir_value(fx.Int32(lds_addr))
    ).result
    return fx.Vector(
        llvm.LoadOp(T.vec(4, T.i32), p, alignment=16, volatile_=True).result
    )


def lds_addr(ptr):
    """The LDS byte address (Int32) of an LDS pointer."""
    return fx.Int32(llvm.ptrtoint(T.i32, ptr.llvm_ptr))


def wait_lds_dma(n: int = 0):
    """Wait until at most ``n`` of this wave's async LDS copies are in flight."""
    rocdl.s_wait_asynccnt(n)


DMA_DEV = CM_DEV  # ``lds_dma_16``'s policy for data stored in this launch


def global_atomic(addr, val, xchg=False):
    """Device-scope atomic add (or exchange) of ``val`` on the int32 at global
    address ``addr`` (Int64); the word's previous value."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.xchg if xchg else llvm.AtomicBinOp.add, ptr, as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic, syncscope="agent",
    )  # fmt: skip
    return fx.Int32(op.result)


def lds_atomic_add(ptr, val):
    """Atomic add of ``val`` to the LDS word at ``ptr`` (workgroup scope); the word's
    previous value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr.llvm_ptr,
        as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
    )
    return fx.Int32(op.result)
