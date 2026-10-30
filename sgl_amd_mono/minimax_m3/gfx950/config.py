# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shapes the MiniMax-M3 mono kernels are compiled for: one rank of a TP group.

The model's own totals are module constants. Everything a rank's share depends
on lives on :class:`Dims`, built once from the TP size the server actually runs
and threaded into the kernel builders.

That split is forced by the framework. SGLang resolves a spec's widths and TP
before it calls ``build`` (``srt/monokernels/runtime.py``), and there is no hook
that could set a module constant in between -- the in-tree ancestor of this file
read TP from ``SGLANG_TEST_MONO_TP`` at import for exactly that reason, which
worked only because a test process could set the variable before the first
import. A plugin has no such moment.
"""

from __future__ import annotations

import os

import msgspec

from sgl_amd_mono.common.plan import padded

HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64

# the model's totals, which the checkpoint's config is checked against: these
# know nothing about TP
TOTAL_Q_HEADS = 64
TOTAL_KV_HEADS = 4
TOTAL_INDEX_HEADS = 4
TOTAL_INTER = 3072

N_ROUTED = 128
TOP_K = 4
SHARED_EXPERT = N_ROUTED  # the fused shared expert's id
MOE_SLOTS = TOP_K + 1

SPARSE_BLOCK = 128
TOPK_BLOCKS = 16
MAX_SPARSE_KEYS = SPARSE_BLOCK * TOPK_BLOCKS
# the longest context served: the score region spans it
MAX_CONTEXT = 1 << 20
MAX_INDEX_BLOCKS = MAX_CONTEXT // SPARSE_BLOCK
PAGE16 = 16

MAX_TOKENS = 16  # tokens one mono step serves (the MFMA B operand holds 16)

# aiter's e8m0_shuffle pads an MXFP4 scale's columns up to a multiple of this
# (measured: 12 -> 16, 24 -> 24, 48 -> 48). At TP8 the w2 scale's 12 columns
# become 16, so anything indexing it must use the padded stride.
SCALE_COL_ALIGN = 8

# indexer CP from here: below it the one-head O(n^2) ranking costs less than CP's fixed overhead
INDEX_CP_FROM_BLOCKS = 288

LAYER_SLOTS = 128  # mailbox epochs: step * LAYER_SLOTS + layer + 1

TP_SIZES = (2, 4, 8)

# The widths a rank can actually run, per TP size, as (low, high) inclusive.
# Both bounds exist, and for different reasons.
#
# The TOP bound is the 160 KB LDS budget, which scales with a rank's head count
# and INTER -- so a *smaller* TP buys *fewer* tokens per step. TP2 carries 32 q
# heads and INTER 1536 against TP4's 16 and 768; measured by COMPILE_ONLY builds,
# widths 12, 13 and 14 compile clean and width 15 reports local memory 172,960
# against the 163,840 limit. The cap is 12 rather than 14 because 12 is the
# widest step a GPU run has actually checked; raising it is a measurement away,
# not a code change.
#
# The BOTTOM bound is a defect. TP8's narrow MoE path faults: measured on an
# 8-rank decode graph capture, widths 16, 12 and 8 captured cleanly and then
# width 4 took ``Memory access fault ... on address (nil)`` on all eight ranks.
# 4 is the first width below ``layout.wide_from`` (5 at TP8), i.e. the first
# step on the narrow path, and that path has needed TP-specific repair once
# before (sized by HIDDEN instead of INTER, which broke TP2 at widths 3-4).
#
# An earlier version of this comment asserted the fault was "unpopulated
# pointers rather than an LDS overflow". That was never measured, and there is
# now an arithmetically confirmed candidate that predicts exactly width 4:
#
#   ``opart`` was declared ``WAVES * O_K_SUB`` floats on the narrow path
#   (``post_attn.py:348``), with ``O_K_SUB = h_sub * 128`` and ``h_sub =
#   local_q_heads // local_kv_heads``. TP2/TP4 get 16 q heads to a GQA group and
#   so 16,384 floats; TP8 gets 8 (``local_kv_heads`` floors at 1, ``config.py:164``)
#   and so only 8,192. The narrow down stage stores at ``opart + ((k+1)*WAVES*64
#   + wave*64 + lane) * 4`` for k in 0..tokens-1 (``post_attn.py:2849``),
#   needing ``(tokens+1) * 2048`` floats. TP8 fits EXACTLY at 3 tokens and
#   overruns by 2,048 floats (8 KB) at 4; TP2 (narrow widths 1-2) and TP4 (1-4)
#   never do.
#
# The overrun lands in ``uexp``/``umask``/``hst``/``misc``/``route``/``rwt``
# (``post_attn.py:348-368``). The route-clobber story for the (nil) signature is
# WEAK and should not be repeated: every ``route``/``rwt`` read for tokens 0-3
# happens BEFORE the token-3 overflow store (``down_load(k+2)`` at ``:2828-2829``
# reading at ``:2088`` and ``:2101-2103``, against the store at ``:2847-2850``),
# so a within-wave wild pointer cannot arise; and ``fx.min(route,
# SHARED_EXPERT)`` at ``:2088`` clamps from above, so a clobbered index would not
# land in a page below 12 KB on five of eight ranks. The recorded signature was
# three (nil), three 0x1000 and two 0x2000 -- a null or tiny base plus a small
# rank-dependent offset, which fits an unpopulated pointer better than an LDS
# clobber. Unresolved, and see the measured result below.
#
# THE SIZING IS NOW FIXED. ``layout.narrow_opart_floats`` takes the max of the
# attention term and the down stage's, so TP8 width 4 gets 10,240 floats instead
# of 8,192, and TP2/TP4 are byte-identical (their attention term already
# dominates). Measured, not assumed: TP2 12/12, TP4 16/16 -- including width 16,
# which has only 176 B spare -- and TP8 16/16 now build clean under COMPILE_ONLY,
# with the harness's own overflow control firing on the same image.
#
# MEASURED 2026-10-18 ON EIGHT RANKS, and the answer was not the expected one.
# Three arms, one ITT invocation each, image built per arm from a guarded tree:
#
#   fixed sizing + floor 1   -> clean, c=4 at 391.14 tok/s, median TPOT 6.47
#   fixed sizing + floor 5   -> clean, c=4 at 279.83 tok/s, median TPOT 10.93
#   OLD sizing   + floor 1   -> ALSO CLEAN, c=4 at 401.53 tok/s, TPOT 6.35
#
# The first two differ only in which code serves c=4, so the gap between them is
# the runtime observable that the monokernel really ran at width 4. The third is
# the positive control, and it did NOT fault. So:
#
#   - The historical TP8 width-4 fault DOES NOT REPRODUCE ON THIS PORT, with or
#     without the sizing fix. It was recorded on the pre-port per-layer tree
#     (m3-5.22-fusion-mono-tp8:20261016). "Clean after the fix" therefore
#     attributes nothing; an earlier version of this comment set up a decision
#     table that would have drawn exactly that wrong conclusion.
#   - The overrun is nonetheless REAL -- the down store reaches element 10,239
#     against an 8,192-float region, which is arithmetic, not a hypothesis --
#     and the old-sizing arm served a full row while committing it. It is
#     therefore SILENT. That is worse than a fault: nothing in a throughput run
#     would reveal wrong output. Keep the fix; it just gets no credit here.
#   - The UG-placement candidate is NOT isolated by width 3, contrary to what
#     this comment used to claim. Unevenness is not unique to TP8 w4: TP4 w2 and
#     w3 are uneven too and run clean on hardware. What is unique to TP8 w4 is
#     per-token placement AND unevenness AND shared_after_routing false.
#
# THE FLOOR STAYS AT 5, now for a better reason than before: widths 1-4 at TP8
# have never been checked for NUMERICS (v4 §12), and the point above shows that
# "it runs" was never evidence about correctness. An accuracy gate is the
# blocker, not another throughput run.
#
# A step outside its TP's range runs the model's normal forward, which the spec
# dispatches: ``DeviceLimits.widths`` is declared per ISA rather than per TP, so
# the framework cannot route this itself. The union below is what the spec
# declares; the per-TP range is what it builds and serves.
WIDTH_RANGE = {2: (1, 12), 4: (1, 16), 8: (5, 16)}
# Test hook for the floor above; see ``Dims.width_range``. Named under the
# SGLANG_TEST_MONO_ prefix because RFC #43949 decision 2 bans per-model env
# vars and permits that prefix for test hooks.
_MIN_WIDTH_ENV = "SGLANG_TEST_MONO_MIN_WIDTH"
WIDTHS = tuple(range(1, MAX_TOKENS + 1))


class Dims(msgspec.Struct, frozen=True):
    """A TP rank's share of MiniMax-M3's widths.

    Built once per server from its TP size. Every field is derived, so two
    ``Dims`` with the same ``tp`` are interchangeable.
    """

    tp: int

    def __post_init__(self):
        if self.tp not in TP_SIZES:
            raise ValueError(f"TP {self.tp}: the kernels are built for {TP_SIZES}")

    # ------------------------------------------------------------ attention
    @property
    def local_q_heads(self) -> int:
        return TOTAL_Q_HEADS // self.tp

    @property
    def local_kv_heads(self) -> int:
        """This rank's KV heads. The model has 4, so past TP4 a head is
        *replicated* rather than split -- which is why this floors at 1 instead
        of using ``Shard.split``, whose refusal of a remainder is right for the
        MoE widths and wrong here."""
        return max(1, TOTAL_KV_HEADS // self.tp)

    @property
    def local_index_heads(self) -> int:
        """Replicated past TP4 for the same reason as ``local_kv_heads``. The
        model gives a rank as many index heads as KV heads, so index head h
        selects blocks for KV head h."""
        return max(1, TOTAL_INDEX_HEADS // self.tp)

    def qkv_rows(self, idx_heads: int) -> int:
        """Rows of a rank's fused projection, q | k | v | index_q | index_k:
        ``local_q_heads`` q heads, ``local_kv_heads`` k and v heads,
        ``idx_heads`` index q heads (the rank's own, or every one under indexer
        context parallelism) and the one replicated index k head."""
        return (self.local_q_heads + 2 * self.local_kv_heads + 1 + idx_heads) * HEAD_DIM

    @property
    def max_qkv_rows(self) -> int:
        """The widest fused projection any build asks for.

        ``IndexHeads`` pins its count to ``TOTAL_INDEX_HEADS`` under CP and to
        ``local_index_heads`` otherwise, and local is never the larger, so CP's
        every-index-head rows are the bound.

        This was ``qkv_rows(tp)``, reading "a rank computes every index q head"
        -- true only because ``TOTAL_INDEX_HEADS`` happens to be 4: ``qkv_rows``
        is injective, so ``qkv_rows(tp)`` and ``qkv_rows(TOTAL_INDEX_HEADS)``
        coincide at TP4 and nowhere else.
        """
        return self.qkv_rows(TOTAL_INDEX_HEADS)

    @property
    def o_k(self) -> int:
        return self.local_q_heads * HEAD_DIM

    # ------------------------------------------------------------------ MoE
    @property
    def inter(self) -> int:
        """A routed expert's intermediate width on this rank (768 at TP4)."""
        return TOTAL_INTER // self.tp

    @property
    def scale_cols(self) -> int:
        return self.inter // 32

    @property
    def padded_scale_cols(self) -> int:
        """``scale_cols`` as aiter's ``e8m0_shuffle`` stores it. At TP8 that is
        12 -> 16, and the loader's view and the kernel's stride disagree unless
        this is used."""
        return padded(self.scale_cols, SCALE_COL_ALIGN)

    # --------------------------------------------------------------- widths
    @property
    def width_range(self) -> tuple[int, int]:
        """The inclusive (low, high) step width this TP can run; see
        ``WIDTH_RANGE``.

        ``SGLANG_TEST_MONO_MIN_WIDTH`` overrides the low bound. This is a TEST
        HOOK -- RFC #43949 decision 2 bans per-model env vars but permits the
        ``SGLANG_TEST_MONO_`` prefix for exactly this -- and it exists to make
        the TP8 narrow-path fault reproducible without editing the table.

        DANGER at TP8. The floor of 5 is there because width 4 faulted on all
        eight ranks (see ``WIDTH_RANGE``). Lowering it re-enables that path.
        Run it only on a node you have to yourself.

        It is read HERE rather than at import because the floor depends on
        ``tp`` and the port resolves TP at build time -- the in-tree ancestor
        could key a module constant off an import-time TP and a plugin has no
        such moment (see this module's docstring).

        BUILD and ROUTING cannot disagree, but not for the reason an earlier
        version of this docstring gave. It claimed ``widths`` below decides
        which kernels are compiled; it does not, and has no caller outside this
        module and its tests. The real chain is in ``minimax_m3/spec.py``: this
        property is read ONCE at ``:210``, and the same ``(low, high)`` both
        filters ``ctx.widths`` into ``self.kernels`` (``:219``) and is stored as
        ``min_rows``/``max_rows`` for the routing guard at ``:240``. One read,
        two uses -- so a mid-process change to the variable could not desync
        them even if something tried.

        It does NOT follow that the ranks agree. ``_gather_tp`` in the
        framework (``srt/monokernels/runtime.py:182``) compares ``served``,
        which is derived from ``spec.widths(isa)`` -- the per-ISA union, which
        is identical on every rank whatever this returns -- and it runs BEFORE
        ``spec.build`` at ``:198``, where this is first read. So that guard
        cannot see a divergent value here. What makes divergence impossible is
        operational, not enforced: all ranks of a TP group share one container
        and one ``environment:`` block. Set per-rank values and the ranks that
        raise would leave the rest waiting in the Packed collectives.
        """
        low, high = WIDTH_RANGE[self.tp]
        raw = os.environ.get(_MIN_WIDTH_ENV)
        if raw is None:
            return low, high
        try:
            want = int(raw)
        except ValueError:
            raise ValueError(f"{_MIN_WIDTH_ENV}={raw!r} is not an integer") from None
        if not 1 <= want <= high:
            raise ValueError(
                f"{_MIN_WIDTH_ENV}={want} is outside 1..{high}, the widths TP "
                f"{self.tp} can build"
            )
        return want, high

    @property
    def widths(self) -> tuple[int, ...]:
        low, high = self.width_range
        return tuple(range(low, high + 1))

    @property
    def index_heads(self) -> IndexHeads:
        """Without indexer context parallelism: the rank's own index q heads."""
        return IndexHeads(dims=self, count=self.local_index_heads)


class IndexHeads(msgspec.Struct, frozen=True):
    """The index q heads in a rank's fused projection, a build parameter.

    Two modes, which must not be inferred from ``count`` -- at TP2 they have the
    same count and different meanings:

    * **own** (``cp=False``): the rank's own ``local_index_heads`` heads, each
      scoring every block for the KV head it pairs with. The model gives a rank
      as many index heads as KV heads (probed: 1/1 at TP4, 2/2 at TP2), so index
      head h selects for KV head h and the split stage's pass h reads its blocks.
    * **cp** (``cp=True``): indexer context parallelism -- all
      ``TOTAL_INDEX_HEADS`` in head order, a rank scoring only its own 1/tp
      stripe of blocks (``own``) and exchanging candidates with its peers.

    ``count > 1`` used to select CP on its own, which is why TP2 was refused:
    its own pair is not context parallelism.

    **No caller builds ``cp=True`` today, at any TP.** The host passes
    ``Dims.index_heads``; the only other constructions carry the flag they were
    given. So every ``cp`` branch in the kernels -- the striped block list, the
    ``// tp`` block index and sub-region size, and ``sel_cp``'s peer exchange --
    is unreachable in this build. They are kept as the upstream author's
    lineage, not because they run: do not infer from their presence that CP is a
    tested path. CP also cannot serve TP2 as written, since its peer exchange
    indexes ``peer_addr(part // 2)`` for ``part < N_SPLIT``, i.e. up to peer 3.
    """

    dims: Dims
    count: int
    own: int = 0
    cp: bool = False

    def __post_init__(self):
        want = TOTAL_INDEX_HEADS if self.cp else self.dims.local_index_heads
        if self.count != want:
            raise ValueError(
                f"{'cp' if self.cp else 'own'} mode takes {want} index heads at "
                f"TP {self.dims.tp}, got {self.count}"
            )
        if not 0 <= self.own < max(self.count, 1):
            raise ValueError(f"own index head {self.own} of {self.count}")
        # pre_attn's SCRATCH_X8 is sized from ``max_qkv_rows``, so a build whose
        # projection is wider than that bound overruns the norm tasks' x8 rows
        # silently.
        if self.rows > self.dims.max_qkv_rows:
            raise ValueError(
                f"{self.count} index heads need {self.rows} projection rows, "
                f"past the bound of {self.dims.max_qkv_rows}"
            )

    @property
    def rows(self) -> int:
        return self.dims.qkv_rows(self.count)

    @property
    def score_blocks(self) -> int:
        """Blocks one head's score region holds. Under CP a rank scores only its
        1/tp stripe; in own mode a head scores every block."""
        return MAX_INDEX_BLOCKS // self.dims.tp if self.cp else MAX_INDEX_BLOCKS

    @property
    def score_words(self) -> int:
        """A token's whole score region, every head. Equal to
        ``MAX_INDEX_BLOCKS`` in both modes wherever a rank has one index head,
        which is why TP4 and TP8 see no change: CP spreads TOTAL heads over
        1/tp blocks each."""
        return self.count * self.score_blocks

    @property
    def iq_off(self) -> int:
        d = self.dims
        return (d.local_q_heads + 2 * d.local_kv_heads + self.own) * HEAD_DIM

    @property
    def ik_off(self) -> int:
        d = self.dims
        return (d.local_q_heads + 2 * d.local_kv_heads + self.count) * HEAD_DIM
