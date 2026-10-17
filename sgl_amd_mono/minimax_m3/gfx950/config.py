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
# (measured over 12 column counts: 12 -> 16, 24 -> 24, 48 -> 48). At TP8 the w2
# scale's 12 columns become 16, so the stored tensor is wider than INTER // 32
# implies and anything indexing it must use the padded stride.
SCALE_COL_ALIGN = 8

# indexer context parallelism serves a request past this many index blocks: its
# selection costs a fixed overhead that the one-head path's O(n^2) ranking only
# exceeds beyond here
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
# Those are unpopulated pointers rather than an LDS overflow, which is why the
# compile gate cannot see it -- every TP8 width builds clean. 4 is the first
# width below ``layout.wide_from`` (5 at TP8), i.e. the first step on the narrow
# path, and that path has needed TP-specific repair once before (it was sized by
# HIDDEN instead of INTER, which broke TP2 at widths 3-4). Treat it as the same
# class, unfixed at TP8.
#
# A step outside its TP's range runs the model's normal forward, which the spec
# dispatches: ``DeviceLimits.widths`` is declared per ISA rather than per TP, so
# the framework cannot route this itself. The union below is what the spec
# declares; the per-TP range is what it builds and serves.
WIDTH_RANGE = {2: (1, 12), 4: (1, 16), 8: (5, 16)}
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
        return (
            self.local_q_heads + 2 * self.local_kv_heads + 1 + idx_heads
        ) * HEAD_DIM

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
        ``WIDTH_RANGE``."""
        return WIDTH_RANGE[self.tp]

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
