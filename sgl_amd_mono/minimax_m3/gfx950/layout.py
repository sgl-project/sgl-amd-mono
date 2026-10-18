# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where the mono layer kernel (K4) puts its tasks and data, per decode token count.

Plain Python, evaluated at build time: which CTA runs each stage's tasks, the
scratch and peer-symmetric regions, and the LDS row pitches. Kept free of FlyDSL
so it is testable without a GPU.

Split the same way as :mod:`config`: what the execution model fixes stays a
module constant, and everything whose extent follows a rank's share of the heads
or of INTER hangs off :class:`Layout`, built from a :class:`Dims`. The
module-level assertions that used to fire at import now fire in
``Layout.__post_init__``, once per build, against that build's TP.
"""

from __future__ import annotations

import msgspec

from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.minimax_m3.gfx950.config import (
    HEAD_DIM,
    HIDDEN,
    MAX_INDEX_BLOCKS,
    MAX_SPARSE_KEYS,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    PAGE16,
    SPARSE_BLOCK,
    TOP_K,
    TOPK_BLOCKS,
    TOTAL_INDEX_HEADS,
    Dims,
)

SPLIT_KEYS = 256  # keys per split task (one paged-attention context partition)
N_SPLIT = MAX_SPARSE_KEYS // SPLIT_KEYS

# A-operand rows of the split stage's QK and PV MFMAs. The instruction is
# 16 x 16 x 32, and its A rows are q heads: a lane reads row ``l16`` and holds
# C rows ``4 * g4 + e``. The stage was written when a GQA group was exactly 16
# heads, so the two coincided and the row index needed neither clamp nor guard.
#
# That is a TP4 (and TP2-per-pass) coincidence, not an invariant. TP8 gives a
# rank 8 q heads over its one KV head, so H_SUB is 8 and rows 8..15 are not
# heads at all: unguarded, their C rows are stored into the next wave's ``hst``
# slots, past ``p8``'s head dimension, and into the next wave's ``opart``
# partials -- all silently, since every one of those is an in-bounds LDS word
# belonging to someone else.
SPLIT_ROWS = 16

CAND_CAP = 256  # blocks counted against each other in the selection floor
SEL_P8_WORDS = 2 * CAND_CAP + 2 + WAVES * TOPK_BLOCKS

O_ROWS = 32  # hidden rows per o_proj / attention-reduce task
# from this many tokens the attention all-reduce is a reduce-scatter to the row
# group's owner rank (t % npes) plus an all-gather of the sums: half the bytes over
# each link for one more hop (smaller batches are latency-bound: one hop wins)
AG_RS_FROM = 12
N_O = HIDDEN // O_ROWS
PAGES_PER_BLOCK = SPARSE_BLOCK // PAGE16
N_ROUTER = N_ROUTED // WAVES  # one expert per wave
UG_ROWS = 16
SH_ROWS = 8  # mid rows per shared-expert up / gate task (a half task)
DN_ROWS = 32
N_DN = HIDDEN // DN_ROWS
# The router logits are written SCORE_COPIES times and CTA b polls copy
# b % SCORE_COPIES: hundreds of expert CTAs spinning on one 1 KB region serialize
# on its cache lines.
SCORE_COPIES = 8

# LDS rows of a token's MXFP8 xn and of its E8M0 scales, padded by 16 B: at the
# natural 1536 / 192-word rows the 16 tokens a MFMA B operand gathers sit in one
# bank group (16-way conflicts)
XN8_ROW = HIDDEN // 4 + 4
XSC_ROW = HIDDEN // 32 + 4


def _layout(items, align=256):
    """Byte offsets of (value, tag) pair regions, 8 bytes per pair."""
    off, out = 0, {}
    for name, pairs in items:
        out[name] = off
        off += (pairs * 8 + align - 1) // align * align
    out["_bytes"] = off
    return out


# a context past THREADS blocks, one-head build: each split task's TOPK_BLOCKS
# best (key, block) (indexer context parallelism: the peers' sel_cp)
SEL_LONG_PAIRS = N_SPLIT * TOPK_BLOCKS * 2


def sym_layout(npes: int):
    """Per-rank symmetric buffer: attn / ffn partial regions, [src rank][token][HIDDEN],
    and the peers' index-selection candidates (``sel_cp``)."""
    rows = npes * MAX_TOKENS * HIDDEN // 2
    # a_ag / ffn_ag: the finished attention / FFN all-reduce rows, each written by
    # its owner rank (the reduce-scatter + all-gather form, S >= AG_RS_FROM)
    ag = MAX_TOKENS * HIDDEN // 2
    return _layout(
        [
            ("attn", rows),
            ("ffn", rows),
            ("a_ag", ag),
            ("ffn_ag", ag),
            # indexer context parallelism: token k's candidates for this rank's
            # index head, source rank s's half h's r-th best at pairs
            # 2 ((k N_SPLIT + 2 s + h) TOPK_BLOCKS + r) + (0 key, 1 block)
            ("sel_cp", SEL_LONG_PAIRS * MAX_TOKENS),
        ]
    )


def ug_ctas_per_token(tokens: int):
    """CTAs per token of the routed up / gate tasks, token k on CTAs k C .. (k + 1)
    C - 1, so a CTA stages one token's xn, not all. None: round-robin, for S < 4
    or S not dividing the grid."""
    return BLOCKS // tokens if tokens >= 4 and BLOCKS % tokens == 0 else None


class Layout(msgspec.Struct, frozen=True):
    """K4's task placement and region sizes for one rank's :class:`Dims`.

    Every value here is derived from ``dims``, so two ``Layout`` with the same
    TP are interchangeable.
    """

    dims: Dims

    def __post_init__(self):
        # The split stage's thread maps and q8 region, checked against this
        # build's head counts rather than at import; see the properties below
        # for what each bound is protecting.
        assert self.h % self.dims.local_kv_heads == 0, (
            f"{self.h} q heads do not divide over {self.dims.local_kv_heads} KV heads"
        )
        assert self.h_sub <= SPLIT_ROWS, (
            f"GQA group {self.h_sub} exceeds the split stage's {SPLIT_ROWS}-row thread map"
        )
        # The surplus rows are a contiguous tail of ``4 * g4 + e``, so while H_SUB is a
        # multiple of 4 the live test is on the lane group alone and never on ``e``.
        assert self.h_sub % 4 == 0, (
            f"GQA group {self.h_sub} leaves a partial 4-lane C-row group in the split stage"
        )
        assert self.split_slots <= THREADS, (
            f"the split stage needs {self.split_slots} thread slots a pass, past THREADS"
        )
        # q8 holds every pass' q at once, so the passes' maps must fit it end to end.
        assert self.split_slots * self.split_passes <= self.o_k // 4, (
            f"{self.split_passes} passes of {self.split_slots} slots overrun q8's "
            f"{self.o_k // 4} words"
        )
        # The tasks are handed out one per CTA (``has_sh = start("shared") < SH_TASKS``,
        # post_attn.py), and ``start`` is taken mod BLOCKS -- so past BLOCKS tasks the
        # guard still admits every CTA while tasks BLOCKS.. simply never run, dropping
        # their mid rows from the shared expert with no diagnostic. INTER scales as
        # 1 / TP, so this holds from TP2 up (192 tasks) and fails at TP1 (384).
        assert self.sh_tasks <= BLOCKS, (
            f"{self.sh_tasks} shared-expert tasks over {BLOCKS} CTAs: INTER = "
            f"{self.dims.inter} needs a CTA to carry more than one, which stage_moe "
            f"does not implement"
        )
        # The MXFP8 pass's scale is an xor-reduction over 4-lane groups, so a round's
        # active slots must not end inside one. THREADS is a multiple of 64, so a round
        # offset never moves a lane out of its wave, and this keeps the tail aligned.
        assert self.mid8_slots % 4 == 0, (
            f"the MXFP8 mid pass leaves a partial 4-lane scale group: INTER = "
            f"{self.dims.inter}"
        )

    # ------------------------------------------------------------- attention
    @property
    def h(self) -> int:
        return self.dims.local_q_heads

    @property
    def o_k(self) -> int:
        return self.dims.o_k

    # The split stage runs one pass per KV head: a pass attends that head's GQA
    # group (H // LOCAL_KV_HEADS q heads) to its own K and V. The model is 64 q over
    # 4 KV heads, so a group is 16 q heads at every TP -- which is also exactly what
    # the stage's thread map covers (the MFMA A operand is 16 rows, a lane holds
    # heads 4 * (lane // 16) + e, and the partial reduction walks O_K_SUB with
    # THREADS * 4 = 2048 elements). The assert in ``__post_init__`` is what ties the
    # two together.
    #
    # At TP4 and TP8 a rank holds one KV head, so SPLIT_PASSES is 1, h0 is 0, and
    # every expression is the original value -- those builds are unchanged.
    #
    # This is also what keeps K4 inside the 160 KB LDS at TP2: the PV partials are
    # WAVES * O_K, which at 32 q heads alone is 131 072 B of a 163 840 B budget.
    # Only ``opart`` is sized for a pass; every other region stays whole-H and is
    # indexed by absolute head.
    @property
    def h_sub(self) -> int:
        return self.h // self.dims.local_kv_heads  # q heads sharing one KV head

    @property
    def o_k_sub(self) -> int:
        return self.h_sub * HEAD_DIM

    def narrow_opart_floats(self, tokens: int) -> int:
        """``opart``'s size on the NARROW MoE path, in floats.

        Two stages share the region and the larger wins:

        * the attention PV partials index ``w * o_k_sub`` over WAVES waves
          (store ``post_attn.py:1198-1209``, reduce ``:1231-1233``), needing
          ``WAVES * o_k_sub``;
        * the narrow MoE down stage stores at ``(k + 1) * WAVES * 64 + wave * 64
          + lane`` scaled by 4, for k in 0..tokens-1 (``post_attn.py:2849``),
          needing ``(tokens + 1) * WAVES * 64 * 4``. Its reduction
          (``:2859-2868``) reads strictly inside that, peaking 60 floats lower.

        The region's other tenants need no term of their own, but they are
        listed because "the larger of two" is only safe if the set really is
        exhaustive: the MoE up/gate partials and their reduction (``:1978``,
        ``:1992``) and the shared-expert partials (``:2822``) peak at 2,048
        floats, and ``route_top4`` -- which takes ``opart`` as its ``sig``
        scratch (``:2708``, writing at ``:1638-1639``) -- needs
        ``tokens * N_ROUTED``, 2,048 at the widest narrow case. ``opart`` is a
        distinct array on the narrow build (``:459``); these stages reuse it in
        time, separated by the barriers at ``:1979``, ``:2000``, ``:2710`` and
        ``:2851``.

        Sizing on the attention term ALONE is the TP8 width-4 defect: ``o_k_sub``
        follows ``h_sub = local_q_heads // local_kv_heads``, and ``local_kv_heads``
        floors at 1, so a TP8 rank has 8 q heads to a GQA group where TP2 and TP4
        have 16 -- half the region -- while the down stage's need does not depend
        on TP at all. TP8 then fits exactly at 3 tokens and overruns by 2,048
        floats at 4, into ``uexp`` / ``route`` / ``rwt``.

        Taking the max is a NO-OP at TP2 and TP4 (16,384 floats against a peak
        need of 6,144 at TP2's narrow widths 1-2 and 10,240 at TP4's 1-4), so it
        cannot grow their LDS. Measured 2026-10-18 over every width below
        ``wide_from`` at each TP: delta 0 B at TP2 and TP4, +8,192 B at TP8
        width 4 and nowhere else.

        That no-op is NOT what protects TP4's 176 B of headroom at width 16, and
        an earlier version of this docstring said it was. Width 16 is WIDE, and
        the declaration at ``post_attn.py:349`` is
        ``1 if WIDE else narrow_opart_floats(tokens)`` -- so this helper is
        never evaluated there at all. Sweeping it across widths 1..16 without
        that filter reports growth on widths that cannot reach the branch; the
        only widths that matter are those below ``wide_from``.
        """
        return max(WAVES * self.o_k_sub, (tokens + 1) * WAVES * 64 * 4)

    @property
    def split_passes(self) -> int:
        return self.dims.local_kv_heads

    # Whether every MFMA row is a real q head. When false the stage clamps its
    # A-operand row reads onto the last real head (so nothing reads out of bounds)
    # and drops the surplus C rows at every store.
    @property
    def split_rows_whole(self) -> bool:
        return self.h_sub == SPLIT_ROWS

    @property
    def split_live_g4(self) -> int:
        return self.h_sub // 4  # lane groups whose C rows are real heads

    # The per-pass q and PV thread maps walk O_K_SUB // 4 slots, one thread each --
    # exactly THREADS at H_SUB == 16 and half of it at TP8, where the surplus
    # threads must not store (``q8`` and the ``sp_o`` row are only O_K_SUB wide).
    @property
    def split_slots(self) -> int:
        return self.o_k_sub // 4

    @property
    def split_slots_whole(self) -> bool:
        return self.split_slots == THREADS

    # ``p8`` (LDS words): the split's FP8 P, H heads x SPLIT_KEYS keys -- and, on
    # top of the same region, the indexer selection's candidates, their counter and
    # the lane groups' maxima. The borrow's safety is argued where it happens
    # (post_attn); what is decided here is only how wide the region must be.
    #
    # It used to be sized by P alone, on the reasoning that the borrow fits under
    # it. That holds at TP4 (1024 words) and TP2 (2048) and fails at TP8, where a
    # rank's 8 q heads need only 512 while the selection still needs 642: sized by
    # P, the candidates would run over the end of the region.
    @property
    def p8_words(self) -> int:
        return max(self.h * SPLIT_KEYS // 4, SEL_P8_WORDS)

    # A page-16 of ONE KV head (``pre_attn.PAGE_BYTES``). The cache is
    # [page16][kv head][...] -- aiter's sparse views are k.view(-1, heads, 8, 16, 16)
    # and v.view(-1, heads, 1, 128, 16), both 2048 B a head -- so stepping one page
    # means stepping over every head, and KV head h of page p starts at
    # (p * LOCAL_KV_HEADS + h) * PAGE_BYTES.
    @property
    def kv_page_bytes(self) -> int:
        return self.dims.local_kv_heads * PAGE16 * HEAD_DIM

    # A token's indexer score region, sized for whichever mode needs more: own mode
    # gives each of the rank's heads every block, CP gives all TOTAL_INDEX_HEADS a
    # 1/TP stripe each. The two are equal wherever a rank has one index head, so
    # TP4 and TP8 keep the original MAX_INDEX_BLOCKS; TP2 doubles it. Scratch is
    # global memory, not LDS -- this does not touch the 160 KB budget.
    @property
    def iscore_blocks(self) -> int:
        return max(
            self.dims.local_index_heads * MAX_INDEX_BLOCKS,
            TOTAL_INDEX_HEADS * (MAX_INDEX_BLOCKS // self.dims.tp),
        )

    # ------------------------------------------------------------------- MoE
    @property
    def ug_per_slot(self) -> int:
        return self.dims.inter // UG_ROWS

    @property
    def sh_tasks(self) -> int:
        return self.dims.inter // SH_ROWS

    # --------------------------------------------------------------- scratch
    @property
    def scratch(self) -> dict:
        """Byte offsets of the global scratch regions."""
        inter = self.dims.inter
        # every region holds MAX_TOKENS rows (token k at k * the one-token size)
        return _layout(
            [
                # indexer block scores, token k's head h block b at pair
                # k * ISCORE_BLOCKS + h * heads.score_blocks + b (``IndexHeads``)
                ("iscore", self.iscore_blocks * MAX_TOKENS),
                # packed bf16 pairs
                ("sp_o", N_SPLIT * self.h * HEAD_DIM // 2 * MAX_TOKENS),
                ("sp_m", N_SPLIT * self.h * MAX_TOKENS),
                ("sp_l", N_SPLIT * self.h * MAX_TOKENS),
                # merged attention, plain FP8 words
                ("attnq", self.o_k // 4 * MAX_TOKENS // 2),
                ("attnq_s", MAX_TOKENS),  # and its per-token scale
                # the finished attention all-reduce + residual, bf16 pairs (o tasks -> router)
                ("a", HIDDEN // 2 * MAX_TOKENS),
                ("xn8", HIDDEN // 4 * MAX_TOKENS),  # MXFP8 xn, 4 fp8 per word
                ("xsc", HIDDEN // 32 * MAX_TOKENS),  # and its per-1x32 E8M0 scales
                # copy c, token k at pairs [(c * MAX_TOKENS + k) * N_ROUTED, ..)
                # per expert (routing key, sigmoid) pairs, see route_key
                ("scores", 2 * N_ROUTED * MAX_TOKENS * SCORE_COPIES),
                ("mid", MOE_SLOTS * inter * MAX_TOKENS),
                # token k's split task p's candidates for index head h: rank r at pairs
                # 2 (((h MAX_TOKENS + k) N_SPLIT + p) TOPK_BLOCKS + r) + (0 key, 1 block)
                #
                # The head term is load-bearing in own mode. A share's candidates are
                # gathered ACROSS CTAs -- every split task polls every other task's
                # entries -- and the mailbox tag is the step's, the same for both passes.
                # Without a per-head range a CTA entering pass 1 would overwrite slots a
                # slower peer is still reading for pass 0. LOCAL_INDEX_HEADS is 1 at TP4
                # and TP8, where this is the original size.
                (
                    "sel_long",
                    SEL_LONG_PAIRS * MAX_TOKENS * self.dims.local_index_heads,
                ),
                # wide: the MXFP8 xn and scales as plain words (every token's row), and a
                # done flag per router task
                ("xn8p", HIDDEN // 4 * MAX_TOKENS // 2),
                ("xscp", HIDDEN // 32 * MAX_TOKENS // 2),
                ("xndone", N_ROUTER * MAX_TOKENS),
                # and the mid rows as MXFP8 words, their E8M0 scales a word each (the down
                # stage's LDS layout), a done flag per up / gate task pair
                ("mid8p", MOE_SLOTS * inter * MAX_TOKENS // 8),
                ("midsp", MOE_SLOTS * inter // 32 * MAX_TOKENS // 2),
                ("ugdone", (TOP_K * MAX_TOKENS + 1) * self.ug_per_slot // 2),
            ]
        )

    @property
    def scratch_bytes(self) -> int:
        return self.scratch["_bytes"]

    # ---------------------------------------------------------- task placing
    def stage_bases(self, tokens: int):
        """Task t of a stage runs on CTA (base + t) % BLOCKS; the merge (S > 1), o and
        down tasks start past the tokens' split CTAs."""
        n_split = N_SPLIT * tokens
        n_merge = tokens if tokens > 1 else 0
        router = (n_split + n_merge + N_O) % BLOCKS
        return {
            "split": 0,
            "merge": n_split,
            "o": n_split + n_merge,
            "router": router,
            "shared": self._shared_base(tokens, router),
            "ug": 0,
            "down": n_split,
        }

    def ug_tasks_of(self, cta: int, tokens: int) -> int:
        """Routed up / gate tasks CTA ``cta`` runs."""
        per_token = TOP_K * self.ug_per_slot
        cpt = ug_ctas_per_token(tokens)
        if cpt is None:
            return (per_token * tokens - cta + BLOCKS - 1) // BLOCKS
        return (per_token - cta % cpt + cpt - 1) // cpt

    def _shared_base(self, tokens: int, router: int) -> int:
        """First of the SH_TASKS consecutive CTAs that run the shared expert's up /
        gate before the routing lands. Ranked by the most routed up / gate tasks a
        window CTA carries (they queue behind it); then, only if that is the most any
        CTA carries, by its router CTAs (they start it late and would delay the last
        routed tasks); then by its down CTAs; then by its router CTAs anyway."""
        n_router = N_ROUTER * tokens
        n_split = N_SPLIT * tokens
        sh_tasks = self.sh_tasks
        most = max(self.ug_tasks_of(c, tokens) for c in range(BLOCKS))

        def cost(base):
            ctas = [(base + j) % BLOCKS for j in range(sh_tasks)]
            busiest = max(self.ug_tasks_of(c, tokens) for c in ctas)
            in_router = sum((c - router) % BLOCKS < n_router for c in ctas)
            in_down = sum(n_split <= c < n_split + N_DN for c in ctas)
            return busiest, in_router if busiest == most else 0, in_down, in_router

        return min(range(BLOCKS), key=cost)

    def shared_after_routing(self, tokens: int) -> bool:
        """Every CTA carries the same number of routed up / gate tasks: a
        shared-expert CTA routes first and runs its shared half task while its first
        routed task's weights are in flight, instead of holding its routing back."""
        loads = [self.ug_tasks_of(c, tokens) for c in range(BLOCKS)]
        return min(loads) == max(loads)

    # ------------------------------------------------------------ MoE widths
    # Token counts from here run the expert-grouped MoE. The narrow path keeps every
    # token's mids three ways at once (see pool_words), so its pool is the budget's
    # swing term and the width at which it stops fitting depends on INTER -- i.e. on
    # TP. Measured by COMPILE_ONLY builds against the 160 KB gfx950 limit: TP2
    # narrow is clean at S=1-2 but needs 168,416 B at S=3 and 192,416 B at S=4,
    # against 163,840. TP4 narrow fits all of S=1-4 (its mids are half the size), so
    # it keeps the original threshold. Set this from a compile, never from a mirror:
    # the LDS overflow is a non-raising diagnostic.
    @property
    def wide_from(self) -> int:
        return 3 if self.dims.tp == 2 else 5

    def wide_moe(self, tokens: int) -> bool:
        """The MoE stage groups its work by expert (every distinct expert's weights
        read once, a B column per token) instead of by (token, slot): past four tokens
        the per-token tasks re-read the experts the tokens share, and their operands
        no longer fit in LDS."""
        return tokens >= self.wide_from

    # --------------------------------------------------------- LDS row pitch
    # the down stage's MXFP8 mid rows / their scales (16 lanes, 16 rows): 196 and
    # 28 words put the rows' 16 B reads, and 4 B scale reads, in distinct banks
    @property
    def mid_row(self) -> int:
        return self.dims.inter // 4 + 4

    @property
    def midsc_row(self) -> int:
        return self.dims.inter // 32 + 4

    # and the o stage's FP8 attention rows (512 words: the same 16-way conflict)
    @property
    def o_row(self) -> int:
        return self.o_k // 4 + 4

    # ------------------------------------------------------------ thread maps
    # Thread maps whose extent is set by TP, each walked in whole THREADS-wide
    # rounds. Every one of these was originally written as a single round, which
    # was correct only because at TP4 the work happens to come to exactly THREADS
    # slots; at TP2 the dimension doubles and one round covers half of it, leaving
    # the rest of the region unwritten for the next stage to read as NaN bits.
    #
    # The merge: a thread holds 4 dims of one head, so the CTA needs
    # H * HEAD_DIM / 4 slots -- 512 at TP4, 1024 at TP2.
    @property
    def merge_slots(self) -> int:
        return self.h * (HEAD_DIM // 4)

    @property
    def merge_passes(self) -> int:
        return -(-self.merge_slots // THREADS)

    # Whether those rounds come out whole. At TP8 (H = 8) the merge needs only half
    # a CTA, so its tail round is masked out of the mailbox polls, the amax
    # reduction and the store alike -- see ``merge_token`` in post_attn. Leaving it
    # unmasked was not merely wasteful: ``hm`` ran past H into the next partition's
    # ``sp_m`` / ``sp_l`` slots (a poll on a tag that is never published, i.e. a
    # hang, not a wrong number), and the store ran past O_K / 4 words into the next
    # token's ``attnq`` row.
    @property
    def merge_rounds_whole(self) -> bool:
        return self.merge_slots == self.merge_passes * THREADS

    # Slots the tail round actually carries (== THREADS when the rounds are whole).
    @property
    def merge_tail_slots(self) -> int:
        return self.merge_slots - (self.merge_passes - 1) * THREADS

    # The narrow MoE path's two mid passes, over a MOE_SLOTS * INTER region: 3840
    # at TP4, 7680 at TP2, 1920 at TP8. (The wide path, from WIDE_FROM tokens up,
    # derives its own counts from the row count it is given.) The copy moves two
    # elements a thread; the MXFP8 pass takes eight.
    @property
    def mid_copy_rounds(self) -> int:
        return -(-(MOE_SLOTS * self.dims.inter) // (2 * THREADS))

    @property
    def mid8_slots(self) -> int:
        return MOE_SLOTS * self.dims.inter // 8

    @property
    def mid8_rounds(self) -> int:
        return -(-self.mid8_slots // THREADS)

    # ----------------------------------------------------------- the LDS pool
    def pool_words(self, tokens: int) -> int:
        """LDS words of the activation pool ``x``, the largest of the stages that take
        turns in it. Wide: the largest of the split's PV partials, the up / gate
        stage's MXFP8 xn + scales + up partials + router sigmoids, and the down
        stage's MXFP8 mids + scales + a zero row."""
        inter = self.dims.inter
        if not self.wide_moe(tokens):
            # The narrow down stage keeps every token's mids three ways at once --
            # bf16 as they land, then MXFP8 plus E8M0 scales -- and that, not the
            # bf16 xn, is the largest narrow operand once INTER passes 983.
            # MOE_SLOTS * INTER * 25/32 is 3000 at TP4, which slips under HIDDEN / 2
            # = 3072 by 2.4% and is why the xn was mistaken for the maximum; it is
            # 6000 at TP2. Sizing this from HIDDEN put the mid tail and every scale
            # word outside the declared array.
            mids = MOE_SLOTS * inter // 2  # bf16, as the mids land
            mid8 = MOE_SLOTS * inter // 4  # the MXFP8 copy the down GEMV reads
            midsc = MOE_SLOTS * inter // 32  # and its E8M0 scales
            return tokens * max(HIDDEN // 2, mids + mid8 + midsc)
        rows = MOE_SLOTS * tokens + 1
        return max(
            # the split's PV partials, one pass' worth (see h_sub above)
            WAVES * self.o_k_sub,
            tokens * (XN8_ROW + XSC_ROW + N_ROUTED) + WAVES * 64 * 4,
            rows * (self.mid_row + self.midsc_row),
        )
