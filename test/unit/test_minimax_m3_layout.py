# SPDX-License-Identifier: Apache-2.0
"""K4's task placement and region sizes at each supported TP, pinned as literals.

Like :mod:`test_minimax_m3_dims`, these values were module-level constants in the
kernels' in-tree ancestor, bound by value at import from a TP read out of the
environment. Moving them onto :class:`Layout` had to be value-exact: a TP4 build
at width 16 uses 163,664 B of the 163,840 B LDS budget, so a region that grew by
one element would stop the working build compiling. Every value below was
compared against that ancestor at TP 2, 4 and 8 -- all 52 of its public names,
functions called over their whole input domain -- before it was written down
here; this test keeps them from drifting afterwards.
"""

import pytest

from sgl_amd_mono.common.plan import BLOCKS, THREADS
from sgl_amd_mono.minimax_m3.gfx950.config import TP_SIZES, WIDTH_RANGE, Dims
from sgl_amd_mono.minimax_m3.gfx950.layout import (
    HEAD_DIM,
    N_DN,
    N_O,
    N_ROUTED,
    N_ROUTER,
    N_SPLIT,
    SEL_LONG_PAIRS,
    SEL_P8_WORDS,
    SPLIT_ROWS,
    XN8_ROW,
    XSC_ROW,
    Layout,
    sym_layout,
    ug_ctas_per_token,
)

# tp -> every extent that follows a rank's share of the heads or of INTER
EXPECTED = {
    2: dict(
        h=32,
        h_sub=16,
        o_k=4096,
        o_k_sub=2048,
        split_passes=2,
        split_rows_whole=True,
        split_live_g4=4,
        split_slots=512,
        split_slots_whole=True,
        p8_words=2048,
        kv_page_bytes=4096,
        iscore_blocks=16384,
        ug_per_slot=96,
        sh_tasks=192,
        wide_from=3,
        mid_row=388,
        midsc_row=52,
        o_row=1028,
        merge_slots=1024,
        merge_passes=2,
        merge_rounds_whole=True,
        merge_tail_slots=512,
        mid_copy_rounds=8,
        mid8_slots=960,
        mid8_rounds=2,
        scratch_bytes=6526720,
    ),
    4: dict(
        h=16,
        h_sub=16,
        o_k=2048,
        o_k_sub=2048,
        split_passes=1,
        split_rows_whole=True,
        split_live_g4=4,
        split_slots=512,
        split_slots_whole=True,
        p8_words=1024,
        kv_page_bytes=2048,
        iscore_blocks=8192,
        ug_per_slot=48,
        sh_tasks=96,
        wide_from=5,
        mid_row=196,
        midsc_row=28,
        o_row=516,
        merge_slots=512,
        merge_passes=1,
        merge_rounds_whole=True,
        merge_tail_slots=512,
        mid_copy_rounds=4,
        mid8_slots=480,
        mid8_rounds=1,
        scratch_bytes=3758080,
    ),
    8: dict(
        h=8,
        h_sub=8,
        o_k=1024,
        o_k_sub=1024,
        split_passes=1,
        split_rows_whole=False,
        split_live_g4=2,
        split_slots=256,
        split_slots_whole=False,
        p8_words=642,
        kv_page_bytes=2048,
        iscore_blocks=8192,
        ug_per_slot=24,
        sh_tasks=48,
        wide_from=5,
        mid_row=100,
        midsc_row=16,
        o_row=260,
        merge_slots=256,
        merge_passes=1,
        merge_rounds_whole=False,
        merge_tail_slots=256,
        mid_copy_rounds=2,
        mid8_slots=240,
        mid8_rounds=1,
        scratch_bytes=2914560,
    ),
}


@pytest.mark.parametrize("tp", TP_SIZES)
def test_rank_extents(tp):
    layout = Layout(dims=Dims(tp=tp))
    got = {name: getattr(layout, name) for name in EXPECTED[tp]}
    assert got == EXPECTED[tp]


def test_execution_model_constants_do_not_follow_tp():
    """What the grid fixes is TP-independent and stays a module constant: a
    rank's share moves none of these."""
    assert (N_SPLIT, N_O, N_ROUTER, N_DN) == (8, 192, 16, 192)
    assert (SEL_P8_WORDS, SEL_LONG_PAIRS, SPLIT_ROWS) == (642, 256, 16)
    assert (XN8_ROW, XSC_ROW) == (1540, 196)


def test_sym_layout_is_set_by_the_peer_count_not_the_tp():
    """``sym_layout`` sizes [src rank][token][HIDDEN] rows, all of HIDDEN rather
    than a rank's slice, so it takes ``npes`` and never a ``Dims``."""
    assert sym_layout(4) == {
        "attn": 0,
        "ffn": 1572864,
        "a_ag": 3145728,
        "ffn_ag": 3538944,
        "sel_cp": 3932160,
        "_bytes": 3964928,
    }
    # the partial regions are one row group per peer; the all-gather rows are not
    assert sym_layout(8)["ffn"] == 2 * sym_layout(4)["ffn"]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_scratch_regions_are_ordered_and_disjoint(tp):
    layout = Layout(dims=Dims(tp=tp))
    scratch = layout.scratch
    assert list(scratch) == [
        "iscore",
        "sp_o",
        "sp_m",
        "sp_l",
        "attnq",
        "attnq_s",
        "a",
        "xn8",
        "xsc",
        "scores",
        "mid",
        "sel_long",
        "xn8p",
        "xscp",
        "xndone",
        "mid8p",
        "midsp",
        "ugdone",
        "_bytes",
    ]
    offsets = [off for name, off in scratch.items() if name != "_bytes"]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert layout.scratch_bytes == scratch["_bytes"] > offsets[-1]
    # ``_layout`` aligns every region to 256 B
    assert all(off % 256 == 0 for off in offsets)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_split_stage_thread_map_fits_its_mfma_rows(tp):
    """The split stage's A rows are q heads, and its per-pass maps must fit both
    the 16-row MFMA and one CTA. TP8's 8-head GQA group is the case that makes
    neither come out whole -- the two ``*_WHOLE`` flags are what the stage masks
    on."""
    layout = Layout(dims=Dims(tp=tp))
    assert layout.h_sub <= SPLIT_ROWS
    assert layout.h_sub % 4 == 0
    assert layout.split_slots <= THREADS
    # q8 holds every pass' q at once
    assert layout.split_slots * layout.split_passes <= layout.o_k // 4
    assert layout.split_rows_whole == (tp != 8)
    assert layout.split_slots_whole == (tp != 8)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_p8_region_holds_the_selection_borrow(tp):
    """``p8`` carries the split's FP8 P and, on the same words, the indexer
    selection's candidates. Sizing it by P alone fails at TP8, where 8 q heads
    need 512 words and the selection needs 642."""
    layout = Layout(dims=Dims(tp=tp))
    assert layout.p8_words >= SEL_P8_WORDS
    assert (layout.p8_words == SEL_P8_WORDS) == (tp == 8)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_shared_expert_tasks_fit_one_per_cta(tp):
    """``start`` is taken mod BLOCKS, so past BLOCKS tasks the surplus silently
    never run. INTER scales as 1 / TP, so this holds from TP2 up."""
    assert Layout(dims=Dims(tp=tp)).sh_tasks <= BLOCKS


@pytest.mark.parametrize("tp", TP_SIZES)
def test_wide_moe_threshold_follows_inter(tp):
    """TP2's mids are twice TP4's, so its narrow pool stops fitting two widths
    earlier: measured at 168,416 B for S=3 against the 163,840 B limit."""
    layout = Layout(dims=Dims(tp=tp))
    assert layout.wide_from == (3 if tp == 2 else 5)
    assert [layout.wide_moe(s) for s in (1, 2, 3, 4, 5)] == (
        [False, False, True, True, True]
        if tp == 2
        else [False, False, False, False, True]
    )


@pytest.mark.parametrize("tp", TP_SIZES)
def test_pool_words_at_the_narrow_wide_seam(tp):
    """The narrow pool is sized by the mids (three ways at once), not by HIDDEN:
    sizing it from HIDDEN put TP2's mid tail outside the declared array."""
    layout = Layout(dims=Dims(tp=tp))
    got = [layout.pool_words(s) for s in (1, 2, 3, 4, 5, 16)]
    assert (
        got
        == {
            2: [6000, 12000, 16384, 16384, 16384, 35640],
            4: [3072, 6144, 9216, 12288, 16384, 31872],
            8: [3072, 6144, 9216, 12288, 11368, 31872],
        }[tp]
    )


@pytest.mark.parametrize("tp", TP_SIZES)
def test_stage_bases_place_the_later_stages_past_the_split_ctas(tp):
    """Only the shared expert's window moves with TP: it is ranked by the routed
    up / gate load, which follows ``ug_per_slot`` and so INTER."""
    layout = Layout(dims=Dims(tp=tp))
    bases = layout.stage_bases(4)
    assert bases == {
        "split": 0,
        "merge": 32,
        "o": 36,
        "router": 228,
        "ug": 0,
        "down": 32,
        "shared": {2: 36, 4: 132, 8: 180}[tp],
    }
    assert bases["merge"] == N_SPLIT * 4
    assert 0 <= bases["shared"] < BLOCKS


def test_ug_ctas_per_token_only_splits_a_dividing_width():
    """Round-robin (None) for S < 4 or a width that does not divide the grid."""
    assert [ug_ctas_per_token(s) for s in (1, 2, 3, 4, 8, 16)] == [
        None,
        None,
        None,
        64,
        32,
        16,
    ]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_ug_tasks_cover_every_routed_task(tp):
    """Whatever the placement, the CTAs between them run every token's
    TOP_K * ug_per_slot tasks exactly once."""
    layout = Layout(dims=Dims(tp=tp))
    for tokens in (1, 3, 4, 16):
        total = sum(layout.ug_tasks_of(c, tokens) for c in range(BLOCKS))
        assert total == 4 * layout.ug_per_slot * tokens


@pytest.mark.parametrize("tp", TP_SIZES)
def test_shared_after_routing_needs_an_even_ug_load(tp):
    """A shared-expert CTA routes first only where every CTA carries the same
    routed load, i.e. where TOP_K * ug_per_slot * S fills the grid. ug_per_slot
    follows INTER, so the width at which that happens doubles with each TP step:
    every second width at TP2, every fourth at TP4, every eighth at TP8."""
    layout = Layout(dims=Dims(tp=tp))
    every = {2: 2, 4: 4, 8: 8}[tp]
    assert [layout.shared_after_routing(s) for s in range(1, 17)] == [
        s % every == 0 for s in range(1, 17)
    ]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_narrow_opart_holds_every_reachable_width(tp):
    """``opart`` must hold the narrow MoE down stage's partials at every width
    that TP actually routes to the kernels.

    This is the test that was missing when TP8 width 4 faulted. ``opart`` is
    declared ``WAVES * o_k_sub`` floats on the narrow path
    (``post_attn.py:342``), and the down stage stores at
    ``((k + 1) * WAVES * 64 + wave * 64 + lane) * 4`` for k in 0..tokens-1
    (``post_attn.py:2827-2830``) -- so it needs ``(tokens + 1) * WAVES * 256``.
    Capacity follows ``o_k_sub = h_sub * 128`` and ``h_sub = local_q_heads //
    local_kv_heads``; because ``local_kv_heads`` floors at 1, TP8 gets 8 q heads
    to a GQA group where TP2 and TP4 get 16, so TP8's region is HALF the size
    while the need is TP-independent. TP8 fits exactly at 3 tokens and overruns
    by 8 KB at 4.

    The COMPILE_ONLY gate cannot catch this: the declaration is inside the LDS
    budget, and it is the dynamic extent of the use that runs past it. Only
    arithmetic over the reachable widths sees it, which is what this does.
    """
    layout = Layout(dims=Dims(tp=tp))
    waves = THREADS // 64
    # Every width the kernels can be BUILT for on the narrow path, not just the
    # ones WIDTH_RANGE currently routes: the TP8 floor of 5 is a quarantine of
    # this very defect, so keying the test off WIDTH_RANGE would make it pass by
    # construction at exactly the TP that is broken.
    narrow = [s for s in range(1, 17) if not layout.wide_moe(s)]
    assert narrow, f"TP{tp} has no narrow widths"
    for tokens in narrow:
        capacity = layout.narrow_opart_floats(tokens)
        # Each entry mirrors the kernel's STORE INDEX EXPRESSION and asserts the
        # highest element it can touch, NOT the capacity formula. That
        # distinction is the point: an earlier version of this test computed
        # ``(tokens + 1) * waves * 64 * 4``, which is what the helper itself
        # returns, so a wrong helper reproduced the same wrong bound here and
        # the test passed. Mirroring the addressing means the helper and the
        # assertion fail independently.
        h_sub = layout.h_sub
        highest = {
            # post_attn.py:1198-1209
            #   wave * O_K_SUB + (g4 * 4 + e) * HEAD_DIM + 16 * jd + l16,
            #   with (g4 * 4 + e) held under h_sub by the SPLIT_LIVE_G4 guard
            "attention PV store": (
                (waves - 1) * layout.o_k_sub + (h_sub - 1) * HEAD_DIM + HEAD_DIM - 1
            ),
            # post_attn.py:1231-1233, guarded to tid < SPLIT_SLOTS so the
            # four-per-thread walk covers exactly o_k_sub elements
            "attention PV reduce": ((waves - 1) * layout.o_k_sub + layout.o_k_sub - 1),
            # post_attn.py:2849
            #   ((k + 1) * WAVES * 64 + wave * 64 + lane) * 4 + 3,
            #   k running 0..tokens-1
            "narrow MoE down store": (
                (tokens * waves * 64 + (waves - 1) * 64 + 63) * 4 + 3
            ),
            # post_attn.py:1638-1639, reached with opart as ``sig`` from :2708
            "route_top4 scores": (tokens - 1) * N_ROUTED + 64 + 63,
            # post_attn.py:1978 and :2822, both (wave * 64 + lane) * 4 + 3
            "MoE up / shared partials": ((waves - 1) * 64 + 63) * 4 + 3,
        }
        for stage, index in highest.items():
            assert index < capacity, (
                f"TP{tp} width {tokens}: the {stage} reaches opart element "
                f"{index}, but the region is only {capacity} floats"
            )


def test_tp8_narrow_opart_is_the_known_defect_not_a_passing_case():
    """Negative control for the test above: prove it would FAIL if the floor of
    5 were lowered, rather than passing because the arithmetic is toothless.

    TP8 width 4 is the configuration that faulted on an 8-rank capture. The
    guard that keeps it out of production is ``WIDTH_RANGE[8]`` starting at 5,
    which is exactly ``wide_from`` -- not a fix to the region. If someone
    reduces that floor, the test above must start failing.
    """
    layout = Layout(dims=Dims(tp=8))
    waves = THREADS // 64
    assert layout.wide_from == 5
    # The OLD sizing -- the attention term alone -- is what the defect was.
    # Pinned here so the regression is described by a test rather than only by
    # a comment: it is 2,048 floats (8 KB) short at width 4, and fits width 3
    # with ZERO spare.
    old_capacity = waves * layout.o_k_sub
    assert (4 + 1) * waves * 64 * 4 - old_capacity == 2048
    assert (3 + 1) * waves * 64 * 4 == old_capacity
    # The fix covers width 4 and leaves TP2/TP4 untouched.
    assert layout.narrow_opart_floats(4) == 10240
    # The floor STAYS at wide_from until hardware says otherwise. Correct
    # sizing makes lowering it plausible, not proven: the (nil) fault has two
    # candidate causes and this addresses only one, so the floor may not drop
    # until a TP8 width-4 decode runs clean on 8 ranks.
    assert WIDTH_RANGE[8][0] == layout.wide_from, (
        "do not lower TP8's floor on the strength of the opart fix alone -- "
        "run a TP8 width-4 decode first; the other candidate (uneven per-token "
        "UG placement at width 4) is untouched by it"
    )
    for tp in (2, 4):
        other = Layout(dims=Dims(tp=tp))
        for s in range(1, other.wide_from):
            assert other.narrow_opart_floats(s) == waves * other.o_k_sub, (
                f"TP{tp} width {s}: the fix must not grow LDS here"
            )
