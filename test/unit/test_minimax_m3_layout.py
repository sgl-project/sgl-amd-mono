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
from sgl_amd_mono.minimax_m3.gfx950.config import TP_SIZES, Dims
from sgl_amd_mono.minimax_m3.gfx950.layout import (
    N_DN,
    N_O,
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
        h=32, h_sub=16, o_k=4096, o_k_sub=2048,
        split_passes=2, split_rows_whole=True, split_live_g4=4,
        split_slots=512, split_slots_whole=True,
        p8_words=2048, kv_page_bytes=4096, iscore_blocks=16384,
        ug_per_slot=96, sh_tasks=192, wide_from=3,
        mid_row=388, midsc_row=52, o_row=1028,
        merge_slots=1024, merge_passes=2,
        merge_rounds_whole=True, merge_tail_slots=512,
        mid_copy_rounds=8, mid8_slots=960, mid8_rounds=2,
        scratch_bytes=6526720,
    ),
    4: dict(
        h=16, h_sub=16, o_k=2048, o_k_sub=2048,
        split_passes=1, split_rows_whole=True, split_live_g4=4,
        split_slots=512, split_slots_whole=True,
        p8_words=1024, kv_page_bytes=2048, iscore_blocks=8192,
        ug_per_slot=48, sh_tasks=96, wide_from=5,
        mid_row=196, midsc_row=28, o_row=516,
        merge_slots=512, merge_passes=1,
        merge_rounds_whole=True, merge_tail_slots=512,
        mid_copy_rounds=4, mid8_slots=480, mid8_rounds=1,
        scratch_bytes=3758080,
    ),
    8: dict(
        h=8, h_sub=8, o_k=1024, o_k_sub=1024,
        split_passes=1, split_rows_whole=False, split_live_g4=2,
        split_slots=256, split_slots_whole=False,
        p8_words=642, kv_page_bytes=2048, iscore_blocks=8192,
        ug_per_slot=24, sh_tasks=48, wide_from=5,
        mid_row=100, midsc_row=16, o_row=260,
        merge_slots=256, merge_passes=1,
        merge_rounds_whole=False, merge_tail_slots=256,
        mid_copy_rounds=2, mid8_slots=240, mid8_rounds=1,
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
        "iscore", "sp_o", "sp_m", "sp_l", "attnq", "attnq_s", "a", "xn8",
        "xsc", "scores", "mid", "sel_long", "xn8p", "xscp", "xndone",
        "mid8p", "midsp", "ugdone", "_bytes",
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
        [False, False, True, True, True] if tp == 2
        else [False, False, False, False, True]
    )


@pytest.mark.parametrize("tp", TP_SIZES)
def test_pool_words_at_the_narrow_wide_seam(tp):
    """The narrow pool is sized by the mids (three ways at once), not by HIDDEN:
    sizing it from HIDDEN put TP2's mid tail outside the declared array."""
    layout = Layout(dims=Dims(tp=tp))
    got = [layout.pool_words(s) for s in (1, 2, 3, 4, 5, 16)]
    assert got == {
        2: [6000, 12000, 16384, 16384, 16384, 35640],
        4: [3072, 6144, 9216, 12288, 16384, 31872],
        8: [3072, 6144, 9216, 12288, 11368, 31872],
    }[tp]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_stage_bases_place_the_later_stages_past_the_split_ctas(tp):
    """Only the shared expert's window moves with TP: it is ranked by the routed
    up / gate load, which follows ``ug_per_slot`` and so INTER."""
    layout = Layout(dims=Dims(tp=tp))
    bases = layout.stage_bases(4)
    assert bases == {
        "split": 0, "merge": 32, "o": 36, "router": 228, "ug": 0, "down": 32,
        "shared": {2: 36, 4: 132, 8: 180}[tp],
    }
    assert bases["merge"] == N_SPLIT * 4
    assert 0 <= bases["shared"] < BLOCKS


def test_ug_ctas_per_token_only_splits_a_dividing_width():
    """Round-robin (None) for S < 4 or a width that does not divide the grid."""
    assert [ug_ctas_per_token(s) for s in (1, 2, 3, 4, 8, 16)] == [
        None, None, None, 64, 32, 16,
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
