# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3's per-rank shapes at each supported TP size, pinned as literals.

These values were derived by module-level constants in the kernels' in-tree
ancestor, where TP was fixed at import. Moving them onto :class:`Dims` had to be
value-exact -- a TP4 build at width 16 uses 163,664 B of the 163,840 B LDS
budget, so a derivation that grew any TP4 shape by one head would stop the
working build compiling. Every value below was compared against that ancestor at
TP 2, 4 and 8 before it was written down here; this test keeps them from
drifting afterwards.
"""

import pytest

from sgl_amd_mono.minimax_m3.gfx950.config import (
    TOTAL_INDEX_HEADS,
    TP_SIZES,
    Dims,
    IndexHeads,
)

# tp -> every shape a rank's share depends on
EXPECTED = {
    2: dict(
        local_q_heads=32,
        local_kv_heads=2,
        local_index_heads=2,
        inter=1536,
        scale_cols=48,
        padded_scale_cols=48,
        o_k=4096,
        max_qkv_rows=5248,
        own_rows=4992,
        width_range=(1, 12),
        score_blocks=8192,
        score_words=16384,
        iq_off=4608,
        ik_off=4864,
    ),
    4: dict(
        local_q_heads=16,
        local_kv_heads=1,
        local_index_heads=1,
        inter=768,
        scale_cols=24,
        padded_scale_cols=24,
        o_k=2048,
        max_qkv_rows=2944,
        own_rows=2560,
        width_range=(1, 16),
        score_blocks=8192,
        score_words=8192,
        iq_off=2304,
        ik_off=2432,
    ),
    8: dict(
        local_q_heads=8,
        local_kv_heads=1,
        local_index_heads=1,
        inter=384,
        scale_cols=12,
        padded_scale_cols=16,
        o_k=1024,
        max_qkv_rows=1920,
        own_rows=1536,
        width_range=(5, 16),
        score_blocks=8192,
        score_words=8192,
        iq_off=1280,
        ik_off=1408,
    ),
}


@pytest.mark.parametrize("tp", TP_SIZES)
def test_rank_shapes(tp):
    dims, want = Dims(tp=tp), EXPECTED[tp]
    heads = dims.index_heads
    got = dict(
        local_q_heads=dims.local_q_heads,
        local_kv_heads=dims.local_kv_heads,
        local_index_heads=dims.local_index_heads,
        inter=dims.inter,
        scale_cols=dims.scale_cols,
        padded_scale_cols=dims.padded_scale_cols,
        o_k=dims.o_k,
        max_qkv_rows=dims.max_qkv_rows,
        own_rows=heads.rows,
        width_range=dims.width_range,
        score_blocks=heads.score_blocks,
        score_words=heads.score_words,
        iq_off=heads.iq_off,
        ik_off=heads.ik_off,
    )
    assert got == want


def test_kv_heads_replicate_past_tp4():
    """The model has 4 KV heads, so TP8 replicates one rather than splitting.

    ``Shard.split`` is right for the MoE widths and wrong here: it would refuse
    TP8 outright on a 4 % 8 remainder.
    """
    assert [Dims(tp=tp).local_kv_heads for tp in (2, 4, 8)] == [2, 1, 1]
    assert [Dims(tp=tp).local_index_heads for tp in (2, 4, 8)] == [2, 1, 1]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_padded_scale_cols_only_pads_at_tp8(tp):
    """aiter's e8m0_shuffle pads MXFP4 scale columns to a multiple of 8, which
    bites only where the rank's column count is not already one: TP8's 12 -> 16.
    Indexing that tensor by ``scale_cols`` would read the wrong stride."""
    dims = Dims(tp=tp)
    assert (dims.padded_scale_cols != dims.scale_cols) == (tp == 8)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_max_qkv_rows_bounds_both_index_modes(tp):
    """``max_qkv_rows`` must bound CP's every-index-head projection, not just
    the rank's own heads. It was ``qkv_rows(tp)``, which coincides with
    ``qkv_rows(TOTAL_INDEX_HEADS)`` at TP4 and nowhere else -- too small at TP2
    for a CP build (4992 against 5248)."""
    dims = Dims(tp=tp)
    assert dims.max_qkv_rows == dims.qkv_rows(TOTAL_INDEX_HEADS)
    assert dims.max_qkv_rows >= dims.index_heads.rows


@pytest.mark.parametrize("tp", TP_SIZES)
def test_index_heads_refuses_a_count_neither_mode_takes(tp):
    dims = Dims(tp=tp)
    with pytest.raises(ValueError, match="index heads"):
        IndexHeads(dims=dims, count=dims.local_index_heads + TOTAL_INDEX_HEADS)


def test_unsupported_tp_is_refused():
    for tp in (1, 3, 16):
        with pytest.raises(ValueError, match="TP"):
            Dims(tp=tp)


def test_tp_is_the_only_field_so_the_kernel_cache_cannot_go_stale():
    """``tp`` must stay the ONLY field of :class:`Dims`. This is not tidiness.

    FlyDSL identifies a compiled kernel partly by the *scalar* values the jit
    function closes over -- where scalar means
    ``(int, float, bool, str, NoneType, tuple, Enum)``
    (``flydsl/compiler/jit_function.py:140``). Anything else is skipped
    silently (``:479-493``). A ``msgspec.Struct`` is not a scalar, so **no
    value carried by ``Dims`` reaches that key**.

    Today that is harmless: every other shape is a property derived from
    ``tp``, and ``tp`` reaches the key anyway as the plain ``int`` ``npes``
    the kernel builders also take. Measured on the real builder at equal row
    counts, TP 2, 4 and 8 give three distinct keys, with the same TP computed
    twice as the stability control.

    Add a second field -- an env-driven INTER padding is exactly the shape of
    thing that would want one -- and that stops being true: the kernel would
    change while nothing in the key did.

    Scope, so this is not read as more than it is. The measurement above is of
    ``_jit_function_cache_key``, and the artifact on disk is named by
    ``_build_full_cache_key`` (a globals snapshot plus argument signatures,
    ``:1363-1371``). This test pins the invariant that keeps ``tp`` reaching a
    scalar at all; it is not a proof that the disk cache is sound. A separate,
    open question is that the globals snapshot captures nothing from
    ``sgl_amd_mono/common/`` -- measured, 73 globals captured and zero from
    there.
    """
    assert Dims.__struct_fields__ == ("tp",)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_min_width_env_is_absent_by_default(tp, monkeypatch):
    """The hook must not change anything unless it is set. Guards against the
    override leaking in from a developer's shell or a container's env block."""
    monkeypatch.delenv("SGLANG_TEST_MONO_MIN_WIDTH", raising=False)
    assert Dims(tp=tp).width_range == {2: (1, 12), 4: (1, 16), 8: (5, 16)}[tp]


def test_min_width_env_lowers_the_floor_and_the_built_widths_follow(monkeypatch):
    """Lowering the floor must move BOTH what is built and what is routed.

    ``widths`` drives the kernels that get compiled and ``spec.py`` reads
    ``width_range`` to decide which steps go to them. They share this one
    property precisely so an override cannot route a step to a kernel that was
    never built; this pins that.
    """
    monkeypatch.setenv("SGLANG_TEST_MONO_MIN_WIDTH", "1")
    dims = Dims(tp=8)
    assert dims.width_range == (1, 16)
    assert dims.widths == tuple(range(1, 17))
    assert 4 in dims.widths  # the width the TP8 floor exists to exclude


def test_min_width_env_can_also_raise_the_floor(monkeypatch):
    monkeypatch.setenv("SGLANG_TEST_MONO_MIN_WIDTH", "8")
    assert Dims(tp=2).width_range == (8, 12)


@pytest.mark.parametrize("bad", ["0", "17", "-1", "", "five", "4.5"])
def test_min_width_env_refuses_a_value_it_cannot_honour(bad, monkeypatch):
    """Loud, not silent: this is operator input, and a typo that quietly fell
    back to the default would make a TP8 experiment look like a clean result
    when the floor never moved."""
    monkeypatch.setenv("SGLANG_TEST_MONO_MIN_WIDTH", bad)
    with pytest.raises(ValueError, match="SGLANG_TEST_MONO_MIN_WIDTH"):
        Dims(tp=8).width_range
