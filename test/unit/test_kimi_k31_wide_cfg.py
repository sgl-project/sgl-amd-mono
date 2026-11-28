# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 wide launch tile configs keep every tile stage of a launch in one BLOCKS round.

A launch runs on BLOCKS CTAs; a tile stage with more tasks than BLOCKS sends CTAs
through a second round of its tiles. The task counts come from the launches' own
plans (``layer.plan`` for a KDA layer, ``dsa_post.plan`` for a DSA layer), so they
are what the kernels run, on CPU.
"""

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
from sgl_amd_mono.kimi_k31 import wide
from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp
from sgl_amd_mono.kimi_k31.gfx950 import layer as lay
from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

# the stages whose task count the tile configs set; the launches' other stages take one
# task a row, a (row, rank) or a (request, head), whatever the configs
TILE_STAGES = ("up", "inproj", "oproj", "rsred", "front", "sdown")

# DSpark verify tokens a request and the most requests a step holds at each deployed
# point (agentx/opt.sh after its run scripts and layouts: 8 tokens at c1/c4/c8, 6 at
# c16-c56; c1 runs Kimi-K3's narrow launch). The decode graphs cover every batch up to
# the most, so a point runs every request count, and every bucket, up to it.
RECIPE = {
    "c1": (8, 1),
    "c4": (8, 2),
    "c8": (8, 8),
    "c16": (6, 8),
    "c24": (6, 12),
    "c32": (6, 16),
    "c48": (6, 48),
    "c56": (6, 56),
}


def _cfg(cfgs) -> lay.LayerCfg:
    return lay.LayerCfg(*(Gemm(*(int(v) for v in c.split(":"))) for c in cfgs))


def _tile_tasks(b: int, L: int, cfgs) -> dict:
    """(launch, stage) -> tasks of every tile stage of bucket b's launches."""
    cfg = _cfg(cfgs)
    plans = {
        "kda": lay.plan(b, cfg, L, 3, 4, wide.NF, False)[0],
        "kda attn_res capture": lay.plan(b, cfg, L, 3, 4, wide.NF, "attn_res")[0],
        "kda prefix capture": lay.plan(b, cfg, L, 3, 4, wide.NF, "prefix")[0],
        "dsa_post": dp.plan(b, cfg.oproj, cfg.front, cfg.sdown, wide.NF, 3)[0],
    }
    tasks = {
        (kind, s.name): s.tasks
        for kind, pl in plans.items()
        for s in pl.stages
        if s.name in TILE_STAGES
    }
    # a renamed or dropped stage must not silently leave the check
    for kind in plans:
        want = TILE_STAGES if kind.startswith("kda") else TILE_STAGES[2:]
        stages = {n for k, n in tasks if k == kind}
        assert stages == set(want), (kind, sorted(tasks))
    return tasks


def _over(b: int, L: int, cfgs) -> dict:
    return {k: t for k, t in _tile_tasks(b, L, cfgs).items() if t > BLOCKS}


@pytest.mark.parametrize(
    "point", sorted(p for p, (L, reqs) in RECIPE.items() if L * reqs > wide.NARROW)
)
def test_every_recipe_bucket_fits_one_round(point: str) -> None:
    L, max_reqs = RECIPE[point]
    assert L in wide.SUPPORTED_L
    over = {}
    for r in wide.REQ_BUCKETS:
        if r <= max_reqs and L * r > wide.NARROW:
            b = L * r
            if bad := _over(b, L, wide.cfg_rows(b, L)):
                over[b] = bad
    assert not over, (point, over)


def _routed(L: int) -> set:
    """Request counts the spec routes to a launch at L draft tokens: the binding takes every
    width of MonoSpec.widths (1 .. NARROW, then wide_widths) that is a multiple of L."""
    widths = (*range(1, wide.NARROW + 1), *wide.wide_widths())
    return {w // L for w in widths if w % L == 0}


def test_every_recipe_step_is_routed_to_a_launch(monkeypatch) -> None:
    """Every request count each point runs reaches a launch; one that does not runs SGLang's
    whole step (or pads to the next graph and pays for the padding)."""
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    missing = {
        point: sorted(set(range(1, max_reqs + 1)) - _routed(L))
        for point, (L, max_reqs) in RECIPE.items()
    }
    assert not any(missing.values()), missing


def test_every_recipe_step_past_narrow_runs_the_wide_launch(monkeypatch) -> None:
    """A routed step past NARROW rows is one the wide launch serves at the point's L (not
    another L's width that falls back to SGLang), and a bucket holds it."""
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    off = {}
    for point, (L, max_reqs) in RECIPE.items():
        served, top = set(wide.widths_at(L)), wide.buckets(L)[-1]
        rows = [L * r for r in range(1, max_reqs + 1) if L * r > wide.NARROW]
        if bad := [b for b in rows if b not in served or b > top]:
            off[point] = bad
    assert not off, off


@pytest.mark.parametrize("L", wide.SUPPORTED_L)
def test_every_draft_length_serves_every_request_count(L: int, monkeypatch) -> None:
    """Each L serves 2 .. REQ_BUCKETS[-1] requests: no width is dropped because another
    draft length reads it as more requests than a bucket holds."""
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    want = {r for r in range(2, wide.REQ_BUCKETS[-1] + 1) if L * r > wide.NARROW}
    assert want <= _routed(L), sorted(want - _routed(L))
    assert set(wide.widths_at(L)) == {L * r for r in want}


@pytest.mark.parametrize("L", wide.SUPPORTED_L)
def test_every_bucket_fits_one_round(L: int) -> None:
    over = {}
    for r in wide.REQ_BUCKETS:
        b = L * r
        if b > wide.NARROW and (bad := _over(b, L, wide.cfg_rows(b, L))):
            over[b] = bad
    assert not over, (L, over)


def test_every_row_count_fits_one_round() -> None:
    """Not only today's buckets: a bucket or draft length added later within 9 ..
    max(SUPPORTED_L) x REQ_BUCKETS[-1] rows lands on a row count kept in one round."""
    top = max(wide.SUPPORTED_L) * wide.REQ_BUCKETS[-1]
    over = {}
    for b in range(wide.NARROW + 1, top + 1):
        if bad := _over(b, 1, wide._row_cfgs(b)):
            over[b] = bad
    assert not over, over


def test_every_row_count_config_meets_tile_constraints() -> None:
    """The tile constraints the launch builds assert, checked on CPU at every row count."""
    top = max(wide.SUPPORTED_L) * wide.REQ_BUCKETS[-1]
    for b in range(wide.NARROW + 1, top + 1):
        cfg = _cfg(wide._row_cfgs(b))
        up, inp, oproj, front, sdown = cfg.up, cfg.inp, cfg.oproj, cfg.front, cfg.sdown
        for g in (up, inp, oproj, front, sdown):
            # tile_mma: 8 waves, 16 x 16 MFMA tiles a wave
            assert g.wm * g.wn == 8, (b, g)
            assert g.bm % (16 * g.wm) == 0 and g.bn % (16 * g.wn) == 0, (b, g)
            assert BLOCKS % g.ks == 0, (b, g)
        # layer.kda_layer: split-K up / in_proj / front, one slice shared down
        assert up.ks > 1 and inp.ks > 1 and front.ks > 1 and sdown.ks == 1, (b, cfg)
        assert up.bm % up.ks == 0 and oproj.bm % oproj.ks == 0, (b, cfg)
        assert wide.SL % up.bn == 0 and wide.SL % oproj.bn == 0, (b, cfg)
        assert wide.H % sdown.bn == 0 and wide.H % oproj.bn == 0, (b, cfg)
        # K slices: whole 64-column steps (gemm_coll, chain), o_proj slices of whole heads
        # (kda_layer), and slices a rank's latent / hidden shard divides or is divided by
        ks_of = ((wide.LAT, up), (wide.H, inp), (wide.H, front), (lay.KO, oproj))
        for k, g in ks_of:
            assert k % g.ks == 0 and (k // g.ks) % 64 == 0, (b, k, g)
        assert (lay.KO // oproj.ks) % k1.HD == 0, (b, cfg)
        ktu, lsl = wide.LAT // up.ks, wide.LAT // wide.TP
        assert ktu % lsl == 0 or lsl % ktu == 0, (b, cfg)
        for kt in (wide.H // inp.ks, wide.H // front.ks):
            assert kt % wide.SL == 0 or wide.SL % kt == 0, (b, cfg)


def test_row_ranges_are_sorted_and_disjoint() -> None:
    for table in (
        wide.FRONT_ROW_RANGES,
        wide.INP_ROW_RANGES,
        wide.OPROJ_ROW_RANGES,
        wide.UP_ROW_RANGES,
        wide.SDOWN_ROW_RANGES,
    ):
        prev_hi = -1
        for lo, hi, _ in table:
            assert lo <= hi
            assert lo > prev_hi, f"overlapping or unsorted range {(lo, hi)}"
            prev_hi = hi
