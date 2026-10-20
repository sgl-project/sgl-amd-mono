"""Forward dispatch chooses a prebuilt CP variant before touching device state."""

from types import SimpleNamespace

import msgspec
import pytest
import torch

from sgl_amd_mono.glm53.gfx950.runner import Glm53Build, Glm53LayerOp


def runtime(*, indexed=True, cp=False, rank=0, **changes):
    key = Glm53Build(
        rows=4,
        tp=4,
        launches_per_step=3,
        with_indexer=indexed,
        index_max_seq=4096 if indexed else 0,
        fp8_shuffled=True,
        indexer_cp=cp,
    )
    return SimpleNamespace(
        key=msgspec.structs.replace(key, **changes),
        rank=rank,
        scratch=torch.empty(0, dtype=torch.uint8),
    )


def op(base, cp=None):
    weights = SimpleNamespace(
        index=object() if base.key.with_indexer else None, shuffled=True
    )
    return Glm53LayerOp(base, weights, 0, cp_runtime=cp)


def test_forward_selects_prebuilt_runtime_and_ignores_process_environment(monkeypatch):
    base, cp = runtime(), runtime(cp=True)
    layer = op(base, cp)
    calls = []
    monkeypatch.setattr(layer, "_launch", lambda rt, *args: calls.append((rt, args)))
    monkeypatch.setenv("SGLANG_AMD_MONO_GLM_INDEXER_CP", "1")
    args = tuple(object() for _ in range(5))
    layer.forward(*args)
    layer.forward(*args, indexer_cp=True)
    layer.forward(*args, indexer_cp=False)
    assert [rt for rt, _ in calls] == [base, cp, base]
    assert all(got == args for _, got in calls)


@pytest.mark.parametrize("value", [0, 1, "1", None, torch.tensor(True)])
def test_forward_refuses_non_python_bools_before_launch(value, monkeypatch):
    layer = op(runtime(), runtime(cp=True))
    monkeypatch.setattr(layer, "_launch", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(TypeError, match="Python bool"):
        layer.forward(*([None] * 5), indexer_cp=value)


@pytest.mark.parametrize("built_cp", [False, True])
def test_forward_does_not_lazily_create_an_unbuilt_variant(built_cp, monkeypatch):
    layer = op(runtime(cp=built_cp))
    monkeypatch.setattr(layer, "_launch", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="not built before forward"):
        layer.forward(*([None] * 5), indexer_cp=not built_cp)


def test_reuse_only_layer_uses_its_original_runtime_for_both_modes(monkeypatch):
    base = runtime(indexed=False)
    layer = op(base)
    monkeypatch.setattr(layer, "_launch", lambda rt, *args: rt)
    for cp in (False, True):
        assert layer.forward(*([None] * 5), indexer_cp=cp) is base


@pytest.mark.parametrize(
    "changes", [{"rows": 2}, {"rank": 1}, {"tp": 8}, {"cp": False}]
)
def test_variants_must_have_matching_geometry_and_rank(changes):
    with pytest.raises(ValueError, match="CP runtime must match"):
        op(runtime(), runtime(**{"cp": True, **changes}))
