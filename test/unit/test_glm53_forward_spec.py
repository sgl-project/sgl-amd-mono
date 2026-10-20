"""The plugin consumes the CP argument and advances only selected epochs."""

from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.glm53.spec import _Built


def test_build_prepares_both_indexer_modes_before_forward(monkeypatch):
    from sglang.srt import runtime_context

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53 import spec
    from sgl_amd_mono.glm53.gfx950 import runner, weights

    events = []

    class Runtime:
        def __init__(self, key, rank, group, device):
            self.key, self.rank = key, rank
            self.scratch = torch.empty(0, dtype=torch.uint8, device=device)
            events.append(("build", key.rows, key.with_indexer, key.indexer_cp))

        def set_deadline(self, seconds):
            assert seconds == 30

        def precompile(self):
            events.append(("compile", self.key.rows))

    parallel = SimpleNamespace(
        tp_size=4, tp_rank=0, tp_group=SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        runtime_context,
        "process_model_config",
        lambda: SimpleNamespace(context_len=4096),
    )
    monkeypatch.setattr(guard, "fault_injection", lambda rank: None)
    monkeypatch.setattr(runner, "Glm53Runtime", Runtime)
    monkeypatch.setattr(weights, "layer_weights", lambda module, *args: module.w)
    table = torch.empty(4096, 32)
    monkeypatch.setattr(weights, "rope_tables", lambda *args: (table, table))
    modules = [
        SimpleNamespace(
            input_layernorm=SimpleNamespace(weight=torch.empty(0)),
            w=SimpleNamespace(index=object() if i == 3 else None, shuffled=True),
        )
        for i in range(5)
    ]
    lm = SimpleNamespace(model=SimpleNamespace(layers=modules))
    built = spec._build(
        lm,
        SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=5),
        SimpleNamespace(widths=(2, 4), copies={}, deadline_s=30),
    )
    assert list(built.runtimes) == [
        (w, kind) for w in (2, 4) for kind in ("index", "index_cp", "shared")
    ]
    assert [event[0] for event in events] == ["build"] * 6 + ["compile"] * 6
    for width in (2, 4):
        assert built._layers[3].ops[width]._runtimes == {
            False: built.runtimes[width, "index"],
            True: built.runtimes[width, "index_cp"],
        }
        assert built._layers[4].ops[width]._runtimes == {
            False: built.runtimes[width, "shared"],
        }


def test_only_selected_width_and_mode_consume_epochs():
    built = object.__new__(_Built)
    calls = []
    built.runtimes = {
        (width, kind): SimpleNamespace(
            advance_step=lambda w=width, k=kind: calls.append((w, k))
        )
        for width in (2, 4)
        for kind in ("index", "index_cp", "shared")
    }
    built._advance_step(2, True)
    built._advance_step(4, False)
    assert calls == [(2, "index_cp"), (2, "shared"), (4, "index"), (4, "shared")]


@pytest.mark.parametrize("cp", [False, True])
def test_fallback_does_not_receive_the_monokernel_only_argument(cp):
    class Model:
        def forward(self, *args, **kwargs):
            return args, kwargs

    built = object.__new__(_Built)
    built.lm, built.max_rows = Model(), 4
    plan = SimpleNamespace(key=SimpleNamespace(rows=5))
    args, kwargs = built.forward(plan, "ids", "pos", "batch", indexer_cp=cp)
    assert args == ("ids", "pos", "batch", None, None)
    assert kwargs == {}


def test_invalid_cp_fails_before_any_backbone_or_fallback():
    built = object.__new__(_Built)
    with pytest.raises(TypeError, match="Python bool"):
        built.forward(None, None, None, None, indexer_cp="1")
