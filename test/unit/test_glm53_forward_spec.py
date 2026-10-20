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
        SimpleNamespace(widths=(2, 4, 6, 8, 12, 16), copies={}, deadline_s=30),
    )
    assert list(built.runtimes) == [
        (w, kind)
        for w in (2, 4, 6, 8, 12, 16)
        for kind in ("index", "index_cp", "shared")
    ]
    assert [event[0] for event in events] == ["build"] * 18 + ["compile"] * 18
    for width in (2, 4, 6, 8, 12, 16):
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


@pytest.mark.parametrize("query", [1, 4, 6, 7, 8])
@pytest.mark.parametrize("expanded", [False, True])
def test_multitoken_metadata_is_causal_and_keeps_request_ownership(
    monkeypatch, query, expanded
):
    from sglang.srt.model_executor import forward_context
    from sglang.srt.monokernels.spec import Plan, ShapeKey

    from sgl_amd_mono.glm53.spec import TOPK, _width

    rows = 2 * query
    width = _width(rows)
    built = _Built(
        SimpleNamespace(model=None),
        {3: None},
        {(width, "index"): None},
        None,
        None,
        4,
        "cpu",
    )
    table = torch.tensor([[10, 7], [29, 4]], dtype=torch.int32)
    if expanded:
        table = table.repeat_interleave(query, dim=0)
    monkeypatch.setattr(
        forward_context,
        "get_attn_backend",
        lambda: SimpleNamespace(
            forward_metadata=SimpleNamespace(real_page_table=table)
        ),
    )
    # Cross a page boundary and the top-k limit, with a padding slot in request 1.
    positions = torch.cat(
        (torch.arange(63, 63 + query), torch.arange(2047, 2047 + query))
    )
    slots = torch.arange(1, rows + 1)
    slots[-1] = 0
    plan = Plan(
        key=ShapeKey(
            forward_mode="target_verify", batch_size=2, tokens_per_request=query
        )
    )
    batch = SimpleNamespace(
        out_cache_loc=slots, seq_lens=torch.tensor([63 + query, 2047 + query])
    )
    step = built._step(plan, positions, batch, rows, width)
    counts = (positions + 1).clamp(max=TOPK)
    counts[-1] = 0
    assert torch.equal(step.indptr[1 : rows + 1] - step.indptr[:rows], counts.int())
    assert (step.indptr[rows:] == counts.sum()).all()
    assert torch.equal(step.positions[:rows], positions)
    assert (step.positions[rows:] == 0).all()
    assert step.slots[-1] == -1 and (step.slots[rows:] == -1).all()
    owners = (
        torch.arange(rows) if expanded else torch.arange(2).repeat_interleave(query)
    )
    assert torch.equal(step.req_ids[:rows], owners.int())
    assert torch.equal(step.page_table[step.req_ids[:rows]], table[owners])
    assert built._step(plan, positions, batch, rows, width) is step


def test_aux_capture_keeps_unpadded_input_before_reused_output(monkeypatch):
    from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
    from sglang.srt.layers.layer_boundary.residual import batch as residual

    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    rows, width = 14, 16
    built = _Built(
        SimpleNamespace(model=None),
        {3: None},
        {(width, "shared"): None},
        None,
        None,
        4,
        "cpu",
    )
    built.check_layers = False
    monkeypatch.setattr(built, "_step", lambda *args: object())
    monkeypatch.setattr(built, "_bind", lambda *args: None)
    monkeypatch.setattr(built, "_advance_step", lambda *args: None)
    monkeypatch.setattr(residual, "fold", lambda h, batch: h + 2)
    monkeypatch.setattr(residual, "take_output", lambda h, batch: h)
    monkeypatch.setattr(residual, "set_written", lambda h, batch: h)
    layer = SimpleNamespace(
        i=3,
        w=SimpleNamespace(index=object()),
        caches=None,
        ops={width: SimpleNamespace(forward=lambda h, *args, **kwargs: h + 10)},
    )
    hidden = torch.ones(rows, HIDDEN, dtype=torch.bfloat16)
    capture = AuxHiddenStatePacker(1)
    result = built._layer(
        layer,
        SimpleNamespace(key=SimpleNamespace(rows=rows)),
        None,
        hidden,
        None,
        None,
        False,
        capture,
    )
    built.pad_h[width].zero_()
    assert result.shape == hidden.shape and (result == 13).all()
    assert capture.finalize().shape == hidden.shape
    assert (capture.finalize() == 3).all()
