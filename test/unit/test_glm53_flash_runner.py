"""The GLM-5.3-Flash launch from the tokens keeps a captured graph's step-input table
apart from eager steps (CPU tensors stand in for the step's), and a KDA launch
refuses a state or verify scratch of another dtype than its build's. Importing the
runner and the kernel's build key reaches AITER's GPU arch detection, so the cases
need a visible GPU."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the runner's imports need a visible GPU"
)


def _op():
    from sgl_amd_mono.glm53_flash.gfx950 import runner
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild

    key = FlashBuild(
        rows=4, tp=4, launches_per_step=1, inter=512, kda=True, dense=True, tokens=True
    )
    t = torch.zeros(1)
    dense = runner.FlashDenseWeights(*([t] * 8))
    embedding = runner.FlashEmbedding(weight=torch.zeros(8, 4), start=0, end=8)
    rt = SimpleNamespace(key=key, width=4)
    return runner.FlashLayerOp(rt, dense, 0, kda=object(), embedding=embedding)


def _step(n):
    from sgl_amd_mono.glm53_flash.gfx950.layout import ROW_WORDS

    ids = torch.zeros(n, dtype=torch.int64)
    return {
        "ids": ids,
        "positions": ids.clone(),
        "out_cache_loc": ids.clone(),
        "req_pool_indices": ids.clone(),
        "slots": torch.zeros(4, dtype=torch.int32),
        "rows": torch.zeros(4, ROW_WORDS, dtype=torch.int32),
    }


def test_an_eager_step_never_rewrites_a_captured_table(monkeypatch):
    op, graph_step = _op(), _step(3)
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    # the eager step before the capture makes the table; the capture reads it
    eager = op._step_table(graph_step)
    capturing[0] = True
    captured = op._step_table(graph_step)
    assert captured is eager
    held = captured.clone()
    # later eager steps of the same row count, on other tensors, take their own
    # tables; the captured one keeps its values
    capturing[0] = False
    for _ in range(3):
        other = op._step_table(_step(3))
        assert other is not captured
    assert torch.equal(captured, held)
    assert op._step_table(graph_step) is captured
    # only the latest uncaptured table of a row count is kept
    assert len(op.step_tabs) == 2
    # a capture of inputs no eager step used first cannot fill a table
    capturing[0] = True
    with pytest.raises(RuntimeError, match="no eager step used first"):
        op._step_table(_step(3))


def test_a_kda_launch_refuses_a_state_of_another_dtype_than_its_build():
    from sgl_amd_mono.glm53_flash.gfx950 import runner
    from sgl_amd_mono.glm53_flash.gfx950.config import HC_MULT, HIDDEN
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild

    S = 4
    ins = (
        torch.zeros(S, HIDDEN, dtype=torch.bfloat16),
        torch.zeros(S, HC_MULT, HIDDEN, dtype=torch.bfloat16),
        torch.zeros(S, HC_MULT, 1),
        torch.zeros(S, HC_MULT, HC_MULT),
    )
    idx = torch.zeros(S, dtype=torch.int32)

    def launch(state_bf16, verify, rec, ir=None):
        key = FlashBuild(
            rows=S,
            tp=4,
            launches_per_step=1,
            inter=512,
            kda=True,
            state_bf16=state_bf16,
            verify=verify,
        )
        rt = SimpleNamespace(key=key, width=S, outs=[(None,) * 4] * 2)
        op = runner.FlashLayerOp(rt, None, 0, kda=object())
        state = runner.FlashKdaState(
            conv_state=torch.zeros(1, dtype=torch.bfloat16),
            rec_state=torch.zeros(1, dtype=rec),
            inter_rec=None if ir is None else torch.zeros(1, dtype=ir),
            inter_conv=None if ir is None else torch.zeros(1, dtype=torch.bfloat16),
        )
        op.launch(*ins, state=state, state_idx=idx)

    f32, bf16 = torch.float32, torch.bfloat16
    for built, rec in ((True, f32), (False, bf16)):
        with pytest.raises(ValueError, match="built for a .* KDA state"):
            launch(built, 1, rec)
    for built, rec, ir in ((True, bf16, f32), (False, f32, bf16)):
        with pytest.raises(ValueError, match="built for a .* verify scratch"):
            launch(built, 4, rec, ir)
