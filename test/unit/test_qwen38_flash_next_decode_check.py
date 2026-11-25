"""The Qwen3.8-Flash-Next decode check mode on CPU: its metrics against the
path's bounds, the worst value kept per metric, the radix track-save comparison,
the fed QSA projections against SGLang's, and the MXFP4 reference's GDN projections
on stand-ins for SGLang's GDN modules."""

import math
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from qwen38_flash_next_decode_fakes import decode_mxfp4

from sgl_amd_mono.qwen38_flash_next.decode import check as C
from sgl_amd_mono.qwen38_flash_next.decode import spec as D
from sgl_amd_mono.qwen38_flash_next.decode import weights as W

# what the check records
METRICS = {
    *(
        f"{kind}.{out}.cosine"
        for kind in ("gdn", "qsa")
        for out in ("streams", "delta")
    ),
    *(f"gdn.{out}.cosine" for out in ("conv", "state", "update")),
    *(f"qsa.{out}.cosine" for out in ("mix", "qkv", "ix")),
    "gdn.track.max_abs",
}


def test_cosine():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(C.cosine(a, 2 * a) - 1.0) < 1e-12
    assert abs(C.cosine(a, -a) + 1.0) < 1e-12
    assert math.isnan(C.cosine(a, torch.tensor([1.0, float("nan"), 3.0])))


def test_violations_against_the_spec_bounds():
    bounds = dict(D.BOUNDS)
    values = {
        "qsa.mix.cosine": (0.99999, 3),
        "gdn.delta.cosine": (0.2, 4),  # a wrong layout
        "gdn.state.cosine": (float("nan"), 8),
        "qsa.delta.cosine": (bounds["qsa.delta.cosine"], 3),  # at the bound: passes
    }
    assert C.violations(values, bounds) == [
        "layer 4 gdn.delta.cosine = 0.2, bound 0.9",
        "layer 8 gdn.state.cosine = nan, bound 0.999",
    ]


def test_every_bound_names_a_metric_the_check_records():
    assert set(D.BOUNDS) == METRICS


def test_check_bounds_by_reference():
    assert D.check_bounds("bf16") == dict(D.BOUNDS)
    mx = D.check_bounds("mxfp4")
    # every metric bounded; the metrics without the MoE each at least as tight as
    # against the bf16 reference
    assert set(mx) == METRICS
    cos = {f"gdn.{o}.cosine" for o in ("conv", "state", "update")}
    cos |= {f"qsa.{o}.cosine" for o in ("mix", "qkv", "ix")}
    assert all(mx[k] >= D.BOUNDS[k] for k in cos)
    assert mx["gdn.track.max_abs"] == D.BOUNDS["gdn.track.max_abs"] == 0.0


def _step(**attrs):
    step = object.__new__(C.StepCheck)
    step.worst, step.states = {}, {}
    step.__dict__.update(attrs)
    return step


def test_note_keeps_the_worst_and_a_nan():
    step = _step()
    step._note("m.x.cosine", 0.99, 1)
    step._note("m.x.cosine", 0.999, 2)
    assert step.worst["m.x.cosine"] == (0.99, 1)
    step._note("m.x.cosine", 0.5, 3)
    assert step.worst["m.x.cosine"] == (0.5, 3)
    step._note("m.x.cosine", float("nan"), 4)
    step._note("m.x.cosine", 0.1, 5)
    v, layer = step.worst["m.x.cosine"]
    assert math.isnan(v) and layer == 4


def test_track_compares_tracked_rows_track_slots_with_their_working_slots():
    """Rows 0 and 2 tracked (working slots 3, 5 -> track slots 6, 8); row 1 is not,
    and its track slot 7 differs without counting. A missed copy shows as max_abs > 0."""
    g = torch.Generator().manual_seed(3)
    states = {
        i: (
            torch.randn(9, 6, 3, generator=g).bfloat16(),
            torch.randn(9, 2, 4, 4, generator=g),
        )
        for i in (0, 4)
    }
    md = NS(
        mamba_cache_indices=torch.tensor([3, 4, 5, 0]),
        mamba_track_indices=torch.tensor([6, 7, 8, 0]),
    )
    # row 3: past the step's rows
    fb = NS(mamba_track_mask=torch.tensor([True, False, True, True]))
    for conv, ssm in states.values():
        for t in (conv, ssm):
            t[[6, 8]] = t[[3, 5]]
    step = _step(rows=3, states=states)
    step.track(fb, md)
    assert step.worst == {"gdn.track.max_abs": (0.0, 0)}
    states[4][1][8, 1, 2, 3] += 0.5
    step.track(fb, md)
    assert step.worst["gdn.track.max_abs"] == (0.5, 4)
    assert C.violations(step.worst, D.check_bounds("mxfp4")) == [
        "layer 4 gdn.track.max_abs = 0.5, bound 0"
    ]
    step.worst = {}
    step.track(NS(mamba_track_mask=None), md)  # radix cache off
    assert step.worst == {}


class _Linear(torch.nn.Module):
    """A stand-in for SGLang's linears: forward -> (output, bias)."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        return F.linear(x, self.w), None


def test_front_compares_the_fed_projections_with_sglang_s():
    g = torch.Generator().manual_seed(5)
    w_qkv, w_ix = torch.randn(12, 16, generator=g), torch.randn(4, 16, generator=g)
    state = torch.randn(3, 32, generator=g)
    layer = NS(
        attn_hyper_connection=NS(mix=lambda s: (s[:, :16] * 2,)),
        qkv_proj=_Linear(w_qkv),
        indexer=NS(index_qk_proj=_Linear(w_ix)),
    )
    x = state[:, :16] * 2  # the launch's block input: SGLang's mix of the state
    step = _step()
    step.front(7, layer, state, x, F.linear(x, w_qkv), -F.linear(x, w_ix))
    assert step.worst["qsa.mix.cosine"] == (pytest.approx(1.0), 7)
    assert step.worst["qsa.qkv.cosine"] == (pytest.approx(1.0), 7)
    assert step.worst["qsa.ix.cosine"] == (pytest.approx(-1.0), 7)
    assert C.violations(step.worst, D.BOUNDS) == [
        "layer 7 qsa.ix.cosine = -1, bound 0.9999"
    ]
    # the other way round: a qkv_out off SGLang's qkv_proj(x)
    step = _step()
    step.front(2, layer, state, x, -F.linear(x, w_qkv), F.linear(x, w_ix))
    assert step.worst["qsa.qkv.cosine"] == (pytest.approx(-1.0), 2)
    assert step.worst["qsa.ix.cosine"] == (pytest.approx(1.0), 2)


class _Gdn(torch.nn.Module):
    """A stand-in for SGLang's GDN module: one input GEMM for qkvz | ba (as its packed
    bf16 path), out_proj on the core."""

    def __init__(self, w_qkvz, w_ba, w_out):
        super().__init__()
        self.w_qkvz = w_qkvz
        self.in_proj_ba = _Linear(w_ba)
        self.out_proj = _Linear(w_out)

    def _forward_input_proj(self, hidden_states):
        y = F.linear(hidden_states, torch.cat([self.w_qkvz, self.in_proj_ba.w]))
        return y[:, : self.w_qkvz.shape[0]], y[:, self.w_qkvz.shape[0] :]

    def forward(self, x, core):
        qkvz, ba = self._forward_input_proj(x)
        return qkvz, ba, self.out_proj(core)[0]


def _copy(rows, k, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 256, (rows, k // 2), generator=g, dtype=torch.uint8)
    s = torch.randint(122, 129, (rows, k // 32), generator=g, dtype=torch.uint8)
    wq, ws = W.mxfp4_copy(torch.zeros(rows, k, dtype=torch.bfloat16), lambda x: (q, s))
    return wq, ws, decode_mxfp4(q, s)


def _gdn_case():
    g = torch.Generator().manual_seed(9)
    qkvz, qkvz_s, qkvz_vals = _copy(64, 256, 10)
    out, out_s, out_vals = _copy(32, 256, 11)
    attn = _Gdn(
        torch.randn(64, 256, generator=g).bfloat16(),
        torch.randn(8, 256, generator=g).bfloat16(),
        torch.randn(32, 256, generator=g).bfloat16(),
    )
    args = {"w_qkvz": qkvz, "w_qkvz_s": qkvz_s, "w_out": out, "w_out_s": out_s}
    x = (torch.randn(3, 256, generator=g) * torch.linspace(0.1, 8, 256)).bfloat16()
    core = (torch.randn(3, 256, generator=g) * 3).bfloat16()
    return attn, args, qkvz_vals, out_vals, x, core


def _assert_restored(attn):
    assert "forward" not in attn.out_proj.__dict__
    assert "_forward_input_proj" not in attn.__dict__


def test_mxfp4_reference_runs_the_kernels_projections():
    """in_proj_qkvz and out_proj: the copies' values on the bf16 inputs (W4A16, as
    the launch); in_proj_ba: SGLang's; SGLang's modules back afterwards."""
    attn, args, qkvz_vals, out_vals, x, core = _gdn_case()
    stock = attn(x, core)
    with C.mxfp4_reference(attn, args):
        qkvz, ba, out = attn(x, core)
    assert torch.equal(qkvz, F.linear(x, qkvz_vals.bfloat16()))
    assert torch.equal(ba, F.linear(x, attn.in_proj_ba.w))
    assert torch.equal(out, F.linear(core, out_vals.bfloat16()))
    assert not torch.equal(qkvz, stock[0]) and not torch.equal(out, stock[2])
    _assert_restored(attn)
    assert all(torch.equal(a, b) for a, b in zip(attn(x, core), stock))


def test_mxfp4_reference_restores_on_error():
    attn, args, *_ = _gdn_case()
    with pytest.raises(RuntimeError, match="boom"), C.mxfp4_reference(attn, args):
        raise RuntimeError("boom")
    _assert_restored(attn)


@pytest.fixture
def residual(monkeypatch):
    import sglang.srt.layers.layer_boundary.residual.batch as batch

    monkeypatch.setattr(batch, "set_written", lambda s, fb: s)
    monkeypatch.setattr(batch, "take_output", lambda out, fb: out)


def test_before_runs_the_gdn_reference_under_the_hooks(residual):
    """A GDN layer's reference call runs with the MXFP4 hooks in that mode; a QSA
    layer's, and a GDN layer's against the bf16 reference, run bare."""
    attn, args, qkvz_vals, _, x, _ = _gdn_case()
    seen = []

    class Layer:
        ple = None
        linear_attn = attn

        def __call__(self, positions, hidden_states, forward_batch):
            qkvz, _ = self.linear_attn._forward_input_proj(hidden_states)
            seen.append(qkvz)
            return hidden_states + 1

    conv, ssm = torch.zeros(5, 2, 3), torch.zeros(5, 2, 2)
    slots = torch.tensor([3, 1], dtype=torch.int32)
    want = F.linear(x, qkvz_vals.bfloat16())
    step = _step(fb="fb", ref_mode="mxfp4", args={4: args})
    step.before(Layer(), NS(idx=4), None, x, slots, (conv, ssm))
    assert torch.equal(seen[0], want) and torch.equal(step.ref["streams"], x + 1)
    assert step.states == {4: (conv, ssm)}
    _assert_restored(attn)
    step.before(Layer(), NS(idx=3), None, x, slots, None)
    assert not torch.equal(seen[1], want) and step.ref["regions"] == {}
    step = _step(fb="fb", ref_mode="bf16", args={4: args})
    step.before(Layer(), NS(idx=4), None, x, slots, (conv, ssm))
    assert not torch.equal(seen[2], want)


def test_before_restores_the_gdn_rows_the_reference_wrote(residual):
    """The reference's state rows are kept for the comparison; the pool holds the
    rows as they were before it, for the launch; pad slots (<= 0) are left alone."""
    conv, ssm = torch.zeros(5, 2, 3), torch.zeros(5, 2, 2)
    slots = torch.tensor([3, 0, 1, -1], dtype=torch.int32)

    class Layer:
        ple = None

        def __call__(self, positions, hidden_states, forward_batch):
            conv.add_(1.0)
            ssm.add_(2.0)
            return hidden_states

    step = _step(fb="fb", ref_mode="bf16", args={})
    step.before(Layer(), NS(idx=0), None, torch.zeros(4, 4), slots, (conv, ssm))
    assert (step.ref["conv"] == 1).all() and (step.ref["state"] == 2).all()
    assert not conv[[3, 1]].any() and not ssm[[3, 1]].any()
    assert (conv[[0, 2, 4]] == 1).all()  # rows no slot of the step names


def test_before_runs_the_layer_without_its_ple(residual):
    seen = []

    class Layer:
        ple = "the PLE"

        def __call__(self, positions, hidden_states, forward_batch):
            seen.append(self.ple)
            return hidden_states

    layer = Layer()
    _step(fb="fb", ref_mode="bf16", args={}).before(
        layer, NS(idx=0), None, torch.zeros(1, 4), None, None
    )
    assert seen == [None] and layer.ple == "the PLE"


def test_after_compares_streams_delta_and_gdn_rows():
    state_in = torch.ones(2, 8)
    out = torch.full((2, 8), 2.0)
    ssm = torch.zeros(5, 2, 2)
    sel = torch.tensor([3, 1])
    ssm[sel] = 4.0  # the launch's update
    step = _step()
    step.ref = {
        "in": state_in,
        "streams": out,
        "regions": {"state": (ssm, sel)},
        "saved": {"state": torch.ones(2, 2, 2)},
        "state": torch.full((2, 2, 2), 4.0),
    }
    step.after(NS(kind="gdn", idx=5), out)
    assert step.worst == {
        "gdn.streams.cosine": (pytest.approx(1.0), 5),
        "gdn.delta.cosine": (pytest.approx(1.0), 5),
        "gdn.state.cosine": (pytest.approx(1.0), 5),
        "gdn.update.cosine": (pytest.approx(1.0), 5),
    }
    assert step.ref is None


def test_reference_mode(monkeypatch):
    monkeypatch.delenv(C.REF_ENV, raising=False)
    assert C.reference_mode() == "bf16"
    monkeypatch.setenv(C.REF_ENV, "mxfp4")
    assert C.reference_mode() == "mxfp4"
    monkeypatch.setenv(C.REF_ENV, "fp8")
    with pytest.raises(ValueError, match="one of"):
        C.reference_mode()
