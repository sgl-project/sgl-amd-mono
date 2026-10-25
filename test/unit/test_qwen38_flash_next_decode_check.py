"""The Qwen3.8-Flash-Next TP2 decode check mode on CPU: its metrics against the
path's bounds, the worst value kept per metric, the radix track-save comparison,
and the MXFP4 reference's GDN projections on stand-ins for SGLang's GDN modules."""

import math
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from qwen38_flash_next_decode_fakes import decode_mxfp4

from sgl_amd_mono.qwen38_flash_next.decode import check as C
from sgl_amd_mono.qwen38_flash_next.decode import spec as D
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import weights as W


def test_cosine():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(C.cosine(a, 2 * a) - 1.0) < 1e-12
    assert abs(C.cosine(a, -a) + 1.0) < 1e-12
    assert math.isnan(C.cosine(a, torch.tensor([1.0, float("nan"), 3.0])))


def test_violations_against_the_spec_bounds():
    bounds = dict(D.BOUNDS)
    values = {
        "front.mix.cosine": (0.99999, 0),
        "gdn.delta.cosine": (0.2, 4),  # a wrong layout
        "gdn.state.cosine": (float("nan"), 8),
        "qsa.delta.cosine": (bounds["qsa.delta.cosine"], 3),  # at the bound: passes
    }
    assert C.violations(values, bounds) == [
        "layer 4 gdn.delta.cosine = 0.2, bound 0.9",
        "layer 8 gdn.state.cosine = nan, bound 0.999",
    ]


def test_every_bound_names_a_metric_the_check_records():
    names = {"front.mix.cosine"} | {
        f"{kind}.{out}.cosine"
        for kind in ("gdn", "qsa")
        for out in ("streams", "delta")
    }
    names |= {f"gdn.{out}.cosine" for out in ("core", "conv", "state", "update")}
    assert set(D.BOUNDS) == names | {"gdn.track.max_abs"}


def test_check_bounds_by_reference():
    assert D.check_bounds("bf16") == dict(D.BOUNDS)
    mx = D.check_bounds("mxfp4")
    # the MoE-free metrics only, each at least as tight as against the bf16 reference
    cos = {"front.mix.cosine"} | {
        f"gdn.{o}.cosine" for o in ("core", "conv", "state", "update")
    }
    assert set(mx) == cos | {"gdn.track.max_abs"}
    assert all(mx[k] >= D.BOUNDS[k] for k in cos)
    assert mx["gdn.track.max_abs"] == D.BOUNDS["gdn.track.max_abs"] == 0.0


def test_note_keeps_the_worst_and_a_nan():
    step = object.__new__(C.StepCheck)
    step.worst = {}
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
    step = object.__new__(C.StepCheck)
    step.worst, step.rows = {}, 3
    step.track(states, fb, md)
    assert step.worst == {"gdn.track.max_abs": (0.0, 0)}
    states[4][1][8, 1, 2, 3] += 0.5
    step.track(states, fb, md)
    assert step.worst["gdn.track.max_abs"] == (0.5, 4)
    assert C.violations(step.worst, D.check_bounds("mxfp4")) == [
        "layer 4 gdn.track.max_abs = 0.5, bound 0"
    ]
    step.worst = {}
    step.track(states, NS(mamba_track_mask=None), md)  # radix cache off
    assert step.worst == {}


class _Linear(torch.nn.Module):
    """A stand-in for SGLang's linears: forward -> (output, bias)."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        return F.linear(x, self.w), None


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
    return W.Fp4(wq, ws), decode_mxfp4(q, s)


def _gdn_case():
    g = torch.Generator().manual_seed(9)
    qkvz, qkvz_vals = _copy(64, 256, 10)
    out, out_vals = _copy(32, 256, 11)
    attn = _Gdn(
        torch.randn(64, 256, generator=g).bfloat16(),
        torch.randn(8, 256, generator=g).bfloat16(),
        torch.randn(32, 256, generator=g).bfloat16(),
    )
    x = (torch.randn(3, 256, generator=g) * torch.linspace(0.1, 8, 256)).bfloat16()
    core = (torch.randn(3, 256, generator=g) * 3).bfloat16()
    return attn, NS(qkvz=qkvz, out=out), qkvz_vals, out_vals, x, core


def _assert_restored(attn):
    assert "forward" not in attn.out_proj.__dict__
    assert "_forward_input_proj" not in attn.__dict__


def test_gdn_reference_bf16_records_core_only():
    attn, gdn, _, _, x, core = _gdn_case()
    want = attn(x, core)
    with C.gdn_reference(attn, gdn, "bf16") as rec:
        got = attn(x, core)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert torch.equal(rec["core"], core)
    _assert_restored(attn)


def test_gdn_reference_mxfp4_runs_the_kernels_projections():
    """in_proj_qkvz and out_proj: the copies' values against MXFP4 inputs;
    in_proj_ba: bf16 on the unquantized rows; SGLang's modules back afterwards."""
    attn, gdn, qkvz_vals, out_vals, x, core = _gdn_case()
    stock = attn(x, core)
    with C.gdn_reference(attn, gdn, "mxfp4") as rec:
        qkvz, ba, out = attn(x, core)
    xq, cq = W.mxfp4_fake_quant(x), W.mxfp4_fake_quant(core)
    assert not torch.equal(xq, x) and not torch.equal(cq, core)
    assert torch.equal(qkvz, F.linear(xq, qkvz_vals.bfloat16()))
    assert torch.equal(ba, F.linear(x, attn.in_proj_ba.w))
    assert torch.equal(out, F.linear(cq, out_vals.bfloat16()))
    assert torch.equal(rec["core"], core)
    _assert_restored(attn)
    assert all(torch.equal(a, b) for a, b in zip(attn(x, core), stock))


def test_gdn_reference_restores_on_error():
    attn, gdn, *_ = _gdn_case()
    with pytest.raises(RuntimeError, match="boom"), C.gdn_reference(attn, gdn, "mxfp4"):
        raise RuntimeError("boom")
    _assert_restored(attn)


def test_before_runs_the_gdn_reference_under_the_hooks(monkeypatch):
    """A GDN layer's reference call runs with the MXFP4 hooks and keeps out_proj's
    input; a QSA layer's runs bare; SGLang's modules are back after."""
    import sglang.srt.layers.layer_boundary.residual.batch as batch

    monkeypatch.setattr(batch, "set_written", lambda s, fb: s)
    monkeypatch.setattr(batch, "take_output", lambda out, fb: out)
    attn, gdn, qkvz_vals, _, x, _ = _gdn_case()
    seen = []

    class Layer:
        ple = None
        linear_attn = attn

        def __call__(self, positions, hidden_states, forward_batch):
            qkvz, _ = self.linear_attn._forward_input_proj(hidden_states)
            seen.append(qkvz)
            self.linear_attn.out_proj(hidden_states * 0.5)
            return hidden_states + 1

    step = object.__new__(C.StepCheck)
    step.fb, step.ref_mode = "fb", "mxfp4"
    step.before(Layer(), NS(gdn=gdn), None, x, None, None)
    want = F.linear(W.mxfp4_fake_quant(x), qkvz_vals.bfloat16())
    assert torch.equal(seen[0], want) and torch.equal(step.ref["core"], x * 0.5)
    assert torch.equal(step.ref["streams"], x + 1) and step.kcore is None
    _assert_restored(attn)
    step.before(Layer(), NS(gdn=None), None, x, None, None)
    assert step.ref["core"] is None and not torch.equal(seen[1], want)


def test_before_restores_the_gdn_rows_the_reference_wrote(monkeypatch):
    """The reference's state rows are kept for the comparison; the pool holds the
    rows as they were before it, for the launch; pad slots (<= 0) are left alone."""
    import sglang.srt.layers.layer_boundary.residual.batch as batch

    monkeypatch.setattr(batch, "set_written", lambda s, fb: s)
    monkeypatch.setattr(batch, "take_output", lambda out, fb: out)
    conv, ssm = torch.zeros(5, 2, 3), torch.zeros(5, 2, 2)
    slots = torch.tensor([3, 0, 1], dtype=torch.int32)

    class Layer:
        ple = None

        def __call__(self, positions, hidden_states, forward_batch):
            conv.add_(1.0)
            ssm.add_(2.0)
            return hidden_states

    step = object.__new__(C.StepCheck)
    step.fb, step.ref_mode = "fb", "bf16"
    step.before(Layer(), NS(gdn=None), None, torch.zeros(3, 4), slots, (conv, ssm))
    assert (step.ref["conv"] == 1).all() and (step.ref["state"] == 2).all()
    assert not conv[[3, 1]].any() and not ssm[[3, 1]].any()
    assert (conv[[0, 2, 4]] == 1).all()  # rows no slot of the step names


def test_reference_mode(monkeypatch):
    monkeypatch.delenv(C.REF_ENV, raising=False)
    assert C.reference_mode() == "bf16"
    monkeypatch.setenv(C.REF_ENV, "mxfp4")
    assert C.reference_mode() == "mxfp4"
    monkeypatch.setenv(C.REF_ENV, "fp8")
    with pytest.raises(ValueError, match="one of"):
        C.reference_mode()


def test_core_metric_on_gdn_layers_only():
    state_in, out = torch.ones(2, 8), torch.full((2, 8), 2.0)
    step = object.__new__(C.StepCheck)
    step.worst, step.ref_mode, step.kcore = {}, "bf16", None
    ref_core = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    step.ref = {
        "in": state_in,
        "streams": out,
        "regions": {},
        "saved": {},
        "core": ref_core,
    }
    step.core(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    step.after(NS(kind="gdn", idx=5), out)
    v, layer = step.worst["gdn.core.cosine"]
    assert layer == 5 and abs(v - 2 / math.sqrt(6)) < 1e-12
    assert step.kcore is None and step.ref is None
    step.ref = {
        "in": state_in,
        "streams": out,
        "regions": {},
        "saved": {},
        "core": None,
    }
    step.after(NS(kind="qsa", idx=7), out)
    assert step.worst["gdn.core.cosine"][1] == 5
    assert step.worst["qsa.streams.cosine"] == (1.0, 7)
