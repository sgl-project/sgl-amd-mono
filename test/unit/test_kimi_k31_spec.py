# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 spec routes one request's verify steps to Kimi-K3's launches and
wider ones to the wide launch."""

import ast
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import MAX_ROWS
from sgl_amd_mono.kimi_k31 import spec as k31
from sgl_amd_mono.kimi_k31 import wide


def test_registered_under_the_k31_architecture():
    assert (
        sgl_amd_mono.SPECS[("KimiK31ForConditionalGeneration", None)]
        == "sgl_amd_mono.kimi_k31.spec:SPEC"
    )


def test_the_narrow_and_wide_widths_meet_without_a_gap():
    # _Built routes rows up to NARROW to Kimi-K3's launches, which are built up to
    # MAX_ROWS; a width between the two would route to a launch never built
    assert wide.NARROW == MAX_ROWS
    widths = k31.SPEC.widths("gfx950")
    assert tuple(range(1, MAX_ROWS + 1)) == widths[:MAX_ROWS]
    assert all(w > wide.NARROW for w in widths[MAX_ROWS:])


def test_every_wide_width_has_a_bucket_at_each_draft_length():
    # the binding serves a width at every L it is a multiple of: an 8-token width
    # that reads as more 6-token requests than the largest bucket holds raised at
    # the launch
    top = max(wide.REQ_BUCKETS)
    for w in wide.wide_widths():
        for L in wide.SUPPORTED_L:
            if w % L == 0:
                assert w // L <= top, (w, L)


def test_the_spec_serves_up_to_56_requests():
    # SPEC reads its widths at import, so read them in a process without a bucket override.
    # The 8-token gaps are widths that 6 tokens read as more than 56 requests.
    env = {
        k: v for k, v in os.environ.items() if k != "SGLANG_AMD_MONO_K31_WIDE_BUCKETS"
    }
    # the child imports exactly what this process does
    script = (
        f"import sys; sys.path[:] = {sys.path!r}; "
        "from sgl_amd_mono.kimi_k31 import spec, wide; "
        "print([w for w in spec.SPEC.widths('gfx950') if w > wide.NARROW])"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    widths = ast.literal_eval(out.strip().splitlines()[-1])
    served = {L: {w // L for w in widths if w % L == 0} for L in (6, 8)}
    assert served[6] == set(range(2, 57))
    assert served[8] == set(range(2, 57)) - {45, 48, 51, 54}


def test_the_trimmed_buckets_still_hold_every_served_width(monkeypatch):
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    # A launch takes the smallest bucket holding its rows;
    # trimming one bucket too many leaves the widest served step without one.
    for L in wide.SUPPORTED_L:
        full = wide.buckets(L=L)
        for top in (w for w in wide.wide_widths() if w % L == 0):
            kept = wide.buckets(L=L, max_rows=top)
            assert kept == full[: len(kept)], (L, top)
            assert kept[-1] >= top and (len(kept) == 1 or kept[-2] < top), (L, top)


def test_every_bucket_resolves_to_a_tuned_config(monkeypatch):
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    # cfg_rows borrows the next tuned count's configs;
    # a bucket past every tuned count must still resolve rather than raise KeyError at warmup.
    for L in wide.SUPPORTED_L:
        for b in wide.buckets(L=L):
            configs = wide.cfg_rows(b=b, L=L)
            assert len(configs) == 5, (L, b)
            assert all(len(c.split(":")) == 5 for c in configs), (L, b, configs)
    assert wide.cfg_rows(b=24, L=8) == wide.CFGS[32]
    # past every tuned count and row range: the largest tuned count's configs
    assert wide._row_cfgs(max(wide.CFGS) + 65) == wide.CFGS[max(wide.CFGS)]


def test_an_override_keeps_its_cut_at_every_draft_length(monkeypatch):
    """A shared wide graph key must clear the override's lower cutoff at every draft length."""
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", value="128,256,384")
    lo = max(r for r in wide.REQ_BUCKETS if r < 16)
    widths = wide.wide_widths()
    for w in widths:
        for L in wide.SUPPORTED_L:
            if w % L == 0:
                assert lo < w // L <= 48, (w, L)
    assert 96 not in widths and 120 in widths


def test_kv_offload_is_accepted_for_hicache_only(monkeypatch):
    monkeypatch.setattr(
        target=k31.k3, name="_config_disable_reasons", value=lambda cfg, hf_config: []
    )
    monkeypatch.delenv(name="SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT", raising=False)
    # the fp32 KDA state the wide launch reads, so the offload flags are the only refusals
    monkeypatch.setenv(name="SGLANG_MAMBA_SSM_DTYPE", value="float32")
    hf_config = SimpleNamespace(dsa_config={"index_topk": 2048})
    off = dict(
        enable_lmcache=False,
        enable_flexkv=False,
        disaggregation_decode_enable_offload_kvcache=False,
    )
    cfg = dict(
        enable_hierarchical_cache=True,
        dcp_size=8,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_num_draft_tokens=6,
        **off,
    )
    assert k31.Feature.KV_OFFLOAD in k31.SPEC.features
    assert (
        k31.SPEC.config_disable_reasons(cfg=SimpleNamespace(**cfg), hf_config=hf_config)
        == []
    )
    for flag in off:
        reasons = k31.SPEC.config_disable_reasons(
            cfg=SimpleNamespace(
                **{**cfg, "enable_hierarchical_cache": False, flag: True}
            ),
            hf_config=hf_config,
        )
        assert len(reasons) == 1 and "HiCache" in reasons[0], (flag, reasons)
    # host-only checkpoints ring ReplaySSM state by request row, not by mamba slot
    monkeypatch.setenv(name="SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT", value="1")
    reasons = k31.SPEC.config_disable_reasons(
        cfg=SimpleNamespace(**cfg), hf_config=hf_config
    )
    assert len(reasons) == 1 and "mamba slot" in reasons[0], reasons


def test_the_launches_name_their_server_args():
    server_args = k31.SPEC.requires.server_args
    assert server_args["enable_linear_replayssm_spec"] == frozenset({True})
    assert server_args["speculative_algorithm"] == frozenset({"DSPARK", "DFLASH"})


# kimi_k31's checks on one covered KDA layer, Kimi-K3's own checks taken as
# passing (test_kimi_k3_spec pins those: they accept a Quark FP8 o_proj).
# ``in_proj=None`` is a layer whose KDA in_proj is not merged; ``o_proj=None`` is
# a qweight-only (GPTQ / AWQ-style) o_proj with no ``weight``.
def _reasons(monkeypatch, *, in_proj=torch.bfloat16, o_proj=torch.bfloat16):
    monkeypatch.setattr(k31.k3, "_model_disable_reasons", lambda *a: [])
    monkeypatch.setattr(wide, "_is_covered", lambda layer: True)

    def lin(dtype):
        return SimpleNamespace(weight=torch.zeros(8, 8, dtype=dtype))

    attn = SimpleNamespace(
        _qkvgbfa_layer=None if in_proj is None else lin(in_proj),
        o_proj=SimpleNamespace(qweight=None) if o_proj is None else lin(o_proj),
    )
    layers = [SimpleNamespace(self_attn=attn)]
    return k31._model_disable_reasons(
        SimpleNamespace(model=SimpleNamespace(layers=layers)), None, {}
    )


def test_a_bf16_kda_layer_passes(monkeypatch):
    assert _reasons(monkeypatch) == []


@pytest.mark.parametrize(
    "dtypes",
    [
        {"o_proj": torch.float8_e4m3fn},
        {"in_proj": torch.float8_e4m3fn},
        {"in_proj": torch.float32},
    ],
    ids=["fp8-o_proj", "fp8-in_proj", "fp32-in_proj"],
)
def test_a_kda_weight_the_wide_launch_cannot_read_is_refused(monkeypatch, dtypes):
    # Kimi-K3's checks accept a Quark FP8 o_proj (K1 and the layer launch read it),
    # but the wide launch reads the in_proj and o_proj weights as bf16: an FP8 one
    # would be read as if it were bf16.
    reasons = _reasons(monkeypatch, **dtypes)
    assert any("not bf16" in r for r in reasons), reasons


def test_an_unmerged_in_proj_is_reported_once(monkeypatch):
    # an unmerged layer has no merged weight to read; it is refused for that alone
    reasons = _reasons(monkeypatch, in_proj=None, o_proj=torch.float8_e4m3fn)
    assert reasons == ["layers [0]: the KDA in_proj is not merged"], reasons


def test_a_qweight_only_o_proj_leaves_the_refusal_to_kimi_k3(monkeypatch):
    """A merged layer whose o_proj has no ``weight`` is reported by Kimi-K3's checks
    ("o_proj has no weight"); kimi_k31's own checks must not raise on it."""
    assert _reasons(monkeypatch, o_proj=None) == []


def test_the_wide_launch_refuses_a_bf16_kda_state(monkeypatch):
    """The wide KDA stage reads the recurrent state as fp32; a bf16 state (which
    Kimi-K3's K1 accepts) must not reach it."""
    from sglang.srt.environ import envs

    monkeypatch.setattr(k31.k3, "_config_disable_reasons", lambda **kwargs: [])
    monkeypatch.setattr(
        k31.k3, "_text_config", lambda hf: SimpleNamespace(dsa_config={})
    )

    def refused():
        cfg = SimpleNamespace(
            dcp_size=8,
            speculative_num_draft_tokens=6,
            enable_lmcache=False,
            enable_flexkv=False,
            disaggregation_decode_enable_offload_kvcache=False,
        )
        reasons = k31._config_disable_reasons(cfg, None)
        return any("the wide launch reads float32" in r for r in reasons)

    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        assert refused()
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
        assert not refused()


def test_wide_capture_stream_and_variant():
    """A captured layer's wide launch writes the stream the DSpark draft asked for,
    not always the attn_res mixture; SGLang without the field still means attn_res."""
    assert wide._capture_stream(SimpleNamespace()) == "attn_res"
    assert (
        wide._capture_stream(SimpleNamespace(dspark_aux_hidden_stream="prefix"))
        == "prefix"
    )
    with pytest.raises(ValueError, match="unsupported dspark_aux_hidden_stream"):
        wide._capture_stream(SimpleNamespace(dspark_aux_hidden_stream="unknown"))

    pk = SimpleNamespace(i=76, nvb=7, write=False, t1=True)
    w = object.__new__(wide.Wide)
    assert w._variant(pk, captures={75}, capture_stream="prefix")["capture"] == "prefix"
    assert (
        w._variant(pk, captures={75}, capture_stream="attn_res")["capture"]
        == "attn_res"
    )
    assert w._variant(pk, captures={79}, capture_stream="prefix")["capture"] is False
