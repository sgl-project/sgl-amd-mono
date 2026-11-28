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


def test_the_spec_serves_up_to_60_requests():
    # SPEC reads its widths at import, so read them in a process without a bucket override.
    # The gaps are widths a shorter draft length reads as more than 60 requests (MonoSpec
    # widths are not per draft length): 6 tokens' 330 and 360 rows are 66 and 72 at L = 5.
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
    served = {L: {w // L for w in widths if w % L == 0} for L in (5, 6, 8)}
    assert served[5] == set(range(2, 61))
    assert served[6] == set(range(2, 61)) - {55, 60}
    assert served[8] == set(range(2, 61)) - {40, 45, 48, 50, 51, 54, 55, 57, 60}


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
    tables = (
        wide.UP_ROW_RANGES,
        wide.INP_ROW_RANGES,
        wide.OPROJ_ROW_RANGES,
        wide.FRONT_ROW_RANGES,
        wide.SDOWN_ROW_RANGES,
    )
    past = max(max(wide.CFGS), *(hi for t in tables for _, hi, _ in t)) + 1
    assert wide._row_cfgs(past) == wide.CFGS[max(wide.CFGS)]


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
    # Kimi-K3's checks (test_kimi_k3_spec pins their offload refusals) pass through
    monkeypatch.setattr(
        target=k31.k3,
        name="_config_disable_reasons",
        value=lambda cfg, hf_config: ["kimi_k3: lmcache"] if cfg.enable_lmcache else [],
    )
    monkeypatch.delenv(name="SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT", raising=False)
    hf_config = SimpleNamespace(dsa_config={"index_topk": 2048})
    cfg = dict(
        enable_hierarchical_cache=True,
        enable_lmcache=False,
        speculative_num_draft_tokens=6,
    )
    assert k31.Feature.KV_OFFLOAD in k31.SPEC.features
    assert (
        k31.SPEC.config_disable_reasons(cfg=SimpleNamespace(**cfg), hf_config=hf_config)
        == []
    )
    reasons = k31.SPEC.config_disable_reasons(
        cfg=SimpleNamespace(**{**cfg, "enable_lmcache": True}), hf_config=hf_config
    )
    assert reasons == ["kimi_k3: lmcache"], reasons
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


def _reasons(monkeypatch, *, in_proj=torch.bfloat16):
    from sgl_amd_mono.kimi_k3 import CORE, HF_CONFIG
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    monkeypatch.setattr(k31.k3, "_model_disable_reasons", lambda *a: [])
    monkeypatch.setattr(k31.k3, "_wide_will_run", lambda: False)
    monkeypatch.setattr(wide, "_is_covered", lambda layer: True)

    hidden = HF_CONFIG["hidden_size"]
    attn = SimpleNamespace(
        use_full_rank_gate=True,
        _bfa_f_b_w=None,
        fused_qkvg_proj=SimpleNamespace(weight=torch.empty(1, 1, dtype=torch.bfloat16)),
        _qkvgbfa_layer=(
            None
            if in_proj is None
            else SimpleNamespace(weight=torch.empty(kda.NPROJ, hidden, dtype=in_proj))
        ),
        f_b_proj=SimpleNamespace(
            weight=torch.empty(kda.PROJ, kda.HD, dtype=torch.bfloat16)
        ),
        o_proj=SimpleNamespace(weight=torch.empty(hidden, CORE, dtype=torch.bfloat16)),
    )
    layers = [SimpleNamespace(self_attn=attn)]
    return k31._model_disable_reasons(
        SimpleNamespace(model=SimpleNamespace(layers=layers)), None, {}
    )


def test_a_bf16_kda_layer_passes(monkeypatch):
    assert _reasons(monkeypatch) == []


def test_a_kda_weight_the_wide_launch_cannot_read_is_refused(monkeypatch):
    reasons = _reasons(monkeypatch, in_proj=torch.float32)
    assert any("merged KDA in-projection" in r for r in reasons), reasons


def test_an_unmerged_in_proj_is_refused(monkeypatch):
    reasons = _reasons(monkeypatch, in_proj=None)
    assert any("not merged" in r for r in reasons), reasons


def test_k31_refuses_k3_wide_disabled(monkeypatch):
    monkeypatch.setattr(k31.k3, "WIDE", False)
    monkeypatch.setattr(k31.k3, "_config_disable_reasons", lambda **kwargs: [])
    cfg = SimpleNamespace(
        speculative_num_draft_tokens=6,
        enable_lmcache=False,
        enable_flexkv=False,
        disaggregation_decode_enable_offload_kvcache=False,
    )
    hf = SimpleNamespace(text_config=SimpleNamespace(dsa_config={"index_topk": 2048}))
    reasons = k31._config_disable_reasons(cfg, hf)
    assert any("SGLANG_AMD_MONO_K3_WIDE=0" in r for r in reasons), reasons


def test_row_count_tile_configs_meet_plan_and_tile_invariants():
    # Guards against out-of-bounds K-splits or LDS/CTA overflows across all (L, bucket) pairs.
    import math

    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm
    from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import BK

    for L in wide.SUPPORTED_L:
        for b in wide.buckets(L):
            gu, gi, go, gf, gs = (
                Gemm(*(int(v) for v in c.split(":"))) for c in wide.cfg_rows(b, L)
            )
            assert gu.ks > 1 and gi.ks > 1 and gf.ks > 1 and gs.ks == 1
            assert (wide.LAT // gu.ks) % BK == 0 and (wide.H // gi.ks) % BK == 0
            assert (wide.KO // go.ks) % BK == 0 and (wide.H // gf.ks) % BK == 0
            assert k1.FA_0 % gi.bn == 0 and wide.SL % go.bn == 0
            assert (
                math.ceil(b / gf.bm) * math.ceil(wide.NF / gf.bn) * gf.ks <= BLOCKS * 4
            )


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


def test_a_bucket_no_served_step_fills_makes_no_launch_sort(monkeypatch):
    """The routed experts read the launch's sort only when a step fills its bucket; a bucket
    whose own width is not served (8-token 480 rows) must not pay for a sort nobody reads."""
    from sgl_amd_mono.kimi_k31 import presort

    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT", value="1")
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK", value="1")
    monkeypatch.setattr(presort, "sort_kind", lambda: "aux16")
    widths = wide.wide_widths()
    assert 480 not in widths and 448 in widths
    assert wide._launch_sort(480) is None
    assert wide._launch_sort(448) == "aux16"
    for L in wide.SUPPORTED_L:
        for b in wide.buckets(L=L):
            assert (wide._launch_sort(b) is not None) <= (b in widths), (L, b)


def test_the_wide_launch_refuses_an_unsupported_state_dtype_before_building(
    monkeypatch,
):
    """Wide raises the KDA state dtype refusal before it reads the model or allocates."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

    monkeypatch.setattr(
        k1_mod, "_state_config", lambda: (False, ["the recurrent state is float16"])
    )
    with pytest.raises(ValueError, match="float16"):
        wide.Wide(model=None, rank=0, group=None, device="cpu", L=6)
