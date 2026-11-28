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


def test_every_width_an_l_serves_has_a_bucket(monkeypatch):
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    # a step takes the smallest bucket holding its rows; one past the largest raised
    for L in wide.SUPPORTED_L:
        top = wide.buckets(L=L)[-1]
        assert all(w <= top for w in wide.widths_at(L)), L


def test_the_spec_serves_up_to_60_requests_at_every_draft_length():
    # SPEC reads its widths at import, so read them in a process without a bucket override.
    # MonoSpec widths are one list for every L: each L's own widths, and nothing else.
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
    widths = set(ast.literal_eval(out.strip().splitlines()[-1]))
    assert widths == {L * r for L in (5, 6, 8) for r in range(2, 61)}


def test_the_trimmed_buckets_still_hold_every_served_width(monkeypatch):
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    # A launch takes the smallest bucket holding its rows;
    # trimming one bucket too many leaves the widest served step without one.
    for L in wide.SUPPORTED_L:
        full = wide.buckets(L=L)
        for top in wide.widths_at(L):
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
    """Each L serves only request counts past the override's lower cut and up to its last
    bucket; another L's width below the cut stays on SGLang's path (Wide.serves)."""
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", value="128,256,384")
    lo = max(r for r in wide.REQ_BUCKETS if r < 16)
    for L in wide.SUPPORTED_L:
        assert all(lo < w // L <= 48 for w in wide.widths_at(L)), L
    # 96 rows: 16 requests at L = 6, 12 (below the cut) at L = 8
    assert 96 in wide.widths_at(6) and 96 in wide.wide_widths()
    assert 96 not in wide.widths_at(8) and 120 in wide.widths_at(8)


def test_kv_offload_is_accepted_for_hicache_only(monkeypatch):
    # Kimi-K3's checks (test_kimi_k3_spec pins their offload refusals) pass through
    monkeypatch.setattr(
        target=k31.k3,
        name="_config_disable_reasons",
        value=lambda cfg, hf_config: ["kimi_k3: lmcache"] if cfg.enable_lmcache else [],
    )
    monkeypatch.delenv(name="SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT", raising=False)
    monkeypatch.setattr(k31.k3, "WIDE", True)  # K3.1 needs the wide launch
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


def test_a_qweight_only_kda_projection_is_refused_with_a_reason(monkeypatch):
    """An unmerged projection with no `weight` (GPTQ / AWQ-style qweight) is refused with
    the not-merged reason instead of raising AttributeError out of the check."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

    attn = SimpleNamespace(
        use_full_rank_gate=True,
        _bfa_f_b_w=None,
        _qkvgbfa_layer=None,
        fused_qkvg_proj=SimpleNamespace(qweight=torch.empty(1, dtype=torch.int32)),
        f_b_proj=SimpleNamespace(weight=torch.empty(1536, 128, dtype=torch.bfloat16)),
    )
    with pytest.raises(ValueError, match="not merged"):
        k1_mod._validate_in_proj_operands(attn)


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


def test_row_count_tile_configs_meet_plan_and_tile_invariants(monkeypatch):
    # Guards against out-of-bounds K-splits or LDS/CTA overflows across all (L, bucket) pairs.
    monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    import math

    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm
    from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import BK

    assert 456 in wide.buckets(8) or 480 in wide.buckets(8)  # the 449-480-row tiles
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


@pytest.mark.parametrize("override", [None, "128,256,384"])
def test_every_bucket_is_a_served_width_so_its_launch_sort_is_read(
    monkeypatch, override
):
    """The routed experts read the launch's sort only when a step fills its bucket: every
    bucket must be a width its L serves, so _launch_sort needs no per-bucket gate."""
    from sgl_amd_mono.kimi_k31 import presort
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

    if override is None:
        monkeypatch.delenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    else:
        monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", value=override)
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT", value="1")
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK", value="1")
    monkeypatch.setattr(presort, "sort_kind", lambda: "aux16")
    for L in wide.SUPPORTED_L:
        for b in wide.buckets(L=L):
            want = "aux16" if b <= mf.sort_max_rows("aux16") else None
            assert wide._launch_sort(b) == want, (L, b)
    for L in wide.SUPPORTED_L:
        assert set(wide.buckets(L=L)) <= set(wide.widths_at(L)), L


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


def test_a_width_only_another_draft_length_serves_runs_sglang(monkeypatch):
    """The binding routes every spec width that is a multiple of L; one only another L
    serves (330 rows: 55 requests at L = 6, 66 at L = 5, past every bucket) runs SGLang's
    own backbone instead of the wide launch, as a step past the widths does."""
    runner = object.__new__(wide.Wide)
    runner.widths = frozenset(
        {300}
    )  # 60 requests at L = 5; 330 is not one of its widths
    runner.backbone = lambda built, ids, fb, embeds: ("wide", ids, fb, embeds)
    built = object.__new__(k31.k3._WideBuilt)
    built._wide, built._positions = runner, "positions"
    built.backbone = lambda *args: ("sglang", *args)
    built._plan = SimpleNamespace(key=SimpleNamespace(rows=330))
    assert built._backbone("ids", "fb", None) == (
        "sglang",
        "ids",
        "positions",
        "fb",
        None,
    )
    built._plan = SimpleNamespace(key=SimpleNamespace(rows=300))
    assert built._backbone("ids", "fb", None) == ("wide", "ids", "fb", None)


def test_a_draft_length_serving_no_routed_wide_width_builds_no_wide_launch(monkeypatch):
    """With the 128,256,384 override and 8 tokens capped at 12 requests the binding still
    routes 96 rows (16 requests at L = 6): `_build` builds no Wide and SGLang runs that
    step; with a width L = 8 serves among them, Wide is sized to that width."""
    import sglang.srt.runtime_context as rc

    k3 = k31.k3
    monkeypatch.setenv(name="SGLANG_AMD_MONO_K31_WIDE_BUCKETS", value="128,256,384")
    made = []

    class FakeWide:
        def __init__(self, model, rank, group, device, *, L, max_rows):
            made.append((L, max_rows))

        def warmup(self):
            pass

    monkeypatch.setattr(wide, "Wide", FakeWide)
    monkeypatch.setattr(wide, "tokens", lambda: 8)
    monkeypatch.setattr(wide, "_say", lambda msg: None)
    monkeypatch.setattr(k3, "_build_narrow", lambda *a: SimpleNamespace(lm=None))
    monkeypatch.setattr(
        rc,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_group=SimpleNamespace(cpu_group=None)),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 0))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    norm = SimpleNamespace(weight=torch.empty(1))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(input_layernorm=norm)])
    )
    routed = (
        *range(1, 9),
        96,
    )  # L = 8 serves 120 (15 requests) and up under the override

    built = k3._build(model, None, SimpleNamespace(widths=routed), use_wide=True)
    assert made == [] and built._wide is None
    built.backbone = lambda *args: ("sglang", *args)
    built._positions = "positions"
    built._plan = SimpleNamespace(key=SimpleNamespace(rows=96))
    assert built._backbone("ids", "fb", None) == (
        "sglang",
        "ids",
        "positions",
        "fb",
        None,
    )

    # sized to the widest width this L serves (120 and 256 sit in different buckets)
    k3._build(model, None, SimpleNamespace(widths=(*routed, 120, 256)), use_wide=True)
    assert made == [(8, 256)]
    # not to a wider routed width another L serves: at L = 5, 270 rows are 54 requests
    # (past the override's 48) while 80 and 160 are 16 and 32
    monkeypatch.setattr(wide, "tokens", lambda: 5)
    made.clear()
    k3._build(
        model, None, SimpleNamespace(widths=(*range(1, 9), 80, 160, 270)), use_wide=True
    )
    assert made == [(5, 160)]


def test_the_narrow_mla_handoff_needs_no_indexer_and_is_cleared_on_error(monkeypatch):
    """A Kimi-K3 MLA layer has no indexer attribute: the handoff skips it, and `_mono_pre`
    is gone again even when SGLang's attention raises."""
    import sglang.srt.layers.layer_boundary.residual.batch as batch

    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    rows = 8
    monkeypatch.setattr(mp, "mla_pre_launch", lambda key, **kw: None)
    monkeypatch.setattr(mp, "scratch_layout", lambda key: {"x": (0, rows * HIDDEN * 2)})
    monkeypatch.setattr(batch, "set_written", lambda *a: None)
    built = object.__new__(k31.k3._Built)
    out = {n: torch.zeros(1) for n in ("q", "latent", "k_nope", "q_lora", "gate")}
    built._mla_out = {rows: out}
    built.runner = SimpleNamespace(
        scratch=torch.zeros(rows * HIDDEN * 2, dtype=torch.uint8), epoch=None
    )
    seen = []

    def attend(layer, x, forward_batch, zero_allocator):
        seen.append(dict(layer.self_attn.__dict__["_mono_pre"]))
        raise RuntimeError("attention failed")

    built._pre_o_proj = attend
    layer = SimpleNamespace(self_attn=SimpleNamespace())  # no indexer attribute
    packed = SimpleNamespace(mla=(SimpleNamespace(write_idx=-1, qn=False), {}), i=3)
    bank = SimpleNamespace(block_residual=None)
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="attention failed"):
        built._mla_core(layer, packed, hidden, bank, None, None)
    assert seen and seen[0]["q"] is out["q"] and seen[0]["q_nope_out"] is None
    assert "_mono_pre" not in layer.self_attn.__dict__
