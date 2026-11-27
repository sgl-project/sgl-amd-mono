# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 spec routes one request's verify steps to Kimi-K3's launches and
wider ones to the wide launch."""

from types import SimpleNamespace

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
    monkeypatch.setattr(k31.k3, "_config_disable_reasons", lambda *a: [])
    cfg = SimpleNamespace(speculative_num_draft_tokens=6)
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
