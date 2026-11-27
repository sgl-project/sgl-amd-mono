"""GLM-5.3-Flash's wide launch on CPU: the one spec that takes it on a server with
``--enable-linear-replayssm-spec``, the widths it adds, its config and model
refusal reasons, and the steps its forward hands back to the model's forward.
SGLang checks the spec's declared requirements, the same way for every spec."""

from types import SimpleNamespace

import pytest
import torch

import sgl_amd_mono
from sgl_amd_mono.glm53_flash import HF_CONFIG, LINEAR_ATTN_CONFIG
from sgl_amd_mono.glm53_flash import spec as flash
from sgl_amd_mono.glm53_flash import wide
from sgl_amd_mono.glm53_flash import wide_spec as ws

TEXT = SimpleNamespace(**HF_CONFIG, linear_attn_config=dict(LINEAR_ATTN_CONFIG))
FLASH = SimpleNamespace(text_config=TEXT)
ARCH = ("Glm5NextForConditionalGeneration", None)


def test_one_spec_serves_both_launches():
    assert sgl_amd_mono.SPECS[ARCH] == "sgl_amd_mono.glm53_flash.spec:SPEC"
    L = wide.SUPPORTED_L[0]
    widths = flash.SPEC.devices["gfx950"].widths
    wide_widths = tuple(L * r for r in range(3, max(wide.REQ_BUCKETS) + 1))
    assert widths == flash.ROWS + wide_widths
    assert max(flash.ROWS) == wide.NARROW < min(wide_widths)
    assert wide.TP in flash.SPEC.devices["gfx950"].tp_sizes
    # the arg the two launches split on is checked by the spec, not required
    assert "enable_linear_replayssm_spec" not in flash.SPEC.requires.server_args
    # HiCache: the config check takes it on the wide launch only
    assert flash.Feature.KV_OFFLOAD in flash.SPEC.features


def _cfg(topk=1, steps=5, draft=6, algorithm="EAGLE", **kw):
    return SimpleNamespace(
        **{
            "enable_hierarchical_cache": False,
            "enable_lmcache": False,
            "enable_flexkv": False,
            "disaggregation_decode_enable_offload_kvcache": False,
            **kw,
        },
        enable_linear_replayssm_spec=True,
        speculative_algorithm=algorithm,
        speculative_eagle_topk=topk,
        speculative_num_steps=steps,
        speculative_num_draft_tokens=draft,
    )


def _flydsl(monkeypatch, installed):
    find_spec = ws.importlib.util.find_spec

    def find(name, *args):
        if name == "flydsl":
            return object() if installed else None
        return find_spec(name, *args)

    monkeypatch.setattr(ws.importlib.util, "find_spec", find)


def _dtypes(temporal=torch.bfloat16, conv=torch.bfloat16):
    return lambda config: SimpleNamespace(temporal=temporal, conv=conv)


@pytest.mark.parametrize("temporal", [torch.float32, torch.bfloat16])
def test_mtp5_on_either_state_dtype_is_served(monkeypatch, temporal):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes(temporal))
    _flydsl(monkeypatch, True)
    assert flash.SPEC.config_disable_reasons(_cfg(), FLASH) == []


@pytest.mark.parametrize(
    "offload, ok",
    [
        ({"enable_hierarchical_cache": True}, True),
        ({"enable_lmcache": True}, False),
        ({"enable_flexkv": True}, False),
        ({"disaggregation_decode_enable_offload_kvcache": True}, False),
    ],
)
def test_hicache_is_the_kv_offload_served(monkeypatch, offload, ok):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    _flydsl(monkeypatch, True)
    reasons = flash.SPEC.config_disable_reasons(_cfg(**offload), FLASH)
    assert (reasons == []) == ok, reasons
    assert ok or "KV offload other than HiCache" in reasons[0], reasons


@pytest.mark.parametrize(
    "cfg, temporal, conv, want",
    [
        (_cfg(steps=3, draft=4), torch.bfloat16, torch.bfloat16, "one draft chain"),
        (_cfg(topk=4, draft=8), torch.bfloat16, torch.bfloat16, "one draft chain"),
        (_cfg(draft=5), torch.bfloat16, torch.bfloat16, "one draft chain"),
        (_cfg(algorithm="NGRAM"), torch.bfloat16, torch.bfloat16, "MTP (EAGLE)"),
        (_cfg(), torch.float16, torch.bfloat16, "KDA state in f32 or bf16"),
        (_cfg(), torch.bfloat16, torch.float32, "conv state in bf16"),
    ],
)
def test_config_reasons(monkeypatch, cfg, temporal, conv, want):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes(temporal, conv))
    _flydsl(monkeypatch, True)
    reasons = flash.SPEC.config_disable_reasons(cfg, FLASH)
    assert len(reasons) == 1 and want in reasons[0], reasons


def test_config_reasons_name_another_model_and_a_missing_flydsl(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    _flydsl(monkeypatch, False)
    other = SimpleNamespace(
        text_config=SimpleNamespace(
            **{**vars(TEXT), "hidden_size": 6144},
        )
    )
    joined = " | ".join(flash.SPEC.config_disable_reasons(_cfg(), other))
    assert "hidden_size = 4096" in joined and "needs FlyDSL" in joined, joined


def _replayssm_spec(monkeypatch, on):
    import sglang.srt.runtime_context as runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(mamba=SimpleNamespace(enable_linear_replayssm_spec=on)),
    )


def test_model_reasons_refuse_other_tp_sizes(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    _replayssm_spec(monkeypatch, True)
    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(tp_size=2)
    )
    model = SimpleNamespace(
        is_mrope_enabled=False, model=SimpleNamespace(layers=[], layers_to_capture=None)
    )
    joined = " | ".join(flash.SPEC.model_disable_reasons(model, FLASH, {}))
    assert f"built for TP{wide.TP}" in joined and "no KDA layer" in joined, joined


@pytest.mark.parametrize("on, launch", [(True, "wide"), (False, "narrow")])
def test_the_replayssm_spec_arg_picks_the_launch(monkeypatch, on, launch):
    import sglang.srt.runtime_context as runtime_context

    def narrow():
        raise LookupError("narrow")

    _replayssm_spec(monkeypatch, on)
    monkeypatch.setattr(runtime_context, "get_parallel", narrow)
    monkeypatch.setattr(ws, "model_disable_reasons", lambda model, hf, c: ["wide"])
    monkeypatch.setattr(ws, "build", lambda model, hf, ctx: "wide")
    try:
        got = flash.SPEC.model_disable_reasons(None, FLASH, {})[0]
    except LookupError as e:
        got = str(e)
    assert got == launch
    if on:
        assert flash.SPEC.build(None, FLASH, None) == "wide"


class _LM:
    def __init__(self):
        self.ran = []

    def forward(self, input_ids, positions, forward_batch, input_embeds, pp):
        self.ran.append("model")
        return "model"


@pytest.mark.parametrize(
    "rows, tokens, embeds",
    [(wide.NARROW, 6, None), (18, 4, None), (18, 6, torch.zeros(1))],
)
def test_narrow_steps_other_chains_and_embeddings_run_the_model_s_forward(
    rows, tokens, embeds
):
    lm = _LM()
    built = ws._Built(lm, SimpleNamespace(L=6))
    plan = SimpleNamespace(key=SimpleNamespace(rows=rows, tokens_per_request=tokens))
    out = built.forward(plan, None, None, SimpleNamespace(), embeds)
    assert out == "model" and lm.ran == ["model"]
