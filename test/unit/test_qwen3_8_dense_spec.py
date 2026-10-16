"""The Qwen3.8-27B dense gfx950 monokernel spec on CPU (RFC #43949): its
declarations, its registry entry, the steps its forward hands back to the model,
its config refusal reasons and the weight and state checks behind its model
refusal reasons. SGLang checks its declared requirements, the same way for every
spec."""

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.qwen3_8_dense import HF_CONFIG
from sgl_amd_mono.qwen3_8_dense import spec as dense

ARCH = "Qwen3_5ForConditionalGeneration"
# Qwen3.8-27B's config.json, as far as the spec reads it
QWEN = SimpleNamespace(
    text_config=SimpleNamespace(**HF_CONFIG),
    architectures=[ARCH],
    quantization_config={"quant_method": "quark"},
)


def test_the_spec_declares_its_widths_requirements_and_bounds():
    gfx950 = dense.SPEC.devices["gfx950"]
    assert dense.SPEC.name == "qwen3_8_dense_hip"
    assert gfx950.widths == tuple(range(1, 9)) and gfx950.tp_sizes == (1, 2)
    assert dense.SPEC.features == frozenset()
    assert dense.SPEC.requires.env == {"SGLANG_USE_AITER": True}
    assert dense.SPEC.requires.server_args == {
        "enable_linear_replayssm": frozenset({False})
    }
    assert dense.SPEC.requires.dtypes == frozenset({"bfloat16"})
    assert dense.SPEC.api_version == registry.MONOKERNEL_API_VERSION == 1
    # K1 and K2 are exact by construction: every bound is bit-exact
    assert dense.SPEC.numerics is dense.NUMERICS
    assert dense.NUMERICS.kind is registry.NumericsKind.BITWISE
    assert dense.NUMERICS.bounds and all(
        name.endswith(".max_abs") and bound == 0.0
        for name, bound in dense.NUMERICS.bounds.items()
    )


def test_each_tp_caps_its_rows_at_a_built_width():
    # TP1 runs up to 4 rows, TP2 up to 8: wider steps lose to SGLang's path
    assert dense.TP_MAX_ROWS == {1: 4, 2: 8}
    assert tuple(dense.TP_MAX_ROWS) == dense.TP_SIZES
    assert all(cap in dense.ROWS for cap in dense.TP_MAX_ROWS.values())


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[ARCH], isa="gfx950"
    )
    assert found is dense.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[ARCH], isa="gfx942"
    )
    assert found is None and "gfx942" in why


class _Untouchable:
    """A model part or runtime that fails on any use."""

    def __init__(self, what):
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name):
        raise AssertionError(f"touched {self._what}.{name}")

    def __setattr__(self, name, value):
        raise AssertionError(f"set {self._what}.{name}")


class _Lm:
    """A model whose own forward records that it ran; its layers fail on any use."""

    def __init__(self):
        self.model, self.ran = _Untouchable("model.model"), []

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


def _plan(rows):
    key = registry.ShapeKey(
        forward_mode="decode", batch_size=rows, tokens_per_request=1
    )
    return registry.Plan(key=key)


@pytest.mark.parametrize("tp", dense.TP_SIZES)
def test_the_model_runs_the_steps_the_kernels_do_not(tp):
    # building wraps nothing and launches nothing; a step wider than the TP's row
    # cap, an embedding step or one fed input embeddings runs the model's own
    # forward, past the runtime
    lm = _Lm()
    built = dense._Built(lm, _Untouchable("rt"), {}, {}, tp=tp, check=False)
    batch = SimpleNamespace(input_embeds=None)
    embeds = SimpleNamespace(input_embeds=torch.zeros(1, 8))
    cap = dense.TP_MAX_ROWS[tp]
    steps = [
        (_plan(cap + 1), batch, False),
        (_plan(1), batch, True),
        (_plan(1), embeds, False),
    ]
    for plan, fb, get_embedding in steps:
        out = built.forward(plan, "ids", "positions", fb, get_embedding, "pp")
        assert out == "the model's forward"
    assert lm.ran == [
        ("ids", "positions", batch, False, "pp"),
        ("ids", "positions", batch, True, "pp"),
        ("ids", "positions", embeds, False, "pp"),
    ]


def _dtypes(temporal=torch.float32, conv=torch.bfloat16):
    return lambda config: SimpleNamespace(temporal=temporal, conv=conv)


def _flydsl(installed):
    find_spec = importlib.util.find_spec

    def find(name, *args):
        if name == "flydsl":
            return SimpleNamespace(name=name) if installed else None
        return find_spec(name, *args)

    return find


def test_config_reasons_name_each_mismatch(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    monkeypatch.setattr(importlib.util, "find_spec", _flydsl(True))
    cfg = SimpleNamespace()
    assert dense.SPEC.config_disable_reasons(cfg, QWEN) == []
    # a text-only config holds the dims at its top level
    flat = SimpleNamespace(**HF_CONFIG, quantization_config={"quant_method": "quark"})
    assert dense.SPEC.config_disable_reasons(cfg, flat) == []
    # another model's dims, bf16 SSM state, FP8 weights and no FlyDSL
    text = SimpleNamespace(
        **{**HF_CONFIG, "hidden_size": 4096, "linear_num_value_heads": 32}
    )
    other = SimpleNamespace(
        text_config=text,
        architectures=[ARCH],
        quantization_config={"quant_method": "fp8"},
    )
    monkeypatch.setattr(
        mamba_utils, "mamba2_state_dtype", _dtypes(temporal=torch.bfloat16)
    )
    monkeypatch.setattr(importlib.util, "find_spec", _flydsl(False))
    reasons = dense.SPEC.config_disable_reasons(cfg, other)
    joined = " | ".join(reasons)
    for want in (
        "hidden_size = 5120",
        "linear_num_value_heads = 48",
        "f32 SSM / bf16 conv state",
        "Quark MXFP4",
        "FlyDSL",
    ):
        assert want in joined, joined
    assert len(reasons) == 5, reasons


# The weight and state checks import the kernels, which need FlyDSL and aiter


@pytest.fixture
def W():
    pytest.importorskip("flydsl")
    pytest.importorskip("aiter")
    from sgl_amd_mono.qwen3_8_dense.gfx950 import weights

    return weights


def _cache(tp, W, temporal=torch.float32, conv=None, replayssm_d=None):
    geo = W.K.Geo(tp)
    if conv is None:
        conv = torch.empty(2, geo.conv, W.K.SL, dtype=torch.bfloat16)
    return SimpleNamespace(
        temporal=torch.empty(2, geo.nv, W.K.HD, W.K.HD, dtype=temporal),
        conv=[conv],
        replayssm_d=replayssm_d,
    )


@pytest.mark.parametrize("tp", dense.TP_SIZES)
def test_state_reasons(W, tp):
    assert W.state_reason(_cache(tp, W), tp=tp) is None
    assert W.state_reason(_cache(tp, W, temporal=torch.bfloat16), tp=tp).startswith(
        "SSM state"
    )
    assert W.state_reason(_cache(tp, W, replayssm_d=object()), tp=tp) == (
        "ReplaySSM ring"
    )
    narrow = torch.empty(2, W.K.Geo(tp).conv // 2, W.K.SL, dtype=torch.bfloat16)
    assert W.state_reason(_cache(tp, W, conv=narrow), tp=tp).startswith("conv state")
    # the other TP's shapes
    assert W.state_reason(_cache(3 - tp, W), tp=tp) is not None


@pytest.fixture
def quark(monkeypatch):
    """SGLang's Quark linear method and MXFP4 scheme, without their imports."""
    classes = SimpleNamespace(
        method=type("QuarkLinearMethod", (), {}),
        scheme=type("QuarkW4A4MXFP4", (), {}),
    )
    base = "sglang.srt.layers.quantization.quark"
    for name, attr, cls in (
        (f"{base}.quark", "QuarkLinearMethod", classes.method),
        (f"{base}.schemes.quark_w4a4_mxfp4", "QuarkW4A4MXFP4", classes.scheme),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, cls)
        monkeypatch.setitem(sys.modules, name, module)
    return classes


def _mxfp4(quark, n, k, **kw):
    # shapes only: meta tensors hold no memory
    proj = {
        "quant_method": quark.method(),
        "scheme": quark.scheme(),
        "weight": torch.empty(n, k // 2, dtype=torch.uint8, device="meta"),
        "weight_scale": torch.empty(n, k // 32, dtype=torch.uint8, device="meta"),
        **kw,
    }
    return SimpleNamespace(**proj)


def test_mxfp4_reasons(W, quark):
    n, k = 256, 512
    assert W.mxfp4_reason(_mxfp4(quark, n, k), n, k) is None
    assert W.mxfp4_reason(_mxfp4(quark, n, k, scheme=object()), n, k) == (
        "not Quark MXFP4"
    )
    bf16 = _mxfp4(quark, n, k, dequantized_bf16=True)
    assert "dequantized to bf16" in W.mxfp4_reason(bf16, n, k)
    shuffled = _mxfp4(quark, n, k)
    shuffled.weight.is_shuffled = True
    assert "preshuffled (is_shuffled)" in W.mxfp4_reason(shuffled, n, k)
    assert W.mxfp4_reason(_mxfp4(quark, n, k), n, 2 * k).startswith("weight ")
    scales = torch.empty(k // 32, n, dtype=torch.uint8, device="meta").t()
    assert "scales" in W.mxfp4_reason(_mxfp4(quark, n, k, weight_scale=scales), n, k)


def _gdn_layer(W, quark, tp, **attn):
    K, D, geo = W.K, W.D, W.K.Geo(tp)
    norm = SimpleNamespace(
        weight=torch.empty(K.HD, dtype=torch.bfloat16, device="meta"),
        activation="silu",
        group_size=None,
        norm_before_gate=True,
    )
    conv = torch.empty(geo.conv, 1, K.CONV_W, dtype=torch.bfloat16, device="meta")
    attn = {
        "attn_tp_size": tp,
        "in_proj_qkvz": _mxfp4(quark, geo.qkvz, K.HIDDEN),
        "in_proj_ba": _mxfp4(quark, geo.ba, K.HIDDEN),
        "out_proj": _mxfp4(quark, K.HIDDEN, geo.val),
        "conv1d": SimpleNamespace(weight=conv, bias=None),
        "A_log": torch.empty(geo.nv, dtype=torch.float32, device="meta"),
        "dt_bias": torch.empty(geo.nv, dtype=torch.bfloat16, device="meta"),
        "norm": norm,
        "num_k_heads": K.NK,
        "num_v_heads": K.NV,
        "head_k_dim": K.HD,
        "head_v_dim": K.HD,
        **attn,
    }
    inter = D.FGeo(tp).inter
    mlp = SimpleNamespace(
        gate_up_proj=_mxfp4(quark, 2 * inter, D.HIDDEN),
        down_proj=_mxfp4(quark, D.HIDDEN, inter),
    )
    return SimpleNamespace(linear_attn=SimpleNamespace(**attn), mlp=mlp)


@pytest.mark.parametrize("tp", dense.TP_SIZES)
def test_gdn_and_ffn_reasons(W, quark, tp):
    layer = _gdn_layer(W, quark, tp)
    assert W.is_gdn(layer)
    assert W.gdn_reason(layer, tp=tp) is None
    assert W.ffn_reason(layer, tp=tp) is None
    # heads split another way, a conv bias, another gated norm, unquantized ba
    split = _gdn_layer(W, quark, tp, attn_tp_size=3 - tp)
    assert W.gdn_reason(split, tp=tp).startswith("GDN heads split")
    conv = SimpleNamespace(weight=layer.linear_attn.conv1d.weight, bias=object())
    biased = _gdn_layer(W, quark, tp, conv1d=conv)
    assert W.gdn_reason(biased, tp=tp) == "conv1d weight not contiguous or with bias"
    gelu = SimpleNamespace(**{**vars(layer.linear_attn.norm), "activation": "gelu"})
    assert W.gdn_reason(_gdn_layer(W, quark, tp, norm=gelu), tp=tp) == (
        "gated norm variant"
    )
    ba = SimpleNamespace(**{**vars(layer.linear_attn.in_proj_ba), "scheme": None})
    assert W.gdn_reason(_gdn_layer(W, quark, tp, in_proj_ba=ba), tp=tp) == (
        "in_proj_ba: not Quark MXFP4"
    )
    # K2 folds out_proj in at TP1 only
    out = SimpleNamespace(**{**vars(layer.linear_attn.out_proj), "scheme": None})
    folded = _gdn_layer(W, quark, tp, out_proj=out)
    if tp == 1:
        assert W.ffn_reason(folded, tp=tp) == "o_proj: not Quark MXFP4"
    else:
        assert W.ffn_reason(folded, tp=tp) is None
        assert W.gdn_reason(folded, tp=tp) == "out_proj: not Quark MXFP4"
