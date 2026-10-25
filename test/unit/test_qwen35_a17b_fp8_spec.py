"""Qwen3.5-397B-A17B block-FP8 TP4 spec on CPU: registration, the gfx950/TP4
limits, and the refusals for MXFP4, per-channel FP8, a non-TP4 shape, a plain
RMSNorm and a projection with a bias."""

from types import SimpleNamespace

import torch
from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.qwen35_a17b_fp8 import spec as fp8
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.layout import LAYERS

ARCH = ["Qwen3_5MoeForConditionalGeneration"]


def _text(**overrides):
    values = dict(fp8._TEXT)
    values["layer_types"] = [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)
    ]
    values["rope_parameters"] = {
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hf(**quant):
    block = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    block.update(quant)
    return SimpleNamespace(
        text_config=_text(),
        architectures=ARCH,
        quantization_config=block,
    )


def _cfg():
    return SimpleNamespace()


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def _fp8(shape):
    return SimpleNamespace(
        weight=_Tensor(shape, torch.float8_e4m3fn),
        weight_scale_inv=_Tensor(tuple(s // 128 for s in shape[-2:]), torch.float32),
    )


def _bf16(shape):
    return SimpleNamespace(weight=_Tensor(shape, torch.bfloat16))


def _experts(e=513):
    return SimpleNamespace(
        w13_weight=_Tensor((e, 512, 4096), torch.float8_e4m3fn),
        w13_weight_scale_inv=_Tensor((e, 4, 32), torch.float32),
        w2_weight=_Tensor((e, 4096, 256), torch.float8_e4m3fn),
        w2_weight_scale_inv=_Tensor((e, 32, 2), torch.float32),
    )


def _mlp():
    return SimpleNamespace(
        enable_shared_expert_fusion=True,
        reduce_results=False,
        experts=_experts(),
        gate=_bf16((512, 4096)),
    )


class GemmaRMSNorm:
    """Stands in for SGLang's, which the spec knows by name."""


class RMSNorm:
    pass


def _norms():
    return dict(input_layernorm=GemmaRMSNorm(), post_attention_layernorm=GemmaRMSNorm())


def _layer(i):
    if i % 4 == 3:
        qkv_n = 4608
        return SimpleNamespace(
            mlp=_mlp(),
            qkv_proj=_fp8((qkv_n, 4096)),
            o_proj=_fp8((4096, 2048)),
            **_norms(),
        )
    return SimpleNamespace(
        mlp=_mlp(),
        linear_attn=SimpleNamespace(
            in_proj_qkvz=_fp8((5120, 4096)),
            in_proj_ba=_bf16((32, 4096)),
            conv1d=_bf16((3072, 1, 4)),
            out_proj=_fp8((4096, 2048)),
        ),
        **_norms(),
    )


def _model():
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[_layer(i) for i in range(LAYERS)], norm=GemmaRMSNorm()
        )
    )


def test_the_plugin_registers_tp4_block_fp8_on_gfx950():
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=ARCH, isa="gfx950"
    )
    assert found is fp8.SPEC, why
    limits = found.devices["gfx950"]
    assert limits.tp_sizes == (4,)
    assert limits.widths == tuple(range(1, 9))
    missing, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=ARCH, isa="gfx942"
    )
    assert missing is None and "gfx942" in why


def test_config_accepts_block_fp8_and_refuses_mxfp4():
    assert fp8.SPEC.config_disable_reasons(_cfg(), _hf()) == []
    quark = _hf(quant_method="quark", activation_scheme=None, weight_block_size=None)
    joined = " | ".join(fp8.SPEC.config_disable_reasons(_cfg(), quark))
    assert "block-FP8" in joined, joined
    channel = _hf(weight_block_size=[1, 4096])
    assert any(
        "block-FP8" in r for r in fp8.SPEC.config_disable_reasons(_cfg(), channel)
    )
    mx = _hf(quant_method="mxfp4")
    assert any("block-FP8" in r for r in fp8.SPEC.config_disable_reasons(_cfg(), mx))


def test_model_accepts_tp4_block_scales_and_refuses_per_channel():
    assert fp8.SPEC.model_disable_reasons(_model(), _hf(), {}) == []
    model = _model()
    # per-channel AttnFP8: a 1-D scale instead of a 128x128 block scale
    model.model.layers[3].o_proj = SimpleNamespace(
        weight=_Tensor((4096, 2048), torch.float8_e4m3fn),
        weight_scale_inv=_Tensor((4096,), torch.float32),
    )
    reasons = fp8.SPEC.model_disable_reasons(model, _hf(), {})
    assert reasons and "layer 3 o_proj" in reasons[0], reasons


def test_model_refuses_a_plain_rmsnorm():
    model = _model()
    model.model.layers[5].post_attention_layernorm = RMSNorm()
    reasons = fp8.SPEC.model_disable_reasons(model, _hf(), {})
    assert reasons == [
        "layer 5 post_attention_layernorm is RMSNorm, not a Gemma RMSNorm"
    ], reasons


def test_model_refuses_a_projection_with_a_bias():
    model = _model()
    model.model.layers[3].qkv_proj.bias = _Tensor((4608,), torch.bfloat16)
    reasons = fp8.SPEC.model_disable_reasons(model, _hf(), {})
    assert reasons == ["layer 3 qkv has a bias, the block-FP8 GEMV adds none"], reasons
