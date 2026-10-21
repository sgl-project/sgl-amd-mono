"""MXFP4 loader refusals and quant-mode dispatch without device allocation."""

from types import SimpleNamespace

import msgspec
import pytest
import torch

from sgl_amd_mono.glm53.gfx950.runner import Glm53Build, Glm53LayerOp
from sgl_amd_mono.glm53.gfx950.weights import experts_reason
from sgl_amd_mono.glm53.spec import _Built


def experts():
    def tensor(shape):
        w = torch.empty(shape, device="meta", dtype=torch.uint8)
        w.is_shuffled = True
        return w

    return SimpleNamespace(
        scheme=type("QuarkW4A4MXFp4MoE", (), {})(),
        moe_runner_config=SimpleNamespace(activation="silu"),
        w13_weight=tensor((257, 1024, 3072)),
        w2_weight=tensor((257, 6144, 256)),
        w13_weight_scale=tensor((257, 1024, 192)),
        w2_weight_scale=tensor((257, 6144, 16)),
    )


def test_loader_accepts_separated_aiter_mxfp4_without_copying():
    e = experts()
    assert experts_reason(SimpleNamespace(experts=e)) is None


def test_layer_weights_preserve_loaded_mxfp4_storage(monkeypatch):
    from sgl_amd_mono.glm53.gfx950 import weights

    def tensor(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def norm(n):
        return SimpleNamespace(weight=tensor(n))

    def linear(rows, k):
        return SimpleNamespace(
            weight=tensor(rows, k, dtype=torch.float8_e4m3fn),
            weight_scale_inv=tensor(
                (rows + 127) // 128, (k + 127) // 128, dtype=torch.float32
            ),
            aiter_bpreshuffled=True,
        )

    e = experts()
    attn = SimpleNamespace(
        q_a_layernorm=norm(2048),
        kv_a_layernorm=norm(512),
        indexer=None,
        fused_qkv_a_proj_with_mqa=linear(2624, 6144),
        q_b_proj=linear(4096, 2048),
        o_proj=linear(6144, 4096),
    )
    module = SimpleNamespace(
        input_layernorm=norm(6144),
        post_attention_layernorm=norm(6144),
        self_attn=attn,
        mlp=SimpleNamespace(
            experts=e,
            gate=SimpleNamespace(
                weight=tensor(256, 6144),
                e_score_correction_bias=tensor(256, dtype=torch.float32),
            ),
        ),
    )
    monkeypatch.setattr(weights, "_uk_uv", lambda *args: (tensor(1),) * 4)
    out = weights.layer_weights(module, 3, 4, {})
    assert out.expert_mxfp4
    assert out.w_ug is e.w13_weight and out.w_dn is e.w2_weight
    assert out.s_ug is e.w13_weight_scale and out.s_dn is e.w2_weight_scale


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda e: setattr(e, "scheme", object()), "MXFP4 expert scheme"),
        (
            lambda e: setattr(e.moe_runner_config, "activation", "situ"),
            "separated gate/up",
        ),
        (lambda e: setattr(e.w13_weight, "is_shuffled", False), "shuffled"),
        (lambda e: setattr(e, "w2_weight_scale", None), "w2_weight_scale"),
        (lambda e: setattr(e, "w13_weight", e.w13_weight[:256]), "fused shared"),
        (lambda e: setattr(e, "w13_weight", e.w13_weight[:, :512]), "TP4"),
    ],
)
def test_loader_rejects_ambiguous_or_mismatched_storage(change, reason):
    e = experts()
    change(e)
    assert reason in experts_reason(SimpleNamespace(experts=e))


@pytest.mark.parametrize("mode, w4", [("fp8", False), ("fp4", True)])
def test_build_key_includes_weight_and_activation_formats(mode, w4):
    key = Glm53Build(
        16, 4, 1, True, 4096, True, expert_mxfp4=w4, moe_activation_dtype=mode
    )
    assert msgspec.json.decode(msgspec.json.encode(key))["moe_activation_dtype"] == mode


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"moe_activation_dtype": "fp4"}, "requires MXFP4"),
        ({"moe_activation_dtype": "int4"}, "fp8 or fp4"),
        ({"expert_mxfp4": True}, "A8W4 is unsupported"),
        ({"tp": 8, "expert_mxfp4": True, "moe_activation_dtype": "fp4"}, "TP4"),
        (
            {"rows": 17, "expert_mxfp4": True, "moe_activation_dtype": "fp4"},
            "1 to 16",
        ),
    ],
)
def test_unsupported_modes_fail_before_allocation(kwargs, reason):
    options = dict(
        rows=4,
        tp=4,
        launches_per_step=1,
        with_indexer=True,
        index_max_seq=4096,
        fp8_shuffled=True,
    )
    with pytest.raises(ValueError, match=reason):
        Glm53Build(**(options | kwargs))


def test_layer_rejects_weight_runtime_format_mismatch():
    rt = SimpleNamespace(
        key=Glm53Build(
            4, 4, 1, False, 0, True, expert_mxfp4=True, moe_activation_dtype="fp4"
        )
    )
    with pytest.raises(ValueError, match="weight format"):
        Glm53LayerOp(rt, SimpleNamespace(index=None, shuffled=True), 0)


@pytest.mark.parametrize(
    "loaded, requested", [("fp8", "fp4"), ("fp4", "fp8"), ("fp4", "int4")]
)
def test_mismatched_activation_fails_before_touching_state(loaded, requested):
    built = object.__new__(_Built)
    built.moe_activation_dtype = loaded
    with pytest.raises(ValueError):
        built.forward(None, None, None, None, moe_activation_dtype=requested)


def test_model_accepts_mxfp4_experts_with_block_fp8_attention():
    from test_glm53_spec import GLM53, _layers, _model

    from sgl_amd_mono.glm53.spec import _model_disable_reasons

    layers = _layers()
    for layer in layers[3:]:
        layer.mlp.experts = experts()
    assert _model_disable_reasons(_model(layers), GLM53, {}) == []


@pytest.mark.parametrize("rows", range(1, 17))
def test_native_physical_rows_have_complete_tiles_and_separate_epochs(rows):
    from sgl_amd_mono.glm53.gfx950.layout import layout

    key = Glm53Build(
        rows, 4, 1, True, 4096, True, expert_mxfp4=True, moe_activation_dtype="fp4"
    )
    sc, sy = layout(key.rows, 16, 4, 2048, True, 4096, inter=512, native_fp4_mfma=True)
    tile, count = sc["_tile_rows"], sc["_tile_count"]
    assert tile <= 12 and tile * count == rows
    if count == 2:
        assert sc["index_q_rot"] + tile * 4096 // 2 * 8 <= sc["_tile_stride"]
    if count > 1:
        assert sc["_tile_barriers"] >= count * sc["_tile_stride"]
        assert sc["_bytes"] - sc["_tile_barriers"] >= (count - 1) * 257 * 8
        assert sy["flags"] == count * sy["_tile_stride"]
