"""The Qwen3.8-Flash-Next gfx950 monokernel spec on CPU (RFC #43949): its registry
entry, the configurations it refuses before load, the layer views it refuses after
load (meta tensors of the real shapes), the verify widths it serves on the NEXTN
recipe, its build keys and its fallback forward. SGLang checks the declared
requirements, the same way for every spec."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry
try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

import sgl_amd_mono
from sgl_amd_mono.common.plan import key_tuple
from sgl_amd_mono.qwen38_flash_next import (
    HF_CONFIG,
    INDEXER_CONFIG,
    LAYER_PATTERN,
    MAX_CTX,
)
from sgl_amd_mono.qwen38_flash_next import spec as q38
from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import weights
from sgl_amd_mono.qwen38_flash_next.gfx950.config import ConvStrides, MonoBuild
from sgl_amd_mono.qwen38_flash_next.gfx950.runner import launch_plans

TARGET = "Qwen4ExpForConditionalGeneration"
# Qwen3.8-Flash-Next's text config (amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-
# PLEFP8 config.json, text_config)
TEXT = SimpleNamespace(
    **HF_CONFIG,
    **INDEXER_CONFIG,
    layer_types=list(LAYER_PATTERN) * 12,
    mamba_ssm_dtype="float32",
    max_position_embeddings=262144,
    eos_token_id=248044,
)
VL = SimpleNamespace(
    text_config=TEXT,
    architectures=[TARGET],
    quantization_config={"quant_method": "quark"},
)
# the NEXTN recipe's server arguments the spec reads
CFG = SimpleNamespace(context_length=None)


def _replace(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def _config_reasons(cfg=CFG, hf_config=VL, flydsl=True):
    with mock.patch.object(q38, "_has_flydsl", return_value=flydsl):
        return q38._config_disable_reasons(cfg, hf_config)


def test_the_plugin_registers_the_target():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[TARGET], isa="gfx950"
    )
    assert found is q38.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[TARGET], isa="gfx942"
    )
    assert found is None and "gfx942" in why


def test_the_recipe_is_accepted():
    assert _config_reasons() == []
    assert _config_reasons(cfg=_replace(CFG, context_length=131072)) == []


@pytest.mark.parametrize(
    "cfg, hf_config, flydsl",
    [
        (CFG, _replace(VL, text_config=_replace(TEXT, hidden_size=4096)), True),
        (CFG, _replace(VL, text_config=_replace(TEXT, num_experts=256)), True),
        (CFG, _replace(VL, text_config=_replace(TEXT, indexer_budget=1024)), True),
        (
            CFG,
            _replace(
                VL,
                text_config=_replace(TEXT, layer_types=["linear_attention"] * 48),
            ),
            True,
        ),
        (CFG, _replace(VL, quantization_config={"quant_method": "fp8"}), True),
        (
            CFG,
            _replace(VL, text_config=_replace(TEXT, mamba_ssm_dtype="float16")),
            True,
        ),
        (_replace(CFG, context_length=MAX_CTX), VL, True),
        (CFG, VL, False),
    ],
    ids=[
        "hidden size",
        "experts",
        "index budget",
        "layer pattern",
        "quantization",
        "ssm dtype",
        "context",
        "no flydsl",
    ],
)
def test_each_unsupported_setting_is_one_reason(cfg, hf_config, flydsl):
    assert len(_config_reasons(cfg=cfg, hf_config=hf_config, flydsl=flydsl)) == 1


def test_the_ple_layer_is_one_reason():
    # PLE on layer 1 (ple_layer_ids counts from 1) is built into the second launch
    text = _replace(TEXT, ple_layer_ids=[2, 6])
    assert len(_config_reasons(hf_config=_replace(VL, text_config=text))) == 1


def test_the_recipe_routes_verify_batches_of_one_to_eight():
    # NEXTN, 3 steps: 4 verify rows a request, the 4- to 32-row launches
    assert q38.SPEC.widths("gfx950") == tuple(range(4, 33, 4))
    assert q38.SPEC.devices["gfx950"].tp_sizes == (1,)
    assert covered_capture_sizes(
        widths=q38.SPEC.widths("gfx950"), tokens_per_request=4, sizes=[1, 2, 4, 8]
    ) == list(range(1, 9))
    # above 8 requests the normal path, at the configured sizes
    assert covered_capture_sizes(
        widths=q38.SPEC.widths("gfx950"), tokens_per_request=4, sizes=[1, 2, 4, 16]
    ) == [*range(1, 9), 16]


def test_the_requirements_are_declared_sglang_reads():
    # SGLang checks each by name: test_contract.py pins the names it reads
    reads = q38.SGLANG_READS
    assert set(q38.SPEC.requires.server_args) <= set(
        reads["srt/arg_groups/fields/*.py"]
    )
    assert set(q38.SPEC.requires.env) <= set(reads["srt/environ.py"])
    # NEXTN resolves to EAGLE; the kernel verifies 4 rows a request of one chain
    args = q38.SPEC.requires.server_args
    assert args["speculative_algorithm"] == {"EAGLE"}
    assert args["speculative_num_draft_tokens"] == {q38.TOKENS}
    assert args["enable_linear_replayssm_spec"] == {False}


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


@pytest.mark.parametrize(
    "key",
    [
        registry.ShapeKey("target_verify", batch_size=2, tokens_per_request=2),
        registry.ShapeKey("target_verify", batch_size=9, tokens_per_request=4),
        registry.ShapeKey("decode", batch_size=4, tokens_per_request=1),
    ],
    ids=["two tokens", "nine requests", "decode"],
)
def test_a_step_the_launches_do_not_take_runs_the_models_forward(key):
    lm = _Lm()
    built = q38._Built(lm, runner=None)
    out = built.forward(registry.Plan(key=key), "ids", "positions", "batch")
    assert out == "the model's forward"
    assert lm.ran == [("ids", "positions", "batch", False, None)]


# ---------------------------------------------------------------- build keys
def test_a_build_key_reaches_the_jit_cache_key():
    key = MonoBuild(plan="gdn,gdn,gdn,qsa", nreq=8, ntok=4, rtt_stride=262208)
    pairs = dict(key_tuple(key, "digest"))
    assert pairs["plan"] == "gdn,gdn,gdn,qsa" and pairs["nreq"] == 8
    assert pairs["iconv.step"] == ConvStrides().step and pairs["sources"] == "digest"
    assert key.layers == ("gdn", "gdn", "gdn", "qsa")
    # a frozen Struct: the runner caches its launchers by key
    assert {key: 1}[MonoBuild(plan="gdn,gdn,gdn,qsa", nreq=8, ntok=4, rtt_stride=262208)]  # fmt: skip


@pytest.mark.parametrize(
    "kw",
    [
        dict(plan="gdn,mla", nreq=1, ntok=4),
        dict(plan="", nreq=1, ntok=4),
        dict(plan="qsa", nreq=9, ntok=1),
        dict(plan="qsa", nreq=1, ntok=5),
        dict(plan="qsa", nreq=8, ntok=4, first_src="hidden"),
    ],
    ids=["kind", "empty", "requests", "tokens", "first source"],
)
def test_a_build_key_out_of_the_kernels_range_is_refused(kw):
    with pytest.raises(ValueError):
        MonoBuild(**kw)


def test_a_verify_is_thirteen_launches_of_the_48_layers():
    kinds = ["qsa" if t == "full_attention" else "gdn" for t in TEXT.layer_types]
    plans = launch_plans(kinds)
    assert len(plans) == 13
    assert plans[0] == ("gdn", [0]) and plans[1] == ("gdn,gdn,qsa", [1, 2, 3])
    assert all(p == "gdn,gdn,gdn,qsa" for p, _ in plans[2:])
    assert [i for _, ids in plans for i in ids] == list(range(48))


# ---------------------------------------------------------------- layout
@pytest.mark.parametrize("R", range(1, 9))
def test_the_scratch_regions_do_not_overlap(R):
    regions, total = LY.scratch_layout(6, 4 * R, R)
    offs = sorted(regions.values())
    assert offs == sorted(set(offs)) and offs[0] == 0 and total > offs[-1]
    assert all(off % 256 == 0 for off in offs)


def test_the_layout_matches_the_spec():
    assert LY.MAX_CTX == MAX_CTX
    assert LY.QP_RG_QKV * 16 + 640 == LY.QP_ROWS
    assert len(LAYER_PATTERN) == HF_CONFIG["full_attention_interval"]


# ---------------------------------------------------------------- layer views
def _meta(*shape, dtype=torch.bfloat16, **attrs):
    t = torch.empty(*shape, dtype=dtype, device="meta")
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


def _fp8(n, k, scale=True):
    """A Quark W8A8 FP8 linear as SGLang keeps it: the [N, K] weight behind a
    transposed view, f32 per-channel scales."""
    return SimpleNamespace(
        weight=_meta(n, k, dtype=torch.float8_e4m3fn).t(),
        weight_scale=_meta(n, dtype=torch.float32) if scale else None,
    )


def _w(*shape, dtype=torch.bfloat16):
    return SimpleNamespace(weight=_meta(*shape, dtype=dtype))


def _hc():
    return SimpleNamespace(
        hc_norm=_w(10240),
        input_mix_weight_down=_w(320, 10240),
        input_mix_weight_up=_w(10240, 320),
        block_inject_weight=_w(4, 10240),
    )


def _experts():
    E = 513
    return SimpleNamespace(
        num_fused_shared_experts=1,
        moe_runner_config=SimpleNamespace(activation="silu"),
        w13_weight=_meta(E, 1536, 1280, dtype=torch.uint8, is_shuffled=True),
        w2_weight=_meta(E, 2560, 384, dtype=torch.uint8, is_shuffled=True),
        w13_weight_scale=_meta(E, 1536, 80, dtype=torch.uint8),
        w2_weight_scale=_meta(E, 2560, 24, dtype=torch.uint8),
    )


def _layer(kind, ple=False, **changes):
    """A decoder layer as SGLang builds it, its tensors on the meta device."""
    layer = SimpleNamespace(
        attn_hyper_connection=_hc(),
        mlp_hyper_connection=_hc(),
        mlp=SimpleNamespace(
            experts=_experts(), gate=_w(512, 2560), shared_expert_gate=_w(1, 2560)
        ),
        ple=None,
    )
    if kind == "gdn":
        layer.linear_attn = SimpleNamespace(
            in_proj_qkvz=_fp8(16384, 2560),
            in_proj_ba=_fp8(96, 2560),
            out_proj=_fp8(2560, 6144),
            A_log=_meta(48, dtype=torch.float32),
            dt_bias=_meta(48),
            conv1d=_w(10240, 1, 4),
            norm=_w(128),
        )
    else:
        layer.is_qsa = True
        layer.qkv_proj = _fp8(13312, 2560)
        layer.o_proj = _fp8(2560, 6144)
        layer.q_norm, layer.k_norm = _w(256), _w(256)
        layer.indexer = SimpleNamespace(
            index_qk_proj=_fp8(640, 2560), q_layernorm=_w(128), k_layernorm=_w(128)
        )
        layer.rotary_emb = SimpleNamespace(cos_sin_cache_as=lambda dtype: None)
    if ple:
        layer.ple = SimpleNamespace(
            key_proj=_w(10240, 2560),
            value_proj=_w(2560, 2560),
            norm_key=_w(10240),
            norm_query=_w(10240),
            norm_conv=_w(10240),
            conv1d=_w(10240, 1, 4),
            ple_embedding=SimpleNamespace(
                ngram_embedding=SimpleNamespace(
                    weight=_meta(1000, 160, dtype=torch.float8_e4m3fn),
                    weight_scale=_meta(1),
                )
            ),
        )
    for path, value in changes.items():
        owner, name = layer, path
        while "." in name:
            head, name = name.split(".", 1)
            owner = getattr(owner, head)
        setattr(owner, name, value)
    return layer


def test_the_views_accept_the_layers():
    assert weights.layer_reasons(_layer("gdn"), 0, "gdn") == []
    assert weights.layer_reasons(_layer("gdn", ple=True), 1, "gdn") == []
    assert weights.layer_reasons(_layer("qsa"), 3, "qsa") == []
    assert weights.ngram_reasons(_layer("gdn", ple=True).ple) == []
    assert weights.mixer_reasons(_hc()) == []


@pytest.mark.parametrize(
    "kind, changes",
    [
        ("gdn", {"mlp.experts.num_fused_shared_experts": 0}),
        (
            "gdn",
            {
                "mlp.experts.w13_weight": _meta(
                    513, 1536, 1280, dtype=torch.uint8, is_shuffled=False
                )
            },
        ),
        ("gdn", {"mlp.experts.moe_runner_config": SimpleNamespace(activation="situ")}),
        ("gdn", {"mlp.experts.w2_weight_scale": _meta(513, 2560, 24)}),
        ("gdn", {"linear_attn.out_proj": _fp8(2560, 6144, scale=False)}),
        ("gdn", {"linear_attn.in_proj_ba": _fp8(256, 2560)}),
        ("gdn", {"linear_attn.in_proj_qkvz": _w(16384, 2560)}),
        ("gdn", {"linear_attn.conv1d": _w(10240, 4)}),
        ("gdn", {"attn_hyper_connection.hc_norm": _w(10240, dtype=torch.float32)}),
        ("qsa", {"indexer.index_qk_proj": _fp8(640, 1280)}),
        ("qsa", {"q_norm": _w(128)}),
        ("qsa", {"mlp.gate": _w(256, 2560)}),
    ],
    ids=[
        "no fused shared expert",
        "unshuffled experts",
        "interleaved gate and up",
        "2-byte scales",
        "dequantized projection",
        "wide b / a projection",
        "bf16 projection",
        "conv weight",
        "f32 norm",
        "index projection",
        "q norm",
        "router",
    ],
)
def test_each_unsupported_view_is_refused(kind, changes):
    assert len(weights.layer_reasons(_layer(kind, **changes), 3, kind)) == 1


def test_an_offloaded_ngram_table_is_refused():
    ple = _layer("gdn", ple=True).ple
    ple.ple_embedding.ngram_embedding.weight = torch.empty(
        8, 160, dtype=torch.float8_e4m3fn
    )
    assert len(weights.ngram_reasons(ple)) == 1


def _model(**changes):
    kinds = ["qsa" if t == "full_attention" else "gdn" for t in TEXT.layer_types]
    layers = [_layer(k, ple=i == 1) for i, k in enumerate(kinds)]
    m = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        hyper_connection_mixer=_hc(),
        embed_tokens=_w(248320, 2560),
    )
    lm = SimpleNamespace(model=m, capture_aux_hidden_states=False)
    for name, value in changes.items():
        setattr(lm, name, value)
    return lm


def _model_reasons(lm, tp=1, context_len=262144):
    parallel = SimpleNamespace(tp_size=tp, pp_size=1, attn_cp_size=1, moe_ep_size=1)
    with (
        mock.patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        mock.patch(
            "sglang.srt.runtime_context.process_model_config",
            return_value=SimpleNamespace(context_len=context_len),
        ),
    ):
        return q38._model_disable_reasons(lm, VL, {})


def test_the_model_is_accepted():
    assert _model_reasons(_model()) == []


def test_each_unsupported_model_is_one_reason():
    assert len(_model_reasons(_model(), tp=2)) == 1
    assert len(_model_reasons(_model(), context_len=MAX_CTX)) == 1
    assert len(_model_reasons(_model(capture_aux_hidden_states=True))) == 1
    lm = _model()
    lm.model.layers[5].ple = lm.model.layers[1].ple
    assert len(_model_reasons(lm)) == 1
    lm = _model()
    lm.model.end_layer = 24  # pipeline stage
    assert len(_model_reasons(lm)) == 1
