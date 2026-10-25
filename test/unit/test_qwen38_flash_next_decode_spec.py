"""The Qwen3.8-Flash-Next gfx950 TP2 decode path on CPU (RFC #43949): what the
model's spec declares for it, the TP size routing its refusals and build, its config
and model refusal reasons, the GDN cache views and the radix track-save call it
needs. SGLang checks the declared requirements and features, the same way for every
spec."""

import importlib.util
from types import SimpleNamespace as NS

import pytest
import torch
from qwen38_flash_next_decode_fakes import (
    TEXT_CONFIG,
    fake_model,
    hf_config,
    meta,
    server_args,
)

try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

from sgl_amd_mono import qwen38_flash_next
from sgl_amd_mono.qwen38_flash_next import spec as S
from sgl_amd_mono.qwen38_flash_next.decode import spec as D
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import layout as KL


def test_the_spec_declares_the_tp2_path():
    limits = S.SPEC.devices["gfx950"]
    assert limits.tp_sizes == (1, 2) and D.TP_SIZES == (2,)
    assert limits.resident_blocks == KL.BLOCKS == 256
    # one token a request: the TP2 launches' 1 to 16 rows, and the TP1 path's 20 to
    # 32, which the TP2 forward hands to the model's own
    assert limits.widths == (*range(1, 17), 20, 24, 28, 32)
    assert D.WIDTHS == tuple(range(1, 17))
    assert covered_capture_sizes(
        widths=limits.widths,
        tokens_per_request=1,
        sizes=[1, 2, 4, 8, 12, 16, 24, 32, 40],
    ) == [*range(1, 17), 20, 24, 28, 32, 40]


def test_the_requirements_keep_the_gdn_state_f32_and_in_place():
    req = S.SPEC.requires.server_args
    assert req["moe_runner_backend"] == frozenset({"auto", "aiter"})
    assert req["enable_linear_replayssm"] == frozenset({False})
    assert req["expert_distribution_recorder_mode"] == frozenset({None})
    assert D.SERVER_ARGS == {
        "speculative_algorithm": frozenset({None}),
        "mamba_ssm_dtype": frozenset({None, "float32"}),
    }


def test_hf_config_is_the_kernels_shapes():
    assert D.HF_CONFIG == {
        k: v for k, v in TEXT_CONFIG.items() if k != "norm_topk_prob"
    }
    # the same model as the package's config
    assert D.HF_CONFIG.items() <= qwen38_flash_next.HF_CONFIG.items()


@pytest.fixture
def flydsl(monkeypatch):
    """FlyDSL found or not, whatever this host has."""
    real = importlib.util.find_spec

    def find(found: bool):
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: (
                (object() if found else None) if name == "flydsl" else real(name, *a)
            ),
        )

    find(True)
    return find


def test_config_reasons_clean(flydsl, monkeypatch):
    monkeypatch.delenv("SGLANG_MAMBA_SSM_DTYPE", raising=False)
    monkeypatch.delenv("SGLANG_MAMBA_CONV_DTYPE", raising=False)
    assert S.SPEC.config_disable_reasons(server_args(), hf_config()) == []
    # norm_topk_prob absent from the checkpoint's config: the router renormalizes
    text = hf_config()
    del text.text_config.norm_topk_prob
    assert S.SPEC.config_disable_reasons(server_args(), text) == []
    # the decode-phase override picks the backend the step runs
    args = server_args(
        linear_attn_backend="flashinfer", linear_attn_decode_backend="triton"
    )
    assert S.SPEC.config_disable_reasons(args, hf_config()) == []


@pytest.mark.parametrize(
    "args, config, env, match",
    [
        ({}, {"moe_intermediate_size": 768}, {}, "moe_intermediate_size = 640"),
        (
            {},
            {"shared_expert_intermediate_size": 768},
            {},
            "shared_expert_intermediate_size = 640",
        ),
        ({}, {"hidden_act": "gelu"}, {}, "hidden_act = 'silu'"),
        ({}, {"num_attention_heads": 32}, {}, "num_attention_heads = 24"),
        ({}, {"norm_topk_prob": False}, {}, "norm_topk_prob false"),
        ({}, {"quant": "fp8"}, {}, "Quark MXFP4 checkpoint, got 'fp8'"),
        ({}, {}, {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}, "GDN state torch.bfloat16"),
        ({}, {}, {"SGLANG_MAMBA_CONV_DTYPE": "float32"}, "conv torch.float32"),
        (
            {"linear_attn_backend": "flashinfer"},
            {},
            {},
            "GDN decode backend flashinfer",
        ),
        (
            {"linear_attn_decode_backend": "flashinfer"},
            {},
            {},
            "GDN decode backend flashinfer",
        ),
        (
            {"mamba_ssm_dtype": "bfloat16"},
            {},
            {},
            "mamba_ssm_dtype 'bfloat16' is not one of ['None', 'float32']",
        ),
        (
            {"speculative_algorithm": "EAGLE"},
            {},
            {},
            "speculative_algorithm 'EAGLE' is not one of ['None']",
        ),
    ],
)
def test_config_reasons_refuse(flydsl, monkeypatch, args, config, env, match):
    monkeypatch.delenv("SGLANG_MAMBA_SSM_DTYPE", raising=False)
    monkeypatch.delenv("SGLANG_MAMBA_CONV_DTYPE", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reasons = S.SPEC.config_disable_reasons(server_args(**args), hf_config(**config))
    assert len(reasons) == 1 and match in reasons[0], reasons


def test_config_reasons_need_flydsl(flydsl):
    flydsl(False)
    assert S.SPEC.config_disable_reasons(server_args(), hf_config()) == ["needs FlyDSL"]


def test_the_tp_size_selects_the_path(flydsl, monkeypatch):
    monkeypatch.delenv("SGLANG_MAMBA_SSM_DTYPE", raising=False)
    # TP1: the NEXTN verify path's server arguments in SGLang's words, then its own
    # refusals
    tp1 = server_args(
        tp_size=1,
        speculative_eagle_topk=None,
        speculative_num_steps=None,
        speculative_num_draft_tokens=None,
        context_length=None,
    )
    reasons = S.SPEC.config_disable_reasons(tp1, hf_config())
    assert reasons[:4] == [
        "speculative_algorithm None is not one of ['EAGLE']",
        "speculative_eagle_topk None is not one of ['1']",
        "speculative_num_steps None is not one of ['3']",
        "speculative_num_draft_tokens None is not one of ['4']",
    ]
    assert reasons[4:] == S._config_disable_reasons(tp1, hf_config()) != []
    assert S._PATHS[1].model_disable_reasons is S._model_disable_reasons
    assert S._PATHS[1].build is S._build
    # another TP size: SGLang refuses it by the spec's tp_sizes
    assert S.SPEC.config_disable_reasons(server_args(tp_size=4), hf_config()) == []


@pytest.fixture
def tp2(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    monkeypatch.setattr(runtime_context, "get_parallel", lambda: NS(tp_size=2))


def _set_track_save(monkeypatch, fn):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )

    monkeypatch.setattr(MambaAttnBackendBase, "_track_mamba_state_decode", fn)


def test_model_reasons_need_the_track_save_the_forward_calls(tp2, monkeypatch):
    model = fake_model()
    # the installed SGLang's backend takes the call
    assert S.SPEC.model_disable_reasons(model, hf_config(), {}) == []
    for fn in (None, lambda self, forward_batch, conv_states, ssm_states: None):
        _set_track_save(monkeypatch, fn)
        reasons = S.SPEC.model_disable_reasons(model, hf_config(), {})
        assert len(reasons) == 1, reasons
        assert reasons[0].endswith(f"does not take {D.TRACK_SAVE_ARGS}")


@pytest.mark.parametrize("rip", [384, 512])
def test_layout_reasons_clean_at_either_padding(rip):
    assert D.layout_reasons(fake_model(rip=rip), 2) == []


def test_layout_reasons_padding_short_of_the_read():
    reasons = D.layout_reasons(fake_model(rip=352), 2)
    assert reasons == [
        f"layer {i} experts: intermediate padded to 352, the launch reads 384"
        for i in range(48)
    ]


def test_layout_reasons_mixed_padding():
    model = fake_model(rip=512)
    model.model.layers[5].mlp = fake_model(rip=384).model.layers[5].mlp
    assert D.layout_reasons(model, 2) == [
        "the layers' experts are padded differently: [384, 512]"
    ]


def test_layout_reasons_name_the_broken_tensor():
    model = fake_model(rip=512)
    layers = model.model.layers
    layers[0].attn_hyper_connection.block_inject_weight.weight = meta(10240, 4)
    layers[4].linear_attn.in_proj_qkvz.weight = meta(4096, 2560)
    layers[7].mlp.experts.num_fused_shared_experts = 0
    layers[8].mlp.experts.w2_weight = None
    layers[9].linear_attn.norm.activation = "swish"
    layers[10].linear_attn.dt_bias = meta(24)  # bf16: accepted
    layers[11].o_proj.weight = meta(2560, 1536, dtype=torch.uint8)
    layers[12].mlp.experts.w2_weight_scale = meta(513, 2560, 8, dtype=torch.uint8)
    layers[13].linear_attn.dt_bias = meta(48)
    layers[14].mlp.experts.w13_weight.is_shuffled = False
    layers[16].linear_attn.conv1d.bias = meta(5120)
    layers[17].linear_attn.in_proj_qkvz.output_sizes = [6144, 6144, 2048, 2048]
    layers[18].linear_attn.activation = "gelu"
    layers[19].attn_output_gate = False
    assert D.layout_reasons(model, 2) == [
        "layer 0 attn_hyper_connection inject: (10240, 4) torch.bfloat16, expected "
        "(4, 10240) torch.bfloat16",
        "layer 4 in_proj_qkvz: (4096, 2560) torch.bfloat16, expected (8192, 2560) "
        "torch.bfloat16",
        "layer 7: the kernels read the shared expert as expert 512",
        "layer 8 experts: missing w13_weight / w2_weight",
        "layer 9 gated norm activation 'swish', the launch gates with sigmoid",
        "layer 11 o_proj: (2560, 1536) torch.uint8, expected (2560, 3072) "
        "torch.bfloat16",
        "layer 12 w2 scales: (513, 2560, 8) torch.uint8, expected 21012480 e8m0",
        "layer 13 dt_bias: (48,) torch.bfloat16, expected (24,) float",
        "layer 14 w13: not in AITER's shuffled layout",
        "layer 16 conv1d has a bias",
        "layer 17 in_proj_qkvz output_sizes [6144, 6144, 2048, 2048], expected "
        "[q, k, v, z]",
        "layer 18 conv activation 'gelu', the launch applies SiLU",
        "layer 19: the o_proj launch applies the output gate",
    ]


def test_layout_reasons_layer_mix():
    model = fake_model(rip=384)
    model.model.layers[3] = model.model.layers[0]
    assert D.layout_reasons(model, 2) == [
        "37 GDN and 11 QSA layers, expected 36 and 12"
    ]


def test_model_reasons_refuse_the_padding_short_of_the_read(tp2):
    assert len(S.SPEC.model_disable_reasons(fake_model(rip=352), hf_config(), {})) == 48


def test_gdn_state_reasons_and_view():
    conv, ssm = meta(9, 5120, 3), meta(9, 24, 128, 128, dtype=torch.float32)
    assert D.gdn_state_reasons(conv, ssm, 2) == []
    bf16 = D.gdn_state_reasons(conv, meta(9, 24, 128, 128), 2)
    assert len(bf16) == 1 and bf16[0].startswith("recurrent state")
    window_first = D.gdn_state_reasons(meta(9, 3, 5120), ssm, 2)
    assert len(window_first) == 1 and window_first[0].startswith("conv state")
    sparse = torch.zeros(3, 24, 128, 256)[..., :128]
    assert D.gdn_state_reasons(
        torch.zeros(3, 5120, 3, dtype=torch.bfloat16), sparse, 2
    )[0].startswith("recurrent state slot not dense")
    # SGLang's pool keeps the conv window first; the view puts it last
    pool = NS(conv=[meta(9, 3, 5120)], temporal=ssm)
    view, state = D.gdn_state_view(pool)
    assert tuple(view.shape) == (9, 5120, 3) and state is ssm
    assert D.gdn_state_reasons(view, state, 2) == []
