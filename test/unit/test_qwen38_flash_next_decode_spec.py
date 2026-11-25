"""The Qwen3.8-Flash-Next gfx950 TP2 decode path on CPU (RFC #43949): what the
model's spec declares for it, the TP size routing its refusals and build, its config
and model refusal reasons, the GDN state checks and the linear-attention backend
calls it needs. SGLang checks the declared requirements and features, the same way
for every spec."""

import importlib.util
import json
import os
import subprocess
import sys
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
from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L

# SGLang's default decode capture sizes without speculative decoding, to 192
DEFAULT_DECODE = [1, 2, 4, 8, 12, *range(16, 193, 8)]


def test_the_spec_declares_the_tp2_path():
    limits = S.SPEC.devices["gfx950"]
    assert limits.tp_sizes == (1, 2) and D.TP_SIZES == (2,)
    assert limits.resident_blocks == L.BLOCKS == 256
    assert list(D.WIDTHS) == DEFAULT_DECODE
    # one token a request: the TP2 widths and the TP1 path's verify widths, of which
    # 20 and 28 are not SGLang's decode sizes
    assert limits.widths == tuple(sorted({*DEFAULT_DECODE, 20, 28}))


@pytest.mark.parametrize(
    "sizes, captured",
    [
        # SGLang's default list at --cuda-graph-max-bs 192: the declared widths
        (DEFAULT_DECODE, sorted({*DEFAULT_DECODE, 20, 28})),
        # a smaller largest size: the declared widths up to it
        ([1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64], [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]),
        # a largest size that is no width: the widths below it, then it (its steps
        # run SGLang's graph)
        ([1, 2, 4, 8, 12, 16, 24, 32, 40, 44], [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44]),
        # sizes past 192 stay SGLang's
        ([*DEFAULT_DECODE, 200, 208], sorted({*DEFAULT_DECODE, 20, 28, 200, 208})),
    ],
)  # fmt: skip
def test_sglang_captures_the_declared_widths(sizes, captured):
    widths = S.SPEC.widths("gfx950")
    assert (
        covered_capture_sizes(widths=widths, tokens_per_request=1, sizes=sizes)
        == captured
    )


def test_the_requirements_keep_the_gdn_state_f32_and_in_place():
    req = S.SPEC.requires.server_args
    assert req["moe_runner_backend"] == frozenset({"auto", "aiter"})
    assert req["enable_linear_replayssm"] == frozenset({False})
    assert req["expert_distribution_recorder_mode"] == frozenset({None})
    assert D.SERVER_ARGS == {
        "speculative_algorithm": frozenset({None}),
        "mamba_ssm_dtype": frozenset({None, "float32"}),
        "enable_unified_memory": frozenset({False}),
        "enable_page_major_kv_layout": frozenset({False}),
    }
    reads = D.SGLANG_READS["srt/arg_groups/fields/*.py"]
    assert set(D.SERVER_ARGS) <= set(reads)


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


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "SGLANG_MAMBA_SSM_DTYPE",
        "SGLANG_MAMBA_CONV_DTYPE",
        "SGLANG_AMD_MONO_ROCM_LLVM",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_reasons_clean(flydsl, clean_env):
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
        ({}, {"num_key_value_heads": 4}, {}, "num_key_value_heads = 2"),
        ({}, {"hidden_act": "gelu"}, {}, "hidden_act = 'silu'"),
        ({}, {"num_attention_heads": 32}, {}, "num_attention_heads = 24"),
        ({}, {"norm_topk_prob": False}, {}, "norm_topk_prob false"),
        ({}, {"quant": "fp8"}, {}, "Quark MXFP4 checkpoint, got 'fp8'"),
        ({}, {}, {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}, "GDN state torch.bfloat16"),
        ({}, {}, {"SGLANG_MAMBA_CONV_DTYPE": "float32"}, "conv torch.float32"),
        ({}, {}, {"SGLANG_AMD_MONO_ROCM_LLVM": "/opt/rocm/llvm"}, "FlyDSL's bundled LLVM"),
        ({"linear_attn_backend": "flashinfer"}, {}, {}, "GDN decode backend flashinfer"),
        ({"linear_attn_decode_backend": "flashinfer"}, {}, {}, "GDN decode backend flashinfer"),
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
        ({"enable_unified_memory": True}, {}, {}, "enable_unified_memory True is not one of ['False']"),
        ({"enable_page_major_kv_layout": True}, {}, {}, "enable_page_major_kv_layout True is not"),
    ],
)  # fmt: skip
def test_config_reasons_refuse(
    flydsl, clean_env, monkeypatch, args, config, env, match
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reasons = S.SPEC.config_disable_reasons(server_args(**args), hf_config(**config))
    assert len(reasons) == 1 and match in reasons[0], reasons


def test_config_reasons_need_flydsl(flydsl, clean_env):
    # Without FlyDSL the spec still imports, and each TP size's path refuses. A
    # child process: this one has imported FlyDSL, and None in sys.modules is a
    # module that neither imports nor is found
    code = (
        "import json, sys\n"
        "sys.modules['flydsl'] = None\n"
        "from qwen38_flash_next_decode_fakes import hf_config, server_args\n"
        "from sgl_amd_mono.qwen38_flash_next.spec import SPEC\n"
        "print(json.dumps([SPEC.config_disable_reasons(server_args(**a), hf_config())"
        " for a in json.loads(sys.argv[1])]))\n"
    )
    tp1 = {
        "tp_size": 1,
        "speculative_eagle_topk": None,
        "speculative_num_steps": None,
        "speculative_num_draft_tokens": None,
        "context_length": None,
    }
    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps([{}, tp1])],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    tp2_reasons, tp1_reasons = json.loads(proc.stdout.splitlines()[-1])
    assert tp2_reasons == ["needs FlyDSL"]
    # TP1's other reasons (its NEXTN arguments, ...) with FlyDSL found, then it
    assert tp1_reasons == [
        *S.SPEC.config_disable_reasons(server_args(**tp1), hf_config()),
        "needs FlyDSL",
    ]


def test_the_tp_size_selects_the_path(flydsl, clean_env):
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
    assert S._PATHS[2].model_disable_reasons is D.model_disable_reasons
    assert S._PATHS[2].build is D.build
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


def test_model_reasons_need_the_backend_s_gdn_state_read(tp2, monkeypatch):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )

    # SGLang before sgl-project/sglang#46980
    monkeypatch.setattr(MambaAttnBackendBase, "state_layer_cache", None, raising=False)
    assert S.SPEC.model_disable_reasons(fake_model(), hf_config(), {}) == [
        "GDN state read: sglang.srt.layers.attention.hybrid_linear_attn_backend."
        "MambaAttnBackendBase has no state_layer_cache (sgl-project/sglang#46980)"
    ]


@pytest.mark.parametrize("rip", [384, 512, 640])
def test_layout_reasons_clean_at_any_padding_of_128(rip):
    assert D.layout_reasons(fake_model(rip=rip)) == []


@pytest.mark.parametrize("rip", [352, 416])
def test_layout_reasons_padding_off_the_128_steps(rip):
    assert D.layout_reasons(fake_model(rip=rip)) == [
        f"layer {i} experts: intermediate padded to {rip}: 320 in steps of 128"
        for i in range(48)
    ]


def test_layout_reasons_mixed_padding():
    model = fake_model(rip=512)
    model.model.layers[5].mlp = fake_model(rip=384).model.layers[5].mlp
    assert D.layout_reasons(model) == [
        "the layers' experts are padded differently: [384, 512]"
    ]


def test_layout_reasons_one_hc_epsilon():
    model = fake_model()
    model.model.layers[6].mlp_hyper_connection.hc_norm.variance_epsilon = 1e-5
    assert D.layout_reasons(model) == ["the HC norms' epsilons differ: [1e-05, 1e-06]"]


def test_layout_reasons_name_the_broken_tensor():
    model = fake_model(rip=512)
    layers = model.model.layers
    layers[0].attn_hyper_connection.block_inject_weight.weight = meta(10240, 4)
    layers[3].qkv_proj.weight = meta(6144, 2560)
    layers[4].linear_attn.in_proj_qkvz.weight = meta(4096, 2560)
    layers[5].linear_attn.in_proj_ba.weight = meta(2560, 48).t()
    layers[7].mlp.experts.num_fused_shared_experts = 0
    layers[8].mlp.experts.w2_weight = None
    layers[9].linear_attn.norm.activation = "swish"
    layers[10].linear_attn.dt_bias = meta(24)  # bf16: accepted
    layers[11].o_proj.weight = meta(2560, 1536, dtype=torch.uint8)
    layers[12].mlp.experts.w2_weight_scale = meta(513, 2560, 8, dtype=torch.uint8)
    layers[13].linear_attn.dt_bias = meta(48)
    layers[14].mlp.experts.w13_weight.is_shuffled = False
    layers[15].indexer.index_qk_proj.bias = meta(640)
    layers[16].linear_attn.conv1d.bias = meta(5120)
    layers[17].linear_attn.in_proj_qkvz.output_sizes = [6144, 6144, 2048, 2048]
    layers[18].linear_attn.activation = "gelu"
    layers[19].attn_output_gate = False
    layers[23].is_qsa = False
    assert D.layout_reasons(model) == [
        "layer 0 attn_hyper_connection inject: (10240, 4) torch.bfloat16, expected "
        "(4, 10240) torch.bfloat16",
        "layer 3 qkv_proj: (6144, 2560) torch.bfloat16, expected (6656, 2560) "
        "torch.bfloat16",
        "layer 4 in_proj_qkvz: (4096, 2560) torch.bfloat16, expected (8192, 2560) "
        "torch.bfloat16",
        "layer 5 in_proj_ba: not contiguous",
        "layer 7: the kernels read the shared expert as expert 512",
        "layer 8 experts: missing w13_weight / w2_weight",
        "layer 9 gated norm activation 'swish', the launch gates with sigmoid",
        "layer 11 o_proj: (2560, 1536) torch.uint8, expected (2560, 3072) "
        "torch.bfloat16",
        "layer 12 w2 scales: (513, 2560, 8) torch.uint8, expected 21012480 dense e8m0",
        "layer 13 dt_bias: (48,) torch.bfloat16, expected (24,) float",
        "layer 14 w13: not in AITER's shuffled layout",
        "layer 15 index_qk_proj has a bias",
        "layer 16 conv1d has a bias",
        "layer 17 in_proj_qkvz output_sizes [6144, 6144, 2048, 2048], expected "
        "[q, k, v, z]",
        "layer 18 conv activation 'gelu', the launch applies SiLU",
        "layer 19: the qsa_back launch applies the output gate",
        "layer 23: the launches feed the QSA indexer",
    ]


def test_layout_reasons_layer_mix():
    model = fake_model(rip=384)
    model.model.layers[3] = model.model.layers[0]
    assert D.layout_reasons(model) == ["37 GDN and 11 QSA layers, expected 36 and 12"]


def test_model_reasons_refuse_the_padding_off_the_steps(tp2):
    assert len(S.SPEC.model_disable_reasons(fake_model(rip=352), hf_config(), {})) == 48


def _pool(slots=9, pitch=None, layer=1):
    """SGLang's conv pool [layers, slots, CONV_DIM, 3] and a layer's view."""
    pitch = pitch or 5120 * 3
    flat = torch.zeros(2 * slots * pitch + 8, dtype=torch.bfloat16)
    pool = flat[: 2 * slots * pitch].view(2, slots, pitch)[..., : 5120 * 3]
    return pool.unflatten(-1, (5120, 3))[layer]


def test_gdn_state_reasons_take_sglang_s_pool():
    conv = _pool()
    ssm = torch.zeros(9, 24, 128, 128)
    assert conv.stride() == (15360, 3, 1)
    assert D.gdn_state_reasons(conv, ssm) == []
    # a slot pitch that is a multiple of 4 elements past the slot's rows
    assert D.gdn_state_reasons(_pool(pitch=15364), ssm) == []


@pytest.mark.parametrize(
    "conv, match",
    [
        (lambda: torch.zeros(9, 3, 5120, dtype=torch.bfloat16), "conv state (9, 3, 5120)"),
        (lambda: torch.zeros(9, 5120, 3), "conv state (9, 5120, 3) torch.float32"),
        (lambda: torch.zeros(9, 3, 5120, dtype=torch.bfloat16).transpose(1, 2), "conv state strides (15360, 1, 5120)"),
        # layer 0: the base aligned, only the pitch off
        (lambda: _pool(pitch=15362, layer=0), "conv state strides (15362, 3, 1)"),
        (lambda: torch.zeros(9 * 15360 + 1, dtype=torch.bfloat16)[1:].view(9, 5120, 3), "8-byte aligned"),
    ],
)  # fmt: skip
def test_gdn_state_reasons_refuse_a_conv_state(conv, match):
    reasons = D.gdn_state_reasons(conv(), torch.zeros(9, 24, 128, 128))
    assert len(reasons) == 1 and match in reasons[0], reasons


def test_gdn_state_reasons_refuse_a_recurrent_state():
    conv = _pool()
    bf16 = D.gdn_state_reasons(conv, torch.zeros(9, 24, 128, 128, dtype=torch.bfloat16))
    assert len(bf16) == 1 and bf16[0].startswith("recurrent state (9, 24, 128, 128)")
    sparse = torch.zeros(3, 24, 128, 256)[..., :128]
    assert D.gdn_state_reasons(conv, sparse) == [
        f"recurrent state slot not dense: strides {sparse.stride()}"
    ]
