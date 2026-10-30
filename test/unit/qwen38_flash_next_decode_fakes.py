"""A Qwen3.8-Flash-Next model as the TP2 decode unit tests read it: namespaces
holding meta tensors in the per-rank shapes SGLang loads at TP2, and the text config
and server args the spec checks."""

from types import SimpleNamespace as NS

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.decode.gfx950.weights import E2M1

# SGLang's ROCm build reads the GPU's properties when some of its modules import
# (memory_pool.py through ``is_fp8_fnuz``, qwen4_exp.py through
# ``is_gfx95_supported``); with CPU torch they import without one.
needs_gpu_on_rocm = pytest.mark.skipif(
    torch.version.hip is not None and not torch.cuda.is_available(),
    reason="SGLang's ROCm build needs a visible GPU to import its runtime modules",
)


def meta(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def scale_cols(k: int) -> int:
    """e8m0 columns of a K-wide MXFP4 row in AITER's shuffled layout."""
    return (k // 32 + 7) // 8 * 8


def decode_mxfp4(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Row-major packed fp4 [N, K / 2] (element 2 j in byte j's low nibble) and e8m0
    [N, K / 32] -> the values, element by element."""
    out = torch.empty(q.shape[0], 2 * q.shape[1])
    for r in range(q.shape[0]):
        for c in range(out.shape[1]):
            code = (int(q[r, c // 2]) >> (4 * (c % 2))) & 0xF
            mag = E2M1[code & 7] * 2.0 ** (int(s[r, c // 32]) - 127)
            out[r, c] = -mag if code & 8 else mag
    return out


def _hc():
    return NS(
        input_mix_weight_down=NS(weight=meta(320, 10240)),
        input_mix_weight_up=NS(weight=meta(10240, 320)),
        block_inject_weight=NS(weight=meta(4, 10240)),
        hc_norm=NS(weight=meta(10240), variance_epsilon=1e-6),
    )


def _moe(rip: int):
    return NS(
        gate=NS(weight=meta(512, 2560)),
        shared_expert_gate=NS(weight=meta(1, 2560)),
        experts=NS(
            w13_weight=meta(513, 2 * rip, 1280, dtype=torch.uint8),
            w2_weight=meta(513, 2560, rip // 2, dtype=torch.uint8),
            w13_weight_scale=meta(513, 2 * rip, scale_cols(2560), dtype=torch.uint8),
            w2_weight_scale=meta(513, 2560, scale_cols(rip), dtype=torch.uint8),
            num_fused_shared_experts=1,
        ),
    )


def _gdn_layer(rip: int):
    return NS(
        linear_attn=NS(
            in_proj_qkvz=NS(
                weight=meta(8192, 2560), output_sizes=[2048, 2048, 6144, 6144]
            ),
            in_proj_ba=NS(weight=meta(48, 2560)),
            conv1d=NS(weight=meta(5120, 1, 4), bias=None),
            out_proj=NS(weight=meta(2560, 3072)),
            A_log=meta(24, dtype=torch.float32),
            dt_bias=meta(24, dtype=torch.float32),
            norm=NS(
                weight=meta(128),
                eps=1e-6,
                activation="sigmoid",
                norm_before_gate=True,
                group_size=None,
            ),
            activation="silu",
        ),
        attn_hyper_connection=_hc(),
        mlp_hyper_connection=_hc(),
        mlp=_moe(rip),
        ple=None,
    )


def _qsa_layer(rip: int):
    return NS(
        o_proj=NS(weight=meta(2560, 3072)),
        attn_output_gate=True,
        attn_hyper_connection=_hc(),
        mlp_hyper_connection=_hc(),
        mlp=_moe(rip),
        ple=None,
    )


def fake_model(rip: int = 512):
    """48 layers, QSA at i % 4 == 3, with the experts padded to ``rip``."""
    layers = [_qsa_layer(rip) if i % 4 == 3 else _gdn_layer(rip) for i in range(48)]
    layers[1].ple = NS()
    return NS(
        model=NS(layers=layers, hyper_connection_mixer=_hc()), is_mrope_enabled=True
    )


# Qwen3.8-Flash-Next's text config fields the TP2 path checks
TEXT_CONFIG = {
    "hidden_size": 2560,
    "hc_count": 4,
    "hc_lowrank": 320,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 24,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hidden_act": "silu",
    "norm_topk_prob": True,
}


def hf_config(quant="quark", **text):
    return NS(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=NS(**{**TEXT_CONFIG, **text}),
        quantization_config={"quant_method": quant},
    )


def server_args(**overrides):
    """A TP2 decode server's arguments the spec reads."""
    return NS(
        **{
            "tp_size": 2,
            "speculative_algorithm": None,
            "mamba_ssm_dtype": None,
            "linear_attn_backend": "triton",
            "linear_attn_decode_backend": None,
            **overrides,
        }
    )
