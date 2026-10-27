# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2 monokernels (TP4).

Projections are Quark per-channel FP8. Routed experts and the fused shared
expert are MXFP4. The family spec selects this package; ``spec.py`` is the
precision spec and ``gfx950`` holds the kernels.
"""

HF_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "hidden_act": "silu",
}
