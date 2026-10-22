# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 MoE monokernels for amd/Qwen3.8-2.4T-A95B-Quark-MXFP4.

Routed experts are MXFP4. Attention and GDN projections stay bf16.
The spec and forward are in ``spec.py``. The gfx950 kernels are in ``gfx950``.
"""

# The Qwen3.8-2.4T text config the kernels are built for (gfx950/layout.py)
HF_CONFIG = {
    "hidden_size": 8192,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 128,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 2048,
    "shared_expert_intermediate_size": 2048,
    "hidden_act": "silu",
}
