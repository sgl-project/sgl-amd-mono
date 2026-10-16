# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B dense (Quark MXFP4) decode on gfx950: per layer, K1 runs the
gated-delta-net share and K2 the FFN share (``gfx950/dense_gdn.py``,
``gfx950/dense_ffn.py``); full-attention layers keep SGLang's attention."""

# The text config the kernels are compiled for
HF_CONFIG = dict(
    model_type="qwen3_5_text",
    hidden_size=5120,
    intermediate_size=17408,
    hidden_act="silu",
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
)
