# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 monokernels (RFC #43949): the spec and its SGLang glue in ``spec``,
and the gfx950 kernels and host runner in ``gfx950``.

The kernels come from ROCm/aiter PR 6047 (MIT License, Copyright (C) 2026,
Advanced Micro Devices, Inc.); the top-k block selection is adapted from ROCm/ATOM
(https://github.com/ROCm/ATOM) at 454d0e3023 (MIT License); each file names its
origin.
"""

# MiniMax-M3's shapes and routing, which the kernels build in (the text config)
HF_CONFIG = dict(
    hidden_size=6144,
    num_local_experts=128,
    num_experts_per_tok=4,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    scoring_func="sigmoid",
)
SWIGLU_ALPHA = 1.702  # aiter's swiglu_mul_batch bakes this in
LAYER_LIMIT = 128  # the mailbox tags' layer slots (``gfx950/config.LAYER_SLOTS``)
