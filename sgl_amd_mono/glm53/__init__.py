# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 monokernels (RFC #43949): the spec and its SGLang glue in ``spec``, and
the gfx950 kernel and host runner in ``gfx950``.

The kernel comes from ROCm/aiter#6173 at 56d6390def (Apache-2.0, Copyright (c)
2025 FlyDSL Project Contributors); each file names its origin.
"""

# GLM-5.3's shapes, routing, rotary layout and indexer schedule, which the kernel
# builds in
HF_CONFIG = dict(
    model_type="glm_moe_dsa",
    hidden_size=6144,
    n_routed_experts=256,
    num_experts_per_tok=8,
    n_shared_experts=1,
    moe_intermediate_size=2048,
    routed_scaling_factor=2.5,
    scoring_func="sigmoid",
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    hidden_act="silu",
    swiglu_limit=None,
    rms_norm_eps=1e-5,
    attention_bias=False,
    num_attention_heads=64,
    q_lora_rank=2048,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    rope_interleave=True,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    indexer_rope_interleave=True,
    index_topk_freq=4,
    index_skip_topk_offset=3,
)
