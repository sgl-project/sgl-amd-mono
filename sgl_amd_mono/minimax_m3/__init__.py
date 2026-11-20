# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 monokernels (RFC #43949): the spec and its SGLang glue in ``spec``,
and the gfx950 kernels and host runner in ``gfx950``.

Adapted from the in-tree MiniMax-M3 mono decode kernels
(sgl-project/sglang#44319, gfx950, TP4), extended to TP2 and TP8; each file
names its origin.

One launch serves each sparse MoE layer, K1 fused in front of K4::

    K1 part   (ar, res) -> h, q; K / V / index cache insert; indexer scores
    K4 part   top-k, attention .. FFN, both all-reduces in-kernel -> (ar, h_mid)

``(ar, res)`` is the (all-reduced MLP output, residual) pair a decoder layer
hands its input norm, so a layer computes the same values either way. A launch
reads the previous layer's ``(ar, h_mid)`` while it writes its own, so both
alternate between two buffers by layer parity. The embedding, the leading dense
MLP layers and the final norm run the model's own modules.
"""

# MiniMax-M3's shapes, routing and activation, which the kernels build in. The
# head counts are the model's totals rather than a rank's share: they are checked
# against the checkpoint's config, which knows nothing about TP.
HF_CONFIG = dict(
    hidden_size=6144,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    num_local_experts=128,
    num_experts_per_tok=4,
    scoring_func="sigmoid",
    swiglu_alpha=1.702,  # aiter's swiglu_mul_batch bakes this in
)

# The HF architectures this package serves. Both are the same decoder stack; the
# conditional-generation one wraps it with a vision tower, whose layers run
# SGLang's own path.
ARCHITECTURES = (
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
)
