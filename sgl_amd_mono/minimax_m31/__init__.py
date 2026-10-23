# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash decode FFN monokernel on gfx950: one persistent launch per sparse
layer from the unreduced attention o_proj partial to the reduced MoE output (attention
TP all-reduce and residual add, Gemma post-attention norm, FP32 router and top-4, the
shared and routed A16W4 experts, MoE TP all-reduce).

The kernel and the DeepSeek-V4.1 mono primitives it builds on (``vcommon``) come from
vLLM (Apache-2.0, Copyright contributors to the vLLM project) at
vllm-project/vllm@a50259bb325366e7dd7c24461e1eb48e9b9ed7cf:
``vllm/models/minimax_m3/amd/specialized_decode/mono`` and
``vllm/models/deepseek_v41/amd/mono/common``.
"""
