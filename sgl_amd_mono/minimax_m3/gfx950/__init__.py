# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 on gfx950: one launch a sparse MoE layer (``post_attn``, with K1 of
``pre_attn`` fused in front), its layouts and weight views, and the host
side that builds and launches it (``packed``)."""
