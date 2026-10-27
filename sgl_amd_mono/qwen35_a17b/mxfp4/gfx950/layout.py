# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2 at TP4.

Shapes are the post-load tensors on one rank, not the checkpoint tensors.
FP8 projections are Quark per-channel: stored as ``[K, N]`` with stride
``(1, K)`` (output row ``n`` is K contiguous E4M3 bytes) and an fp32 scale
per output channel. Routed experts plus the fused shared expert are MXFP4
``[gate; up]`` through ``shuffle_weight(layout=(16, 16))`` and ``e8m0_shuffle``.
"""

TP = 4
HIDDEN = 4096
BLOCKS = 256  # one resident CTA per CU

# GDN, per rank. Full model: 16 key heads, 64 value heads, head dim 128.
HD = 128
NK = 16 // TP  # 4
NV = 64 // TP  # 16
KEY = NK * HD  # 512
VAL = NV * HD  # 2048
QKVZ = 2 * KEY + 2 * VAL  # 5120: [q | k | v | z]
BA = 2 * NV  # 32: [b | a]
CONV = 2 * KEY + VAL  # 3072 conv channels: q | k | v
CONV_W = 4
CORE = VAL  # 2048: out_proj K, and the full-attention o_proj K

# MoE, per rank. Experts are TP-sharded on the intermediate, not EP.
E = 512
TOPK = 10
RI = 1024 // TP  # 256
SI = 1024 // TP  # 256
EPR = E // TP  # 128 router rows this rank owns; the gate weight is replicated

MAX_TOKENS = 8
MAX_U = MAX_TOKENS * TOPK

ROWS = 16
FP4_STEP = 128


def need(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"Qwen3.5 AttnFP8 mono: {what}")


def fp4_group_dwords(k):
    return ROWS * k // 8


def fp4_tile_dword(expert_base, rg, k, st, lane):
    return expert_base + rg * fp4_group_dwords(k) + st * 256 + lane * 4


def w13_base(e):
    """First dword of expert ``e``'s w13. The product is folded before ``e``
    (i32 in the kernel) multiplies it."""
    return e * (2 * RI * HIDDEN // 8)


def w13_group(gate_or_up, g):
    """``[gate; up]`` rows, not SiTU-interleaved. Silu MXFP4 uses this layout."""
    return gate_or_up * (RI // ROWS) + g


def w2_base(e):
    return e * (HIDDEN * RI // 8)


def scale_index(row, col, cols):
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, f = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def w13_scale_index(e, n, kb):
    return scale_index(e * 2 * RI + n, kb, HIDDEN // 32)


def w2_scale_index(e, n, kb):
    return scale_index(e * HIDDEN + n, kb, RI // 32)
