# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shapes, CTA roles and memory layout of the DSpark draft launch (adapted from vLLM's
``vllm/models/minimax_m3/amd/mono/drafter/layout.py``, Apache-2.0).

The MiniMax-M3.1 DSpark draft model (5 layers) at TP4 on gfx950: 256 CTAs of 512
threads, one CTA per CU. A draft step of up to 2 requests (7 query rows each) is one
launch. Roles are fixed per CTA for the whole launch:

- G, CTAs 0..191: the MXFP8 GEMVs of every layer (QKV 12 rows, o_proj 32,
  gate 16 + up 16, down 32 a CTA).
- L, CTAs 192..239: attention partitions and the merge tasks.
- N, CTAs 240..255: the row norms; a CTA keeps its row's residual in LDS.

Scratch regions are per layer and written once a launch; flag regions and the
symmetric (IPC) regions are shared by the layers and told apart by their tags
(``step * 8 + layer + 1``). The draft's K / V caches are SGLang's: bf16 [slots, 1,
128] a layer, a token's slot from the request-to-token table.
"""

HIDDEN = 6144
QS = 2048  # 16 q heads x 128 a rank
QKV_N = 2304  # q | k | v rows a rank
IR = 3072  # intermediate size a rank
LAYERS = 5
HEADS = 16
HD = 128
ROT = 64
WINDOW = 4096
BLOCK = 128  # keys a partition block (the pool's page)
KV_ROW = HD * 2  # bytes of a token's bf16 K (or V) row
QPR = 7  # query rows a request
MAX_ROWS = 16
MAX_REQS = 2
THREADS = 512
WAVES = 8
BLOCKS = 256

NG = 192
ATT0 = 192
NATT = 48
NORM0 = 240
NP = 34  # partitions: a request's window spans <= 34 blocks
MAXB = 34

QKV_R = 12
O_R = 32
GU_R = 16
D_R = 32

EPS = 1e-6
FP8_MAX = 448.0
TINY = 1.1754943508222875e-38

# an attention partial: m [128] f32, l [128] f32, o [112][128] f32
AP_STRIDE = 1024 + 112 * 128 * 4


def _pack(regions, start=0):
    offsets, off = {}, start
    for name, nbytes in regions:
        offsets[name] = off
        off = (off + nbytes + 255) // 256 * 256
    return offsets, off


L_OFF, LAYER_BYTES = _pack(
    [
        ("x8", 16 * 6144),
        ("xs", 16 * 192),
        ("qkv", 16 * 2304 * 2),
        ("ap", 2 * 34 * AP_STRIDE),  # request r's block b at slot 34 r + b
        ("a8", 16 * 2048),
        ("as", 16 * 64),
        ("h8", 16 * 6144),
        ("hs", 16 * 192),
        ("act", 16 * 3072 * 2),
    ]
)
F_OFF, SCRATCH_BYTES = _pack(
    [
        (name, count * 8)
        for name, count in (
            ("x8f", 16),
            ("qkvf", 192),
            ("apf", 64),
            ("attf", 64),
            ("h8f", 16),
            ("actf", 192),
        )
    ],
    LAYER_BYTES * LAYERS,
)

# symmetric buffer: o_proj / down partials, [4 src ranks][16 rows][3072 pairs]
ROW_PAIRS = 3072
RANK_STRIDE = 16 * 3072 * 8
Y_OP = 0
Y_DP = 4 * RANK_STRIDE
SYM_BYTES = 8 * RANK_STRIDE

R_X8, R_XS, R_QKV, R_AP = L_OFF["x8"], L_OFF["xs"], L_OFF["qkv"], L_OFF["ap"]
R_A8, R_AS, R_H8, R_HS = L_OFF["a8"], L_OFF["as"], L_OFF["h8"], L_OFF["hs"]
R_ACT = L_OFF["act"]
F_X8, F_QKV, F_AP = F_OFF["x8f"], F_OFF["qkvf"], F_OFF["apf"]
F_ATT, F_H8, F_ACT = F_OFF["attf"], F_OFF["h8f"], F_OFF["actf"]

# timeline builds: a CTA's 64-bit clock marks past the scratch, point 10 layer + j
# (G: 0 QKV out, 1 o_proj out, 2 gate / up out, 3 down out, 8 x8 seen; L: 4 QKV seen,
# 5 partial out, 6 merges out; N: 7 x8 out, 9 h8 out), then 50 h_out and 51 the start
TL_POINTS = 52
TL_BYTES = BLOCKS * TL_POINTS * 8

# per-layer pointer table: int64 [5][16]
T_IN_LN, T_POST_LN, T_Q_NORM, T_K_NORM, T_K_CACHE, T_V_CACHE, T_SLOTS = range(7)

# layer-packed GEMV weights: bytes a layer
QKV_W_B, QKV_S_B = 2304 * 6144, 2304 * 192
O_W_B, O_S_B = 6144 * 2048, 6144 * 64
GU_W_B, GU_S_B = 6144 * 6144, 6144 * 192
D_W_B, D_S_B = 6144 * 3072, 6144 * 96
