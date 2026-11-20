# SPDX-License-Identifier: Apache-2.0
"""The shapes the Muse Spark gfx1250 kernels are built for: one rank at TP8 / EP8."""

import os

HIDDEN = 8192
HEAD_DIM = 64
LOCAL_Q_HEADS = 16  # 128 / TP8
LOCAL_KV_HEADS = 2  # 16 / TP8
Q_K = LOCAL_Q_HEADS * HEAD_DIM  # 1024: the q, o_gate and attention-output widths
KV_K = LOCAL_KV_HEADS * HEAD_DIM  # 128
# the fused projection's rows: q | k | v | gate (QKVOGateParallelLinear's order)
Q_OFF = 0
K_OFF = Q_K
V_OFF = K_OFF + KV_K
G_OFF = V_OFF + KV_K  # 1280
HEAD_ROWS = G_OFF  # the rows the head tasks finish: q | k | v
QKVG_ROWS = G_OFF + Q_K  # 2304: q | k | v | o_gate rows of the fused projection
EXPERTS = 256
LOCAL_EXPERTS = 32  # 256 / EP8
TOP_K = 8
MOE_HIDDEN = 4096  # the pooled width between the pooler and the unpooler
INTER = 4096  # an expert's intermediate width (whole: EP8 keeps expert TP at 1)
MXFP4_GROUP = 32
WINDOW = 2047  # a sliding layer attends to sliding_window - 1 keys
PAGE = 16
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTE_EPS = 1e-15
ROUTER_MULT = 0.17677669529663687
ROUTE_SCALE = 1.0
QK_SCALE = 5.4730064863875  # applied to q after its per-head norm
SOFTMAX_SCALE = HEAD_DIM**-0.5
LOG2E = 1.4426950408889634
GATE_TEMPERATURE = 0.3
LAYERS = 62
LAYER_SLOTS = 128  # mailbox tags: step * LAYER_SLOTS + layer + 1
MAX_TOKENS = 64  # C8 x 8 draft tokens
# The widths the monokernels serve: rows = rows a request (1 for plain decode, the draft tokens for a
# speculative verify) x running requests. Default: every plain decode step of one to sixteen requests, a row a
# request and a kernel set a batch size. SGLang captures a decode graph a width and drops its own sizes below the
# widest, so a batch size without a width runs the next width's kernels with padded rows, and a padded row costs
# about what a real one does (the replay leaves its token id as it was: it routes to experts of its own).
# SGLANG_AMD_MONO_MUSE_WIDTHS picks them (fewer widths: a shorter first start). The draft's kernels take 16 rows
# at most (``draft.DRAFT_WIDTHS``).
WIDTHS = tuple(
    int(x)
    for x in os.environ.get(
        "SGLANG_AMD_MONO_MUSE_WIDTHS", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
    ).split(",")
)
assert all(w in (*range(1, 17), 32, 64) for w in WIDTHS) and WIDTHS, WIDTHS
