# SPDX-License-Identifier: Apache-2.0
"""K2 on gfx1250: one Muse Spark layer from the attention output to the experts'
input, for S = ``tokens`` rows; per token::

    o (bf16, the attention output), gate (K1's o_gate)
      -> per-head RMSNorm without weight (eps RMS_EPS) x sigmoid(gate), one bf16
         rounding (the fused norm-gate)
      -> o_proj GEMV (bf16 WMMA, fp32 accumulate, one rounding) -> the bf16
         partial rows pushed to every rank's ``attn`` region -> rank-ordered fp32
         sum, one rounding (the one-stage all-reduce's numerics)
      -> attention boundary (weightless post norm): nb = bf16(attn_out rstd) (eps
         POST_EPS); stream = alpha stream + beta nb in fp32, in place; x32 =
         stream rstd2 w_pre in fp32 (eps RMS_EPS), the router's input; x =
         bf16(x32), the FFN input
      -> router: the model's fp32 weight [HIDDEN][EXPERTS], read in place, times
         x32 in fp32 (each rank its own experts' logits over every K part, pushed
         to every rank), x ROUTER_MULT, sigmoid, + expert bias, top-8 in fp32's
         order (a tie to the lower expert), the picks' sigmoids / (sum +
         ROUTE_EPS) x ROUTE_SCALE -> ids (int32), scores (f32)
      -> pooler GEMV (bf16 WMMA, the K parts' fp32 partials summed, one rounding)
         -> pre-expert norm (eps RMS_EPS, weight) -> expert input (bf16 [S,
         MOE_HIDDEN]; with ``xq_pre`` also its MX FP8 blocks for K3)

The work is task lists over the BLOCKS CTAs (one a compute unit)::

    og      every CTA: every token's normed, gated o into LDS (the o GEMV's B rows)
    o       512 tasks (16 rows each), CTA b takes b, b + BLOCKS, ..: o_proj partial
            rows -> every rank's ``attn`` region
    a_part  a_parts(S) tasks a token on the last CTAs (from AP0): the all-reduce sum
            of the part's columns, x's inputs (``xbuf``: alpha stream in fp32,
            ``xbuf2``: the summed attention output in bf16) and the part's norm sums
            (``ss``, their flag); the stream's update after the CTA's pooler job
            (``resflag``)
    passes  the router jobs (this rank's experts, R32_Q a CTA) and the pooler (256
            row tiles) GEMVs, K split in P_PARTS eighths: CTA b works eighth b %
            P_PARTS; the eighth's BLOCKS / P_PARTS CTAs share its work (a router job
            on some, then a contiguous range of pooler tiles), the tiles in groups of
            G held in two register slots (a slot's next group issued right after
            its WMMAs) -> ``lgx`` [part][token][expert] on every rank, ``pool``
            [part][token][row]
    route   S tasks on CTAs ROUTE0 + t: top-8 -> ids / scores
    xn      S tasks on CTAs XN0 + t: the pre-expert norm -> expert input
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    WAVE_OFFSETS,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    permlane_swap,
    popcount,
    rsrc,
    uniform,
    wmma_bf16,
    wmma_row_sum,
    xshfl,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    PAIR_BYTES,
    THREADS,
    WAVES,
    cdiv,
    pair_layout,
)
from sgl_amd_mono.common.sync import preg
from sgl_amd_mono.muse_spark.gfx1250.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    TOP_K,
)
from sgl_amd_mono.muse_spark.gfx1250.device import (
    ATTN_CM,
    ATTN_SLOTS,
    FLAG_STRIDE,
    OG_ONCE,
    PUSH_SET,
    acquire_loads,
    async_copy16,
    l2_prefetch,
    lds_addr,
    lds_dma_16,
    lds_load16,
    poll_flags,
    rank_base,
    release_stores,
    stamp,
    wait_lds_dma,
    wave_rank,
    wstamp,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    GQA,
    TRACE_LAYERS,
    TRACE_SLOTS,
    scratch_bytes,
    token_tiles,
)
from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import (
    POOL_PARTS,
    XQW,
    XSW,
    lds_barrier,
)
from sgl_amd_mono.muse_spark.gfx1250.stages.mxfp4 import fp8_word, mx8_scale

# ---------------------------------------------------------------- K2 geometry
ROWS = 16  # weight rows a WMMA tile: the WMMA's M
TILE = 16  # tokens a WMMA B operand holds: a token tile
K_STEP = 32  # the bf16 WMMA's K
ACC = 8  # fp32 accumulator values a lane
RED_FLOATS = WAVES * WAVE * ACC  # one row tile's accumulators of every wave
O_TASKS = HIDDEN // ROWS  # 512 o_proj row groups
O_PER_CTA = cdiv(O_TASKS, BLOCKS)  # o tasks a CTA (the last ones past O_TASKS: dead)
O_K_PER_WAVE = Q_K // WAVES  # the o GEMV's K over the waves
O_STEPS = O_K_PER_WAVE // K_STEP
OG_THREADS = Q_K // 4  # 256: a token's og threads, 4 o elements each
P_PARTS = POOL_PARTS  # the router / pooler K split (eighths)
K_PER_PART = HIDDEN // P_PARTS  # 1024
P_STEPS = K_PER_PART // WAVES // K_STEP  # WMMA steps a wave a (row tile, token tile)
R_PARTS = P_PARTS
P_TILES = MOE_HIDDEN // ROWS  # 256 pooler row tiles a K part
NJ = BLOCKS // P_PARTS  # CTAs a K part
G = 2 * cdiv(EXPERTS // ROWS, NJ)  # row tiles a pass group
A_PARTS_MAX = 8  # CTAs a token's attention boundary at most
A_PAIRS_MAX = HIDDEN // (
    2 * THREADS
)  # 8: a thread's column pairs with the whole row on one CTA
A_TASKS_MAX = MAX_TOKENS  # boundary tasks at most
AP0 = BLOCKS - A_TASKS_MAX  # the boundary tasks' CTAs: the last A_TASKS_MAX
# SGLANG_AMD_MONO_RSAG=1: the attention all-reduce as a reduce-scatter + all-gather. An o task's partial rows go
# to the rank owning their columns (HIDDEN / npes a rank) only; the owner's boundary threads sum the ranks'
# partials (rank order, one bf16 rounding: the one-stage sums) and push the reduced columns to every rank's
# ``ared``; every other boundary thread polls those. A quarter of the one-stage push bytes, one hop more.
RSAG = os.environ.get("SGLANG_AMD_MONO_RSAG", "0")
assert RSAG in ("0", "1", "auto"), RSAG
# auto: from this width on. Off by default: a layer bench with uniform routing gains from 8 rows on, a served
# batch of 8 loses.
RSAG_MIN = 8


def rsag(tokens: int) -> bool:
    return RSAG == "1" or (RSAG == "auto" and tokens >= RSAG_MIN)


# With pshard the pooler's K-eighth partials stay on their rank (``pool``, device scope, fp32); each pred task sums
# its token's rows of this rank's slice over the eighths (part order, fp32) and pushes those fp32 sums to every
# rank's ``poolr``; the xn task reads the whole row from there and rounds it once, as the model does.
# the boundary CTAs' pred task before (1) or after (0) their stream update (K3 waits for the update's flags)
PRED_FIRST = os.environ.get("SGLANG_AMD_MONO_PRED_FIRST", "1") == "1"
XN0 = 0  # the xn tasks' CTAs (0 .. S)
# SGLANG_AMD_MONO_ROUTE_SPLIT (1 / 0; default auto: at <= ROUTE_SPLIT_MAX rows; the fp32 form): a route task's
# waves 0 .. R_PARTS poll a K part's partial logits each (4 pairs a lane a round trip) and hand them to wave 0
# through LDS (``rdump``, free once the router jobs are done). Wave 0 polling every part itself takes 32 loads a
# lane a round trip: the logits were seen that much later than they landed.
ROUTE_SPLIT = os.environ.get("SGLANG_AMD_MONO_ROUTE_SPLIT", "auto")
assert ROUTE_SPLIT in ("0", "1", "auto"), ROUTE_SPLIT
# auto: up to this width (with ffn.PLAN_OVL it gains at 2 and 4 rows served, nothing at 1 and loses at 8). The
# 32-spec poll takes a build to 256 VGPRs (no scratch at 8 rows; beside RSAG it spills).
ROUTE_SPLIT_MAX = 4


def route_split(tokens: int) -> bool:
    return ROUTE_SPLIT == "1" or (ROUTE_SPLIT == "auto" and tokens <= ROUTE_SPLIT_MAX)


# SGLANG_AMD_MONO_O_CHUNK: o tasks a reduction round at most (default 4; 6: a CTA's o tasks in one round, every
# tile's weights in before the first push)
O_CHUNK = int(os.environ.get("SGLANG_AMD_MONO_O_CHUNK", "4"))
RED_SETS = max(G, 4, O_CHUNK)  # LDS reduction row sets
# ``device.PUSH_SET`` ("o"): the o tasks' pushes go out from every wave of the CTA, a wave to one rank (its
# resource stays in SGPRs), 16 B a lane. A wave's remote stores leave one after the other, and its later
# loads wait behind them: the first form (a thread stores its pair to the 8 ranks in
# turn) queues 48 stores on each of two waves; this form queues 3 a wave. "pred": the same for the pooled slices.
PUSH_ALL = "o" in PUSH_SET
# the weight ring: every GEMV tile is 16 rows x 1024 K (o_proj: Q_K; router / pooler: a K
# eighth); wave w streams its 64-K slice of the tile's rows (128 B a row) into its own LDS
# ring by async copies (eight lanes a row: contiguous reads), RING tiles ahead of its WMMAs
RING = int(
    os.environ.get("SGLANG_AMD_MONO_K2_RING", "4")
)  # a wave's tiles in LDS at once (6 fits up to 16 rows)
SLICE_BYTES = K_PER_PART // WAVES * 2  # a wave's bytes of a tile row
CHUNKS = SLICE_BYTES // 16
RTILE_BYTES = ROWS * SLICE_BYTES  # a wave's share of a tile
COPIES = RTILE_BYTES // (WAVE * 16)  # async copies a lane a tile
RING_FLOATS = WAVES * RING * RTILE_BYTES // 4
assert Q_K == K_PER_PART and CHUNKS == 8 and COPIES * WAVE * 16 == RTILE_BYTES
# The router: no weight copies and the model's own numerics. It reads the model's fp32 weight ([HIDDEN][EXPERTS],
# a k row 1 KB) and x in fp32, and each rank computes only its own experts'
# logits (the rank's slice ``local_start`` .. + R32_LOCAL, 128 B of a k row), which it pushes to every rank's
# ``lgx`` [K part][token][expert]: an eighth of the multiply-adds a rank, the route task reads the eight owners'
# partials as it read the local ones. A K part's first R32_JOBS CTAs take R32_Q experts each (one 16 B load a k
# row); a thread holds its two k rows' weights in registers and multiplies them into every token's x32. x itself is
# made exactly as the model makes it (``stage_x32``: nb = bf16(v rstd), stream' = alpha st + beta nb, x32 = stream'
# rstd2 pre_w in fp32, one bf16 rounding for the pooler's operand) from the boundary's ``xbuf`` (alpha st, fp32)
# and ``xbuf2`` (v, bf16).
R32_JOBS = 8  # router CTAs a K part
R32_Q = 4  # experts a router CTA
# SGLANG_AMD_MONO_ROUTER_PURE=1 (default): a K part's router CTAs are, where the step is narrow enough, CTAs with no
# xn / route / boundary task. A wave's later loads wait behind its remote stores, so a CTA that pushes its logits
# and then polls (the xn rows, the logits, the pred task's partials) polls late, which delays the experts' start.
ROUTER_PURE = os.environ.get("SGLANG_AMD_MONO_ROUTER_PURE", "1") == "1"


def router_js(tokens: int) -> list:
    """The indices j (CTA part + P_PARTS j) of a K part's R32_JOBS router CTAs at ``tokens`` rows, job order."""
    first = list(range(R32_JOBS))
    if not ROUTER_PURE:
        return first
    # preference: no task at all, then the xn / route CTAs (their polls wait for a cross-GPU hop anyway), last the
    # boundary CTAs (their pred task polls this GPU's partials right away)
    poll_js = {b // P_PARTS for b in range(XN0, XN0 + 2 * tokens)}
    bnd_js = {b // P_PARTS for b in range(AP0, AP0 + a_parts(tokens) * tokens)}
    order = sorted(
        range(NJ), key=lambda j: (2 if j in bnd_js else (1 if j in poll_js else 0), j)
    )
    return sorted(order[:R32_JOBS])


def router_slot(c, j):
    """CTA index ``j``'s router job (0 .. R32_JOBS - 1) at this build's rows, or -1 (traced)."""
    q = fx.Int32(-1)
    for idx, jj in enumerate(router_js(c["S"])):
        q = (j == jj).select(fx.Int32(idx), q)
    return q


R32_LOCAL = R32_JOBS * R32_Q  # experts a rank's slice
RD_TOK = 8  # tokens ``rdump`` holds at once (every thread's R32_Q sums a token)
XS_PITCH = (
    Q_K // 2 + 4
)  # floats a token's staged x row (bf16 pairs) in ``xs``: a 16 B skew spreads the tokens' banks
assert (
    R32_JOBS <= NJ
    and EXPERTS % R32_LOCAL == 0
    and K_PER_PART == 2 * THREADS
    and R32_Q == 4
)
assert WAVE == 32 and BLOCKS % P_PARTS == 0 and NJ >= 1, (WAVE, BLOCKS)
assert (
    O_K_PER_WAVE % K_STEP == 0 and P_STEPS >= 1 and K_PER_PART % (WAVES * K_STEP) == 0
)
assert (
    EXPERTS == 8 * WAVE and TOP_K == 8
)  # a lane holds 8 routing keys; the picks on one wave
assert HIDDEN == 16 * THREADS and MOE_HIDDEN == 8 * THREADS and Q_K == 2 * THREADS
assert MAX_TOKENS % TILE == 0 and OG_THREADS <= THREADS and AP0 > 0


def route0(s: int) -> int:
    """The route tasks' first CTA (past the xn tasks, before the boundary CTAs)."""
    assert 2 * s <= AP0, f"K2 at {s} rows needs more than the {BLOCKS} resident CTAs"
    return XN0 + s


def a_parts(s: int) -> int:
    """CTAs a token's attention boundary at width ``s``: the largest power of two
    (at most A_PARTS_MAX) whose tasks fit the A_TASKS_MAX boundary CTAs."""
    p = A_PARTS_MAX
    while p > 1 and p * s > A_TASKS_MAX:
        p //= 2
    assert p * s <= A_TASKS_MAX and A_PAIRS_MAX % p == 0, (s, p)
    return p


def pool_split(j: int, p_tiles: int = P_TILES, tokens: int = MAX_TOKENS) -> tuple:
    """K part CTA j's (router jobs, first pooler tile, pooler tiles) of a pooler of
    ``p_tiles`` row tiles: a router job is worth about two row tiles, the pooler tiles
    spread so every CTA of the part works about as many row tiles."""
    nr = [(1 if i in router_js(tokens) else 0) for i in range(NJ)]
    total = p_tiles + 2 * sum(nr)
    cnt = [0] * NJ
    left = p_tiles
    for i in range(NJ):
        share = -(-(total - sum(2 * nr[k] + cnt[k] for k in range(i))) // (NJ - i))
        cnt[i] = max(0, min(left, share - 2 * nr[i]))
        left -= cnt[i]
    cnt[NJ - 1] += left
    starts = [sum(cnt[:i]) for i in range(NJ)]
    return nr[j], starts[j], cnt[j]


def k2_scratch_layout() -> dict:
    """K2's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS: the
    boundary parts' norm sums (the flags of their x halves), the router parts'
    partial logits (fp32), the pooled rows (bf16 pairs), x as plain bf16 words in
    two halves (``k2_xbuf``, ``k2_xbuf2``, bytes given as pairs), the next K1's first
    half (deferred), the expert input's MX FP8 blocks (``k2_xinq``: K3's staged-column
    layout, XQW code words then XSW scale words a token), a flag a boundary part once
    its stream columns are out, and the merged launch's hand-off flags."""
    return pair_layout(
        [
            ("k2_og", MAX_TOKENS * Q_K // 2),
            ("k2_xflag", MAX_TOKENS),
            ("k2_ss", 4 * MAX_TOKENS * A_PARTS_MAX),
            (
                "k2_pool",
                P_PARTS * MAX_TOKENS * MOE_HIDDEN,
            ),  # fp32 partials, a value a pair
            ("k2_xbuf", MAX_TOKENS * HIDDEN * 4 // PAIR_BYTES),
            ("k2_xbuf2", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_c1", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_xinq", MAX_TOKENS * (XQW + XSW) * 4 // PAIR_BYTES),
            ("k2_resflag", MAX_TOKENS * A_PARTS_MAX * FLAG_STRIDE),
            ("k2_rflag", MAX_TOKENS * FLAG_STRIDE),
            # the picks again, tagged, then the scores, tagged (K3's plan and fin tasks poll them)
            ("k2_idx", 2 * MAX_TOKENS * TOP_K),
            ("k2_xinflag", MAX_TOKENS * FLAG_STRIDE),
            ("k2_oflag", MAX_TOKENS * LOCAL_KV_HEADS * 4 * FLAG_STRIDE),
            ("k2_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric buffer (one layout for the layer: K3's): K2 pushes
    ``attn``, K3 ``ffn`` and ``unp``. The guard's peer flags follow at
    ``sym_bytes(npes)``."""
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import k3_sym_layout

    return k3_sym_layout(npes)


def sym_bytes(npes: int) -> int:
    """The peer buffer's data bytes: the guard flags' offset."""
    return scratch_bytes(sym_layout(npes))


def sym_init(peer_bytes, npes: int) -> None:
    """A fresh peer buffer: nothing to set on gfx1250 (no K-sharded regions)."""


def k2_xs_floats(tokens: int) -> int:
    """The ``xs`` LDS region: every token's og row (the o GEMV's B rows), then every token's x over the CTA's K
    eighth as bf16 pairs (``stage_x32``)."""
    return tokens * XS_PITCH


def k2_rdump_floats(tokens: int) -> int:
    """The ``rdump`` LDS region: every thread's R32_Q partial logits of up to RD_TOK tokens."""
    return min(tokens, RD_TOK) * THREADS * R32_Q


def attn_at(src, tok, col):
    """The dword (a value; a bf16 pair and its tag take two) of column ``col`` of (source rank ``src``, token
    ``tok``)'s partial o row in ``attn``: [source][token][column], or CTA-major (``device.ATTN_CM``:
    [source][o CTA][token][task slot][16 columns], o task k on CTA k % BLOCKS, slot k / BLOCKS)."""
    if not ATTN_CM:
        return (src * MAX_TOKENS + tok) * HIDDEN + col
    task = col // ROWS
    return (
        ((src * BLOCKS + task % BLOCKS) * MAX_TOKENS + tok) * (O_PER_CTA * ROWS)
        + (task // BLOCKS) * ROWS
        + col % ROWS
    )


def o_chunk(tokens: int) -> int:
    """o tasks a reduction round (their rows for every token fit ``misc``)."""
    return max(1, min(O_PER_CTA, RED_SETS, O_CHUNK, MAX_TOKENS // tokens))


def k2_lds_bytes(tokens: int) -> int:
    return 4 * (
        k2_xs_floats(tokens)
        + RED_SETS * RED_FLOATS
        + ROWS * MAX_TOKENS
        + EXPERTS
        + RING_FLOATS
        + k2_rdump_floats(tokens)
    )


def k2_lds(tokens: int):
    """K2's LDS for a step of ``tokens`` rows: the o GEMV's B rows, the reduction
    row sets, a task's output rows for every token / the tokens' scalars, a token's
    sigmoids."""

    @fx.struct
    class Lds:
        xs: fx.Array[fx.Float32, k2_xs_floats(tokens), 16]
        red: fx.Array[fx.Float32, RED_SETS * RED_FLOATS, 16]
        misc: fx.Array[fx.Float32, ROWS * MAX_TOKENS, 16]
        sig: fx.Array[fx.Float32, EXPERTS, 16]
        ring: fx.Array[fx.Float32, RING_FLOATS, 16]
        rdump: fx.Array[fx.Float32, k2_rdump_floats(tokens), 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def _cs(x, y):
    """A compare-exchange, descending."""
    return fx.max(x, y), fx.min(x, y)


def _sort8_bitonic(v):
    """A bitonic list of 8 keys sorted descending (three compare-exchange levels)."""
    v = list(v)
    for st in (4, 2, 1):
        for i in range(8):
            if i & st == 0:
                v[i], v[i + st] = _cs(v[i], v[i + st])
    return v


def _sort4(k):
    a, b = _cs(k[0], k[1])
    c, d = _cs(k[2], k[3])
    a, c = _cs(a, c)
    b, d = _cs(b, d)
    b, c = _cs(b, c)
    return [a, b, c, d]


def top8_sorted(keys):
    """The wave's 8 largest of its 8 keys a lane (distinct), descending, on every
    lane: each lane sorts its 8 (two sorted 4s, a bitonic merge), then five butterfly
    steps (xor 1 .. 16) each keep the top 8 of its list and its partner's (a bitonic
    merge of two descending lists: the same set on both lanes)."""
    assert len(keys) == 8 and TOP_K == 8
    lo4, hi4 = _sort4(keys[:4]), _sort4(keys[4:])
    top = _sort8_bitonic(lo4 + hi4[::-1])
    for off in (1, 2, 4, 8, 16):
        if off >= 16:
            pairs = [
                permlane_swap(off, v, v) for v in top
            ]  # (the lower lane's, the upper lane's)
            lo, hi = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            lo, hi = top, [xshfl(v, off) for v in top]
        top = _sort8_bitonic([fx.max(lo[i], hi[7 - i]) for i in range(8)])
    return top


# SGLANG_AMD_MONO_TOP8X (default 1; 0: ``top8_sorted``): the picks by ``top8_extract`` (about half the
# instructions, on the one wave the routing waits for; the same keys in the same order).
TOP8X = os.environ.get("SGLANG_AMD_MONO_TOP8X", "1") == "1"


def top8_extract(keys):
    """``top8_sorted``'s result by eight extractions: each lane sorts its 8 keys, then eight times the wave's
    largest head (a butterfly max: the same value on every lane) is taken and the lane that held it moves its
    list up (the keys are distinct, so one lane does). After p extractions a lane's first 8 - p keys are all
    that can still be picked."""
    assert len(keys) == 8 and TOP_K == 8
    lo4, hi4 = _sort4(keys[:4]), _sort4(keys[4:])
    s = _sort8_bitonic(lo4 + hi4[::-1])
    top = []
    for p in range(8):
        m = butterfly(s[0], WAVE_OFFSETS, fx.max)
        top.append(m)
        hit = s[0] == m
        s = [hit.select(s[i + 1], s[i]) for i in range(7 - p)]
    return top


def route_top8(keys):
    return top8_extract(keys) if TOP8X else top8_sorted(keys)


def route_bits(v):
    """Score ``v`` as a signed-orderable i32 (every bit of the fp32)."""
    b = v.bitcast(fx.Int32)
    return (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)


def route_key(bits, e):
    """``route_bits`` with the low 8 bits replaced by 255 - expert ``e``: a pick is one integer max over
    distinct keys. The keys order the scores' top 24 bits only; ``stage_route`` settles the 8th place's
    tie group on the low 8 (``route_tie_ids``)."""
    return (bits & fx.Int32(-256)) | (255 - e)


def route_tie_ids(top, keys, bits, e0):
    """The exact picks when experts past the 8th share the 8th key's top 24 score bits (the tie group: keys
    lo .. lo + 255, lo = top[7] with its low 8 bits cleared). The picks above the group stand; the group's
    places go to its members with the largest low 8 score bits (equal scores: the lower expert), found by
    a second pass of the sorting network over the group alone. Returns the 8 expert ids, in place order."""
    lo = top[7] & fx.Int32(-256)
    above = fx.Int32(0)  # picks above the tie group (the same on every lane)
    for p in range_constexpr(TOP_K):
        above = above + (top[p] >= lo + 256).select(fx.Int32(1), fx.Int32(0))
    sub = [
        ((keys[h] >= lo) & (keys[h] < lo + 256)).select(
            ((bits[h] & 255) << 8) | (255 - (e0 + h)), fx.Int32(-1)
        )
        for h in range(8)
    ]
    st = top8_sorted(sub)
    ids = []
    for p in range_constexpr(TOP_K):
        i = 255 - (top[p] & 255)
        for q in range_constexpr(p + 1):
            i = (above == p - q).select(
                255 - (st[q] & 255), i
            )  # place p is the group's q-th
        ids.append(i)
    return ids


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` over its threads, in wave order; ``slot`` picks the
    WAVES words of ``red``."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = butterfly(v, WAVE_OFFSETS)
    if lane == 0:
        fx.ptr_store(ws, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(red + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


@flyc.jit
def block_sum4(c, v0, v1, v2, v3):
    """Four CTA sums at once (one barrier): each wave's four sums to ``red`` slots 0
    .. 3, every thread adds the waves in wave order."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = [butterfly(v, WAVE_OFFSETS) for v in (v0, v1, v2, v3)]
    if lane == 0:
        for k in range_constexpr(4):
            fx.ptr_store(ws[k], red + (k * WAVES + wave))
    gpu.barrier()
    outs = []
    for k in range_constexpr(4):
        tot = fx.ptr_load(red + k * WAVES)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + (k * WAVES + w))
        outs.append(tot)
    return outs


def ld_bf(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) bf16 at element ``e`` (even) of a bf16 buffer, as
    f32; at most 4 dwords a load."""
    out = []
    for e0 in range(0, n, 8):
        m = min(8, n - e0)
        w = bo.buffer_load(
            rsrc(ptr), (e + e0) // 2, vec_width=m // 2, dtype=T.i32, cache_modifier=cm
        )
        if m == 2:
            lo, hi = bf2_f32(fx.Int32(w))
            out += [lo, hi]
        else:
            v = fx.Vector(w).bitcast(fx.BFloat16).to(fx.Float32)
            out += [fx.Float32(v[j]) for j in range(m)]
    return out


def ld_f32(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) f32 at element ``e`` of an f32 buffer; at most 4 a load."""
    out = []
    for e0 in range(0, n, 4):
        m = min(4, n - e0)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(ptr), e + e0, vec_width=m, dtype=T.f32, cache_modifier=cm
            )
        )
        out += [fx.Float32(v[j]) for j in range(m)]
    return out


def st_f32(ptr, e, vals, cm=0):
    """``vals`` (2, 4, 8 or 16 f32) at element ``e`` of an f32 buffer; at most 4 a store."""
    for e0 in range(0, len(vals), 4):
        bo.buffer_store(fx.Vector.from_elements(vals[e0 : e0 + 4], fx.Float32), rsrc(ptr), e + e0, cache_modifier=cm)  # fmt: skip


def st_bf(ptr, e, vals, cm=0):
    """``vals`` (4 or 8 f32) as bf16 at element ``e`` of a bf16 buffer."""
    for e0 in range(0, len(vals), 4):
        bo.buffer_store(
            fx.Vector.from_elements(vals[e0 : e0 + 4], fx.Float32).to(fx.BFloat16),
            rsrc(ptr),
            e + e0,
            cache_modifier=cm,
        )


def st_words(ptr, d, words, cm=0):
    """``words`` (1 .. 8 i32) at dword ``d`` of a buffer; at most 4 a store."""
    for d0 in range(0, len(words), 4):
        chunk = words[d0 : d0 + 4]
        if len(chunk) == 1:
            bo.buffer_store(chunk[0], rsrc(ptr), d + d0, cache_modifier=cm)
        else:
            bo.buffer_store(fx.Vector.from_elements(chunk, fx.Int32), rsrc(ptr), d + d0, cache_modifier=cm)  # fmt: skip


def _cat2(a, b):
    return fx.Vector(a).shuffle(fx.Vector(b), list(range(2 * len(fx.Vector(a)))))


def w_step(r_w, row, k_row, k, cm=CM_NT):
    """The 16 bf16 at row ``row``, k .. k + 16 of a row-major [.., k_row] bf16 weight:
    a WMMA operand's lane half (8 dwords, two 16 B loads)."""
    at = (row * k_row + k) // 2
    return _cat2(
        bo.buffer_load(r_w, at, vec_width=4, dtype=T.i32, cache_modifier=cm),
        bo.buffer_load(r_w, at + 4, vec_width=4, dtype=T.i32, cache_modifier=cm),
    )


# ---------------------------------------------------------------- og rows and o tasks
def ring_slot(c, it):
    """This wave's ring slot of tile ``it`` (LDS byte address)."""
    return (
        lds_addr(c["ring"])
        + c["wave"] * (RING * RTILE_BYTES)
        + (it % RING) * RTILE_BYTES
    )


def ring_copy(c, it, desc):
    """Tile ``it`` = ``desc`` (its row 0's address at the K range's start, row pitch in
    bytes) into this wave's ring slot: lane l copies 16 B chunk l % 8 of row 4 i + l / 8
    of the wave's slice, stored at chunk (l % 8) ^ (row % 8) (``ring_frag`` reads a
    chunk's 16 rows from distinct banks)."""
    lane, wave = c["lane"], c["wave"]
    addr0, pitch = desc
    slot = ring_slot(c, it)
    ch = lane % CHUNKS
    for i in range(COPIES):
        r = i * (WAVE // CHUNKS) + lane // CHUNKS
        g = fx.Int64(addr0) + fx.Int64(r * pitch + wave * SLICE_BYTES + ch * 16)
        async_copy16(g, slot + r * SLICE_BYTES + ((ch ^ (r % 8)) * 16))


def ring_frag(c, it, s):
    """Step ``s`` of tile ``it``'s WMMA A operand from the ring: row l % 16, K 32 s + 16 h
    .. + 16 of the wave's slice (lane half h): the K labelling of the B operands."""
    lane = c["lane"]
    m, h = lane % 16, lane // 16
    slot = ring_slot(c, it)
    c0 = 4 * s + 2 * h
    a = lds_load16(slot + m * SLICE_BYTES + ((c0 ^ (m % 8)) * 16))
    b = lds_load16(slot + m * SLICE_BYTES + (((c0 + 1) ^ (m % 8)) * 16))
    return _cat2(a, b)


def ring_start(c, descs):
    """The first RING tiles' copies (issued ahead of whatever the CTA waits on next)."""
    for it in range(min(RING, len(descs))):
        ring_copy(c, it, descs[it])
    rocdl.sched_barrier(0)


def ring_acc(c, descs, it, xbs):
    """Tile ``it``'s accumulators, a token tile each (``xbs[t][s]``: the B operands):
    wait for its copies, the WMMAs from the ring, then the slot's next tile's copies."""
    n = len(descs)
    issued = min(n, RING + it)
    wait_lds_dma((issued - it - 1) * COPIES)
    steps = len(xbs[0])
    frags = [ring_frag(c, it, s).bitcast(fx.BFloat16) for s in range(steps)]
    accs = []
    for t in range(len(xbs)):
        acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
        for s in range(steps):
            acc = wmma_bf16(frags[s], fx.Vector(xbs[t][s]).bitcast(fx.BFloat16), acc)
        accs.append(acc)
    if it + RING < n:
        rocdl.s_wait_dscnt(0)  # this tile's reads are done before the slot's next copy
        ring_copy(c, it + RING, descs[it + RING])
    return accs


# (the CTA-major ``attn`` region: the all-waves tagged push; o task k is on CTA k % BLOCKS, slot k / BLOCKS)
assert not ATTN_CM or (
    PUSH_ALL and RSAG == "0" and ATTN_SLOTS == O_PER_CTA and ROWS == 16
)


def o_descs(c):
    """This CTA's o tasks (bid + BLOCKS k; past the last: the last task's rows, computed,
    not published) as ring tiles."""
    w_o = c["args"]["w_o"]
    tasks = [c["bid"] + BLOCKS * k for k in range(O_PER_CTA)]
    descs = [
        (w_o + fx.Int64(fx.min(tk, O_TASKS - 1) * (ROWS * Q_K * 2)), Q_K * 2)
        for tk in tasks
    ]
    return tasks, descs


@flyc.jit
def stage_og_rows(c, o_flag=None, QL=8):
    """Every token's normed gated o into LDS, computed here from o and the gate (16
    heads of 64: per-head RMSNorm without weight x sigmoid(gate), one bf16 rounding;
    thread tid holds head (tid % 256) / 16, dims 4 (tid % 16) .. of token 2 round +
    tid / 256): the o tasks' B rows. ``o_flag`` (a merged launch): the requests' (KV
    head, row tile) flags first, then device-scope loads."""
    S, tid, xs = c["S"], c["tid"], c["xs"]
    a = c["args"]
    if const_expr(OG_ONCE and o_flag is not None):
        # the attention's merge wrote the og rows where o goes (``attention.merge_l2``): a token's row is 2 KB,
        # four waves' lanes of 16 B; four tokens a copy
        lane, wave = c["lane"], c["wave"]
        RT = (QL * GQA + 15) // 16
        poll_flags(c, [(o_flag, 0, (S // QL) * LOCAL_KV_HEADS * RT, FLAG_STRIDE)])
        for i in range_constexpr(cdiv(S, 4)):
            tok = fx.min(4 * i + wave // 4, S - 1)
            lds_dma_16(a["o"] + fx.Int64(tok * (Q_K * 2)), xs + (tok * (Q_K // 2) + (wave % 4) * (WAVE * 4)), ((wave % 4) * WAVE + lane) * 16, CM_DEV)  # fmt: skip
        wait_lds_dma(0)
        gpu.barrier()
        return
    TPR = THREADS // OG_THREADS  # tokens a round: 2
    e = ((tid % OG_THREADS) // 16) * HEAD_DIM + (tid % 16) * 4
    rounds = list(range(0, S, TPR))
    ts = [fx.min(t0 + tid // OG_THREADS, S - 1) for t0 in rounds]
    sg = [[sigmoid(g) for g in ld_bf(a["g"], t * Q_K + e, 4)] for t in ts]
    o_cm = 0  # o from memory the host wrote: the default cache policy
    if const_expr(o_flag is not None):
        o_cm = CM_DEV  # the attention's device-scope stores, earlier in this launch
        RT = (QL * GQA + 15) // 16  # the attention's row tiles a (request, KV head)
        poll_flags(c, [(o_flag, 0, (S // QL) * LOCAL_KV_HEADS * RT, FLAG_STRIDE)])
        acquire_loads()
    ovs = [ld_bf(a["o"], t * Q_K + e, 4, o_cm) for t in ts]
    for i in range_constexpr(len(rounds)):
        t, ov = ts[i], ovs[i]
        ss = fx.Float32(0.0)
        for j in range_constexpr(4):
            ss = ss + ov[j] * ov[j]
        ss = butterfly(ss, (8, 4, 2, 1))
        rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
        vals = [ov[j] * rstd * sg[i][j] for j in range(4)]
        if rounds[i] + tid // OG_THREADS < S:
            fx.ptr_store(bf16_pair(vals[0], vals[1]), xs + (t * (Q_K // 2) + e // 2))
            fx.ptr_store(
                bf16_pair(vals[2], vals[3]), xs + (t * (Q_K // 2) + e // 2 + 1)
            )
    gpu.barrier()


@flyc.jit
def o_push_all(c, j0, n):
    """The round's reduced o rows (``misc``: tasks j0 .. j0 + n of this CTA) to every rank's ``attn`` region from
    every wave: threads TPD p .. TPD (p + 1) store to rank p (itself too), store k of the round holding rows
    4 (k % 4) .. + 4 of (task k / (4 S), token (k / 4) % S) as two tagged pairs (16 B)."""
    S, misc = c["S"], c["misc"]
    peer_bases, my_rank, sy_attn = c["bases"], c["rank"], c["sy_attn"]
    QT = ROWS // 4  # 16 B stores a (task, token)
    V4 = fx.Vector.make_type(4, fx.Float32)
    dest, td, TPD = wave_rank(c, len(peer_bases))
    region = preg(rank_base(peer_bases, dest), sy_attn, "attn")
    n_quads = n * S * QT
    if const_expr(ATTN_CM):
        # CTA-major: store k of the round is (token k / (4 n), slot j0 + (k / 4) % n, rows 4 (k % 4) ..): a wave's
        # 32 lanes are 512 B of the CTA's tokens' blocks in memory order (a round's slots of a token are whole
        # 128 B units). A dead slot's rows are stored too: they complete its unit, and nothing reads them.
        bid = c["bid"]
        for qq in range_constexpr(cdiv(n_quads, TPD)):
            k = td + qq * TPD
            tok = k // (n * QT)
            jj = (k // QT) % n
            r4 = (k % QT) * 4
            if k < n_quads:
                vv = fx.ptr_load(misc + ((jj * S + tok) * ROWS + r4), result_type=V4)
                c["put_bf"](
                    region,
                    ((my_rank * BLOCKS + bid) * MAX_TOKENS + tok) * (O_PER_CTA * ROWS)
                    + (j0 + jj) * ROWS
                    + r4,
                    [fx.Float32(vv[e]) for e in range(4)],
                )
        return
    for qq in range_constexpr(cdiv(n_quads, TPD)):
        k = td + qq * TPD
        jj = k // (S * QT)
        tok = (k // QT) % S
        r4 = (k % QT) * 4
        otask = c["bid"] + BLOCKS * (j0 + jj)
        if (k < n_quads) & (otask < O_TASKS):
            vv = fx.ptr_load(misc + ((jj * S + tok) * ROWS + r4), result_type=V4)
            c["put_bf"](
                region,
                (my_rank * MAX_TOKENS + tok) * HIDDEN + otask * ROWS + r4,
                [fx.Float32(vv[e]) for e in range(4)],
            )


@flyc.jit
def o_push_first(c, tasks, j0, n):
    """The first push form (``PUSH_ALL`` off, and the reduce-scatter half): pair i of a task (token i / 8, rows
    2 (i % 8), + 1) from thread i, stored to the ranks in turn (itself too)."""
    S, tid, misc, put_bf = c["S"], c["tid"], c["misc"], c["put_bf"]
    bases, rank, sy_attn = c["bases"], c["rank"], c["sy_attn"]
    n_pairs = (ROWS // 2) * S
    for j in range_constexpr(n):
        task = tasks[j0 + j]
        for q in range_constexpr(-(-n_pairs // THREADS)):
            i = tid + q * THREADS
            if (i < n_pairs) & (task < O_TASKS):
                tk = i // (ROWS // 2)
                rr = (i % (ROWS // 2)) * 2
                v0 = fx.ptr_load(misc + ((j * S + tk) * ROWS + rr))
                v1 = fx.ptr_load(misc + ((j * S + tk) * ROWS + rr + 1))
                if const_expr(rsag(S) and len(bases) > 1):
                    # to the rank owning these columns only (the reduce-scatter half)
                    own_p = (task * ROWS) // (HIDDEN // len(bases))
                    b = bases[0]
                    for p in range_constexpr(1, len(bases)):
                        b = (own_p == p).select(bases[p], b)
                    put_bf(preg(b, sy_attn, "attn"), (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr, [v0, v1])  # fmt: skip
                else:
                    for p in range_constexpr(len(bases)):
                        put_bf(preg(bases[p], sy_attn, "attn"), (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr, [v0, v1])  # fmt: skip


@flyc.jit
def stage_o(c, tasks, descs, j0, n, rnd=9):
    """o tasks ``tasks[j0 .. j0 + n]`` (ring tiles ``descs``, started): rows 16 task ..
    of o_proj for every token tile (B from the og rows in LDS), the waves'
    accumulators summed in wave order, one bf16 rounding, the partial rows pushed to
    every rank's ``attn`` region (a task past the last: nothing)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    xs, red, misc, put_bf = c["xs"], c["red"], c["misc"], c["put_bf"]
    bases, rank, sy_attn = c["bases"], c["rank"], c["sy_attn"]
    l16, h = lane % 16, lane // 16
    V4F = fx.Vector.make_type(4, fx.Float32)
    tiles = token_tiles(S)

    def x_at(t, s):
        tok = fx.min(t * TILE + l16, S - 1)
        at = xs + (tok * (Q_K // 2) + (wave * O_K_PER_WAVE + s * K_STEP + 16 * h) // 2)
        return _cat2(
            fx.ptr_load(at, result_type=V4F), fx.ptr_load(at + 4, result_type=V4F)
        ).bitcast(fx.Int32)

    xbs = [[x_at(t, s) for s in range(O_STEPS)] for t in range(tiles)]
    accs = [ring_acc(c, descs, j0 + j, xbs) for j in range(n)]
    if const_expr(rnd < 2):
        stamp(
            c, 24 + 4 * rnd
        )  # round rnd: the tiles' WMMAs issued (their weights landed in the ring)
        wstamp(c, 2 * rnd)  # ... on every wave
    for t in range_constexpr(tiles):
        for j in range_constexpr(n):
            fx.ptr_store(
                accs[j][t], red + (j * RED_FLOATS + (wave * WAVE + lane) * ACC)
            )
        lds_barrier()
        if tid < ROWS * TILE:
            r = tid % ROWS
            tk = t * TILE + tid // ROWS
            if tk < S:
                for j in range_constexpr(n):
                    fx.ptr_store(
                        bf16_round(wmma_row_sum(red + j * RED_FLOATS, r, tid // ROWS)),
                        misc + ((j * S + tk) * ROWS + r),
                    )
        lds_barrier()
    if const_expr(rnd < 2):
        stamp(
            c, 25 + 4 * rnd
        )  # round rnd: the rows reduced over the waves (in ``misc``)
    if const_expr(PUSH_ALL and not (rsag(S) and len(bases) > 1)):
        o_push_all(c, j0, n)
    else:
        o_push_first(c, tasks, j0, n)
    if const_expr(rnd < 2):
        stamp(c, 26 + 4 * rnd)  # round rnd: the pushes issued
        wstamp(c, 2 * rnd + 1)  # ... on every wave
    lds_barrier()  # ``misc`` is reused by the CTA's next tasks


@flyc.jit
def stage_o_all(c, o_flag=None, QL=8):
    """This CTA's o tasks (bid + BLOCKS k): the ring's first tiles copied first, the og
    rows staged meanwhile, then the tasks in rounds of ``o_chunk``."""
    S = c["S"]
    tasks, descs = o_descs(c)
    ring_start(c, descs)
    stage_og_rows(c, o_flag, QL)
    stamp(c, 1)  # the og rows staged (the attention's o of every head arrived)
    ch = o_chunk(S)
    for k0 in range_constexpr(0, O_PER_CTA, ch):
        stage_o(c, tasks, descs, k0, min(ch, O_PER_CTA - k0), k0 // ch)
        if const_expr(k0 // ch < 2):
            stamp(c, 16 + k0 // ch)  # o round k0 / ch done


# ---------------------------------------------------------------- boundary tasks
@flyc.jit
def stage_a_part(c, t, part):
    """Part ``part`` of token ``t``'s attention boundary (columns HIDDEN / A_PARTS
    part .., 2 PAIRS a thread): the ranks' partial o rows summed in rank order, one
    bf16 rounding; then the pooler / router operands x = rstd2 (b1 + rstd b2) as its
    two halves b1 = bf16(pre_w alpha st), b2 = bf16(pre_w beta v) (plain bf16 words)
    and the part's four norm sums (``ss``, after the halves: the consumers' flag).
    The stream's update is ``stage_a_part_res``, after the CTA's pooler job."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, xbuf, xbuf2 = c["own_attn"], c["ss"], c["xbuf"], c["xbuf2"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS  # columns a thread
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    al = ld_f32(a["alpha"], e0, NC)
    be = ld_f32(a["beta"], e0, NC)
    st = ld_f32(a["res"], t * HIDDEN + e0, NC)
    pb = t * (HIDDEN // 2) + e0 // 2
    PW = 2 if PAIRS % 2 == 0 else 1  # pairs a spec
    # (RSAG) this thread's columns lie in one rank's block (whole waves do): that rank's threads reduce the
    # ranks' partials, every other thread polls the reduced columns that rank pushed (W copies of one spec,
    # so both kinds poll the same number of words)
    mine = (e0 // (HIDDEN // W)) == c["rank"]
    if const_expr(rsag(S) and W > 1):
        src_reg = preg(mine.select(own.value, c["own_ared"].value), 0, "attn")
        ws = poll(
            [(src_reg, mine.select(src * (MAX_TOKENS * HIDDEN // 2) + pb + PW * q, pb + PW * q), PW) for q in range(PAIRS // PW) for src in range(W)],
            batch=max(8, (PAIRS // PW) * W),
        )  # fmt: skip
    else:
        if const_expr(ATTN_CM):
            specs = [
                (own, attn_at(src, t, e0 + 2 * PW * q) // 2, PW)
                for q in range(PAIRS // PW)
                for src in range(W)
            ]
        else:
            specs = [
                (own, src * (MAX_TOKENS * HIDDEN // 2) + pb + PW * q, PW)
                for q in range(PAIRS // PW)
                for src in range(W)
            ]
        ws = poll(specs, batch=max(8, (PAIRS // PW) * W))
    stamp(c, 8)  # every rank's partial o rows arrived (thread 0's columns)
    v = []
    for j in range_constexpr(PAIRS):
        acc0, acc1 = bf2_f32(ws[(j // PW) * W][j % PW])
        f0, f1 = acc0, acc1
        for src in range_constexpr(1, W):
            lo, hi = bf2_f32(ws[(j // PW) * W + src][j % PW])
            acc0 = acc0 + lo
            acc1 = acc1 + hi
        if const_expr(rsag(S) and W > 1):
            # the owner's sum, or the owner's already rounded value (bf16_round leaves it as is)
            acc0 = mine.select(acc0, f0)
            acc1 = mine.select(acc1, f1)
        v += [bf16_round(acc0), bf16_round(acc1)]
    if const_expr(rsag(S) and W > 1):
        if mine:
            for p in range_constexpr(W):
                c["put_bf"](
                    c["peer_ared"][p], t * HIDDEN + e0, v
                )  # the all-gather half
    # ``stage_x32``'s inputs: alpha st in fp32, and v as it is (bf16)
    st_f32(xbuf, t * HIDDEN + e0, [al[k] * st[k] for k in range(NC)], CM_DEV)
    st_words(xbuf2, (t * HIDDEN + e0) // 2, [bf16_pair(v[2 * k], v[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)], CM_DEV)  # fmt: skip
    ssp, A_, B_, C_ = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
    for k in range_constexpr(NC):
        ssp = ssp + v[k] * v[k]
        A_ = A_ + (al[k] * st[k]) * (al[k] * st[k])
        B_ = B_ + al[k] * be[k] * st[k] * v[k]
        C_ = C_ + (be[k] * v[k]) * (be[k] * v[k])
    ssp, A_, B_, C_ = block_sum4(c, ssp, A_, B_, C_)
    release_stores()  # this wave's halves out ...
    gpu.barrier()  # ... every wave's, before the sums that flag them
    if tid < 4:
        put(ss_mb, (t * A_PARTS_MAX + part) * 4 + tid, (tid == 0).select(ssp, (tid == 1).select(A_, (tid == 2).select(B_, C_))))  # fmt: skip
    return v


@flyc.jit
def stage_a_part_res(c, t, part, v):
    """The rest of boundary part ``part`` of token ``t``, after the CTA's pooler job:
    ``v`` (this part's all-reduced o columns, kept from ``stage_a_part``: the peers'
    pushes are not read again, so a rank already in its next step may overwrite them), the
    parts' sums of squares (``ss``), the post norm without weight, the stream updated
    in place (fp32), the next K1's first half (deferred K1), the (token, part) flag
    (``resflag``)."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, resflag = c["own_attn"], c["ss"], c["resflag"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    al = ld_f32(a["alpha"], e0, NC)
    be = ld_f32(a["beta"], e0, NC)
    st = ld_f32(a["res"], t * HIDDEN + e0, NC)
    if const_expr(c["defer_k1"]):
        npw1 = ld_bf(a["npw"], e0, NC)
        al2 = ld_f32(a["alpha2"], e0, NC)
    if const_expr(A_PARTS > 1):
        got = poll(
            [(ss_mb, (t * A_PARTS_MAX + q) * 4, 2) for q in range(A_PARTS)],
            batch=A_PARTS,
        )
        stamp(c, 9)  # the parts' norm sums arrived
        tot = fx.Float32(0.0)
        for q in range_constexpr(A_PARTS):
            tot = tot + got[q][0].bitcast(fx.Float32)
    else:
        tot = fx.Float32(0.0)
        for k in range_constexpr(NC):
            tot = tot + v[k] * v[k]
        tot = block_sum4(c, tot, tot, tot, tot)[0]
    rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
    n = [bf16_round(v[k] * rstd) for k in range(NC)]
    new = [al[k] * st[k] + be[k] * n[k] for k in range(NC)]
    st_f32(a["res"], t * HIDDEN + e0, new, CM_DEV)
    if const_expr(c["defer_k1"]):
        c1w = [bf16_pair(npw1[2 * k] * al2[2 * k] * new[2 * k], npw1[2 * k + 1] * al2[2 * k + 1] * new[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
        st_words(c["c1"], (t * HIDDEN + e0) // 2, c1w, CM_DEV)
    release_stores()
    gpu.barrier()
    if tid == 0:
        put(
            resflag, (t * A_PARTS + part) * FLAG_STRIDE, fx.Int32(1)
        )  # the stream's part (and c1) is out


# ---------------------------------------------------------------- router and pooler GEMVs
SCAL0 = 0  # misc floats: a token's (rstd, rstd2) at 2 t
SUMS0 = 2 * MAX_TOKENS  # then the (token, part) norm sums, 4 each


@flyc.jit
def wait_x(c):
    """Every (token, part)'s four norm sums (stored after its x halves: their flag;
    one round, two words a thread), then each token's rstd (the post norm) and rstd2
    (the pre-FFN norm) into LDS (``misc``) for ``stage_x32``."""
    S, tid, ss_mb, misc, poll = c["S"], c["tid"], c["ss"], c["misc"], c["poll"]
    A_PARTS = a_parts(S)
    NP = S * A_PARTS
    assert 2 * NP <= THREADS and SUMS0 + 4 * NP <= ROWS * MAX_TOKENS
    if tid < 2 * NP:
        tq = tid // 2
        got = poll([(ss_mb, ((tq // A_PARTS) * A_PARTS_MAX + tq % A_PARTS) * 4 + 2 * (tid % 2), 2)])  # fmt: skip
        fx.ptr_store(got[0][0].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid))
        fx.ptr_store(got[0][1].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid + 1))
    gpu.barrier()
    acquire_loads()
    if tid < S:
        tot, At, Bt, Ct = (
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
        )
        for q in range_constexpr(A_PARTS):
            b = misc + (SUMS0 + (tid * A_PARTS + q) * 4)
            tot = tot + fx.ptr_load(b)
            At = At + fx.ptr_load(b + 1)
            Bt = Bt + fx.ptr_load(b + 2)
            Ct = Ct + fx.ptr_load(b + 3)
        rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
        tot2 = At + fx.Float32(2.0) * rstd * Bt + rstd * rstd * Ct
        fx.ptr_store(rstd, misc + (SCAL0 + 2 * tid))
        fx.ptr_store(
            hw_rsq(tot2 / float(HIDDEN) + RMS_EPS), misc + (SCAL0 + 2 * tid + 1)
        )
    gpu.barrier()


def pool_tiles(c) -> int:
    """The pooler row tiles this rank works: all, or (``pshard``) its row slice's."""
    return P_TILES // c["W"] if c["pshard"] else P_TILES


def pool_max(c) -> int:
    """The most pooler tiles a CTA's job holds (``pool_split``), rounded to groups of G."""
    n = max(pool_split(j, pool_tiles(c), c["S"])[2] for j in range(NJ))
    return cdiv(n, G) * G


def pool_job(c):
    """This CTA's (K part, its index j among the part's CTAs, first pooler tile, pooler
    tiles) as traced values (``pool_split`` per j)."""
    bid = c["bid"]
    part = bid % P_PARTS
    j = bid // P_PARTS
    p0, cnt = fx.Int32(0), fx.Int32(0)
    for jj in range(NJ):
        _, s0, n0 = pool_split(jj, pool_tiles(c), c["S"])
        p0 = (j == jj).select(fx.Int32(s0), p0)
        cnt = (j == jj).select(fx.Int32(n0), cnt)
    return part, j, p0, cnt


def pass_descs(c):
    """This CTA's pooler job as ring tiles: its pooler tiles p0 .. p0 + cnt in groups of G (a
    tile past the job or the table: clamped, computed, not published)."""
    a = c["args"]
    part, j, p0, cnt = pool_job(c)
    k0 = part * (K_PER_PART * 2)
    descs = []
    n = pool_tiles(c)
    for i in range(pool_max(c)):
        tile = fx.min(p0 + i, n - 1)
        descs.append(
            (a["w_pool"] + fx.Int64(tile * (ROWS * HIDDEN * 2) + k0), HIDDEN * 2)
        )
    return (part, j, p0, cnt), descs


@flyc.jit
def stage_group(c, job, accs, g):
    """One pooler group of G row tiles' accumulators (``accs[k][t]``, k the tile, t the token
    tile) summed over the waves through ``red`` (a set a tile) and published: the K eighth's
    partial rows in fp32 to ``pool`` [part][token][row] (the model rounds the pooled row once,
    after the whole K sum: ``stage_pred`` / ``stage_xn``)."""
    S, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    pool = c["pool"]
    part, j, p0, cnt = job
    for ps in range_constexpr(token_tiles(S)):
        t0 = ps * TILE
        nt = min(TILE, S - t0)
        for k in range_constexpr(G):
            fx.ptr_store(
                accs[k][ps], red + (k * RED_FLOATS + (wave * WAVE + lane) * ACC)
            )
        lds_barrier()
        # thread (tile k, token tk, row pair rp)
        for rnd in range_constexpr(cdiv(G * TILE * (ROWS // 2), THREADS)):
            i = tid + rnd * THREADS
            k = i // (TILE * (ROWS // 2))
            tk = (i // (ROWS // 2)) % TILE
            rp = (i % (ROWS // 2)) * 2
            gi = g * G + k
            if (k < G) & (tk < nt) & (gi < cnt):
                rb = red + k * RED_FLOATS
                prow = (p0 + gi) * ROWS + rp
                vals = [wmma_row_sum(rb, rp, tk), wmma_row_sum(rb, rp + 1, tk)]
                c["put_words"](pool, (part * MAX_TOKENS + t0 + tk) * MOE_HIDDEN + prow, [vals[0].bitcast(fx.Int32), vals[1].bitcast(fx.Int32)])  # fmt: skip
        lds_barrier()  # ``red`` is reused by the next round


def local_start(c):
    """The rank's first expert: the launch's ``local_start`` argument (the model's), else rank R32_LOCAL (a K2
    launch on its own)."""
    ls = c["args"].get("local_start")
    return fx.Int32(c["rank"]) * R32_LOCAL if ls is None else ls


def router_w(c, job, vr=0):
    """The fp32 router's weights this thread holds for its CTA's router job: its two k rows (the part's k ``tid``
    and ``tid`` + THREADS) x the CTA's R32_Q experts (the rank's slice ``local_start`` + R32_LOCAL vr .., CTA j's
    quad): two 16 B loads of the model's own [HIDDEN][EXPERTS] fp32 router, and the quad's first expert. A CTA
    past the part's router jobs loads the last job's (unused)."""
    a, tid = c["args"], c["tid"]
    part, j, _, _ = job
    e0 = (
        local_start(c) + R32_LOCAL * vr + R32_Q * fx.max(router_slot(c, j), fx.Int32(0))
    )
    k0 = part * K_PER_PART + tid
    return (
        ld_f32(a["w_router"], k0 * EXPERTS + e0, R32_Q),
        ld_f32(a["w_router"], (k0 + THREADS) * EXPERTS + e0, R32_Q),
        e0,
    )


def router_prefetch(c, job):
    """The lines ``router_w`` reads, into GL2 ahead of the norm sums' wait (no counter moves)."""
    a, tid = c["args"], c["tid"]
    part, j, _, _ = job
    e0 = local_start(c) + R32_Q * fx.max(router_slot(c, j), fx.Int32(0))
    k0 = part * K_PER_PART + tid
    for kk in (k0, k0 + THREADS):
        l2_prefetch(fx.Int64(a["w_router"]) + fx.Int64((kk * EXPERTS + e0) * 4))


@flyc.jit
def stage_x32(c, part):
    """Every token's x over K eighth ``part`` into LDS, made as the model makes it: nb = bf16(v rstd), stream' =
    alpha st + beta nb, x32 = stream' rstd2 pre_w in fp32, from the boundary's ``xbuf`` (alpha st, fp32) and
    ``xbuf2`` (v, bf16) and the token's scalars (``wait_x``). ``red`` [token][k] takes x32 (the router's operand),
    ``xs`` [token][k / 2] its bf16 pairs (the pooler's B rows, one rounding; a token's row XS_PITCH floats). A
    thread takes four k of a token a round."""
    S, tid, xs, red, misc = c["S"], c["tid"], c["xs"], c["red"], c["misc"]
    a = c["args"]
    QK = K_PER_PART // 4  # four-k units a token
    TPR = THREADS // QK  # tokens a round
    assert THREADS % QK == 0 and S * K_PER_PART <= RED_SETS * RED_FLOATS, S
    k4 = tid % QK
    k = part * K_PER_PART + 4 * k4
    be = ld_f32(a["beta"], k, 4)
    pw = ld_bf(a["pre_w"], k, 4)
    rounds = cdiv(S, TPR)
    ts = [fx.min(tid // QK + TPR * r, S - 1) for r in range(rounds)]
    s1 = [ld_f32(c["xbuf"], t * HIDDEN + k, 4, CM_DEV) for t in ts]
    vv = [ld_bf(c["xbuf2"], t * HIDDEN + k, 4, CM_DEV) for t in ts]
    for r in range_constexpr(rounds):
        t = ts[r]
        rstd = fx.ptr_load(misc + (SCAL0 + 2 * t))
        rstd2 = fx.ptr_load(misc + (SCAL0 + 2 * t + 1))
        x32 = [
            ((s1[r][q] + be[q] * bf16_round(vv[r][q] * rstd)) * rstd2) * pw[q]
            for q in range(4)
        ]
        if tid // QK + TPR * r < S:
            fx.ptr_store(
                fx.Vector.from_elements(x32, fx.Float32),
                red + (t * K_PER_PART + 4 * k4),
            )
            fx.ptr_store(
                fx.Vector.from_elements(
                    [bf16_pair(x32[0], x32[1]), bf16_pair(x32[2], x32[3])], fx.Float32
                ),
                xs + (t * XS_PITCH + 2 * k4),
            )
    gpu.barrier()


def x_ops32(c, part, t0):
    """Token tile t0 .. t0 + 16's x over the CTA's K eighth, this wave's steps, as the WMMA's B operands from
    ``stage_x32``'s bf16 rows in ``xs`` (lane l16's token, clamped; lane half h holds K 16 h ..)."""
    S, lane, wave = c["S"], c["lane"], c["wave"]
    tl = fx.min(t0 + lane % 16, S - 1)
    base = lds_addr(c["xs"]) + tl * (XS_PITCH * 4)
    out = []
    for kc in range_constexpr(P_STEPS):
        at = base + ((wave * P_STEPS + kc) * K_STEP + 16 * (lane // 16)) * 2
        out.append(_cat2(lds_load16(at), lds_load16(at + 16)))
    return out


@flyc.jit
def stage_router_sh(c, job, w):
    """This CTA's router job (``w`` = ``router_w``): R32_Q experts of the rank's slice over the part's K for
    every token. A thread's two k rows as fp32 multiply-adds against x32 (``red``), its R32_Q sums a token to
    ``rdump``; wave i then sums token i's THREADS partials (thread order, then the lanes by butterflies: one
    order whatever the rank), and the sums go to every rank's ``lgx`` [part][token][expert] as tagged pairs, a
    wave to one rank (``wave_rank``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, rdump, sig = c["red"], c["rdump"], c["sig"]
    part = job[0]
    w0, w1, e0 = w
    V4 = fx.Vector.make_type(4, fx.Float32)
    assert S * R32_Q <= EXPERTS and 2 * S <= THREADS // c["W"], S
    for b0 in range_constexpr(0, S, RD_TOK):
        nb = min(RD_TOK, S - b0)
        for i in range_constexpr(nb):
            x0 = fx.ptr_load(red + ((b0 + i) * K_PER_PART + tid))
            x1 = fx.ptr_load(red + ((b0 + i) * K_PER_PART + tid + THREADS))
            fx.ptr_store(
                fx.Vector.from_elements(
                    [w0[q] * x0 + w1[q] * x1 for q in range(R32_Q)], fx.Float32
                ),
                rdump + (i * THREADS + tid) * R32_Q,
            )
        lds_barrier()
        if wave < nb:
            sums = [fx.Float32(0.0) for _ in range(R32_Q)]
            for m in range_constexpr(WAVES):
                pv = fx.Vector(
                    fx.ptr_load(
                        rdump + (wave * THREADS + lane + WAVE * m) * R32_Q,
                        result_type=V4,
                    )
                )
                sums = [sums[q] + fx.Float32(pv[q]) for q in range(R32_Q)]
            tots = [butterfly(sv, WAVE_OFFSETS) for sv in sums]
            if lane == 0:
                fx.ptr_store(
                    fx.Vector.from_elements(tots, fx.Float32), sig + (b0 + wave) * R32_Q
                )
        lds_barrier()
    dest, td, _ = wave_rank(c, c["W"])
    if td < 2 * S:
        tok, pr = td // 2, td % 2
        v0 = fx.ptr_load(sig + (tok * R32_Q + 2 * pr))
        v1 = fx.ptr_load(sig + (tok * R32_Q + 2 * pr + 1))
        region = preg(rank_base(c["bases"], dest), c["sy"]["lgx"][0], "lgx")
        c["put_words"](region, (part * MAX_TOKENS + tok) * EXPERTS + e0 + 2 * pr, [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)])  # fmt: skip


@flyc.jit
def router_jobs(c, job, w):
    """The CTA's router job for the rank's slice (and, on a group of fewer ranks, for the slices the absent ranks
    would own: one job a slice, its weights loaded here)."""
    for vr in range_constexpr(EXPERTS // (c["W"] * R32_LOCAL)):
        stage_router_sh(c, job, w if vr == 0 else router_w(c, job, vr))


@flyc.jit
def stage_passes(c, first):
    """This CTA's router / pooler job (``first`` = ``passes_first``: its ring tiles, the first ones' copies in
    flight): x into LDS (``stage_x32``), the CTA's router job (the router CTAs of a K part: ``router_js``),
    then the pooler groups, G ring tiles a group."""
    S = c["S"]
    job, descs = first[0], first[1]
    part = job[0]
    stage_x32(c, part)
    if router_slot(c, job[1]) >= 0:
        router_jobs(c, job, first[2])
    stamp(c, 10)  # the router job done (thread 0's wave)
    stamp(c, 11)  # ... its logits pushed
    xbs = [x_ops32(c, part, ps * TILE) for ps in range(token_tiles(S))]
    for g in range_constexpr(len(descs) // G):
        accs = [ring_acc(c, descs, g * G + k, xbs) for k in range(G)]
        if const_expr(g < 2):
            stamp(c, 12 + 2 * g)  # pooler group g's WMMAs done (thread 0's wave)
        stage_group(c, job, accs, g)
        if const_expr(g < 2):
            stamp(c, 13 + 2 * g)  # ... reduced and published


def passes_first(c):
    """The router / pooler job's ring tiles, the first RING copies issued now; the router job's weight lines
    prefetched (and, after the norm sums' wait, loaded: ``router_w`` in ``run_k2``)."""
    job, descs = pass_descs(c)
    ring_start(c, descs)
    router_prefetch(c, job)
    return job, descs


# ---------------------------------------------------------------- route and xn tasks
@flyc.jit
def stage_route(c, t, route_flag=None):
    """route task: token ``t``'s top-8 on wave 0 (lane l holds experts 8 l .. 8 l +
    8): the parts' partial logits summed in part order, sigmoid(logit x
    ROUTER_MULT), + bias, a pick an integer max over ``route_key`` (a tie to the
    lower expert), the picks' sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE."""
    tid, lane, wave, sig = c["tid"], c["lane"], c["wave"], c["sig"]
    put, poll = c["put"], c["poll"]
    logits = c["own_lgx"]  # every rank's own experts' partial logits, pushed here
    a = c["args"]
    split = route_split(c["S"])
    V4 = fx.Vector.make_type(4, fx.Float32)
    if const_expr(split):
        rd = c["rdump"]
        assert R_PARTS <= WAVES and R_PARTS * WAVE * 8 <= k2_rdump_floats(c["S"]), (
            R_PARTS,
            c["S"],
        )
        bias_all = ld_f32(
            a["bias"], 8 * lane, 8
        )  # (every wave issues it, ahead of the polls; wave 0 uses it)
        if wave < R_PARTS:
            gw = poll(
                [
                    (logits, (wave * MAX_TOKENS + t) * EXPERTS + 8 * lane + 2 * q, 2)
                    for q in range(4)
                ],
                batch=4,
            )
            for q in range_constexpr(2):
                fx.ptr_store(
                    fx.Vector.from_elements(
                        [
                            gw[2 * q + e // 2][e % 2].bitcast(fx.Float32)
                            for e in range(4)
                        ],
                        fx.Float32,
                    ),
                    rd + ((wave * WAVE + lane) * 8 + 4 * q),
                )
        lds_barrier()
    if wave == 0:
        e0 = 8 * lane
        if const_expr(split):
            bias = bias_all
            lg = []  # [part][expert of the lane]
            for p in range_constexpr(R_PARTS):
                for q in range_constexpr(2):
                    pv = fx.Vector(
                        fx.ptr_load(
                            rd + ((p * WAVE + lane) * 8 + 4 * q), result_type=V4
                        )
                    )
                    lg += [fx.Float32(pv[e]) for e in range(4)]
        else:
            bias = ld_f32(a["bias"], e0, 8)
            got = poll(
                [(logits, (p * MAX_TOKENS + t) * EXPERTS + e0 + 2 * q, 2) for p in range(R_PARTS) for q in range(4)],
                batch=4 * R_PARTS,  # one round trip for the parts' logits
            )  # fmt: skip
            lg = [
                got[p * 4 + h // 2][h % 2].bitcast(fx.Float32)
                for p in range(R_PARTS)
                for h in range(8)
            ]
        stamp(c, 22)  # the parts' partial logits of the token arrived (route CTAs)
        keys, bits = [], []
        for h in range_constexpr(8):
            logit = lg[h]
            for p in range_constexpr(1, R_PARTS):
                logit = logit + lg[p * 8 + h]
            sc = sigmoid(logit * fx.Float32(ROUTER_MULT))
            fx.ptr_store(sc, sig + (e0 + h))
            bits.append(route_bits(sc + bias[h]))
            keys.append(route_key(bits[h], e0 + h))
        top = route_top8(keys)
        pid = fx.Int32(0)
        for p in range_constexpr(TOP_K):
            pid = (lane == p).select(255 - (top[p] & 255), pid)
        # the picks through LDS (``misc``, free on a route CTA by now): an expert past the 8th whose key is in the
        # 8th's tie group (the same top 24 score bits) may beat a pick of that group on the low 8 bits. Then, and
        # only then, the group's places are taken again on every bit (``route_tie_ids``): fp32's own order.
        lo7 = top[7] & fx.Int32(-256)
        tied = (keys[0] >= lo7) & (keys[0] < top[7])
        for h in range_constexpr(1, 8):
            tied = tied | ((keys[h] >= lo7) & (keys[h] < top[7]))
        if lane < TOP_K:
            fx.ptr_store(pid.bitcast(fx.Float32), c["misc"] + lane)
        if uniform(popcount(ballot(tied))) != 0:
            ids = route_tie_ids(top, keys, bits, e0)
            pid2 = fx.Int32(0)
            for p in range_constexpr(TOP_K):
                pid2 = (lane == p).select(ids[p], pid2)
            if lane < TOP_K:
                fx.ptr_store(pid2.bitcast(fx.Float32), c["misc"] + lane)
        if lane < TOP_K:
            pid = fx.ptr_load(c["misc"] + lane).bitcast(fx.Int32)
            if const_expr(route_flag is not None):
                put(
                    c["idx"], t * TOP_K + lane, pid
                )  # first: the experts' plan waits on it
                stamp(c, 31)  # the token's tagged picks stored (route CTAs)
            w = fx.ptr_load(sig + pid)
            tot = butterfly(w, (1, 2, 4))
            score = (
                w / (tot + fx.Float32(ROUTE_EPS)) * fx.Float32(ROUTE_SCALE)
            ).bitcast(fx.Int32)
            if const_expr(route_flag is not None):
                put(c["idx"], MAX_TOKENS * TOP_K + t * TOP_K + lane, score)
            bo.buffer_store(
                pid, rsrc(a["ids"]), t * TOP_K + lane, cache_modifier=CM_DEV
            )
            bo.buffer_store(
                score, rsrc(a["scores"]), t * TOP_K + lane, cache_modifier=CM_DEV
            )
    if const_expr(route_flag is not None):
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(route_flag, t * FLAG_STRIDE, fx.Int32(1))
    stamp(c, 23)  # the token's routing flag raised (route CTAs)


def _xn_pool(c, pool, t, e0):
    """The P_PARTS K-eighths' pooled rows e0 .. e0 + 8 of token ``t`` (fp32 partials, a value a pair), summed in
    part order (four rows a round of the eight parts)."""
    v = []
    for h in range(2):
        got = c["poll"](
            [(pool, (p * MAX_TOKENS + t) * MOE_HIDDEN + e0 + 4 * h + 2 * j, 2) for p in range(P_PARTS) for j in range(2)],
            batch=2 * P_PARTS,
        )  # fmt: skip
        for j in range(2):
            for q in range(2):
                sm = got[j][q].bitcast(fx.Float32)
                for p in range(1, P_PARTS):
                    sm = sm + got[p * 2 + j][q].bitcast(fx.Float32)
                v.append(sm)
    return v


def _bor(x, y):
    return x | y


def pool_pair(c, t, prow):
    """Rows ``prow``, + 1 of token ``t``'s pooled slice on this rank: the K eighths' fp32 partials (``pool``, a
    value a pair) summed in part order."""
    got = c["poll"]([(c["pool"], (pp * MAX_TOKENS + t) * MOE_HIDDEN + prow, 2) for pp in range(P_PARTS)], batch=P_PARTS)  # fmt: skip
    lo, hi = got[0][0].bitcast(fx.Float32), got[0][1].bitcast(fx.Float32)
    for pp in range(1, P_PARTS):
        lo = lo + got[pp][0].bitcast(fx.Float32)
        hi = hi + got[pp][1].bitcast(fx.Float32)
    return lo, hi


@flyc.jit
def pred_push_all(c, t, sp, RP):
    """``stage_pred`` from every wave: threads TPD p .. TPD (p + 1) store to rank p, the k-th of them rows
    sp RP + 2 k, + 1 of this rank's slice (each sums the K eighths itself: device-scope reads of the same pairs)."""
    W = c["W"]
    SL = MOE_HIDDEN // W
    dest, td, TPD = wave_rank(c, W)
    region = preg(rank_base([r.value for r in c["peer_poolr"]], dest), 0, "poolr")
    n_items = RP // 2
    for qq in range_constexpr(cdiv(n_items, TPD)):
        kk = td + qq * TPD
        if kk < n_items:
            prow = sp * RP + 2 * kk
            lo, hi = pool_pair(c, t, prow)
            c["put_words"](
                region,
                t * MOE_HIDDEN + c["rank"] * SL + prow,
                [lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)],
            )


@flyc.jit
def stage_pred(c, t, sp, n_sp):
    """pred task (with pshard): part ``sp`` of ``n_sp`` of token ``t``'s rows of this rank's pooled slice, the K
    eighths' partials summed in part order (fp32), pushed to every rank's ``poolr`` (two rows a thread, 16 B a
    push). On the boundary CTAs right after their passes: many CTAs push a little each (a single CTA's remote
    stores would stream too slowly)."""
    tid, W = c["tid"], c["W"]
    SL = MOE_HIDDEN // W  # this rank's pooled rows
    RP = SL // n_sp  # rows a task
    if const_expr("pred" in PUSH_SET and W > 1):
        pred_push_all(c, t, sp, RP)
        return
    if tid < RP // 2:
        prow = sp * RP + 2 * tid
        lo, hi = pool_pair(c, t, prow)
        grow = c["rank"] * SL + prow
        for pe in range_constexpr(W):
            c["put_words"](
                c["peer_poolr"][pe],
                t * MOE_HIDDEN + grow,
                [lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)],
            )


@flyc.jit
def stage_xn(c, t, xin_flag=None):
    """xn task: token ``t``'s pre-expert norm of its pooled row (8 elements a
    thread, the K parts' partial rows summed), weight ``xn_w``, one bf16 rounding
    -> the expert input; with ``xq_pre`` also its MX FP8 blocks (``k2_xinq``, the
    blocks K3 would quantize from the bf16 row: ``ffn.stage_xq_cols``)."""
    tid, put = c["tid"], c["put"]
    a = c["args"]
    e0 = tid * 8
    xw = ld_bf(a["xn_w"], e0, 8)  # before the poll: after it, a dependent round trip
    if const_expr(c["pshard"]):
        # every rank's slice of token t, summed over the eighths on its rank (``stage_pred``)
        got = c["poll"](
            [(c["own_poolr"], t * MOE_HIDDEN + e0 + 2 * j, 2) for j in range(4)],
            batch=4,
        )
        v = [got[j // 2][j % 2].bitcast(fx.Float32) for j in range(8)]
    else:
        v = _xn_pool(c, c["pool"], t, e0)
    stamp(
        c, 18
    )  # every rank's pooled slice of the token arrived (xn CTAs, thread 0's columns)
    v = [
        bf16_round(x) for x in v
    ]  # the pooled row's one rounding, before its norm (as the model's)
    ss = fx.Float32(0.0)
    for j in range_constexpr(8):
        ss = ss + v[j] * v[j]
    rstd = hw_rsq(block_sum(c, ss, 0) / float(MOE_HIDDEN) + RMS_EPS)
    stamp(c, 19)  # the pre-expert norm's sum
    out = [v[j] * rstd * xw[j] for j in range(8)]
    st_bf(a["xin"], t * MOE_HIDDEN + e0, out, CM_DEV)
    if const_expr(c["xq_pre"]):
        vals = [bf16_round(x) for x in out]
        amax = abs(vals[0])
        for k in range_constexpr(1, 8):
            amax = fx.max(amax, abs(vals[k]))
        amax = butterfly(amax, (1, 2), fx.max)
        e8, inv = mx8_scale(amax)
        q = [x * inv for x in vals]
        w0 = fp8_word(q[0], q[1], q[2], q[3])
        w1 = fp8_word(q[4], q[5], q[6], q[7])
        sw = butterfly(e8 << (8 * ((tid // 4) % 4)), (4, 8), _bor)
        base = t * (XQW + XSW)
        bo.buffer_store(fx.Vector.from_elements([w0, w1], fx.Int32), rsrc(c["xinq"]), base + 2 * tid, cache_modifier=CM_DEV)  # fmt: skip
        if tid % 16 == 0:
            bo.buffer_store(
                sw, rsrc(c["xinq"]), base + XQW + tid // 16, cache_modifier=CM_DEV
            )
    stamp(c, 20)  # the expert input stored (bf16 row, MX FP8 blocks)
    if const_expr(xin_flag is not None):
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(xin_flag, t * FLAG_STRIDE, fx.Int32(1))
    stamp(c, 21)  # the token's expert-input flag raised


# ---------------------------------------------------------------- the CTAs' roles
@flyc.jit
def run_k2(c, route_flag=None, xin_flag=None, o_flag=None, QL=8):
    """K2's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    in a launch that goes on to the experts, the route and xn tasks raise a token's
    flag once its routing / expert input is out. ``o_flag``: the attention ran
    earlier in this launch, a flag a (request of ``QL`` rows, KV head, row tile)
    once its o rows are out."""
    S, bid = c["S"], c["bid"]
    A_PARTS = a_parts(S)
    R0 = route0(S)
    stamp(c, 0)
    stage_o_all(c, o_flag, QL)
    stamp(c, 2)  # the o tasks done
    boundary = (bid >= AP0) & (bid < AP0 + A_PARTS * S)
    NC = 2 * (A_PAIRS_MAX // A_PARTS)
    v = [fx.Float32(0.0) for _ in range(NC)]
    if boundary:
        v = stage_a_part(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS)
    stamp(c, 3)  # the boundary part done (boundary CTAs)
    # the router / pooler weights: issued before the norm sums' wait, which streams them
    first = passes_first(c)
    wait_x(c)
    stamp(c, 4)  # every token's x arrived
    first = (
        *first,
        router_w(c, first[0]),
    )  # (issued with the x stage's loads: one round trip)
    stage_passes(c, first)
    stamp(c, 5)  # the router / pooler passes done
    if const_expr(c["pshard"] and PRED_FIRST):
        if boundary:
            stage_pred(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS, A_PARTS)
        stamp(c, 14)  # the pred task done (boundary CTAs)
    if boundary:
        stage_a_part_res(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS, v)
    stamp(c, 6)  # the boundary's stream update done (boundary CTAs)
    if const_expr(c["pshard"] and not PRED_FIRST):
        if boundary:
            stage_pred(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS, A_PARTS)
        stamp(c, 14)  # the pred task done (boundary CTAs)
    if (bid >= R0) & (bid < R0 + S):
        stage_route(c, bid - R0, route_flag)
    if (bid >= XN0) & (bid < XN0 + S):
        stage_xn(c, bid - XN0, xin_flag)
    stamp(c, 7)  # end
