# SPDX-License-Identifier: Apache-2.0
"""K3 on gfx1250: one Muse Spark layer from the experts' input to the next layer's
attention input, for S = ``tokens`` rows on a rank of the TP 8 / EP 8 group. Per
token and each of its picks this rank holds, the chain of the normal gfx1250 expert
path (aiter's A8W4 TDM grouped MoE + Muse's finalize) on the experts as SGLang keeps
them on gfx1250 (``prepare_gfx1250_a8w4_weights``: w13 gate / up rows interleaved
then 16 x 16 byte-tile shuffled, w2 tile shuffled, n32k4 E8M0 scales)::

    x (bf16: K2's pre-expert-normed pooled input)
      -> MX FP8 (E4M3, E8M0 a 32-value block, RoundUp; ``stages.mxfp4``)
      -> gate / up on the scaled FP4 x FP8 WMMA (fp32)
      -> silu(gate) up in fp32 (sigmoid through tanh, as aiter's epilogue)
      -> MX FP8 again (no bf16 rounding first)
      -> down on the scaled WMMA -> y (bf16, a row a (token, slot))
      -> the finalize: v = bf16(y pen_w), out += score v / sqrt(mean v^2 + 1e-8)
         over the slots in order -> bf16 partial rows
      -> pushed to every rank, summed in rank order (the one-stage all-reduce) -> bf16
      -> the unpooler (rank r's rows r HIDDEN / npes .., bf16 after the fp32 sum),
         gathered into every rank
      -> the FFN boundary: post_input = bf16(branch pfw), nb = bf16(post_input
         rstd) (eps 1e-8), stream = alpha stream + beta nb (fp32, in place), the
         next layer's pre-attention norm (eps 1e-5, weight npw) -> bf16; on the
         last layer the stream is rounded to bf16 before that norm

The experts' work is grouped by expert. Every CTA builds the step's plan in LDS
(a ballot a local expert over rounds of WAVE picks; a pick's column is its rank
among its expert's picks), then the picked experts' columns are cut into column
tiles of 16 (a WMMA's N). The ug items (64 interleaved w13 rows: 32 mids) and the dn
items (32 w2 rows) are tile-major, spread as contiguous ranges over the CTAs; the
waves split K (UG_CHUNKS chunks of 128 a wave) and their accumulators are summed in
wave order through LDS. A WMMA takes the weights as A (FP4, rows = M) and the staged
columns as B (FP8, columns = N); a 32-row super-row's E8M0 scales are one word a
lane (n32k4), tile 0 / 1 selected by the scale lane half.

Stages (CTA map in ``k3_bases``; every wait points to an earlier stage, all CTAs
co-resident)::

    plan    every CTA
    ug      n_tiles x UG_T items: mids -> ``k3_mid`` blocks (FP8 codes + scale)
    dn      n_tiles x DN_T items: y rows -> ``k3_y``
    fin     S x AR_PARTS tasks on CTAs S ..: the finalize, the push to every rank's
            ``ffn`` region, the rank-ordered sum -> ``k3_fbuf`` + a flag
    unpool  the rank's row tiles on the CTAs past the fin tasks: rows pushed to every
            rank's ``unp`` region
    bnd     S tasks on CTAs 0 ..: the whole row's boundary -> stream, attention input
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.ops import (
    CM_DEV,
    WAVE_OFFSETS,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fresh,
    hw_rsq,
    popcount,
    read_lane_i32,
    rsrc,
    sum_partials,
    uniform,
    uniform64,
    wave_incl_scan,
    wmma_bf16,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, cdiv, pair_layout
from sgl_amd_mono.common.sync import preg
from sgl_amd_mono.muse_spark.gfx1250.config import (
    EXPERTS,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    MAX_TOKENS,
    MOE_HIDDEN,
    MXFP4_GROUP,
    POST_EPS,
    RMS_EPS,
    TOP_K,
)
from sgl_amd_mono.muse_spark.gfx1250.device import (
    ATTN_COLS,
    FLAG_STRIDE,
    PF_LINE,
    PUSH_SET,
    async_copy4,
    async_copy4_at,
    async_copy16,
    async_copy16_at,
    l2_prefetch_gvs,
    lds_addr,
    lds_dma_16,
    lds_load4,
    lds_load16,
    pin64,
    pin_add,
    poll_flags,
    rank_base,
    release_stores,
    stamp,
    wait_lds_dma,
    wait_xcnt,
    wave_rank,
    wstamp,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import TRACE_LAYERS, TRACE_SLOTS
from sgl_amd_mono.muse_spark.gfx1250.stages.mxfp4 import (
    fp8_byte,
    fp8_word,
    lanes_before,
    mx8_scale,
)

POOL_PARTS = 8  # the pooler's K parts (post_attn.P_PARTS)
# weight super-rows a CTA keeps in flight in the ug / dn loops: 2 (two register slots) or 4
K3_SLOTS = int(os.environ.get("SGLANG_AMD_MONO_K3_SLOTS", "2"))
# SGLANG_AMD_MONO_RSAG3=1: the experts' all-reduce (fin) as a reduce-scatter + all-gather, as post_attn's: a fin
# thread pushes its partial pair to the rank owning those columns (MOE_HIDDEN / npes a rank) only; the owner sums
# the ranks' partials (rank order, one rounding) and pushes the result to every rank's ``fred``.
RSAG = os.environ.get("SGLANG_AMD_MONO_RSAG3", "0") == "1"
# SGLANG_AMD_MONO_UNP_FULL=1: every rank unpools whole rows (all UNP_TASKS row tiles of the unpooler, which the
# model keeps replicated: ``post_expert_proj``) into its own scratch ``k3_unpl``, on every CTA (a fin CTA after its
# fin task), and the boundary reads them there: the unpooled rows take no exchange (no AG4). The rank reads the
# whole unpooler (67 MB) instead of its slice (8.4 MB). SGLANG_AMD_MONO_PF_UNP=1 (with it): the CTAs without a fin
# task prefetch the unpooler into GL2 while the fin tasks wait for the ranks' partials, once every fin task saw
# its token's y rows (``k3_pflag``: the rank's dn reads are over).
UNP_FULL = os.environ.get("SGLANG_AMD_MONO_UNP_FULL", "0") == "1"
PF_UNP = UNP_FULL and os.environ.get("SGLANG_AMD_MONO_PF_UNP", "0") == "1"

# ---------------------------------------------------------------- the plan
COLS = MAX_TOKENS  # picking columns an expert can have: a token picks it at most once
TILE_COLS = 16  # columns an expert item takes: the WMMA's N
TILES_PER_EXPERT = COLS // TILE_COLS  # 4
MAX_TILES = LOCAL_EXPERTS * TILES_PER_EXPERT
# ---------------------------------------------------------------- the experts
CHUNK = 128  # k of one scaled WMMA
UG_CHUNKS = MOE_HIDDEN // CHUNK // WAVES  # chunks a wave: the waves split K
DN_CHUNKS = INTER // CHUNK // WAVES
XS_LDS_MAX = 16  # tokens an unpool pass stages in LDS
XQW = MOE_HIDDEN // 4  # FP8 code words of a staged column (four codes a word)
XQP = XQW + 4  # LDS words a staged column: the 16 B skew spreads the columns' banks
XSW = MOE_HIDDEN // MXFP4_GROUP // 4  # its E8M0 scales, four a word (a 128-K chunk's)
XS_BASE = TILE_COLS * XQP  # the staged scales' first word in ``xcol``
MID_W = 9  # mailbox pairs a mid block: eight words of four FP8 codes, the E8M0 code
UG_T = (
    INTER // 32
)  # ug items a column tile: 32 mids each (64 interleaved gate / up rows)
DN_T = MOE_HIDDEN // 32  # dn items a column tile: 32 y rows each (one super-row)
MID_PAIRS = UG_T * MID_W  # a (token, slot)'s mid blocks
W_ROW_BYTES = MOE_HIDDEN // 2  # a packed FP4 row (w13 and w2 alike: K = 4096)
W_SCALE_COLS = MOE_HIDDEN // MXFP4_GROUP  # a row's E8M0 scales
TILE_BYTES = 16 * W_ROW_BYTES  # a 16-row byte tile of the shuffled weights
SR_SCALE_BYTES = 32 * W_SCALE_COLS  # a 32-row n32k4 scale super-row
ACC = 8  # fp32 accumulator values a lane (16 x 16 / WAVE)
RED_WT = (
    WAVE * ACC + ACC
)  # a wave's tile in ``red`` (floats): 32 lanes x 8, lanes 16.. 32 B further
RED_SLOT_FLOATS = (
    WAVES * RED_WT + 64
)  # one tile's accumulators of every wave, padded 256 B
RED_SLOTS = 8  # two parities x four tiles (ug); dn uses 2 x 2
assert (
    RED_SLOT_FLOATS % 64 == 0
    and 48 + (WAVES - 1) * RED_WT + WAVE * ACC + ACC <= RED_SLOT_FLOATS
)


def red_at(slot, w, lane):
    """Wave ``w``'s lane ``lane`` accumulators of tile slot ``slot`` in ``red`` (float index).
    The lanes past 16 sit 32 B further and slot k of a group of four 64 B further, so the 32
    lanes of one column's reduction (four slots x two lane halves x four row pairs) read 64
    distinct banks."""
    return (
        slot * RED_SLOT_FLOATS
        + (slot % 4) * 16
        + w * RED_WT
        + lane * ACC
        + (lane // 16) * ACC
    )


# ---------------------------------------------------------------- fin / unpool / bnd
AR_PARTS_MAX = 8
UNP_ROWS = 16  # unpooler rows an unpool task: the WMMA's M
UNP_TASKS = HIDDEN // UNP_ROWS  # 512 row tiles, 1 / npes of them a rank
UNP_STEPS = MOE_HIDDEN // WAVES // 32  # bf16 WMMA steps a wave
UNP_PITCH = MOE_HIDDEN // 2 + 4  # LDS floats a staged unpool row (16 B skew)
BND_ELEMS = HIDDEN // THREADS  # 16 elements a thread of a token's row
assert WAVE == 32, "the gfx1250 K3 maps a 32-value MX block onto a wave32"
assert XQW == 2 * THREADS and INTER == MOE_HIDDEN and UG_CHUNKS >= 1 and DN_CHUNKS >= 1
assert MOE_HIDDEN % (CHUNK * WAVES) == 0 and XSW * 4 * 32 == MOE_HIDDEN
assert MOE_HIDDEN == 8 * THREADS and HIDDEN == 16 * THREADS and TOP_K == 8
assert LOCAL_EXPERTS <= WAVE and LOCAL_EXPERTS % WAVES == 0 and UNP_STEPS >= 1
assert MAX_TOKENS % TILE_COLS == 0 and BND_ELEMS == 16 and DN_T % (4 * WAVE) == 0
MAX_PICKS = MAX_TOKENS * TOP_K


def tile_cols(tokens: int) -> int:
    """Columns a tile can hold at this width: a tile's rounds of x / mid staging."""
    return min(TILE_COLS, tokens)


# SGLANG_AMD_MONO_FIN_PARTS: fin tasks a token at S <= 8 (default 4; 8: twice the CTAs push the experts'
# all-reduce, half the unpool CTAs take two jobs each)
FIN_PARTS = int(os.environ.get("SGLANG_AMD_MONO_FIN_PARTS", "0"))
assert FIN_PARTS in (0, 2, 4, 8), FIN_PARTS
FIN8_MAX = 4  # 8 fin tasks a token up to this width (FIN_PARTS=0: 8, then 4 up to FIN4_MAX rows, then 2)
# (4 tasks a token keep ``fin_push_idle`` and leave the unpool a third of the CTAs at 16 rows, two jobs each: the
# layer bench is faster at 9, 12 and 16 rows than with 2 tasks a token, with the same output bits)
FIN4_MAX = 16
# A wave's loads wait for the acknowledgements of the cross-GPU stores it issued before them, so a
# task that pushes and then polls its peers sees their data late. Two role splits (default on; 0: the first forms):
# SGLANG_AMD_MONO_FIN_IDLE: a fin task's partial goes out from the CTA's idle waves (``fin_push_idle``, where the
# part leaves enough of them: 4 tasks a token); the waves that poll the ranks' partials issue no cross-GPU store.
FIN_IDLE = os.environ.get("SGLANG_AMD_MONO_FIN_IDLE", "1") == "1"
# SGLANG_AMD_MONO_BND_FIN: the bnd tasks on the first S fin CTAs (their pushes are long acknowledged by
# then) instead of CTAs 0 .. S, whose unpool job pushes its rows right before the bnd's poll.
BND_FIN = os.environ.get("SGLANG_AMD_MONO_BND_FIN", "1") == "1"


def ar_parts(tokens: int) -> int:
    """fin / AR tasks a token (``MOE_HIDDEN / parts`` columns each): S x parts CTAs
    spread the pushes and polls; every other CTA (the S bnd CTAs too) unpools."""
    if FIN_PARTS and tokens <= 8:
        return FIN_PARTS
    return 8 if tokens <= FIN8_MAX else (4 if tokens <= FIN4_MAX else 2)


def unp_tokens(tokens: int) -> int:
    """Tokens an unpool pass (a job's B columns)."""
    return min(tokens, XS_LDS_MAX)


def fin_ppt(parts: int) -> int:
    """Column pairs a fin thread (1 or 2: a poll spec of at most two pairs). ``SGLANG_AMD_MONO_FIN_PPT2=1``: two
    whenever the part's columns allow, so each push is a 16 B store (a CTA's remote stores stream faster than
    with 8 B ones), on half the threads."""
    ppt = cdiv(MOE_HIDDEN // parts // 2, THREADS)
    if (
        os.environ.get("SGLANG_AMD_MONO_FIN_PPT2", "1") == "1"
        and (MOE_HIDDEN // parts) % 4 == 0
    ):
        ppt = 2
    assert ppt in (1, 2), ppt
    return ppt


def k3_bases(tokens: int, npes: int) -> dict:
    """K3's CTA map: the bnd tasks on CTAs 0 .. S, the fin tasks (``ar_parts`` a
    token) from ``fin0`` to ``unp0``; the unpool jobs on every other CTA (unpool slot
    u: CTAs 0 .. S, then ``unp0`` ..), ``unp_stride`` slots taking ``ku`` jobs each,
    so a fin CTA's polls wait behind no unpool weights. ``UNP_FULL``: every row tile,
    on every CTA (job bid + BLOCKS k)."""
    parts = ar_parts(tokens)
    fin0 = tokens
    unp0 = fin0 + tokens * parts
    stride = BLOCKS if UNP_FULL else BLOCKS - tokens * parts
    tasks = UNP_TASKS if UNP_FULL else UNP_TASKS // npes
    passes = cdiv(tokens, unp_tokens(tokens))  # token passes: a job each
    assert UNP_TASKS % npes == 0 and stride > 0, (
        f"K3 at {tokens} rows needs more than the {BLOCKS} resident CTAs"
    )
    return {
        "ar_parts": parts,
        "fin0": fin0,
        "bnd0": fin0 if BND_FIN else 0,  # the bnd tasks' first CTA
        "unp0": unp0,
        "unp_stride": stride,
        "unp_tasks": tasks,
        "unp_passes": passes,
        "unp_jobs": tasks * passes,  # (row tile, token pass) pairs over the unpool CTAs
        "ku": cdiv(tasks * passes, stride),
    }


def k3_scratch_layout() -> dict:
    """K3's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS: the
    mid blocks and y rows a (token, slot), a flag a (token, AR part), the dn items'
    sums of squares of the finalized values a (token, slot, item), and the
    all-reduce's result as plain bf16 words (``k3_fbuf``, read by async LDS copies)."""
    return pair_layout(
        [
            ("k3_mid", MAX_TOKENS * TOP_K * MID_PAIRS),
            ("k3_y", MAX_TOKENS * TOP_K * MOE_HIDDEN // 2),
            ("k3_fflag", MAX_TOKENS * AR_PARTS_MAX),
            ("k3_ss", MAX_TOKENS * TOP_K * DN_T),
            ("k3_fbuf", MAX_TOKENS * MOE_HIDDEN // 4),
            ("k3_nscal", MAX_TOKENS * 2),  # a token's (post rstd, pre rstd) for the next K1 (deferred)
            ("k3_ugq", 2),  # (unused on gfx1250: no ug queue)
            ("k3_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
            ("k3_unpl", MAX_TOKENS * HIDDEN // 2),  # UNP_FULL: this rank's whole unpooled rows (tagged bf16 pairs)
            ("k3_pflag", MAX_TOKENS * AR_PARTS_MAX),  # PF_UNP: a fin task's flag once its token's y rows are in
        ],
        align=256,
    )  # fmt: skip


def k3_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric regions, name -> (byte offset, bytes): ``attn`` and
    ``ffn`` [source rank][token][width / 2] bf16 pairs, ``unp`` [token][HIDDEN / 2]
    (each rank's unpooled rows from that rank), then K2's regions. Must equal
    post_attn.sym_layout (the same names, order and formula)."""
    return pair_layout(
        [
            ("attn", npes * MAX_TOKENS * ATTN_COLS // 2),
            ("ffn", npes * MAX_TOKENS * MOE_HIDDEN // 2),
            ("unp", MAX_TOKENS * HIDDEN // 2),
            # (TP-expert regions of the gfx950 build; kept so the offsets match post_attn's)
            ("midx", MAX_TOKENS * TOP_K * MID_PAIRS),
            ("vx", MAX_TOKENS * TOP_K * MOE_HIDDEN // 2),
            ("ssx", MAX_TOKENS * TOP_K * DN_T),
            # (a region of the gfx950 build's sharded pooler, not written here; kept so the offsets stay)
            ("poolx", POOL_PARTS * MAX_TOKENS * MOE_HIDDEN // 2),
            # the K-sharded boundary (post_attn ``kshard``)
            ("kss", MAX_TOKENS * POOL_PARTS * 8 * 4),
            ("vx2", MAX_TOKENS * HIDDEN // 2),
            ("lgx", POOL_PARTS * MAX_TOKENS * EXPERTS),
            ("poolq", POOL_PARTS * MAX_TOKENS * MOE_HIDDEN // 4),
            ("poolf", MAX_TOKENS * MOE_HIDDEN // 2),
            ("scx", MAX_TOKENS * TOP_K),
            # the attention all-reduce's all-gather half (``SGLANG_AMD_MONO_RSAG``): [token][HIDDEN / 2], each
            # rank's reduced column block from that rank
            ("ared", MAX_TOKENS * HIDDEN // 2),
            # the experts' all-reduce's all-gather half (``SGLANG_AMD_MONO_RSAG``): [token][MOE_HIDDEN / 2]
            ("fred", MAX_TOKENS * MOE_HIDDEN // 2),
            # the sharded pooler: every rank's rows of the pooled row, summed over the K parts on their rank:
            # [token][MOE_HIDDEN], an fp32 value a pair
            ("poolr", MAX_TOKENS * MOE_HIDDEN),
        ],
        align=256,
    )


# SGLANG_AMD_MONO_K3_WAVE: how a CTA's waves take the expert (ug / dn) items. 0 (this module's default): the waves
# split an item's K and their accumulators are summed through LDS; 1: a wave runs whole items over the full K,
# the next chunks' weights in a register FIFO; 2 (the server's default, ``muse_spark/spec.py``): the same with the
# weights through the wave's own LDS ring of async copies (it reads K2's staged MX FP8 input, so it needs an
# ``xq_pre`` build, as the runner's layer is).
K3_WAVE = int(os.environ.get("SGLANG_AMD_MONO_K3_WAVE", "0"))
# the wave-owned items' async LDS ring (``K3_WAVE == 2``): a wave's weights stream through
# its own ring of chunk slots (``global_load_async_to_lds``: no VGPRs in flight, counted by asynccnt);
# a super-row's chunk takes tiles 0 / 1 (lo, hi) x 32 lanes x 16 B, then its 32 scale words
SR_SLOT_BYTES = 4 * 512 + 128
K3W_RING_UG = int(
    os.environ.get("SGLANG_AMD_MONO_K3W_RING_UG", "2")
)  # ug chunks a wave holds
RING_W_BYTES = (
    K3W_RING_UG * 2 * SR_SLOT_BYTES
)  # a wave's ring (dn: twice the chunks, one super-row each)


def k3_lds_floats() -> dict:
    """K3's LDS, name -> floats (ints for the plan), independent of the width."""
    return {
        # (the LDS ring build reduces nothing across waves: fin / unpool use one slot)
        "red": (1 if K3_WAVE == 2 else RED_SLOTS) * RED_SLOT_FLOATS,
        # a tile's FP8 columns + their scales; the unpool passes' rows
        "xcol": max(XS_BASE + TILE_COLS * XSW, XS_LDS_MAX * UNP_PITCH),
        "misc": max(UNP_ROWS * XS_LDS_MAX, 2 * WAVES * TILE_COLS),  # dn's wave partials
        "sums": 2 * WAVES,  # ``block_sum``'s wave sums
        "pick_of": LOCAL_EXPERTS
        * COLS,  # expert e's column j: its pick (token TOP_K + slot)
        "cnt_of": EXPERTS,  # expert e's picks
        "start_of": 4,  # (unused on gfx1250)
        "tile_of": MAX_TILES,  # column tile i: e COLS + its first column
        "nt": 4,
        "pw": MOE_HIDDEN
        // 2,  # the post-expert norm weight (bf16 pairs), staged by the plan
        "ring": WAVES * RING_W_BYTES // 4 if K3_WAVE == 2 else 4,
    }


def k3_lds_bytes(tokens: int) -> int:
    return 4 * sum(k3_lds_floats().values())


def k3_lds(tokens: int):
    """K3's LDS: the tiles' accumulators, a tile's staged columns (also the
    unpool passes' rows), the reductions, the step's plan."""
    f = k3_lds_floats()
    n_red, n_xcol, n_misc, n_sums = f["red"], f["xcol"], f["misc"], f["sums"]
    n_pick_of, n_cnt_of, n_start_of = f["pick_of"], f["cnt_of"], f["start_of"]
    n_tile_of, n_nt, n_pw, n_ring = f["tile_of"], f["nt"], f["pw"], f["ring"]

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, n_red, 16]
        xcol: fx.Array[fx.Float32, n_xcol, 16]
        misc: fx.Array[fx.Float32, n_misc, 16]
        sums: fx.Array[fx.Float32, n_sums, 16]
        pick_of: fx.Array[fx.Int32, n_pick_of, 16]
        cnt_of: fx.Array[fx.Int32, n_cnt_of, 16]
        start_of: fx.Array[fx.Int32, n_start_of, 16]
        tile_of: fx.Array[fx.Int32, n_tile_of, 16]
        nt: fx.Array[fx.Int32, n_nt, 16]
        pw: fx.Array[fx.Int32, n_pw, 16]
        ring: fx.Array[fx.Int32, n_ring, 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def _v(n, ty=fx.Float32):
    return fx.Vector.make_type(n, ty)


def silu_mul(g, u):
    """silu(g) u as aiter's stage-1 epilogue: g (0.5 (1 + tanh(g / 2))) u."""
    th = fx.Float32(rocdl.tanh(T.f32, (g * fx.Float32(0.5)).ir_value()))
    sig = fx.Float32(0.5) * (fx.Float32(1.0) + th)
    return g * sig * u


def bor(x, y):
    return x | y


def cat(parts):
    """Concatenate equal-length vectors (in order)."""
    while len(parts) > 1:
        n = 2 * len(fx.Vector(parts[0]))
        parts = [
            fx.Vector(parts[i]).shuffle(fx.Vector(parts[i + 1]), list(range(n)))
            for i in range(0, len(parts), 2)
        ]
    return fx.Vector(parts[0])


def ld4(r, i, cm=0):
    return fx.Vector(bo.buffer_load(r, i, vec_width=4, dtype=T.i32, cache_modifier=cm))


def lds_barrier():
    """A workgroup barrier for an LDS hand-off only: waits for this wave's LDS
    operations, not for its global loads (``gpu.barrier`` drains those too, which
    stalls every weight prefetch in flight)."""
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def acc_pair(red, slot, m, j):
    """Rows m, m + 1 (m even) of tile ``slot``, column j: the waves' WMMA accumulators
    summed in wave order (wave w's lane l at ``red + red_at(slot, w, l)``)."""
    lane = j + 16 * (m // 8)
    a0 = fx.Float32(0.0)
    a1 = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        v = fx.Vector(
            fx.ptr_load(
                red + (red_at(slot, w, lane) + m % 8),
                result_type=_v(2),
            )
        )
        a0 = a0 + fx.Float32(v[0])
        a1 = a1 + fx.Float32(v[1])
    return a0, a1


def tile_store(c, slot, acc):
    """This wave's accumulator of tile ``slot`` into ``red``."""
    fx.ptr_store(acc, c["red"] + red_at(slot, c["wave"], c["lane"]))


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` through ``sums`` row ``slot`` (one barrier)."""
    lane, wave, sums = c["lane"], c["wave"], c["sums"]
    ws = butterfly(v, WAVE_OFFSETS)
    if lane == 0:
        fx.ptr_store(ws, sums + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(sums + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(sums + (slot * WAVES + w))
    return tot


def pick_at(c, e, col):
    """Expert e's column ``col``'s pick: token TOP_K + slot."""
    return fx.ptr_load(c["pick_of"] + (e * COLS + col))


def tile_info(c, ti, n_tiles):
    """Column tile ``ti``'s (local expert, first column, columns), clamped to valid
    values past the plan (a CTA with no item still issues loads: expert 0's)."""
    tc = fx.max(fx.min(ti, n_tiles - 1), fx.Int32(0))
    packed = uniform(fx.ptr_load(c["tile_of"] + tc))
    packed = fx.max(fx.min(packed, fx.Int32(LOCAL_EXPERTS * COLS - 1)), fx.Int32(0))
    e = packed // COLS
    col0 = packed % COLS
    cnt = uniform(fx.ptr_load(c["cnt_of"] + e))
    n = fx.max(fx.min(cnt - col0, fx.Int32(TILE_COLS)), fx.Int32(1))
    return e, col0, n


def sr_loads(c, r_w, r_s, srow, chunks):
    """This lane's loads of the 32-row super-row from weight row ``srow`` (a multiple
    of 32) over the wave's ``chunks`` k chunks, in flight: [tile 0 words, tile 1
    words, scale words], a tile's words a chunk the 8 dwords of its WMMA A operand.
    The shuffled layout keeps a row's 32-value block b at byte (row / 16) TILE_BYTES
    + 256 b + 16 (row % 16): lane (m, h) of a tile takes blocks 4 kc + h and 4 kc + 2
    + h of chunk kc; the n32k4 scales a lane's word of super-row row ``lane``."""
    lane, wave = c["lane"], c["wave"]
    l16, h = lane % 16, lane // 16
    # the scale words first: a wait for them leaves the weight words in flight
    scs = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                ((srow // 32) * SR_SCALE_BYTES + (wave * chunks + kc) * 128 + lane * 4) // 4,
                vec_width=1,
                dtype=T.i32,
            )
        )
        for kc in range(chunks)
    ]  # fmt: skip
    tiles = []
    for t in range(2):
        base = (srow // 16 + t) * TILE_BYTES + l16 * 16
        ws = []
        for kc in range(chunks):
            kcg = wave * chunks + kc
            lo = ld4(r_w, (base + (4 * kcg + h) * 256) // 4)
            hi = ld4(r_w, (base + (4 * kcg + 2 + h) * 256) // 4)
            ws.append(cat([lo, hi]))
        tiles.append(ws)
    return [tiles[0], tiles[1], scs]


def tile_wmmas(ld, t, xb):
    """Tile t's (0 / 1 of the super-row loads ``ld``) fp32 accumulator over the
    wave's chunks: the scaled FP4 x FP8 WMMA on the weights against the staged
    column blocks ``xb[kc]`` = (16 words, scale word)."""
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    for kc in range(len(xb)):
        acc = fx.Vector(
            rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                T.vec(ACC, T.f32), ld[t][kc], xb[kc][0], acc, ld[2][kc], xb[kc][1],
                fmtA=4, fmtB=0, scaleAType=t, scaleBType=0,
            )
        )  # fmt: skip
    return acc


def w_addrs(c, e, dn):
    """Expert e's w13 (``dn`` false) or w2 weight and scale base addresses."""
    a = c["args"]
    rows = MOE_HIDDEN if dn else 2 * INTER
    w = a["w2"] if dn else a["w13"]
    s = a["s2"] if dn else a["s13"]
    return w + fx.Int64(e) * (rows * W_ROW_BYTES), s + fx.Int64(e) * (
        rows * W_SCALE_COLS
    )


W_ALL = 0xFFFFFFFF  # a live slot's buffer range (the expert's rows; offsets stay below 4 GB)


def ug_slot_loads(c, q, pr, dq, n_tiles, use_dn, live):
    """ug item ``q``'s super-row ``pr``, or (``use_dn``) dn item ``dq``'s super-row, as
    ``ug_or_dn_pair_loads``; ``live`` false: no item (a dead slot of the unrolled loops), the
    loads read a zero-size buffer, zeros and no memory traffic."""
    e_u, _, _ = tile_info(c, q // UG_T, n_tiles)
    e_d, _, _ = tile_info(c, dq // DN_T, n_tiles)
    wu, su = w_addrs(c, e_u, False)
    wd, sd = w_addrs(c, e_d, True)
    srow = use_dn.select((dq % DN_T) * 32, (q % UG_T) * 64 + pr * 32)
    nb = live.select(fx.Int64(W_ALL), fx.Int64(0))
    w = use_dn.select(wd, wu)
    sc = use_dn.select(sd, su)
    return sr_loads(c, rsrc(w, nb), rsrc(sc, nb), srow, UG_CHUNKS)


def dn_slot_loads(c, d0, dn_n, pos, n_tiles):
    """dn position ``pos`` of the range from item ``d0`` (``dn_n`` items): its super-row's
    loads, or none past the range (zeros, no traffic)."""
    dq = d0 + fx.max(fx.min(pos, dn_n - 1), fx.Int32(0))
    return ug_slot_loads(c, d0, 0, dq, n_tiles, fx.Int32(0) == fx.Int32(0), pos < dn_n)


def ug_or_dn_pair_loads(c, q, pr, dq, n_tiles, use_dn):
    """ug item ``q``'s super-row ``pr`` (interleaved w13 rows 64 (q % UG_T) + 32 pr ..),
    or (``use_dn``) dn item ``dq``'s super-row: the same load shape (w13 and w2 rows
    are both MOE_HIDDEN = INTER wide), so the ug loop's last prefetch fetches the dn
    stage's first weights."""
    e_u, _, _ = tile_info(c, q // UG_T, n_tiles)
    e_d, _, _ = tile_info(c, dq // DN_T, n_tiles)
    wu, su = w_addrs(c, e_u, False)
    wd, sd = w_addrs(c, e_d, True)
    srow = use_dn.select((dq % DN_T) * 32, (q % UG_T) * 64 + pr * 32)
    return sr_loads(
        c, rsrc(use_dn.select(wd, wu)), rsrc(use_dn.select(sd, su)), srow, UG_CHUNKS
    )


def dn_pair_loads(c, q, n_tiles):
    """dn item q's super-row: 32 rows of the expert's down weight."""
    e, _, _ = tile_info(c, q // DN_T, n_tiles)
    w, s = w_addrs(c, e, True)
    return sr_loads(c, rsrc(w), rsrc(s), (q % DN_T) * 32, DN_CHUNKS)


def xq_loads(c, e, col0, n):
    """The bf16 x of expert e's columns col0 .. col0 + n, every round's load at
    once (in flight): round i is column i, eight values a thread. A column past n
    reads the last one's (its store is skipped); the token is clamped to the step's
    rows, as a rank no pick reached stages its (unused) first tile from a plan it
    never wrote."""
    S, tid = c["S"], c["tid"]
    r_xin = rsrc(c["args"]["xin"])

    def x_unit(i):
        tk = pick_at(c, e, col0 + fx.min(fx.Int32(i), n - 1)) // TOP_K
        tk = fx.max(fx.min(tk, fx.Int32(S - 1)), fx.Int32(0))
        return ld4(r_xin, tk * (MOE_HIDDEN // 2) + tid * 4, CM_DEV)

    return [x_unit(i) for i in range(tile_cols(S))]


def xq_all_loads(c):
    """Every token's MX FP8 blocks (K2's ``xinq``: XQW code words then XSW scale words a
    token), 16 B a thread a round, in flight (S <= TILE_COLS)."""
    S, tid = c["S"], c["tid"]
    n4 = S * (XQW + XSW) // 4
    r = rsrc(c["args"]["xinq"])
    return [ld4(r, 4 * fx.min(tid + THREADS * i, n4 - 1), CM_DEV) for i in range(cdiv(n4, THREADS))]  # fmt: skip


@flyc.jit
def stage_xq_all(c, xv):
    """The blocks (``xq_all_loads``) into LDS: token t's code words at xcol[t XQP ..],
    its scale words at xcol[XS_BASE + t XSW ..] (a 16 B unit never straddles)."""
    S, tid, xcol = c["S"], c["tid"], c["xcol"]
    W_TOK = XQW + XSW
    n4 = S * W_TOK // 4
    for i in range_constexpr(len(xv)):
        v = tid + THREADS * i
        if v < n4:
            d = 4 * v
            tk, o = d // W_TOK, d % W_TOK
            at = (o < XQW).select(tk * XQP + o, XS_BASE + tk * XSW + (o - XQW))
            fx.ptr_store(fx.Vector(xv[i]).bitcast(fx.Float32), xcol + at)
    lds_barrier()


def xq_b(c, col, chunks, kc):
    """This lane's B operand of chunk ``kc`` from staged LDS column ``col``: the 16
    words of K 32 j + 16 h .. (j = 0 .. 3, h the lane half) and the chunk's scale word."""
    lane, wave, xcol = c["lane"], c["wave"], c["xcol"]
    kcg = wave * chunks + kc
    base = col * XQP + kcg * 32 + (lane // 16) * 4
    parts = [fx.ptr_load(xcol + (base + 8 * j), result_type=_v(4)) for j in range(4)]
    q = cat(parts).bitcast(fx.Int32)
    sc = fx.ptr_load(xcol + (XS_BASE + col * XSW + kcg)).bitcast(fx.Int32)
    return q, sc


def lane_col(c, n, cb=0):
    """The staged LDS column of this lane's tile column (lane % 16; past n: the last)."""
    return cb + fx.min(c["lane"] % 16, n - 1)


def lane_token(c, e, col0, n):
    """The token of this lane's column (lane % 16) of a tile (the last column's past n)."""
    tk = pick_at(c, e, col0 + fx.min(c["lane"] % 16, n - 1)) // TOP_K
    return fx.max(fx.min(tk, fx.Int32(c["S"] - 1)), fx.Int32(0))


def unp_gtask(c, k, before_fin=False):
    """CTA ``bid``'s k-th unpool job: (its index among the rank's jobs, its row tile of
    the unpooler, its token pass; outside the rank's jobs: the tile past the buffer, no
    traffic). ``UNP_FULL``: job bid + BLOCKS k of every row tile; ``before_fin``: a fin
    CTA has none yet (its first job's weights go out after its fin task)."""
    b = c["bases"]
    bid = c["bid"]
    fin = (bid >= b["fin0"]) & (bid < b["unp0"])
    if UNP_FULL:
        job = bid + BLOCKS * k
        if before_fin:
            job = fin.select(fx.Int32(-1), job)
        live = (job >= 0) & (job < b["unp_jobs"])
        return (
            job,
            live.select(job // b["unp_passes"], fx.Int32(UNP_TASKS)),
            job % b["unp_passes"],
        )
    slot = (bid < b["fin0"]).select(bid, bid - (b["unp0"] - b["fin0"]))
    job = fin.select(fx.Int32(-1), slot + b["unp_stride"] * k)
    live = (job >= 0) & (job < b["unp_jobs"])
    tile = job // b["unp_passes"]
    ps = job % b["unp_passes"]
    return job, live.select(c["rank"] * b["unp_tasks"] + tile, fx.Int32(UNP_TASKS)), ps


def unp_w_loads(c, gtask):
    """A lane's loads of unpooler row tile ``gtask`` (16 rows x MOE_HIDDEN, row-major
    bf16): row l16, this wave's K slice, 16 contiguous values a step (lane half h holds
    K 16 h .. of a 32-K step); in flight."""
    lane, wave = c["lane"], c["wave"]
    l16, h = lane % 16, lane // 16
    r_u = rsrc(
        c["args"]["w_unpool"], HIDDEN * MOE_HIDDEN * 2
    )  # a tile past the last: zeros
    row = gtask * UNP_ROWS + l16
    out = []
    for kc in range(UNP_STEPS):
        k0 = wave * (MOE_HIDDEN // WAVES) + kc * 32 + 16 * h
        at = (row * MOE_HIDDEN + k0) // 2
        out.append(cat([ld4(r_u, at), ld4(r_u, at + 4)]))
    return out


def unp_prefetch(c, slot, n_slots: int):
    """``PF_UNP``: CTA ``slot`` of ``n_slots``' share of the whole unpooler (HIDDEN x MOE_HIDDEN bf16, row-major)
    into GL2, in row order, a line a thread and instruction."""
    total = HIDDEN * MOE_HIDDEN * 2 // PF_LINE
    per = -(-total // (n_slots * THREADS)) * THREADS
    assert n_slots * per == total, (
        n_slots,
        per,
        total,
    )  # the shares tile the unpooler exactly
    base = uniform64(fx.Int64(c["args"]["w_unpool"]) + fx.Int64(slot * per) * PF_LINE)
    for i in range(per // THREADS):
        l2_prefetch_gvs(base, c["tid"] * PF_LINE, i * THREADS * PF_LINE)


# ---------------------------------------------------------------- 0. the plan
def plan_pw_loads(c):
    """The post-expert norm weight's loads of ``stage_plan`` (16 B a thread), in flight. K3's entry issues them
    before it waits for its flags: loads return in issue order, and these come from HBM, so issued inside the
    plan they held the picks' loads back."""
    PW4 = MOE_HIDDEN // 2 // 4  # 16 B units
    tid = c["tid"]
    return [
        ld4(rsrc(c["args"]["pen_w"]), 4 * fx.min(tid + THREADS * i, PW4 - 1))
        for i in range(cdiv(PW4, THREADS))
    ]


# SGLANG_AMD_MONO_PLAN_IDX=1 (default; a merged launch): the plan takes the picks from K2's tagged ``idx`` pairs (the
# route task puts them right after its top-8), and K3's entry waits for the expert input alone. The routing flag
# comes later (the route task's plain stores drained, a barrier, the flag, a poll): with it the experts wait for
# the routing whenever the logits arrive last.
PLAN_IDX = os.environ.get("SGLANG_AMD_MONO_PLAN_IDX", "1") == "1"
# SGLANG_AMD_MONO_PLAN_OVL (1 / 0; default auto: at <= PLAN_OVL_MAX rows; with PLAN_IDX and the entry's x staging):
# wave 0 polls the tagged picks (every round's pairs in one round trip) while the expert input's staging loads are
# in flight, and the staging's LDS barrier hands both over (after the staging, the picks' poll, a round trip a
# round, its barrier and the reads came on top although the picks had landed).
PLAN_OVL = os.environ.get("SGLANG_AMD_MONO_PLAN_OVL", "auto")
assert PLAN_OVL in ("0", "1", "auto"), PLAN_OVL
PLAN_OVL_MAX = 4  # auto: up to this width (as ``post_attn.ROUTE_SPLIT_MAX``: no gain measured above)


def plan_ovl(tokens: int) -> bool:
    return PLAN_OVL == "1" or (PLAN_OVL == "auto" and tokens <= PLAN_OVL_MAX)


@flyc.jit
def plan_picks_poll(c):
    """``PLAN_OVL``: wave 0's poll of the tagged picks into ``misc`` (a lane a pick a round); the caller's next LDS
    barrier publishes them (``stage_plan`` then reads them: ``picks_staged``)."""
    S, lane, wave = c["S"], c["lane"], c["wave"]
    PICKS = S * TOP_K
    ROUNDS = cdiv(PICKS, WAVE)
    if wave == 0:
        got = c["poll"](
            [
                (c["plan_idx"], fx.min(rr * WAVE + lane, PICKS - 1), 1)
                for rr in range(ROUNDS)
            ],
            batch=ROUNDS,
        )
        for rr in range_constexpr(ROUNDS):
            fx.ptr_store(
                fx.Int32(got[rr][0]).bitcast(fx.Float32), c["misc"] + (rr * WAVE + lane)
            )


@flyc.jit
def stage_plan(c):
    """The step's plan in LDS, the same on every CTA: wave w owns local experts
    EPW w .. EPW w + EPW and walks every pick (token p / TOP_K, slot p % TOP_K) in
    rounds of WAVE; a pick's column is its rank among its expert's picks in pick
    order (a ballot an expert a round, the count a uniform scalar); then wave 0
    lists the experts' column tiles of 16 in expert order. Returns the tiles."""
    S, lane, wave = c["S"], c["lane"], c["wave"]
    pick_of, cnt_of, tile_of, nt = c["pick_of"], c["cnt_of"], c["tile_of"], c["nt"]
    a = c["args"]
    local_start = a["local_start"]
    EPW = LOCAL_EXPERTS // WAVES
    r_ids = rsrc(a["ids"])
    PICKS = S * TOP_K
    ROUNDS = cdiv(PICKS, WAVE)
    # the post-expert norm weight into LDS (dn reads a word an item from there: a global load
    # inside the dn loop would make its wait drain every weight stream issued before it)
    PW4 = MOE_HIDDEN // 2 // 4  # 16 B units
    tid = c["tid"]
    pw_v = c.get("pw_loads")
    if pw_v is None:
        pw_v = plan_pw_loads(c)
    if const_expr(c.get("plan_idx") is not None):
        # the tagged picks (``PLAN_IDX``): wave 0 polls them (a lane a pick a round, until its tag is this launch's)
        # and hands them to the CTA's waves through LDS. Every wave of the GPU polling (re-reading the same lines)
        # saw the picks later than a few threads a CTA do
        if const_expr(
            not c.get("picks_staged")
        ):  # (``PLAN_OVL``: polled and published at K3's entry)
            if wave == 0:
                for rr in range_constexpr(ROUNDS):
                    got = c["poll"](
                        [(c["plan_idx"], fx.min(rr * WAVE + lane, PICKS - 1), 1)]
                    )
                    fx.ptr_store(
                        fx.Int32(got[0][0]).bitcast(fx.Float32),
                        c["misc"] + (rr * WAVE + lane),
                    )
            lds_barrier()
        p_ids = [
            fx.ptr_load(c["misc"] + (rr * WAVE + lane)).bitcast(fx.Int32)
            for rr in range(ROUNDS)
        ]
    else:
        p_ids = [
            fx.Int32(
                bo.buffer_load(
                    r_ids, fx.min(rr * WAVE + lane, PICKS - 1), vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
            for rr in range(ROUNDS)
        ]  # fmt: skip
    stamp(c, 19)  # the picks arrived (the tagged pairs' poll, or the plain loads)
    cnts = [
        fx.Int32(0) for _ in range(EPW)
    ]  # this wave's experts' picks so far (uniform)
    assert PICKS <= EXPERTS - LOCAL_EXPERTS, (
        "the picks' table takes cnt_of's unused words"
    )
    for rr in range_constexpr(ROUNDS):
        p = rr * WAVE + lane
        le = p_ids[rr] - local_start
        ok = (p < PICKS) & (le >= 0) & (le < LOCAL_EXPERTS)
        if (wave == 0) & (p < PICKS):
            # pick p's local expert, or -1 (``fin_held``: the fin tasks' slots, from the picks this plan took)
            fx.ptr_store(ok.select(le, fx.Int32(-1)), cnt_of + (LOCAL_EXPERTS + p))
        col = fx.Int32(0)
        mine = fx.Int32(0) == fx.Int32(1)
        for k in range_constexpr(EPW):
            hit = ok & (le == wave * EPW + k)
            m = ballot(hit)
            col = hit.select(cnts[k] + lanes_before(m, lane), col)
            cnts[k] = cnts[k] + uniform(popcount(m))
            mine = mine | hit
        if mine & (col < COLS):
            fx.ptr_store(p, pick_of + (le * COLS + col))
    for k in range_constexpr(EPW):
        if lane == 0:
            fx.ptr_store(cnts[k], cnt_of + (wave * EPW + k))
    for i in range_constexpr(len(pw_v)):
        if tid + THREADS * i < PW4:
            fx.ptr_store(fx.Vector(pw_v[i]), c["pw"] + 4 * (tid + THREADS * i))
    gpu.barrier()
    if wave == 0:
        # lane e < LOCAL_EXPERTS: expert e's count, its tiles and their first index
        mine0 = lane < LOCAL_EXPERTS
        cnt = mine0.select(
            fx.ptr_load(cnt_of + fx.min(lane, LOCAL_EXPERTS - 1)), fx.Int32(0)
        )
        tiles = mine0.select(
            (fx.min(cnt, COLS) + TILE_COLS - 1) // TILE_COLS, fx.Int32(0)
        )
        incl = wave_incl_scan(tiles, lane)
        tile0 = incl - tiles
        n_tiles = read_lane_i32(incl, WAVE - 1)
        if lane == 0:
            # an empty plan's first tile is expert 0, column 0 (a later real tile 0 wins)
            fx.ptr_store(fx.Int32(0), tile_of)
            fx.ptr_store(n_tiles, nt)
        for j in range_constexpr(TILES_PER_EXPERT):
            if j < tiles:
                fx.ptr_store(lane * COLS + j * TILE_COLS, tile_of + (tile0 + j))
    gpu.barrier()
    return uniform(fx.ptr_load(nt))


# ---------------------------------------------------------------- 1 + 2. the experts
@flyc.jit
def stage_xq_cols(c, e, col0, n, xv=None):
    """Expert e's columns col0 .. col0 + n as MX FP8 blocks into LDS: column j's
    code words at xcol[j XQP ..], its scale words at xcol[XS_BASE + j XSW ..]; a
    32-value block is four lanes' (``xv``: the loads, already in flight)."""
    S, tid, xcol = c["S"], c["tid"], c["xcol"]
    if const_expr(xv is None):
        xv = xq_loads(c, e, col0, n)
    for i in range_constexpr(tile_cols(S)):
        vb = fx.Vector(xv[i]).bitcast(fx.BFloat16).to(fx.Float32)
        vals = [fx.Float32(vb[k]) for k in range(8)]
        amax = abs(vals[0])
        for k in range_constexpr(1, 8):
            amax = fx.max(amax, abs(vals[k]))
        amax = butterfly(amax, (1, 2), fx.max)
        e8, inv = mx8_scale(amax)
        q = [v * inv for v in vals]
        w0 = fp8_word(q[0], q[1], q[2], q[3])
        w1 = fp8_word(q[4], q[5], q[6], q[7])
        sw = butterfly(e8 << (8 * ((tid // 4) % 4)), (4, 8), bor)
        if fx.Int32(i) < n:
            fx.ptr_store(
                fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Float32),
                xcol + (i * XQP + 2 * tid),
            )
            if (tid % 16) == 0:
                fx.ptr_store(
                    sw.bitcast(fx.Float32), xcol + (XS_BASE + i * XSW + tid // 16)
                )
    lds_barrier()


@flyc.jit
def stage_mid_q(c, e, col0, n):
    """Expert e's columns col0 .. col0 + n's mid blocks (MX FP8, from the ug items)
    into LDS in ``stage_xq_cols``' layout; every round polled in one spin (a column
    past n polls the last one's, its store skipped)."""
    S, tid, xcol, poll, mid = c["S"], c["tid"], c["xcol"], c["poll"], c["mid"]
    XR = cdiv(tile_cols(S) * UG_T, THREADS)

    def specs(i):
        v = tid + THREADS * i  # block v % UG_T of column v / UG_T
        j = fx.min(v // UG_T, n - 1)
        base = (pick_at(c, e, col0 + j) * UG_T + v % UG_T) * MID_W
        return [(mid, base + 2 * w, 2) for w in range(4)] + [(mid, base + 8, 1)]

    got = poll([sp for i in range(XR) for sp in specs(i)], batch=5 * XR)
    for i in range_constexpr(XR):
        v = tid + THREADS * i
        g = 5 * i
        blk = v % UG_T
        sw = butterfly((got[g + 4][0] & 0xFF) << (8 * (blk % 4)), (1, 2), bor)
        if v < n * UG_T:
            words = [got[g + w // 2][w % 2] for w in range(8)]
            at = (v // UG_T) * XQP + blk * 8
            fx.ptr_store(
                fx.Vector.from_elements(words[:4], fx.Int32).bitcast(fx.Float32),
                xcol + at,
            )
            fx.ptr_store(
                fx.Vector.from_elements(words[4:], fx.Int32).bitcast(fx.Float32),
                xcol + (at + 4),
            )
            if (blk % 4) == 0:
                fx.ptr_store(
                    sw.bitcast(fx.Float32),
                    xcol + (XS_BASE + (v // UG_T) * XSW + blk // 4),
                )
    lds_barrier()


@flyc.jit
def put_mid_block(c, blk, e, col0, n, par):
    """Each column's 32 mids of block ``blk`` as one MX FP8 block to ``k3_mid``:
    wave j = column, lane r = mid r of the block (a whole wave a block). Mid r is
    the item's interleaved rows 2 r (gate) and 2 r + 1 (up): super-row r / 16, tile
    (r % 16) / 8, rows M = 2 (r % 8), M + 1 of that tile."""
    lane, wave, put, mid, red = c["lane"], c["wave"], c["put"], c["mid"], c["red"]
    j = wave
    r = lane
    slot = 4 * par + 2 * (r // 16) + (r % 16) // 8
    gate, up = acc_pair(red, slot, 2 * (r % 8), fx.min(j, n - 1))
    act = silu_mul(gate, up)
    amax = butterfly(abs(act), WAVE_OFFSETS, fx.max)
    e8, inv = mx8_scale(amax)
    word = butterfly(fp8_byte(act * inv) << (8 * (r % 4)), (1, 2), bor)
    if (j < n) & ((r % 4) == 0):
        base = (pick_at(c, e, col0 + j) * UG_T + blk) * MID_W
        put(mid, base + r // 4, word)
        if r == 0:
            put(mid, base + 8, e8)


@flyc.jit
def put_rows(c, region, width, r, rows, e, col0, n, vals_of):
    """Thread (column j = tid % 16, row pair rp = tid / 16): rows 2 rp, + 1 of the
    item's ``rows`` for column j < n -> ``region`` (``width`` elements a row) at
    the column's pick's row, ``vals_of(row, j)`` the value, bf16 pairs."""
    tid, put_bf = c["tid"], c["put_bf"]
    j = tid % TILE_COLS
    for g in range_constexpr(cdiv(rows // 2, THREADS // TILE_COLS)):
        rp = tid // TILE_COLS + g * (THREADS // TILE_COLS)
        if (j < n) & (rp < rows // 2):
            p = pick_at(c, e, col0 + j)
            put_bf(region, p * width + r * rows + 2 * rp, [vals_of(2 * rp, j), vals_of(2 * rp + 1, j)])  # fmt: skip


def item_range(slot, total):
    """The first item and the item count of CTA slot ``slot`` of ``total`` items in
    contiguous ranges: the last ``total % BLOCKS`` slots take one item more. The dn
    ranges go to the slots in reverse (``BLOCKS - 1 - bid``)."""
    m, r = total // BLOCKS, total % BLOCKS
    first = slot * m + fx.max(slot - (BLOCKS - r), 0)
    return first, m + (slot >= BLOCKS - r).select(fx.Int32(1), fx.Int32(0))


@flyc.jit
def stage_ug(c, n_tiles):
    """This CTA's contiguous range of the ug items (tile, 32-mid block), two a loop turn in
    four fixed register slots (two super-rows an item): each slot's next super-row, two
    items ahead, is issued right after its WMMAs, so two items (four super-rows) stream while
    one computes and no loop-carried register is copied (a copy waits for its load). An odd
    range's last turn has a dead second item (zeros, no traffic, nothing published). The
    tile's x is restaged at a tile boundary; after both super-rows the SwiGLU mids of the 32
    rows leave as an MX FP8 block. The reduction rows alternate between the turn's two
    items: one barrier an item. Returns the dn stage's first four items' weights, in flight
    (the slots' prefetch past the ug range)."""
    bid = c["bid"]
    TU = n_tiles * UG_T
    q0, n_items = item_range(bid, TU)
    t0 = q0 // UG_T
    e0, c0, n0 = tile_info(c, t0, n_tiles)
    pre = c["xq_pre"]
    xv0 = xq_all_loads(c) if pre else xq_loads(c, e0, c0, n0)
    TD = n_tiles * DN_T
    dq0, dn_n = item_range(BLOCKS - 1 - bid, TD)
    NE = ((n_items + 1) // 2) * 2  # positions the loop covers (an odd range: one dead)

    def pos_loads(pos, pr):
        """Slot ``pr`` (super-row) of range position ``pos``: an ug item's super-row, none
        (a dead position), or past the loop the dn stage's items 2 (pos - NE) + pr."""
        dpos = 2 * (pos - NE) + pr
        dq = dq0 + fx.max(fx.min(dpos, dn_n - 1), fx.Int32(0))
        is_dn = pos >= NE
        live = (pos < n_items) | (is_dn & (dpos < dn_n))
        return ug_slot_loads(
            c,
            q0 + fx.max(fx.min(pos, n_items - 1), fx.Int32(0)),
            pr,
            dq,
            n_tiles,
            is_dn,
            live,
        )

    a0 = pos_loads(fx.Int32(0), 0)
    a1 = pos_loads(fx.Int32(0), 1)
    b0 = pos_loads(fx.Int32(1), 0)
    b1 = pos_loads(fx.Int32(1), 1)
    if const_expr(pre):
        stage_xq_all(c, xv0)
    else:
        stage_xq_cols(c, e0, c0, n0, xv0)

    def item(pos, par, s0, s1, cur_t):
        """Position ``pos`` (reduction rows ``par``) from slots ``s0`` / ``s1``; their next
        weights (position pos + 2) and the staged tile."""
        live = pos < n_items
        q = q0 + fx.max(fx.min(pos, n_items - 1), fx.Int32(0))
        ti = q // UG_T
        e, col0, n = tile_info(c, ti, n_tiles)
        if const_expr(not pre):
            if ti != cur_t:
                lds_barrier()  # every thread is done with the previous tile's columns
                stage_xq_cols(c, e, col0, n)
                cur_t = ti
        if const_expr(pre):
            col = lane_token(c, e, col0, n)
        else:
            col = lane_col(c, n)
        xb = [xq_b(c, col, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        for t in range_constexpr(2):
            tile_store(c, 4 * par + t, tile_wmmas(s0, t, xb))
        n0_ = pos_loads(pos + 2, 0)
        for t in range_constexpr(2):
            tile_store(c, 4 * par + 2 + t, tile_wmmas(s1, t, xb))
        n1_ = pos_loads(pos + 2, 1)
        lds_barrier()
        if live:
            put_mid_block(c, q % UG_T, e, col0, n, par)
        return n0_, n1_, cur_t

    cur_t = t0
    for k in range(0, NE // 2):
        a0, a1, cur_t = item(2 * k, 0, a0, a1, cur_t)
        b0, b1, cur_t = item(2 * k + 1, 1, b0, b1, cur_t)
    return a0, a1, b0, b1


@flyc.jit
def stage_dn(c, n_tiles, first=None):
    """This CTA's contiguous range of the dn items (tile, 32-row block), four a loop
    turn in four fixed register slots (each slot's item four ahead issued after its
    WMMAs; past the range a dead item: zeros, no traffic, nothing published): the tile's
    mid blocks restaged at a tile boundary, each item's 32 y rows of its columns to
    ``k3_y``, and their finalized values' sums of squares (v = bf16(bf16(y) pen_w), a
    wave partial a column in LDS, summed and published an item later, behind that item's
    barrier) to ``k3_ss``. ``first``: the first four items' weights, in flight."""
    bid, red = c["bid"], c["red"]
    TD = n_tiles * DN_T
    q0, n_items = item_range(BLOCKS - 1 - bid, TD)
    slots = (
        list(first)
        if first is not None
        else [dn_slot_loads(c, q0, n_items, fx.Int32(j), n_tiles) for j in range(4)]
    )
    tid, lane, wave, misc, put = c["tid"], c["lane"], c["wave"], c["misc"], c["put"]
    j16, rp = tid % TILE_COLS, tid // TILE_COLS

    def publish_ss(par, pq, pe, pc0, pn):
        if (pq >= 0) & (tid < TILE_COLS) & (tid < pn):
            tot = fx.ptr_load(misc + (par * (WAVES * TILE_COLS) + tid))
            for w in range_constexpr(1, WAVES):
                tot = tot + fx.ptr_load(
                    misc + (par * (WAVES * TILE_COLS) + w * TILE_COLS + tid)
                )
            put(c["ss"], pick_at(c, pe, pc0 + tid) * DN_T + pq % DN_T, tot)

    def item(pos, par, w, cur_t, prev):
        """dn item at range position ``pos`` (its weights ``w``; past the range: dead),
        reduction rows / ss partials by the position's parity ``par``. Returns (the slot's
        next weights, the staged tile, this item as ``prev``)."""
        live = pos < n_items
        q = q0 + fx.max(fx.min(pos, n_items - 1), fx.Int32(0))
        ti = q // DN_T
        e, col0, n = tile_info(c, ti, n_tiles)
        if ti != cur_t:
            lds_barrier()  # every thread is done with the previous tile's columns
            stage_mid_q(c, e, col0, n)
            stamp(c, 12)  # a tile's mids arrived (the CTA's last tile's survives)
            cur_t = ti
        rr = 2 * fx.min(rp, fx.Int32(15))  # this thread's row pair of the item
        pw = bf2_f32(fx.ptr_load(c["pw"] + (((q % DN_T) * 32 + rr) // 2)))
        col = lane_col(c, n)
        xb = [xq_b(c, col, DN_CHUNKS, kc) for kc in range(DN_CHUNKS)]
        for t in range_constexpr(2):
            tile_store(c, 2 * par + t, tile_wmmas(w, t, xb))
        nxt = dn_slot_loads(c, q0, n_items, pos + 4, n_tiles)
        lds_barrier()
        publish_ss(1 - par, *prev)
        n_pub = live.select(n, fx.Int32(0))  # a dead item publishes nothing
        ya, yb = acc_pair(red, 2 * par + rr // 16, rr % 16, j16)
        put_rows(c, c["y"], MOE_HIDDEN, q % DN_T, 32, e, col0, n_pub, lambda row, j: (row % 2 == 0).select(ya, yb))  # fmt: skip
        # (row pairs 16 .. 31 are no rows of the item: threads 256 .. 511 add 0)
        v0 = bf16_round(bf16_round(ya) * pw[0])
        v1 = bf16_round(bf16_round(yb) * pw[1])
        ss = ((j16 < n_pub) & (rp < 16)).select(v0 * v0 + v1 * v1, fx.Float32(0.0))
        ss = butterfly(ss, (16,))  # the wave's two row pairs of column j16
        if lane < TILE_COLS:
            fx.ptr_store(
                ss, misc + (par * (WAVES * TILE_COLS) + wave * TILE_COLS + lane)
            )
        return nxt, cur_t, live.select(q, fx.Int32(-1)), e, col0, n_pub

    s0, s1, s2, s3 = slots
    cur_t = fx.Int32(-1)
    pq, pe, pc0, pn = fx.Int32(-1), fx.Int32(0), fx.Int32(0), fx.Int32(0)
    for k in range(0, (n_items + 3) // 4):
        s0, cur_t, pq, pe, pc0, pn = item(4 * k, 0, s0, cur_t, (pq, pe, pc0, pn))
        s1, cur_t, pq, pe, pc0, pn = item(4 * k + 1, 1, s1, cur_t, (pq, pe, pc0, pn))
        s2, cur_t, pq, pe, pc0, pn = item(4 * k + 2, 0, s2, cur_t, (pq, pe, pc0, pn))
        s3, cur_t, pq, pe, pc0, pn = item(4 * k + 3, 1, s3, cur_t, (pq, pe, pc0, pn))
    lds_barrier()  # ``red`` is reused by the fin and unpool stages; the last item's partials are in
    # the last turn's last item (parity 1; past the range: dead, and its predecessor published)
    publish_ss(fx.Int32(1), pq, pe, pc0, pn)


# ---------------------------------------------------------------- 3. fin + all-reduce
@flyc.jit
def stage_ug2(c, n_tiles):
    """This CTA's contiguous range of the ug items (tile, 32-mid block), each two
    super-rows held in two fixed register slots: a slot's next super-row (the next
    item's) is issued right after its WMMAs, so two super-rows stream while the
    other computes and no loop-carried register is copied (a copy waits for its
    load). The tile's x is restaged at a tile boundary; after both super-rows the
    SwiGLU mids of the 32 rows leave as an MX FP8 block. The reduction rows
    alternate between items: one barrier an item. Returns the dn stage's first two
    items' weights, in flight (the last item's prefetch)."""
    bid = c["bid"]
    TU = n_tiles * UG_T
    q0, n_items = item_range(bid, TU)
    t0 = q0 // UG_T
    e0, c0, n0 = tile_info(c, t0, n_tiles)
    pre = c["xq_pre"]
    xv0 = xq_all_loads(c) if pre else xq_loads(c, e0, c0, n0)
    TD = n_tiles * DN_T
    dq0, dn_n = item_range(BLOCKS - 1 - bid, TD)
    dq1 = dq0 + fx.min(
        fx.Int32(1), dn_n - 1
    )  # dn's second item (its first again in a range of one)
    none = n_items == 0  # a CTA without ug items: dn's first two items first thing
    cur0 = ug_or_dn_pair_loads(c, q0, 0, dq0, n_tiles, none)
    cur1 = ug_or_dn_pair_loads(c, q0, 1, dq1, n_tiles, none)
    if const_expr(pre):
        stage_xq_all(c, xv0)
    else:
        stage_xq_cols(c, e0, c0, n0, xv0)
    cur_t = t0
    for i in range(0, n_items):
        q = q0 + i
        ti = q // UG_T
        e, col0, n = tile_info(c, ti, n_tiles)
        if const_expr(not pre):
            if ti != cur_t:
                lds_barrier()  # every thread is done with the previous tile's columns
                stage_xq_cols(c, e, col0, n)
                cur_t = ti
        last = i + 1 >= n_items
        if const_expr(pre):
            col = lane_token(c, e, col0, n)
        else:
            col = lane_col(c, n)
        xb = [xq_b(c, col, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        par = i % 2
        for t in range_constexpr(2):
            tile_store(c, 4 * par + t, tile_wmmas(cur0, t, xb))
        nxt0 = ug_or_dn_pair_loads(c, q + 1, 0, dq0, n_tiles, last)
        for t in range_constexpr(2):
            tile_store(c, 4 * par + 2 + t, tile_wmmas(cur1, t, xb))
        nxt1 = ug_or_dn_pair_loads(c, q + 1, 1, dq1, n_tiles, last)
        lds_barrier()
        put_mid_block(c, q % UG_T, e, col0, n, par)
        cur0 = nxt0
        cur1 = nxt1
    return cur0, cur1


@flyc.jit
def stage_dn2(c, n_tiles, first=None):
    """This CTA's contiguous range of the dn items (tile, 32-row block), two a loop
    turn in two fixed register slots (each slot's item two ahead issued after its
    WMMAs; an odd range's last turn recomputes its last item, publishing the same
    values again): the tile's mid blocks restaged at a tile boundary, each item's
    32 y rows of its columns to ``k3_y``, and their finalized values' sums of squares
    (v = bf16(bf16(y) pen_w), a wave partial a column in LDS, summed and published
    an item later, behind that item's barrier) to ``k3_ss``. ``first``: the first
    two items' weights, in flight."""
    bid, red = c["bid"], c["red"]
    TD = n_tiles * DN_T
    q0, n_items = item_range(BLOCKS - 1 - bid, TD)
    cur_a, cur_b = (
        first
        if first is not None
        else (
            dn_pair_loads(c, q0, n_tiles),
            dn_pair_loads(c, q0 + fx.min(fx.Int32(1), n_items - 1), n_tiles),
        )
    )
    tid, lane, wave, misc, put = c["tid"], c["lane"], c["wave"], c["misc"], c["put"]
    j16, rp = tid % TILE_COLS, tid // TILE_COLS

    def publish_ss(par, pq, pe, pc0, pn):
        if (pq >= 0) & (tid < TILE_COLS) & (tid < pn):
            tot = fx.ptr_load(misc + (par * (WAVES * TILE_COLS) + tid))
            for w in range_constexpr(1, WAVES):
                tot = tot + fx.ptr_load(
                    misc + (par * (WAVES * TILE_COLS) + w * TILE_COLS + tid)
                )
            put(c["ss"], pick_at(c, pe, pc0 + tid) * DN_T + pq % DN_T, tot)

    def item(pos, w, cur_t, prev):
        """dn item at range position ``pos`` (its weights ``w``; past the range: the
        last item again), reduction rows / ss partials by the position's parity.
        Returns (the slot's next weights, the staged tile, this item as ``prev``)."""
        q = q0 + fx.min(pos, n_items - 1)
        ti = q // DN_T
        e, col0, n = tile_info(c, ti, n_tiles)
        if ti != cur_t:
            lds_barrier()  # every thread is done with the previous tile's columns
            stage_mid_q(c, e, col0, n)
            stamp(c, 12)  # a tile's mids arrived (the CTA's last tile's survives)
            cur_t = ti
        rr = 2 * fx.min(rp, fx.Int32(15))  # this thread's row pair of the item
        pw = bf2_f32(fx.ptr_load(c["pw"] + (((q % DN_T) * 32 + rr) // 2)))
        par = pos % 2
        col = lane_col(c, n)
        xb = [xq_b(c, col, DN_CHUNKS, kc) for kc in range(DN_CHUNKS)]
        for t in range_constexpr(2):
            tile_store(c, 2 * par + t, tile_wmmas(w, t, xb))
        nxt = dn_pair_loads(c, q0 + fx.min(pos + 2, n_items - 1), n_tiles)
        lds_barrier()
        publish_ss(1 - par, *prev)
        ya, yb = acc_pair(red, 2 * par + rr // 16, rr % 16, j16)
        put_rows(c, c["y"], MOE_HIDDEN, q % DN_T, 32, e, col0, n, lambda row, j: (row % 2 == 0).select(ya, yb))  # fmt: skip
        # (row pairs 16 .. 31 are no rows of the item: threads 256 .. 511 add 0)
        v0 = bf16_round(bf16_round(ya) * pw[0])
        v1 = bf16_round(bf16_round(yb) * pw[1])
        ss = ((j16 < n) & (rp < 16)).select(v0 * v0 + v1 * v1, fx.Float32(0.0))
        ss = butterfly(ss, (16,))  # the wave's two row pairs of column j16
        if lane < TILE_COLS:
            fx.ptr_store(
                ss, misc + (par * (WAVES * TILE_COLS) + wave * TILE_COLS + lane)
            )
        return nxt, cur_t, q, e, col0, n

    cur_t = fx.Int32(-1)
    pq, pe, pc0, pn = fx.Int32(-1), fx.Int32(0), fx.Int32(0), fx.Int32(0)
    for k in range(0, (n_items + 1) // 2):
        cur_a, cur_t, pq, pe, pc0, pn = item(2 * k, cur_a, cur_t, (pq, pe, pc0, pn))
        cur_b, cur_t, pq, pe, pc0, pn = item(2 * k + 1, cur_b, cur_t, (pq, pe, pc0, pn))
    lds_barrier()  # ``red`` is reused by the fin and unpool stages; the last item's partials are in
    # the last turn's second item (parity 1; past an odd range: the last item again)
    publish_ss(fx.Int32(1), pq, pe, pc0, pn)


# ---------------------------------------------------------------- 3. fin + all-reduce
# ---------------------------------------------------------------- wave-owned expert items
# K3_WAVE = 1: a wave runs whole ug / dn items over the full K (no cross-wave K split): no ``red``
# reduction and no CTA barrier per item, and a FIFO of the next chunks' weights in flight per wave
# (``K3W_DEPTH_UG`` / ``K3W_DEPTH_DN`` chunks of 128 K).
K3W_DEPTH_UG = int(os.environ.get("SGLANG_AMD_MONO_K3W_DEPTH_UG", "4"))
K3W_DEPTH_DN = int(os.environ.get("SGLANG_AMD_MONO_K3W_DEPTH_DN", "8"))
K_CHUNKS = MOE_HIDDEN // CHUNK  # 32 chunks of 128 over the full K (w13 and w2 alike)
# items a wave takes at most: a CTA's range of at most cdiv(LOCAL_EXPERTS UG_T, BLOCKS) items
# (S <= 16: an expert has one column tile), split over the waves
K3W_ITEMS = cdiv(cdiv(LOCAL_EXPERTS * max(UG_T, DN_T), BLOCKS), WAVES)
# SGLANG_AMD_MONO_K3_WSLOT: which wave takes a CTA's k-th expert item (ring build). A CTA at a narrow step has
# fewer items than waves, and the waves that stream share the CU's SIMDs: an item takes longer the more waves of
# its CU are busy, and the ring depth hardly matters: the stream is bound by the waves' share of a SIMD, not by
# memory. 0: item k on wave k (the first waves); 1: the waves in the order 0, 4, 8, 12, 1, 5, .. (one a group of four first); 2 (default):
# 0, 5, 10, 15, 1, 6, 11, 12, .. (the first four differ in both wave / 4 and wave % 4, and so does every next four).
K3_WSLOT = int(os.environ.get("SGLANG_AMD_MONO_K3_WSLOT", "2"))
# TIMING PROBE ONLY (wrong results below 4): tiles of a ug item a wave streams: 4 (the item), 2 (one super-row:
# half the copies and WMMAs), 1 (one super-row's copies, one tile's WMMA). What a wave's stream costs when the
# item's tiles go to several waves.
K3_PROBE_TILES = int(os.environ.get("SGLANG_AMD_MONO_K3_PROBE_TILES", "4"))
assert K3_PROBE_TILES in (1, 2, 4)
assert K3_WSLOT in (0, 1, 2) and WAVES == 16, (K3_WSLOT, WAVES)


def wave_slot(c):
    """This wave's item slot in its CTA (``K3_WSLOT``): the wave takes items slot + WAVES i of the CTA's range."""
    wave = c["wave"]
    if K3_WSLOT == 0:
        return wave
    if K3_WSLOT == 1:
        return (wave % 4) * 4 + wave // 4
    return ((wave % 4 - wave // 4 + 4) % 4) * 4 + wave // 4


def chunk_loads(c, r_w, r_s, srow, kc):
    """This lane's loads of super-row ``srow``'s chunk ``kc`` (k 128 kc ..), in flight: the two
    tiles' A operands (8 dwords each, the shuffled layout as ``sr_loads``) and the chunk's
    n32k4 scale word."""
    lane = c["lane"]
    l16, h = lane % 16, lane // 16
    sc = fx.Int32(
        bo.buffer_load(r_s, ((srow // 32) * SR_SCALE_BYTES + kc * 128 + lane * 4) // 4, vec_width=1, dtype=T.i32)
    )  # fmt: skip
    ts = []
    for t in range(2):
        base = (srow // 16 + t) * TILE_BYTES + l16 * 16
        lo = ld4(r_w, (base + (4 * kc + h) * 256) // 4)
        hi = ld4(r_w, (base + (4 * kc + 2 + h) * 256) // 4)
        ts.append(cat([lo, hi]))
    return ts[0], ts[1], sc


def xq_b_at(c, xcol, col, kc):
    """This lane's B operand of chunk ``kc`` (of the full K) from LDS column ``col`` of the
    staged region ``xcol``: 16 code words and the chunk's scale word (``xq_b`` layout)."""
    lane = c["lane"]
    base = col * XQP + kc * 32 + (lane // 16) * 4
    parts = [fx.ptr_load(xcol + (base + 8 * j), result_type=_v(4)) for j in range(4)]
    q = cat(parts).bitcast(fx.Int32)
    sc = fx.ptr_load(xcol + (XS_BASE + col * XSW + kc)).bitcast(fx.Int32)
    return q, sc


def wmma_s(a, b, acc, sa, sb, t):
    return fx.Vector(
        rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            T.vec(ACC, T.f32), a, b, acc, sa, sb, fmtA=4, fmtB=0, scaleAType=t, scaleBType=0
        )
    )  # fmt: skip


def stream_rows(c, srcs, xcol, col, depth):
    """The full-K product of the super-rows ``srcs`` [(r_w, r_s, srow)] with LDS column ``col``
    of ``xcol``: per super-row the two tiles' fp32 accumulators. The weights stream through a
    FIFO ``depth`` chunks deep (each chunk's loads issued ``depth`` chunks before its WMMAs; the
    loads of a chunk complete in issue order, so a chunk's wait leaves the later ones in flight)."""
    accs = [[fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(2)] for _ in srcs]
    fifo = []
    for kc in range_constexpr(min(depth, K_CHUNKS)):
        fifo.append([chunk_loads(c, rw, rs, sr, kc) for (rw, rs, sr) in srcs])
    rocdl.sched_barrier(0)
    for kc in range_constexpr(K_CHUNKS):
        cur = fifo.pop(0)
        if kc + depth < K_CHUNKS:
            fifo.append(
                [chunk_loads(c, rw, rs, sr, kc + depth) for (rw, rs, sr) in srcs]
            )
            rocdl.sched_barrier(0)
        xb, xs = xq_b_at(c, xcol, col, kc)
        for i in range_constexpr(len(srcs)):
            a0, a1, sa = cur[i]
            accs[i][0] = wmma_s(a0, xb, accs[i][0], sa, xs, 0)
            accs[i][1] = wmma_s(a1, xb, accs[i][1], sa, xs, 1)
    return accs


# SGLANG_AMD_MONO_K3_GVS=1 (default): the experts' async copies in the instruction's SGPR-base form. An item's
# weight / scale bases are made wave-uniform once (``u_srcs``: SGPRs); a copy's address is base + this lane's
# fixed offset (one register for every weight copy of the kernel) + a constant (the offset field). The first form
# rebuilt a per-lane 32-bit offset for every copy: 12 VALU ops a ug chunk of 53 instructions, and no-ops behind the
# WMMAs whose operand registers they rewrote. A wave's stream is bound by its instruction rate.
K3_GVS = os.environ.get("SGLANG_AMD_MONO_K3_GVS", "0") == "1"


def u_srcs(srcs):
    """The super-rows [(w, s, srow)] with wave-uniform bases (``K3_GVS``): [(w + the super-row's first tile, s + its
    scale words, None)]; as they are otherwise, or when already so."""
    if not K3_GVS or srcs is None:
        return srcs
    out = []
    for w, s, srow in srcs:
        if srow is None:
            out.append((w, s, None))
        else:
            out.append((uniform64(w + fx.Int64((srow // 16) * TILE_BYTES)), uniform64(s + fx.Int64((srow // 32) * SR_SCALE_BYTES)), None))  # fmt: skip
    return out


def ring_dst(c, n, sn):
    """This lane's LDS byte addresses of a chunk's copies in ring slot ``sn`` (items of ``n`` super-rows), each
    pinned in a register (an opaque move: the compiler otherwise rebuilds every one behind the chunk's WMMAs, and
    a VALU write to a WMMA's operand register right behind it costs eight no-ops): (the weight halves [4 i + 2 t +
    hf], the scale words [i])."""
    lane = c["lane"]
    slot = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES + sn * n * SR_SLOT_BYTES
    return [fresh(slot + j * 512 + lane * 16) for j in range(4 * n)], [fresh(slot + n * 2048 + i * 128 + lane * 4) for i in range(n)]  # fmt: skip


def ring_issue(c, srcs, kc, slot, dst=None):
    """Async copies of chunk ``kc`` of the super-rows ``srcs`` [(w, s, srow): weight / scale base
    addresses, or ``u_srcs``' uniform bases] into the ring slot at LDS byte address ``slot``
    (``chunk_loads``' bytes); ``dst``: the slot's pinned addresses (``ring_dst``)."""
    lane = c["lane"]
    l16, h = lane % 16, lane // 16
    n = len(srcs)
    lw = fx.Int64(
        fx.Uint32(l16 * 16 + h * 256)
    )  # this lane's fixed offset in a tile's chunk half
    ls = fx.Int64(fx.Uint32(lane * 4))
    for i, (w, s, srow) in enumerate(srcs):
        if srow is None:
            for t in range(2):
                for hf in range(2):
                    j = 4 * i + 2 * t + hf
                    async_copy16(w + lw + fx.Int64(t * TILE_BYTES + (4 * kc + 2 * hf) * 256), slot + j * 512 + lane * 16 if dst is None else dst[0][j])  # fmt: skip
            async_copy4(
                s + ls + fx.Int64(kc * 128),
                slot + n * 2048 + i * 128 + lane * 4 if dst is None else dst[1][i],
            )
            continue
        for t in range(2):
            base = (srow // 16 + t) * TILE_BYTES + l16 * 16
            for hf in range(2):
                async_copy16(w + fx.Int64(base + (4 * kc + 2 * hf + h) * 256), slot + (4 * i + 2 * t + hf) * 512 + lane * 16)  # fmt: skip
        async_copy4(s + fx.Int64((srow // 32) * SR_SCALE_BYTES + kc * 128 + lane * 4), slot + n * 2048 + i * 128 + lane * 4)  # fmt: skip


def ring_read(c, n, slot):
    """This lane's A operands and scale words of a ring slot holding ``n`` super-rows' chunk."""
    lane = c["lane"]
    out = []
    for i in range(n):
        ts = []
        for t in range(2):
            lo = lds_load16(slot + (4 * i + 2 * t) * 512 + lane * 16)
            hi = lds_load16(slot + (4 * i + 2 * t + 1) * 512 + lane * 16)
            ts.append(cat([lo, hi]))
        out.append((ts[0], ts[1], lds_load4(slot + n * 2048 + i * 128 + lane * 4)))
    return out


def ring_depth(n):
    """Chunk slots a wave's ring holds for items of ``n`` super-rows."""
    return min(RING_W_BYTES // (n * SR_SLOT_BYTES), K_CHUNKS)


def ring_prologue(c, srcs):
    """Issue the first ``ring_depth`` chunks of an item into this wave's ring (slots 0 ..)."""
    if K3_RAW and K3_PROBE_TILES == 4:
        return ring_prologue_raw(c, srcs)
    n = len(srcs)
    srcs = u_srcs(srcs)
    ring0 = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES
    for kc in range_constexpr(ring_depth(n)):
        ring_issue(c, srcs, kc, ring0 + kc * n * SR_SLOT_BYTES)


# SGLANG_AMD_MONO_K3_RAW=1 (default): the item form's ring in a lockstep layout, its copies as raw instructions. A
# wave's stream is bound by its instruction rate, and a ug chunk was 52 instructions for 4 WMMAs: 10 address
# rebuilds for its 10 refill copies (the copy instruction adds its one offset field to the LDS and to the global
# address, so the compiler subtracts the offset from an LDS address it had just built) and 8 no-ops (a VALU write
# to a WMMA's operand register right behind it). Here tile tau = 2 i + t of the item keeps its own run of the
# ring, a chunk 1 KB as in memory: chunk k's copy of tile tau is LDS[L + O] <- memory[G(tau) + O] with O = (tau
# depth + k) 1024 + 512 hf, L the lap's one base register (chunks depth lap ..: ring0 + 16 lane - 1024 depth lap,
# renewed once a lap) and G(tau) the tile's pinned base (the tile's own bytes - 1024 depth tau). The scale words
# likewise (128 B a chunk, behind the weights). A chunk is then its loads, its WMMAs and its copies: 31
# instructions (ug), 19 (dn).
K3_RAW = os.environ.get("SGLANG_AMD_MONO_K3_RAW", "1") == "1"
RAW_W = 8192  # the weights' bytes of a wave's ring (2 n tiles x depth chunks x 1 KB); the scales' 512 B follow
assert RAW_W + 512 <= RING_W_BYTES


def raw_bases(c, srcs):
    """An item's pinned global bases in the lockstep layout: (G[tile 2 i + t], Gs[super-row i]) for this lane."""
    lane = c["lane"]
    n = len(srcs)
    depth = ring_depth(n)
    assert 2 * n * depth * 1024 == RAW_W and n * depth * 128 == 512, (n, depth)
    lw = fx.Int64(fx.Uint32((lane % 16) * 16 + (lane // 16) * 256))
    ls = fx.Int64(fx.Uint32(lane * 4))
    G, Gs = [], []
    for i, (w, s, srow) in enumerate(srcs):
        wb = w + fx.Int64((srow // 16) * TILE_BYTES) + lw
        for t in range(2):
            G.append(pin64(wb + fx.Int64(t * TILE_BYTES - (2 * i + t) * depth * 1024)))
        Gs.append(
            pin64(
                s
                + fx.Int64((srow // 32) * SR_SCALE_BYTES)
                + ls
                + fx.Int64(-i * depth * 128)
            )
        )
    return G, Gs


def raw_issue(bases, lw_reg, ls_reg, kc, n, depth):
    """Chunk ``kc``'s copies of an item (``raw_bases``) into the lap whose base registers are ``lw_reg`` (weights)
    and ``ls_reg`` (scales): five a super-row, ``ring_issue``'s. Then ``wait_xcnt``: the compiler does not know
    these copies read their operand registers, and may move or reuse one behind them (without the waits a server
    run stopped in a wait that never completed, at 1 row)."""
    G, Gs = bases
    for i in range(n):
        for t in range(2):
            tau = 2 * i + t
            for hf in range(2):
                async_copy16_at(lw_reg, G[tau], (tau * depth + kc) * 1024 + hf * 512)
        async_copy4_at(ls_reg, Gs[i], (i * depth + kc) * 128)
    wait_xcnt()


def ring_prologue_raw(c, srcs):
    """``ring_prologue`` in the lockstep layout."""
    lane = c["lane"]
    n = len(srcs)
    depth = ring_depth(n)
    ring0 = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES
    bases = raw_bases(c, srcs)
    l0w, l0s = fresh(ring0 + lane * 16), fresh(ring0 + RAW_W + lane * 4)
    for kc in range(depth):
        raw_issue(bases, l0w, l0s, kc, n, depth)
    wait_xcnt()


def stream_rows_raw(c, srcs, xcol, col, issued=False, nxt=None, mark=None):
    """``stream_rows_ring`` in the lockstep layout (``K3_RAW``): the same products in the same order."""
    lane = c["lane"]
    n = len(srcs)
    depth = ring_depth(n)
    assert K_CHUNKS % depth == 0, (K_CHUNKS, depth)
    per = 5 * n  # async copies a chunk
    ring0 = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES
    if not issued:
        ring_prologue_raw(c, srcs)
    bases = raw_bases(c, srcs)
    nbases = raw_bases(c, nxt) if nxt is not None else None
    l0w, l0s = fresh(ring0 + lane * 16), fresh(ring0 + RAW_W + lane * 4)
    lr, sr = l0w, l0s  # the lap being read
    lf, sf = (
        pin_add(l0w, -depth * 1024),
        pin_add(l0s, -depth * 128),
    )  # the lap being refilled
    xb0 = fresh(lds_addr(xcol) + (col * XQP + (lane // 16) * 4) * 4)
    xs0 = fresh(lds_addr(xcol) + (XS_BASE + col * XSW) * 4)
    rocdl.sched_barrier(0)
    accs = [[fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(2)] for _ in srcs]
    for kc in range_constexpr(K_CHUNKS):
        if kc and kc % depth == 0:
            # a new lap, before the chunk's loads (ten copies behind the last WMMAs)
            wait_xcnt()
            lr, sr = lf, sf
            if kc + depth < K_CHUNKS:
                lf = pin_add(l0w, -(kc // depth + 1) * depth * 1024)
                sf = pin_add(l0s, -(kc // depth + 1) * depth * 128)
            else:
                lf, sf = l0w, l0s  # the last lap refills the next item's first
        inflight = kc + depth if nxt is not None else min(kc + depth, K_CHUNKS)
        wait_lds_dma(
            (inflight - kc - 1) * per
        )  # this chunk's copies landed (in issue order)
        if mark is not None and kc == K_CHUNKS // 2:
            wstamp(c, mark)  # (trace builds, the per-wave CTAs) half the item streamed
        cur = []
        for i in range(n):
            ts = []
            for t in range(2):
                o = ((2 * i + t) * depth + kc) * 1024
                ts.append(cat([lds_load16(lr + o), lds_load16(lr + (o + 512))]))
            cur.append((ts[0], ts[1], lds_load4(sr + (i * depth + kc) * 128)))
        xb = cat([lds_load16(xb0 + (kc * 32 + 8 * j) * 4) for j in range(4)])
        xs = lds_load4(xs0 + kc * 4)
        for i in range_constexpr(n):
            a0, a1, sa = cur[i]
            accs[i][0] = wmma_s(a0, xb, accs[i][0], sa, xs, 0)
            accs[i][1] = wmma_s(a1, xb, accs[i][1], sa, xs, 1)
        if kc + depth < K_CHUNKS:
            rocdl.s_wait_dscnt(
                0
            )  # the slot's reads are done before the refill overwrites it
            rocdl.sched_barrier(0)
            raw_issue(bases, lf, sf, kc + depth, n, depth)
        elif nxt is not None:
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)
            raw_issue(nbases, lf, sf, kc + depth - K_CHUNKS, n, depth)
    wait_xcnt()  # (the copies' operand registers are free from here)
    return accs


def stream_rows_ring(
    c, srcs, xcol, col, issued=False, nxt=None, mark=None, probe_one=False
):
    """``stream_rows`` through this wave's async LDS ring: as many chunk slots as the ring holds
    in flight, each slot refilled (chunk + depth) right after its WMMAs read it. ``issued``: the
    first slots were issued already (``ring_prologue`` or the previous item's tail); ``nxt``: the
    next item's super-rows, whose first chunks take the slots this item's last chunks free (an
    item's chunk count is a multiple of the depth, so its chunk j sits in slot j % depth)."""
    if K3_RAW and K3_PROBE_TILES == 4:
        return stream_rows_raw(c, srcs, xcol, col, issued, nxt, mark)
    n = len(srcs)
    srcs, nxt = u_srcs(srcs), u_srcs(nxt)  # (uniform bases, once an item)
    sb = n * SR_SLOT_BYTES
    depth = ring_depth(n)
    assert K_CHUNKS % depth == 0, (K_CHUNKS, depth)
    per = 5 * n  # async copies a chunk
    ring0 = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES
    accs = [[fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(2)] for _ in srcs]
    if not issued:
        ring_prologue(c, srcs)
    dst = [ring_dst(c, n, sn) for sn in range(depth)] if K3_GVS else [None] * depth
    for kc in range_constexpr(K_CHUNKS):
        issued = kc + depth if nxt is not None else min(kc + depth, K_CHUNKS)
        wait_lds_dma(
            (issued - kc - 1) * per
        )  # this chunk's copies landed (in issue order)
        slot = ring0 + (kc % depth) * sb
        if mark is not None and kc == K_CHUNKS // 2:
            wstamp(c, mark)  # (trace builds, the per-wave CTAs) half the item streamed
        cur = ring_read(c, n, slot)
        xb, xs = xq_b_at(c, xcol, col, kc)
        for i in range_constexpr(n):
            a0, a1, sa = cur[i]
            accs[i][0] = wmma_s(a0, xb, accs[i][0], sa, xs, 0)
            if not probe_one:
                accs[i][1] = wmma_s(a1, xb, accs[i][1], sa, xs, 1)
        if kc + depth < K_CHUNKS:
            rocdl.s_wait_dscnt(
                0
            )  # the slot's reads are done before the refill overwrites it
            rocdl.sched_barrier(0)
            ring_issue(c, srcs, kc + depth, slot, dst[kc % depth])
        elif nxt is not None:
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)
            ring_issue(c, nxt, kc + depth - K_CHUNKS, slot, dst[kc % depth])
    return accs


def stream_any(c, srcs, xcol, col, depth):
    """The wave's full-K product of the super-rows ``srcs`` [(w, s, srow)], by the build's mode."""
    if const_expr(K3_WAVE == 2):
        return stream_rows_ring(c, srcs, xcol, col)
    return stream_rows(
        c, [(rsrc(w), rsrc(s), sr) for (w, s, sr) in srcs], xcol, col, depth
    )


@flyc.jit
def ug_item_w(c, q, n_tiles):
    """ug item ``q`` (tile q / UG_T, mid block q % UG_T: interleaved w13 rows 64 blk ..) on this
    wave over the full K, then its MX FP8 mid block a column (``put_mid_block``'s numerics:
    silu(g) u in fp32, the block's E8M0 RoundUp scale, E4M3 codes). Lane (column N = lane % 16,
    half h) holds rows M = 8 h .. 8 h + 7 of each tile: mids 16 sr + 8 t + 4 h + j (j < 4)."""
    lane, mid, put = c["lane"], c["mid"], c["put"]
    blk = q % UG_T
    e, col0, n = tile_info(c, q // UG_T, n_tiles)
    w, s = w_addrs(c, e, False)
    col = lane_token(c, e, col0, n)
    accs = stream_any(
        c, [(w, s, blk * 64), (w, s, blk * 64 + 32)], c["xcol"], col, K3W_DEPTH_UG
    )
    acts = []
    for sr in range_constexpr(2):
        for t in range_constexpr(2):
            a = accs[sr][t]
            acts.append(
                [
                    silu_mul(fx.Float32(a[2 * j]), fx.Float32(a[2 * j + 1]))
                    for j in range(4)
                ]
            )
    am = abs(acts[0][0])
    for v in acts:
        for x in v:
            am = fx.max(am, abs(x))
    am = butterfly(am, (16,), fx.max)
    e8, inv = mx8_scale(am)
    N, h = lane % 16, lane // 16
    if N < n:
        base = (pick_at(c, e, col0 + N) * UG_T + blk) * MID_W
        for k in range_constexpr(4):  # (sr, t) = (k // 2, k % 2): word 4 sr + 2 t + h
            v = acts[k]
            word = fp8_byte(v[0] * inv) | (fp8_byte(v[1] * inv) << 8) | (fp8_byte(v[2] * inv) << 16) | (fp8_byte(v[3] * inv) << 24)  # fmt: skip
            put(mid, base + (2 * k + h), word)
        if h == 0:
            put(mid, base + 8, e8)


@flyc.jit
def stage_ug_w(c, n_tiles):
    """This CTA's contiguous range of the ug items, item q0 + w + WAVES i on wave w; the step's
    MX FP8 x staged once (``xq_pre``)."""
    bid, wave = c["bid"], c["wave"]
    TU = n_tiles * UG_T
    q0, n_items = item_range(bid, TU)
    if const_expr(not c.get("xq_staged", False)):
        stage_xq_all(c, xq_all_loads(c))
    for i in range_constexpr(K3W_ITEMS):
        q = q0 + wave + WAVES * i
        if q < q0 + n_items:
            ug_item_w(c, q, n_tiles)


@flyc.jit
def dn_item_w(c, q, n_tiles, xcol):
    """dn item ``q`` (tile q / DN_T, w2 rows 32 (q % DN_T) ..) on this wave over the full K
    against the tile's mids staged in ``xcol``; its y rows (bf16) to ``k3_y`` and each column's
    sum of squares of v = bf16(bf16(y) pen_w) over the 32 rows to ``k3_ss``. Lane (column N,
    half h) holds rows 16 t + 8 h + k of the item."""
    lane, put, put_bf = c["lane"], c["put"], c["put_bf"]
    blk = q % DN_T
    e, col0, n = tile_info(c, q // DN_T, n_tiles)
    w, s = w_addrs(c, e, True)
    N, h = lane % 16, lane // 16
    col = fx.min(N, n - 1)
    accs = stream_any(c, [(w, s, blk * 32)], xcol, col, K3W_DEPTH_DN)[0]
    ss = fx.Float32(0.0)
    for t in range_constexpr(2):
        a = accs[t]
        for j in range_constexpr(4):
            pw = bf2_f32(fx.ptr_load(c["pw"] + (blk * 16 + 8 * t + 4 * h + j)))
            v0 = bf16_round(bf16_round(fx.Float32(a[2 * j])) * pw[0])
            v1 = bf16_round(bf16_round(fx.Float32(a[2 * j + 1])) * pw[1])
            ss = ss + (v0 * v0 + v1 * v1)
    ss = butterfly(ss, (16,))
    if N < n:
        p = pick_at(c, e, col0 + N)
        for t in range_constexpr(2):
            a = accs[t]
            for g in range_constexpr(
                2
            ):  # 128-bit stores (a 256-bit buffer store does not select on gfx1250)
                put_bf(c["y"], p * MOE_HIDDEN + blk * 32 + 16 * t + 8 * h + 4 * g, [fx.Float32(a[4 * g + k]) for k in range(4)])  # fmt: skip
        if h == 0:
            put(c["ss"], p * DN_T + blk, ss)


@flyc.jit
def stage_dn_w(c, n_tiles):
    """This CTA's contiguous range of the dn items (reversed CTA order, as ``stage_dn``), item
    q0 + w + WAVES i on wave w. The range spans at most two column tiles: the first's mids
    staged in ``xcol``, the second's in ``red`` (free while the experts run)."""
    bid, wave = c["bid"], c["wave"]
    if const_expr(K3_WAVE == 2):
        # BLOCKS // n_tiles CTAs a tile, the tile's items strided over them: one staged tile a CTA
        cpt = fx.max(fx.Int32(BLOCKS) // fx.max(n_tiles, fx.Int32(1)), fx.Int32(1))
        ti, sub_ = bid // cpt, bid % cpt
        lds_barrier()  # every wave is done with the staged x
        if ti < n_tiles:
            e, col0, n = tile_info(c, ti, n_tiles)
            stage_mid_q(c, e, col0, n)
            for i in range_constexpr(K3W_ITEMS):
                blk = sub_ + cpt * (wave + WAVES * i)
                if blk < DN_T:
                    dn_item_w(c, ti * DN_T + blk, n_tiles, c["xcol"])
        return
    TD = n_tiles * DN_T
    q0, n_items = item_range(BLOCKS - 1 - bid, TD)
    lds_barrier()  # every wave is done with the staged x
    ta = fx.max(q0, fx.Int32(0)) // DN_T
    tb = fx.max(q0 + n_items - 1, fx.Int32(0)) // DN_T
    cb = dict(c)
    cb["xcol"] = c["red"]
    if (
        n_items > 0
    ):  # (a CTA without items polls no tile: an empty plan's mids were never written)
        ea, ca, na = tile_info(c, ta, n_tiles)
        stage_mid_q(c, ea, ca, na)
        if tb != ta:
            eb, cbb, nb = tile_info(c, tb, n_tiles)
            stage_mid_q(cb, eb, cbb, nb)
    for i in range_constexpr(K3W_ITEMS):
        q = q0 + wave + WAVES * i
        if q < q0 + n_items:
            if q // DN_T == ta:
                dn_item_w(c, q, n_tiles, c["xcol"])
            else:
                dn_item_w(c, q, n_tiles, c["red"])


def ug_srcs(c, q, n_tiles):
    """ug item ``q``'s (expert, first column, columns, block, super-rows [(w, s, srow)])."""
    blk = q % UG_T
    e, col0, n = tile_info(c, q // UG_T, n_tiles)
    w, s = w_addrs(c, e, False)
    return e, col0, n, blk, [(w, s, blk * 64), (w, s, blk * 64 + 32)]


def dn_srcs(c, ti, blk, n_tiles):
    """dn item (tile ``ti``, block ``blk``)'s (expert, first column, columns, super-rows)."""
    e, col0, n = tile_info(c, ti, n_tiles)
    w, s = w_addrs(c, e, True)
    return e, col0, n, [(w, s, blk * 32)]


@flyc.jit
def ug_epilogue(c, accs, e, col0, n, blk):
    """An ug item's MX FP8 mid block a column from its accumulators (``ug_item_w``'s epilogue)."""
    lane, mid, put = c["lane"], c["mid"], c["put"]
    acts = []
    for sr in range_constexpr(2):
        for t in range_constexpr(2):
            a = accs[2 * sr + t]
            acts.append(
                [
                    silu_mul(fx.Float32(a[2 * j]), fx.Float32(a[2 * j + 1]))
                    for j in range(4)
                ]
            )
    am = abs(acts[0][0])
    for v in acts:
        for x in v:
            am = fx.max(am, abs(x))
    am = butterfly(am, (16,), fx.max)
    e8, inv = mx8_scale(am)
    N, h = lane % 16, lane // 16
    if N < n:
        base = (pick_at(c, e, col0 + N) * UG_T + blk) * MID_W
        for k in range_constexpr(4):  # (sr, t) = (k // 2, k % 2): word 4 sr + 2 t + h
            v = acts[k]
            word = fp8_byte(v[0] * inv) | (fp8_byte(v[1] * inv) << 8) | (fp8_byte(v[2] * inv) << 16) | (fp8_byte(v[3] * inv) << 24)  # fmt: skip
            put(mid, base + (2 * k + h), word)
        if h == 0:
            put(mid, base + 8, e8)


def tile_ss(c, a, blk, t):
    """A dn tile's (rows 32 blk + 16 t ..) sum of squares of v = bf16(bf16(y) pen_w) a column: this lane's eight
    rows, then the column's two lanes."""
    h = c["lane"] // 16
    ss = fx.Float32(0.0)
    for j in range(4):
        pw = bf2_f32(fx.ptr_load(c["pw"] + (blk * 16 + 8 * t + 4 * h + j)))
        v0 = bf16_round(bf16_round(fx.Float32(a[2 * j])) * pw[0])
        v1 = bf16_round(bf16_round(fx.Float32(a[2 * j + 1])) * pw[1])
        ss = ss + (v0 * v0 + v1 * v1)
    return butterfly(ss, (16,))


@flyc.jit
def dn_epilogue(c, accs, e, col0, n, blk):
    """A dn item's y rows and its columns' sums of squares (``dn_item_w``'s epilogue)."""
    lane, put, put_bf = c["lane"], c["put"], c["put_bf"]
    N, h = lane % 16, lane // 16
    # (a tile's sum of squares, then the two tiles': the tile form's order, ``stage_dn_q``)
    ss = tile_ss(c, accs[0], blk, 0) + tile_ss(c, accs[1], blk, 1)
    if N < n:
        p = pick_at(c, e, col0 + N)
        for t in range_constexpr(2):
            a = accs[t]
            for g in range_constexpr(2):  # 128-bit stores
                put_bf(c["y"], p * MOE_HIDDEN + blk * 32 + 16 * t + 8 * h + 4 * g, [fx.Float32(a[4 * g + k]) for k in range(4)])  # fmt: skip
        if h == 0:
            put(c["ss"], p * DN_T + blk, ss)


# ``SGLANG_AMD_MONO_PF_EXP=1`` (ring build, merged launch): K3 waits for the routing alone, builds the plan,
# issues each wave's first ug item's ring prologue and prefetches its ug items' next chunks into GL2, and only
# then waits for the expert input. The routing lands right after the router pass; the expert input waits for
# the pooled rows' all-gather, later, while HBM idles. PF_EXP_CH: chunks an item past the ring's at
# most; PF_EXP_MB: a rank's prefetch budget (the items' chunk count shrinks to keep within it, so the lines
# stay cached until the ug reads them).
PF_EXP = os.environ.get("SGLANG_AMD_MONO_PF_EXP", "0") == "1"
PF_EXP_CH = min(int(os.environ.get("SGLANG_AMD_MONO_PF_EXP_CH", "8")), K_CHUNKS - 4)
PF_EXP_MB = int(os.environ.get("SGLANG_AMD_MONO_PF_EXP_MB", "48"))


@flyc.jit
def stage_ug_pre(c, n_tiles):
    """``PF_EXP``: this wave's first ug item's ring prologue (``stage_ug_r`` then skips its own) and GL2
    prefetches of chunks depth .. depth + f of each of its ug items, f = min(PF_EXP_CH, the budget over the
    rank's items). An ug item is 4 consecutive 16-row tiles of the expert's w13 (its two super-rows), a
    chunk 1 KB of each tile, and a 128 B scale line of each super-row."""
    bid, wave, lane = c["bid"], c["wave"], c["lane"]
    TU = n_tiles * UG_T
    q0, n_items = item_range(bid, TU)
    end = q0 + n_items
    last = fx.max(TU - 1, fx.Int32(0))
    qs = [q0 + wave_slot(c) + WAVES * i for i in range(K3W_ITEMS)]
    ring_prologue(c, ug_srcs(c, fx.min(qs[0], last), n_tiles)[4])
    depth = ring_depth(2)
    f = fx.Int32(PF_EXP_MB << 20) // fx.max(TU * (2 * SR_SLOT_BYTES), fx.Int32(1))
    f = fx.min(f, fx.Int32(PF_EXP_CH))
    # a lane's line offsets, fixed over the run (the item's bases in SGPRs, the chunks in the offset field):
    # the 4 tiles x 2 KB of a chunk pair as two instructions, and a super-row's scale line of chunk depth + lane
    woffs = [
        ((h * WAVE + lane) // 16) * TILE_BYTES + ((h * WAVE + lane) % 16) * 128
        for h in range(2)
    ]
    soff = (
        fx.min(lane, fx.max(f - 1, fx.Int32(0))) * 128
    )  # (lanes past f repeat the last)
    for i in range_constexpr(K3W_ITEMS):
        if (qs[i] < end) & (f > 0):
            srcs = ug_srcs(c, qs[i], n_tiles)[4]
            w, s, srow0 = srcs[0]
            for _, s_, srow in srcs:
                l2_prefetch_gvs(
                    uniform64(
                        s_ + fx.Int64((srow // 32) * SR_SCALE_BYTES + depth * 128)
                    ),
                    soff,
                )
            wb = uniform64(w + fx.Int64((srow0 // 16) * TILE_BYTES + depth * 1024))
            for g in range_constexpr(0, PF_EXP_CH, 2):
                if g < f:
                    for off in woffs:
                        l2_prefetch_gvs(wb, off, g * 1024)


@flyc.jit
def stage_ug_r(c, n_tiles, issued=False):
    """The ring build's ug: this CTA's contiguous range, item q0 + w + WAVES i on wave w; the
    first item's weights stream while the x is staged, each item's tail streams the next one's
    first chunks (items past the range read clamped, valid rows: no result, drained at the end).
    ``issued``: ``stage_ug_pre`` issued the first item's prologue."""
    bid, wave = c["bid"], c["wave"]
    TU = n_tiles * UG_T
    q0, n_items = item_range(bid, TU)
    end = q0 + n_items
    last = fx.max(TU - 1, fx.Int32(0))
    qs = [q0 + wave_slot(c) + WAVES * i for i in range(K3W_ITEMS)]
    if const_expr(not issued):
        ring_prologue(
            c,
            ug_srcs(c, fx.min(qs[0], last), n_tiles)[4][
                : 1 if K3_PROBE_TILES < 4 else 2
            ],
        )
    if const_expr(not c.get("xq_staged", False)):
        stage_xq_all(c, xq_all_loads(c))
    for i in range_constexpr(K3W_ITEMS):
        if qs[i] < end:
            e, col0, n, blk, srcs = ug_srcs(c, qs[i], n_tiles)
            nxt = (
                ug_srcs(c, fx.min(qs[i + 1], last), n_tiles)[4]
                if i + 1 < K3W_ITEMS
                else None
            )
            if i == 0:
                wstamp(
                    c, 0
                )  # (trace builds, the per-wave CTAs) the wave's first ug item begins (x staged)
            if const_expr(K3_PROBE_TILES < 4):
                # (timing probe: one super-row of the item, its next item's too; the mids are wrong)
                pa = stream_rows_ring(c, srcs[:1], c["xcol"], lane_token(c, e, col0, n), issued=True, nxt=None if nxt is None else nxt[:1], mark=1 if i == 0 else None, probe_one=K3_PROBE_TILES == 1)  # fmt: skip
                accs = [pa[0], pa[0]]
            else:
                accs = stream_rows_ring(
                    c,
                    srcs,
                    c["xcol"],
                    lane_token(c, e, col0, n),
                    issued=True,
                    nxt=nxt,
                    mark=1 if i == 0 else None,
                )
            if i == 0:
                wstamp(c, 2)  # ... its accumulators done
            ug_epilogue(
                c, [accs[0][0], accs[0][1], accs[1][0], accs[1][1]], e, col0, n, blk
            )
            if i == 0:
                wstamp(c, 3)  # ... its mid blocks stored
    wait_lds_dma(0)  # a dead item's prefetched chunks land before the ring is reused


# SGLANG_AMD_MONO_K3_UGQ=1 (default, ring build): a ug item's four 16-row tiles go to four waves. A wave's stream is
# bound by its async copies (a wave issues only so many a microsecond, fewer when every wave of the CU streams): an
# item on one wave is 320 copies however idle the CU's other waves are (with few experts a rank most are). A tile
# is 64 weight copies and 8 for its super-row's scale words (all 32 chunks at once). A CTA works its
# items four a round, item m of the round on the waves of slots 4 m .. 4 m + 3 (``wave_slot``); the block's E8M0
# scale needs the max over the four tiles' mids, exchanged through LDS at the round's barrier. The accumulators,
# the mids and their codes are the item form's, bit for bit.
K3_UGQ = os.environ.get("SGLANG_AMD_MONO_K3_UGQ", "1") == "1"
# a CTA takes the tile form while its items fit this many (rounds of four), else the item form: a tile's stream
# costs more instructions an item (the x operand's loads a tile, the rounds' barriers), which loses once most of the
# CU's waves stream anyway: the tile form wins at 1 and 2 rows, is even at 4 and loses from 8 rows on.
# Default (-1): up to 12 items a CTA at 1 and 2 rows, the item form alone above.
UGQ_MAX = int(os.environ.get("SGLANG_AMD_MONO_K3_UGQ_MAX", "-1"))
assert UGQ_MAX % 4 == 0 or UGQ_MAX == -1


def ugq_max(tokens: int) -> int:
    """Items a CTA up to which it takes the tile form at ``tokens`` rows."""
    return UGQ_MAX if UGQ_MAX >= 0 else (12 if tokens <= 2 else 0)


UQ_DEPTH = 4  # weight chunks of its tile a wave holds (1 KB each: the lo / hi halves)
UQ_SC = (
    UQ_DEPTH * 1024
)  # then the super-row's scale words of every chunk (K_CHUNKS x 128 B)
assert (
    UQ_SC + K_CHUNKS * 128 <= RING_W_BYTES
    and K_CHUNKS % UQ_DEPTH == 0
    and (K_CHUNKS * 128) % 512 == 0
)


def ugq_rounds(tokens: int) -> int:
    """Rounds of four ug items a CTA in the tile form at most at ``tokens`` rows."""
    experts = min(LOCAL_EXPERTS, tokens * TOP_K)
    return min(cdiv(cdiv(experts * UG_T, BLOCKS), 4), ugq_max(tokens) // 4)


def tile_addr(w, srow, t):
    """The 16-row tile ``t`` of super-row ``srow``'s weight base (wave-uniform with ``K3_GVS``)."""
    b = w + fx.Int64((srow // 16 + t) * TILE_BYTES)
    return uniform64(b) if K3_GVS else b


def tile_issue(c, wt, kc, slot):
    """Async copies of chunk ``kc`` of the tile at ``wt`` (``tile_addr``) into the 1 KB ring slot at LDS byte
    address ``slot``: ``ring_issue``'s lo and hi halves of that tile."""
    lane = c["lane"]
    lw = fx.Int64(fx.Uint32((lane % 16) * 16 + (lane // 16) * 256))
    for hf in range(2):
        async_copy16(
            wt + lw + fx.Int64((4 * kc + 2 * hf) * 256), slot + hf * 512 + lane * 16
        )


def tile_prologue(c, w, s, srow, t):
    """A tile's stream begins: its super-row's scale words of every chunk (4 KB at UQ_SC of the wave's ring, eight
    copies), then its first UQ_DEPTH chunks."""
    lane = c["lane"]
    ring0 = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES
    sbase = s + fx.Int64((srow // 32) * SR_SCALE_BYTES)
    if K3_GVS:
        sbase = uniform64(sbase)
    ll = fx.Int64(fx.Uint32(lane * 16))
    for j in range(K_CHUNKS * 128 // 512):
        async_copy16(
            sbase + ll + fx.Int64(j * 512), ring0 + UQ_SC + j * 512 + lane * 16
        )
    wt = tile_addr(w, srow, t)
    for kc in range(UQ_DEPTH):
        tile_issue(c, wt, kc, ring0 + kc * 1024)


def stream_tile(c, w, srow, t, xcol, col):
    """The full-K product of one 16-row tile (``tile_prologue`` issued its first chunks) with LDS column ``col`` of
    ``xcol``: its fp32 accumulator, chunk by chunk as ``stream_rows_ring`` accumulates that tile. The tile's scale
    words are lanes 16 t .. of the super-row's: read into lanes 0 .. 15 (the WMMA's row-0 scale operand).

    A chunk is the instructions that do the work and nothing else: its loads take a pinned base and a constant
    offset; its refill copies take the one base register of the ring's lap (the LDS address of chunk k is the
    lap's base + 1024 k: the copy instruction adds one offset to both of its addresses, so a refill whose LDS and
    global offsets move together needs no address arithmetic) and this lane's pinned global base. A VALU write to
    a WMMA's operand register right behind it costs eight no-ops: nothing but the copies follows the WMMA."""
    lane = c["lane"]
    ring0 = lds_addr(c["ring"]) + c["wave"] * RING_W_BYTES
    wt = tile_addr(w, srow, t)
    ga = pin64(wt + fx.Int64(fx.Uint32((lane % 16) * 16 + (lane // 16) * 256)))
    lap0 = fresh(ring0 + lane * 16)
    lb = lap0  # the lap's base: chunk k of lap i (chunks UQ_DEPTH i ..) is at lap0 - UQ_DEPTH 1024 i + 1024 k
    ln = pin_add(lap0, -UQ_DEPTH * 1024)
    sbase = fresh(ring0 + UQ_SC + ((lane + 16 * t) % WAVE) * 4)
    xbase = fresh(lds_addr(xcol) + (col * XQP + (lane // 16) * 4) * 4)
    xsbase = fresh(lds_addr(xcol) + (XS_BASE + col * XSW) * 4)
    rocdl.sched_barrier(0)
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    for kc in range_constexpr(K_CHUNKS):
        issued = min(kc + UQ_DEPTH, K_CHUNKS)
        if kc and kc % UQ_DEPTH == 0:
            # the next lap's base, before this chunk's loads (not behind a WMMA)
            wait_xcnt()
            lb = ln
            ln = pin_add(lap0, -(kc // UQ_DEPTH + 1) * (UQ_DEPTH * 1024))
        wait_lds_dma(
            (issued - kc - 1) * 2
        )  # this chunk's copies (and the scales') landed (in issue order)
        a = cat([lds_load16(lb + kc * 1024), lds_load16(lb + (kc * 1024 + 512))])
        sa = lds_load4(sbase + kc * 128)
        xb = cat([lds_load16(xbase + (kc * 32 + 8 * j) * 4) for j in range(4)])
        xs = lds_load4(xsbase + kc * 4)
        acc = wmma_s(a, xb, acc, sa, xs, 0)
        if kc + UQ_DEPTH < K_CHUNKS:
            rocdl.s_wait_dscnt(
                0
            )  # the slot's reads are done before the refill overwrites it
            rocdl.sched_barrier(0)
            kn = kc + UQ_DEPTH
            for hf in range(2):
                async_copy16_at(ln, ga, kn * 1024 + hf * 512)
            wait_xcnt()  # (as ``raw_issue``)
    return acc


def ugq_item(c, r, n_tiles):
    """The wave's tile of round ``r``: (live, expert, first column, columns, block, w, s, super-row, tile of the
    super-row, tile of the item, the item's slot in the round); the item clamped to a valid one past the range."""
    TU = n_tiles * UG_T
    q0, n_items = item_range(c["bid"], TU)
    ws = wave_slot(c)
    m, tt = ws // 4, ws % 4
    qi = 4 * r + m
    q = fx.max(fx.min(q0 + qi, TU - 1), fx.Int32(0))
    e, col0, n, blk, srcs = ug_srcs(c, q, n_tiles)
    w, s, srow0 = srcs[0]
    return qi < n_items, e, col0, n, blk, w, s, srow0 + 32 * (tt // 2), tt % 2, tt, m


@flyc.jit
def ugq_round(c, r, n_tiles, rounds):
    """Round ``r`` of the tile form's ug on this CTA (a CTA with an item in it): the wave's tile streamed, its
    four mids a lane kept in ``red`` and its columns' max in ``misc`` (a parity a round); after the barrier the
    block's max over the item's four tiles, the E8M0 scale and the tile's codes (``ug_epilogue``'s numerics)."""
    lane, wave, mid, put = c["lane"], c["wave"], c["mid"], c["put"]
    red, misc = c["red"], c["misc"]
    live, e, col0, n, blk, w, s, srow, t, tt, m = ugq_item(c, r, n_tiles)
    nxt = (
        ugq_item(c, r + 1, n_tiles) if r + 1 < rounds else None
    )  # (a wave live next round is live in this one)
    N, h = lane % 16, lane // 16
    par = (r % 2) * (WAVES * TILE_COLS)
    V4 = _v(4)
    if live:
        a = stream_tile(c, w, srow, t, c["xcol"], lane_token(c, e, col0, n))
        acts = [
            silu_mul(fx.Float32(a[2 * j]), fx.Float32(a[2 * j + 1])) for j in range(4)
        ]
        am = abs(acts[0])
        for x in acts[1:]:
            am = fx.max(am, abs(x))
        am = butterfly(am, (16,), fx.max)
        fx.ptr_store(
            fx.Vector.from_elements(acts, fx.Float32), red + (wave * WAVE + lane) * 4
        )
        if h == 0:
            fx.ptr_store(am, misc + (par + (m * 4 + tt) * TILE_COLS + N))
        if const_expr(nxt is not None):
            if nxt[0]:
                tile_prologue(
                    c, nxt[5], nxt[6], nxt[7], nxt[8]
                )  # the next round's tile streams over the barrier
    lds_barrier()
    if live:
        amx = fx.ptr_load(misc + (par + (m * 4) * TILE_COLS + N))
        for k in range_constexpr(1, 4):
            amx = fx.max(amx, fx.ptr_load(misc + (par + (m * 4 + k) * TILE_COLS + N)))
        e8, inv = mx8_scale(amx)
        v = fx.Vector(fx.ptr_load(red + (wave * WAVE + lane) * 4, result_type=V4))
        if N < n:
            base = (pick_at(c, e, col0 + N) * UG_T + blk) * MID_W
            word = fp8_byte(fx.Float32(v[0]) * inv) | (fp8_byte(fx.Float32(v[1]) * inv) << 8) | (fp8_byte(fx.Float32(v[2]) * inv) << 16) | (fp8_byte(fx.Float32(v[3]) * inv) << 24)  # fmt: skip
            put(mid, base + (2 * tt + h), word)
            if (tt == 0) & (h == 0):
                put(mid, base + 8, e8)


@flyc.jit
def stage_ug_a(c, n_tiles):
    """The ring build's ug by the CTA's item count: the tile form up to UGQ_MAX items, else the item form."""
    _, n_items = item_range(c["bid"], n_tiles * UG_T)
    if n_items <= ugq_max(c["S"]):
        stage_ug_q(c, n_tiles)
    else:
        stage_ug_r(c, n_tiles)


@flyc.jit
def stage_ug_q(c, n_tiles):
    """The tile form's ug (``K3_UGQ``): this CTA's contiguous range of items, four a round."""
    S = c["S"]
    TU = n_tiles * UG_T
    _, n_items = item_range(c["bid"], TU)
    rounds = ugq_rounds(S)
    first = ugq_item(c, 0, n_tiles)
    if first[0]:
        tile_prologue(
            c, first[5], first[6], first[7], first[8]
        )  # streams while the x is staged
    if const_expr(not c.get("xq_staged", False)):
        stage_xq_all(c, xq_all_loads(c))
    for r in range_constexpr(rounds):
        if 4 * r < n_items:
            ugq_round(c, r, n_tiles, rounds)


# ``SGLANG_AMD_MONO_PF_DN=1`` (ring build): a dn wave prefetches chunks depth .. depth + PF_DN_CH of its items
# into GL2 with its ring prologue, before it waits for the tile's mids (the ug items of the tile, other CTAs'):
# that wait and the ug tails it overlaps leave HBM time spare.
PF_DN = os.environ.get("SGLANG_AMD_MONO_PF_DN", "0") == "1"
PF_DN_CH = min(int(os.environ.get("SGLANG_AMD_MONO_PF_DN_CH", "12")), K_CHUNKS - 8)


def dn_prefetch(c, srcs):
    """GL2 prefetches of chunks depth .. depth + PF_DN_CH of a dn item (one super-row: two 16-row tiles of
    w2, a chunk 1 KB of each, and the super-row's 128 B scale line a chunk), four chunks an instruction pair."""
    lane = c["lane"]
    w, s, srow = srcs[0]
    depth = ring_depth(1)
    l2_prefetch_gvs(
        uniform64(s + fx.Int64((srow // 32) * SR_SCALE_BYTES + depth * 128)),
        fx.min(lane, fx.Int32(PF_DN_CH - 1)) * 128,
    )
    # chunks depth + g, + 1 of both tiles: 2 x 2 KB = 32 lines (a lane: tile l / 16, line l % 16), the base in
    # SGPRs and the lane's offset fixed, the chunk pair in the offset field
    wb = uniform64(w + fx.Int64((srow // 16) * TILE_BYTES + depth * 1024))
    off = (lane // 16) * TILE_BYTES + (lane % 16) * 128
    for g in range(0, PF_DN_CH, 2):
        l2_prefetch_gvs(wb, off, g * 1024)


@flyc.jit
def stage_dn_r(c, n_tiles):
    """The ring build's dn: BLOCKS // n_tiles CTAs a tile, the tile's items strided over them
    (one staged tile a CTA); the first item's weights stream while the mids are polled and
    staged, each item's tail streams the next one's first chunks."""
    bid, wave = c["bid"], c["wave"]
    cpt = fx.max(fx.Int32(BLOCKS) // fx.max(n_tiles, fx.Int32(1)), fx.Int32(1))
    ti, sub_ = bid // cpt, bid % cpt
    tic = fx.min(ti, fx.max(n_tiles - 1, fx.Int32(0)))
    blks = [sub_ + cpt * (wave_slot(c) + WAVES * i) for i in range(K3W_ITEMS)]
    lds_barrier()  # every wave is done with the staged x (and its ring)
    ring_prologue(c, dn_srcs(c, tic, fx.min(blks[0], fx.Int32(DN_T - 1)), n_tiles)[3])
    if const_expr(PF_DN):
        for i in range_constexpr(K3W_ITEMS):
            if (ti < n_tiles) & (blks[i] < DN_T):
                dn_prefetch(c, dn_srcs(c, ti, blks[i], n_tiles)[3])
    if ti < n_tiles:
        e, col0, n = tile_info(c, ti, n_tiles)
        stage_mid_q(c, e, col0, n)
        col = fx.min(c["lane"] % 16, n - 1)
        for i in range_constexpr(K3W_ITEMS):
            if blks[i] < DN_T:
                _, _, _, srcs = dn_srcs(c, ti, blks[i], n_tiles)
                nxt = (
                    dn_srcs(c, ti, fx.min(blks[i + 1], fx.Int32(DN_T - 1)), n_tiles)[3]
                    if i + 1 < K3W_ITEMS
                    else None
                )
                accs = stream_rows_ring(c, srcs, c["xcol"], col, issued=True, nxt=nxt)[
                    0
                ]
                dn_epilogue(c, [accs[0], accs[1]], e, col0, n, blks[i])
    wait_lds_dma(0)


# SGLANG_AMD_MONO_K3_DNQ=1 (default, ring build): a dn item's two 16-row tiles go to two waves (as ``K3_UGQ``) while
# the CTA's items fit DNQ_MAX (two tiles a wave pair: one round). A tile's y rows are its own; the item's sum of
# squares is the two tiles', exchanged through LDS at one barrier (``tile_ss``'s order: the item form's bits).
K3_DNQ = os.environ.get("SGLANG_AMD_MONO_K3_DNQ", "1") == "1"
# (Default (-1): up to 8 items a CTA at 1 and 2 rows, the item form alone above: no gain measured there.)
DNQ_MAX = int(os.environ.get("SGLANG_AMD_MONO_K3_DNQ_MAX", "-1"))
assert -1 <= DNQ_MAX <= WAVES // 2


def dnq_max(tokens: int) -> int:
    """Items a CTA up to which it takes the dn tile form at ``tokens`` rows."""
    return DNQ_MAX if DNQ_MAX >= 0 else (8 if tokens <= 2 else 0)


@flyc.jit
def stage_dn_q(c, n_tiles):
    """The tile form's dn: the CTA's items (``stage_dn_r``'s: BLOCKS // n_tiles CTAs a tile, the tile's items
    strided over them), item k's tile t on the wave of slot 2 k + t; one round."""
    bid, wave, lane = c["bid"], c["wave"], c["lane"]
    put, put_bf, misc = c["put"], c["put_bf"], c["misc"]
    cpt = fx.max(fx.Int32(BLOCKS) // fx.max(n_tiles, fx.Int32(1)), fx.Int32(1))
    ti, sub_ = bid // cpt, bid % cpt
    tic = fx.min(ti, fx.max(n_tiles - 1, fx.Int32(0)))
    ws = wave_slot(c)
    k, t = ws // 2, ws % 2
    blk = sub_ + cpt * k
    live = (ti < n_tiles) & (blk < DN_T)
    blkc = fx.min(blk, fx.Int32(DN_T - 1))
    N, h = lane % 16, lane // 16
    lds_barrier()  # every wave is done with the staged x (and its ring)
    e, col0, n, srcs = dn_srcs(c, tic, blkc, n_tiles)
    w, s, srow = srcs[0]
    if live:
        tile_prologue(c, w, s, srow, t)  # streams while the mids are polled and staged
    if ti < n_tiles:
        stage_mid_q(c, e, col0, n)
        if live:
            a = stream_tile(c, w, srow, t, c["xcol"], fx.min(lane % 16, n - 1))
            sq = tile_ss(c, a, blkc, t)
            if h == 0:
                fx.ptr_store(sq, misc + ((k * 2 + t) * TILE_COLS + N))
            if N < n:
                p = pick_at(c, e, col0 + N)
                for g in range_constexpr(2):  # 128-bit stores
                    put_bf(c["y"], p * MOE_HIDDEN + blkc * 32 + 16 * t + 8 * h + 4 * g, [fx.Float32(a[4 * g + q]) for q in range(4)])  # fmt: skip
        lds_barrier()
        if live & (t == 0) & (h == 0) & (N < n):
            ss = fx.ptr_load(misc + ((k * 2) * TILE_COLS + N)) + fx.ptr_load(
                misc + ((k * 2 + 1) * TILE_COLS + N)
            )
            put(c["ss"], pick_at(c, e, col0 + N) * DN_T + blkc, ss)


@flyc.jit
def stage_dn_a(c, n_tiles):
    """The ring build's dn by the CTA's item count: the tile form up to DNQ_MAX items, else the item form."""
    bid = c["bid"]
    cpt = fx.max(fx.Int32(BLOCKS) // fx.max(n_tiles, fx.Int32(1)), fx.Int32(1))
    sub_ = bid % cpt
    n_dn = (
        fx.Int32(DN_T - 1) - sub_
    ) // cpt + 1  # the CTA's items: blocks sub_, sub_ + cpt, .. below DN_T
    if n_dn <= dnq_max(c["S"]):
        stage_dn_q(c, n_tiles)
    else:
        stage_dn_r(c, n_tiles)


@flyc.jit
def fin_push_first(c, t, col, vals, act, owner, rs):
    """The fin task's first push form: an active thread stores its columns to the ranks in turn (itself too);
    ``rs``: to the columns' owner only (the reduce-scatter half)."""
    W, rank, put_bf = c["W"], c["rank"], c["put_bf"]
    if act:
        if const_expr(rs):
            ob = c["peer_ffn"][0].value
            for pe in range_constexpr(1, W):
                ob = (owner == pe).select(c["peer_ffn"][pe].value, ob)
            put_bf(preg(ob, 0, "ffn"), (rank * MAX_TOKENS + t) * MOE_HIDDEN + col, vals)
        else:
            for pe in range_constexpr(W):
                put_bf(
                    c["peer_ffn"][pe], (rank * MAX_TOKENS + t) * MOE_HIDDEN + col, vals
                )


@flyc.jit
def fin_push_all(c, t, part, vals, act, NV: int, AR_COLS: int):
    """The fin task's partial (``vals``: this thread's NV columns, when ``act``) to every rank's ``ffn`` region
    from every wave (``device.PUSH_SET`` "fin"): staged in LDS (``xcol``: free between the dn items and the
    unpool), then threads TPD p .. TPD (p + 1) store to rank p, four columns (16 B) a lane. The polls that
    follow wait behind AR_COLS / 4 / TPD stores a wave, not W."""
    W, xcol = c["W"], c["xcol"]
    # the ring build's waves end their dn items one by one, each reading the tile's mids staged in ``xcol``:
    # every wave is past them before the values go there
    lds_barrier()
    if act:
        for e in range_constexpr(NV):
            fx.ptr_store(vals[e], xcol + (NV * c["tid"] + e))
    lds_barrier()
    dest, td, TPD = wave_rank(c, W)
    region = preg(rank_base([r.value for r in c["peer_ffn"]], dest), 0, "ffn")
    n_items = AR_COLS // 4
    for qq in range_constexpr(cdiv(n_items, TPD)):
        kk = td + qq * TPD
        if kk < n_items:
            fv = fx.ptr_load(xcol + 4 * kk, result_type=_v(4))
            c["put_bf"](region, (c["rank"] * MAX_TOKENS + t) * MOE_HIDDEN + part * AR_COLS + 4 * kk, [fx.Float32(fv[e]) for e in range(4)])  # fmt: skip
    lds_barrier()  # ``xcol`` is the unpool's next (a fin CTA of a whole-row unpool build)


@flyc.jit
def fin_push_idle(c, t, part, vals, act, NV: int, AR_COLS: int):
    """The fin task's partial (``vals``: an active thread's NV columns) to every rank's ``ffn`` region from
    idle waves of the CTA (``FIN_IDLE``): staged in LDS (``xcol``, as ``fin_push_all``), then the wave after the
    active ones stores every item (four columns, 16 B a lane and store) to rank 0, the next wave to rank 1, ..
    (itself too). The active waves issue no cross-GPU store, so their polls of the ranks' partials wait behind
    none; a pushing wave stores to one rank (stores to several ranks in turn leave a wave slowly)."""
    W, xcol, tid, lane, wave = c["W"], c["xcol"], c["tid"], c["lane"], c["wave"]
    n_act = AR_COLS // NV
    WA = n_act // WAVE  # the active waves; the W after them push
    assert NV == 4 and n_act % WAVE == 0 and WA + W <= WAVES, (NV, AR_COLS, W)
    lds_barrier()  # (every wave is past the dn items' reads of ``xcol``)
    if act:
        for e in range_constexpr(NV):
            fx.ptr_store(vals[e], xcol + (NV * tid + e))
    lds_barrier()
    fvs = [
        fx.ptr_load(xcol + 4 * (lane + WAVE * q), result_type=_v(4)) for q in range(WA)
    ]
    lds_barrier()  # ``xcol`` is free again; the active waves go on to their polls
    if (wave >= WA) & (wave < WA + W):
        dest = uniform(fx.min(wave - WA, W - 1))
        region = preg(rank_base([r.value for r in c["peer_ffn"]], dest), 0, "ffn")
        for q in range_constexpr(WA):
            c["put_bf"](region, (c["rank"] * MAX_TOKENS + t) * MOE_HIDDEN + part * AR_COLS + 4 * (lane + WAVE * q), [fx.Float32(fvs[q][e]) for e in range(4)])  # fmt: skip


def fin_held(c, t):
    """Token t's eight picks as local experts, or -1 (another rank's): the table ``stage_plan`` left behind
    ``cnt_of``'s counts, from the picks it built the items of."""
    return [
        fx.ptr_load(c["cnt_of"] + (LOCAL_EXPERTS + t * TOP_K + sl))
        for sl in range(TOP_K)
    ]


def fin_score_specs(c, t):
    """Token t's eight scores as poll specs: K2's tagged ones (``k2_idx``, behind the picks: this launch's or not
    there yet, two pairs a load) when the plan took the tagged picks; none for a standalone K3, whose plain scores
    the host wrote."""
    if c.get("plan_idx") is None:
        return []
    return [
        (c["plan_idx"], MAX_PICKS + t * TOP_K + 2 * j, 2) for j in range(TOP_K // 2)
    ]


@flyc.jit
def stage_fin(c, t, part):
    """fin task (token t, part): the part's columns of the token's slots (two or
    four elements a thread), the finalize (v = bf16(y pen_w), each slot's sum of
    squares over its whole row summed from the dn items' publishes in ``k3_ss``,
    score / rms weighted sum in slot order), pushed as bf16 pairs to every rank's ``ffn``
    region, then the same columns of every rank's partial summed in rank order ->
    ``k3_fbuf`` (plain bf16) and the (token, part) flag. A token none of whose
    picks this rank holds pushes zeros and polls nothing."""
    W, tid, lane, wave, red = c["W"], c["tid"], c["lane"], c["wave"], c["red"]
    put, put_bf, poll = c["put"], c["put_bf"], c["poll"]
    y_mb, ss_mb, fbuf, own_ffn = c["y"], c["ss"], c["fbuf"], c["own_ffn"]
    a = c["args"]
    rank = c["rank"]
    AP = c["ar_parts"]
    AR_COLS = MOE_HIDDEN // AP
    PPT = fin_ppt(AP)  # column pairs a thread
    NV = 2 * PPT
    act = NV * tid < AR_COLS
    col = part * AR_COLS + NV * fx.min(
        tid, AR_COLS // NV - 1
    )  # an idle thread loads the last columns
    pwv = bo.buffer_load(rsrc(a["pen_w"]), col // 2, vec_width=PPT, dtype=T.i32)
    if const_expr(PPT == 1):
        pw_words = [fx.Int32(pwv)]
    else:
        pwv = fx.Vector(pwv)
        pw_words = [pwv[q] for q in range(PPT)]
    pw = [bf2_f32(w)[h] for w in pw_words for h in range(2)]
    # The token's slots this rank holds come from the plan (``fin_held``), which took K2's tagged picks ahead of the
    # routing's plain stores and its flag. A rank none of the step's picks reached has no expert item and runs
    # this task at once: a plain read of the picks here could still return the previous layer's, a slot it then
    # waits for and no item writes.
    es = fin_held(c, t)
    # the scores ride with the poll of the dn items' rows below (a rank without a pick needs none)
    sc_specs = fin_score_specs(c, t)
    sc_plain = []
    if const_expr(len(sc_specs) == 0):
        sc_plain = [
            fx.Float32(bo.buffer_load(rsrc(a["scores"]), t * TOP_K + sl, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))
            for sl in range(TOP_K)
        ]  # fmt: skip
    # the first slot this rank holds (a slot it does not hold polls that one's row)
    fb = fx.Int32(-1)
    for sl in range_constexpr(TOP_K):
        fb = ((fb < 0) & (es[sl] >= 0)).select(fx.Int32(sl), fb)
    a0 = fx.Float32(0.0)
    a1 = fx.Float32(0.0)
    a2 = fx.Float32(0.0)
    a3 = fx.Float32(0.0)
    # the slot this wave sums the squares of (a slot this rank does not hold, or a
    # wave past TOP_K: a held one's, taken as 0)
    psl = fx.Int32(fb)
    held_w = fx.Int32(0) == fx.Int32(1)
    for s_ in range_constexpr(TOP_K):
        hit = (wave == s_) & (es[s_] >= 0)
        psl = hit.select(fx.Int32(s_), psl)
        held_w = held_w | hit
    SSL = DN_T // WAVE  # ss publishes a lane
    if fb >= 0:
        got = poll(
            [
                (y_mb, (t * TOP_K + (es[sl] >= 0).select(fx.Int32(sl), fb)) * (MOE_HIDDEN // 2) + col // 2, PPT)
                for sl in range(TOP_K)
            ]
            + [(ss_mb, (t * TOP_K + psl) * DN_T + SSL * lane + 2 * k, 2) for k in range(SSL // 2)]
            + sc_specs,
            batch=TOP_K + SSL // 2 + len(sc_specs),
        )  # fmt: skip
        stamp(c, 15)  # the token's y rows and their sums of squares arrived (fin CTAs)
        if const_expr(len(sc_specs) == 0):
            wsc = sc_plain
        else:
            wsc = [
                fx.Int32(got[TOP_K + SSL // 2 + sl // 2][sl % 2]).bitcast(fx.Float32)
                for sl in range(TOP_K)
            ]
        vs = []
        for sl in range_constexpr(TOP_K):
            held = es[sl] >= 0
            v = []
            for q in range_constexpr(PPT):
                lo, hi = bf2_f32(got[sl][q])
                v.append(held.select(bf16_round(lo * pw[2 * q]), fx.Float32(0.0)))
                v.append(held.select(bf16_round(hi * pw[2 * q + 1]), fx.Float32(0.0)))
            vs.append(v)
        tot = fx.Float32(0.0)
        for k in range_constexpr(SSL // 2):
            for h in range_constexpr(2):
                tot = tot + got[TOP_K + k][h].bitcast(fx.Float32)
        tot = held_w.select(tot, fx.Float32(0.0))
        tot = butterfly(tot, WAVE_OFFSETS)
        if lane == 0:
            rinv = fx.Float32(1.0) / fmath.sqrt(tot / float(MOE_HIDDEN) + POST_EPS)
            fx.ptr_store(rinv, red + (TOP_K * WAVES + wave))
        gpu.barrier()
        for sl in range_constexpr(TOP_K):
            sc = wsc[sl] * fx.ptr_load(red + (TOP_K * WAVES + sl))
            a0 = a0 + sc * vs[sl][0]
            a1 = a1 + sc * vs[sl][1]
            if const_expr(NV == 4):
                a2 = a2 + sc * vs[sl][2]
                a3 = a3 + sc * vs[sl][3]
    if const_expr(PF_UNP):
        if tid == 0:
            put(
                c["pflag"], t * AP + part, fx.Int32(1)
            )  # this token's y rows are in (a timing hint, no data)
    vals = [a0, a1] if NV == 2 else [a0, a1, a2, a3]
    rs = RSAG and W > 1
    owner = col // (
        MOE_HIDDEN // W
    )  # the rank owning these columns (whole waves share one; used with rs)
    mine = owner == rank
    idle = (
        FIN_IDLE
        and W > 1
        and not rs
        and NV == 4
        and (AR_COLS // NV) // WAVE + W <= WAVES
    )
    if const_expr(idle):
        fin_push_idle(c, t, part, vals, act, NV, AR_COLS)
    elif const_expr("fin" in PUSH_SET and W > 1 and not rs):
        fin_push_all(c, t, part, vals, act, NV, AR_COLS)
    else:
        fin_push_first(c, t, col, vals, act, owner, rs)
    stamp(c, 16)  # this rank's partial pushed (fin CTAs)
    if act:
        # the ranks' partials of the same columns, summed in rank order, one rounding
        words = []
        for q in range_constexpr(PPT):

            def pair_at(src, q=q):
                return ((src * MAX_TOKENS + t) * MOE_HIDDEN + col) // 2 + q

            if const_expr(rs):
                # the owner sums the partials; every other rank polls the owner's result (W copies of one spec)
                fq = (t * MOE_HIDDEN + col) // 2 + q
                reg = preg(mine.select(own_ffn.value, c["own_fred"].value), 0, "ffn")
                ws = poll([(reg, mine.select(pair_at(src), fq), 1) for src in range(W)])
                lo, hi = bf2_f32(ws[0][0])
                f0, f1 = lo, hi
                for src in range_constexpr(1, W):
                    l2, h2 = bf2_f32(ws[src][0])
                    lo = lo + l2
                    hi = hi + h2
                lo = mine.select(lo, f0)
                hi = mine.select(hi, f1)
            else:
                lo, hi = sum_partials(poll, own_ffn, pair_at, W)
            words.append(bf16_pair(lo, hi).bitcast(fx.Int32))
        if const_expr(rs):
            if mine:
                # the reduced pairs to every rank's ``fred`` (the all-gather half)
                for pe in range_constexpr(W):
                    c["put_words"](
                        c["peer_fred"][pe], (t * MOE_HIDDEN + col) // 2, words
                    )
        r_f = rsrc(fbuf.value)
        if const_expr(PPT == 1):
            bo.buffer_store(
                words[0], r_f, (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV
            )
        else:
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), r_f, (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV)  # fmt: skip
    stamp(c, 17)  # every rank's partials arrived and summed (fin CTAs)
    if const_expr(idle):
        # only the active waves stored the result; a pushing wave does not wait for its cross-GPU stores here
        if wave < (AR_COLS // NV) // WAVE:
            release_stores()
    else:
        release_stores()
    gpu.barrier()
    if tid == 0:
        put(c["fflag"], t * AP + part, fx.Int32(1))


# ---------------------------------------------------------------- 4. unpool
@flyc.jit
def unp_push_first(c, live_k, t0, gtask, nt: int):
    """An unpool job's rows (``red`` slot 0), the first push form: thread (token, row pair) stores its pair to
    the ranks in turn (``UNP_FULL``: to this rank's own ``unpl``)."""
    W, tid, red, put_bf = c["W"], c["tid"], c["red"], c["put_bf"]
    if live_k & (tid < (UNP_ROWS // 2) * nt):
        tk = tid // (UNP_ROWS // 2)
        rr = (tid % (UNP_ROWS // 2)) * 2
        v0, v1 = acc_pair(red, 0, rr, tk)
        if const_expr(UNP_FULL):
            put_bf(
                c["unpl"], (t0 + tk) * HIDDEN + gtask * UNP_ROWS + rr, [v0, v1]
            )  # this rank's own rows
        else:
            for pe in range_constexpr(W):
                put_bf(
                    c["peer_unp"][pe],
                    (t0 + tk) * HIDDEN + gtask * UNP_ROWS + rr,
                    [v0, v1],
                )


@flyc.jit
def unp_push_all(c, live_k, t0, gtask, nt: int):
    """An unpool job's rows (``red`` slot 0) to every rank's ``unp`` region from every wave (``device.PUSH_SET``
    "unp"): threads TPD p .. TPD (p + 1) store to rank p, the k-th of them rows 4 (k % 4) .. + 4 of token k / 4
    (16 B): one store a wave at 8 or 16 rows a pass."""
    W, red = c["W"], c["red"]
    dest, td, TPD = wave_rank(c, W)
    region = preg(rank_base([r.value for r in c["peer_unp"]], dest), 0, "unp")
    QT = UNP_ROWS // 4
    n_items = QT * nt
    for qq in range_constexpr(cdiv(n_items, TPD)):
        kk = td + qq * TPD
        if live_k & (kk < n_items):
            tk = kk // QT
            rr = (kk % QT) * 4
            v0, v1 = acc_pair(red, 0, rr, tk)
            v2, v3 = acc_pair(red, 0, rr + 2, tk)
            c["put_bf"](
                region, (t0 + tk) * HIDDEN + gtask * UNP_ROWS + rr, [v0, v1, v2, v3]
            )


@flyc.jit
def stage_unpool(c, first_w):
    """This CTA's unpool jobs (``first_w``: the first's weights, issued before the
    fin flags' wait, which streams them): the summed rows are staged in LDS by async
    copies in passes of XS_LDS_MAX tokens, each pass's 16 x tokens bf16 sums pushed
    to every rank's ``unp`` region. A further job's weights go out before the
    previous job's WMMAs."""
    S, W, tid, lane, wave = c["S"], c["W"], c["tid"], c["lane"], c["wave"]
    red, xcol, put_bf = c["red"], c["xcol"], c["put_bf"]
    b = c["bases"]
    AP, KU = c["ar_parts"], b["ku"]
    h = lane // 16
    fbuf = c["fbuf"].value
    assert S % unp_tokens(S) == 0, S  # a job is one whole pass
    task0, _, _ = unp_gtask(c, 0)
    live = (task0 >= 0) & (task0 < b["unp_jobs"])
    bid = c["bid"]
    fin = (bid >= b["fin0"]) & (bid < b["unp0"])
    if const_expr(PF_UNP):
        # the CTAs without a fin task: the unpooler into GL2 once the rank's dn reads are over
        if live & ((bid < b["fin0"]) | (bid >= b["unp0"])):
            poll_flags(c, [(c["pflag"], S * AP)])
            unp_prefetch(
                c, (bid < b["fin0"]).select(bid, bid - S * AP), BLOCKS - S * AP
            )
        stamp(c, 12)  # the unpooler's prefetch issued (PF_UNP)
    if live:
        poll_flags(c, [(c["fflag"], S * AP)])
    stamp(c, 8)  # the fin flags arrived (unpool CTAs)
    if live:
        cur = first_w
        if const_expr(UNP_FULL):
            # a fin CTA's first job: its weights go out now (``first_w`` holds the dead tile's zeros there, and
            # these loads read the dead tile on the other CTAs)
            late = unp_w_loads(c, fin.select(unp_gtask(c, 0)[1], fx.Int32(UNP_TASKS)))
            cur = [bor(fx.Vector(x), fx.Vector(y)) for x, y in zip(first_w, late)]
        for k in range_constexpr(KU):
            task, gtask, job_ps = unp_gtask(c, k)
            live_k = (task >= 0) & (
                task < b["unp_jobs"]
            )  # a dead job computes on a clamped pass, publishes nothing
            nt = unp_tokens(S)  # every served width is whole passes
            t0 = fx.min(job_ps, b["unp_passes"] - 1) * nt
            tl = fx.min(lane % 16, nt - 1)
            # the pass's rows by async copies: round i is token t0 + i, 16 B a thread (one pass: staged once)
            restage = k == 0 or b["unp_passes"] > 1
            if const_expr(restage):
                for i in range_constexpr(nt * (MOE_HIDDEN // 8) // THREADS):
                    lds_dma_16(
                        fbuf + fx.Int64((t0 + i) * (MOE_HIDDEN * 2)),
                        xcol + (i * UNP_PITCH + wave * WAVE * 4),
                        tid * 16,
                        CM_DEV,
                    )
            if const_expr(k + 1 < KU):
                nxt = unp_w_loads(c, unp_gtask(c, k + 1)[1])
            if const_expr(restage):
                wait_lds_dma(0)
                lds_barrier()
            if const_expr(k == 0):
                stamp(c, 9)  # the first job's rows landed in LDS
            acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
            for kc in range_constexpr(UNP_STEPS):
                k0 = wave * (MOE_HIDDEN // WAVES) + kc * 32 + 16 * h
                at = xcol + (tl * UNP_PITCH + k0 // 2)
                bv = cat(
                    [
                        fx.ptr_load(at, result_type=_v(4)),
                        fx.ptr_load(at + 4, result_type=_v(4)),
                    ]
                )
                acc = wmma_bf16(
                    fx.Vector(cur[kc]).bitcast(fx.BFloat16),
                    bv.bitcast(fx.BFloat16),
                    acc,
                )
            tile_store(c, 0, acc)
            lds_barrier()
            if const_expr(k == 0):
                stamp(c, 10)  # the first job's WMMAs done
            # branch = bf16(the fp32 sum over the waves, in wave order): thread (token, row pair)
            if const_expr("unp" in PUSH_SET and W > 1 and not UNP_FULL):
                unp_push_all(c, live_k, t0, gtask, nt)
            else:
                unp_push_first(c, live_k, t0, gtask, nt)
            lds_barrier()
            if const_expr(k == 0):
                stamp(c, 11)  # the first job's rows pushed
            if const_expr(k + 1 < KU):
                cur = nxt


# ---------------------------------------------------------------- 5. the boundary
@flyc.jit
def stage_bnd(c, t, x_flag=None, x_put=None):
    """bnd task t: the token's whole unpooled row (16 elements a thread, from every
    rank's pushes) through the post norm with input scale (eps 1e-8), the gated
    residual (the fp32 stream in place) and the next layer's pre-attention norm
    (eps 1e-5) -> ``attn_out``. On the last layer the stream is rounded to bf16
    before that norm (the model's final-norm input dtype): ``c["last"]`` compiles the
    rule in, else the launch's ``last`` argument selects it. ``x_flag`` (with
    ``x_put``, the next layer's mailbox): the next layer's K1 runs later in this
    launch; the token's flag."""
    tid, poll = c["tid"], c["poll"]
    own_unp = (
        c["unpl"] if UNP_FULL else c["own_unp"]
    )  # UNP_FULL: this rank's own whole rows (device scope)
    a = c["args"]
    e0 = tid * BND_ELEMS
    fw_w = [ld4(rsrc(a["pfw"]), e0 // 2 + 4 * i) for i in range(2)]
    al = [fx.Vector(bo.buffer_load(rsrc(a["alpha"]), e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    be = [fx.Vector(bo.buffer_load(rsrc(a["beta"]), e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    st = [fx.Vector(bo.buffer_load(rsrc(a["res"]), t * HIDDEN + e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    nw_w = [ld4(rsrc(a["npw"]), e0 // 2 + 4 * i) for i in range(2)]
    got = poll([(own_unp, (t * HIDDEN + e0) // 2 + 2 * j, 2) for j in range(4)])
    stamp(c, 13)  # every rank's unpooled rows arrived (thread 0's columns)
    u = [bf2_f32(got[j][q])[h] for j in range(4) for q in range(2) for h in range(2)]
    fw = [
        bf2_f32(fx.Int32(fw_w[k // 8][(k % 8) // 2]))[k % 2] for k in range(BND_ELEMS)
    ]
    pp = [bf16_round(u[k] * fw[k]) for k in range(BND_ELEMS)]
    ssp = pp[0] * pp[0]
    for k in range_constexpr(1, BND_ELEMS):
        ssp = ssp + pp[k] * pp[k]
    rstd = hw_rsq(block_sum(c, ssp, 0) / float(HIDDEN) + POST_EPS)
    stamp(c, 14)  # the post norm's sum
    nb = [bf16_round(pp[k] * rstd) for k in range(BND_ELEMS)]
    new = [
        fx.Float32(al[k // 4][k % 4]) * fx.Float32(st[k // 4][k % 4]) + fx.Float32(be[k // 4][k % 4]) * nb[k]
        for k in range(BND_ELEMS)
    ]  # fmt: skip
    for i in range_constexpr(4):
        bo.buffer_store(fx.Vector.from_elements(new[4 * i : 4 * i + 4], fx.Float32), rsrc(a["res"]), t * HIDDEN + e0 + 4 * i)  # fmt: skip
    if const_expr(c["last"]):
        pre = [bf16_round(v) for v in new]
    else:
        is_last = a["last"] != 0
        pre = [is_last.select(bf16_round(v), v) for v in new]
    ss2 = pre[0] * pre[0]
    for k in range_constexpr(1, BND_ELEMS):
        ss2 = ss2 + pre[k] * pre[k]
    rstd2 = hw_rsq(block_sum(c, ss2, 1) / float(HIDDEN) + RMS_EPS)
    if const_expr(c["defer_k1"]):
        if c["tid"] < 2:
            c["put"](c["nscal"], 2 * t + c["tid"], (c["tid"] == 0).select(rstd, rstd2))
    nw = [
        bf2_f32(fx.Int32(nw_w[k // 8][(k % 8) // 2]))[k % 2] for k in range(BND_ELEMS)
    ]
    for i in range_constexpr(2):
        words = [
            bf16_pair(pre[k] * rstd2 * nw[k], pre[k + 1] * rstd2 * nw[k + 1]).bitcast(
                fx.Int32
            )
            for k in range(8 * i, 8 * i + 8, 2)
        ]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(a["attn_out"]), (t * HIDDEN + e0) // 2 + 4 * i, cache_modifier=CM_DEV)  # fmt: skip
    if const_expr(x_flag is not None):
        release_stores()
        gpu.barrier()
        if tid == 0:
            x_put(x_flag, t * FLAG_STRIDE, fx.Int32(1))


# ---------------------------------------------------------------- the launch's order
@flyc.jit
def run_k3(
    c,
    route_flag=None,
    xin_flag=None,
    x_flag=None,
    x_put=None,
    res_flag=None,
    ids_mb=None,
    sc_mb=None,
):
    """K3's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    the routing and the expert input were written earlier in this launch, flagged a
    token each. ``x_flag`` / ``x_put``: see ``stage_bnd``. ``res_flag`` = (region,
    count): K2's boundary parts flag their stream columns. ``ids_mb``: K2's tagged picks,
    the tagged scores behind them (``PLAN_IDX``: the plan and the fin tasks poll them;
    ``sc_mb``, the gfx950 build's TP-expert hand-off, is not taken on gfx1250)."""
    S, bid = c["S"], c["bid"]
    b = c["bases"]
    AP = c["ar_parts"]
    stamp(c, 0)
    late_res = res_flag is not None and not c["defer_k1"]
    res_flags = (
        [(res_flag[0], 0, res_flag[1], FLAG_STRIDE)]
        if res_flag is not None and not late_res
        else []
    )
    early = (
        PF_EXP and K3_WAVE == 2 and route_flag is not None
    )  # the routing's wait apart (``stage_ug_pre``)
    if const_expr(route_flag is not None):
        c["pw_loads"] = plan_pw_loads(
            c
        )  # (the plan's weight loads, ahead of the flags' wait)
        if const_expr(early):
            poll_flags(c, [(route_flag, 0, S, FLAG_STRIDE)])
        elif const_expr(PLAN_IDX and ids_mb is not None):
            # the plan polls the tagged picks itself (``stage_plan``), and the fin tasks the tagged picks and
            # scores (``stage_fin``): nothing here reads the route task's plain stores, which its flag covers
            c["plan_idx"] = ids_mb
            poll_flags(c, [(xin_flag, 0, S, FLAG_STRIDE)] + res_flags)
            if const_expr(K3_WAVE == 2 and c["xq_pre"]):
                # the expert input is staged while the picks are still on their way (the routing comes last)
                if const_expr(plan_ovl(S)):
                    xv = xq_all_loads(c)
                    plan_picks_poll(
                        c
                    )  # (behind the staging loads; its stores before the staging's barrier)
                    stage_xq_all(c, xv)
                    c["picks_staged"] = True
                else:
                    stage_xq_all(c, xq_all_loads(c))
                c["xq_staged"] = True
        else:
            poll_flags(
                c,
                [(route_flag, 0, S, FLAG_STRIDE), (xin_flag, 0, S, FLAG_STRIDE)]
                + res_flags,
            )
    stamp(
        c, 1
    )  # the routing (PF_EXP) / routing and expert input arrived (a merged launch) / entry
    n_tiles = stage_plan(c)
    stamp(c, 2)  # the plan built
    if const_expr(early):
        stage_ug_pre(c, n_tiles)
        poll_flags(c, [(xin_flag, 0, S, FLAG_STRIDE)] + res_flags)
        stamp(c, 18)  # the expert input arrived (PF_EXP)
    if const_expr(K3_WAVE == 2):
        assert c["xq_pre"], (
            "the wave-owned expert items read the staged MX FP8 x (xq_pre)"
        )
        if const_expr(K3_UGQ and not early and K3_PROBE_TILES == 4):
            stage_ug_a(c, n_tiles)
        else:
            stage_ug_r(c, n_tiles, issued=early)
        stamp(c, 3)  # this CTA's ug items done
        if const_expr(K3_DNQ):
            stage_dn_a(c, n_tiles)
        else:
            stage_dn_r(c, n_tiles)
    elif const_expr(K3_WAVE == 1):
        assert c["xq_pre"], (
            "the wave-owned expert items read the staged MX FP8 x (xq_pre)"
        )
        stage_ug_w(c, n_tiles)
        stamp(c, 3)  # this CTA's ug items done
        stage_dn_w(c, n_tiles)
    else:
        dn_first = (stage_ug2 if K3_SLOTS == 2 else stage_ug)(c, n_tiles)
        stamp(c, 3)  # this CTA's ug items done
        (stage_dn2 if K3_SLOTS == 2 else stage_dn)(c, n_tiles, dn_first)
    stamp(c, 4)  # this CTA's dn items done
    # the unpool CTAs' first job's weights go out now, behind nothing, so the fin
    # hops stream them; the fin CTAs carry none (their tile is past the buffer)
    first_w = unp_w_loads(c, unp_gtask(c, 0, before_fin=True)[1])
    if (bid >= b["fin0"]) & (bid < b["fin0"] + S * AP):
        stage_fin(c, (bid - b["fin0"]) // AP, (bid - b["fin0"]) % AP)
    stamp(c, 5)  # the fin task done (fin CTAs)
    stage_unpool(c, first_w)
    stamp(c, 6)  # the unpool job(s) done (unpool CTAs)
    bnd0 = b["bnd0"]
    if (bid < S) if bnd0 == 0 else ((bid >= bnd0) & (bid < bnd0 + S)):
        if const_expr(late_res):
            poll_flags(c, [(res_flag[0], 0, res_flag[1], FLAG_STRIDE)])
        stage_bnd(c, bid if bnd0 == 0 else bid - bnd0, x_flag, x_put)
    stamp(c, 7)  # the boundary done (end)
