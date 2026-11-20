# SPDX-License-Identifier: Apache-2.0
"""K-attn: one Muse Spark layer's attention over the fp8 KV cache, for S =
``tokens`` rows of R = S / QL requests, ``QL`` consecutive rows a request (a
DFlash verify's 8 draft tokens, or one for decode). Per request and KV head, the
rank's 8 query heads of each of its QL rows attend the request's keys, as the
ROCm model's verify kernels (``swa_verify_splitkv``, ``gqa_verify_splitkv_2kv``)
and aiter's unified attention compute them::

    S = scale log2(e) (q . k)   q bf16 (K1 already applied the query scale),
                                K E4M3 upcast to bf16, fp32
    masked: key < context + qpos + 1 (causal), context + qpos - key < window
    online softmax in the exp2 domain, P -> bf16, O += P V (V upcast)
    O / l -> bf16

gfx1250 (wave32, WMMA): every wave takes 32 keys of a chunk (two 16-key tiles,
one 32-key PV step), so a chunk is ``WAVES`` x 32 keys. K and V move by async
global -> LDS copies (``s_wait_asynccnt``); K is upcast fp8 -> bf16 exactly and
the scores come from ``v_wmma_f32_16x16x32_bf16`` (S^T = K Q^T: a lane holds one
row and keys 8 (lane / 16) .. + 7 of each key tile, which is the PV WMMA's A
operand as it stands); V^T comes from ``ds_load_tr8_b64`` (16 x 16 fp8 tiles),
upcast the same way.

Tasks. An attention task is one (request, KV head, ``nt`` row tiles, partition
of ``part`` keys; the partitions a sliding window masks whole are skipped): a
chunk's K / V loads and its V^T reads serve the task's tiles (QL = 8 is 64 rows
= four tiles). Each is a loop over chunks with the waves splitting the keys: QK
on wave-local key tiles, the row statistics, P in registers, V transposed from
LDS, PV; the waves' key splits merge in LDS; the partial (m, l, O) a row goes
out as plain fp32 words with one flag a task. The partitions are sized to at
most ``MAX_PARTS`` a request from the longest request's span (``part_in``
overrides); the partition grows with the step's total chunk work so a step
stays near one task a CTA whatever the request mix.

Merge. A (request, KV head, row tile) with at most ``DIRECT_MAX`` partitions
folds them in one job; a longer one in two levels: ``MERGE_GROUPS`` level-1 jobs
each fold a contiguous eighth of the partitions (a thread a (row, 8 dims,
partition lane), the lanes combined in LDS) into a level-2 partial, and the
level-2 job folds those. The merge jobs count down from below K2's chain CTAs
(the task loop's extra tasks fall on the first); a CTA runs all its level-1
jobs before its level-2 jobs, so every wait points to work that is already
running.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.ops import (
    CM_DEV,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    permlane_swap,
    rsrc,
    uniform,
    wmma_bf16,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.muse_spark.gfx1250.config import (
    HEAD_DIM,
    LOCAL_KV_HEADS,
    LOG2E,
    MAX_TOKENS,
    Q_K,
    RMS_EPS,
    WINDOW,
)
from sgl_amd_mono.muse_spark.gfx1250.device import (
    FLAG_STRIDE,
    OG_ONCE,
    lds_dma_16,
    poll_flags,
    release_stores,
    stamp,
    wait_lds_dma,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_CHUNK as CHUNK
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_CHUNK_PAGES as PPC
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_IDS as IDS
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_IDS_CHUNKS as IDS_CHUNKS
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_KEY_SPLITS as KEY_SPLITS
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_KEY_TILES as KEY_TILES
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_KEYS_PER_WAVE as KEYS_PER_WAVE
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_MAX_PARTS as MAX_PARTS
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_MERGE_GROUPS as GROUPS
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_PART_ROW as PART_ROW
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_Q_PITCH as Q_PITCH
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_RING_DEPTH as RING_DEPTH
from sgl_amd_mono.muse_spark.gfx1250.layout import ATTN_V_STEPS as V_STEPS
from sgl_amd_mono.muse_spark.gfx1250.layout import (
    GQA,
)

TILE = 16  # keys a WMMA tile
NEG = -1.0e30  # a masked score, and "no key yet" of a row's running max
NO_MAX = -1.0e29  # a running max below this is "no key yet": exp2 then takes 0
O_OFF = 4  # a partial row: m, l, pad, pad, O[64] (O 16 B aligned)
COMB_ROW = PART_ROW  # a tile row's (m, l, pad, pad, O[64]) in the LDS tree
# the ring's wait at a chunk's head: chunk cc + 1's four DMAs may stay in flight
# (SGLANG_AMD_MONO_ATTN_RING_WAIT0=1: wait for both chunks, a diagnostic)
MERGE_TOP = (
    BLOCKS - MAX_TOKENS
)  # the merge jobs' CTAs count down from below this one (K2's chain CTAs)
# SGLANG_AMD_MONO_ATTN_EARLY (the bf16 5D cache, the general build): a task's first chunk is copied early (1 / 0;
# default auto: at <= 4 rows, where it gains at long contexts and costs nothing at short ones; at 8 rows it loses).
EARLY = os.environ.get("SGLANG_AMD_MONO_ATTN_EARLY", "auto")
assert EARLY in ("0", "1", "auto"), EARLY


def early_on(tokens: int) -> bool:
    return EARLY == "1" or (EARLY == "auto" and tokens <= 4)


# SGLANG_AMD_MONO_ATTN_FDIV (default 1; 0: integer divisions): the partition planning's divisions by run-time
# values (a few a request) through a float reciprocal instead of the integer division's expansion. Exact for the
# counts here: 0 <= n < 2^20, d >= 1, and (n + 0.5) / d lies at least 0.5 / d off every integer.
ATTN_FDIV = os.environ.get("SGLANG_AMD_MONO_ATTN_FDIV", "1") == "1"


def _rcp(d):
    """1 / d as f32 (``_fdiv``'s reciprocal, made once a divisor)."""
    return fx.Float32(1.0) / fx.Float32(fx.Int32(d))


def _fdiv(n, d, r=None):
    """n // d for small non-negative n and positive d (``r`` = ``_rcp(d)``)."""
    if not ATTN_FDIV:
        return n // d
    return fx.Int32(
        (fx.Float32(fx.Int32(n)) + fx.Float32(0.5)) * (_rcp(d) if r is None else r)
    )


assert MERGE_TOP > 0, (BLOCKS, MAX_TOKENS)


def merge_ctas(s: int, ql: int) -> tuple:
    """The CTAs [lo, hi) of the level-2 merge jobs (``run_kattn``): they reach K2 last.
    More jobs than CTAs below ``MERGE_TOP``: the CTAs take them in rounds."""
    n_l2 = (s // ql) * LOCAL_KV_HEADS * ((ql * GQA + 15) // 16)
    return max(MERGE_TOP - n_l2, 0), MERGE_TOP


DMA_PER_PAGE = (
    16 * HEAD_DIM // (16 * WAVE)
)  # async 16 B copies a wave lands a 1 KB page with
DMA_PER_CHUNK = 2 * 2 * DMA_PER_PAGE  # a wave's chunk: K and V, two pages each
RING_WAIT = (
    0
    if os.environ.get("SGLANG_AMD_MONO_ATTN_RING_WAIT0", "0") == "1"
    else DMA_PER_CHUNK
)
MERGE_LANES = 4  # partitions folded side by side a (row, 8 dims): 16 x 8 x 4 = THREADS
PAGE_WORDS = (
    16 * HEAD_DIM // 4
)  # a 16-key fp8 page of one head in LDS: 1 KB, chunk-major (dims 16 c .., key)
V_SKEW = 16  # words between the two V pages beyond a page: the transposed reads of both hit distinct banks
SLOT_WORDS = (
    2 * KEY_TILES * PAGE_WORDS + V_SKEW
)  # a ring slot: a wave's chunk, its K pages then its V pages
# bf16 KV in SGLang's "vectorized_5d" layout (``c["kv5d"]``: --kv-cache-dtype bf16, --page-size 64,
# SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d): K [page][head][dim / 8][key][8], V [page][head][key / 8][dim][8], i.e.
# 16 B units of eight bf16: a key's 8 dims (K) or a dim's 8 keys (V), which are the WMMA operands' lane halves as they
# stand (no fp8 upcast, no transposed read). A wave's 32 keys of a chunk are half a page: per dim chunk 512 B of K
# (8 async copies), and 4 KB of V contiguous (8 copies).
PAGE5 = 64  # keys a page
PPC5 = CHUNK // PAGE5  # page ids a chunk: 8
IDS5 = (
    IDS_CHUNKS * PPC5
)  # page ids a task stages at once (the same chunks as the fp8 form)
KV5_HEAD_BYTES = PAGE5 * HEAD_DIM * 2  # a (page, head) of K or of V: 8 KB
KV5_PAGE_BYTES = LOCAL_KV_HEADS * KV5_HEAD_BYTES
KV5_K_WORDS = (
    KEYS_PER_WAVE * HEAD_DIM * 2 // 4
)  # a wave's K of a chunk in LDS: [dim chunk][key][16 B], 4 KB
SLOT5_WORDS = (
    2 * KV5_K_WORDS
)  # a ring slot: the wave's K, then its V ([key group][dim][16 B])
DMA5_PER_CHUNK = (
    2 * (KV5_K_WORDS * 4) // (WAVE * 16)
)  # async copies a wave's chunk: 8 K + 8 V
assert CHUNK % PAGE5 == 0 and KEYS_PER_WAVE * 2 == PAGE5 and IDS5 % THREADS == 0, (
    CHUNK,
    KEYS_PER_WAVE,
    IDS5,
)
TAU = 8.0  # a row's reference max moves only when a chunk's max exceeds it by this (log2 units): P <= 256
DIRECT_MAX = (
    16  # partitions a level-2 job folds itself (four a lane); more go through level 1
)


def fp8_words_to_bf16(words):
    """The fp8 E4M3 bytes of ``words`` (i32, little-endian byte order) as one bf16
    vector: ``v_cvt_f32_fp8`` a byte (exact), then bf16 (exact: three mantissa bits)."""
    vals = []
    for w in words:
        for b in range_constexpr(4):
            vals.append(
                fx.Float32(
                    llvm.call_intrinsic(
                        T.f32,
                        "llvm.amdgcn.cvt.f32.fp8",
                        [as_ir_value(fx.Int32(w)), as_ir_value(fx.Int32(b))],
                        [],
                        [],
                    )
                )
            )
    return fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16)


def tr8_offset(lane):
    """The byte a lane addresses in a 16 x 16 fp8 tile (16 B a row) for
    ``ds_load_tr8_b64``: lane l then holds column l % 16 of rows 8 (l / 16) .. + 7."""
    row = (lane // 16) * 8 + lane % 4 + ((lane % 16) // 8) * 4
    col = (((lane % 16) // 4) % 2) * 8
    return row * 16 + col


def comb_floats(nt: int) -> int:
    """The key-split merge's LDS floats: every wave's state of ``nt`` tiles."""
    return WAVES * nt * 16 * COMB_ROW


def kattn_lds(nt: int, kv5d: bool = False):
    """K-attn's LDS as two aliasing views (a union of them, built by the launch): a
    task's Q tiles and the merge jobs' reduction, then either each wave's P tiles
    and V^T (the chunk loop) or the key splits' merge tree (after it). ``kv5d``: the
    bf16 5D cache's ring slots and page ids."""
    slot_words, n_ids = (SLOT5_WORDS, IDS5) if kv5d else (SLOT_WORDS, IDS)

    @fx.struct
    class Loop:
        qs: fx.Array[
            fx.BFloat16, nt * 16 * Q_PITCH, 16
        ]  # the task's Q rows (its row tiles)
        red: fx.Array[fx.Float32, THREADS * 10, 16]  # the merge jobs' lane combine
        ring: fx.Array[
            fx.Int32, WAVES * RING_DEPTH * slot_words, 16
        ]  # each wave's K / V chunks (DMA ring)
        ids: fx.Array[fx.Int32, n_ids, 16]  # the task's page ids, a block of chunks

    @fx.struct
    class Tree:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]
        red: fx.Array[fx.Float32, THREADS * 10, 16]
        comb: fx.Array[fx.Float32, comb_floats(nt), 16]

    return Loop, Tree


def lds_views(loop, tree):
    """``c`` entries of the two ``kattn_lds`` views (``.peek()`` of union members)."""
    return {"qs": loop.qs.ptr, "red": loop.red.ptr, "ring": loop.ring.ptr, "ids": loop.ids.ptr, "comb": tree.comb.ptr}  # fmt: skip


# The step's attention plan: [layer kind: full, sliding][the partition in keys, then a request's partition count
# each], plain int32 words in the scratch's ``attn_plan`` region (``layout.ATTN_PLAN_WORDS``).
PLAN_WORDS = 1 + MAX_TOKENS


def plan_compute(lens, QL, window, part_in, task_tiles):
    """(the partition in keys, [request r's partitions]) of a layer with ``window`` (0: full) over requests of
    ``lens`` keys: ``run_kattn``'s planning (the general build's: the same operations on the same values)."""
    R = len(lens)
    spans = []
    for rr in range(R):
        hi = (lens[rr] + CHUNK - 1) // CHUNK
        first = fx.max(lens[rr] - QL - window + 1, fx.Int32(0))
        lo = (window > 0).select(first // CHUNK, fx.Int32(0))
        spans.append(hi - lo)
    max_span = spans[0]
    total_span = spans[0]
    for rr in range(1, R):
        max_span = fx.max(max_span, spans[rr])
        total_span = total_span + spans[rr]

    def tasks_of(pc_):
        n = fx.Int32(0)
        r_pc = _rcp(pc_) if ATTN_FDIV else None
        for rr in range(R):
            n = n + _fdiv(spans[rr] + pc_ - 1, pc_, r_pc)
        return n * (LOCAL_KV_HEADS * task_tiles)

    per_cta = (total_span * (LOCAL_KV_HEADS * task_tiles) + BLOCKS - 1) // BLOCKS
    floor_ = (max_span + MAX_PARTS - 1) // MAX_PARTS
    pc = fx.max(fx.max(per_cta, floor_), fx.Int32(1))
    for _ in range(3):
        pc = (tasks_of(pc) > BLOCKS).select(pc + 1, pc)
    part = (part_in > 0).select(part_in, pc * CHUNK)
    rcp_part = _rcp(part) if ATTN_FDIV else None
    return part, [
        _fdiv(spans[rr] * CHUNK + part - 1, part, rcp_part) for rr in range(R)
    ]


@flyc.jit
def run_attn_plan(S, QL, NT, a, plan):
    """The plan launch's body (one wave): both layer kinds' plans of the step into ``plan`` (the scratch's
    ``attn_plan`` region; device scope, plain words: the layers' launches follow this one)."""
    R = S // QL
    task_tiles = ((QL * GQA + 15) // 16) // NT
    lens = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["seq_lens"]),
                (a["lens64"] != 0).select(2 * rr, rr),
                vec_width=1,
                dtype=T.i32,
            )
        )
        + a["extra"]
        for rr in range(R)
    ]
    r_plan = rsrc(plan.value)
    words = []
    for kind in range_constexpr(2):
        part, nps = plan_compute(
            lens, QL, a["window"] if kind else fx.Int32(0), a["part_in"], task_tiles
        )
        words.append([part] + nps)
    if fx.thread_idx.x == 0:
        for kind in range_constexpr(2):
            for j in range_constexpr(R + 1):
                bo.buffer_store(
                    fx.Int32(words[kind][j]),
                    r_plan,
                    kind * PLAN_WORDS + j,
                    cache_modifier=CM_DEV,
                )


@flyc.jit
def run_kattn(c, o_flag=None):
    """K-attn's work on this CTA for ``c["S"]`` rows, ``c["QL"]`` a request.
    ``o_flag`` (a mailbox region): each level-2 merge raises its (request, KV
    head, row tile)'s flag once its o rows are out (a launch that goes on to K2)."""
    S, QL = c["S"], c["QL"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    put, poll = c["put"], c["poll"]
    part_w, pflag, l2_w, l2flag = c["part"], c["pflag"], c["l2"], c["l2flag"]
    a = c["args"]
    R = S // QL
    ROWS = QL * GQA  # rows a (request, KV head): 64 or 8
    ROW_TILES = (ROWS + 15) // 16  # 4 or 1
    NT = c["NT"]  # row tiles a task (the build's; 4 spills registers, 2 fits)
    # the SLIDING-ONLY build (``c["KIND"] == "sliding"``, QL 8, at most 7 requests): the window is on, a
    # (request, head)'s span is <= 9 chunks, so a task is ONE chunk (part = 1 chunk: the step's span x
    # 4 tasks <= 256) and every merge is direct (<= 9 partitions): the full layers' paths (multi-chunk
    # ring, lazy rescale over chunks, unmasked chunks, level-1 merges, window == 0) are compiled out.
    # The same values and the same order: bit-exact to the general build on a sliding layer.
    TASK_TILES = ROW_TILES // NT
    SL = (
        c.get("KIND", "") == "sliding"
        and QL * GQA == 64
        and S // QL <= 7
        and (WINDOW + QL + CHUNK - 1) // CHUNK + 1 <= DIRECT_MAX
        and (S // QL) * ((WINDOW + QL) // CHUNK + 2) * LOCAL_KV_HEADS * TASK_TILES
        <= BLOCKS
    )  # (a step too wide for one chunk a task takes the general build: the same values)
    # the FULL-ONLY build (``c["KIND"] == "full"``): window == 0 (no window term, no first-chunk edge: a
    # full layer's first chunk is masked by nothing), the job mapping in whole groups of 8 partitions
    FL = c.get("KIND", "") == "full" and QL * GQA == 64
    l16 = lane % 16
    g2 = lane // 16  # the lane's half of the wave
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    V2I = fx.Vector.make_type(2, fx.Int32)
    V4I = fx.Vector.make_type(4, fx.Int32)
    qs, ring, ids, comb, red = c["qs"], c["ring"], c["ids"], c["comb"], c["red"]
    KV5 = c.get(
        "kv5d", False
    )  # bf16 KV, 64-key pages, the vectorized_5d layout (else fp8 NHD, 16-key pages)
    PPC_, IDS_, SLOT_ = (PPC5, IDS5, SLOT5_WORDS) if KV5 else (PPC, IDS, SLOT_WORDS)
    RING_WAIT_ = (0 if RING_WAIT == 0 else DMA5_PER_CHUNK) if KV5 else RING_WAIT
    wring = ring + wave * (RING_DEPTH * SLOT_)  # this wave's K / V ring
    table_stride, extra, window, part_in = (
        a["table_stride"],
        a["extra"],
        a["window"],
        a["part_in"],
    )
    lens64 = a["lens64"]
    qk_scale = a["scale_bits"].bitcast(fx.Float32) * fx.Float32(LOG2E)
    KV_ROWS = c.get(
        "kv_rows", 0
    )  # a head-major full pool's rows (``kv_rows``), else NHD
    SWA_ROWS = c.get("swa_rows", 0)  # a head-major sliding (SWA) pool's rows, else NHD
    assert not (KV5 and (KV_ROWS or SWA_ROWS)), "the 5D cache has one layout"
    k_base, v_base = fx.Int64(a["k_cache"]), fx.Int64(a["v_cache"])
    r_pt = rsrc(a["page_table"], a["table_bytes"])
    r_q = rsrc(a["q"])
    r_part = rsrc(part_w.value)
    r_l2 = rsrc(l2_w.value)
    stamp(c, 0)

    # every request's keys (int32, or the low word of its int64), the step's rows
    # included, loaded once at entry: a task's chain starts at its Q and page ids
    lens = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["seq_lens"]),
                (lens64 != 0).select(2 * rr, rr),
                vec_width=1,
                dtype=T.i32,
            )
        )
        + extra
        for rr in range(R)
    ]

    def seq_len_of(r):
        """Request r's keys."""
        if const_expr(isinstance(r, int)):
            return lens[r]
        out = lens[0]
        for rr in range_constexpr(1, R):
            out = (r == rr).select(lens[rr], out)
        return out

    def chunk_range(keys, qpos_lo):
        """The chunks [lo, hi) of a request with ``keys`` keys a window from its
        first row's position leaves unmasked (every chunk of a full one)."""
        hi = (keys + CHUNK - 1) // CHUNK
        ctx = keys - QL
        first = fx.max(ctx + qpos_lo - window + 1, fx.Int32(0))
        lo = (
            first // CHUNK
            if SL
            else (
                fx.Int32(0) if FL else (window > 0).select(first // CHUNK, fx.Int32(0))
            )
        )
        return lo, hi

    # ``plan_in`` (a build flag): the partition and every request's partition count come from the scratch's
    # ``attn_plan`` words, computed once a step by the plan launch (``plan_compute``: the planning below, the
    # same in every layer of a kind): its divisions were most of the instructions ahead of the first KV copy
    PLAN = bool(c.get("plan_in", False)) and not SL
    if const_expr(PLAN):
        r_plan = rsrc(c["plan"].value)
        pk = (window > 0).select(
            fx.Int32(PLAN_WORDS), fx.Int32(0)
        )  # the layer kind's words
        pv = [
            fx.Int32(
                bo.buffer_load(
                    r_plan, pk + j, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
            for j in range(R + 1)
        ]
        part = pv[0]
        nps = pv[1:]

        def n_parts_of(r):
            """Request r's partitions (the plan's)."""
            if const_expr(isinstance(r, int)):
                return nps[r]
            out = nps[0]
            for rr in range_constexpr(1, R):
                out = (r == rr).select(nps[rr], out)
            return out

    else:
        # the partitions: a request's keys from its first unmasked chunk (a sliding
        # window's start; 0 for a full layer), the longest such span over at most
        # MAX_PARTS of them
        def span_of(r):
            lo_r, hi_r = chunk_range(seq_len_of(r), 0)
            return hi_r - lo_r

        max_span = span_of(0)
        total_span = max_span
        for rr in range_constexpr(1, R):
            sp = span_of(rr)
            max_span = fx.max(max_span, sp)
            total_span = total_span + sp

        # the partition (chunks a task) from the step's total chunk work, so that the
        # step has one task a CTA whatever the request mix (a task costs several chunks'
        # time beside its chunks: its Q and page ids, the first chunk's exposed
        # latency, the key-split tree and the publish), the per-request rounding
        # checked so that no CTA gets a second task, and at least what keeps the
        # longest request within MAX_PARTS
        def tasks_of(pc_):
            n = fx.Int32(0)
            r_pc = _rcp(pc_) if ATTN_FDIV else None
            for rr in range_constexpr(R):
                lo_r, hi_r = chunk_range(seq_len_of(rr), 0)
                n = n + _fdiv((hi_r - lo_r) + pc_ - 1, pc_, r_pc)
            return n * (LOCAL_KV_HEADS * TASK_TILES)

        if const_expr(SL):
            # (sliding-only: one chunk a task: the step's span x LOCAL_KV_HEADS x TASK_TILES <= BLOCKS)
            part = CHUNK
            n_parts = max_span
        else:
            per_cta = (
                total_span * (LOCAL_KV_HEADS * TASK_TILES) + BLOCKS - 1
            ) // BLOCKS
            floor_ = (max_span + MAX_PARTS - 1) // MAX_PARTS
            pc = fx.max(fx.max(per_cta, floor_), fx.Int32(1))
            for _ in range_constexpr(3):
                pc = (tasks_of(pc) > BLOCKS).select(pc + 1, pc)
            part = (part_in > 0).select(part_in, pc * CHUNK)
            rcp_part = _rcp(part) if ATTN_FDIV else None
            n_parts = _fdiv(
                max_span * CHUNK + part - 1, part, rcp_part
            )  # partitions a (request, head) at most

        def n_parts_of(r):
            """Request r's partitions (its span's, the rest of the task slots idle)."""
            lo, hi = chunk_range(seq_len_of(r), 0)
            if const_expr(SL):
                return hi - lo
            return _fdiv((hi - lo) * CHUNK + part - 1, part, rcp_part)

    def pflag_at(r, h, rt, p):
        """Pair index of the flag of the task holding row tile ``rt`` of (request r,
        head h) over partition p: a task's own (its first tile's), not the
        partition's: the tasks of a partition's row tiles finish apart (a merge
        that took the first one's flag read the other tiles' rows of the last layer)."""
        return ((r * LOCAL_KV_HEADS + h) * ROW_TILES + (rt // NT) * NT) * MAX_PARTS + p

    def part_row(r, h, p, row):
        """Float index of (request r, head h, partition p)'s row record."""
        return (((r * LOCAL_KV_HEADS + h) * MAX_PARTS + p) * ROWS + row) * PART_ROW

    def l2_row(r, h, rt, g, row16):
        """Float index of (request r, head h, row tile rt, group g)'s level-2 row record."""
        return (
            (((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g) * 16 + row16
        ) * PART_ROW

    def fold(m, l, ov, m2, l2, v2, ok):
        """(m, l, O) with a partial folded in (``ok``: it exists)."""
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, ov * a1 + v2 * a2

    # ============================================ attention tasks
    def load_ids(r, b_lo, c_last):
        """The page ids of the block of chunks from ``b_lo`` (clamped to ``c_last``),
        this thread's share."""
        out = []
        if const_expr(SL):
            # (sliding-only: the task's one chunk: its 16 page ids)
            return [
                fx.Int32(
                    bo.buffer_load(
                        r_pt,
                        r * table_stride + b_lo * PPC_ + tid % PPC_,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            ]
        for i in range_constexpr(IDS_ // THREADS):
            idx = tid + THREADS * i
            cc_i = fx.min(b_lo + idx // PPC_, c_last)
            out.append(
                fx.Int32(
                    bo.buffer_load(
                        r_pt,
                        r * table_stride + cc_i * PPC_ + idx % PPC_,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            )
        return out

    def store_ids(pgs):
        if const_expr(SL):
            if tid < PPC_:
                fx.ptr_store(pgs[0], ids + tid)
            return
        for i in range_constexpr(IDS_ // THREADS):
            fx.ptr_store(pgs[i], ids + (tid + THREADS * i))

    def issue_pg(pg, h, sl):
        """(KV5) The wave's half page (keys 32 (wave % 2) ..) of page ``pg``, head ``h`` -> ring slot ``sl``: K per
        dim chunk (512 B: the 32 keys' 16 B units), then V (the half's 4 key groups x 64 dims: 4 KB contiguous);
        DMA5_PER_CHUNK copies."""
        slot = wring + sl * SLOT_
        base = fx.Int64(pg) * KV5_PAGE_BYTES + fx.Int64(fx.Int32(h) * KV5_HEAD_BYTES)
        half = wave % 2
        for dc in range_constexpr(HEAD_DIM // 8):
            lds_dma_16(
                k_base + base + fx.Int64((dc * PAGE5 + half * KEYS_PER_WAVE) * 16),
                slot + dc * (WAVE * 4),
                lane * 16,
            )
        for j in range_constexpr(KV5_K_WORDS * 4 // (WAVE * 16)):
            lds_dma_16(
                v_base
                + base
                + fx.Int64(half * (KV5_HEAD_BYTES // 2) + j * (WAVE * 16)),
                slot + (KV5_K_WORDS + j * (WAVE * 4)),
                lane * 16,
            )

    def attn_task(r, h, rt0, p):
        """Row tiles rt0 .. rt0 + NT of (request r, KV head h) over partition p: one
        K / V load and one V^T staging a chunk serve every tile of the task."""
        keys = seq_len_of(r)
        ctx = keys - QL
        ks = wave  # this wave's key split
        lo, hi = chunk_range(keys, 0)
        c_lo = fx.max(lo, lo + p * (part // CHUNK))
        c_hi = fx.min(hi, lo + (p + 1) * (part // CHUNK))
        c_last = fx.max(c_hi - 1, c_lo)
        # (KV5, ``EARLY``) the wave's page id of the task's first chunk, loaded directly and first: its K / V copies
        # go out as soon as it is back, ahead of the Q / ids staging's two barriers (the copies come from HBM:
        # the first chunk's WMMAs waited for them)
        early_pg = None
        if const_expr(KV5 and early_on(S) and not SL):
            early_pg = bo.buffer_load(
                r_pt, r * table_stride + c_lo * PPC5 + ks // 2, vec_width=1, dtype=T.i32
            )
        # Q: rows 16 (rt0 + it) + i = 8 qpos + hg of (request r, head h), 16 B a thread,
        # loaded beside the first block's page ids (one round trip for both), into LDS
        # after a barrier (the ids alias the previous task's merge tree)
        q_thr = tid % (NT * 16 * HEAD_DIM // 8)
        i_t = q_thr // (16 * HEAD_DIM // 8)
        q_row = (q_thr % (16 * HEAD_DIM // 8)) // (HEAD_DIM // 8)
        q_d0 = (q_thr % (HEAD_DIM // 8)) * 8
        q_r = fx.min((rt0 + i_t) * 16 + q_row, ROWS - 1)
        qv = fx.Vector(
            bo.buffer_load(
                r_q,
                ((r * QL + q_r // GQA) * Q_K + (h * GQA + q_r % GQA) * HEAD_DIM + q_d0)
                // 2,
                vec_width=4,
                dtype=T.i32,
            )  # fmt: skip
        )
        ids0 = load_ids(r, c_lo, c_last)
        if const_expr(early_pg is not None):
            if c_lo < c_hi:
                issue_pg(uniform(fx.Int32(early_pg)), h, 0)
        gpu.barrier()
        if tid < NT * 16 * HEAD_DIM // 8:
            fx.ptr_store(
                qv.bitcast(fx.BFloat16), qs + ((i_t * 16 + q_row) * Q_PITCH + q_d0)
            )
        store_ids(ids0)
        gpu.barrier()

        # each tile's B operands of this lane: row l16, dims 32 step + 16 g2 .. + 15 (the
        # WMMA's K a lane holds is ours to label, the K operand below takes the same dims)
        def q_op(it, step):
            p0 = qs + ((it * 16 + l16) * Q_PITCH + 32 * step + 16 * g2)
            lo = fx.Vector(fx.ptr_load(p0, result_type=V8B))
            hi = fx.Vector(fx.ptr_load(p0 + 8, result_type=V8B))
            return lo.shuffle(hi, list(range(16)))

        qa = [[q_op(it, step) for step in range(2)] for it in range(NT)]
        # S^T tiles: K is the WMMA's A operand (key l16, dims 32 step + 16 g2 ..), Q^T
        # its B (row l16): a lane holds row l16 of each row tile and keys 8 g2 .. 8 g2 + 7
        # of each key tile, so a row's running statistics are one lane's (two lanes
        # share a row), and its P values are, as they stand, the PV WMMA's A operand
        # (keys of tile 0 then tile 1; the V^T reads give the same key order). The
        # reference max of a row moves only when a chunk's max exceeds it by TAU (lazy
        # rescale, every row of the wave at once).
        r_lane = [(rt0 + it) * 16 + l16 for it in range(NT)]
        qpos = [fx.min(fx.min(r_lane[it], ROWS - 1) // GQA, QL - 1) for it in range(NT)]
        row_live = [r_lane[it] < ROWS for it in range(NT)]
        m_ref = [
            fx.Float32(NEG) for _ in range(NT)
        ]  # the row's reference max (its running max, lazily)
        l_lane = [
            fx.Float32(0.0) for _ in range(NT)
        ]  # this lane's keys' share of the row sum
        accs = [
            [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(4)] for _ in range(NT)
        ]

        # The chunk loop: a wave's 32 keys of a chunk are two K pages and two V pages
        # (1 KB each, one head), landed in LDS by async copies (16 B a lane, a page in
        # DMA_PER_PAGE copies; chunk-major: dims 16 c .. of key k at bytes 256 c + 16 k,
        # so a K operand is one 16 B read and a V^T operand one ds_load_tr8_b64 a 16 x 16
        # tile) in the wave's ring slot; two chunks are in flight while one computes.
        # The page ids of a block of IDS_CHUNKS chunks are staged in LDS first.
        assert KEY_TILES == 2 and V_STEPS == 1
        tr_w = (
            tr8_offset(lane) // 4
        )  # this lane's word in a V 16 x 16 tile (8 B aligned)

        def issue(cc, sl, b_lo):
            """Chunk cc's four pages -> ring slot sl (DMA_PER_CHUNK copies: the ring's counts)."""
            slot = wring + sl * SLOT_
            if const_expr(KV5):
                issue_pg(
                    fx.ptr_load(ids + ((cc - b_lo) * PPC5 + ks // 2)), h, sl
                )  # (uniform)
                return
            for kt in range_constexpr(KEY_TILES):
                pg = fx.ptr_load(ids + ((cc - b_lo) * PPC + 2 * ks + kt))  # uniform
                # NHD [page][key][head][dim]: the page's base (64-bit), a lane's offset in it
                row0 = fx.Int64(pg) * (16 * LOCAL_KV_HEADS * HEAD_DIM)
                kstride = LOCAL_KV_HEADS * HEAD_DIM
                hoff = h * HEAD_DIM
                if const_expr(KV_ROWS > 0 and not SL):
                    # a full layer reads the head-major pool (``kv_rows``): the head's 16 keys of a page are
                    # 1 KB contiguous. The full-only build takes it whole; the any-layer build by window == 0
                    hm0 = (fx.Int64(h) * KV_ROWS + fx.Int64(pg) * 16) * HEAD_DIM
                    if const_expr(FL):
                        row0, kstride, hoff = hm0, HEAD_DIM, fx.Int32(0)
                    else:
                        hm = window == 0
                        row0 = hm.select(hm0, row0)
                        kstride = hm.select(fx.Int32(HEAD_DIM), fx.Int32(kstride))
                        hoff = hm.select(fx.Int32(0), hoff)
                if const_expr(SWA_ROWS > 0 and not FL):
                    # a sliding layer reads the head-major SWA pool (``swa_rows``) the same way: the
                    # sliding-only build whole, the any-layer build by window > 0
                    hs0 = (fx.Int64(h) * SWA_ROWS + fx.Int64(pg) * 16) * HEAD_DIM
                    if const_expr(SL):
                        row0, kstride, hoff = hs0, HEAD_DIM, fx.Int32(0)
                    else:
                        hs = window > 0
                        row0 = hs.select(hs0, row0)
                        kstride = hs.select(fx.Int32(HEAD_DIM), fx.Int32(kstride))
                        hoff = hs.select(fx.Int32(0), hoff)
                for j in range_constexpr(DMA_PER_PAGE):
                    # lane -> (key, dims chunk) of the page's j-th 16 B a lane block
                    blk = j * WAVE + lane
                    voff = (blk % 16) * kstride + hoff + (blk // 16) * 16
                    lds_dma_16(
                        k_base + row0, slot + (kt * PAGE_WORDS + j * WAVE * 4), voff
                    )
                    lds_dma_16(
                        v_base + row0,
                        slot
                        + ((KEY_TILES + kt) * PAGE_WORDS + kt * V_SKEW + j * WAVE * 4),
                        voff,
                    )

        def pair_max(v):
            """The max over the two lanes sharing a row (l16, l16 + 16)."""
            x, y = permlane_swap(16, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
            return fx.max(x.bitcast(fx.Float32), y.bitcast(fx.Float32))

        def pair_sum(v):
            x, y = permlane_swap(16, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
            return x.bitcast(fx.Float32) + y.bitcast(fx.Float32)

        NV = KEY_TILES * 8  # scores a lane a row tile: keys 8 g2 + e of each key tile

        def key_of(j):
            """Key offset in the wave's 32 of a lane's score j."""
            return (j // 8) * TILE + 8 * g2 + j % 8

        n_blk = 1 if SL else (c_hi - c_lo + IDS_CHUNKS - 1) // IDS_CHUNKS
        stamp(c, 3)  # a task's chunk loop begins (the last task's stamps survive)
        for b in range(0, n_blk):
            b_lo = c_lo + b * IDS_CHUNKS
            b_hi = b_lo + 1 if SL else fx.min(b_lo + IDS_CHUNKS, c_hi)
            b_last = b_hi - 1
            if b > 0:  # (the first block's ids came with Q)
                gpu.barrier()  # the previous block's ids are consumed
                store_ids(load_ids(r, b_lo, c_last))
                gpu.barrier()
                if const_expr(early_pg is not None):
                    issue(b_lo, 0, b_lo)
            if const_expr(early_pg is None):
                issue(b_lo, 0, b_lo)
            if const_expr(not SL):
                if b_lo < b_last:
                    issue(b_lo + 1, 1, b_lo)
            for cc in range(b_lo, b_hi):  # (sliding-only: one chunk, b_hi = b_lo + 1)
                sl = (cc - b_lo) % 2
                slot = wring + sl * SLOT_
                key0 = cc * CHUNK + ks * KEYS_PER_WAVE
                # chunk cc's pages landed (cc + 1's in flight, unless cc is the last)
                if const_expr(SL):
                    wait_lds_dma(0)  # (the task's one chunk)
                elif cc < b_last:
                    wait_lds_dma(RING_WAIT_)
                else:
                    wait_lds_dma(0)
                # the chunk's K operands (tile kt, step: key l16, dims 32 step + 16 g2 ..)
                kb = []
                for kt in range_constexpr(KEY_TILES):
                    for step in range_constexpr(2):
                        if const_expr(KV5):
                            # key 16 kt + l16 of the wave's 32, dim chunks 4 step + 2 g2 and the next: 16 bf16
                            dcb, key = 4 * step + 2 * g2, kt * TILE + l16
                            w0 = fx.Vector(
                                fx.ptr_load(
                                    slot + (dcb * KEYS_PER_WAVE + key) * 4,
                                    result_type=V4I,
                                )
                            )
                            w1 = fx.Vector(
                                fx.ptr_load(
                                    slot + ((dcb + 1) * KEYS_PER_WAVE + key) * 4,
                                    result_type=V4I,
                                )
                            )
                            kb.append(
                                w0.shuffle(w1, list(range(8))).bitcast(fx.BFloat16)
                            )
                        else:
                            widx = (
                                slot
                                + kt * PAGE_WORDS
                                + ((2 * step + g2) * 256 + l16 * 16) // 4
                            )
                            w = fx.Vector(fx.ptr_load(widx, result_type=V4I))
                            kb.append(fp8_words_to_bf16([w[e] for e in range(4)]))
                # ---- S^T: tile (it, kt): row l16, keys 8 g2 + e of key tile kt; scaled and masked
                edge = (
                    (cc == c_last) if FL else ((cc == c_last) | (cc == lo))
                )  # a chunk that may hold masked keys
                sc = []  # [it][kt * 8 + e]
                mx = []
                for it in range_constexpr(NT):
                    sts = []
                    for kt in range_constexpr(KEY_TILES):
                        st = fx.Vector.filled(8, 0.0, fx.Float32)
                        for step in range_constexpr(2):
                            st = wmma_bf16(kb[kt * 2 + step], qa[it][step], st)
                        sts.append(st)
                    raw = [fx.Float32(sts[j // 8][j % 8]) * qk_scale for j in range(NV)]
                    if const_expr(SL):
                        # (the one chunk is an edge chunk: masked; the window is on)
                        vals = [
                            (
                                row_live[it]
                                & (key0 + key_of(j) < ctx + qpos[it] + 1)
                                & ((ctx + qpos[it] - (key0 + key_of(j))) < window)
                            ).select(raw[j], fx.Float32(NEG))
                            for j in range(NV)
                        ]
                    else:
                        vals = [
                            row_live[it].select(raw[j], fx.Float32(NEG))
                            for j in range(NV)
                        ]
                        if edge:  # (wave-uniform: only a chunk that may hold masked keys tests them)
                            if const_expr(FL):
                                vals = [
                                    (
                                        row_live[it]
                                        & (key0 + key_of(j) < ctx + qpos[it] + 1)
                                    ).select(raw[j], fx.Float32(NEG))
                                    for j in range(NV)
                                ]
                            else:
                                vals = [
                                    (
                                        row_live[it]
                                        & (key0 + key_of(j) < ctx + qpos[it] + 1)
                                        & (
                                            (window == 0)
                                            | (
                                                (ctx + qpos[it] - (key0 + key_of(j)))
                                                < window
                                            )
                                        )
                                    ).select(raw[j], fx.Float32(NEG))
                                    for j in range(NV)
                                ]
                    sc.append(vals)
                    m8 = vals[0]
                    for j in range_constexpr(1, NV):
                        m8 = fx.max(m8, vals[j])
                    mx.append(pair_max(m8))
                # ---- the lazy rescale: when any row's chunk max exceeds its reference by TAU,
                # every row takes its new max and its sum and O are rescaled (the step is
                # wave-uniform: a branch, skipped in the common chunk that moves nothing)
                if const_expr(SL):
                    # (the task's first and only chunk: the reference max is the chunk's when any row of
                    # the wave has a key (as the lazy step takes it then), and O / l are still zero)
                    need = mx[0] > m_ref[0] + TAU
                    for it in range_constexpr(1, NT):
                        need = need | (mx[it] > m_ref[it] + TAU)
                    move = ballot(need) != 0
                    for it in range_constexpr(NT):
                        m_ref[it] = move.select(fx.max(m_ref[it], mx[it]), m_ref[it])
                    move = False  # (no rescale: a static branch)
                else:
                    need = mx[0] > m_ref[0] + TAU
                    for it in range_constexpr(1, NT):
                        need = need | (mx[it] > m_ref[it] + TAU)
                    move = ballot(need) != 0
                if move:  # (else every alpha is 1: nothing to rescale)
                    for it in range_constexpr(NT):
                        m_new = fx.max(m_ref[it], mx[it])
                        m_g = (m_new > fx.Float32(NO_MAX)).select(
                            m_new, fx.Float32(0.0)
                        )
                        alpha = hw_exp2(m_ref[it] - m_g)  # 0 from NEG (a first chunk)
                        l_lane[it] = l_lane[it] * alpha
                        m_ref[it] = m_new
                        # O rows 8 g2 + e take the alpha of row 8 g2 + e: from lane 8 g2 + e
                        a_rows = [
                            fx.Int32(
                                rocdl.ds_bpermute(
                                    T.i32, (8 * g2 + e) * 4, alpha.bitcast(fx.Int32)
                                )
                            ).bitcast(fx.Float32)
                            for e in range(8)
                        ]
                        al = fx.Vector.from_elements(a_rows, fx.Float32)
                        accs[it] = [accs[it][dt] * al for dt in range(4)]
                # ---- P = exp2(S - m) as the PV A operand (16 bf16 a lane), the lane's row sums
                p_ops = []
                for it in range_constexpr(NT):
                    m_g = (m_ref[it] > fx.Float32(NO_MAX)).select(
                        m_ref[it], fx.Float32(0.0)
                    )
                    ps = [hw_exp2(sc[it][j] - m_g) for j in range(NV)]
                    acc_l = ps[0]
                    for j in range_constexpr(1, len(ps)):
                        acc_l = acc_l + ps[j]
                    l_lane[it] = l_lane[it] + acc_l
                    p_ops.append(
                        fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16)
                    )
                # ---- PV: V^T operands by transposed LDS reads (dim tile dt: lane (g2, l16) gets
                # dim 16 dt + l16 at keys 8 g2 .. + 7 of page 0 then of page 1)
                b_v = []
                for dt in range_constexpr(4):
                    if const_expr(KV5):
                        # dim 16 dt + l16 at keys 8 g2 .. + 7 of key tile 0 (key group g2 of the wave's four), then of
                        # key tile 1 (group 2 + g2): two 16 B units, the PV operand as it stands
                        v0 = fx.Vector(
                            fx.ptr_load(
                                slot
                                + (KV5_K_WORDS + (g2 * HEAD_DIM + 16 * dt + l16) * 4),
                                result_type=V4I,
                            )
                        )
                        v1 = fx.Vector(
                            fx.ptr_load(
                                slot
                                + (
                                    KV5_K_WORDS
                                    + ((2 + g2) * HEAD_DIM + 16 * dt + l16) * 4
                                ),
                                result_type=V4I,
                            )
                        )
                        b_v.append(v0.shuffle(v1, list(range(8))).bitcast(fx.BFloat16))
                    else:
                        words = []
                        for pgi in range_constexpr(KEY_TILES):
                            vp = (
                                slot
                                + (
                                    (KEY_TILES + pgi) * PAGE_WORDS
                                    + pgi * V_SKEW
                                    + dt * 64
                                )
                                + tr_w
                            )
                            w = fx.Vector(
                                rocdl_raw.ds_load_tr8_b64(T.vec(2, T.i32), vp.llvm_ptr)
                            )
                            words += [w[0], w[1]]
                        b_v.append(fp8_words_to_bf16(words))
                accs = [
                    [wmma_bf16(p_ops[it], b_v[dt], accs[it][dt]) for dt in range(4)]
                    for it in range(NT)
                ]
                rocdl.s_wait_dscnt(0)  # every read of the slot is done
                if const_expr(not SL):
                    if cc + 2 <= b_last:
                        issue(cc + 2, sl, b_lo)  # chunk cc + 2 into this slot
                if cc == b_lo:
                    stamp(c, 4)  # the first chunk done
        stamp(c, 5)  # the chunk loop done
        gpu.barrier()  # every wave is done with the ring: the merge tree reuses the LDS

        # ---- the key splits merge in LDS in one round: every wave writes its
        # (m, l, O) rows (a (split, tile, row) record is COMB_ROW floats, O from
        # O_OFF); then each wave owns RPW of the task's rows, folds the splits
        # of them (a lane: one row, DPL dims) and publishes them with vector stores
        def comb_at(slot, it, row):
            return comb + ((slot * NT + it) * 16 + row) * COMB_ROW

        for it in range_constexpr(NT):
            l_row = pair_sum(l_lane[it])
            if g2 == 0:
                base = comb_at(ks, it, l16)
                fx.ptr_store(m_ref[it], base)
                fx.ptr_store(l_row, base + 1)
            for e in range_constexpr(8):
                base = comb_at(ks, it, 8 * g2 + e)
                for dt in range_constexpr(4):
                    fx.ptr_store(
                        fx.Float32(accs[it][dt][e]), base + (O_OFF + dt * 16 + l16)
                    )
        gpu.barrier()
        RPW = NT * 16 // WAVES  # rows a wave
        assert RPW >= 1 and WAVE % RPW == 0, (NT, WAVES)
        LPR = WAVE // RPW  # lanes a row
        DPL = HEAD_DIM // LPR  # dims a lane
        VD = fx.Vector.make_type(DPL, fx.Float32)
        trow = wave * RPW + lane // LPR  # the task row (0 .. NT 16)
        it_f, row_f = trow // 16, trow % 16
        dd = (lane % LPR) * DPL
        m_f = fx.Float32(NEG)
        l_f = fx.Float32(0.0)
        o_f = fx.Vector.filled(DPL, 0.0, fx.Float32)
        for src in range_constexpr(KEY_SPLITS):
            base = comb_at(src, it_f, row_f)
            m2 = fx.ptr_load(base)
            l2 = fx.ptr_load(base + 1)
            v2 = fx.Vector(fx.ptr_load(base + (O_OFF + dd), result_type=VD))
            mn = fx.max(m_f, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m_f - mg)
            a2 = hw_exp2(m2 - mg)
            m_f, l_f, o_f = mn, l_f * a1 + l2 * a2, o_f * a1 + v2 * a2
        # ---- the task's partial rows: plain fp32 words, then the flag
        row = rt0 * 16 + trow
        if row < ROWS:
            base = part_row(r, h, p, row)
            if lane % LPR == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([m_f, l_f], fx.Float32),
                    r_part,
                    base,
                    cache_modifier=CM_DEV,
                )
            bo.buffer_store(o_f, r_part, base + O_OFF + dd, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(pflag, pflag_at(r, h, rt0, p), fx.Int32(1))

    # the tasks, request-major over the requests' own partition counts (a short
    # request has few): job -> (request, head, tile pair, partition) through the
    # requests' cumulative task counts, so a CTA iterates real tasks only
    if const_expr(FL):
        # the job mapping's partitions a (request, head, tile group), rounded up to whole groups of 8
        # past DIRECT_MAX when every padded task still has its own CTA (a pad's task runs no chunk and
        # publishes an empty partial nobody reads): a partition's TASK_TILES tasks AND its two heads'
        # tasks then sit a multiple of 8 jobs apart, on one XCD, sharing the K / V lines in its L2
        # (an unpadded partition count splits both pairings, which costs time)
        pads = [((n_parts_of(rr) + 7) // 8) * 8 for rr in range(R)]
        tot_pad = pads[0]
        for rr in range_constexpr(1, R):
            tot_pad = tot_pad + pads[rr]
        use_pad = tot_pad * (LOCAL_KV_HEADS * TASK_TILES) <= BLOCKS
        counts = [
            (use_pad & (n_parts_of(rr) > DIRECT_MAX)).select(pads[rr], n_parts_of(rr))
            * (LOCAL_KV_HEADS * TASK_TILES)
            for rr in range(R)
        ]
    else:
        counts = [n_parts_of(rr) * (LOCAL_KV_HEADS * TASK_TILES) for rr in range(R)]
    starts = [fx.Int32(0)]
    for rr in range_constexpr(1, R):
        starts.append(starts[-1] + counts[rr - 1])
    total = starts[-1] + counts[-1]
    for job in range(bid, total, BLOCKS):
        r = fx.Int32(0)
        for rr in range_constexpr(1, R):
            r = (job >= starts[rr]).select(fx.Int32(rr), r)
        base = starts[0]
        for rr in range_constexpr(1, R):
            base = (r == rr).select(starts[rr], base)
        np_r = n_parts_of(r)
        if const_expr(FL):
            end = total
            for rr in range_constexpr(R - 1):
                end = (r == rr).select(starts[rr + 1], end)
            np_r = (end - base) // (
                LOCAL_KV_HEADS * TASK_TILES
            )  # (the mapping's, padded)
        local = job - base
        # a partition's TASK_TILES tasks (its row tiles' groups) sit 8 jobs apart, on
        # CTAs of one XCD (CTAs go to the XCDs in turn): they stream the same K / V
        # pages at about the same time, so the later reads hit that XCD's L2 (before,
        # the tasks were np_r apart, on different XCDs, each reading the pages from
        # HBM). Groups of 8 partitions x TASK_TILES jobs; the last np_r % 8 partitions
        # in a tail group.
        if const_expr(ATTN_FDIV):
            tn = TASK_TILES * np_r
            qj = _fdiv(local, tn)
            j = local - qj * tn
            h = qj % LOCAL_KV_HEADS
        else:
            j = local % (TASK_TILES * np_r)
            h = (local // (TASK_TILES * np_r)) % LOCAL_KV_HEADS
        full = (np_r // 8) * 8
        g, w = j // (8 * TASK_TILES), j % (8 * TASK_TILES)
        jt = j - TASK_TILES * full
        tail = fx.max(np_r - full, fx.Int32(1))
        in_full = j < TASK_TILES * full
        if const_expr(ATTN_FDIV):
            qt = _fdiv(
                fx.max(jt, fx.Int32(0)), tail
            )  # (jt < 0 on a full-group job: its values unused)
            p = in_full.select(g * 8 + w % 8, full + (jt - qt * tail))
            rt0 = in_full.select(w // 8, qt) * NT
        else:
            p = in_full.select(g * 8 + w % 8, full + jt % tail)
            rt0 = in_full.select(w // 8, jt // tail) * NT
        attn_task(r, h, rt0, p)
    stamp(c, 1)  # the attention tasks done

    # ============================================ merge jobs
    # a thread: row 16 rt + tid / 32, dims 8 ((tid / 4) % 8) .., partition lane tid % 4
    row16 = tid // 32
    dch = (tid // 4) % 8
    pl = tid % 4
    d0 = dch * 8
    # the merge jobs count down from below K2's chain CTAs (the last MAX_TOKENS run K2's
    # boundary parts first thing: a merge there delayed them), the task loop's extra
    # tasks sit on the first (``merge_ctas``)
    M_CTAS = MERGE_TOP
    mbid = M_CTAS - 1 - bid
    n_l1 = R * LOCAL_KV_HEADS * ROW_TILES * GROUPS
    n_l2 = R * LOCAL_KV_HEADS * ROW_TILES

    def load_rec(r_src, base):
        """A record's (m, l) and this thread's 8 dims of O (plain words)."""
        ml = fx.Vector(
            bo.buffer_load(r_src, base, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
        )
        o0 = fx.Vector(
            bo.buffer_load(
                r_src,
                base + O_OFF + d0,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        o1 = fx.Vector(
            bo.buffer_load(
                r_src,
                base + O_OFF + d0 + 4,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        return ml, o0, o1

    def fold_rec(m, l, o0, o1, rec, ok):
        ml, p0, p1 = rec
        m2, l2 = fx.Float32(ml[0]), fx.Float32(ml[1])
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2

    def combine_lanes(m, l, o0, o1):
        """The four partition lanes of a (row, 8 dims) fold in LDS: every lane folds
        the four in lane order (the same result on each; lane 0 stores)."""
        slot = red + tid * 10
        fx.ptr_store(m, slot)
        fx.ptr_store(l, slot + 1)
        for j in range_constexpr(4):
            fx.ptr_store(fx.Float32(o0[j]), slot + (2 + j))
            fx.ptr_store(fx.Float32(o1[j]), slot + (6 + j))
        gpu.barrier()
        base = red + (tid - pl) * 10
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        for src in range_constexpr(MERGE_LANES):
            s2 = base + src * 10
            m2 = fx.ptr_load(s2)
            l2 = fx.ptr_load(s2 + 1)
            p0 = fx.Vector.from_elements(
                [fx.ptr_load(s2 + (2 + j)) for j in range(4)], fx.Float32
            )
            p1 = fx.Vector.from_elements(
                [fx.ptr_load(s2 + (6 + j)) for j in range(4)], fx.Float32
            )
            mn = fx.max(m, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m - mg)
            a2 = hw_exp2(m2 - mg)
            m, l, o0, o1 = mn, l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2
        gpu.barrier()  # ``red`` is reused by the CTA's next job
        return m, l, o0, o1

    def merge_l1(r, h, rt, g):
        """Level-1 job: partitions [g PPG, (g + 1) PPG) of (request r, head h) for
        row tile rt, folded four lanes a (row, 8 dims) -> a level-2 record."""
        np_r = n_parts_of(r)
        ppg = (np_r + GROUPS - 1) // GROUPS
        p_lo = g * ppg
        cnt = fx.max(fx.min(np_r - p_lo, ppg), fx.Int32(0))
        poll_flags(c, [(pflag, pflag_at(r, h, rt, p_lo), cnt)])
        row = fx.min(rt * 16 + row16, ROWS - 1)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        # four partitions a lane in flight a round
        for k0 in range(0, (ppg + MERGE_LANES * 4 - 1) // (MERGE_LANES * 4)):
            ps = [p_lo + (k0 * 4 + k) * MERGE_LANES + pl for k in range(4)]
            recs = [
                load_rec(r_part, part_row(r, h, fx.min(pk, p_lo + cnt - 1), row))
                for pk in ps
            ]
            for k in range_constexpr(4):
                m, l, o0, o1 = fold_rec(m, l, o0, o1, recs[k], ps[k] < p_lo + cnt)
        m, l, o0, o1 = combine_lanes(m, l, o0, o1)
        if (pl == 0) & (rt * 16 + row16 < ROWS):
            base = l2_row(r, h, rt, g, row16)
            if dch == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([m, l], fx.Float32),
                    r_l2,
                    base,
                    cache_modifier=CM_DEV,
                )
            bo.buffer_store(o0, r_l2, base + O_OFF + d0, cache_modifier=CM_DEV)
            bo.buffer_store(o1, r_l2, base + O_OFF + d0 + 4, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(
                l2flag,
                ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g,
                fx.Int32(1),
            )

    def merge_l2(r, h, rt):
        """Level-2 job: the groups' records (or, with at most DIRECT_MAX partitions,
        the partitions themselves) of (request r, head h, row tile rt) -> o rows."""
        np_r = n_parts_of(r)
        direct = True if SL else np_r <= DIRECT_MAX  # (sliding-only: <= 9 partitions)
        row = fx.min(rt * 16 + row16, ROWS - 1)
        og = OG_ONCE and o_flag is not None
        if const_expr(og):
            # the gate of this thread's (row, 8 dims), ahead of the flags' wait
            g_at = (r * QL + row // GQA) * Q_K + (h * GQA + row % GQA) * HEAD_DIM + d0
            g_w = fx.Vector(
                bo.buffer_load(rsrc(a["g"]), g_at // 2, vec_width=4, dtype=T.i32)
            )
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        # the source records: the partitions themselves (at most GROUPS of them), or
        # the level-1 records; one resource over the selected base address
        if const_expr(SL):
            poll_flags(c, [(pflag, pflag_at(r, h, rt, 0), np_r)])
            r_src = rsrc(part_w.value)
        else:
            if direct:
                poll_flags(c, [(pflag, pflag_at(r, h, rt, 0), np_r)])
            else:
                poll_flags(
                    c,
                    [
                        (
                            l2flag,
                            ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS,
                            GROUPS,
                        )
                    ],
                )
            r_src = rsrc(direct.select(part_w.value, l2_w.value))
        for k in range_constexpr(DIRECT_MAX // MERGE_LANES):
            g = k * MERGE_LANES + pl  # a partition (direct) or a group (g < GROUPS)
            if const_expr(SL):
                idx = part_row(r, h, fx.min(g, np_r - 1), row)
                ok = g < np_r
            else:
                idx = direct.select(
                    part_row(r, h, fx.min(g, np_r - 1), row),
                    l2_row(r, h, rt, fx.min(g, GROUPS - 1), row16),
                )
                ok = direct.select(g < np_r, g < GROUPS)
            m, l, o0, o1 = fold_rec(m, l, o0, o1, load_rec(r_src, idx), ok)
        m, l, o0, o1 = combine_lanes(m, l, o0, o1)
        if const_expr(og):
            # the og row: o as its bf16, the head's RMSNorm without weight (the og rows' sum: four dims, then
            # the butterfly over the head's sixteen groups, here two groups a lane), x sigmoid(gate), one bf16
            # rounding; every lane of a (row, 8 dims) holds the same values
            g_inv = fx.Float32(1.0) / l
            g_of = [bf16_round(fx.Float32(o0[j]) * g_inv) for j in range(4)] + [bf16_round(fx.Float32(o1[j]) * g_inv) for j in range(4)]  # fmt: skip
            s_lo = fx.Float32(0.0)
            s_hi = fx.Float32(0.0)
            for j in range_constexpr(4):
                s_lo = s_lo + g_of[j] * g_of[j]
                s_hi = s_hi + g_of[4 + j] * g_of[4 + j]
            s_lo = butterfly(s_lo, (16, 8, 4))
            s_hi = butterfly(s_hi, (16, 8, 4))
            g_rstd = hw_rsq((s_lo + s_hi) / float(HEAD_DIM) + RMS_EPS)
            g_in = [bf2_f32(fx.Int32(g_w[q]))[e] for q in range(4) for e in range(2)]
            g_vals = [g_of[j] * g_rstd * (fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(g_in[j] * fx.Float32(-LOG2E)))) for j in range(8)]  # fmt: skip
            g_words = [
                bf16_pair(g_vals[2 * q], g_vals[2 * q + 1]).bitcast(fx.Int32)
                for q in range(4)
            ]
            if (pl == 0) & (rt * 16 + row16 < ROWS):
                bo.buffer_store(fx.Vector.from_elements(g_words, fx.Int32), rsrc(a["o"]), g_at // 2, cache_modifier=CM_DEV)  # fmt: skip
        else:
            if (pl == 0) & (rt * 16 + row16 < ROWS):
                inv = fx.Float32(1.0) / l
                t = r * QL + (rt * 16 + row16) // GQA
                ov = fx.Vector.from_elements([fx.Float32(o0[j]) for j in range(4)] + [fx.Float32(o1[j]) for j in range(4)], fx.Float32)  # fmt: skip
                bo.buffer_store(
                    (ov * inv).to(fx.BFloat16),
                    rsrc(a["o"]),
                    t * Q_K + (h * GQA + (rt * 16 + row16) % GQA) * HEAD_DIM + d0,
                    cache_modifier=CM_DEV,
                )
        if const_expr(o_flag is not None):
            # the (request, head, row tile)'s o rows are out: its flag, for the og tasks
            release_stores()
            gpu.barrier()
            if tid == 0:
                put(
                    o_flag,
                    ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * FLAG_STRIDE,
                    fx.Int32(1),
                )

    for j in range(
        (mbid < 0).select(fx.Int32(n_l1), mbid), n_l1 if not SL else 0, M_CTAS
    ):
        g = j % GROUPS
        rt = (j // GROUPS) % ROW_TILES
        h = (j // (GROUPS * ROW_TILES)) % LOCAL_KV_HEADS
        r = j // (GROUPS * ROW_TILES * LOCAL_KV_HEADS)
        if (
            n_parts_of(r) > DIRECT_MAX
        ):  # else the level-2 job folds the partitions itself
            merge_l1(r, h, rt, g)
    for j2 in range((mbid < 0).select(fx.Int32(n_l2), mbid), n_l2, M_CTAS):
        rt = j2 % ROW_TILES
        h = (j2 // ROW_TILES) % LOCAL_KV_HEADS
        r = j2 // (ROW_TILES * LOCAL_KV_HEADS)
        merge_l2(r, h, rt)
    stamp(c, 2)  # the merge done (end)
