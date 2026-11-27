# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 draft layer's attention half around D-attn, as two launches on a
rank of a TP group, for S = ``tokens`` rows (a block of 8 consecutive rows a
request)::

  D-pre : res' = first ? x : bf16(res + x)         (the input norm's fused add)
          h    = RMSNorm(res') w_in                 (eps 1e-5, bf16)
          delta = h W_kp^T                           (2048: the attention conv's coefficients, both sides -> ``coef``)
          hc   = conv0(h)                            (side 0)
          qkv  = hc W_qkv^T                          (1280 rows a rank: 8 q heads, k, v of 128)
          q, k : per-head RMSNorm (weights, eps 1e-5) -> bf16 -> neox RoPE from the table
          q -> ``q`` bf16 [S, 1024]; k, v -> the layer's fp8 cache at the rows' slots
  D-post: d   = o W_o^T                              (this rank's partial, 8192)
          d   = sum over the ranks (bf16 pairs through the peer regions, rank order)
          attn_out = conv1(d) with side 1's coefficients (bf16 [S, 8192])

The stages, mailboxes, the K-part / column-block plan and the GEMV machinery are
D-MLP's (``draft_mlp.py``): the qkv GEMV is a K-part GEMV like the gate / up one,
the o GEMV a column-block GEMV over the whole K like the down one.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.arch import WAVE
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    lane_gather,
    peer_bases,
    rsrc,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx950 import compat
from sgl_amd_mono.muse_spark.gfx1250.config import MAX_TOKENS
from sgl_amd_mono.muse_spark.gfx1250.device import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    poll_flags,
    release_stores,
    stamp,
    step_tag,
)
from sgl_amd_mono.muse_spark.gfx1250.draft_mlp import (
    BLOCK,
    COEF_ROWS,
    EPS,
    GROUP,
    GROUPS,
    K_PART,
    K_PARTS,
    MAXC,
    MAXG,
    RED_FLOATS,
    RGS,
    RING_BYTES,
    ROWS,
    TAPS,
    XP,
    XPAD,
    H,
    cf1_item,
    coef_shard,
    col_block,
    conv0_base,
    gemv,
    kpart_tiles,
    lds_rows_b,
    stage_coef,
    stage_coef_reduce,
    stage_conv0,
    stage_norm,
    tp_sum_conv1,
)
from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes

BRIDGE = compat.install()
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

# the attention geometry (a rank of TP 8)
HEAD_DIM = 128
Q_HEADS = 8
Q_K = Q_HEADS * HEAD_DIM  # 1024
QKV_ROWS = Q_K + 2 * HEAD_DIM  # 1280: q heads, k, v
QKV_HEADS = Q_HEADS + 2  # head-rows a token of the qkv reduce
XO = Q_K + XPAD  # the o rows' LDS pitch
DIMS = HEAD_DIM // WAVE  # dims a lane of a head-row: lane + WAVE m
assert HEAD_DIM == 2 * 64 and DIMS % 2 == 0


def dp_scratch_layout() -> dict:
    """D-pre's scratch regions (pairs): the K parts' sums of squares, the coefficient
    partials, the qkv partials, the trace."""
    return pair_layout(
        [
            ("dp_ss", K_PARTS * MAX_TOKENS),
            ("dp_cpart", K_PARTS * MAX_TOKENS * COEF_ROWS),
            ("dp_coef", MAX_TOKENS * COEF_ROWS),
            ("dp_apart", K_PARTS * MAX_TOKENS * QKV_ROWS),
            ("dp_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def do_scratch_layout() -> dict:
    """D-post's scratch regions (pairs): the trace (the o GEMV sums its K itself)."""
    return pair_layout([("do_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS)], align=256)


def do_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric regions: ``do`` [source rank][token][H / 2] bf16 pairs,
    ``dpc`` D-pre's coefficients [token][COEF_ROWS] (a launch that shards them),
    ``dao`` attn_out [token][H / 2] bf16 pairs (each column block's owner pushes it)."""
    return pair_layout(
        [
            ("do", npes * MAX_TOKENS * H // 2),
            ("dpc", MAX_TOKENS * COEF_ROWS),
            ("dao", MAX_TOKENS * H // 2),
        ],
        align=256,
    )


def do_sym_bytes(npes: int) -> int:
    return scratch_bytes(do_sym_layout(npes))


def dp_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        ring: fx.Array[fx.Int32, RING_BYTES // 4, 16]
        red: fx.Array[fx.Float32, RED_FLOATS, 16]
        xa: fx.Array[fx.BFloat16, 2 * S * XP, 16]  # h, then hc (the qkv GEMV's B)
        ssl: fx.Array[fx.Float32, S * K_PARTS, 16]
        ssw: fx.Array[fx.Float32, S * 2, 16]
        nf: fx.Array[fx.Float32, S, 16]

    return Lds


def do_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        ring: fx.Array[fx.Int32, RING_BYTES // 4, 16]
        red: fx.Array[fx.Float32, RED_FLOATS, 16]
        xo: fx.Array[
            fx.BFloat16, S * XO, 16
        ]  # the attention output rows (the o GEMV's B)
        dbuf: fx.Array[fx.Float32, S * MAXC, 16]
        cf1: fx.Array[fx.Float32, S * TAPS * MAXG, 16]

    return Lds


def _pick(pair, idx):
    """Element ``idx % 2`` of a (lo, hi) bf16 pair for a dynamic ``idx``."""
    return (idx % 2 == 0).select(pair[0], pair[1])


def _cs(r_cs, idx):
    """Element ``idx`` of the rotary table (fp32 [positions, 128]: cos then sin)."""
    return fx.Float32(bo.buffer_load(r_cs, idx, vec_width=1, dtype=T.f32))


# ---------------------------------------------------------------- D-pre stages
@flyc.jit
def stage_qkv(c, bid):
    """The qkv GEMV task: this CTA's row tiles of W_qkv over its K part against hc;
    fp32 partials to ``apart`` [kp][token][row]."""
    put, apart = c["put"], c["apart"]
    kp, lo, n, mt = kpart_tiles(c, QKV_ROWS // ROWS, bid)

    def publish(r, tk, v):
        put(apart, (kp * MAX_TOKENS + tk) * QKV_ROWS + r, v)

    gemv(
        c,
        c["args"]["w_qkv"],
        H,
        kp * K_PART,
        K_PART,
        lo,
        n,
        mt,
        lds_rows_b(c, c["xhc"], XP),
        publish,
    )


@flyc.jit
def stage_heads_w(c, bid):
    """Head-rows (token, head) 10 S in all, one a wave (CTA ``bid``'s wave w takes job
    bid + w BLOCKS), on the wave's lanes (no LDS, no barrier): lane l holds dims l + WAVE m of the head-row. The K parts' qkv partials
    summed in part order, bf16 (the GEMM's output); q / k: the head's RMSNorm with its
    weight, bf16, neox RoPE (dims d, d + 64: one lane's) from the table at the row's
    position, bf16; q -> ``q`` (bf16 pairs), k / v -> the fp8 cache at the row's slot
    (quads: the neighbours' outputs by lane gathers)."""
    S, lane, wave, poll = c["S"], c["lane"], c["wave"], c["poll"]
    apart = c["apart"]
    a = c["args"]
    r_cs = rsrc(a["cos_sin"])
    r_q = rsrc(a["q"])
    xcm = c.get("xcm", 0)
    N = QKV_HEADS * S
    if wave < -(
        -N // BLOCKS
    ):  # (a job a wave: the CTAs' first waves, then their second ..)
        for p in range_constexpr(1):
            job = bid + wave * BLOCKS
            live = job < N
            jb = fx.min(job, N - 1)
            t, hd = jb // QKV_HEADS, jb % QKV_HEADS
            row0 = hd * HEAD_DIM
            ds = [lane + WAVE * m for m in range(DIMS)]
            got = poll(
                [(apart, (kp * MAX_TOKENS + t) * QKV_ROWS + row0 + d, 1) for d in ds for kp in range(K_PARTS)]
            )  # fmt: skip
            vs = []
            for m in range_constexpr(DIMS):
                v = got[m * K_PARTS][0].bitcast(fx.Float32)
                for kp in range_constexpr(1, K_PARTS):
                    v = v + got[m * K_PARTS + kp][0].bitcast(fx.Float32)
                vs.append(bf16_round(v))
            is_v = hd == QKV_HEADS - 1
            is_k = hd == QKV_HEADS - 2
            sq = vs[0] * vs[0]
            for m in range_constexpr(1, DIMS):
                sq = sq + vs[m] * vs[m]
            tot = butterfly(sq, (1, 2, 4, 8, 16))
            inv = hw_rsq(tot / float(HEAD_DIM) + EPS)
            r_nw = rsrc(is_k.select(a["k_norm"], a["q_norm"]))
            vals = []
            for m in range_constexpr(DIMS):
                d = ds[m]
                wn = fx.Float32(_pick(bf2_f32(fx.Int32(bo.buffer_load(r_nw, d // 2, vec_width=1, dtype=T.i32))), d % 2))  # fmt: skip
                vals.append(is_v.select(vs[m], bf16_round(vs[m] * inv * wn)))
            pos = fx.Int32(bo.buffer_load(rsrc(a["positions"]), (a["pos64"] != 0).select(2 * t, t), vec_width=1, dtype=T.i32))  # fmt: skip
            HM = DIMS // 2  # the first half's dims a lane (d < 64), partners m + HM
            outs = [None] * DIMS
            for m in range_constexpr(HM):
                half = lane + WAVE * m
                cs = _cs(r_cs, pos * HEAD_DIM + half)
                sn = _cs(r_cs, pos * HEAD_DIM + 64 + half)
                x1, x2 = vals[m], vals[m + HM]
                outs[m] = is_v.select(x1, bf16_round(x1 * cs - x2 * sn))
                outs[m + HM] = is_v.select(x2, bf16_round(x2 * cs + x1 * sn))

            def nb(x, k):
                return lane_gather(
                    x.bitcast(fx.Int32), fx.min(lane + k, fx.Int32(WAVE - 1))
                ).bitcast(fx.Float32)

            nbs = [[nb(o, k) for k in range(1, 4)] for o in outs]
            if live:
                if hd < Q_HEADS:
                    if (lane % 2) == 0:
                        for m in range_constexpr(DIMS):
                            bo.buffer_store(fx.Vector.from_elements([outs[m], nbs[m][0]], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_q, (t * Q_K + row0 + ds[m]) // 2, cache_modifier=xcm)  # fmt: skip
                else:
                    slot = fx.Int32(bo.buffer_load(rsrc(a["slots"]), (a["slot64"] != 0).select(2 * t, t), vec_width=1, dtype=T.i32))  # fmt: skip
                    if (lane % 4) == 0:
                        for m in range_constexpr(DIMS):
                            q4 = [outs[m]] + nbs[m]
                            packed = fp8_pack4(clamp_fp8(q4[0]), clamp_fp8(q4[1]), clamp_fp8(q4[2]), clamp_fp8(q4[3]))  # fmt: skip
                            bo.buffer_store(
                                packed,
                                rsrc(is_v.select(a["v_cache"], a["k_cache"])),
                                (slot * HEAD_DIM + ds[m]) // 4,
                                cache_modifier=xcm,
                            )


@flyc.jit
def run_dpre(c):
    bid = c["bid"]
    kp = bid % K_PARTS
    a = c["args"]
    stamp(c, 0)
    xflag = c.get("xflag")
    c["in_cm"] = c.get("xcm", 0)
    if const_expr(xflag is not None):
        # x / res from this launch's previous layer (its D-MLP CTAs of this K part)
        if a["first"] == 0:
            poll_flags(c, [(xflag, kp * RGS, RGS)])
    stage_norm(c, kp, add_first=a["first"] != 0)
    stamp(c, 1)
    stage_coef(c, bid)
    b0 = conv0_base(c, kp)
    stamp(c, 2)
    stage_coef_reduce(c, bid)
    stamp(c, 3)
    stage_conv0(c, kp, b0)
    stamp(c, 4)
    stage_qkv(c, bid)
    stamp(c, 5)
    stage_heads_w(c, bid)
    hflag, put = c.get("hflag"), c["put"]
    if const_expr(hflag is not None):
        # this CTA's q rows and block K / V are out: its flag, for the attention tasks
        release_stores()
        gpu.barrier()
        if c["tid"] == 0:
            put(hflag, bid, fx.Int32(1))
    stamp(c, 6)


# ---------------------------------------------------------------- D-post stages
@flyc.jit
def stage_o(c, bid):
    """The o GEMV task: this CTA's column block of W_o rows over the whole K (1024)
    against the attention output (``o``, or the tagged pairs of ``c["o_box"]``), staged
    in LDS rows; this rank's partial rows -> ``dbuf`` [token][MAXC]."""
    S, tid, xo, dbuf, o_box = c["S"], c["tid"], c["xo"], c["dbuf"], c.get("o_box")
    a = c["args"]
    r_o = rsrc(a["o"])
    Q = Q_K // 8  # 16 B (eight bf16) a thread item
    N = S * Q
    for q in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + q * THREADS, N - 1)
        t, k8 = idx // Q, idx % Q
        if const_expr(o_box is not None):
            got = c["poll"](
                [(o_box, (t * Q_K + 8 * k8) // 2 + 2 * h, 2) for h in range(2)]
            )
            v = fx.Vector.from_elements(
                [fx.Int32(w) for h in range(2) for w in got[h]], fx.Int32
            )
        else:
            v = fx.Vector(bo.buffer_load(r_o, (t * Q_K + 8 * k8) // 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))  # fmt: skip
        if tid + q * THREADS < N:
            fx.ptr_store(v.bitcast(fx.BFloat16), xo + (t * XO + 8 * k8))
    gpu.barrier()
    g_lo, n_g = col_block(bid)

    def publish(r, tk, v):
        fx.ptr_store(v, dbuf + (tk * MAXC + r - g_lo * GROUP))

    gemv(c, a["w_o"], Q_K, 0, Q_K, g_lo, n_g, MAXG, lds_rows_b(c, xo, XO), publish)
    gpu.barrier()


@flyc.jit
def stage_post(c, bid):
    """The block's rank partial summed over the ranks, conv1 with side 1's coefficients
    (D-pre's: the plain ``coef`` [S, 2048] fp32 tensor, or the ranks' pushed rows of
    ``c["coef_box"]``), attn_out (bf16); with ``c["ao_box"]`` (TP > 1) a reduce-scatter
    and an all-gather (``draft_mlp.tp_sum_conv1``)."""
    a = c["args"]
    box = c.get("coef_box")

    def coef_cf1(idx, g_lo):
        t, tap, g = cf1_item(idx)
        at = t * COEF_ROWS + (1 * TAPS + tap) * GROUPS + g_lo + g
        if const_expr(box is not None):
            return c["poll"]([(box, at, 1)])[0][0].bitcast(fx.Float32)
        return fx.Float32(
            bo.buffer_load(
                rsrc(a["coef"]), at, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV
            )
        )

    tp_sum_conv1(
        c,
        bid,
        a["attn_out"],
        coef_cf1,
        (c["own_do"], c["peer_do"], c.get("ao_box"), c.get("peer_ao")),
    )


@flyc.jit
def run_dpost(c, ws=None):
    """D-post's stages on this CTA's column block (``ws``: unused, the weights stream
    through the ring)."""
    bid = c["bid"]
    stamp(c, 0)
    stage_o(c, bid)
    stamp(c, 1)
    stage_post(c, bid)
    aflag, put = c.get("aflag"), c["put"]
    if const_expr(aflag is not None):
        # this CTA's attn_out columns are out: its flag, for D-MLP's norm
        release_stores()
        gpu.barrier()
        if c["tid"] == 0:
            put(aflag, bid, fx.Int32(1))
    stamp(c, 2)


# ---------------------------------------------------------------- the launches
class DPreBuild(msgspec.Struct, frozen=True):
    tokens: int
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


class DPostBuild(msgspec.Struct, frozen=True):
    tokens: int
    npes: int = 8
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


def _common(key, lds_members: dict, args: dict, tag, extra: dict, layer=None) -> dict:
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": key.tokens, "tid": tid, "bid": fx.block_idx.x, "lane": tid % WAVE, "wave": tid // WAVE,
        "put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll, "args": args, "trace_on": key.trace,
        "layer": layer,
    }  # fmt: skip
    c.update(lds_members)
    c.update(extra)
    return c


def dp_members(lds, S: int) -> dict:
    """D-pre's ``c`` entries of its LDS (``dp_lds``) and the D-MLP stage names."""
    return {
        "ring": lds.ring.ptr, "red": lds.red.ptr, "xh": lds.xa.ptr, "xhc": lds.xa.ptr + S * XP,
        "ssl": lds.ssl.ptr, "ssw": lds.ssw.ptr, "nf": lds.nf.ptr,
    }  # fmt: skip


def dp_args(args: dict) -> dict:
    """D-MLP stage arguments of D-pre: the norm's inputs, the projection's weights."""
    return {
        "norm": (args["res"], args["x"], args["w_in"], args["res_out"]),
        "w_kp": args["w_kp"], "base": args["base"],
    }  # fmt: skip


def do_members(lds) -> dict:
    return {"ring": lds.ring.ptr, "red": lds.red.ptr, "xo": lds.xo.ptr, "dbuf": lds.dbuf.ptr, "cf1": lds.cf1.ptr, "base": None}  # fmt: skip


def build_dpre(key: DPreBuild):
    """``@flyc.jit`` launcher of D-pre for ``key.tokens`` rows. Arguments: ``x`` /
    ``res`` bf16 [S, H] (``first``: res is x), ``w_in`` bf16 [H], ``w_kp`` bf16 [2048,
    H], ``base`` bf16 [2, 2, H], ``w_qkv`` bf16 [1280, H], ``q_norm`` / ``k_norm`` bf16
    [128], ``cos_sin`` fp32 [*, 128], ``positions`` (int32 / int64 [S]; ``pos64``),
    ``slots`` (int32 / int64 [S]: the rows' cache slots; ``slot64``), ``k_cache`` /
    ``v_cache`` fp8 [slots, 1, 128], ``res_out`` bf16 [S, H], ``q`` bf16 [S, 1024],
    ``coef`` fp32 [S, 2048], ``scratch``, ``epoch``, ``layer``."""
    S = key.tokens
    assert S % BLOCK == 0 and 1 <= S <= MAX_TOKENS
    name = kernel_symbol("muse_spark_dpre", s=S, f=key.fault_site, t=int(key.trace))
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = dp_scratch_layout()
    Lds = dp_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpre(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
        cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, k_cache: Int64, v_cache: Int64,
        res_out: Int64, q: Int64, coef: Int64, scratch: Int64, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "x": x, "res": res, "first": first, "w_in": w_in, "w_kp": w_kp, "base": base, "w_qkv": w_qkv,
                "q_norm": q_norm, "k_norm": k_norm, "cos_sin": cos_sin, "positions": positions, "pos64": pos64,
                "slots": slots, "slot64": slot64, "k_cache": k_cache, "v_cache": v_cache, "res_out": res_out,
                "q": q,
            }  # fmt: skip
            extra = dp_args(args)
            extra["coef_out"] = coef
            c = _common(
                key,
                dp_members(lds, S),
                args,
                step_tag(epoch, layer),
                extra,
                layer=layer,
            )
            for region, (off, _) in layout.items():
                c[region[3:]] = sreg(scratch, off, region)  # "dp_cpart" -> c["cpart"]
            coef_shard(c, 1, 0)
            run_dpre(c)

    @flyc.jit
    def launch(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
        cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, k_cache: Int64, v_cache: Int64,
        res_out: Int64, q: Int64, coef: Int64, scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dpre(
            x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin, positions, pos64, slots, slot64, k_cache,
            v_cache, res_out, q, coef, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


def build_dpost(key: DPostBuild):
    """``@flyc.jit`` launcher of D-post for ``key.tokens`` rows on a rank of a
    ``key.npes``-way TP group. Arguments: ``o`` bf16 [S, 1024], ``w_o`` bf16 [H, 1024],
    ``coef`` fp32 [S, 2048] (D-pre's), ``base`` bf16 [2, 2, H], ``attn_out`` bf16 [S,
    H], ``scratch``, ``sym``, ``peers``, ``rank``, ``epoch``, ``layer``."""
    S, npes = key.tokens, key.npes
    assert S % BLOCK == 0 and 1 <= S <= MAX_TOKENS
    name = kernel_symbol(
        "muse_spark_dpost", s=S, w=npes, f=key.fault_site, t=int(key.trace)
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout, sym_layout = do_scratch_layout(), do_sym_layout(npes)
    Lds = do_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpost(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64, scratch: Int64, sym: Int64, peers: Int64,
        rank: Int32, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "o": o,
                "w_o": w_o,
                "coef": coef,
                "base": base,
                "attn_out": attn_out,
            }
            bases = peer_bases(peers, npes)
            do_off = sym_layout["do"][0]
            extra = {
                "W": npes, "rank": rank, "own_do": preg(sym, do_off, "do"),
                "peer_do": [preg(b, do_off, "do") for b in bases],
            }  # fmt: skip
            members = do_members(lds)
            members["base"] = base
            c = _common(key, members, args, step_tag(epoch, layer), extra, layer=layer)
            for region, (off, _) in layout.items():
                c[region[3:]] = sreg(scratch, off, region)
            run_dpost(c)

    @flyc.jit
    def launch(
        o: Int64, w_o: Int64, coef: Int64, base: Int64, attn_out: Int64, scratch: Int64, sym: Int64, peers: Int64,
        rank: Int32, epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dpost(
            o, w_o, coef, base, attn_out, scratch, sym, peers, rank, epoch, layer
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
