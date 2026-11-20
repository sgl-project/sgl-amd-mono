# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#67421 at 28f094565b (Apache-2.0):
# vllm/models/qwen3_5/amd/specialized_decode/mono/blocks.py, the layer launch
"""``layer_block``: a decoder layer's second half, one persistent launch,
`BLOCKS` x `THREADS`, S <= `MAXT` rows:

    layer  (out_proj / o_proj, its TP all-reduce, the residual add +
           post_attention_layernorm, the sparse MoE and its TP all-reduce):
           core -> every rank's AR -> h + residual -> the residual out, x ->
           router, sigmoid shared gate, top-10, the MXFP4 experts with the
           shared expert fused as expert 512 -> every rank's FINAL -> the
           layer's output [S, H]

It keeps no x [S, H] in LDS (at 16 rows a bf16 x is 262 KB): the norm tasks
(CTAs 0 .. NRED - 1) put their columns' sums of squares, and every consumer
builds what it reads of x from the residual add's inputs in global memory
(`stages.norm`): the router a bf16 K slice, the routed experts the rows' MXFP4
codes, the shared gate a dot product. From `LAYER_XQ_MIN` rows the norm tasks
quantize their columns once into a shared MXFP4 image (`xq`) that the experts
copy instead. The launch's input is the SGLang front's output (plain loads),
except the AR sums (`xbuf`), which its norm tasks write in the launch.

The peer regions sit past K2's and the guard flags in the same peer buffer,
and every wait runs under the plugin's guard, as in K2.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950._k3 import KernelAbi, shift
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.layout import need
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import mx_layout as QL
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.common import (
    BLOCKS,
    THREADS,
    WAVES,
    pair_layout,
    peer_bases,
)
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.ops import (
    CM_DEV,
    ld_i32,
    ld_ptr64,
    traced,
    uniform,
)
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.stages import gemv, moe
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.stages import norm as N
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.stages.moe import (
    PS_MIN,
    step_tag,
    xq_scales,
)

SOURCES = kernel_sources(__name__)

H, ROWS, TP = L.HIDDEN, QL.ROWS, L.TP
E, TOPK = L.EXPERTS, L.TOPK
EPR = E // TP  # router logits a rank computes: 64
RIP = L.RI  # the experts' intermediate rows as the weights hold them
MOE = QL.MoeDims(H, L.RI, RIP)
KT = TOPK + 1  # picks a token: the top-10 and the fused shared expert
MAXT = L.SPLIT_MAX
NRED = N.NRED  # 64: the norm / final tasks, CTAs 0 ..
OTASKS = H // ROWS  # 512: o_proj / down row groups, bid and bid + 256
# the router: 4 row groups of the rank's 64 experts x 4 K quarters, CTAs R0 ..
R_PARTS = 4
R_K = H // R_PARTS
R_TASKS = EPR // ROWS * R_PARTS  # 16
R0 = NRED  # 64
RT0 = R0 + R_TASKS  # 80: shared gate CTAs, a token each
PS0 = RT0 + MAXT  # 96: the fused shared expert's early ug CTAs
LAYER_XQ_MIN = 5
# rows from which down runs the slots of run_ug's full rounds first, under its
# last round (needs the fused shared expert's tasks first: ``ps``)
SPLIT_MIN = 8

assert SPLIT_MIN >= PS_MIN
assert OTASKS == 2 * BLOCKS and PS0 + L.RI // moe.UG_COLS <= BLOCKS
assert RIP % 128 == 0 and k2.TAG_SLOTS == moe.TAG_SLOTS

_STREAM = fx.Stream(None)


class LayerBuild(msgspec.Struct, frozen=True):
    tokens: int
    eps: float = 1e-6  # post_attention_layernorm
    oq: bool = True  # out_proj / o_proj MXFP4 (ASM FP4 GEMM layout)
    og: bool = False  # o_proj's input is core x sigmoid(gate) (full attention)
    # SGLANG_AMD_MONO_FAULT_INJECT: the wait this rank stalls at (0: none), once
    # fault_after steps have run
    fault_site: int = 0
    fault_after: int = 0
    # False: a failed wait keeps the launch running and leaves its error words (tests)
    stop: bool = True

    @property
    def u1(self) -> bool:
        return self.tokens == 1

    @property
    def ps(self) -> bool:
        return self.tokens >= PS_MIN

    @property
    def xq(self) -> bool:
        return self.tokens >= LAYER_XQ_MIN

    @property
    def split(self) -> bool:
        return self.tokens >= SPLIT_MIN


LAYER_ABI = KernelAbi(
    (
        "core",
        "gate",
        "residual",
        "w_o",
        "w_os",
        "ln_w",
        "w_gate",
        "w_sg",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "res_out",
        "scratch",
        "peers",
        "rank",
        "epoch",
        "layer",
    )
)


LAYER_REGIONS = ("msq", "xqf", "rpart", "sgate", "route", "inter")


def layer_scratch_layout(key: LayerBuild) -> dict:
    """The mailbox regions, then `xbuf`: the AR sums [S, H] bf16 the norm
    tasks write for the consumers' x (16 B aligned), then with `key.xq` the x
    rows' shared MXFP4 image (16 B aligned)."""
    s = key.tokens
    lay = pair_layout(
        {
            "msq": NRED * s,
            "xqf": NRED,
            "rpart": R_TASKS * s * ROWS,
            "sgate": s,
            "route": s * 2 * TOPK,
            "inter": s * KT * moe.inter_pairs(MOE),
        }.items()
    )
    lay |= pair_layout([("xbuf", s * H * 2 // 8)], 16, scratch_bytes(lay))
    if key.xq:
        lay |= pair_layout([("xq", N.xq_dwords(s) // 2)], 16, scratch_bytes(lay))
    return lay


def scratch_bytes(layout: dict) -> int:
    return max(o + n for o, n in layout.values())


PEER_REGIONS = ("ar", "rlog", "final")


def peer_layout(s):
    """One parity's peer regions: the o_proj partials, the router logits and
    the MoE partials of every rank."""
    return pair_layout(
        {
            "ar": TP * s * H // 2,
            "rlog": TP * s * EPR // 2,
            "final": TP * s * H // 2,
        }.items()
    )


def peer_bytes(s):
    """Both parities' regions at `s` rows."""
    return 2 * scratch_bytes(peer_layout(s))


# past K2's regions and the guard flags: a K2 step and a layer_block step
# never write the same bytes
PEER_BASE = (k2.peer_buffer_bytes() + 255) // 256 * 256


def peer_buffer_bytes() -> int:
    """A rank's peer buffer with K2 and layer_block: K2's regions, the flags,
    then both parities of the regions here at `MAXT` rows."""
    return PEER_BASE + peer_bytes(MAXT)


def _peers(peers, own, epoch):
    """The peer regions at the widest width, this step's parity half: every
    rank's and this rank's own (`own`)."""
    play = peer_layout(MAXT)
    half = peer_bytes(MAXT) // 2
    par = uniform(ld_i32(epoch, 0)) % 2
    bases = peer_bases(peers, TP)
    peer = {}
    for region in PEER_REGIONS:
        off = PEER_BASE + play[region][0]
        peer[region] = [
            shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
        ]
        peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
    return peer


def _xwords(s):
    """LDS words of the x a CTA holds at once: a bf16 K slice (the router) or
    the rows' MXFP4 codes."""
    return max(s * (R_K + gemv.LDS_PAD) // 2, xq_scales(s, H) + s * H // 32)


_LAYER: dict = {}


def router_loads(c, i, q):
    """Router task (i, q): the rank's experts 16 i .., K quarter q."""
    return gemv.loads(
        c["lane"],
        c["wave"],
        c["args"]["w_gate"],
        c["rank"] * EPR + i * ROWS,
        R_K,
        ldk=H,
        k0=q * R_K,
    )


@traced
def stage_router(c, i, q, ops, src):
    """Router task (i, q) on x's K quarter q (`rs` loaded). Parts q > 0
    publish fp32 partials; part 0 adds them in part order, rounds to bf16
    (GateLinear's output) and puts the logits to every rank's RLOG."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    N.x_rows_to_lds(c, src, R_K, q * R_K)
    moe.put_acc(c, gemv.mfmas(lane, wave, c["xl"], R_K + gemv.LDS_PAD, R_K, s, ops))

    def out(t, r, vs):
        for p in range_constexpr(TP):
            c["put_bf"](
                c["peer"]["rlog"][p], (c["rank"] * s + t) * EPR + i * ROWS + r, vs
            )

    moe.ksplit_rows(
        c,
        c["rpart"],
        lambda qq, t, r: ((i * R_PARTS + qq) * s + t) * ROWS + r,
        q,
        R_PARTS,
        out,
    )


def _rlog_at(rlog, s, t, e):
    """Token t's logit of expert e: rank e / EPR's slot of every rank's RLOG."""
    return (rlog, ((e // EPR * s + t) * EPR + e % EPR) // 2, 1)


@traced
def _x_fp4(c, src, missing):
    """The routed experts' x (RCEIL MXFP4, as the stock decode MoE quantizes
    it) -> LDS; `missing`: the CTA has no rstd yet."""
    if missing:
        N.load_rstd(c)
    N.x_fp4_to_lds(c, src)


@traced
def _x_ug(c, src, xq, missing):
    """The routed experts' x -> LDS: the shared MXFP4 image (`xq`), else
    built here (`_x_fp4`)."""
    if const_expr(xq):
        N.xq_to_lds(c)
    else:
        _x_fp4(c, src, missing)


def build_layer(key: LayerBuild):
    if key in _LAYER:
        return _LAYER[key]
    s = key.tokens
    need(1 <= s <= MAXT, f"{s} rows, layer_block takes 1..{MAXT}")
    scr = layer_scratch_layout(key)
    max_u = s * TOPK + 1  # distinct experts the step can touch + the fused shared
    d = MOE

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * KT, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, max_u, 16]
        kof: fx.Array[fx.Int32, max_u * s, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class XLds:
        xl: fx.Array[
            fx.Int32, max(_xwords(s), s * (L.VALUE_DIM + gemv.LDS_PAD) // 2), 16
        ]

    @fx.struct
    class DownLds:
        il: fx.Array[fx.Int32, s * KT * (moe.irow(d) // 4 + d.rip // 32), 16]
        contrib: fx.Array[fx.Float32, (1 + int(key.split)) * s * KT * ROWS, 16]

    @fx.union
    class StageLds:
        x: XLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, 16, 16]

    keyed = key_tuple(key, SOURCES)
    name = f"qwen38_mono_layer_s{s}_o{int(key.oq)}_a{int(key.og)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def klayer(
        core: Int64,
        gate: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl = st_lds.x.peek(), st_lds.down.peek()
        mb = Mailbox(step_tag(epoch, layer))
        own = ld_ptr64(peers, rank)
        # every wait of this launch runs under the guard (deadline, error words);
        # a failure flags every rank through the peer buffers' flags region
        with guard.scope(
            guard.Guard(
                epoch,
                own=own,
                peers=peers,
                rank=rank,
                npes=TP,
                flags_offset=k2.FLAGS_OFF,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
                stop=key.stop,
            )
        ):
            peer = _peers(peers, own, epoch)
            rlog = peer["rlog_own"]
            c = {
                "S": s,
                "TP": TP,
                "moe": d,
                "rlog_at": lambda t, lane, i: _rlog_at(rlog, s, t, lane + 64 * i),
                "rlog_last": (
                    TP,
                    lambda i: (rlog, ((i * s + s - 1) * EPR + EPR - 2) // 2, 1),
                ),
                "route_cta": RT0,
                "tid": tid,
                "bid": bid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rank": rank,
                "eps": key.eps,
                "oq": key.oq,
                "og": key.og,
                "shuffled": True,
                "u1": key.u1,
                "KT": KT,
                "kof_t": s,
                "max_u": max_u,
                "put": mb.put,
                "put_bf": mb.put_bf,
                "put_words": mb.put_words,
                "poll": mb.poll,
                "red": lds.red.ptr,
                "rs": lds.rs.ptr,
                "peer": peer,
                "xl": xl.xl.ptr,
                "il": dl.il.ptr,
                "contrib": dl.contrib.ptr,
                "rt": {
                    "route": rl.route.ptr,
                    "flag": rl.flag.ptr,
                    "expert": rl.expert.ptr,
                    "kof": rl.kof.ptr,
                    "scan": rl.scan.ptr,
                },
                "xbuf": scratch + fx.Int64(scr["xbuf"][0]),
                "xq": scratch + fx.Int64(scr["xq"][0] if key.xq else 0),
                "args": {
                    "core": core,
                    "gate": gate,
                    "residual": residual,
                    "w_o": w_o,
                    "w_os": w_os,
                    "ln_w": ln_w,
                    "w_gate": w_gate,
                    "w_sg": w_sg,
                    "w13": w13,
                    "w13s": w13s,
                    "w2": w2,
                    "w2s": w2s,
                    "out": out,
                    "res_out": res_out,
                },
            }
            for region in LAYER_REGIONS:
                c[region] = sreg(scratch, scr[region][0], region)
            src = (c["xbuf"], CM_DEV, residual)
            is_r = (bid >= R0) & (bid < RT0)
            is_g = (bid >= RT0) & (bid < RT0 + s)
            ck = L.VALUE_DIM
            # ---- o_proj (both tasks' weights in flight first; a router CTA's
            # router weights behind them), then the AR + residual add + norm
            # sums, the router and the shared gate
            if is_r:
                oops = [moe.o_loads(c, bid + BLOCKS * j, ck) for j in range(2)]
                ri = bid - R0
                rops = router_loads(c, ri // R_PARTS, ri % R_PARTS)
                moe.run_oproj(c, oops, ck)
                N.load_rstd(c)
                stage_router(c, ri // R_PARTS, ri % R_PARTS, rops, src)
            if ~is_r:
                moe.run_oproj(
                    c, [moe.o_loads(c, bid + BLOCKS * j, ck) for j in range(2)], ck
                )
            if bid < NRED:
                N.stage_norm(c, bid)
                if const_expr(key.xq):
                    N.load_rstd(c)
                    N.stage_xq(c, bid, src)
            if is_g:
                N.load_rstd(c)
                N.stage_sgate(c, bid - RT0, src)
            # ---- the routed experts, then down: two row groups a CTA
            missing = ~(is_r | is_g)
            nu = moe.run_ug(c, key.ps, PS0, lambda: _x_ug(c, src, key.xq, missing))
            dops = moe.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            if const_expr(key.split):
                cut = moe.early_slots(c, nu)
                moe.load_inter(c, cut, True)
                b2 = bid + BLOCKS
                moe.down_zero(c, 2)
                dops = moe.down_slots(c, bid, nu, dops, 0, cut, (b2, 0, cut))
                dops = moe.down_slots(c, b2, nu, dops, 0, cut, (bid, cut, nu), 1)
                moe.load_inter(c, cut, False)
                dops = moe.down_slots(c, bid, nu, dops, cut, nu, (b2, cut, nu))
                moe.down_slots(c, b2, nu, dops, cut, nu, None, 1)
                moe.down_put(c, bid)
                moe.down_put(c, b2, 1)
            else:
                moe.load_inter(c)
                dops2 = moe.stage_down(c, bid, nu, dops, bid + BLOCKS)
                moe.stage_down(c, bid + BLOCKS, nu, dops2)
            if bid < NRED:
                N.stage_final(c, bid)

    @flyc.jit
    def launch(
        core: Int64,
        gate: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w_gate: Int64,
        w_sg: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        klayer(
            core,
            gate,
            residual,
            w_o,
            w_os,
            ln_w,
            w_gate,
            w_sg,
            w13,
            w13s,
            w2,
            w2s,
            out,
            res_out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    LAYER_ABI.check(klayer, launch)
    _LAYER[key] = launch
    return launch


def layer_block(
    key: LayerBuild,
    *,
    core,
    residual,
    w_o,
    ln_w,
    w_gate,
    w_sg,
    w13,
    w13s,
    w2,
    w2s,
    out,
    res_out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    w_os=None,
    gate=None,
):
    """out_proj / o_proj and its TP all-reduce, the residual add +
    post_attention_layernorm, the sparse MoE and its TP all-reduce: `core`
    [S, CORE] bf16 (x sigmoid(`gate`) with `key.og`) and `residual` [S,
    H] -> `res_out` (the sum over the ranks + residual) and `out` (the
    MoE's output) [S, H] bf16. Every weight as SGLang holds it after loading:
    `w_o` with `key.oq` fp4x2 with e8m0 `w_os` in the ASM FP4 GEMM layout,
    the routed experts as the AITER_MXFP4_MXFP4 backend holds them (shuffled,
    e8m0 scales), the shared expert fused as expert 512. `peers`: the int64
    table of every rank's peer buffer (`peer_buffer_bytes()` each)."""
    s = key.tokens
    k = L.VALUE_DIM
    bf = torch.bfloat16
    fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
    ne = E + 1
    rows = (s, H)
    need(core.shape == (s, k) and core.dtype == bf, f"core: bf16 [{s}, {k}]")
    need(
        (gate is not None) == key.og and (gate is None or gate.shape == core.shape),
        "gate: core's shape, with og only",
    )
    for name, t in (("residual", residual), ("out", out), ("res_out", res_out)):
        need(t.shape == rows and t.dtype == bf, f"{name}: bf16 [S, H]")
    need(ln_w.dtype == bf and ln_w.numel() == H, "post_attention_layernorm weight")
    if key.oq:
        need(
            w_o.shape == (H, k // 2)
            and w_o.dtype in fp4s
            and w_os is not None
            and w_os.numel() == H * QL.scale_cols(k)
            and w_os.element_size() == 1,
            f"o_proj: fp4x2 [{H}, {k // 2}] + e8m0 scales, one a 32-block",
        )
    else:
        need(w_o.shape == (H, k) and w_o.dtype == bf, f"o_proj: bf16 [{H}, {k}]")
    need(w_gate.shape == (E, H) and w_sg.numel() == H, "router shape")
    need(all(t.dtype == bf for t in (w_gate, w_sg)), "router weights: bf16")
    need(
        w13.shape == (ne, 2 * RIP, H // 2) and w13.dtype in fp4s,
        f"w13: fp4x2 [{ne}, {2 * RIP}, {H // 2}]",
    )
    need(
        w2.shape == (ne, H, RIP // 2) and w2.dtype in fp4s,
        f"w2: fp4x2 [{ne}, {H}, {RIP // 2}]",
    )
    need(
        w13s.numel() == ne * 2 * RIP * QL.scale_cols(H) and w13s.element_size() == 1,
        "w13 scales: e8m0, one a 32-block, shuffled",
    )
    need(
        w2s.numel() == ne * H * QL.scale_cols(RIP) and w2s.element_size() == 1,
        "w2 scales: e8m0, one a 32-block, shuffled (K blocks padded to 8)",
    )
    used = [core, gate, residual, w_o, w_os, ln_w, w_gate, w_sg, w13, w13s, w2, w2s]
    used += [out, res_out]
    need(
        all(t.is_contiguous() for t in used if t is not None), "a non-contiguous input"
    )
    vals = {
        "core": core.data_ptr(),
        "gate": 0 if gate is None else gate.data_ptr(),
        "residual": residual.data_ptr(),
        "w_o": w_o.data_ptr(),
        "w_os": w_os.data_ptr() if key.oq else 0,
        "ln_w": ln_w.data_ptr(),
        "w_gate": w_gate.data_ptr(),
        "w_sg": w_sg.data_ptr(),
        "w13": w13.data_ptr(),
        "w13s": w13s.data_ptr(),
        "w2": w2.data_ptr(),
        "w2s": w2s.data_ptr(),
        "out": out.data_ptr(),
        "res_out": res_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "peers": peers.data_ptr(),
        "rank": rank,
        "epoch": epoch.data_ptr(),
        "layer": layer,
    }
    build_layer(key)(*LAYER_ABI.pack(vals), stream=torch.cuda.current_stream())
