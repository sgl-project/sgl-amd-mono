# SPDX-License-Identifier: Apache-2.0
"""One rank's worker for ``test_qwen38_flash_next_wide_exchange.py``: the wide decode's
TP exchange between two processes on two GPUs, through ``common.peer.PeerBuffer``.

Each launch runs a layer's two exchanges as a ``gdn`` or ``qsa_back`` launch does
(``wide/layer.py``): the attention block's (``gdn.out``'s push of a column group's
LDS tile, then ``hc.combine``), a grid barrier, then the MLP block's (``moe.push_group``
up to ``moe.SMALL_MAX`` rows, ``moe.combine`` above, then ``hc.combine``), with the
launch's peer table, alarm word, health words, slot and tag. Launch ``i`` is layer
``i % 48`` of step ``i // 48``. Inputs cycle through ``SETS`` random sets, so a stale
line from the slot's previous use differs from the expected one.

Every launch's new state must equal, bit for bit, a host reference with the kernel's
rounding points: each rank's lines (bf16, or the INT4 codec's round trip,
``wide_reference.int4_roundtrip``) summed in f32 in rank order, rounded to bf16,
scaled and added to the state in f32, rounded to bf16. The scales (the injection and
the MoE routing weights) are powers of two, so every product is exact and the
reference is exact whether or not the compiler fuses a multiply-add. The two ranks'
states and ``rrms`` must also agree (a digest a launch, compared across ranks).

``--setting`` builds the kernel in vLLM's setting (a wait gives up and the launch runs
to its end) or SGLang's: stop mode (``grid``) at ``--wait-s`` (default 60 s), with
every rank's flags in its peer buffer's head, and the fault injection
(``SGLANG_AMD_MONO_FAULT_INJECT``, ``guard.fault_injection``). In SGLang's setting a
stop is a deliberate fault, so the worker leaves no core file. ``--progress`` syncs
after every launch and appends ``<launch> <monotonic s>`` to a file, so a test can
see when a stopped rank's last launch ended.

``--control`` (vLLM's setting) breaks launch ``--control-iter`` on rank 1 to show
each check can fail: ``lost_publish`` (rank 1 publishes its lines to its own mailbox
only, so rank 0's wait gives up after ``abi.WAIT_TICKS``), ``corrupt_input`` (rank
1's attention partial is not the one the reference uses) or ``rank_mismatch`` (rank
1's injection scales differ from rank 0's). Writes the per-width records to
``--out`` (JSON)."""

import argparse
import json
import os
import resource
import statistics
import time

import torch
import torch.distributed as dist
from wide_reference import int4_roundtrip

EPS = 1e-6
SETS = 5  # odd: launches 2 apart (the same slot) never share inputs
LAYERS = 48  # launch i is layer i % LAYERS; the epoch counts steps
POW2 = (0.5, 1.0, 2.0)
RRMS_REL = 1e-3  # rsqrt of a wave-order sum vs float64


def build(rows: int, codec: str, stop=False, wait_s=None, fault=(0, 0)):
    """The launcher of `rows` and `codec`; `stop`, `wait_s` (None:
    `abi.WAIT_SECONDS`) and `fault` (site, after) as `layer.LayerBuild`'s."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int32, Int64

    from sgl_amd_mono.common.guard import deadline_ticks
    from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import PEER_HEAD, WAIT_SECONDS
    from sgl_amd_mono.qwen38_flash_next.decode.wide.common.grid import (
        Grid,
        grid_sync,
        prologue,
    )
    from sgl_amd_mono.qwen38_flash_next.decode.wide.common.ops import (
        CM_DEV,
        ld,
        ld_i32x4,
        ld_ptr64,
        lds_st,
    )
    from sgl_amd_mono.qwen38_flash_next.decode.wide.common.plan import source_digest
    from sgl_amd_mono.qwen38_flash_next.decode.wide.layout import (
        BLOCKS,
        HIDDEN,
        THREADS,
    )
    from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import exchange as ex
    from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import hc, moe

    digest = source_digest()  # FlyDSL does not hash the stages' sources
    small = moe.small(rows)
    default_stream = fx.Stream(None)
    # scalars, so FlyDSL's key sees them
    ticks = deadline_ticks(wait_s or WAIT_SECONDS)
    fault_site, fault_after = fault
    name = f"t_xch2_{rows}_{codec}" + ("_stop" if stop else "")
    name += f"_f{fault_site}" if fault_site else ""

    @fx.struct
    class Smem:
        tile: fx.Array[fx.Int32, ex.GroupTile.words(rows), 16]
        word: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kern(
        y_a: Int64,
        y_m: Int64,
        route: Int64,
        part: Int64,
        split: Int64,
        state: Int64,
        inj_a: Int64,
        inj_m: Int64,
        rrms: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        alarm: Int64,
    ):
        _ = digest
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        g = Grid(
            tid,
            bid,
            sync,
            sync_next,
            folds,
            health,
            alarm,
            sm.word.ptr,
            ticks=ticks,
            stop=stop,
        )
        prologue(g)
        bases = [ld_ptr64(peers, r) for r in range(2)]
        own = (rank == 1).select(bases[1], bases[0])
        if fx.const_expr(stop):
            # as layer.build: every rank's flag words, PEER_HEAD bytes before its mailbox
            head = fx.Int64(PEER_HEAD)
            g.watch([b - head for b in bases], own - head, rank)
        ep = ld(epoch, 0, fx.Int32)
        if fx.const_expr(fault_site != 0):
            g.inject(fault_site, fault_after, ep)
        # layer.build's exchange of `kind` (0 attention, 1 MLP)
        slot_a, tag_a = (layer % 2) * 2, ep * 128 + layer * 2 + 1
        xa = ex.Exchange(bases, own, rank, 2, slot_a, tag_a, rows, codec)
        tile = ex.GroupTile(sm.tile.ptr, rows)
        for grp in range(bid, ex.GROUPS, BLOCKS):
            for idx in range(tid, rows * 8, THREADS):
                token, piece = idx // 8, idx % 8
                w = ld_i32x4(y_a, (token * HIDDEN + grp * 64 + piece * 8) // 2, CM_DEV)
                lds_st(tile.ptr + token * ex.GroupTile.STRIDE + piece * 4, w)
            fx.barrier()
            ex.push_tile(g, xa, tile, grp)
            fx.barrier()
        hc.combine(g, xa, state, inj_a, rrms, rows, EPS)
        grid_sync(g)
        xm = ex.Exchange(bases, own, rank, 2, slot_a + 1, tag_a + 1, rows, codec)
        if fx.const_expr(small):
            for grp in range(bid, ex.GROUPS, BLOCKS):
                moe.push_group(g, xm, y_m, grp, rows)
        else:
            # no split halves: `split` holds no tag, so `part2` is never read
            moe.combine(g, xm, route, part, part, split, rows)
        hc.combine(g, xm, state, inj_m, rrms, rows, EPS)

    @flyc.jit
    def launch(
        y_a: Int64,
        y_m: Int64,
        route: Int64,
        part: Int64,
        split: Int64,
        state: Int64,
        inj_a: Int64,
        inj_m: Int64,
        rrms: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        alarm: Int64,
        stream: fx.Stream = default_stream,
    ):
        _ = digest
        kern(
            y_a,
            y_m,
            route,
            part,
            split,
            state,
            inj_a,
            inj_m,
            rrms,
            peers,
            rank,
            epoch,
            layer,
            sync,
            sync_next,
            folds,
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def _pow2(gen, shape) -> torch.Tensor:
    return torch.tensor(POW2)[torch.randint(0, len(POW2), shape, generator=gen)]


def _lines(y: torch.Tensor, codec: str) -> torch.Tensor:
    """A rank's partial as its exchange lines decode it (f32)."""
    if codec == "int4":
        return int4_roundtrip(y.float())
    return y.to(torch.bfloat16).float()


def make_set(rows: int, codec: str, seed: int, rank: int, dev) -> dict:
    """One input set: both ranks' partials (every rank draws both, so both hold the
    reference), this rank's device inputs and the reference state and rrms."""
    from sgl_amd_mono.qwen38_flash_next.decode.wide.layout import HC, HIDDEN, SLOTS
    from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import moe

    gen = torch.Generator().manual_seed(seed)
    y_a = [torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16) for _ in "01"]
    if moe.small(rows):
        y_m = [torch.randn(rows, HIDDEN, generator=gen) for _ in "01"]
        part = [torch.zeros(1) for _ in "01"]
        wts = [torch.zeros(rows, SLOTS) for _ in "01"]
    else:
        # moe.combine's sum: sum_j w[b, j] part[b, j] in slot order, from zero
        part = [torch.randn(rows * SLOTS, HIDDEN, generator=gen) for _ in "01"]
        wts = [_pow2(gen, (rows, SLOTS)) for _ in "01"]
        y_m = []
        for p, w in zip(part, wts):
            pv, acc = p.view(rows, SLOTS, HIDDEN), torch.zeros(rows, HIDDEN)
            for j in range(SLOTS):
                acc = acc + w[:, j, None] * pv[:, j]
            y_m.append(acc)
    x = (torch.randn(rows, HC * HIDDEN, generator=gen) * 2).to(torch.bfloat16)
    inj_a, inj_m = _pow2(gen, (rows, HC)), _pow2(gen, (rows, HC))

    def combine(state, ys, inj):
        y = (_lines(ys[0], codec) + _lines(ys[1], codec)).to(torch.bfloat16).float()
        s = state.float().view(rows, HC, HIDDEN) + y[:, None, :] * inj[..., None]
        return s.to(torch.bfloat16).view(rows, HC * HIDDEN)

    new = combine(combine(x, y_a, inj_a), y_m, inj_m)
    rrms = torch.rsqrt(new.double().view(rows, HC, HIDDEN).pow(2).mean(-1) + EPS)
    route = torch.zeros(2 * rows * SLOTS, dtype=torch.int32)
    route[rows * SLOTS :] = wts[rank].flatten().view(torch.int32)
    return {
        "y_a": y_a[rank].to(dev),
        "y_m": y_m[rank].to(dev),
        "route": route.to(dev),
        "part": part[rank].to(dev),
        "x": x.to(dev),
        "inj_a": inj_a.to(dev),
        "inj_m": inj_m.to(dev),
        "ref": new.view(torch.int16).to(dev),
        "ref_rrms": rrms.float().to(dev),
    }


class Rank:
    """This rank's launch state, as the runtime's launcher keeps it."""

    def __init__(self, rank: int, group, dev):
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import (
            PEER_HEAD,
            health_words,
        )
        from sgl_amd_mono.qwen38_flash_next.decode.wide.common.grid import (
            FOLD_BASE,
            SLOT_WORDS,
            sync_words,
        )
        from sgl_amd_mono.qwen38_flash_next.decode.wide.layout import (
            BLOCKS,
            MAX_TOKENS,
            SLOTS,
        )
        from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import exchange

        self.rank, self.group, self.dev = rank, group, dev
        self.peers = PeerBuffer(
            PEER_HEAD + exchange.mailbox_bytes(2), group, rank, 2, dev
        )
        self.peers.bytes.zero_()
        self.peer_head = PEER_HEAD
        self.table = torch.zeros(4, dtype=torch.int64, device=dev)
        self.table[:2] = self.peers.addresses + PEER_HEAD
        # lost_publish: both entries this rank's own mailbox, so no line reaches the peer
        self.own_only = torch.zeros(4, dtype=torch.int64, device=dev)
        self.own_only[:2] = self.peers.addresses[rank] + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
        base = self.sync.data_ptr()
        self.slots = (base, base + 4 * SLOT_WORDS)
        self.folds = base + 4 * FOLD_BASE
        self.health = torch.zeros(health_words(BLOCKS), dtype=torch.int32, device=dev)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=dev)
        self.split = torch.zeros(MAX_TOKENS * SLOTS, dtype=torch.int32, device=dev)
        self.launches = 0

    def flags(self) -> list:
        """The output rank's alarm words (rank 0's peer buffer head), on rank 0."""
        return self.peers.bytes[:8].view(torch.int32).tolist()

    def reset_health(self) -> None:
        """After a failed width: every rank's health words and alarm words zero."""
        torch.cuda.synchronize()
        dist.barrier(self.group)
        self.health.zero_()
        self.peers.bytes[: self.peer_head].zero_()
        torch.cuda.synchronize()
        dist.barrier(self.group)


def run_width(st: Rank, rows: int, codec: str, iters: int, args) -> dict:
    from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled

    from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import RECORD_WORDS
    from sgl_amd_mono.qwen38_flash_next.decode.wide.layout import HC, HIDDEN

    rec = {"rows": rows, "codec": codec, "iters": iters}
    stream = torch.cuda.current_stream()
    t0 = time.perf_counter()
    launch = build(rows, codec, args.stop, args.wait_s, args.fault)
    _preload_compiled(launch, *([0] * 18), stream)
    torch.cuda.synchronize()
    rec["compile_s"] = round(time.perf_counter() - t0, 2)
    sets = [
        make_set(rows, codec, args.seed * 1_000_003 + rows * 101 + p, st.rank, st.dev)
        for p in range(SETS)
    ]
    gen = torch.Generator(device=st.dev).manual_seed(args.seed)
    mult = torch.randint(
        1, 1 << 31, (rows * HC * HIDDEN,), generator=gen, device=st.dev
    )
    mult_r = torch.randint(1, 1 << 31, (rows * HC,), generator=gen, device=st.dev)
    state = torch.empty(rows, HC * HIDDEN, dtype=torch.bfloat16, device=st.dev)
    rrms = torch.empty(rows, HC, device=st.dev)
    # the control launch's inputs on rank 1
    bad = args.control if st.rank == 1 else "none"
    ctrl = args.control_iter
    has_ctrl = 0 <= ctrl < iters
    corrupt = sets[ctrl % SETS]["y_a"].clone()
    corrupt[0, 0] += 64
    other_inj = sets[ctrl % SETS]["inj_a"].clone()
    other_inj[0, 0] *= 2
    mism, digests, rr_err, events = [], [], [], []
    progress = open(args.progress, "a") if args.progress else None  # noqa: SIM115
    torch.cuda.synchronize()
    dist.barrier(st.group)
    t0 = time.perf_counter()
    for i in range(iters):
        s = sets[i % SETS]
        layer = st.launches % LAYERS
        if layer == 0:
            st.epoch[:1].add_(1)  # a step's first launch
        state.copy_(s["x"])
        is_ctrl = i == ctrl
        y_a = corrupt if is_ctrl and bad == "corrupt_input" else s["y_a"]
        inj_a = other_inj if is_ctrl and bad == "rank_mismatch" else s["inj_a"]
        table = st.own_only if is_ctrl and bad == "lost_publish" else st.table
        k = st.launches % 2
        ev = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        ev[0].record()
        _run_compiled(
            launch,
            y_a.data_ptr(),
            s["y_m"].data_ptr(),
            s["route"].data_ptr(),
            s["part"].data_ptr(),
            st.split.data_ptr(),
            state.data_ptr(),
            inj_a.data_ptr(),
            s["inj_m"].data_ptr(),
            rrms.data_ptr(),
            table.data_ptr(),
            st.rank,
            st.epoch.data_ptr(),
            layer,
            st.slots[k],
            st.slots[1 - k],
            st.folds,
            st.health.data_ptr(),
            st.alarm,
            stream,
        )
        ev[1].record()
        st.launches += 1
        events.append(ev)
        if progress:
            torch.cuda.synchronize()
            progress.write(f"{i} {time.monotonic():.6f}\n")
            progress.flush()
        mism.append((state.view(torch.int16) != s["ref"].view(rows, -1)).sum())
        digests.append(
            (state.view(torch.int16).flatten().to(torch.int64) * mult).sum()
            + (rrms.view(torch.int32).flatten().to(torch.int64) * mult_r).sum()
        )
        rr_err.append(((rrms - s["ref_rrms"]).abs() / s["ref_rrms"]).max())
    torch.cuda.synchronize()
    rec["loop_s"] = round(time.perf_counter() - t0, 2)
    if progress:
        progress.close()
    mism = torch.stack(mism).cpu()
    mine = torch.stack(digests).cpu()
    rr_err = torch.stack(rr_err).cpu()
    ms = [a.elapsed_time(b) for a, b in events]
    theirs = [torch.empty_like(mine) for _ in range(2)]
    dist.all_gather(theirs, mine, group=st.group)
    differ = (theirs[0] != theirs[1]).nonzero().flatten().tolist()
    bad_iters = mism.nonzero().flatten().tolist()
    health = st.health[: 2 * RECORD_WORDS].cpu().tolist()
    failed_rec = None
    if health[0]:
        # the CTAs' records of the first wait each gave up on: [1, site, seen,
        # target, cta, clock, 0, 0] (grid.give_up)
        recs = st.health.view(-1, RECORD_WORDS)[1:].cpu()
        recs = recs[recs[:, 0] != 0]
        sites = recs[:, 1].tolist()
        failed_rec = {
            "ctas": len(sites),
            "sites": {str(x): sites.count(x) for x in sorted(set(sites))},
            "first": recs[recs[:, 5].argmin()].tolist() if len(sites) else None,
        }
    rec |= {
        "failed": health[0],
        "failed_record": failed_rec,
        "alarm_words": st.flags()[:2] if st.rank == 0 else None,
        "mismatch_iters": len(bad_iters),
        "first_mismatch_iters": bad_iters[:5],
        "mismatch_elems_max": int(mism.max()),
        "digest_differ_iters": len(differ),
        "first_digest_differ_iters": differ[:5],
        "rrms_rel_max": float(rr_err.max()),
        "launch_ms_median": round(statistics.median(ms), 4),
        "launch_ms_max": round(max(ms), 4),
        "launch_ms_control": round(ms[ctrl], 2) if has_ctrl else None,
        "mismatch_at_control": int(mism[ctrl]) if has_ctrl else None,
        "digest_differs_at_control": (ctrl in differ) if has_ctrl else None,
    }
    rec["pass"] = (
        rec["failed"] == 0
        and rec["mismatch_iters"] == 0
        and rec["digest_differ_iters"] == 0
        and rec["rrms_rel_max"] < RRMS_REL
    )
    # the ranks reset together: a wait that gave up on either rank poisons both
    any_failed = torch.tensor([rec["failed"]], dtype=torch.int32)
    dist.all_reduce(any_failed, op=dist.ReduceOp.MAX, group=st.group)
    if any_failed.item():
        st.reset_health()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--init-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codecs", default="bf16,int4")
    ap.add_argument("--widths", required=True, help="comma-separated rows")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument(
        "--control",
        default="none",
        choices=("none", "lost_publish", "corrupt_input", "rank_mismatch"),
    )
    ap.add_argument("--control-iter", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--setting", choices=("vllm", "sglang"), default="vllm")
    ap.add_argument("--wait-s", type=float, help="sglang: the deadline (default 60)")
    ap.add_argument("--progress", help="sync every launch and log it to this file")
    args = ap.parse_args()
    if args.control != "none" and not 0 <= args.control_iter < args.iters:
        ap.error("--control needs --control-iter below --iters")
    rank = args.rank
    from sgl_amd_mono.common.guard import fault_injection

    args.stop = args.setting == "sglang"
    args.fault = fault_injection(rank) or (0, 0)
    if args.stop:
        args.wait_s = args.wait_s or 60.0
        if args.control != "none":
            ap.error("--control breaks a launch in vLLM's setting")
        # a stop is a deliberate fault
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    elif args.wait_s is not None or args.fault != (0, 0):
        ap.error("--wait-s and SGLANG_AMD_MONO_FAULT_INJECT are SGLang's setting")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    # a file init method: a port picked in advance can be taken before rank 0 listens on it
    dist.init_process_group(
        backend="gloo", init_method=f"file://{args.init_file}", rank=rank, world_size=2
    )
    st = Rank(rank, dist.group.WORLD, dev)
    props = torch.cuda.get_device_properties(dev)
    out = {
        "rank": rank,
        "device": props.gcnArchName,
        "pci_bus_id": getattr(props, "pci_bus_id", None),
        "HSA_ENABLE_IPC_MODE_LEGACY": os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY"),
        "control": args.control,
        "control_iter": args.control_iter,
        "setting": args.setting,
        "wait_s": args.wait_s,
        "fault": args.fault,
        "records": [],
    }
    for codec in args.codecs.split(","):
        for rows in map(int, args.widths.split(",")):
            rec = run_width(st, rows, codec, args.iters, args)
            out["records"].append(rec)
            print(json.dumps(rec), flush=True)
            with open(args.out, "w") as f:  # each width as it ends
                json.dump(out, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
