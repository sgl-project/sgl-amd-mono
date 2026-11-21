"""Kernel-only time of the Qwen3.8-Flash-Next decode launches at TP2 or TP1 (the
torchrun world size), under torchrun:

    torchrun --nproc-per-node 2 test/kernels/qwen38_flash_next/time_layers.py \\
        --kind model --rows 1,16,64,192
    torchrun --nproc-per-node 1 test/kernels/qwen38_flash_next/time_layers.py \\
        --kind model --rows 1,16,64

One HIP graph holds a step of launches, as ``decode/launcher.py`` builds and runs them
in SGLang's setting, the streams chained launch to launch. Each of the 48 layers has
its own random weights in the per-rank shapes SGLang loads at that TP
(``layout.shard``; seeded by layer and rank, so they stream from HBM as in a served
step; the GDN projections' MXFP4 copies made by ``weights.build_layers``) and its own
GDN states, one slot a row. A QSA layer's attention output and gate, which SGLang's
attention computes between the layer's two launches, are random. ``--kind model`` is
the served step's launches in model order (36 ``gdn``, 12 ``qsa_front`` and 12
``qsa_back``); ``gdn``, ``qsa_front`` and ``qsa_back`` run only that kind's launches.
After an eager step and 20 warm-up replays, each of ``--trials`` trials replays the
graph ``--reps`` times; rank 0 prints per-launch microseconds (trial time / (reps x
launches)): the median, minimum and maximum over trials, the slowest rank's median,
and whether every health word stayed clean."""

import argparse
import gc
import math
import os
import statistics
import sys
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from sgl_amd_mono.qwen38_flash_next.decode import launcher, weights
from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L
from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import FAILED
from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import qsa

# SGLang pads a rank's expert rows, 640 at TP1 and 320 at TP2 (``weights.moe_rip``)
RIP = {1: 768, 2: 512}
STD = 0.02  # the bf16 weights' scale
KINDS = ("model", "gdn", "qsa_front", "qsa_back")


def _bf16(gen, *shape, std=STD):
    return (torch.randn(shape, generator=gen, device=gen.device) * std).bfloat16()


def _bytes(gen, *shape, low=0, high=256):
    return torch.randint(
        low, high, shape, generator=gen, device=gen.device, dtype=torch.uint8
    )


def _ones(gen, n):
    return torch.ones(n, dtype=torch.bfloat16, device=gen.device)


def _hc(gen):
    return NS(
        input_mix_weight_down=NS(weight=_bf16(gen, L.LOWRANK, L.HC_HIDDEN)),
        input_mix_weight_up=NS(weight=_bf16(gen, L.HC_HIDDEN, L.LOWRANK)),
        block_inject_weight=NS(weight=_bf16(gen, L.HC, L.HC_HIDDEN)),
        hc_norm=NS(weight=_ones(gen, L.HC_HIDDEN), variance_epsilon=1e-6),
    )


def _moe(rep, own, rip: int):
    # the router replicated, the experts a rank's share of each intermediate: random
    # FP4 codes, e8m0 scales 2^-9 .. 2^-4 (as the stage tests' ``_expert_weights``)
    e, cols = L.E_ALL, weights.scale_cols
    return NS(
        gate=NS(weight=_bf16(rep, L.EXPERTS, L.HIDDEN)),
        shared_expert_gate=NS(weight=_bf16(rep, 1, L.HIDDEN)),
        experts=NS(
            w13_weight=_bytes(own, e, 2 * rip, L.HIDDEN // 2),
            w2_weight=_bytes(own, e, L.HIDDEN, rip // 2),
            w13_weight_scale=_bytes(own, e, 2 * rip, cols(L.HIDDEN), low=118, high=124),
            w2_weight_scale=_bytes(own, e, L.HIDDEN, cols(rip), low=118, high=124),
        ),
    )


def _gdn(gen, sh: L.Shard):
    return NS(
        in_proj_qkvz=NS(weight=_bf16(gen, sh.QKVZ, L.HIDDEN)),
        in_proj_ba=NS(weight=_bf16(gen, sh.BA, L.HIDDEN)),
        conv1d=NS(weight=_bf16(gen, sh.CONV_DIM, 1, L.CONV_TAPS, std=0.5)),
        out_proj=NS(weight=_bf16(gen, L.HIDDEN, sh.VALUE_DIM)),
        A_log=torch.rand(sh.NV, generator=gen, device=gen.device).add(1).log(),
        dt_bias=torch.randn(sh.NV, generator=gen, device=gen.device),
        norm=NS(weight=_ones(gen, L.GDN_DV), eps=1e-6),
    )


def _layer(i: int, rank: int, tp: int, device):
    """Layer i's modules as ``weights.build_layers`` reads them (QSA at i % 4 == 3,
    PLE on layer 1): what SGLang replicates (the hyper-connections, the router, the
    indexer's projection) the same on every rank, its shards a rank's own."""
    sh = L.shard(tp)
    rep = torch.Generator(device=device).manual_seed(1000 * i)
    own = torch.Generator(device=device).manual_seed(1000 * i + 1 + rank)
    layer = NS(
        attn_hyper_connection=_hc(rep),
        mlp_hyper_connection=_hc(rep),
        mlp=_moe(rep, own, RIP[tp]),
        ple=NS() if i == 1 else None,
    )
    if i % 4 == 3:
        layer.o_proj = NS(weight=_bf16(own, L.HIDDEN, sh.QSA_CORE))
        layer.qkv_proj = NS(weight=_bf16(own, sh.QKV, L.HIDDEN))
        layer.indexer = NS(index_qk_proj=NS(weight=_bf16(rep, qsa.IX, L.HIDDEN)))
    else:
        layer.linear_attn = _gdn(own, sh)
    return layer


def _launcher(args, widths, rank: int, tp: int, group, device):
    """Every layer's operands and GDN states, and the launcher of ``widths``, its
    builds compiled."""
    model = NS(model=NS(layers=[_layer(i, rank, tp, device) for i in range(48)]))
    layers = weights.build_layers(model, {})
    la = launcher.Launcher(
        layers, widths, tp, rank, group, device, RIP[tp], args.deadline
    )
    torch.cuda.synchronize()
    launcher.compile_all(la.all_keys())
    slots = max(widths) + 1  # slot 0 is SGLang's reserved one
    gen = torch.Generator(device=device).manual_seed(7 + rank)
    for i, ly in enumerate(layers):
        if ly.kind == "gdn":
            conv = _bf16(gen, slots, la.shard.CONV_DIM, L.CONV_TAPS - 1, std=1.0)
            ssm = torch.randn(
                slots, la.shard.NV, L.GDN_DV, L.GDN_DK, generator=gen, device=device
            )
            la.bind(i, conv, ssm.mul_(0.01))
    return la


def _time(args, la, rows: int, rank: int, tp: int, group, device) -> None:
    rep = torch.Generator(device=device).manual_seed(rows)
    own = torch.Generator(device=device).manual_seed(rows + 1 + rank)
    slots = la.st_idx[rows]
    slots.copy_(torch.arange(1, rows + 1, dtype=torch.int32))
    # the streams are replicated; the attention output and gate are a rank's heads
    state = _bf16(rep, rows, L.HC_HIDDEN, std=1.0)
    attn = _bf16(own, rows, la.shard.QSA_CORE, std=1.0)
    gate = _bf16(own, rows, la.shard.QSA_CORE, std=1.0)
    # the step's launches: (build, layer), the served step's builds
    plan = [
        (key, i)
        for i, ly in enumerate(la.layers)
        for key in la.keys(i, rows, i == 0 or ly.ple)
        if args.kind in ("model", key.kind)
    ]
    # the launches of a step alternate the two barrier slots
    assert len(plan) % 2 == 0, len(plan)

    def step():
        la.begin_step()
        seq = 0
        for key, i in plan:
            dyn = {"state": state, "st_idx": slots}
            if key.kind == "qsa_back":
                dyn |= {"attn": attn, "gate": gate}
            seq = la.launch(key, i, seq, dyn)

    step()  # eager first: each launch loads its code outside the capture
    torch.cuda.synchronize()
    peak = state.float().abs().max().item()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_launch = []
    for _ in range(args.trials):
        dist.barrier(group=group)
        begin.record()
        for _ in range(args.reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_launch.append(begin.elapsed_time(end) * 1e3 / (args.reps * len(plan)))
    clean = int(la.health[FAILED]) == 0
    median = statistics.median(per_launch)
    medians = [None] * tp
    dist.all_gather_object(medians, (median, clean), group=group)
    if rank == 0:
        print(
            f"TIME kind={args.kind} rows={rows} tp={tp} launches={len(plan)} "
            f"us_per_launch median={median:.2f} min={min(per_launch):.2f} "
            f"max={max(per_launch):.2f} "
            f"slowest_rank_median={max(m for m, _ in medians):.2f} "
            f"clean={all(c for _, c in medians)} "
            f"state_after_eager_step max_abs={peak:.3g} finite={math.isfinite(peak)}",
            flush=True,
        )
    del graph
    dist.barrier(group=group)
    # free this width's graph now: an unload during the next capture breaks it
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=KINDS, default="model")
    ap.add_argument(
        "--rows", help="comma-separated widths (default: 1,16,64 and the TP's widest)"
    )
    ap.add_argument("--reps", type=int, default=50, help="graph replays a trial")
    ap.add_argument("--trials", type=int, default=7)
    # SGLang's: half the watchdog timeout, at most 60 s (the build key's wait_s)
    ap.add_argument("--deadline", type=float, default=60.0)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    top = L.shard(tp).MAX_TOKENS  # ValueError at a TP the launches are not built for
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    widths = (
        [int(r) for r in args.rows.split(",")]
        if args.rows
        else sorted({1, 16, 64, top})
    )
    la = _launcher(args, widths, rank, tp, group, device)
    for rows in widths:
        _time(args, la, rows, rank, tp, group, device)
    dist.barrier(group=group)
    la.peers.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
