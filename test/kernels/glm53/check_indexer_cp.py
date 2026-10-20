"""GLM-5.3 paged FP8 CP on/off, with nonzero weights and optional paired timing.

torchrun --standalone --nproc-per-node=4 test/kernels/glm53/check_indexer_cp.py

Uses the plugin's real runner and cache ABI, without importing SGLang. This is a
synthetic layer/protocol check, not a checkpoint or serving accuracy result.
The independent score oracle consumes the unchanged upstream stage's queries
and head weights; it does not establish a full-layer independent golden.
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sgl_amd_mono.glm53.gfx950.layout import (
    INDEX_LOCAL_CAPACITY,
    index_cp_world,
    index_score_capacity,
    index_shard_tiles,
)
from sgl_amd_mono.glm53.gfx950.packing import pack_bf16, pack_fp8, pack_ptpc_fp8
from sgl_amd_mono.glm53.gfx950.runner import (
    INDEX_PAGE_BYTES,
    Glm53Build,
    Glm53Caches,
    Glm53LayerOp,
    Glm53Runtime,
    Glm53Step,
    Glm53Weights,
)

HIDDEN, INDEX_DIM, TOPK = 6144, 128, 2048
LAYERS = 3  # odd count exercises slot parity across step boundaries


def make_weights(tp, device, ties=False):
    g = torch.Generator(device=device).manual_seed(719)
    heads, inter = 64 // tp, 2048 // tp

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    def fp8(rows, k, *, packed=False, scale_k=128, experts=None):
        shape = (rows, k) if experts is None else (experts, rows, k)
        q = rnd(*shape, scale=0.2).to(torch.float8_e4m3fn)
        scales = torch.full(
            (*shape[:-2], math.ceil(rows / 128), math.ceil(k / scale_k)),
            k**-0.5,
            dtype=torch.float32,
            device=device,
        )
        return (pack_fp8(q) if packed else pack_ptpc_fp8(q)), scales

    qa, sqa = fp8(2624, HIDDEN)
    qb, sqb = fp8(heads * 256, 2048)
    uk, suk = fp8(heads * 512, 192, packed=True, scale_k=64)
    uv, suv = fp8(heads * 256, 512, packed=True)
    wo, so = fp8(HIDDEN, heads * 256)
    ug, sug = fp8(2 * inter, HIDDEN, experts=257)
    dn, sdn = fp8(HIDDEN, inter, experts=257)
    ik, sik = fp8(INDEX_DIM, HIDDEN)
    iq, siq = fp8(32 * INDEX_DIM, 2048)
    iw = pack_bf16(rnd(32, HIDDEN, scale=0.0 if ties else HIDDEN**-0.5).bfloat16())
    keep = (
        ik,
        sik,
        iw,
        iq,
        siq,
        torch.ones(128, device=device),
        torch.zeros(128, device=device),
    )
    index = torch.tensor(
        [t.data_ptr() for t in keep] + [0], dtype=torch.int64, device=device
    )
    return Glm53Weights(
        g_in=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        g_q=torch.ones(2048, dtype=torch.bfloat16, device=device),
        g_kv=torch.ones(512, dtype=torch.bfloat16, device=device),
        g_post=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        w_qkv_a=qa,
        s_qkv_a=sqa,
        w_q_b=qb,
        s_q_b=sqb,
        w_uk=uk,
        s_uk=suk,
        w_uv=uv,
        s_uv=suv,
        w_o=wo,
        s_o=so,
        w_r=pack_bf16(rnd(256, HIDDEN, scale=0.02).bfloat16()),
        bias=rnd(256, scale=0.05),
        w_ug=ug,
        s_ug=sug,
        w_dn=dn,
        s_dn=sdn,
        shuffled=True,
        index=index,
        keep=keep,
    )


def fixture(rows, capacity, device, pad_rows, full_context=False):
    g = torch.Generator(device=device).manual_seed(720)
    pages_per_request = capacity // 64
    pages = rows * pages_per_request + 1
    table = torch.zeros(rows, pages_per_request + 3, dtype=torch.int32, device=device)
    table[:, :pages_per_request] = (
        torch.randperm(pages - 1, generator=g, device=device).reshape(rows, -1) + 1
    )
    req = torch.arange(rows - 1, -1, -1, dtype=torch.int32, device=device)
    lengths = [max(1, capacity - 3), 65, 2049, 1, 4097, 127, 2048, 129][:rows]
    if full_context:
        lengths = [capacity - 3 - row for row in range(rows)]
    lengths = [min(n, capacity) for n in lengths]
    if pad_rows:
        lengths[-pad_rows:] = [0] * pad_rows
    pos = torch.tensor(
        [max(0, n - 1) for n in lengths], dtype=torch.int64, device=device
    )
    slots = table[req.long(), pos // 64].long() * 64 + pos % 64
    for row, n in enumerate(lengths):
        if not n:
            slots[row] = -1
    counts = torch.tensor(
        [min(n, TOPK) for n in lengths], dtype=torch.int32, device=device
    )
    indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0).int()]
    )
    h = torch.randn(rows, HIDDEN, generator=g, device=device).bfloat16()
    angles = (
        torch.arange(capacity, device=device).float()[:, None]
        * (8000000 ** (-torch.arange(0, 64, 2, device=device).float() / 64))[None, :]
    )
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kv = (torch.randn(pages * 64, 576, generator=g, device=device) * 0.1).bfloat16()
    values = (torch.randn(pages, 64, 128, generator=g, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    shuffled = (
        values.view(torch.uint8)
        .reshape(pages, 4, 16, 8, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    scales = torch.ones(pages, 64, dtype=torch.float32, device=device)
    idx = torch.cat([shuffled.reshape(pages, -1), scales.view(torch.uint8)], dim=1)
    idx = idx.reshape(pages, 64, 132)
    step = Glm53Step(
        pos,
        slots,
        indptr,
        req,
        table,
        torch.full((rows * TOPK + 1,), -7, dtype=torch.int32, device=device),
    )
    return h, cos, sin, Glm53Caches(kv, idx), step, lengths


def pairs(rt, name, shape, bf16=False):
    n = math.prod(shape)
    off = rt.sc[name]
    raw = rt.scratch
    if name == "index_q" and "index_cp_q" in rt.sy:
        launch_index = int(rt.epoch[0]) * LAYERS - 1
        off = rt.sy["index_cp_q"] + (launch_index & 1) * rt.sy["_index_cp_q_stride"]
        raw = rt.peer.bytes
    count = n // 2 if bf16 else n
    words = (
        raw[off : off + count * 8]
        .view(torch.int32)
        .reshape(count, 2)[:, 0]
        .contiguous()
    )
    return words.view(torch.bfloat16 if bf16 else torch.float32).reshape(shape)


def scores(rt, lengths):
    capacity = rt.key.index_max_seq
    stride = min(capacity, INDEX_LOCAL_CAPACITY) if rt.key.indexer_cp else capacity
    n = rt.width * stride
    if rt.key.indexer_cp:
        # advance_step() ran after the last layer in run_step().
        launch_index = int(rt.epoch[0]) * LAYERS - 1
        off = rt.sy["index_scores"] + (launch_index & 1) * rt.sy["_index_scores_stride"]
        raw = rt.peer.bytes
    else:
        off, raw = rt.sc["index_scores"], rt.scratch
    result = (
        raw[off : off + n * 8]
        .view(torch.int32)
        .reshape(n, 2)[:, 0]
        .contiguous()
        .view(torch.float32)
        .reshape(rt.width, -1)
    )
    if not rt.key.indexer_cp or capacity <= INDEX_LOCAL_CAPACITY:
        return result
    full = result.new_zeros(rt.width, capacity)
    full[:, :stride] = result
    cp_ranks = index_cp_world(rt.width, rt.key.tp, capacity)
    shard_stride = index_score_capacity(capacity, cp_ranks)
    local = pairs(rt, "index_scores", (rt.width, shard_stride)).cpu()
    shards = [torch.empty_like(local) for _ in range(rt.key.tp)]
    dist.all_gather(shards, local)
    group_start = (dist.get_rank() // cp_ranks) * cp_ranks
    for rank, values in enumerate(shards[group_start : group_start + cp_ranks]):
        for row, length in enumerate(lengths):
            if length > INDEX_LOCAL_CAPACITY:
                begin, end = index_shard_tiles(length, rank, cp_ranks)
                begin, end = begin * 64, min(end * 64, length)
                full[row, begin:end] = values[row, : end - begin].to(full.device)
    return full


def score_oracle(rt, cache, step, cos, sin, lengths):
    """Independent torch score arithmetic from the score stage's inputs."""
    rows = rt.width
    q = pairs(rt, "index_q", (rows, 32, 128), bf16=True).float()
    w = pairs(rt, "index_w", (rows, 32))
    new_k = pairs(rt, "index_k_new", (rows, 128), bf16=True).float()
    c, s = cos[step.positions].float()[:, None], sin[step.positions].float()[:, None]
    q0, q1 = q[:, :, :64:2].clone(), q[:, :, 1:64:2].clone()
    q[:, :, :64:2], q[:, :, 1:64:2] = q0 * c - q1 * s, q0 * s + q1 * c
    for span in (1, 2, 4, 8, 16, 32, 64):
        z = q.reshape(rows, 32, -1, 2, span)
        a, b = z[..., 0, :].clone(), z[..., 1, :].clone()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
    q = (q * (128**-0.5)).bfloat16().float()
    pages = cache.idx.shape[0]
    raw = cache.idx.reshape(pages, -1)
    kvals = (
        raw[:, :8192]
        .contiguous()
        .reshape(pages, 4, 8, 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    kvals = kvals.view(torch.float8_e4m3fn).float().reshape(pages, 64, 128)
    scales = raw[:, 8192:].contiguous().view(torch.float32)
    actual = scores(rt, lengths)
    worst = 0.0
    for row, length in enumerate(lengths):
        if not length:
            continue
        logical = torch.arange(length, device=q.device)
        physical = step.page_table[step.req_ids[row].long(), logical // 64].long()
        keys = kvals[physical, logical % 64] * scales[physical, logical % 64, None]
        keys[-1] = new_k[row]
        ref = ((q[row] @ keys.T).relu() * w[row, :, None]).sum(dim=0)
        got = actual[row, :length]
        # The independent RoPE/butterfly path can cross BF16 rounding boundaries.
        rel = (got - ref).norm() / ref.norm().clamp_min(1e-20)
        assert torch.isfinite(got).all() and float(rel) < 0.01, (row, float(rel))
        worst = max(worst, float(rel))
    return worst


def check_untouched(before, after, written_slots):
    valid = written_slots[written_slots >= 0].long().unique()
    mask = torch.ones(before.kv.shape[0], dtype=torch.bool, device=valid.device)
    mask[valid] = False
    assert torch.equal(before.kv[mask], after.kv[mask]), "untouched KV rows changed"
    changed = (before.idx != after.idx).reshape(-1)
    d = torch.arange(128, device=valid.device)
    for slot in valid.tolist():
        page, row = slot // 64, slot % 64
        offset = (
            page * INDEX_PAGE_BYTES
            + (row // 16) * 2048
            + (d // 16) * 256
            + (row % 16) * 16
            + d % 16
        )
        changed[offset] = False
        scale = page * INDEX_PAGE_BYTES + 8192 + row * 4
        changed[scale : scale + 4] = False
    assert not changed.any(), "untouched index-cache bytes changed"


def run(args):
    rank, tp = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    assert tp in (4, 8) and args.rows in (2, 4, 8)
    assert 0 <= args.pad_rows < args.rows
    assert args.capacity >= 64 and args.capacity % 64 == 0
    assert "gfx950" in torch.cuda.get_device_properties(device).gcnArchName
    weights = make_weights(tp, device, args.ties)
    h, cos, sin, initial, step, lengths = fixture(
        args.rows, args.capacity, device, args.pad_rows, args.full_context
    )
    runtimes, caches, steps = [], [], []
    try:
        for cp in (False, True):
            rt = Glm53Runtime(
                Glm53Build(
                    rows=args.rows,
                    tp=tp,
                    launches_per_step=LAYERS,
                    with_indexer=True,
                    index_max_seq=args.capacity,
                    fp8_shuffled=True,
                    indexer_cp=cp,
                ),
                rank,
                dist.group.WORLD,
                device,
            )
            runtimes.append(rt)
            rt.set_deadline(30)
            # Poison payloads, retaining generation zero and clear guard words.
            rt.scratch.view(torch.int32)[::2] = 0x7FC00000
            rt.peer.bytes[: rt.sy["flags"]].view(torch.int32)[::2] = 0x7FC00000
            caches.append(Glm53Caches(initial.kv.clone(), initial.idx.clone()))
            steps.append(
                Glm53Step(
                    step.positions,
                    step.slots,
                    step.indptr,
                    step.req_ids,
                    step.page_table,
                    step.topk.clone(),
                )
            )
            print(f"rank={rank} compile cp={cp}", flush=True)
            rt.precompile()
        ops = [
            Glm53LayerOp(runtimes[0], weights, layer, cp_runtime=runtimes[1])
            for layer in range(LAYERS)
        ]
        dist.barrier()

        def run_step(which):
            x = h
            for op in ops:
                x = op.forward(
                    x, caches[which], cos, sin, steps[which], indexer_cp=bool(which)
                )
            runtimes[which].advance_step()
            return x

        outputs = [None, None]

        def compare():
            torch.cuda.synchronize()
            for rt in runtimes:
                assert not rt.epoch[3:6].any(), "guard error"
            assert torch.isfinite(outputs[0]).all()
            assert torch.equal(
                *[
                    pairs(rt, "index_q", (args.rows, 32, 128), bf16=True)
                    for rt in runtimes
                ]
            ), "CP changed projected queries"
            assert torch.equal(caches[0].kv, caches[1].kv), "CP changed KV bytes"
            assert torch.equal(caches[0].idx, caches[1].idx), (
                "CP changed index-cache bytes"
            )
            score_vectors = [scores(rt, lengths) for rt in runtimes]
            for row, length in enumerate(lengths):
                assert torch.equal(
                    score_vectors[0][row, :length], score_vectors[1][row, :length]
                ), "CP changed scores"
                if length:
                    # Independent exact membership/tie oracle from complete scores.
                    logical = torch.argsort(
                        score_vectors[0][row, :length], descending=True, stable=True
                    )[:TOPK]
                    physical = (
                        step.page_table[step.req_ids[row].long(), logical // 64].long()
                        * 64
                        + logical % 64
                    )
                    start = int(step.indptr[row])
                    selected = steps[1].topk[start : start + len(logical)].long()
                    assert torch.equal(
                        selected.sort().values, physical.sort().values
                    ), "top-k differs from stable torch sort"
                if args.ties and length:
                    logical = torch.arange(min(length, TOPK), device=device)
                    slots = (
                        step.page_table[step.req_ids[row].long(), logical // 64].long()
                        * 64
                        + logical % 64
                    )
                    start = int(step.indptr[row])
                    selected = steps[1].topk[start : start + len(logical)].long()
                    assert torch.equal(selected, slots), (
                        "ties did not select earliest logical positions"
                    )
            assert torch.equal(steps[0].topk, steps[1].topk), (
                "CP changed ordered physical top-k"
            )
            assert torch.equal(outputs[0], outputs[1]), "CP changed layer output"
            # Reconstruct TP ownership: every rank must produce the same hidden stream.
            local_output = outputs[1].cpu()
            root_output = local_output.clone()
            dist.broadcast(root_output, src=0)
            assert torch.equal(root_output, local_output), "TP ranks disagree"

        for repeat in range(2):
            for which in (0, 1):
                if repeat == 1 and which == 1:
                    dist.barrier()
                    if rank == 0:
                        time.sleep(0.05)
                outputs[which] = run_step(which)
            compare()
        oracle_rel = score_oracle(runtimes[0], caches[0], step, cos, sin, lengths)
        graphs = []
        for which in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[which] = run_step(which)
            graphs.append(graph)
        compare()
        generator = torch.Generator(device=device).manual_seed(723)
        initial_lengths = list(lengths)
        written_slots = [step.slots.clone()]

        def set_lengths(new_lengths):
            lengths[:] = new_lengths
            step.positions.copy_(
                torch.tensor([max(0, n - 1) for n in lengths], device=device)
            )
            slots = (
                step.page_table[step.req_ids.long(), step.positions // 64].long() * 64
                + step.positions % 64
            )
            slots[torch.tensor(lengths, device=device) == 0] = -1
            step.slots.copy_(slots)
            counts = torch.tensor([min(n, TOPK) for n in lengths], device=device)
            step.indptr[1:].copy_(counts.cumsum(0))
            written_slots.append(slots.clone())

        replay_lengths = []
        for replay in range(args.replays):
            h.copy_(torch.randn(h.shape, generator=generator, device=device))
            # Cross the long/short boundary through the same captured pointers,
            # including mixed rows and empty shards; no host graph reselection.
            new_lengths = [max(1, n - replay % 6) if n else 0 for n in initial_lengths]
            if args.capacity > INDEX_LOCAL_CAPACITY:
                boundaries = [
                    args.capacity - 3,
                    INDEX_LOCAL_CAPACITY,
                    INDEX_LOCAL_CAPACITY + 1,
                    4095,
                    args.capacity // 2,
                    args.capacity - 2,
                    1,
                    INDEX_LOCAL_CAPACITY - 1,
                ]
                for row, n in enumerate(initial_lengths):
                    if n:
                        new_lengths[row] = boundaries[(replay + row) % len(boundaries)]
            set_lengths(new_lengths)
            replay_lengths.append(list(lengths))
            for graph in graphs:
                graph.replay()
            compare()
        set_lengths(initial_lengths)
        for graph in graphs:
            graph.replay()
        compare()
        # Separate captured modes can be replayed in a different order/count.
        # Only the selected runtime may consume an epoch.
        before = [int(rt.epoch[0]) for rt in runtimes]
        for which in (1, 1, 0):
            graphs[which].replay()
        compare()
        assert [int(rt.epoch[0]) for rt in runtimes] == [before[0] + 1, before[1] + 2]
        for cache in caches:
            check_untouched(initial, cache, torch.cat(written_slots))
        result = dict(
            tp=tp,
            rows=args.rows,
            capacity=args.capacity,
            initial_live_keys=initial_lengths,
            final_live_keys=lengths,
            ties=args.ties,
            graph_replays=args.replays,
            replay_live_keys=replay_lengths,
            dispatch="forward_parameter",
            extra_mode_replays=[1, 2],
            delayed_peer_ms=50,
            correctness="passed",
            score_oracle_rel=oracle_rel,
            benchmark="not_run",
        )
        if args.save_control or args.check_control:
            snapshot = {
                "input": h.cpu(),
                "output": outputs[0].cpu(),
                "indices": steps[0].topk.cpu(),
            }
            for name, shape in (
                ("a", (args.rows, HIDDEN)),
                ("o", (args.rows, 64 // tp, 256)),
                ("q_lat", (args.rows, 64 // tp, 512)),
            ):
                snapshot[name] = pairs(runtimes[0], name, shape, bf16=True).cpu()
            if args.save_control:
                target = Path(f"{args.save_control}.rank{rank}.pt")
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, target)
            if args.check_control:
                control = torch.load(
                    f"{args.check_control}.rank{rank}.pt", weights_only=True
                )
                metrics = {}
                for name in snapshot:
                    if name in ("input", "indices"):
                        assert torch.equal(snapshot[name], control[name]), (
                            f"source control changed {name}"
                        )
                    else:
                        actual, expected = snapshot[name].float(), control[name].float()
                        relative = (
                            actual - expected
                        ).norm() / expected.norm().clamp_min(1e-30)
                        max_abs = (actual - expected).abs().max()
                        # A separate source-control gate for changed attention
                        # reduction tiles, stricter than the serving cosine gate.
                        # CP off/on above remains bitwise exact in every run.
                        assert (
                            torch.isfinite(actual).all() and float(relative) <= 1e-3
                        ), (name, float(relative))
                        metrics[name] = [float(relative), float(max_abs)]
                gathered = [None for _ in range(tp)]
                dist.all_gather_object(gathered, metrics)
                result["source_control_max_rank"] = {
                    name: [max(m[name][i] for m in gathered) for i in range(2)]
                    for name in metrics
                }
        if args.bench_iters:
            trials = [[], []]
            for trial in range(4):
                for which in (0, 1) if trial % 2 == 0 else (1, 0):
                    for _ in range(10):
                        graphs[which].replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.bench_iters):
                        graphs[which].replay()
                    end.record()
                    end.synchronize()
                    us = torch.tensor(
                        [start.elapsed_time(end) * 1000 / (args.bench_iters * LAYERS)]
                    )
                    dist.all_reduce(us, op=dist.ReduceOp.MAX)
                    trials[which].append(float(us))
            medians = [statistics.median(t) for t in trials]
            result.update(
                benchmark="synthetic_shared_weights",
                max_rank_us=trials,
                median_us=medians,
                speedup=medians[0] / medians[1],
            )
        if rank == 0:
            print("RESULT " + json.dumps(result), flush=True)
    finally:
        for rt in runtimes:
            rt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--pad-rows", type=int, default=1)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--ties", action="store_true")
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Start every active request near the compiled context limit.",
    )
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument(
        "--save-control",
        help="Remote prefix for small control tensors, one file per rank.",
    )
    parser.add_argument(
        "--check-control",
        help="Compare output and selected intermediates to a saved source control.",
    )
    dist.init_process_group("gloo")
    try:
        run(parser.parse_args())
    finally:
        dist.destroy_process_group()
