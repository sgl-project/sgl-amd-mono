"""Compare multi-token GLM layers across unchanged source snapshots.

Select the implementation using PYTHONPATH. The shared fixture also works with
the pre-CP runner. ``serial`` is a causal per-token control. ``origin`` and
``origin-chunk4`` force unsupported multi-query shapes for diagnostics only.
All schedules write the same request-major output without a timed host copy.
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from fixtures import HIDDEN, TOPK, check_untouched, fixture, make_weights, tensor_digest
from msgspec.structs import replace

from sgl_amd_mono.glm53.gfx950.runner import (
    Glm53Build,
    Glm53Caches,
    Glm53LayerOp,
    Glm53Runtime,
    Glm53Step,
)


def read_pairs(rt, name, columns, dtype=torch.bfloat16):
    rows = rt.sc.get("_tile_rows", rt.width)
    result = []
    for tile in range(rt.width // rows):
        raw = rt.scratch
        offset = rt.sc[name] + tile * rt.sc.get("_tile_stride", 0)
        if name == "index_q" and "index_cp_q" in rt.sy:
            slot = (int(rt.epoch[0]) - 1) & 1
            raw = rt.peer.bytes
            offset = (
                rt.sy["index_cp_q"]
                + slot * rt.sy["_index_cp_q_stride"]
                + tile * rt.sy.get("_tile_stride", 0)
            )
        count = rows * columns // (2 if dtype == torch.bfloat16 else 1)
        words = raw[offset : offset + count * 8].view(torch.int32).reshape(count, 2)
        result.append(words[:, 0].contiguous().view(dtype).reshape(rows, columns))
    return torch.cat(result)


def main(args):
    rank, tp = dist.get_rank(), dist.get_world_size()
    assert tp == 4
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    total = args.requests * args.query_tokens
    assert 1 < total <= 16 and args.requests >= 2
    weights = make_weights(tp, device, seed=args.seed)
    if args.route_ties:
        weights.w_r.zero_()
        weights.bias.zero_()
    h, cos, sin, initial, step, lengths = fixture(
        total,
        args.capacity,
        device,
        0,
        True,
        args.seed + 1,
        request_width=args.query_tokens,
    )
    caches = Glm53Caches(initial.kv.clone(), initial.idx.clone())
    external = (
        torch.load(f"{args.external_control}.rank{rank}.pt", weights_only=True)
        if args.external_control
        else None
    )
    if external is not None:
        weights = replace(weights, index=None)
    state_columns = [("q_lat", 16 * 512), ("a", HIDDEN), ("o", 16 * 256)]
    if external is None:
        state_columns.insert(0, ("index_q", 4096))
    dtypes = {}
    if args.check_routing:
        state_columns += [("sel", 9), ("prob", 9), ("mid", 9 * 512)]
        dtypes = {"sel": torch.int32, "prob": torch.float32, "mid": torch.float32}
    compared_names = ["output", *(name for name, _ in state_columns if name != "sel")]
    if args.schedule == "serial":
        groups = [
            list(range(token, total, args.query_tokens))
            for token in range(args.query_tokens)
        ]
    else:
        chunk = (
            4
            if args.schedule == "origin-chunk4"
            else 8
            if args.schedule == "origin" and total > 12
            else total
        )
        groups = [
            list(range(begin, min(begin + chunk, total)))
            for begin in range(0, total, chunk)
        ]
    runtimes, calls, gathered_outputs = [], [], []
    try:
        for group in groups:
            count = len(group)
            options = dict(
                rows=count,
                tp=tp,
                launches_per_step=1,
                with_indexer=external is None,
                index_max_seq=args.capacity if external is None else 0,
                fp8_shuffled=True,
            )
            if "indexer_cp" in Glm53Build.__struct_fields__:
                options["indexer_cp"] = args.indexer_cp
            else:
                assert not args.indexer_cp, "the origin source does not support CP"
            rt = Glm53Runtime(Glm53Build(**options), rank, dist.group.WORLD, device)
            runtimes.append(rt)
            rt.set_deadline(30)
            rt.precompile()
            indices = torch.tensor(group, dtype=torch.long, device=device)
            counts = torch.tensor(
                [min(lengths[i], TOPK) for i in group], dtype=torch.int32, device=device
            )
            st = Glm53Step(
                step.positions[indices].contiguous(),
                step.slots[indices].contiguous(),
                torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32, device=device),
                        counts.cumsum(0).int(),
                    )
                ),
                step.req_ids[indices].contiguous(),
                step.page_table,
                torch.full((count * TOPK + 1,), -7, dtype=torch.int32, device=device),
            )
            if external is not None:
                for row, original_row in enumerate(group):
                    nkeys = min(lengths[original_row], TOPK)
                    begin = int(st.indptr[row])
                    st.topk[begin : begin + nkeys].copy_(
                        external["indices"][original_row, :nkeys]
                    )
            calls.append((Glm53LayerOp(rt, weights, 0), h[indices].contiguous(), st))
            gathered_outputs.append(
                torch.empty(count, HIDDEN, dtype=h.dtype, device=device)
            )
            rt.outs[0] = gathered_outputs[-1]
        dist.barrier()

        def run_step():
            for (op, hidden, st), rt in zip(calls, runtimes):
                op.launch(hidden, caches, cos, sin, st)
                rt.advance_step()

        def snapshot():
            torch.cuda.synchronize()
            out = torch.empty_like(h)
            selected = torch.full((total, TOPK), -7, dtype=torch.int32, device=device)
            state = {
                name: torch.empty(
                    total,
                    columns,
                    dtype=dtypes.get(name, torch.bfloat16),
                    device=device,
                )
                for name, columns in state_columns
            }
            for group, rt, output, (_, _, st) in zip(
                groups, runtimes, gathered_outputs, calls
            ):
                assert not rt.epoch[3:6].any(), "guard error"
                out[group] = output
                for row, original_row in enumerate(group):
                    count = min(lengths[original_row], TOPK)
                    begin = int(st.indptr[row])
                    selected[original_row, :count] = st.topk[begin : begin + count]
                for name, tensor in state.items():
                    tensor[group] = read_pairs(rt, name, tensor.shape[1], tensor.dtype)
            assert torch.isfinite(out).all()
            check_untouched(initial, caches, step.slots)
            root = out.clone()
            dist.broadcast(root, src=0)
            assert torch.equal(out, root), "TP ranks disagree"
            return dict(
                input=h.cpu(),
                output=out.cpu(),
                indices=selected.cpu(),
                **{name: value.cpu() for name, value in state.items()},
                cache_sha256=dict(
                    kv=tensor_digest(caches.kv), index=tensor_digest(caches.idx)
                ),
            )

        run_step()
        first = snapshot()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_step()
        for _ in range(8):
            graph.replay()
        observed = snapshot()
        replay_changes = {}
        for name in first:
            if name == "cache_sha256":
                replay_changes[name] = first[name] != observed[name]
            else:
                diff = observed[name].float() - first[name].float()
                replay_changes[name] = dict(
                    changed=int(torch.count_nonzero(diff)),
                    max_abs=float(diff.abs().max()),
                )
        if rank == 0:
            print("REPLAY " + json.dumps(replay_changes), flush=True)
        for name in first:
            if name == "cache_sha256":
                assert first[name] == observed[name], "cache replay changed"
            else:
                assert torch.equal(first[name], observed[name]), (
                    f"replay changed {name}"
                )
        metrics = {}
        if args.save_control:
            path = Path(f"{args.save_control}.rank{rank}.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(observed, path)
        if args.check_control:
            control = torch.load(
                f"{args.check_control}.rank{rank}.pt", weights_only=True
            )
            assert control["cache_sha256"] == observed["cache_sha256"], (
                "cache bytes differ"
            )
            assert torch.equal(control["input"], observed["input"]), "fixture changed"
            if args.check_routing:
                assert torch.equal(control["sel"], observed["sel"]), (
                    "ordered expert routes differ"
                )
            for name in compared_names:
                actual, expected = observed[name].float(), control[name].float()
                relative = float(
                    torch.linalg.vector_norm(actual - expected)
                    / torch.linalg.vector_norm(expected).clamp_min(1e-12)
                )
                metrics[name] = dict(
                    relative_l2=relative,
                    max_abs=float((actual - expected).abs().max()),
                    exact=torch.equal(observed[name], control[name]),
                )
                assert relative <= 0.001, (name, metrics[name])
            assert torch.equal(control["indices"], observed["indices"]), (
                "ordered top-k differs"
            )
        result = dict(
            tp=tp,
            requests=args.requests,
            query_tokens=args.query_tokens,
            total_rows=total,
            capacity=args.capacity,
            seed=args.seed,
            schedule=args.schedule,
            rows_per_launch=[len(g) for g in groups],
            cp=args.indexer_cp,
            correctness="passed",
            with_indexer=external is None,
            routing_checked=args.check_routing,
            route_ties=args.route_ties,
            control_metrics=metrics,
        )
        if args.check_causality:
            # Change only the final token of request 0. Earlier tokens and the
            # independent request must not observe its input or new cache row.
            changed_row = args.query_tokens - 1
            saved = [hidden.clone() for _, hidden, _ in calls]
            for group, (_, hidden, _) in zip(groups, calls):
                if changed_row in group:
                    hidden[group.index(changed_row)].neg_().add_(1)
            caches.kv.copy_(initial.kv)
            caches.idx.copy_(initial.idx)
            run_step()
            changed = snapshot()
            unaffected = [row for row in range(total) if row != changed_row]
            for name in [*compared_names, "indices"]:
                assert torch.equal(
                    changed[name][unaffected], observed[name][unaffected]
                ), f"future token or another request changed {name}"
            for (_, hidden, _), original in zip(calls, saved):
                hidden.copy_(original)
            caches.kv.copy_(initial.kv)
            caches.idx.copy_(initial.idx)
            run_step()
            restored = snapshot()
            for name in observed:
                assert (
                    restored[name] == observed[name]
                    if name == "cache_sha256"
                    else torch.equal(restored[name], observed[name])
                ), f"restored inputs changed {name}"
            result["future_token_and_request_isolation"] = "passed"
        if args.bench:
            timed = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timed):
                for _ in range(32):
                    run_step()
            trials = []
            for _ in range(4):
                for _ in range(10):
                    timed.replay()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                timed.replay()
                start.record()
                for _ in range(10):
                    timed.replay()
                end.record()
                end.synchronize()
                duration = torch.tensor([start.elapsed_time(end) * 1000 / 320])
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                trials.append(float(duration))
            final = snapshot()
            assert all(
                torch.equal(final[n], observed[n]) for n in [*compared_names, "indices"]
            )
            assert final["cache_sha256"] == observed["cache_sha256"]
            result.update(
                max_rank_us=trials,
                median_us=statistics.median(trials),
                benchmark="synthetic_one_layer_query_block",
                graph_steps=32,
                graph_replays=10,
            )
        source = Path(inspect.getfile(Glm53Runtime)).parent
        if args.profile_launches:
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                run_step()
                torch.cuda.synchronize()
            kernels = Counter(
                event.name
                for event in profile.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            )
            mono_launches = sum(
                count for name, count in kernels.items() if "glm5_monokernel" in name
            )
            assert mono_launches == len(groups), (mono_launches, kernels)
            result["profiled_mono_launches"] = mono_launches
            result["profiled_gpu_kernels"] = dict(kernels)
        result["source_sha256"] = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("kernel.py", "layout.py", "runner.py")
        }
        all_metrics = [None] * tp
        dist.all_gather_object(all_metrics, metrics)
        result["control_metrics_all_ranks"] = all_metrics
        if rank == 0:
            print("RESULT " + json.dumps(result), flush=True)
    finally:
        for rt in reversed(runtimes):
            rt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--query-tokens", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument(
        "--schedule",
        choices=("batched", "origin", "origin-chunk4", "serial"),
        default="batched",
    )
    parser.add_argument("--indexer-cp", action="store_true")
    parser.add_argument("--save-control")
    parser.add_argument("--check-control")
    parser.add_argument(
        "--external-control",
        help="Reuse this control's physical top-k slots without running the indexer",
    )
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--check-causality", action="store_true")
    parser.add_argument(
        "--profile-launches",
        action="store_true",
        help="Verify the GPU layer launch count outside timing",
    )
    parser.add_argument(
        "--check-routing",
        action="store_true",
        help="Also compare expert slots, probabilities and up/gate intermediates",
    )
    parser.add_argument(
        "--route-ties",
        action="store_true",
        help="Force identical tied router scores for all query tokens",
    )
    args = parser.parse_args()
    if args.external_control and args.indexer_cp:
        parser.error("external top-k layers have no indexer CP work")
    dist.init_process_group("gloo")
    try:
        main(args)
    finally:
        dist.destroy_process_group()
