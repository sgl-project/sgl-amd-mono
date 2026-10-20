"""TP4 native A4W4 layer: independent MoE golden, caches and replay.

Run with torchrun --standalone --nproc-per-node=4. Control weights and tensors
stay on the GPU server. The golden checks the changed MoE subgraph using the
layer's post-attention boundary; attention/indexer retain their existing tests.
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
from check_multitoken import read_pairs
from fixtures import HIDDEN, check_untouched, fixture, make_weights, tensor_digest
from msgspec.structs import replace
from mxfp4_reference import expert_factory, quantize, unpack

from sgl_amd_mono.glm53.gfx950.runner import (
    Glm53Build,
    Glm53Caches,
    Glm53LayerOp,
    Glm53Runtime,
)


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return dict(
        relative_l2=float(
            torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-20)
        ),
        max_abs=float((a - b).abs().max()),
        exact=bool(torch.equal(actual, expected)),
    )


def check(args):
    rank = dist.get_rank()
    assert dist.get_world_size() == 4, "native A4W4 requires TP4"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    originals = {}
    weights = replace(
        make_weights(
            4, device, seed=args.seed, expert_factory=expert_factory(originals, rank)
        ),
        expert_mxfp4=True,
    )
    if args.ties:
        weights.w_r.zero_()
        weights.bias.zero_()
    h, cos, sin, initial, step, lengths = fixture(
        args.rows,
        args.capacity,
        device,
        args.pad_rows,
        True,
        args.seed + 1,
        request_width=args.rows // 2,
    )
    if args.quant_edges:
        # With O=0, a=h. Unit rows make normalized inputs round back to these
        # BF16-exact values, including E2M1 midpoints and E8M0 scale boundaries.
        weights.w_o.zero_()
        h.fill_(1)
        h[0].zero_()
        values = torch.tensor(
            [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 6],
            device=device,
        )
        signs = torch.cat((values, -values))
        powers = torch.tensor([-30, -10, -1, 0, 1, 5], device=device).repeat(
            HIDDEN // 32 // 6
        )
        weights.g_post.copy_(
            torch.ldexp(signs[None, :].expand(HIDDEN // 32, -1), powers[:, None])
            .flatten()
            .bfloat16()
        )
    if args.external:
        # External physical-slot lists with per-query causal extents.
        for row, length in enumerate(lengths):
            logical = torch.arange(min(length, 2048), device=device)
            begin = int(step.indptr[row])
            step.topk[begin : begin + logical.numel()] = (
                step.page_table[step.req_ids[row].long(), logical // 64] * 64
                + logical.int() % 64
            )
        weights = replace(weights, index=None)
    caches = Glm53Caches(initial.kv.clone(), initial.idx.clone())
    rt = Glm53Runtime(
        Glm53Build(
            args.rows,
            4,
            1,
            not args.external,
            args.capacity if not args.external else 0,
            True,
            indexer_cp=args.cp,
            expert_mxfp4=True,
            moe_activation_dtype="fp4",
        ),
        rank,
        dist.group.WORLD,
        device,
    )
    rt.set_deadline(60)
    rt.precompile()
    op = Glm53LayerOp(rt, weights, 0)

    def run():
        op.launch(h, caches, cos, sin, step)
        rt.advance_step()

    try:
        run()
        torch.cuda.synchronize()
        first = rt.outs[0].clone()
        assert not rt.epoch[3:6].any(), "guard error"
        root_output = first.clone()
        dist.broadcast(root_output, src=0)
        assert torch.equal(first, root_output), "TP ranks disagree"
        a = read_pairs(rt, "a", HIDDEN)
        selected = read_pairs(rt, "sel", 9, torch.int32).long()
        probability = read_pairs(rt, "prob", 9, torch.float32)
        mid = read_pairs(rt, "mid", 9 * 512, torch.float32).reshape(args.rows, 9, 512)
        # Independently normalize the attention boundary and quantize it.
        normalized = (
            a.float()
            * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-5)
            * weights.g_post.float()
        )
        x, codes, scales = quantize(normalized)
        native_scales = read_pairs(rt, "xqs", HIDDEN // 32, torch.int32)
        assert torch.equal(native_scales, scales.int()), "input E8M0 scales"
        native_words = read_pairs(rt, "xq", HIDDEN // 4, torch.int32)
        chunks = codes.reshape(args.rows, HIDDEN // 128, 16, 8).long()
        packed = sum(chunks[..., j] << (4 * j) for j in range(8)).int()
        expected_words = torch.cat((packed, torch.zeros_like(packed)), -1).reshape_as(
            native_words
        )
        code_differences = int((native_words != expected_words).sum())
        assert code_differences == 0, f"input quantized words: {code_differences}"

        expected_mid = torch.empty_like(mid)
        down_from_actual = torch.empty(args.rows, HIDDEN, device=device)
        down_composed = torch.empty_like(down_from_actual)
        for row in range(args.rows):
            total_actual = torch.zeros(HIDDEN, device=device)
            total_composed = torch.zeros_like(total_actual)
            for slot in range(9):
                expert = int(selected[row, slot])
                qw, sw = originals["ug"]
                wug = unpack(qw[expert], sw[expert])
                ug = wug @ x[row]
                gate, up = ug.chunk(2)
                expected_mid[row, slot] = torch.nn.functional.silu(gate) * up
                qw, sw = originals["dn"]
                wdn = unpack(qw[expert], sw[expert])
                qa = quantize(mid[row, slot])[0]
                qr = quantize(expected_mid[row, slot])[0]
                total_actual += (wdn @ qa) * probability[row, slot]
                total_composed += (wdn @ qr) * probability[row, slot]
            down_from_actual[row] = total_actual
            down_composed[row] = total_composed

        def output_reference(local):
            parts = [torch.empty_like(local, dtype=torch.bfloat16) for _ in range(4)]
            dist.all_gather(parts, local.bfloat16())
            total = torch.zeros_like(local)
            for part in parts:
                total += part.float()
            return (a.float() + total).bfloat16()

        metrics = {
            "up_gate": error(mid, expected_mid),
            "down_and_reduce": error(first, output_reference(down_from_actual)),
            "composed_moe": error(first, output_reference(down_composed)),
        }
        if any(metric["relative_l2"] > 0.001 for metric in metrics.values()):
            qa, ca, sa = quantize(mid)
            qr, cr, sr = quantize(expected_mid)
            locations = (ca != cr).nonzero()
            detail = []
            for location in locations[:12]:
                at = tuple(location.tolist())
                detail.append(
                    dict(
                        index=at,
                        actual=float(mid[at]),
                        expected=float(expected_mid[at]),
                        actual_code=int(ca[at]),
                        expected_code=int(cr[at]),
                    )
                )
            print(
                "DIAGNOSTIC "
                + json.dumps(
                    dict(
                        rank=rank,
                        metrics=metrics,
                        mid_codes_different=int((ca != cr).sum()),
                        mid_scales_different=int((sa != sr).sum()),
                        quantized_mid=error(qa, qr),
                        samples=detail,
                    )
                ),
                flush=True,
            )
        for name, metric in metrics.items():
            assert metric["relative_l2"] <= 0.001, (name, metric)
        check_untouched(initial, caches, step.slots)
        cache_hash = (tensor_digest(caches.kv), tensor_digest(caches.idx))
        initial_indices = step.topk.clone()
        control = dict(
            output=first.cpu(),
            a=a.cpu(),
            selected=selected.cpu(),
            probability=probability.cpu(),
            mid=mid.cpu(),
            indices=initial_indices.cpu(),
            cache_hash=cache_hash,
        )
        if args.save_control:
            path = Path(f"{args.save_control}.rank{rank}.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(control, path)
        if args.check_control:
            expected = torch.load(
                f"{args.check_control}.rank{rank}.pt", weights_only=True
            )
            for name, value in control.items():
                assert (
                    value == expected[name]
                    if name == "cache_hash"
                    else torch.equal(value, expected[name])
                ), name
        for _ in range(2):
            caches.kv.copy_(initial.kv)
            caches.idx.copy_(initial.idx)
            run()
            torch.cuda.synchronize()
            assert torch.equal(first, rt.outs[0]), "output replay"
            assert torch.equal(initial_indices, step.topk), "ordered indices replay"
            assert cache_hash == (
                tensor_digest(caches.kv),
                tensor_digest(caches.idx),
            ), "cache replay"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(4):
            caches.kv.copy_(initial.kv)
            caches.idx.copy_(initial.idx)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(first, rt.outs[0]), "graph replay"
            assert torch.equal(initial_indices, step.topk), "graph ordered indices"
            assert cache_hash == (
                tensor_digest(caches.kv),
                tensor_digest(caches.idx),
            ), "graph caches"
        assert not rt.epoch[3:6].any(), "replay guard error"
        trials = []
        if args.bench:
            timed = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timed):
                for _ in range(32):
                    run()
            for _ in range(8):
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
            assert not rt.epoch[3:6].any(), "timing guard error"
            assert torch.equal(first, rt.outs[0]), "timing changed output"
            assert torch.equal(initial_indices, step.topk), "timing changed indices"
            assert cache_hash == (
                tensor_digest(caches.kv),
                tensor_digest(caches.idx),
            ), "timing changed caches"
        launch_counts = None
        if args.profile_launches:
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                run()
                torch.cuda.synchronize()
            launch_counts = dict(
                Counter(
                    event.name
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                )
            )
            assert (
                sum(
                    count
                    for name, count in launch_counts.items()
                    if "glm5_monokernel" in name
                )
                == 1
            ), launch_counts
        source = Path(inspect.getfile(Glm53Runtime)).parent
        result = dict(
            rank=rank,
            control_exact=True if args.check_control else None,
            profiled_gpu_kernels=launch_counts,
            source_sha256={
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("kernel.py", "layout.py", "runner.py")
            },
            max_rank_us=trials,
            median_us=statistics.median(trials) if trials else None,
            graph_steps=32 if trials else None,
            graph_replays=10 if trials else None,
            requests=2,
            query_tokens=args.rows // 2,
            live_lengths=lengths,
            benchmark="synthetic_one_layer_query_block" if trials else None,
            rows=args.rows,
            activation="fp4",
            cp=args.cp,
            external=args.external,
            capacity=args.capacity,
            seed=args.seed,
            pad_rows=args.pad_rows,
            ties=args.ties,
            quant_edges=args.quant_edges,
            input_codes_exact=True,
            input_scales_exact=True,
            metrics=metrics,
            replay="passed",
            graph_replay="passed",
            caches="passed",
        )
        all_results = [None] * 4
        dist.all_gather_object(all_results, result)
        if rank == 0:
            print("RESULT " + json.dumps(all_results), flush=True)
    finally:
        rt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2, choices=(2, 4, 6, 8, 12, 16))
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=821)
    parser.add_argument("--pad-rows", type=int, default=0)
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--ties", action="store_true")
    parser.add_argument("--quant-edges", action="store_true")
    parser.add_argument("--save-control")
    parser.add_argument("--check-control")
    parser.add_argument("--profile-launches", action="store_true")
    parser.add_argument("--bench", action="store_true")
    dist.init_process_group("gloo")
    try:
        check(parser.parse_args())
    finally:
        dist.destroy_process_group()
