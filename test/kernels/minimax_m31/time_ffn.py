"""Time MiniMax-M3.1-Flash's FFN launch (torchrun, TP4): layers with their own weight
copies (experts cold from HBM), chained in one graph, routed to a fixed set of experts.

    torchrun --nproc-per-node 4 test/kernels/minimax_m31/time_ffn.py --widths 8,16 --experts 8,14
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_minimax_m31_ffn as t  # noqa: E402

from sgl_amd_mono.minimax_m31 import ffn  # noqa: E402

COPIED = (
    "w13",
    "w2",
    "s13",
    "s2",
    "w13s",
    "s13s",
    "w2s",
    "s2s",
    "wo",
    "wos",
    "post_norm",
    "router",
)


def layer_weights(base, n_layers, experts, rank):
    """Every layer's own copy of the weights; the bias steers every row to `experts`."""
    from types import SimpleNamespace

    g = torch.Generator(device=base.router.device).manual_seed(100 + rank)
    picked = torch.randperm(ffn.EXPERTS, generator=g, device=base.router.device)[
        :experts
    ]
    layers = []

    def copy(x):
        return (
            x.view(torch.uint8).clone().view(x.dtype)
            if x.element_size() == 1
            else x.clone()
        )

    for _ in range(n_layers):
        w = SimpleNamespace(**{k: copy(getattr(base, k)) for k in COPIED})
        w.router_bias = base.router_bias.clone()
        w.router_bias[picked] += 100.0
        layers.append(w)
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,8,16")
    ap.add_argument(
        "--experts", default="8,14,18", help="distinct routed experts a step"
    )
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--trials", type=int, default=5, help="timed batches of replays")
    ap.add_argument(
        "--timeline", action="store_true", help="the last layer's stage marks"
    )
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.minimax_m31.runner import FfnRunner

    widths = tuple(int(w) for w in args.widths.split(","))
    peer = PeerBuffer(ffn.peer_bytes(tp), cpu, rank, tp, dev)
    runner = FfnRunner(
        tp, rank, peer.local, peer.addresses, dev, widths, timeline=args.timeline
    )
    base = t.weights._get_wrapped_function()()
    runner.warm(base, torch.cuda.get_device_properties(dev).multi_processor_count)
    for E in (int(e) for e in args.experts.split(",")):
        layers = layer_weights(base, args.layers, E, rank)
        for s in widths:
            g = torch.Generator(device=dev).manual_seed(7 + s)
            attn = torch.randn(s, ffn.ATTN_K, device=dev, generator=g).mul(3).bfloat16()
            residual = (
                torch.randn(s, ffn.HIDDEN, device=dev, generator=g).mul(4).bfloat16()
            )
            slots = torch.arange(s, dtype=torch.int64, device=dev)

            def chain():
                out = None
                for w in layers:
                    out, _ = runner(attn, residual, w, slots)
                return out

            chain()  # eager: builds the width's scratch
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            pers = []
            for _ in range(args.trials):
                dist.barrier()
                t0, t1 = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                t0.record()
                for _ in range(args.replays):
                    graph.replay()
                t1.record()
                torch.cuda.synchronize()
                pers.append(t0.elapsed_time(t1) * 1e3 / (args.replays * len(layers)))
            sums = torch.tensor(pers, device=dev)
            dist.all_reduce(sums)
            sums = (sums / tp).sort().values
            route = (
                runner.region(s, "route")[: s * 2 * ffn.TOPK * 8]
                .view(torch.int32)
                .view(-1, 2)[:, 0]
            )
            distinct = route.view(s, 2 * ffn.TOPK)[:, : ffn.TOPK].unique().numel()
            if rank == 0:
                print(
                    f"TP{tp} S={s} experts={E} (distinct {distinct}): {sums[0].item():.2f} min "
                    f"{sums[len(sums) // 2].item():.2f} median us a layer",
                    flush=True,
                )
            if args.timeline and rank == 0:
                tl = runner.region(s, "tl")[: ffn.BLOCKS * ffn.TL_POINTS * 8].view(
                    torch.int64
                )
                print(timeline(tl.view(ffn.BLOCKS, ffn.TL_POINTS).cpu()), flush=True)
    dist.destroy_process_group()


def timeline(tl) -> str:
    """Each stage mark's median and last CTA, in µs after the first CTA's start mark."""
    t0 = tl[:, 0][tl[:, 0] > 0].min()
    names = ("start",) + tuple(ffn.TL_STAGES)
    lines = []
    for j in range(tl.shape[1]):
        v = tl[:, j][tl[:, j] > 0].double()
        if v.numel():
            lines.append(
                f"  {names[j] if j < len(names) else j:>8}: median {((v - t0) / 100).median():7.2f} "
                f"last {((v - t0) / 100).max():7.2f} ({v.numel()} CTAs)"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
