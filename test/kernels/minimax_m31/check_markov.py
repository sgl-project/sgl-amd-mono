"""Check the M3.1 DSpark draft's sampled Markov block (minimax_m31/markov, bf16 step logits)
against SGLang's arithmetic for the VanillaMarkov head (torchrun, TP4, M3.1's vocab and
block): the tokens on every rank alike, the bf16 step logits and the noise counter, eager
and under graph replay.

    torchrun --nproc-per-node 4 test/kernels/minimax_m31/check_markov.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

VOCAB, DIM = 200064, 256


def reference(base, w1, w2s, anchor, temps, key, offset, vocab_start):
    """(tokens [bs, gamma], bf16 step logits [bs, gamma, width]) as SGLang computes them."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import _philox

    bs, gamma, width = base.shape
    prev = anchor.clone()
    tokens, logits = [], torch.empty_like(base)
    g = torch.arange(width, device=base.device) + vocab_start
    for s in range(gamma):
        bias = F.linear(w1[prev], w2s)  # [bs, width], bf16
        value = (base[:, s].float() + bias.float()).bfloat16()
        logits[:, s] = value
        keys = value.float()
        for r in range(bs):
            t = float(temps[r])
            if t != 0.0:
                c0 = (g // 4).to(torch.int64)
                c1 = torch.full_like(c0, r)
                c2 = torch.full_like(c0, s)
                c3 = torch.full_like(c0, int(offset) & 0xFFFFFFFF)
                words = _philox(key, c0, c1, c2, c3)
                w = torch.stack(words, 1).gather(1, (g % 4)[:, None]).squeeze(1)
                u = (w.to(torch.float32) + 0.5) * 2.3283064365386963e-10
                keys[r] = value[r].float() / t - torch.log(-torch.log1p(-u))
        best, idx = keys.max(1)  # the lowest index on ties
        pair = torch.stack([best, (idx + vocab_start).float()], 1)
        every = [torch.empty_like(pair) for _ in range(dist.get_world_size())]
        dist.all_gather(every, pair)
        every = torch.stack(every)  # [tp, bs, 2]
        top = every[:, :, 0].argmax(0)  # the lowest rank on ties
        prev = every[top, torch.arange(bs), 1].long()
        tokens.append(prev)
    return torch.stack(tokens, 1), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, default=7)
    ap.add_argument("--replays", type=int, default=20)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.minimax_m31 import markov

    g = torch.Generator(device=dev).manual_seed(7)
    w1 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.05).bfloat16()
    w2 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.5).bfloat16()
    width = VOCAB // tp  # SGLang's partition: the vocab is a multiple of 64 already
    vocab_start = rank * width
    w2s = w2[vocab_start : vocab_start + width]
    ops, x = markov.ops(), markov.Exchange(cpu, rank, tp, dev)
    scratch = torch.empty(
        markov.MAX_SEQUENCES, markov.blocks(width), 2, dtype=torch.float32, device=dev
    )
    key, gamma, fails = 0x1234_5678_9ABC, args.gamma, []
    for bs in range(1, markov.MAX_SEQUENCES + 1):
        gr = torch.Generator(device=dev).manual_seed(100 + 13 * rank + bs)
        base = (torch.randn(bs, gamma, width, generator=gr, device=dev) * 3).bfloat16()
        anchor = torch.randint(
            0,
            VOCAB,
            (bs,),
            device=dev,
            generator=torch.Generator(device=dev).manual_seed(bs),
        )
        temps = torch.tensor([1.0, 0.0, 0.7, 1.3][:bs], device=dev)
        offset0 = 17 * bs
        want, want_logits = reference(
            base, w1, w2s, anchor, temps, key, offset0, vocab_start
        )
        offset = torch.tensor([offset0], dtype=torch.int64, device=dev)
        out = torch.zeros(bs * gamma, dtype=torch.int64, device=dev)
        logits = torch.zeros(bs, gamma, width, dtype=torch.bfloat16, device=dev)

        def block():
            for s in range(gamma):
                ops.markov_sample_step(
                    base[:, s], w1, w2s, anchor, logits[:, s], temps, offset, key,
                    vocab_start, s, scratch, out, gamma, x.tok, x.calls, x.peers, rank,
                    s == gamma - 1, 20.0, True,
                )  # fmt: skip

        block()
        torch.cuda.synchronize()
        bad_tok = int((out.view(bs, gamma) != want).sum())
        # the bias GEMV sums in another order: a bf16 flip of the bias, then of the sum
        w, got = want_logits.float(), logits.float()
        bias = (w - base.float()).abs()
        apart = (got - w).abs()
        bad_logit = int((apart > (w.abs() + bias) * 2**-7 + 1e-6).sum())
        counter = int(offset.item()) == offset0 + 1
        # graph replay: the same block, the counter put back each time
        graph = torch.cuda.CUDAGraph()
        offset.fill_(offset0)
        with torch.cuda.graph(graph):
            block()
        bad_replay = 0
        for _ in range(args.replays):
            offset.fill_(offset0)
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            bad_replay += int((out.view(bs, gamma) != want).sum())
        every = [torch.empty_like(out) for _ in range(tp)]
        dist.all_gather(every, out)
        alike = all(torch.equal(every[0], e) for e in every)
        ok = bad_tok == 0 and bad_logit == 0 and counter and bad_replay == 0 and alike
        if rank == 0:
            print(
                f"bs={bs} {'PASS' if ok else 'FAIL'} | tokens differing {bad_tok} of "
                f"{bs * gamma} | step logits past a bf16 flip {bad_logit} | max "
                f"{apart.max().item():.3g} | counter {counter} | replays differing "
                f"{bad_replay} | ranks alike {alike}",
                flush=True,
            )
        if not ok:
            fails.append(bs)
    x.close()
    dist.barrier()
    if rank == 0:
        print("ALL PASS" if not fails else f"FAIL at {fails}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
