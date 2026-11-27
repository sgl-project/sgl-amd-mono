# SPDX-License-Identifier: Apache-2.0
"""A chain of ``--layers`` wide T1 launches (``glm53_flash.gfx950.wide.layer``, t1=True)
on ``tp`` gfx950 GPUs, a stand-in MoE between them (each rank's partial = x_f @ W_m,r^T,
torch), the way a step runs them: layer i reads layer i - 1's R_mid / post_f / comb_f
and its MoE partial. Eager once, then the same chain captured in a HIP graph and
replayed:

    ranks   every rank's final R_mid, x_f, ids, weights are the same bits (eager, replay)
    replay  each replay's outputs against the eager chain's (rel; split-K bf16 atomics
            make proj, and so everything after it, differ in the last bits run to run)
    ids     picked sets of the replay against the eager chain (rows whose 8th and 9th
            keys are within the drift, 2e-2, excused)

    torchrun --nproc-per-node 4 chain_bench.py [--reqs 3,16] [--layers 4] [--replays 3] [--time]
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

os.environ.setdefault("SGLANG_USE_AITER", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from layer_bench import gather, rel  # noqa: E402

from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.config import (  # noqa: E402
    HIDDEN,
    MOE_SLOTS,
    N_EXPERTS,
    TOP_K,
)
from sgl_amd_mono.glm53_flash.gfx950.wide import kda, layer, mhc  # noqa: E402

HC = mhc.HC
LB = -5.0
ONORM_EPS = 1e-5
NH, KD, P, NPROJ = kda.NH, kda.KD, kda.P, kda.NPROJ


def run(a, reqs, rank, tp, dev, cpu):
    tt = a.tt
    m = reqs * tt
    slots = reqs + 2
    gen = torch.Generator(device=dev).manual_seed(2000 + reqs)
    lg = torch.Generator(device=dev).manual_seed(200 + rank)

    def rnd(*s, scale=1.0, dt=torch.float32, g=gen):
        return (torch.randn(*s, device=dev, generator=g) * scale).to(dt)

    def z(*s, dt=torch.float32):
        return torch.zeros(*s, device=dev, dtype=dt)

    # one set of weights for every layer (replicated, then this rank's)
    mw = [
        dict(
            fn=rnd(24, HC * HIDDEN, scale=0.02),
            scale=torch.tensor([0.8, 0.6, 1.1], device=dev),
            base=rnd(24, scale=0.3),
            norm=(1 + rnd(HIDDEN, scale=0.1)).to(torch.bfloat16),
        )
        for _ in range(2)
    ]
    w_r = rnd(N_EXPERTS, HIDDEN, scale=0.02, dt=torch.bfloat16)
    bias_r = rnd(N_EXPERTS, scale=0.5)
    w_in = rnd(NPROJ, HIDDEN, scale=0.02, dt=torch.bfloat16, g=lg)
    w_o = rnd(HIDDEN, P, scale=0.02, dt=torch.bfloat16, g=lg)
    w_m = rnd(HIDDEN, HIDDEN, scale=0.01, dt=torch.bfloat16, g=lg)
    conv_w = rnd(3 * P, 4, scale=0.3, g=lg)
    w_fgb = rnd(2, P, KD, scale=0.08, dt=torch.bfloat16, g=lg)
    a_log = torch.log(torch.rand(NH, device=dev, generator=lg) * 8 + 1)
    dt_bias = rnd(P, scale=0.5, g=lg)
    on_w = 1 + rnd(KD, scale=0.1, g=lg)
    idx = torch.arange(1, reqs + 1, device=dev, dtype=torch.int32)
    L = tt

    # per layer: its own KDA cache and outputs; layer 0's inputs are the step's
    layers = []
    for _ in range(a.layers):
        conv_st = rnd(slots, 3, 3 * P, scale=0.5, dt=torch.bfloat16, g=lg).transpose(
            1, 2
        )
        layers.append(
            dict(
                conv_st=conv_st,
                state=rnd(slots, NH, KD, KD, scale=0.05, dt=torch.bfloat16, g=lg),
                rawv=z(slots, NH, L, KD, dt=torch.bfloat16),
                rawk=z(slots, NH, L, KD, dt=torch.bfloat16),
                ring_g=z(slots, NH, L, KD),
                ring_beta=z(slots, NH, L),
                iw=z(slots, tt, 3, 3 * P, dt=torch.bfloat16).transpose(2, 3),
                core=z(m, P, dt=torch.bfloat16),
                x_a=z(m, HIDDEN, dt=torch.bfloat16),
                post_a=z(m, HC),
                comb_a=z(m, HC * HC),
                proj=z(m, NPROJ, dt=torch.bfloat16),
                acc_o=z(m, HIDDEN, dt=torch.bfloat16),
                res_mid=z(m, HC, HIDDEN, dt=torch.bfloat16),
                x_f=z(m, HIDDEN, dt=torch.bfloat16),
                post_f=z(m, HC),
                comb_f=z(m, HC * HC),
                ids=z(m, MOE_SLOTS, dt=torch.int32),
                wts=z(m, MOE_SLOTS),
                logits=z(m, N_EXPERTS),
                moe=z(m, HIDDEN, dt=torch.bfloat16),
                res_cur=z(m, HC, HIDDEN, dt=torch.bfloat16),
            )
        )
    r0 = rnd(m, HC, HIDDEN, dt=torch.bfloat16)
    p0 = torch.rand(m, HC, device=dev, generator=gen) * 2
    c0 = torch.rand(m, HC * HC, device=dev, generator=gen) / 2
    m0 = rnd(m, HIDDEN, scale=0.5, dt=torch.bfloat16, g=lg)

    launch = layer.build(m, tt, tp, LB, ONORM_EPS, True, True, "route", t1=True)
    plan = launch.plan
    epoch = z(plan["epoch_words"], dt=torch.int32)
    scratch = z(plan["scratch_words"], dt=torch.int32)
    peers = PeerBuffer(plan["peer_bytes"], cpu, rank, tp, dev)
    peers.bytes.zero_()
    fa, ff = mw
    stream = torch.cuda.current_stream

    def step():
        prev = dict(res_mid=r0, post_f=p0, comb_f=c0, moe=m0)
        for ly in layers:
            iw_s = ly["iw"].stride()
            ints = (
                ly["conv_st"].stride(0),
                ly["conv_st"].stride(2),
                iw_s[0],
                iw_s[1],
                iw_s[3],
                ly["rawv"].stride(0),
                ly["rawk"].stride(0),
                ly["ring_g"].stride(0),
                ly["ring_beta"].stride(0),
                L,
            )
            ptrs = (
                prev["res_mid"],
                fa["fn"],
                fa["scale"],
                fa["base"],
                fa["norm"],
                ly["x_a"],
                ly["post_a"],
                ly["comb_a"],
                w_in,
                ly["proj"],
                conv_w,
                ly["conv_st"],
                idx,
                ly["iw"],
                idx,
                w_fgb,
                a_log,
                dt_bias,
                ly["state"],
                idx,
                ly["rawv"],
                ly["rawk"],
                ly["ring_g"],
                ly["ring_beta"],
                idx,
                on_w,
                ly["core"],
                w_o,
                ly["acc_o"],
                ff["fn"],
                ff["scale"],
                ff["base"],
                ff["norm"],
                ly["res_mid"],
                ly["x_f"],
                ly["post_f"],
                ly["comb_f"],
                w_r,
                bias_r,
                ly["ids"],
                ly["wts"],
                ly["logits"],
                prev["moe"],
                prev["post_f"],
                prev["comb_f"],
                ly["res_cur"],
                epoch,
                scratch,
            )
            launch(
                *[t.data_ptr() for t in ptrs],
                peers.addresses.data_ptr(),
                rank,
                *ints,
                0,
                0,
                m,
                stream=stream(),
            )
            torch.matmul(ly["x_f"], w_m.T, out=ly["moe"])
            prev = ly

    def snap():
        last = layers[-1]
        return {
            k: last[k].clone()
            for k in ("res_mid", "x_f", "ids", "wts", "post_f", "comb_f")
        }

    def key_set_bad(ids_a, ids_b, logits, bias):
        keys = torch.sigmoid(logits) + bias.to(torch.bfloat16).float()
        kv = keys.topk(TOP_K + 1, dim=1).values
        tie = (kv[:, TOP_K - 1] - kv[:, TOP_K]) < 2e-2
        same = (ids_a[:, :TOP_K].sort(1).values == ids_b[:, :TOP_K].sort(1).values).all(
            1
        )
        return int((~same & ~tie).sum().item())

    errs = {}
    dist.barrier(cpu)
    step()
    torch.cuda.synchronize()
    ref = snap()
    errs["ranks_eager"] = max(
        max(float((t != v).any()) for t in gather(v)) for v in ref.values()
    )
    # the graph: the whole chain, captured once (the epoch moves on in device memory)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        step()
    worst = {}
    for _ in range(a.replays):
        g.replay()
        torch.cuda.synchronize()
        out = snap()
        for k in ("res_mid", "x_f", "post_f", "comb_f"):
            worst[k] = max(worst.get(k, 0.0), rel(out[k], ref[k]))
        worst["ids_bad"] = max(
            worst.get("ids_bad", 0),
            key_set_bad(out["ids"], ref["ids"], layers[-1]["logits"], bias_r),
        )
        worst["ranks"] = max(
            worst.get("ranks", 0.0),
            max(max(float((t != v).any()) for t in gather(v)) for v in out.values()),
        )
    errs.update(worst)
    errs["ep"] = epoch[0].item()
    line = f"rank {rank} reqs {reqs:2d} rows {m:3d} layers {a.layers} " + "  ".join(
        f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}" for k, v in errs.items()
    )
    if a.time:
        torch.cuda.synchronize()
        st, en = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        st.record()
        for _ in range(10):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        step_us = st.elapsed_time(en) / 10 * 1000
        gm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gm, stream=s):
            for ly in layers:
                torch.matmul(ly["x_f"], w_m.T, out=ly["moe"])
        gm.replay()
        torch.cuda.synchronize()
        st.record()
        for _ in range(10):
            gm.replay()
        en.record()
        torch.cuda.synchronize()
        moe_us = st.elapsed_time(en) / 10 * 1000
        line += f"  step {step_us:7.1f} us  stand-in MoE {moe_us:6.1f} us  per layer {(step_us - moe_us) / a.layers:6.1f} us"
    print(line, flush=True)
    peers.close()
    tol = {
        "ranks_eager": 0.0,
        "ranks": 0.0,
        "res_mid": 2e-2,
        "x_f": 2e-2,
        "post_f": 2e-2,
        "comb_f": 2e-2,
        "ids_bad": 0,
    }
    bad = [k for k, v in tol.items() if errs[k] > v]
    if bad and rank == 0:
        print("  over tolerance:", bad, flush=True)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="3,16")
    ap.add_argument("--tt", type=int, default=6)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    tp = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    ok = True
    for reqs in map(int, a.reqs.split(",")):
        ok &= run(a, reqs, rank, tp, dev, cpu)
    flags = gather(torch.tensor([int(ok)], device=dev))
    if rank == 0:
        print("PASS" if all(f.item() for f in flags) else "FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
