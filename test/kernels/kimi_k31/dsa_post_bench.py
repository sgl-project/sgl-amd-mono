# SPDX-License-Identifier: Apache-2.0
"""One wide-row DSA post-attention launch (``kimi_k31/gfx950/dsa_post.py``) at TP8: check and time.

o_proj -> RS + residual -> MLP AttnRes -> MoE front -> route (+ sort) / shared SiTU -> shared down.
Checks prefix_out (with OOB prefix clamping), latent, shared_out and logits against the f32 reference; the top-k
ids and weights against the launch's own logits, and the same routing on every rank; the sort against aiter's
(aux16) or SGLang's (sgl) on the launch's own ids. Checks eager launches, then one CUDA-graph replay on fresh
inputs, and times the graph with cold-L2 flushing.

    python3 -m torch.distributed.run --nproc-per-node 8 test/kernels/kimi_k31/dsa_post_bench.py \
        --rows 12,24,48 --nvb-m 3 --sort aux16 --front-map --timeline
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from chain_bench import EPS
from gemm_coll_bench import graph_us
from layer_bench import KO, LAT, SI, TOPK, TP, E, H, rel, situ, sort_check_aux16

BF = torch.bfloat16


def attn_res_f32_ref(prefix, bank, nvb, w_s, w_p, w_o):
    """AttnRes + post-attn RMSNorm with FP32 scoring matching ``attnres_row``."""
    if nvb == 0:
        mixed = prefix
    else:
        rows = torch.cat([bank[..., :nvb, :], prefix.unsqueeze(-2)], dim=-2).float()
        normed = F.rms_norm(rows, (H,), w_s.float(), EPS)
        probs = torch.softmax(normed @ w_p.float(), dim=-1)
        mixed = (probs.unsqueeze(-1) * rows).sum(dim=-2).to(BF)
    return F.rms_norm(mixed.float(), (H,), w_o.float(), EPS).to(BF)


def timeline(pl, scratch, m, rank):
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp

    o = pl.scratch_offsets[0]["tl"]
    tl = (
        scratch[o : o + BLOCKS * dp.TL_POINTS * 8]
        .view(torch.int64)
        .view(BLOCKS, dp.TL_POINTS)
        .clone()
    )
    every = [torch.empty_like(tl) for _ in range(TP)]
    dist.all_gather(every, tl)
    if rank != 0:
        return
    t = torch.stack(every).cpu()
    t0 = torch.stack([t[r, :, 0][t[r, :, 0] > 0].min() for r in range(TP)]).view(
        TP, 1, 1
    )
    t = torch.where(t > 0, t - t0, torch.full_like(t, -1))
    print(f"timeline dsa_post m {m}: mark  CTAs  first  median  last (us)", flush=True)
    for k, name in enumerate(dp.MARKS):
        v = t[..., k]
        v = v[v >= 0].double() / 100.0
        if v.numel():
            print(
                f"  {k:2d} {name:11s} {v.numel():5d} {v.min():7.1f} {v.median():7.1f} {v.max():7.1f}",
                flush=True,
            )
    ends = [t[r, :, -1].max().item() / 100.0 for r in range(TP)]
    print("  end by rank: " + " ".join(f"{e:5.1f}" for e in ends), flush=True)


def run(args, m, rank, dev, cpu):
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf
    from sgl_amd_mono.kimi_k31.wide import cfg_rows

    nl, nf, nbk, nvm = args.launches, args.nf, args.nbk, args.nvb_m
    sl = H // TP
    cols = slice(rank * sl, (rank + 1) * sl)
    col0 = 2 * rank * sl
    _, _, go_s, gf_s, gs_s = cfg_rows(m, args.L)
    go = Gemm(*(int(v) for v in go_s.split(":")))
    gf = Gemm(*(int(v) for v in gf_s.split(":")))
    gs = Gemm(*(int(v) for v in gs_s.split(":")))
    n_sgu = nf - mf.E_R - mf.L_R
    front_rows = (
        (n_sgu + rank * mf.E_R, n_sgu + mf.E + rank * mf.L_R, 0)
        if args.front_map
        else None
    )
    launch, pl = dp.dsa_post(
        m,
        go,
        gf,
        gs,
        nf,
        nvb_m=nvm,
        nbk=nbk,
        launches=nl,
        eps=EPS,
        timeline=args.timeline,
        front_rows=front_rows,
        sort=args.sort,
    )
    half = pl.peer_offsets[1]
    peers = PeerBuffer(2 * half + guard.FLAG_BYTES, cpu, rank, TP, dev)
    peers.bytes.zero_()
    scr_pb = PeerBuffer(pl.scratch_bytes(), cpu, rank, 1, dev)
    scratch = scr_pb.bytes
    scratch.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(120.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    torch.manual_seed(1000 + m)
    valid_m = max(1, m - args.pfx_short)
    cores_all = [
        torch.randn(m, TP * KO, dtype=BF, device=dev) / (KO**0.5) for _ in range(nl)
    ]
    wo_all = [
        torch.randn(H, TP * KO, dtype=BF, device=dev) / (KO**0.5) for _ in range(nl)
    ]
    pfx_full = [torch.randn(valid_m, H, dtype=BF, device=dev) * 0.5 for _ in range(nl)]
    bank_full = [torch.randn(m, nbk, H, dtype=BF, device=dev) * 0.5 for _ in range(nl)]
    w_s = [torch.randn(H, dtype=BF, device=dev) * 0.5 + 1.0 for _ in range(nl)]
    w_p = [torch.randn(H, dtype=BF, device=dev) * 0.1 for _ in range(nl)]
    w_ln = [torch.randn(H, dtype=BF, device=dev) * 0.5 + 1.0 for _ in range(nl)]
    w_router = [torch.randn(E, H, dtype=BF, device=dev) / (H**0.5) for _ in range(nl)]
    w_down = [torch.randn(LAT, H, dtype=BF, device=dev) / (H**0.5) for _ in range(nl)]
    w_sgu = [
        torch.randn(TP * 2 * SI, H, dtype=BF, device=dev) / (H**0.5) for _ in range(nl)
    ]
    w_sd_all = [
        torch.randn(H, TP * SI, dtype=BF, device=dev) / (SI**0.5) for _ in range(nl)
    ]
    bias = [torch.randn(E, dtype=torch.float32, device=dev) * 0.05 for _ in range(nl)]

    cores = [c[:, rank * KO : (rank + 1) * KO].contiguous() for c in cores_all]
    wo = [w[:, rank * KO : (rank + 1) * KO].contiguous() for w in wo_all]
    bank = [b[:, :, cols].contiguous() for b in bank_full]
    cw_m = [(s.float() * p.float()).contiguous() for s, p in zip(w_s, w_p)]
    w_sd = [w[:, rank * SI : (rank + 1) * SI].contiguous() for w in w_sd_all]

    if args.front_map:
        wf = [
            torch.cat(
                [
                    sg[rank * n_sgu : (rank + 1) * n_sgu],
                    wr,
                    wd,
                ],
                dim=0,
            ).contiguous()
            for sg, wr, wd in zip(w_sgu, w_router, w_down)
        ]
    else:
        wf = [
            torch.cat(
                [
                    wr[rank * mf.E_R : (rank + 1) * mf.E_R],
                    wd[rank * mf.L_R : (rank + 1) * mf.L_R],
                    sg[rank * n_sgu : (rank + 1) * n_sgu],
                ],
                dim=0,
            ).contiguous()
            for sg, wr, wd in zip(w_sgu, w_router, w_down)
        ]

    pout = [torch.zeros(m, H, dtype=BF, device=dev) for _ in range(nl)]
    flat = [torch.ones(m * (LAT + H), dtype=BF, device=dev) for _ in range(nl)]
    moe_out = [f[: m * LAT].view(m, LAT) for f in flat]
    sh_out = [f[m * LAT :].view(m, H) for f in flat]
    lat_out = [torch.zeros(m, LAT, dtype=BF, device=dev) for _ in range(nl)]
    tk_ids = [torch.zeros(m, TOPK, dtype=torch.int32, device=dev) for _ in range(nl)]
    tk_w = [torch.zeros(m, TOPK, dtype=torch.float32, device=dev) for _ in range(nl)]
    logits = [torch.zeros(m, E, dtype=torch.float32, device=dev) for _ in range(nl)]
    acc_o = torch.zeros(m * H, dtype=BF, device=dev)
    cnt = torch.zeros(dp.counters(m, go, gf, nf), dtype=torch.int32, device=dev)
    if args.sort:
        sortb = [
            torch.zeros(
                mf.sort_layout(m, args.sort)["nbytes"], dtype=torch.uint8, device=dev
            )
            for _ in range(nl)
        ]
        for sb in sortb:
            mf.sort_init(sb, m, args.sort)
    else:
        sortb = [torch.zeros(256, dtype=torch.uint8, device=dev) for _ in range(nl)]

    def call_one(i):
        pfx_bytes = pfx_full[i].shape[0] * H * 2 - col0
        _run_compiled(
            launch,
            cores[i].data_ptr(),
            wo[i].data_ptr(),
            pfx_full[i].data_ptr() + col0,
            pout[i].data_ptr() + col0,
            bank[i].data_ptr(),
            cw_m[i].data_ptr(),
            w_ln[i].data_ptr(),
            wf[i].data_ptr(),
            acc_o.data_ptr(),
            bias[i].data_ptr(),
            lat_out[i].data_ptr(),
            tk_ids[i].data_ptr(),
            tk_w[i].data_ptr(),
            w_sd[i].data_ptr(),
            sh_out[i].data_ptr(),
            moe_out[i].data_ptr(),
            logits[i].data_ptr(),
            cnt.data_ptr(),
            scratch.data_ptr(),
            peers.addresses.data_ptr(),
            rank,
            epoch.data_ptr(),
            i,
            pfx_bytes,
            sortb[i].data_ptr(),
            torch.cuda.current_stream(),
        )

    def make_refs():
        refs = []
        for i in range(nl):
            pfx_pad = torch.zeros(m, H, dtype=BF, device=dev)
            pfx_pad[:valid_m] = pfx_full[i]
            o_ref = cores_all[i].float() @ wo_all[i].float().T
            p_ref = (o_ref + pfx_pad.float()).to(BF)
            x_ref = attn_res_f32_ref(p_ref, bank_full[i], nvm, w_s[i], w_p[i], w_ln[i])
            logits_ref = x_ref.float() @ w_router[i].float().T
            lat_ref = (x_ref.float() @ w_down[i].float().T).to(BF)
            sgu_r = (
                x_ref.float() @ w_sgu[i][rank * n_sgu : (rank + 1) * n_sgu].float().T
            ).to(BF)
            h_r = situ(sgu_r[:, :SI], sgu_r[:, SI:])
            sh_ref = (h_r.float() @ w_sd[i].float().T).to(BF)
            refs.append((p_ref, lat_ref, sh_ref, logits_ref))
        return refs

    if args.sort == "sgl":
        from layer_bench import sort_check

    def check(refs, st):
        """Check this repetition's outputs of every launch against refs, into st."""
        bad_ids = 0
        for i in range(nl):
            p_ref, lat_ref, sh_ref, logits_ref = refs[i]
            sig_own = torch.sigmoid(logits[i])
            _, own_idx = torch.topk(sig_own + bias[i], TOPK, dim=-1)

            st["p"] = max(st["p"], rel(pout[i][:, cols], p_ref[:, cols]))
            st["lat"] = max(st["lat"], rel(lat_out[i], lat_ref))
            st["sh"] = max(st["sh"], rel(sh_out[i], sh_ref))
            st["lg"] = max(st["lg"], rel(logits[i], logits_ref))
            # the launch's ids must be a top-k of its own scores; at an exact tie on the
            # k-th score either expert is one (the launch and torch.topk break it apart)
            sc = sig_own + bias[i]
            kth = sc.gather(-1, own_idx).min(-1, keepdim=True).values
            ids = tk_ids[i].long()
            # an out-of-range id fails its row; clamped only to gather it here
            in_range = ((ids >= 0) & (ids < E)).all(-1)
            bad_ids += int((~in_range).sum())
            ids = ids.clamp(0, E - 1)
            mine = sc.gather(-1, ids)
            ids_s = ids.sort(-1).values
            distinct = (ids_s.diff(dim=-1) != 0).all(-1)
            above = (sc > kth).sum(-1) == (mine > kth).sum(-1)
            top_ok = in_range & (mine >= kth).all(-1) & above & distinct
            st["own"] += int((~top_ok).sum())
            mine_w = sig_own.gather(-1, ids)
            st["own_w"] = max(
                st["own_w"], rel(tk_w[i], mine_w / mine_w.sum(-1, keepdim=True))
            )
            # every rank must route a row to the same experts, a tie broken the same way
            ids_0 = ids_s.clone()
            dist.broadcast(ids_0, src=0)
            st["route"] += int((ids_s != ids_0).any(-1).sum())
        # the sort validators index with the raw ids: skip them on every rank when any
        # rank saw an out-of-range id (that rank already fails on it)
        bad = torch.tensor([bad_ids], dtype=torch.int64, device=dev)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()):
            st["sort_skipped"] += bool(args.sort)
            return
        if args.sort == "aux16":
            st["sort"] += sort_check_aux16(m, nl, tk_ids, tk_w, moe_out, sortb, rank)
        elif args.sort == "sgl":
            st["sort"] += sort_check(m, nl, tk_ids, tk_w, lat_out, moe_out, sortb, rank)

    def verdict(st):
        """The worst rank's errors of one phase (eager or replay) and whether they pass."""
        worst = torch.tensor(
            [st["p"], st["lat"], st["sh"], st["lg"], st["own_w"]],
            dtype=torch.float64,
            device=dev,
        )
        counts = torch.tensor(
            [st["own"], st["route"], st["sort"], st["sort_skipped"]],
            dtype=torch.int64,
            device=dev,
        )
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
        e = dict(zip(("p", "lat", "sh", "lg", "own_w"), worst.tolist()))
        keys = ("own", "route", "sort", "sort_skipped")
        e.update(zip(keys, (int(c) for c in counts.tolist())))
        ok = (
            max(e["p"], e["lat"], e["sh"], e["lg"]) < 2e-2
            and e["own_w"] < 1e-5
            and e["own"] == e["route"] == e["sort"] == e["sort_skipped"] == 0
        )
        return e, ok

    phases = {}
    eager = dict(
        p=0.0,
        lat=0.0,
        sh=0.0,
        lg=0.0,
        own_w=0.0,
        own=0,
        route=0,
        sort=0,
        sort_skipped=0,
    )
    refs = make_refs()
    for _ in range(args.reps_check):
        for mo in moe_out:
            mo.fill_(1.0)
        epoch[0] += 1
        for i in range(nl):
            call_one(i)
        torch.cuda.synchronize()
        dist.barrier()
        check(refs, eager)
    phases["eager"] = verdict(eager)

    # Time in CUDA graph with optional L2 flush
    flush = (
        torch.empty(args.flush << 20, dtype=torch.uint8, device=dev)
        if args.flush
        else None
    )

    def step_fn(idx):
        if idx == 0:
            epoch[0] += 1
        if flush is not None:
            flush.add_(1)
        call_one(idx)

    graph, us = graph_us(step_fn, nl)
    # One more replay on fresh inputs (the same tensors the graph reads): a replay that skips
    # a launch or an input-dependent write keeps the previous inputs' outputs, which fail
    # here (a sort field whose value does not change with the inputs is not covered).
    torch.manual_seed(5000 + m)
    for i in range(nl):
        cores_all[i].copy_(torch.randn_like(cores_all[i]) / (KO**0.5))
        cores[i].copy_(cores_all[i][:, rank * KO : (rank + 1) * KO])
        pfx_full[i].copy_(torch.randn_like(pfx_full[i]) * 0.5)
    for mo in moe_out:
        mo.fill_(1.0)
    refs = make_refs()
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    replay = dict(
        p=0.0,
        lat=0.0,
        sh=0.0,
        lg=0.0,
        own_w=0.0,
        own=0,
        route=0,
        sort=0,
        sort_skipped=0,
    )
    check(refs, replay)
    phases["replay"] = verdict(replay)
    ok = all(phase_ok for _, phase_ok in phases.values())
    if rank == 0:
        for name, (e, phase_ok) in phases.items():
            print(
                f"dsa_post m {m:3d} nvb {nvm} pfx_short {args.pfx_short} {name:6s} "
                f"rel_l2 pout {e['p']:.2e} lat {e['lat']:.2e} sh {e['sh']:.2e} "
                f"logits {e['lg']:.2e} own_w {e['own_w']:.1e} own_diff {e['own']} "
                f"route_diff {e['route']} sort_bad {e['sort']}"
                + (
                    f" (sort check skipped in {e['sort_skipped']} reps: bad ids)"
                    if e["sort_skipped"]
                    else ""
                )
                + "  "
                + ("PASS" if phase_ok else "FAIL"),
                flush=True,
            )
        print(
            f"dsa_post m {m:3d} us: {us:5.1f}  " + ("PASS" if ok else "FAIL"),
            flush=True,
        )
    if args.timeline:
        timeline(pl, scratch, m, rank)
    peers.close()
    scr_pb.close()
    dist.barrier()
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=None)
    p.add_argument("--L", type=int, default=6)
    p.add_argument("--nf", type=int, default=2096)
    p.add_argument("--nbk", type=int, default=8)
    p.add_argument("--nvb-m", type=int, default=3)
    p.add_argument("--launches", type=int, default=8)
    p.add_argument("--reps-check", type=int, default=3)
    p.add_argument("--pfx-short", type=int, default=0)
    p.add_argument("--flush", type=int, default=256)
    p.add_argument("--front-map", action="store_true")
    p.add_argument("--timeline", action="store_true")
    p.add_argument("--sort", default=None, choices=(None, "sgl", "aux16"))
    args = p.parse_args()
    if args.reps_check < 1:
        p.error("--reps-check must be at least 1: the eager check would not run")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    all_ok = True
    rows_list = (
        [int(v) for v in args.rows.split(",")]
        if args.rows is not None
        else [2 * args.L, 4 * args.L, 8 * args.L]
    )
    for m in rows_list:
        all_ok &= bool(run(args, m, rank, dev, cpu))
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
