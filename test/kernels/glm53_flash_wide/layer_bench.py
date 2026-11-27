# SPDX-License-Identifier: Apache-2.0
"""Wide layer launch (``glm53_flash.gfx950.wide.layer``) on ``tp`` gfx950 GPUs against
torch / the standalone stages:

    R_in    (``--t1``) ``_mhc_post_torch`` of the rank-order sum of the MoE partials
            (bitwise); then the residual every check below uses
    core    the KDA task (``kda.standalone``) on the launch's own proj (bitwise)
    o       sum over ranks of core_r @ W_o,r^T (bf16 tolerance); every rank the same bits
    R_mid   ``_mhc_post_torch`` of the launch's o (bitwise)
    x_f, post_f, comb_f   ``_mhc_pre_torch`` + RMSNorm of R_mid
    ids, w  top-8 of sigmoid(x_f @ W_r^T) + bias (torch, the launch's x_f): the picked
            set (near-ties excused), weights; every rank the same bits

    torchrun --nproc-per-node 4 layer_bench.py [--reqs 3,6,16] [--stop fgate] [--time]
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

os.environ.setdefault("SGLANG_USE_AITER", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kimi_k31"))

from mhc_bench import mhc_post_ref, mhc_pre_ref, rmsnorm_ref  # noqa: E402
from tile_bench import bench  # noqa: E402

from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.config import (  # noqa: E402
    HIDDEN,
    MOE_SLOTS,
    N_EXPERTS,
    TOP_K,
)
from sgl_amd_mono.glm53_flash.gfx950.wide import coll, kda, layer, mhc  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.wide import sort as wsort  # noqa: E402

HC = mhc.HC
LB = -5.0
ONORM_EPS = 1e-5
NH, KD, P, NPROJ = kda.NH, kda.KD, kda.P, kda.NPROJ


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return out


def run(a, reqs, sbf, rank, tp, dev, cpu):
    tt = a.tt
    m = reqs * tt
    slots = reqs + 2
    L = tt
    # replicated: same seed on every rank
    gen = torch.Generator(device=dev).manual_seed(1000 + reqs)

    def rnd(*s, scale=1.0, dt=torch.float32, g=gen):
        return (torch.randn(*s, device=dev, generator=g) * scale).to(dt)

    res_in = rnd(m, HC, HIDDEN, dt=torch.bfloat16)
    # t1: res_in is the previous layer's R_mid; its post_f / comb_f; each rank's MoE partial
    post_p = torch.rand(m, HC, device=dev, generator=gen) * 2
    comb_p = torch.rand(m, HC * HC, device=dev, generator=gen) / 2
    moe_in = rnd(
        m,
        HIDDEN,
        scale=0.5,
        dt=torch.bfloat16,
        g=torch.Generator(device=dev).manual_seed(7000 + rank),
    )
    mhc_w = []
    for _ in range(2):
        mhc_w.append(
            dict(
                fn=rnd(24, HC * HIDDEN, scale=0.02),
                scale=torch.tensor([0.8, 0.6, 1.1], device=dev),
                base=rnd(24, scale=0.3),
                norm=(1 + rnd(HIDDEN, scale=0.1)).to(torch.bfloat16),
            )
        )
    w_r = rnd(N_EXPERTS, HIDDEN, scale=0.02, dt=torch.bfloat16)
    bias_r = rnd(N_EXPERTS, scale=0.5)
    # this rank's heads
    lg = torch.Generator(device=dev).manual_seed(100 + rank)
    w_in = rnd(NPROJ, HIDDEN, scale=0.02, dt=torch.bfloat16, g=lg)
    w_o = rnd(HIDDEN, P, scale=0.02, dt=torch.bfloat16, g=lg)
    conv_w = rnd(3 * P, 4, scale=0.3, g=lg)
    w_fgb = rnd(2, P, KD, scale=0.08, dt=torch.bfloat16, g=lg)
    a_log = torch.log(torch.rand(NH, device=dev, generator=lg) * 8 + 1)
    dt_bias = rnd(P, scale=0.5, g=lg)
    on_w = 1 + rnd(KD, scale=0.1, g=lg)
    conv_st = rnd(slots, 3, 3 * P, scale=0.5, dt=torch.bfloat16, g=lg).transpose(1, 2)
    sdt = torch.bfloat16 if sbf else torch.float32
    state = rnd(slots, NH, KD, KD, scale=0.05, dt=sdt, g=lg)
    idx = torch.arange(1, reqs + 1, device=dev, dtype=torch.int32)
    cs_idx = idx.clone()
    if reqs > 2:
        cs_idx[1] = -1

    def kbufs():
        return dict(
            rawv=torch.zeros(slots, NH, L, KD, device=dev, dtype=torch.bfloat16),
            rawk=torch.zeros(slots, NH, L, KD, device=dev, dtype=torch.bfloat16),
            ring_g=torch.zeros(slots, NH, L, KD, device=dev),
            ring_beta=torch.zeros(slots, NH, L, device=dev),
            iw=torch.zeros(
                slots, tt, 3, 3 * P, device=dev, dtype=torch.bfloat16
            ).transpose(2, 3),
            core=torch.zeros(m, P, device=dev, dtype=torch.bfloat16),
        )

    mine, ref = kbufs(), kbufs()
    z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)  # noqa: E731
    x_a, post_a, comb_a = z(m, HIDDEN, dt=torch.bfloat16), z(m, HC), z(m, HC * HC)
    proj = z(m, NPROJ, dt=torch.bfloat16)
    acc_o = z(m, HIDDEN, dt=torch.bfloat16)
    res_mid = z(m, HC, HIDDEN, dt=torch.bfloat16)
    x_f, post_f, comb_f = z(m, HIDDEN, dt=torch.bfloat16), z(m, HC), z(m, HC * HC)
    ids, wts, logits = z(m, MOE_SLOTS, dt=torch.int32), z(m, MOE_SLOTS), z(m, N_EXPERTS)
    res_cur = z(m, HC, HIDDEN, dt=torch.bfloat16)
    launch = layer.build(
        m,
        tt,
        tp,
        LB,
        ONORM_EPS,
        bool(sbf),
        True,
        a.stop,
        trace=a.trace,
        oneshot=m <= a.oneshot_rows,
        t1=a.t1,
        moe_sort=a.sort,
        body=not a.no_body,
    )
    plan = launch.plan
    epoch = z(plan["epoch_words"], dt=torch.int32)
    scratch = z(plan["scratch_words"], dt=torch.int32)
    peers = PeerBuffer(plan["peer_bytes"], cpu, rank, tp, dev)
    peers.bytes.zero_()
    iw_s = mine["iw"].stride()
    ints = (
        conv_st.stride(0),
        conv_st.stride(2),
        iw_s[0],
        iw_s[1],
        iw_s[3],
        mine["rawv"].stride(0),
        mine["rawk"].stride(0),
        mine["ring_g"].stride(0),
        mine["ring_beta"].stride(0),
        L,
    )
    fa, ff = mhc_w
    ptrs = (
        res_in,
        fa["fn"],
        fa["scale"],
        fa["base"],
        fa["norm"],
        x_a,
        post_a,
        comb_a,
        w_in,
        proj,
        conv_w,
        conv_st,
        cs_idx,
        mine["iw"],
        idx,
        w_fgb,
        a_log,
        dt_bias,
        state,
        idx,
        mine["rawv"],
        mine["rawk"],
        mine["ring_g"],
        mine["ring_beta"],
        idx,
        on_w,
        mine["core"],
        w_o,
        acc_o,
        ff["fn"],
        ff["scale"],
        ff["base"],
        ff["norm"],
        res_mid,
        x_f,
        post_f,
        comb_f,
        w_r,
        bias_r,
        ids,
        wts,
        logits,
        moe_in,
        post_p,
        comb_p,
        res_cur,
        epoch,
        scratch,
    )
    pl = [t.data_ptr() for t in ptrs] + [peers.addresses.data_ptr(), rank]
    sly = wsort.layout(m)
    sbuf = torch.empty(sly["nbytes"], device=dev, dtype=torch.uint8)
    wsort.init(sbuf, m)
    moe_out = torch.full((m, HIDDEN), 7.0, device=dev, dtype=torch.bfloat16)

    def go(mt=m):
        launch(
            *pl,
            *ints,
            sbuf.data_ptr(),
            moe_out.data_ptr(),
            mt,
            stream=torch.cuda.current_stream(),
        )

    dist.barrier(cpu)
    go()
    torch.cuda.synchronize()
    errs = {}
    fa, ff = mhc_w
    r_attn = res_in
    if a.t1:
        moes = gather(moe_in)
        msum = moes[0].float()
        for t in moes[1:]:
            msum = msum + t.float()
        errs["R_in"] = rel(
            res_cur, mhc_post_ref(msum.to(torch.bfloat16), res_in, post_p, comb_p)
        )
        errs["rin_ranks"] = max(float((t != res_cur).any()) for t in gather(res_cur))
        r_attn = res_cur
    pa_ref, ca_ref, lia = mhc_pre_ref(r_attn, fa["fn"], fa["scale"], fa["base"])
    errs["x_a"] = rel(x_a, rmsnorm_ref(lia, fa["norm"]))
    errs["post_a"] = rel(post_a, pa_ref)
    errs["comb_a"] = rel(comb_a, ca_ref)
    if a.no_body:
        if a.stop == "route":
            errs.update(route_check(x_a, w_r, bias_r, ids, wts))
            errs["logits"] = rel(logits, x_a.float() @ w_r.float().T)
            errs["route_ranks"] = max(
                float((t != ids).any()) for t in gather(ids)
            ) + max(float((t != wts).any()) for t in gather(wts))
    # core: the KDA task alone on our proj
    kl = kda.standalone(reqs, tt, LB, ONORM_EPS, bool(sbf), True)
    kp = (
        proj,
        conv_w,
        conv_st,
        cs_idx,
        ref["iw"],
        idx,
        w_fgb,
        a_log,
        dt_bias,
        state,
        idx,
        ref["rawv"],
        ref["rawk"],
        ref["ring_g"],
        ref["ring_beta"],
        idx,
        on_w,
        ref["core"],
    )
    if not a.no_body:
        kl(*[t.data_ptr() for t in kp], *ints, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        errs["core"] = rel(mine["core"], ref["core"])
        errs["ring_bad"] = sum(
            int((mine[k] != ref[k]).sum())
            for k in ("rawv", "rawk", "ring_g", "ring_beta", "iw")
        )
        errs["proj"] = rel(proj, (x_a.float() @ w_in.float().T).to(torch.bfloat16))
    if a.stop != "kda" and not a.no_body:
        offs, half = coll.layout(
            m, plan["rb_o"], plan["oproj"].bn, tp, plan["oneshot"], plan["t1"]
        )
        par = (epoch[0].item() - 1) % 2
        if plan["oneshot"]:
            os_ = peers.bytes[par * half + offs["os"] :][: tp * m * HIDDEN * 2]
            os_ = os_.view(torch.bfloat16).view(tp, m, HIDDEN).float()
            o = os_[0]
            for p_ in range(1, tp):
                o = o + os_[p_]
            o = o.to(torch.bfloat16)
        else:
            o = peers.bytes[par * half + offs["ag"] :][: m * HIDDEN * 2]
            o = o.view(torch.bfloat16).view(m, HIDDEN).clone()
        cores, wos = gather(mine["core"]), gather(w_o)
        o_ref = sum(c.float() @ w.float().T for c, w in zip(cores, wos)).to(
            torch.bfloat16
        )
        errs["o"] = rel(o, o_ref)
        errs["o_ranks"] = max(float((t != o).any()) for t in gather(o))
        if layer.STOPS.index(a.stop) >= 2:
            r_ref = mhc_post_ref(o, r_attn, post_a, comb_a)
            errs["R_mid"] = rel(res_mid, r_ref)
            p_ref, c_ref, li_ref = mhc_pre_ref(
                res_mid, ff["fn"], ff["scale"], ff["base"]
            )
            xr = rmsnorm_ref(li_ref, ff["norm"])
            errs["x_f"] = rel(x_f, xr)
            if errs["x_f"] > 1e-3 and rank == 0:
                print(
                    "x_f",
                    x_f.float().norm().item(),
                    "ref",
                    xr.float().norm().item(),
                    "x_a",
                    x_a.float().norm().item(),
                    "rows",
                    x_f.float().norm(dim=1)[:4].tolist(),
                    "x_a vs ref",
                    rel(x_a, xr),
                    flush=True,
                )
            errs["post_f"] = rel(post_f, p_ref)
            errs["comb_f"] = rel(comb_f, c_ref)
            errs["xf_ranks"] = max(float((t != x_f).any()) for t in gather(x_f))
        if a.stop == "route":
            errs.update(route_check(x_f, w_r, bias_r, ids, wts))
            errs["logits"] = rel(logits, x_f.float() @ w_r.float().T)
            errs["route_ranks"] = max(
                float((t != ids).any()) for t in gather(ids)
            ) + max(float((t != wts).any()) for t in gather(wts))
    if a.sort and a.stop == "route":
        bad = sort_check(sbuf, sly, ids, wts, m, m, moe_out)
        xa0 = x_a.clone()
        pad_bad = 0
        # fewer tokens than the bucket, then back: the previous launch's rows reset
        for mt in (max(tt, m - tt), max(1, m // 2 - 1), m):
            moe_out.fill_(7.0)
            go(mt)
            torch.cuda.synchronize()
            bad += sort_check(sbuf, sly, ids, wts, m, mt, moe_out)
            if not a.t1:
                # R_in rows past mt read 0: rows below are unchanged, pad x_a is 0
                # (x_f is not bitwise run to run: split-K in_proj accumulates atomically)
                pad_bad += int((x_a[:mt] != xa0[:mt]).sum()) + int(
                    (x_a[mt:] != 0).sum()
                )
        errs["pad_bad"] = pad_bad
        errs["sort_bad"] = bad
    # repeated launches: the epoch hand-offs, across ranks
    for _ in range(a.launches):
        go()
    torch.cuda.synchronize()
    errs["ep"] = epoch[0].item()
    line = (
        f"rank {rank} reqs {reqs:2d} rows {m:3d} os {int(plan['oneshot'])} state {'bf16' if sbf else 'f32 '} "
        + "  ".join(
            f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}"
            for k, v in errs.items()
        )
    )
    if a.time:
        dist.barrier(cpu)
        line += f"  layer {bench(lambda _w: go(), [None] * 4):6.1f} us"
    print(line, flush=True)
    if a.trace:
        from sgl_amd_mono.common.plan import BLOCKS
        from sgl_amd_mono.glm53_flash.gfx950.wide.m1a import TRACE_POINTS

        sc = layer.scratch_layout(m, tt, True)
        acc = None
        for _ in range(20):
            go()
            torch.cuda.synchronize()
            tr = scratch[sc["trace"] :][: BLOCKS * TRACE_POINTS * 2].view(torch.int64)
            tr = tr.view(BLOCKS, TRACE_POINTS).double()
            tr = (tr - tr[:, 0].min()) / 100.0
            acc = tr if acc is None else acc + tr
        tr = acc / 20
        names = [
            "start",
            "aseam",
            "agate",
            "inproj",
            "kda",
            "oproj",
            "reduce",
            "fseam",
            "fgate",
            "epoch",
            "router",
            "route",
            "rwait",
            "sort",
        ]
        for p_, nm in sorted(
            enumerate(names), key=lambda x: {9: 99, 12: 10.5}.get(x[0], x[0])
        ):
            col = tr[:, p_]
            col = col[(col > -1e5) & (col < 1e5)]
            if col.numel() and rank == 0:
                print(
                    f"    {nm:>7}  median {col.median():6.2f}  max {col.max():6.2f} us",
                    flush=True,
                )
    peers.close()
    tol = {
        "R_in": 1e-6,
        "rin_ranks": 0.0,
        "x_a": 1e-3,
        "post_a": 1e-5,
        "comb_a": 1e-5,
        "core": 1e-6,
        "proj": 1e-2,
        "o": 1e-2,
        "o_ranks": 0.0,
        "R_mid": 1e-6,
        "x_f": 1e-3,
        "post_f": 1e-5,
        "comb_f": 1e-5,
        "xf_ranks": 0.0,
        "ids_bad": 0,
        "w": 1e-5,
        "logits": 1e-5,
        "shared": 0,
        "route_ranks": 0.0,
        "sort_bad": 0,
        "ring_bad": 0,
        "pad_bad": 0,
    }
    return all(errs[k] <= v for k, v in tol.items() if k in errs)


def sort_check(sbuf, ly, ids, wts, b, mt, moe_out):
    """Mismatches of the launch's sort of tokens 0 .. mt-1 against AITER's BM16 aux sort
    layout built here (an expert's pairs by token, pad b), and of the zeroed output."""
    v = {k: t.cpu() for k, t in wsort.views(sbuf, ly, mt).items()}
    ids_c, w_c = ids[:mt].cpu().long(), wts[:mt].cpu()
    e_ids, e_w, e_mi, e_eids = [], [], [], []
    rev = torch.zeros(mt * MOE_SLOTS, dtype=torch.int32)
    for e in range(wsort.E):
        ts, ss = (ids_c == e).nonzero(as_tuple=True)
        n = ts.numel()
        if n == 0:
            continue
        nb = -(-n // wsort.BM)
        for i in range(n):
            t, sl = int(ts[i]), int(ss[i])
            rev[t * MOE_SLOTS + sl] = len(e_ids)
            e_ids.append((sl << 24) | t)
            e_w.append(float(w_c[t, sl]))
            e_mi.append(t)
        for _ in range(nb * wsort.BM - n):
            e_ids.append(b)
            e_w.append(0.0)
            e_mi.append(b)
        e_eids += [e] * nb
    nv = len(e_ids)
    bad = int(v["num_valid_ids"][0] != nv) + int(v["num_valid_ids"][1] != mt)
    bad += int(
        (v["sorted_token_ids"][:nv] != torch.tensor(e_ids, dtype=torch.int32)).sum()
    )
    bad += int((v["sorted_weights"][:nv] != torch.tensor(e_w)).sum())
    bad += int((v["m_indices"][:nv] != torch.tensor(e_mi, dtype=torch.int32)).sum())
    bad += int(
        (
            v["sorted_expert_ids"][: nv // wsort.BM]
            != torch.tensor(e_eids, dtype=torch.int32)
        ).sum()
    )
    bad += int((v["reverse_sorted"] != rev).sum())
    bad += int((moe_out != 0).any(1).sum())
    return bad


def route_check(x_f, w_r, bias_r, ids, wts):
    """Picked sets against torch (a row whose 8th and 9th keys are within 1e-5 is
    excused), the weights in id order, the shared slot."""
    s = torch.sigmoid(x_f.float() @ w_r.float().T)
    keys = s + bias_r.to(torch.bfloat16).float()
    kv, ki = keys.topk(TOP_K + 1, dim=1)
    ref_ids = ki[:, :TOP_K]
    tie = (kv[:, TOP_K - 1] - kv[:, TOP_K]) < 1e-5
    mi = ids[:, :TOP_K].long()
    same = (mi.sort(1).values == ref_ids.sort(1).values).all(1)
    bad = int((~same & ~tie).sum().item())
    ws = s.gather(1, mi)
    w_ref = ws / ws.sum(1, keepdim=True) * 2.5
    shared = int(((ids[:, TOP_K] != N_EXPERTS) | (wts[:, TOP_K] != 1.0)).sum().item())
    return {
        "ids_bad": bad,
        "ties": int(tie.sum().item()),
        "w": rel(wts[:, :TOP_K], w_ref),
        "shared": shared,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="3,6,16")
    ap.add_argument("--tt", type=int, default=6)
    ap.add_argument("--state-bf16", default="1")
    ap.add_argument("--stop", default="route", choices=layer.STOPS + ("agate",))
    ap.add_argument("--launches", type=int, default=10)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--oneshot-rows", type=int, default=0)
    ap.add_argument("--t1", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--sort", action="store_true")
    ap.add_argument(
        "--no-body",
        action="store_true",
        help="a DSA layer's seam launch (stop agate / route)",
    )
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    tp = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    ok = True
    for reqs in map(int, a.reqs.split(",")):
        for sbf in map(int, a.state_bf16.split(",")):
            ok &= run(a, reqs, sbf, rank, tp, dev, cpu)
    flags = gather(torch.tensor([int(ok)], device=dev))
    if rank == 0:
        print("PASS" if all(f.item() for f in flags) else "FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
