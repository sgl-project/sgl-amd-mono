# SPDX-License-Identifier: Apache-2.0
"""Wide launch milestone 1a (``glm53_flash.gfx950.wide.m1a``: aseam -> agate -> in_proj
-> KDA verify in one persistent launch) on 1 gfx950 GPU against the standalone stages:
mHC pre (``mhc.standalone``), torch bf16 in_proj, and the KDA task (``kda.standalone``)
run on the launch's own proj. Repeats the launch to exercise the epoch hand-offs.

    python m1a_bench.py [--reqs 3,6,16] [--tt 6] [--state-bf16 1] [--launches 20] [--time]
"""

import argparse
import os
import sys

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kimi_k31"
    ),
)

from tile_bench import bench  # noqa: E402

from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.wide import kda, m1a, mhc, tile_dma  # noqa: E402

HC = mhc.HC
LB = -5.0
ONORM_EPS = 1e-5
NH, KD, P, NPROJ = kda.NH, kda.KD, kda.P, kda.NPROJ


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="3,6,16")
    ap.add_argument("--tt", type=int, default=6)
    ap.add_argument("--state-bf16", default="1")
    ap.add_argument("--launches", type=int, default=20)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    tt = a.tt
    fn = torch.randn(24, HC * HIDDEN, device=dev) * 0.02
    hc_scale = torch.tensor([0.8, 0.6, 1.1], device=dev)
    hc_base = torch.randn(24, device=dev) * 0.3
    norm_w = (1 + 0.1 * torch.randn(HIDDEN, device=dev)).to(torch.bfloat16)
    w_in = (torch.randn(NPROJ, HIDDEN, device=dev) * 0.02).to(torch.bfloat16)
    conv_w = torch.randn(3 * P, 4, device=dev) * 0.3
    w_fgb = (torch.randn(2, P, KD, device=dev) * 0.08).to(torch.bfloat16)
    a_log = torch.log(torch.rand(NH, device=dev) * 8 + 1)
    dt_bias = torch.randn(P, device=dev) * 0.5
    on_w = 1 + 0.1 * torch.randn(KD, device=dev)
    worst = 0.0
    for reqs in map(int, a.reqs.split(",")):
        for sbf in map(int, a.state_bf16.split(",")):
            m = reqs * tt
            slots = reqs + 2
            res_in = torch.randn(m, HC, HIDDEN, device=dev).to(torch.bfloat16)
            conv_st = (
                (torch.randn(slots, 3, 3 * P, device=dev) * 0.5)
                .to(torch.bfloat16)
                .transpose(1, 2)
            )
            sdt = torch.bfloat16 if sbf else torch.float32
            state = (torch.randn(slots, NH, KD, KD, device=dev) * 0.05).to(sdt)
            perm = torch.randperm(slots - 1, device=dev)[:reqs].to(torch.int32) + 1
            cs_idx = perm.clone()
            if reqs > 2:
                cs_idx[1] = -1
            h0_idx, ring_idx, iw_idx = perm.clone(), perm.clone(), perm.clone()
            L = tt

            def bufs():
                return dict(
                    rawv=torch.zeros(
                        slots, NH, L, KD, device=dev, dtype=torch.bfloat16
                    ),
                    rawk=torch.zeros(
                        slots, NH, L, KD, device=dev, dtype=torch.bfloat16
                    ),
                    ring_g=torch.zeros(slots, NH, L, KD, device=dev),
                    ring_beta=torch.zeros(slots, NH, L, device=dev),
                    iw=torch.zeros(
                        slots, tt, 3, 3 * P, device=dev, dtype=torch.bfloat16
                    ).transpose(2, 3),
                    core=torch.zeros(m, P, device=dev, dtype=torch.bfloat16),
                )

            mine, ref = bufs(), bufs()
            x = torch.zeros(m, HIDDEN, device=dev, dtype=torch.bfloat16)
            post_out = torch.zeros(m, HC, device=dev)
            comb_out = torch.zeros(m, HC * HC, device=dev)
            proj = torch.full((m, NPROJ), 7.0, device=dev, dtype=torch.bfloat16)
            launch = m1a.build(m, tt, LB, ONORM_EPS, bool(sbf), True)
            plan = launch.plan
            epoch = torch.zeros(plan["epoch_words"], device=dev, dtype=torch.int32)
            scratch = torch.zeros(plan["scratch_words"], device=dev, dtype=torch.int32)
            iw_s = mine["iw"].stride()
            ptrs = (
                res_in,
                fn,
                hc_scale,
                hc_base,
                norm_w,
                x,
                post_out,
                comb_out,
                w_in,
                proj,
                conv_w,
                conv_st,
                cs_idx,
                mine["iw"],
                iw_idx,
                w_fgb,
                a_log,
                dt_bias,
                state,
                h0_idx,
                mine["rawv"],
                mine["rawk"],
                mine["ring_g"],
                mine["ring_beta"],
                ring_idx,
                on_w,
                mine["core"],
                epoch,
                scratch,
            )
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
            pl = [t.data_ptr() for t in ptrs]

            def run():
                launch(*pl, *ints, stream=torch.cuda.current_stream())

            run()
            torch.cuda.synchronize()
            # reference chain: mHC pre standalone, torch in_proj, KDA standalone on our proj
            pmix = torch.zeros(mhc.pmix_words(m), device=dev)
            x_r = torch.zeros_like(x)
            post_r, comb_r = torch.zeros_like(post_out), torch.zeros_like(comb_out)
            dummy = torch.zeros(1, device=dev)
            mhc.standalone(m, False)(
                *[
                    t.data_ptr()
                    for t in (
                        res_in,
                        dummy,
                        dummy,
                        dummy,
                        dummy,
                        fn,
                        hc_scale,
                        hc_base,
                        norm_w,
                        pmix,
                        x_r,
                        post_r,
                        comb_r,
                    )
                ],
                stream=torch.cuda.current_stream(),
            )
            proj_r = (x_r.float() @ w_in.float().T).to(torch.bfloat16)
            kl = kda.standalone(reqs, tt, LB, ONORM_EPS, bool(sbf), True)
            kp = (
                proj,
                conv_w,
                conv_st,
                cs_idx,
                ref["iw"],
                iw_idx,
                w_fgb,
                a_log,
                dt_bias,
                state,
                h0_idx,
                ref["rawv"],
                ref["rawk"],
                ref["ring_g"],
                ref["ring_beta"],
                ring_idx,
                on_w,
                ref["core"],
            )
            kl(*[t.data_ptr() for t in kp], *ints, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            errs = {
                "x": rel(x, x_r),
                "post": rel(post_out, post_r),
                "comb": rel(comb_out, comb_r),
                "proj": rel(proj, proj_r),
            }
            for k in ("core", "rawk", "rawv", "ring_g", "ring_beta", "iw"):
                errs[k] = rel(mine[k], ref[k])
            outs = dict(mine, proj=proj, x=x, comb=comb_out)
            first = {k: v.clone() for k, v in outs.items()}
            # repeated launches: every epoch must reproduce the first (the split-K adds
            # may round differently, so proj and what follows compare by tolerance)
            drift = 0.0
            for _ in range(a.launches):
                run()
            torch.cuda.synchronize()
            dk = {k: rel(outs[k], first[k]) for k in outs}
            drift = max(dk.values())
            exact_mhc = all(torch.equal(outs[k], first[k]) for k in ("x", "comb"))
            ep = epoch[0].item()
            errs["drift"] = drift
            errs["drift_proj"] = dk["proj"]
            assert exact_mhc, "mHC outputs changed across launches"
            worst = max(worst, *errs.values())
            line = f"reqs {reqs:2d} rows {m:3d} ep {ep:3d}  " + "  ".join(
                f"{k} {v:.1e}" for k, v in errs.items()
            )
            if a.time:
                t_m1a = bench(lambda _w: run(), [None] * 4)
                g = plan["gemm"]
                dl = tile_dma.standalone(
                    m,
                    NPROJ,
                    HIDDEN,
                    g.bm,
                    g.bn,
                    g.wm,
                    g.wn,
                    g.ks,
                    g.bk,
                    g.stages,
                    "linbf",
                )
                pz = torch.zeros(m, NPROJ, device=dev, dtype=torch.bfloat16)
                ml = mhc.standalone(m, False)
                mp = [
                    t.data_ptr()
                    for t in (
                        res_in,
                        dummy,
                        dummy,
                        dummy,
                        dummy,
                        fn,
                        hc_scale,
                        hc_base,
                        norm_w,
                        pmix,
                        x_r,
                        post_r,
                        comb_r,
                    )
                ]
                kpl = [t.data_ptr() for t in kp]

                def chain():
                    ml(*mp, stream=torch.cuda.current_stream())
                    pz.zero_()
                    dl(
                        x_r.data_ptr(),
                        w_in.data_ptr(),
                        pz.data_ptr(),
                        stream=torch.cuda.current_stream(),
                    )
                    kl(*kpl, *ints, stream=torch.cuda.current_stream())

                t_chain = bench(lambda _w: chain(), [None] * 4)
                line += f"  m1a {t_m1a:6.1f} us  standalone chain {t_chain:6.1f} us"
            print(line, flush=True)
    print("PASS" if worst < 2e-2 else "FAIL", f"worst rel {worst:.1e}")


if __name__ == "__main__":
    main()
