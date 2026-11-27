# SPDX-License-Identifier: Apache-2.0
"""Wide KDA verify task (``glm53_flash.gfx950.wide.kda``, standalone, one CTA a
(request, head)) on 1 gfx950 GPU against SGLang's chain on the same inputs: torch bf16
f_b / g_b (``fused_fg_b_proj``), the FlyDSL ring verify (sglang#51126,
``kda_verify_gfx950_flydsl``), then the gated RMSNorm (``FusedRMSNormGated``, sigmoid).

    python kda_bench.py [--reqs 3,6,16] [--tt 6] [--state-bf16 0,1] [--time]
"""

import argparse
import importlib.util
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

from sgl_amd_mono.glm53_flash.gfx950.wide import kda  # noqa: E402

REF = os.environ.get(
    "SGL_KDA_VERIFY_FLYDSL",
    "/shared_nfs/jacchang/wide1120/kref/sgl_kda_verify_flydsl.py",
)
LB = -5.0
ONORM_EPS = 1e-5
NH, KD, P, NPROJ = kda.NH, kda.KD, kda.P, kda.NPROJ


def load_ref():
    spec = importlib.util.spec_from_file_location("sgl_kda_verify_flydsl", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_kda_verify_kernel


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="3,6,16")
    ap.add_argument("--tt", type=int, default=6)
    ap.add_argument("--state-bf16", default="0,1")
    ap.add_argument("--iw", type=int, default=1)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    create_ref = load_ref()
    torch.manual_seed(0)
    dev = "cuda"
    tt = a.tt
    worst = 0.0
    for reqs in map(int, a.reqs.split(",")):
        for sbf in map(int, a.state_bf16.split(",")):
            m = reqs * tt
            slots = reqs + 2
            proj = (torch.randn(m, NPROJ, device=dev) * 0.5).to(torch.bfloat16)
            conv_w = torch.randn(3 * P, 4, device=dev) * 0.3
            conv_st = (
                (torch.randn(slots, 3, 3 * P, device=dev) * 0.5)
                .to(torch.bfloat16)
                .transpose(1, 2)
            )
            w_fgb = (torch.randn(2, P, KD, device=dev) * 0.08).to(torch.bfloat16)
            a_log = torch.log(torch.rand(NH, device=dev) * 8 + 1)
            dt_bias = torch.randn(P, device=dev) * 0.5
            sdt = torch.bfloat16 if sbf else torch.float32
            state = (torch.randn(slots, NH, KD, KD, device=dev) * 0.05).to(sdt)
            on_w = 1 + 0.1 * torch.randn(KD, device=dev)
            perm = torch.randperm(slots - 1, device=dev)[:reqs].to(torch.int32) + 1
            cs_idx = perm.clone()
            if reqs > 2:
                cs_idx[1] = -1  # a pad request
            h0_idx = perm.clone()
            ring_idx = perm.clone()
            iw_idx = perm.clone()
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
                )

            mine, ref = bufs(), bufs()
            core = torch.zeros(m, P, device=dev, dtype=torch.bfloat16)
            iw_s = mine["iw"].stride()
            launch = kda.standalone(reqs, tt, LB, ONORM_EPS, bool(sbf), bool(a.iw))
            ptrs = (
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
                core,
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
            # SGLang's chain
            fa = proj[:, kda.FA_0 : kda.FA_0 + KD]
            ga = proj[:, kda.GA_0 : kda.GA_0 + KD]
            f_out = (fa @ w_fgb[0].T).to(torch.bfloat16).contiguous()
            g_out = (ga @ w_fgb[1].T).to(torch.bfloat16)
            b2 = proj[:, kda.BETA_0 : kda.BETA_0 + NH]
            o = torch.empty(m, NH, KD, device=dev, dtype=torch.bfloat16)
            rlaunch = create_ref(tt, 64, NH, NH, LB, KD**-0.5, bool(a.iw), bool(sbf))
            riw = ref["iw"].stride()
            rlaunch(
                proj.data_ptr(),
                conv_w.data_ptr(),
                conv_st.data_ptr(),
                cs_idx.data_ptr(),
                ref["iw"].data_ptr(),
                iw_idx.data_ptr(),
                f_out.data_ptr(),
                b2.data_ptr(),
                a_log.data_ptr(),
                dt_bias.data_ptr(),
                state.data_ptr(),
                h0_idx.data_ptr(),
                o.data_ptr(),
                ref["rawv"].data_ptr(),
                ref["rawk"].data_ptr(),
                ref["ring_g"].data_ptr(),
                ref["ring_beta"].data_ptr(),
                ring_idx.data_ptr(),
                NPROJ,
                conv_st.stride(0),
                conv_st.stride(2),
                riw[0],
                riw[1],
                riw[3],
                f_out.stride(0),
                NPROJ,
                ref["rawv"].stride(0),
                ref["rawk"].stride(0),
                ref["ring_g"].stride(0),
                ref["ring_beta"].stride(0),
                L,
                reqs * NH,
                torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            of = o.float()
            rstd = torch.rsqrt(of.square().mean(-1, keepdim=True) + ONORM_EPS)
            y = (of * rstd * on_w * torch.sigmoid(g_out.float().view(m, NH, KD))).to(
                torch.bfloat16
            )
            live = (cs_idx >= 0).repeat_interleave(tt)
            y[~live] = 0
            y = y.view(m, P)
            errs = {"core": rel(core, y)}
            for k in ("rawk", "rawv", "ring_g", "ring_beta", "iw"):
                errs[k] = rel(mine[k], ref[k])
            exact = {
                k: (mine[k] == ref[k]).float().mean().item()
                for k in ("rawk", "rawv", "ring_beta", "iw")
            }
            worst = max(worst, *errs.values())
            line = (
                f"reqs {reqs:2d} rows {m:3d} state {'bf16' if sbf else 'f32 '}  "
                + "  ".join(f"{k} {v:.1e}" for k, v in errs.items())
                + "  exact "
                + " ".join(f"{k}={v:.3f}" for k, v in exact.items())
            )
            if a.time:
                line += f"  wide {bench(lambda _w: run(), [None] * 4):6.1f} us"
                fg_in = torch.stack((fa, ga))
                fg_out = torch.empty(2, m, P, device=dev, dtype=torch.bfloat16)
                wt = w_fgb.transpose(1, 2)

                def sgl():
                    torch.bmm(fg_in, wt, out=fg_out)
                    rlaunch(
                        proj.data_ptr(),
                        conv_w.data_ptr(),
                        conv_st.data_ptr(),
                        cs_idx.data_ptr(),
                        ref["iw"].data_ptr(),
                        iw_idx.data_ptr(),
                        fg_out[0].data_ptr(),
                        b2.data_ptr(),
                        a_log.data_ptr(),
                        dt_bias.data_ptr(),
                        state.data_ptr(),
                        h0_idx.data_ptr(),
                        o.data_ptr(),
                        ref["rawv"].data_ptr(),
                        ref["rawk"].data_ptr(),
                        ref["ring_g"].data_ptr(),
                        ref["ring_beta"].data_ptr(),
                        ring_idx.data_ptr(),
                        NPROJ,
                        conv_st.stride(0),
                        conv_st.stride(2),
                        riw[0],
                        riw[1],
                        riw[3],
                        P,
                        NPROJ,
                        ref["rawv"].stride(0),
                        ref["rawk"].stride(0),
                        ref["ring_g"].stride(0),
                        ref["ring_beta"].stride(0),
                        L,
                        reqs * NH,
                        torch.cuda.current_stream(),
                    )

                us = bench(lambda _w: sgl(), [None] * 4)
                line += f"  sgl bmm+verify {us:6.1f} us (+ o_norm)"
            print(line, flush=True)
    print("PASS" if worst < 2e-2 else "FAIL", f"worst rel {worst:.1e}")


if __name__ == "__main__":
    main()
