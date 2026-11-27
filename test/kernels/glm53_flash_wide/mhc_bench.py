# SPDX-License-Identifier: Apache-2.0
"""Wide mHC pre stages (``glm53_flash.gfx950.wide.mhc``, standalone slice + gate
launches) on 1 gfx950 GPU against SGLang's ``_mhc_post_torch`` + ``_mhc_pre_torch``.

    python mhc_bench.py [--rows 16,36,96] [--post 0,1] [--time]
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

from sgl_amd_mono.glm53_flash.gfx950.config import (  # noqa: E402
    EPS,
    HC_EPS,
    HC_POST_MULT,
    HIDDEN,
    SINKHORN_ITERS,
)
from sgl_amd_mono.glm53_flash.gfx950.wide import mhc  # noqa: E402

HC = 4


def mhc_pre_ref(residual, fn, hc_scale, hc_base):
    """SGLang ``_mhc_pre_torch`` at GLM-5.3-Flash's constants, plus the layer's RMSNorm input."""
    s, n, h = residual.shape
    x_flat = residual.view(s, n * h).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + EPS)
    mixes = torch.nn.functional.linear(x_flat, fn) * rsqrt
    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + HC_EPS
    post = HC_POST_MULT * torch.sigmoid(
        mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n]
    )
    comb = mixes[:, 2 * n :].view(s, n, n) * hc_scale[2] + hc_base[2 * n :].view(n, n)
    comb = comb.softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    li = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    return post, comb.reshape(s, n * n), li


def rmsnorm_ref(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * w.float()).to(
        x.dtype
    )


def mhc_post_ref(x, residual, post, comb):
    out = post.unsqueeze(-1) * x.unsqueeze(1) + (
        comb.view(-1, HC, HC).unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    return out.type_as(x)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16,36,96")
    ap.add_argument("--post", default="0,1")
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    fn = torch.randn(24, HC * HIDDEN, device=dev) * 0.02
    hc_scale = torch.tensor([0.8, 0.6, 1.1], device=dev)
    hc_base = torch.randn(24, device=dev) * 0.3
    norm_w = (1 + 0.1 * torch.randn(HIDDEN, device=dev)).to(torch.bfloat16)
    worst = 0.0
    for s in map(int, a.rows.split(",")):
        for post in map(int, a.post.split(",")):
            res_in = torch.randn(s, HC, HIDDEN, device=dev).to(torch.bfloat16)
            x_in = torch.randn(s, HIDDEN, device=dev).to(torch.bfloat16)
            post_in = 2 * torch.sigmoid(torch.randn(s, HC, device=dev))
            comb_in = torch.softmax(torch.randn(s, HC, HC, device=dev), -1).reshape(
                s, HC * HC
            )
            res_out = torch.zeros_like(res_in)
            pmix = torch.zeros(mhc.pmix_words(s), device=dev)
            x_out = torch.zeros(s, HIDDEN, device=dev, dtype=torch.bfloat16)
            post_out = torch.zeros(s, HC, device=dev)
            comb_out = torch.zeros(s, HC * HC, device=dev)
            launch = mhc.standalone(s, bool(post))
            ptrs = [
                t.data_ptr()
                for t in (
                    res_in,
                    res_out,
                    x_in,
                    post_in,
                    comb_in,
                    fn,
                    hc_scale,
                    hc_base,
                    norm_w,
                    pmix,
                    x_out,
                    post_out,
                    comb_out,
                )
            ]

            def fn_run():
                launch(*ptrs, stream=torch.cuda.current_stream())

            fn_run()
            torch.cuda.synchronize()
            r_ref = mhc_post_ref(x_in, res_in, post_in, comb_in) if post else res_in
            p_ref, c_ref, li_ref = mhc_pre_ref(r_ref, fn, hc_scale, hc_base)
            x_ref = rmsnorm_ref(li_ref, norm_w)
            errs = {
                "x": rel(x_out, x_ref),
                "post": rel(post_out, p_ref),
                "comb": rel(comb_out, c_ref),
            }
            if post:
                errs["R'"] = rel(res_out, r_ref)
                errs["R'_ulp"] = (
                    (res_out.float() != r_ref.float()).float().mean().item()
                )
            worst = max(worst, errs["x"], errs["post"], errs["comb"])
            line = f"rows {s:3d} post {post}  " + "  ".join(
                f"{k} {v:.1e}" for k, v in errs.items()
            )
            if a.time:
                line += f"  {bench(lambda _w: fn_run(), [None] * 4):6.1f} us"
                for which in ("slice", "gate"):
                    one = mhc.standalone(s, bool(post), which)
                    us = bench(
                        lambda _w, one=one: one(
                            *ptrs, stream=torch.cuda.current_stream()
                        ),
                        [None] * 4,
                    )
                    line += f"  {which} {us:5.1f}"
            print(line, flush=True)
    print("PASS" if worst < 1e-2 else "FAIL", f"worst rel {worst:.1e}")


if __name__ == "__main__":
    main()
