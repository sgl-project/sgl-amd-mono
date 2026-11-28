# SPDX-License-Identifier: Apache-2.0
"""Check and time wide MLA projection-front launch (``gfx950/mla_pre_wide.py``) on 1 MI355X.

Checks all 8 outputs (q, latent, k_nope, q_lora, gate, idx_q, idx_k, idx_w) against
the f32 reference across repeated launches (verifying fbuf1/fbuf2 zeroing & tag
advancement), and times cold-L2 CUDA-graph execution against the unfused bf16 GEMM +
RMSNorm sequence for both DCP1 (qb_rows=2304) and DCP8 (qb_rows=18432).

    SGLANG_USE_AITER=1 HIP_VISIBLE_DEVICES=0 python3 test/kernels/kimi_k31/mla_pre_wide_bench.py \
        --rows 16,30,48,64 --qb-rows 2304,18432 --timeline
"""

import argparse

import torch
import torch.nn.functional as F
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mpw

BF = torch.bfloat16
Q_EPS = 1e-6
KV_EPS = 1e-6


def rel(a, b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return float("inf")
    val = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).item()
    return val if val == val else float("inf")


def print_timeline(lay, scratch, m, qb_rows):
    o = lay["offs"]["tl"]
    t = (
        scratch[o : o + BLOCKS * mpw.TL_POINTS * 8]
        .view(torch.int64)
        .view(BLOCKS, mpw.TL_POINTS)
        .cpu()
    )
    t0 = t[:, 0][t[:, 0] > 0].min()
    t = torch.where(t > 0, t - t0, torch.full_like(t, -1))
    print(
        f"  timeline m={m} qb={qb_rows}: mark  CTAs  first  median  last (us)",
        flush=True,
    )
    for k, name in enumerate(mpw.MARKS):
        v = t[:, k]
        v = v[v >= 0].double() / 100.0
        if v.numel():
            print(
                f"    {k:2d} {name:8s} {v.numel():5d} {v.min():7.1f} {v.median():7.1f} {v.max():7.1f}",
                flush=True,
            )


def time_graph(fn, nl, flush_mb=256, reps=10):
    flush = (
        torch.empty(flush_mb << 20, dtype=torch.uint8, device="cuda")
        if flush_mb
        else None
    )
    for i in range(nl):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(nl):
            if flush is not None:
                flush.add_(1)
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    # subtract flush time
    flush_us = 0.0
    if flush is not None:
        gf = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gf):
            for _ in range(nl):
                flush.add_(1)
        for _ in range(3):
            gf.replay()
        torch.cuda.synchronize()
        f0 = torch.cuda.Event(enable_timing=True)
        f1 = torch.cuda.Event(enable_timing=True)
        f0.record()
        for _ in range(reps):
            gf.replay()
        f1.record()
        torch.cuda.synchronize()
        flush_us = f0.elapsed_time(f1) * 1000.0 / (reps * nl)
    return g, flush, e0.elapsed_time(e1) * 1000.0 / (reps * nl) - flush_us


def run_one(m, qb_rows, nl, flush_mb, show_tl, check_only=False):
    dev = torch.device("cuda:0")
    launch, lay = mpw.mla_pre_wide(
        m, qb_rows=qb_rows, launches=nl, q_eps=Q_EPS, kv_eps=KV_EPS, timeline=show_tl
    )
    torch.manual_seed(42 + m + qb_rows)

    x = [torch.randn(m, mpw.H, dtype=BF, device=dev) * 0.5 for _ in range(nl)]
    w_qkv = [
        torch.randn(mpw.LATW, mpw.H, dtype=BF, device=dev) / (mpw.H**0.5)
        for _ in range(nl)
    ]
    w_g = [
        torch.randn(mpw.GATE, mpw.H, dtype=BF, device=dev) / (mpw.H**0.5)
        for _ in range(nl)
    ]
    w_k = [
        torch.randn(mpw.IK, mpw.H, dtype=BF, device=dev) / (mpw.H**0.5)
        for _ in range(nl)
    ]
    w_p = [
        torch.randn(mpw.IW, mpw.H, dtype=BF, device=dev) / (mpw.H**0.5)
        for _ in range(nl)
    ]
    q_nw = [torch.randn(mpw.QL, dtype=BF, device=dev) * 0.2 + 1.0 for _ in range(nl)]
    kv_nw = [torch.randn(mpw.KV, dtype=BF, device=dev) * 0.2 + 1.0 for _ in range(nl)]
    w_iq = [
        torch.randn(mpw.IQ, mpw.QL, dtype=BF, device=dev) / (mpw.QL**0.5)
        for _ in range(nl)
    ]
    w_qb = [
        torch.randn(qb_rows, mpw.QL, dtype=BF, device=dev) / (mpw.QL**0.5)
        for _ in range(nl)
    ]

    q_out = [torch.zeros(m, qb_rows, dtype=BF, device=dev) for _ in range(nl)]
    lat_out = [
        torch.zeros(m, mpw.KV + mpw.ROPE, dtype=BF, device=dev) for _ in range(nl)
    ]
    kn_out = [torch.zeros(m, mpw.KV, dtype=BF, device=dev) for _ in range(nl)]
    ql_out = [torch.zeros(m, mpw.QL, dtype=BF, device=dev) for _ in range(nl)]
    g_out = [torch.zeros(m, mpw.GATE, dtype=BF, device=dev) for _ in range(nl)]
    iq_out = [torch.zeros(m, mpw.IQ, dtype=BF, device=dev) for _ in range(nl)]
    ik_out = [torch.zeros(m, mpw.IK, dtype=BF, device=dev) for _ in range(nl)]
    iw_out = [torch.zeros(m, mpw.IW, dtype=BF, device=dev) for _ in range(nl)]

    cnt = torch.zeros(mpw.counters(qb_rows), dtype=torch.int32, device=dev)
    from sgl_amd_mono.common.peer import PeerBuffer

    scr_pb = PeerBuffer(lay["nbytes"], None, 0, 1, dev)
    scratch = scr_pb.bytes
    scratch.zero_()
    from sgl_amd_mono.common import guard

    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(60.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def call_fused(i):
        if i == 0:
            epoch[0].add_(1)
        _run_compiled(
            launch,
            x[i].data_ptr(),
            w_qkv[i].data_ptr(),
            w_g[i].data_ptr(),
            w_k[i].data_ptr(),
            w_p[i].data_ptr(),
            q_nw[i].data_ptr(),
            kv_nw[i].data_ptr(),
            w_iq[i].data_ptr(),
            w_qb[i].data_ptr(),
            q_out[i].data_ptr(),
            lat_out[i].data_ptr(),
            kn_out[i].data_ptr(),
            ql_out[i].data_ptr(),
            g_out[i].data_ptr(),
            iq_out[i].data_ptr(),
            ik_out[i].data_ptr(),
            iw_out[i].data_ptr(),
            cnt.data_ptr(),
            scratch.data_ptr(),
            epoch.data_ptr(),
            i,
            torch.cuda.current_stream(),
        )

    w_kg = [torch.cat([wk, wp], dim=0).contiguous() for wk, wp in zip(w_k, w_p)]
    try:
        from sglang.srt.layers.quantization.unquant_hip import a16w16_linear as _lin
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
            fused_qk_rmsnorm_bf16 as _qk_norm,
        )

        unfused_tag = "sglang_aiter"
    except Exception:
        _lin, _qk_norm, unfused_tag = F.linear, None, "torch_fallback"

    def call_unfused(i):
        qkv = _lin(x[i], w_qkv[i])
        q_l, kv_l = qkv[:, : mpw.QL], qkv[:, mpw.QL :]
        if _qk_norm is not None:
            ql, kn = _qk_norm(q_l, q_nw[i], Q_EPS, kv_l[:, : mpw.KV], kv_nw[i], KV_EPS)
        else:
            ql = F.rms_norm(q_l, (mpw.QL,), q_nw[i], Q_EPS)
            kn = F.rms_norm(kv_l[:, : mpw.KV], (mpw.KV,), kv_nw[i], KV_EPS)
        _ = _lin(x[i], w_g[i])
        _ = _lin(x[i], w_kg[i])
        _ = _lin(ql, w_iq[i])
        _ = _lin(ql, w_qb[i])
        return kn

    def check_outputs():
        errs = {}
        for i in range(nl):
            qkv_r = (x[i].float() @ w_qkv[i].float().T).to(BF)
            q_lat_r = qkv_r[:, : mpw.QL]
            lat_r = qkv_r[:, mpw.QL :]
            ql_r = F.rms_norm(q_lat_r.float(), (mpw.QL,), q_nw[i].float(), Q_EPS).to(BF)
            kn_r = F.rms_norm(
                lat_r[:, : mpw.KV].float(), (mpw.KV,), kv_nw[i].float(), KV_EPS
            ).to(BF)
            g_r = (x[i].float() @ w_g[i].float().T).to(BF)
            ik_r = (x[i].float() @ w_k[i].float().T).to(BF)
            iw_r = (x[i].float() @ w_p[i].float().T).to(BF)
            iq_r = (ql_r.float() @ w_iq[i].float().T).to(BF)
            qb_r = (ql_r.float() @ w_qb[i].float().T).to(BF)
            e = dict(
                q=rel(q_out[i], qb_r),
                lat=rel(lat_out[i], lat_r),
                kn=rel(kn_out[i], kn_r),
                ql=rel(ql_out[i], ql_r),
                g=rel(g_out[i], g_r),
                iq=rel(iq_out[i], iq_r),
                ik=rel(ik_out[i], ik_r),
                iw=rel(iw_out[i], iw_r),
            )
            for k, v in e.items():
                prev = errs.get(k, 0.0)
                errs[k] = float("inf") if (v != v or prev != prev) else max(prev, v)
        return errs

    # 1. Check 4 eager steps, each on new inputs (tests fbuf zeroing + tag advance): a step
    #    that skips a write keeps the previous step's outputs and fails its check
    errs = {}
    for step in range(4):
        if step:
            torch.manual_seed(7000 + m * 31 + qb_rows + step)
            for i in range(nl):
                x[i].copy_(torch.randn(m, mpw.H, dtype=BF, device=dev) * 0.5)
        for i in range(nl):
            call_fused(i)
        torch.cuda.synchronize()
        for k, v in check_outputs().items():
            prev = errs.get(k, 0.0)
            errs[k] = float("inf") if (v != v or prev != prev) else max(prev, v)

    # 2. Capture CUDA graph (and time unless check_only), then replay with fresh inputs
    if check_only:
        g_fused = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_fused):
            for i in range(nl):
                call_fused(i)
        _flush_fused = _flush_unfused = None
        us_fused = us_unfused = 0.0
    else:
        g_fused, _flush_fused, us_fused = time_graph(call_fused, nl, flush_mb=flush_mb)
        _, _flush_unfused, us_unfused = time_graph(call_unfused, nl, flush_mb=flush_mb)
    torch.manual_seed(9000 + m * 31 + qb_rows)
    for i in range(nl):
        x[i].copy_(torch.randn(m, mpw.H, dtype=BF, device=dev) * 0.5)
    g_fused.replay()
    torch.cuda.synchronize()
    for k, v in check_outputs().items():
        prev = errs.get(k, 0.0)
        errs[k] = float("inf") if (v != v or prev != prev) else max(prev, v)
    ok = all(v == v and v < 2e-2 for v in errs.values())
    if not check_only:
        print(
            f"mla_pre_wide m={m:2d} qb={qb_rows:5d} | fused={us_fused:5.1f}us vs {unfused_tag}={us_unfused:5.1f}us "
            f"(save={us_unfused - us_fused:+5.1f}us, 24L={(us_unfused - us_fused) * 24 / 1000:+.2f}ms) | "
            f"max_err={max(errs.values()):.2e} ({'PASS' if ok else 'FAIL'})",
            flush=True,
        )
        if show_tl:
            print_timeline(lay, scratch, m, qb_rows)
    del _flush_fused, _flush_unfused
    scr_pb.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16,30,48,64")
    ap.add_argument("--qb-rows", default="2304,18432")
    ap.add_argument("--launches", type=int, default=4)
    ap.add_argument("--flush", type=int, default=256)
    ap.add_argument("--timeline", action="store_true")
    args = ap.parse_args()
    all_ok = True
    for qb in (int(v) for v in args.qb_rows.split(",")):
        for m in (int(v) for v in args.rows.split(",")):
            all_ok &= run_one(m, qb, args.launches, args.flush, args.timeline)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
