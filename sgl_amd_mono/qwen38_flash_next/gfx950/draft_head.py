# SPDX-License-Identifier: Apache-2.0
"""The MTP draft's vocab projection with an FP8 copy of the lm_head (W8A16).

The draft's logits feed only the top-1 draft proposal (or, with rejection
sampling, the draft distribution, which only has to match what the draft
proposed), so the draft can read an e4m3 copy of the lm_head with one scale a
vocab row: half the bytes of the bf16 GEMM each draft step reads (0.64 GB, not
1.27 GB). The hidden states stay bf16; the weight is dequantized in registers and
accumulated in fp32, the scale applied once a row. The target verify keeps the
bf16 lm_head.

``gemv`` writes the full [rows, vocab] logits in the dtype SGLang's lm_head path
returns, so everything after the projection (SGLang's top-1 kernels, NaN checks,
draft probabilities) is unchanged. ``gemv_argmax`` is the fused variant that
returns the top-1 ids without writing the logits (micro-benchmarks only: SGLang
takes the argmax of the logits itself).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


def quantize_rows(weight: torch.Tensor, chunk: int = 16384):
    """e4m3 copy of a bf16 [N, K] weight with one fp32 scale per row."""
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    s = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for i in range(0, n, chunk):
        w = weight[i : i + chunk].float()
        amax = w.abs().amax(dim=1).clamp_min(1e-12)
        scale = amax / FP8_MAX
        q[i : i + chunk] = (
            (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )
        s[i : i + chunk] = scale
    return q, s


@triton.jit
def _gemv_tile(
    X,
    W,
    S,
    sx,
    M,
    N,
    K: tl.constexpr,
    pid,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    nmask = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BK):
        # the weight is read once: stream it past the caches
        w = tl.load(
            W + rn[:, None] * K + (k0 + rk)[None, :],
            mask=nmask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        x = tl.load(
            X + rm[:, None] * sx + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0
        )
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    scale = tl.load(S + rn, mask=nmask, other=0.0)
    return acc * scale[None, :], rm, rn, nmask


@triton.jit
def _gemv_kernel(
    X,
    W,
    S,
    O,
    sx,
    so,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    acc, rm, rn, nmask = _gemv_tile(X, W, S, sx, M, N, K, pid, BM, BN, BK)
    tl.store(
        O + rm[:, None] * so + rn[None, :],
        acc.to(O.dtype.element_ty),
        mask=(rm[:, None] < M) & nmask[None, :],
    )


@triton.jit
def _gemv_argmax_kernel(
    X,
    W,
    S,
    PV,
    PI,
    sx,
    M,
    N,
    K: tl.constexpr,
    NT,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    acc, rm, rn, nmask = _gemv_tile(X, W, S, sx, M, N, K, pid, BM, BN, BK)
    acc = tl.where(nmask[None, :], acc, -float("inf"))
    v = tl.max(acc, axis=1)
    i = tl.argmax(acc, axis=1) + pid * BN
    tl.store(PV + rm * NT + pid, v, mask=rm < M)
    tl.store(PI + rm * NT + pid, i, mask=rm < M)


@triton.jit
def _argmax_finalize_kernel(PV, PI, OUT, NT, BT: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, BT)
    v = tl.load(PV + row * NT + t, mask=t < NT, other=-float("inf"))
    j = tl.argmax(v, axis=0)
    tl.store(OUT + row, tl.load(PI + row * NT + j).to(tl.int64))


def _config(m: int):
    bm = max(16, triton.next_power_of_2(m))
    # (BM, BN, BK, warps): tuned on MI355X for K = 2560, N = 248320
    if m <= 4:
        return bm, 64, 256, 2
    if m <= 16:
        return bm, 64, 256, 4
    return bm, 128, 256, 4


def gemv(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    out_dtype=torch.bfloat16,
    out: torch.Tensor | None = None,
    cfg=None,
):
    m, k = x.shape
    n = w_q.shape[0]
    assert x.stride(1) == 1 and w_q.is_contiguous() and k == w_q.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m == 0:
        return out
    bm, bn, bk, nw = cfg or _config(m)
    assert k % bk == 0
    if m > bm:  # rows beyond one tile: one launch a tile of rows
        for i in range(0, m, bm):
            gemv(x[i : i + bm], w_q, w_s, out=out[i : i + bm], cfg=cfg)
        return out
    _gemv_kernel[(triton.cdiv(n, bn),)](
        x,
        w_q,
        w_s,
        out,
        x.stride(0),
        out.stride(0),
        m,
        n,
        k,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=nw,
    )
    return out


def gemv_argmax(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, cfg=None):
    m, k = x.shape
    n = w_q.shape[0]
    bm, bn, bk, nw = cfg or _config(m)
    assert m <= bm and k % bk == 0
    nt = triton.cdiv(n, bn)
    pv = torch.empty((m, nt), dtype=torch.float32, device=x.device)
    pi = torch.empty((m, nt), dtype=torch.int32, device=x.device)
    ids = torch.empty((m,), dtype=torch.int64, device=x.device)
    _gemv_argmax_kernel[(nt,)](
        x,
        w_q,
        w_s,
        pv,
        pi,
        x.stride(0),
        m,
        n,
        k,
        nt,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=nw,
    )
    _argmax_finalize_kernel[(m,)](
        pv, pi, ids, nt, BT=triton.next_power_of_2(nt), num_warps=8
    )
    return ids
