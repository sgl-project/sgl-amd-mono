# SPDX-License-Identifier: Apache-2.0
"""Decode conv and gated delta for one Qwen3.5 GDN layer.

The conv window matches causal_conv1d_update (kernel width 4): state columns
are oldest to newest, ``acc = c0*w0 + c1*w1 + c2*w2 + x*w3 + bias``, then
SiLU, then the window shifts so the newest column is ``x``. The recurrence
matches the packed decode rule: per-head L2 norm of q and k, ``q *= K**-0.5``,
``g = -exp(A_log) * softplus(a + dt_bias)``, beta rounded through the
projection dtype, then ``h *= exp(g); v -= h k; v *= beta; h += v k``.
Pad rows (cache index < 0) write nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b_fp8.gfx950.layout import (
    BA_N,
    CONV_KERNEL,
    GDN_HEAD,
    GDN_K_HEADS,
    GDN_V_HEADS,
    QKVZ_N,
    SOFTPLUS_THRESHOLD,
)


@triton.jit
def _scatter_rows(
    dst,
    src,
    idx,
    valid,
    n_cols,
    stride_dst,
    stride_src,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(valid + row) == 0:
        return
    slot = tl.load(idx + row).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    col = 0
    while col < n_cols:
        o = col + offs
        mask = o < n_cols
        v = tl.load(src + row * stride_src + o, mask=mask, other=0)
        tl.store(dst + slot * stride_dst + o, v, mask=mask)
        col += BLOCK


def _zeros_if_pad(gathered: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """A pad row (cache index < 0) reads zeros, not whatever lives in slot 0."""
    mask = valid.view(-1, *([1] * (gathered.ndim - 1))).to(dtype=torch.bool)
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def _scatter(table: torch.Tensor, rows: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor):
    flat_src = rows.reshape(rows.shape[0], -1).contiguous()
    flat_dst = table.view(table.shape[0], -1)
    n_cols = flat_src.shape[1]
    _scatter_rows[(flat_src.shape[0],)](
        flat_dst,
        flat_src,
        idx,
        valid,
        n_cols,
        flat_dst.stride(0),
        flat_src.stride(0),
        BLOCK=1024,
    )


def _split_qkvzba(qkvz: torch.Tensor, ba: torch.Tensor):
    hk, hv, d = GDN_K_HEADS, GDN_V_HEADS, GDN_HEAD
    q_dim = hk * d
    v_dim = hv * d
    if qkvz.shape[-1] != QKVZ_N or ba.shape[-1] != BA_N:
        raise RuntimeError(
            f"GDN projections are {tuple(qkvz.shape)} and {tuple(ba.shape)}, "
            f"TP4 wants [*, {QKVZ_N}] and [*, {BA_N}]"
        )
    q, k, v, z = qkvz.split([q_dim, q_dim, v_dim, v_dim], dim=-1)
    b, a = ba.split([hv, hv], dim=-1)
    s = qkvz.shape[0]
    return (
        q.view(s, hk, d),
        k.view(s, hk, d),
        v.view(s, hv, d),
        z.view(s, hv, d),
        b,
        a,
    )


def _conv_update(x, conv, idx, valid, weight, bias):
    """One decode token. ``conv`` is ``[slots, dim, 3]`` or ``[slots, 3, dim]``."""
    safe = idx.clamp(min=0)
    window_last = conv.shape[-1] == CONV_KERNEL - 1
    gathered = _zeros_if_pad(conv.index_select(0, safe), valid)
    if window_last:
        state = gathered
    else:
        state = gathered.transpose(1, 2).contiguous()
    w = weight.view(weight.shape[0], CONV_KERNEL)
    acc = state[..., 0].float() * w[:, 0] + state[..., 1].float() * w[:, 1]
    acc = acc + state[..., 2].float() * w[:, 2] + x.float() * w[:, 3]
    if bias is not None:
        acc = acc + bias.float()
    y = (acc / (1.0 + torch.exp(-acc))).to(dtype=x.dtype)
    new = torch.stack(
        [state[..., 1], state[..., 2], x.to(dtype=state.dtype)], dim=-1
    ).contiguous()
    if not window_last:
        new = new.transpose(1, 2).contiguous()
    _scatter(conv, new, idx, valid)
    return y


def _gated_delta(q, k, v, a, b, state, idx, valid, a_log, dt_bias):
    s, hk, d = q.shape
    hv = v.shape[1]
    rep = hv // hk
    qf = q.float()
    kf = k.float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    qf = qf * (d**-0.5)
    qf = qf.repeat_interleave(rep, dim=1)
    kf = kf.repeat_interleave(rep, dim=1)
    a_log = a_log.reshape(-1)
    dt_bias = dt_bias.reshape(-1)
    x = a.float() + dt_bias.float()
    softplus = torch.where(
        x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x.clamp(max=SOFTPLUS_THRESHOLD))), x
    )
    g = -torch.exp(a_log.float()) * softplus
    beta = torch.sigmoid(b.float()).to(dtype=b.dtype).float()
    safe = idx.clamp(min=0)
    h = _zeros_if_pad(state.index_select(0, safe), valid).float()
    vf = v.float()
    h = h * torch.exp(g)[:, :, None, None]
    vf = vf - (h * kf[:, :, None, :]).sum(-1)
    vf = vf * beta[:, :, None]
    h = h + vf[:, :, :, None] * kf[:, :, None, :]
    out = (h * qf[:, :, None, :]).sum(-1)
    _scatter(state, h.to(dtype=state.dtype), idx, valid)
    out = torch.where(valid.view(-1, 1, 1).bool(), out, torch.zeros_like(out))
    return out.to(dtype=v.dtype)


def gdn_core(mod, hidden, conv, ssm, idx, rows: int) -> torch.Tensor:
    """Input RMSNorm is the caller's. Returns the local core, ``[rows, V]``."""
    from sgl_amd_mono.qwen35_a17b_fp8.gfx950.gemv import block_fp8_gemv

    x = hidden[:rows]
    qkvz = block_fp8_gemv(x, mod.in_proj_qkvz.weight, mod.in_proj_qkvz.weight_scale_inv)
    ba = F.linear(x, mod.in_proj_ba.weight, getattr(mod.in_proj_ba, "bias", None))
    q, k, v, z, b, a = _split_qkvzba(qkvz, ba)
    mixed = torch.cat([q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1)], -1)
    cache_idx = idx[:rows].to(torch.long)
    valid = (cache_idx >= 0).to(torch.int32)
    mixed = _conv_update(
        mixed, conv, cache_idx, valid, mod.conv1d.weight, mod.conv1d.bias
    )
    q_dim = GDN_K_HEADS * GDN_HEAD
    v_dim = GDN_V_HEADS * GDN_HEAD
    q, k, v = mixed.split([q_dim, q_dim, v_dim], dim=-1)
    q = q.view(rows, GDN_K_HEADS, GDN_HEAD)
    k = k.view(rows, GDN_K_HEADS, GDN_HEAD)
    v = v.view(rows, GDN_V_HEADS, GDN_HEAD)
    core = _gated_delta(
        q, k, v, a, b, ssm, cache_idx, valid, mod.A_log, mod.dt_bias
    )
    flat = core.reshape(-1, GDN_HEAD)
    zflat = z.reshape(-1, GDN_HEAD)
    gated = mod.norm(flat, zflat)
    if isinstance(gated, tuple):
        gated = gated[0]
    return gated.reshape(rows, -1).contiguous()
