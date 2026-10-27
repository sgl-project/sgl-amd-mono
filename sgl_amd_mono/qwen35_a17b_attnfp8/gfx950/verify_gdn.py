# SPDX-License-Identifier: Apache-2.0
"""Verify-step GDN for Qwen3.5-397B AttnFP8 at TP4.

The stock verify layer launches a Gemma norm, a per-token FP8 quant, the
in_proj GEMM, a split, a conv, a compare, the delta rule and a gated norm.
K2 already owns the MoE. This folds the Gemma norm and the quant into one
aiter kernel (the GEMM then reads that FP8 activation), and replaces the
split, conv, delta and gated norm with the draft-chain conv, the delta rule
and the gated RMSNorm. topk is 1, so the chain is linear and the main SSM
slot is left for the commit to fill from the per-token snapshots.

Rounding follows the Triton verify kernels: SiLU in fp32 stored as bf16, the
bf16 SSM state re-rounded between draft tokens, and the gated norm applied
after the delta output is rounded to bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b_attnfp8.gfx950 import layout as L

_LOGGED = False


@triton.jit
def _conv_chain_kernel(
    x_ptr,
    stride_x,
    w_ptr,
    stride_w,
    bias_ptr,
    state_ptr,
    stride_slot,
    stride_dim,
    stride_win,
    slot_ptr,
    inter_ptr,
    stride_i_slot,
    stride_i_step,
    stride_i_dim,
    stride_i_win,
    inter_slot_ptr,
    out_ptr,
    stride_out,
    tokens,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    req = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < C
    slot = tl.load(slot_ptr + req).to(tl.int64)
    live = slot >= 0
    cell = state_ptr + tl.maximum(slot, 0) * stride_slot + cols * stride_dim
    c0 = tl.load(cell, mask=mask & live, other=0.0).to(tl.float32)
    c1 = tl.load(cell + stride_win, mask=mask & live, other=0.0).to(tl.float32)
    c2 = tl.load(cell + 2 * stride_win, mask=mask & live, other=0.0).to(tl.float32)
    w0 = tl.load(w_ptr + cols * stride_w, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + cols * stride_w + 1, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + cols * stride_w + 2, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + cols * stride_w + 3, mask=mask, other=0.0).to(tl.float32)
    bias = tl.zeros((BLOCK,), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    islot = tl.load(inter_slot_ptr + req).to(tl.int64)
    save = live & (islot >= 0)
    ibase = inter_ptr + tl.maximum(islot, 0) * stride_i_slot + cols * stride_i_dim
    for t in range(tokens):
        x = tl.load(
            x_ptr + (req * tokens + t) * stride_x + cols, mask=mask, other=0.0
        ).to(tl.float32)
        acc = c0 * w0 + c1 * w1 + c2 * w2 + x * w3 + bias
        y = acc / (1.0 + tl.exp(-acc))
        tl.store(
            out_ptr + (req * tokens + t) * stride_out + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=mask & live,
        )
        c0 = c1
        c1 = c2
        c2 = x
        tl.store(ibase + t * stride_i_step, c0, mask=mask & save)
        tl.store(ibase + t * stride_i_step + stride_i_win, c1, mask=mask & save)
        tl.store(ibase + t * stride_i_step + 2 * stride_i_win, c2, mask=mask & save)
    tl.store(cell, c0.to(state_ptr.dtype.element_ty), mask=mask & live)
    tl.store(cell + stride_win, c1.to(state_ptr.dtype.element_ty), mask=mask & live)
    tl.store(cell + 2 * stride_win, c2.to(state_ptr.dtype.element_ty), mask=mask & live)


@triton.jit
def _delta_chain_kernel(
    mixed_ptr,
    stride_mixed,
    qkvz_ptr,
    stride_qkvz,
    ba_ptr,
    stride_ba,
    state_ptr,
    stride_s_slot,
    stride_s_head,
    stride_s_v,
    stride_s_k,
    slot_ptr,
    inter_ptr,
    stride_i_slot,
    stride_i_step,
    stride_i_head,
    stride_i_v,
    stride_i_k,
    inter_slot_ptr,
    a_log_ptr,
    dt_bias_ptr,
    sumsq_ptr,
    out_ptr,
    stride_out,
    scale,
    tokens,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    NVT: tl.constexpr,
    THRESHOLD: tl.constexpr,
    ROUND: tl.constexpr,
):
    # One value tile of the draft chain. h is [K, BV], matching the stock
    # verify kernel's tile so the 256 programs stay latency-bound the same way.
    req = tl.program_id(0)
    hv = tl.program_id(1)
    tile = tl.program_id(2)
    kh = hv // (HV // HK)
    k_off = tl.arange(0, D)
    v_off = tile * BV + tl.arange(0, BV)
    slot = tl.load(slot_ptr + req).to(tl.int64)
    live = slot >= 0
    cell = (
        state_ptr
        + tl.maximum(slot, 0) * stride_s_slot
        + hv * stride_s_head
        + k_off[:, None] * stride_s_k
        + v_off[None, :] * stride_s_v
    )
    h = tl.load(cell, mask=live, other=0.0).to(tl.float32)
    islot = tl.load(inter_slot_ptr + req).to(tl.int64)
    save = live & (islot >= 0)
    a_log = tl.load(a_log_ptr + hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + hv).to(tl.float32)
    for t in range(tokens):
        row = req * tokens + t
        base = mixed_ptr + row * stride_mixed
        q = tl.load(base + kh * D + k_off).to(tl.float32)
        k = tl.load(base + HK * D + kh * D + k_off).to(tl.float32)
        v = tl.load(base + 2 * HK * D + hv * D + v_off).to(tl.float32)
        b = tl.load(ba_ptr + row * stride_ba + hv).to(tl.float32)
        a = tl.load(ba_ptr + row * stride_ba + HV + hv).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6)
        q = q * scale
        x = a + dt_bias
        sp = tl.where(x <= THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g = -tl.exp(a_log) * sp
        beta = 1.0 / (1.0 + tl.exp(-b))
        h = h * tl.exp(g)
        v = (v - tl.sum(h * k[:, None], axis=0)) * beta
        h = h + k[:, None] * v[None, :]
        o = tl.sum(h * q[:, None], axis=0)
        if save:
            iptr = (
                inter_ptr
                + islot * stride_i_slot
                + t * stride_i_step
                + hv * stride_i_head
                + k_off[:, None] * stride_i_k
                + v_off[None, :] * stride_i_v
            )
            tl.store(iptr, h.to(inter_ptr.dtype.element_ty))
        if ROUND:
            h = h.to(tl.bfloat16).to(tl.float32)
        o = o.to(tl.bfloat16).to(tl.float32)
        tl.store(sumsq_ptr + (row * HV + hv) * NVT + tile, tl.sum(o * o, axis=0))
        tl.store(
            out_ptr + row * stride_out + hv * D + v_off,
            tl.where(live, o, 0.0).to(out_ptr.dtype.element_ty),
        )


@triton.jit
def _norm_kernel(
    out_ptr,
    stride_out,
    qkvz_ptr,
    stride_qkvz,
    sumsq_ptr,
    norm_w_ptr,
    eps,
    HV: tl.constexpr,
    D: tl.constexpr,
    NVT: tl.constexpr,
):
    row = tl.program_id(0)
    hv = tl.program_id(1)
    offs = tl.arange(0, D)
    ss = tl.zeros((), dtype=tl.float32)
    base = (row * HV + hv) * NVT
    for tile in tl.static_range(NVT):
        ss += tl.load(sumsq_ptr + base + tile)
    rstd = tl.rsqrt(ss / D + eps)
    norm_w = tl.load(norm_w_ptr + offs).to(tl.float32)
    o = tl.load(out_ptr + row * stride_out + hv * D + offs).to(tl.float32)
    z = tl.load(qkvz_ptr + row * stride_qkvz + (2 * 4 * D + HV * D) + hv * D + offs).to(
        tl.float32
    )
    y = o * rstd * norm_w * (z * tl.sigmoid(z))
    tl.store(out_ptr + row * stride_out + hv * D + offs, y.to(out_ptr.dtype.element_ty))


def _slots(forward_batch):
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    lin = getattr(backend, "linear_attn_backend", backend)
    md = lin.forward_metadata
    cache = md.mamba_cache_indices
    nreq = md.query_start_loc.shape[0] - 1
    inter = select_verify_intermediate_state_indices(
        lin.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        cache[:nreq] >= 0,
        lin.req_to_token_pool.size,
    )
    return lin, cache, inter


def _gemma_fp8(layer, hidden, residual):
    """Gemma add + RMSNorm + per-token FP8 quant, one launch.

    ``residual is None`` on the first layer: the residual K2 adds is ``hidden``
    itself, matching ``GemmaRMSNorm`` without a residual. The scale is ``[rows, 1]``,
    which is what ``apply_fp8_linear`` consumes for a pre-quantized activation.
    """
    from aiter.ops.rmsnorm import add_rmsnorm_quant, rmsnorm_quant
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    norm = layer.input_layernorm
    rows, hidden_size = hidden.shape
    quant = torch.empty((rows, hidden_size), dtype=fp8_dtype, device=hidden.device)
    scale = torch.empty(rows, dtype=torch.float32, device=hidden.device)
    if residual is None:
        rmsnorm_quant(
            quant,
            hidden,
            scale,
            norm.weight,
            norm.variance_epsilon,
            0,
            False,
            True,
        )
        return quant, scale.unsqueeze(1), hidden
    residual_out = torch.empty_like(hidden)
    add_rmsnorm_quant(
        quant,
        hidden,
        residual,
        residual_out,
        scale,
        norm.weight,
        norm.variance_epsilon,
        0,
        False,
        True,
    )
    return quant, scale.unsqueeze(1), residual_out


def _proj_prequant(mod, quant, scale):
    """Packed in_proj on a pre-quantized activation. ``None`` when the GEMM cannot."""
    weight = getattr(mod, "_fused_in_proj_weight", None)
    weight_scale = getattr(mod, "_fused_in_proj_scale", None)
    if weight is None or weight_scale is None:
        return None
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

    fused = apply_fp8_linear(
        quant,
        weight.t(),
        weight_scale,
        input_scale=scale,
        use_per_token_if_dynamic=True,
    )
    split = mod._fused_in_proj_qkvz_width
    width = mod._fused_in_proj_ba_width
    return fused[:, :split], fused[:, split : split + width]


def verify_layer(layer, hidden, residual, forward_batch):
    """``(core, residual)`` for one verify GDN layer. ``out_proj`` is K2's.

    Falls back to SGLang's GDN when the draft chain is not 4 tokens or the
    packed FP8 projection is absent.
    """
    mod = layer.linear_attn
    rows = hidden.shape[0]
    tokens = forward_batch.spec_info.draft_token_num
    if tokens == 4 and rows % tokens == 0:
        quant, scale, residual_out = _gemma_fp8(layer, hidden, residual)
        projected = _proj_prequant(mod, quant, scale)
        if projected is not None:
            core = _chain(mod, projected[0], projected[1], forward_batch, rows, tokens)
            return core, residual_out
    if residual is None:
        residual_out, x = hidden, layer.input_layernorm(hidden)
    else:
        x, residual_out = layer.input_layernorm(hidden, residual)
    with _identity_out_proj(mod):
        return mod(x, forward_batch).contiguous(), residual_out


def _chain(mod, qkvz, ba, forward_batch, rows, tokens) -> torch.Tensor:
    """Core output ``[rows, 2048]`` from the packed projection."""
    global _LOGGED
    batch = rows // tokens
    lin, cache, inter = _slots(forward_batch)
    if qkvz.shape != (rows, L.QKVZ) or ba.shape[-1] != L.BA:
        raise RuntimeError(
            f"verify GDN projections are {tuple(qkvz.shape)} and {tuple(ba.shape)}, "
            f"TP4 wants [{rows}, {L.QKVZ}] and [*, {L.BA}]"
        )
    st = lin.req_to_token_pool.mamba2_layer_cache(mod.layer_id)
    conv = st.conv[0]
    ssm = st.temporal
    inter_conv = st.intermediate_conv_window[0]
    inter_ssm = st.intermediate_ssm
    weight = mod.conv1d.weight.reshape(L.CONV, L.CONV_W)
    bias = mod.conv1d.bias
    if not _LOGGED and not torch.cuda.is_current_stream_capturing():
        _LOGGED = True
        print(
            "verify GDN fused: "
            f"conv {tuple(conv.shape)} ssm {tuple(ssm.shape)} "
            f"inter_conv {tuple(inter_conv.shape)} inter_ssm {tuple(inter_ssm.shape)}",
            flush=True,
        )
    mixed = torch.empty(rows, L.CONV, dtype=torch.bfloat16, device=qkvz.device)
    _conv_chain_kernel[(batch, triton.cdiv(L.CONV, 256))](
        qkvz,
        qkvz.stride(0),
        weight,
        weight.stride(0),
        weight if bias is None else bias,
        conv,
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        cache,
        inter_conv,
        inter_conv.stride(0),
        inter_conv.stride(1),
        inter_conv.stride(2),
        inter_conv.stride(3),
        inter,
        mixed,
        mixed.stride(0),
        tokens,
        C=L.CONV,
        BLOCK=256,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    core = torch.empty(rows, L.CORE, dtype=torch.bfloat16, device=qkvz.device)
    sumsq = torch.empty(rows, L.NV, L.HD // 16, dtype=torch.float32, device=qkvz.device)
    _delta_chain_kernel[(batch, L.NV, L.HD // 16)](
        mixed,
        mixed.stride(0),
        qkvz,
        qkvz.stride(0),
        ba,
        ba.stride(0),
        ssm,
        ssm.stride(0),
        ssm.stride(1),
        ssm.stride(2),
        ssm.stride(3),
        cache,
        inter_ssm,
        inter_ssm.stride(0),
        inter_ssm.stride(1),
        inter_ssm.stride(2),
        inter_ssm.stride(3),
        inter_ssm.stride(4),
        inter,
        mod.A_log.reshape(-1),
        mod.dt_bias.reshape(-1),
        sumsq,
        core,
        core.stride(0),
        L.HD**-0.5,
        tokens,
        HK=L.NK,
        HV=L.NV,
        D=L.HD,
        BV=16,
        NVT=L.HD // 16,
        THRESHOLD=20.0,
        ROUND=ssm.dtype == torch.bfloat16,
        num_warps=2,
    )
    _norm_kernel[(rows, L.NV)](
        core,
        core.stride(0),
        qkvz,
        qkvz.stride(0),
        sumsq,
        mod.norm.weight.reshape(-1),
        mod.norm.eps,
        HV=L.NV,
        D=L.HD,
        NVT=L.HD // 16,
        num_warps=4,
    )
    return core


def _identity_out_proj(mod):
    from sgl_amd_mono.qwen35_a17b_attnfp8.spec import _proj_input

    return _proj_input(mod, "out_proj")
