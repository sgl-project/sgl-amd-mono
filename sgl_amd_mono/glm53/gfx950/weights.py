# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/weights.py
"""GLM-5.3's layer tensors as the kernel reads them, from SGLang's loaded modules
(``Glm53Weights``). Projections and experts are SGLang's FP8 tensors in place;
W_UK and W_UV, the router, the indexer's weights_proj and its k LayerNorm are
copies, recorded in ``copies`` for the startup log."""

from __future__ import annotations

from typing import Optional

import torch

from .config import GLM5_GLOBAL_HEADS, HIDDEN, KV_LORA, NOPE_DIM, PE_DIM, Q_LORA, V_DIM
from .layout import INDEX_DIM, INDEX_HEADS
from .packing import pack_bf16, pack_fp8

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj",
        "o_proj",
        "w_kc",
        "w_vc",
        "w_scale",
        "indexer",
        "rotary_emb",
        "gate",
        "e_score_correction_bias",
        "experts",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": (
        "wq_b",
        "wk",
        "weights_proj",
        "k_norm",
        "rotary_emb",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache",),
    "srt/layers/quantization/fp8.py": (
        "weight_scale_inv",
        "aiter_bpreshuffled",
        "is_shuffled",
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
        "runner",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("quant_method",),
    "srt/layers/moe/moe_runner/runner.py": ("runner_backend",),
    "srt/layers/moe/utils.py": ("is_aiter",),
}

N_EXPERTS = 256
# SGLang appends the shared expert to the routed ones
PHYSICAL_EXPERTS = N_EXPERTS + 1
QKV_A = Q_LORA + KV_LORA + PE_DIM


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _need(name: str, t: torch.Tensor, shape: tuple, dtypes: tuple) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous {'/'.join(map(str, dtypes))} {shape}, got "
            f"{t.dtype} {tuple(t.shape)} {tuple(t.stride())}"
        )
    return t


def _fp8(name: str, weight, scale, rows: int, cols: int):
    """An FP8 E4M3 [rows, cols] weight with its f32 128 x 128 block scales."""
    _need(f"{name} weight", weight, (rows, cols), (torch.float8_e4m3fn,))
    if scale is None:
        raise ValueError(f"needs {name} as block FP8; it has no weight_scale_inv")
    _need(
        f"{name} scale", scale, (_cdiv(rows, 128), _cdiv(cols, 128)), (torch.float32,)
    )
    return weight, scale


def _shuffled(linear) -> bool:
    # SGLang sets these only on a linear it rewrote into AITER's (16, 16) layout
    return bool(
        getattr(linear, "aiter_bpreshuffled", False)
        or getattr(linear.weight, "is_shuffled", False)
    )


def _linear(name: str, linear, rows: int, cols: int):
    """A block-FP8 linear as SGLang keeps it after loading: (weight, scales,
    shuffled), shuffled meaning AITER's (16, 16) layout in place."""
    w, sc = _fp8(
        name, linear.weight, getattr(linear, "weight_scale_inv", None), rows, cols
    )
    return w, sc, _shuffled(linear)


def unshuffled_projections(layer) -> list[str]:
    """The layer's FP8 projections that SGLang left row-major: the kernel reads
    every one in place in AITER's (16, 16) layout."""
    attn = layer.self_attn
    linears = {
        "fused_qkv_a_proj_with_mqa": attn.fused_qkv_a_proj_with_mqa,
        "q_b_proj": attn.q_b_proj,
        "o_proj": attn.o_proj,
    }
    if attn.indexer is not None:
        linears["indexer.wq_b"] = attn.indexer.wq_b
        linears["indexer.wk"] = attn.indexer.wk
    return [name for name, linear in linears.items() if not _shuffled(linear)]


def experts_reason(moe) -> Optional[str]:
    """Why the kernel cannot read this MoE's experts in place, or None."""
    experts = getattr(moe, "experts", None)
    w13 = getattr(experts, "w13_weight", None)
    if w13 is None:
        return "no fused experts (w13_weight)"
    if w13.dtype != torch.float8_e4m3fn:
        return f"needs block-FP8 experts, got {w13.dtype}"
    if w13.shape[0] != PHYSICAL_EXPERTS:
        return (
            f"needs the shared expert fused as expert {N_EXPERTS} "
            f"({PHYSICAL_EXPERTS} experts), got {w13.shape[0]}"
        )
    # SGLang shuffles FP8 experts to (16, 16) only for its AITER MoE runner
    runner = getattr(experts.quant_method, "runner", None)
    if runner is None or not runner.runner_backend.is_aiter():
        return "needs the experts on SGLang's AITER MoE runner, which shuffles them"
    return None


def _uk_uv(attn, heads: int, idx: int, copies: dict):
    """W_UK [H * 512, 192] and W_UV [H * 256, 512] packed from SGLang's per-tensor
    FP8 ``w_kc`` / ``w_vc``, every block scale ``w_scale``: the values SGLang's
    own absorbed attention multiplies by."""
    w_kc, w_vc, w_scale = attn.w_kc, attn.w_vc, getattr(attn, "w_scale", None)
    if w_kc.dtype != torch.float8_e4m3fn or w_scale is None or w_scale.numel() != 1:
        raise ValueError(
            f"needs w_kc / w_vc as per-tensor FP8 E4M3 with a scalar w_scale, got "
            f"{w_kc.dtype} and {None if w_scale is None else tuple(w_scale.shape)}"
        )
    if tuple(w_kc.shape) != (heads, NOPE_DIM, KV_LORA) or tuple(w_vc.shape) != (
        heads,
        KV_LORA,
        V_DIM,
    ):
        raise ValueError(
            f"needs w_kc [{heads}, {NOPE_DIM}, {KV_LORA}] and w_vc [{heads}, "
            f"{KV_LORA}, {V_DIM}], got {tuple(w_kc.shape)} and {tuple(w_vc.shape)}"
        )
    uk = w_kc.transpose(1, 2).reshape(heads * KV_LORA, NOPE_DIM)
    uv = w_vc.transpose(1, 2).reshape(heads * V_DIM, KV_LORA)
    scale = float(w_scale.float().item())
    dev = uk.device
    w_uk, w_uv = pack_fp8(uk.contiguous()), pack_fp8(uv.contiguous())
    s_uk = torch.full(
        (heads * KV_LORA // 128, NOPE_DIM // 64), scale, dtype=torch.float32, device=dev
    )
    s_uv = torch.full(
        (heads * V_DIM // 128, KV_LORA // 128), scale, dtype=torch.float32, device=dev
    )
    for name, t in (("w_uk", w_uk), ("w_uv", w_uv), ("s_uk", s_uk), ("s_uv", s_uv)):
        copies["W_UK / W_UV packed", idx, name] = t
    return w_uk, s_uk, w_uv, s_uv


def _indexer(attn, idx: int, copies: dict):
    """An indexer layer's pointer table (wk, its scales, weights_proj packed, wq_b,
    its scales, k LayerNorm weight and bias in f32, no timeline) and what it
    points at."""
    ind = attn.indexer
    w_k, s_k, sh_k = _linear("indexer.wk", ind.wk, INDEX_DIM, HIDDEN)
    w_q, s_q, sh_q = _linear("indexer.wq_b", ind.wq_b, INDEX_HEADS * INDEX_DIM, Q_LORA)
    w_w = _need(
        "indexer.weights_proj",
        ind.weights_proj.weight,
        (INDEX_HEADS, HIDDEN),
        (torch.bfloat16,),
    )
    w_w = pack_bf16(w_w)
    ln = ind.k_norm
    g_k = ln.weight.detach().float().contiguous()
    b_k = ln.bias.detach().float().contiguous()
    copies["indexer weights_proj packed", idx] = w_w
    copies["indexer k LayerNorm f32", idx, "weight"] = g_k
    copies["indexer k LayerNorm f32", idx, "bias"] = b_k
    keep = (w_k, s_k, w_w, w_q, s_q, g_k, b_k)
    table = torch.tensor(
        [t.data_ptr() for t in keep] + [0], dtype=torch.int64, device=w_k.device
    )
    return table, keep, (sh_k, sh_q)


def layer_weights(layer, idx: int, tp: int, copies: dict):
    """A covered layer's tensors (``Glm53Weights``)."""
    from .runner import Glm53Weights

    attn, moe = layer.self_attn, layer.mlp
    heads = GLM5_GLOBAL_HEADS // tp
    bf16 = (torch.bfloat16,)
    g_in = _need("input_layernorm", layer.input_layernorm.weight, (HIDDEN,), bf16)
    g_post = _need(
        "post_attention_layernorm",
        layer.post_attention_layernorm.weight,
        (HIDDEN,),
        bf16,
    )
    g_q = _need("q_a_layernorm", attn.q_a_layernorm.weight, (Q_LORA,), bf16)
    g_kv = _need("kv_a_layernorm", attn.kv_a_layernorm.weight, (KV_LORA,), bf16)
    w_qkv, s_qkv, sh_qkv = _linear(
        "fused_qkv_a_proj_with_mqa", attn.fused_qkv_a_proj_with_mqa, QKV_A, HIDDEN
    )
    w_qb, s_qb, sh_qb = _linear(
        "q_b_proj", attn.q_b_proj, heads * (NOPE_DIM + PE_DIM), Q_LORA
    )
    w_o, s_o, sh_o = _linear("o_proj", attn.o_proj, HIDDEN, heads * V_DIM)
    w_uk, s_uk, w_uv, s_uv = _uk_uv(attn, heads, idx, copies)
    why = experts_reason(moe)
    if why is not None:
        raise ValueError(why)
    experts = moe.experts
    inter = experts.w13_weight.shape[1] // 2
    w_ug = _need(
        "w13_weight",
        experts.w13_weight,
        (PHYSICAL_EXPERTS, 2 * inter, HIDDEN),
        (torch.float8_e4m3fn,),
    )
    s_ug = _need(
        "w13_weight_scale_inv",
        experts.w13_weight_scale_inv,
        (PHYSICAL_EXPERTS, 2 * inter // 128, HIDDEN // 128),
        (torch.float32,),
    )
    w_dn = _need(
        "w2_weight",
        experts.w2_weight,
        (PHYSICAL_EXPERTS, HIDDEN, inter),
        (torch.float8_e4m3fn,),
    )
    s_dn = _need(
        "w2_weight_scale_inv",
        experts.w2_weight_scale_inv,
        (PHYSICAL_EXPERTS, HIDDEN // 128, inter // 128),
        (torch.float32,),
    )
    gate = moe.gate
    w_r = pack_bf16(
        _need("gate", gate.weight, (N_EXPERTS, HIDDEN), (torch.bfloat16, torch.float32))
        .to(torch.bfloat16)
        .contiguous()
    )
    copies["router packed", idx] = w_r
    bias = _need(
        "e_score_correction_bias",
        gate.e_score_correction_bias,
        (N_EXPERTS,),
        (torch.float32,),
    )
    index, keep, shuffles = None, (), (sh_qkv, sh_qb, sh_o)
    if attn.indexer is not None:
        index, keep, ind_shuffles = _indexer(attn, idx, copies)
        shuffles += ind_shuffles
    # the kernel reads every in-place FP8 weight of a launch in one layout
    if len(set(shuffles)) != 1:
        raise ValueError(
            f"needs every block-FP8 projection in one layout, got shuffled {shuffles}"
        )
    return Glm53Weights(
        g_in=g_in,
        g_q=g_q,
        g_kv=g_kv,
        g_post=g_post,
        w_qkv_a=w_qkv,
        s_qkv_a=s_qkv,
        w_q_b=w_qb,
        s_q_b=s_qb,
        w_uk=w_uk,
        s_uk=s_uk,
        w_uv=w_uv,
        s_uv=s_uv,
        w_o=w_o,
        s_o=s_o,
        w_r=w_r,
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        shuffled=shuffles[0],
        index=index,
        keep=keep,
    )


def _cos_sin(rotary):
    """A rotary's (cos, sin) [positions, 32]: SGLang's f32 ``cos_sin_cache``
    [positions, 64] (cos, then sin), or AITER's ``cos_cache`` and ``sin_cache``."""
    cache = getattr(rotary, "cos_sin_cache", None)
    if cache is not None:
        return cache[:, : PE_DIM // 2], cache[:, PE_DIM // 2 :]
    cos, sin = rotary.cos_cache, rotary.sin_cache
    return cos.reshape(cos.shape[0], -1), sin.reshape(sin.shape[0], -1)


def rope_tables(layers: list, copies: dict):
    """The bf16 cos and sin tables [positions, 32] both rotaries of every covered
    layer read, from the first layer's MLA rotary; each indexer's must equal it."""
    cos, sin = _cos_sin(layers[0].self_attn.rotary_emb)
    for layer in layers:
        indexer = layer.self_attn.indexer
        if indexer is None:
            continue
        ic, is_ = _cos_sin(indexer.rotary_emb)
        if ic.shape != cos.shape or not (
            torch.equal(ic, cos) and torch.equal(is_, sin)
        ):
            raise ValueError("needs the indexer's rotary table to equal the MLA one")
    if cos.dim() != 2 or cos.shape[1] != PE_DIM // 2:
        raise ValueError(
            f"needs rotary tables [positions, {PE_DIM // 2}], got {tuple(cos.shape)}"
        )
    cos = cos.to(torch.bfloat16).contiguous()
    sin = sin.to(torch.bfloat16).contiguous()
    copies["rotary cos / sin bf16", "cos"] = cos
    copies["rotary cos / sin bf16", "sin"] = sin
    return cos, sin
