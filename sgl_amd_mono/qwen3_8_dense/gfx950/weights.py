# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang PR #43558 (Apache-2.0):
# python/sglang/srt/layers/qwen3_8_dense_mono.py
"""Which Qwen3.8-27B layers the dense launches take as SGLang loaded them, and
their weights as the launches read them: SGLang's parameters in place, except
the MXFP4 weights, which the launches read in the (16, 16)-shuffled order
(``dense_ffn.shuffle_w``) with plain e8m0 scales. A weight on SGLang's Triton
GEMM is copied once into that order; one on its ASM GEMM is in that order
already and is read in place (in_proj_qkvz, which SGLang may repack after the
build, is copied as it is), with its scales copied out of the ASM tile order
(``mxfp4_operands``; the caller records the copies in ``copies``)."""

from __future__ import annotations

import functools
from typing import Optional

import torch

from . import dense_ffn as D
from . import dense_gdn as K
from .runner import DenseFfnWeights, DenseGdnState, DenseGdnWeights

# The SGLang names these functions read
SGLANG_READS = {
    "srt/layers/quantization/quark/quark.py": ("QuarkLinearMethod",),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py": (
        "QuarkW4A4MXFP4",
        "dequantized_bf16",
        "use_aiter_asm_fp4_gemm",
        "_swizzle_asm_fp4_scale",
    ),
    "srt/layers/linear.py": ("quant_method",),
    "srt/layers/quantization/quark/*.py": ("scheme",),
    "srt/layers/quantization/quark/schemes/*.py": ('"weight_scale"',),
    "srt/layers/quantization/fp8.py": ("aiter_bpreshuffled", "is_shuffled"),
    "srt/models/qwen3_5.py": (
        "linear_attn",
        "mlp",
        "o_proj",
        "input_layernorm",
        "post_attention_layernorm",
        "attn_tp_size",
        "in_proj_qkvz",
        "in_proj_ba",
        "conv1d",
        "A_log",
        "dt_bias",
        "norm",
        "out_proj",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
    ),
    "srt/models/qwen2_moe.py": ("gate_up_proj", "down_proj"),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "eps",
        "activation",
        "group_size",
        "norm_before_gate",
    ),
    "srt/mem_cache/memory_pool.py": ("conv", "temporal", "replayssm_d"),
}


def is_gdn(layer) -> bool:
    return hasattr(layer, "linear_attn")


def o_proj(layer):
    """The projection K2 folds in at TP1: out_proj of a GDN layer, else o_proj."""
    return layer.linear_attn.out_proj if is_gdn(layer) else layer.o_proj


def asm_layout(proj) -> bool:
    """Whether SGLang runs ``proj`` on AITER's ASM FP4 GEMM (``--fp4-gemm-backend
    aiter``, or ``auto`` on gfx95 for the shapes AITER tunes). SGLang has then
    replaced the weight with AITER's ``shuffle_weight(layout=(16, 16))`` of it,
    the order ``dense_ffn.shuffle_w`` makes, and the scales with their ASM tile
    order (``_swizzle_asm_fp4_scale``), and marked neither ``is_shuffled``."""
    return bool(getattr(proj, "use_aiter_asm_fp4_gemm", False))


def unswizzle_scales(ws: torch.Tensor) -> torch.Tensor:
    """Plain e8m0 [N, K / 32] scales from the ASM tile order: the inverse of
    SGLang's ``_swizzle_asm_fp4_scale``, whose tiles are 32 rows by 8 groups."""
    n, groups = ws.shape
    return (
        ws.view(n // 32, groups // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .contiguous()
        .view(n, groups)
    )


@functools.lru_cache(maxsize=None)
def _unswizzles(n: int, groups: int) -> bool:
    """Whether ``unswizzle_scales`` inverts SGLang's ``_swizzle_asm_fp4_scale`` on
    [n, groups] scales; both only permute, so an index tensor shows it."""
    if n % 32 or groups % 8:
        return False
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
        _swizzle_asm_fp4_scale,
    )

    idx = torch.arange(n * groups, dtype=torch.int32).view(n, groups)
    return torch.equal(_swizzle_asm_fp4_scale(unswizzle_scales(idx)), idx)


def mxfp4_reason(proj, n: int, k: int, *, bitwise: bool = False) -> Optional[str]:
    """Why ``proj`` is not an MXFP4 [n, k] linear the kernels read, row-major on
    the Triton ``gemm_afp4wfp4`` path (the kernels' reference) or in the ASM
    GEMM's layout (``asm_layout``), else None. The kernels match the Triton path
    bit for bit, not the ASM one; ``bitwise`` refuses the ASM layout."""
    from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
        QuarkW4A4MXFP4,
    )

    if not isinstance(proj.quant_method, QuarkLinearMethod) or not isinstance(
        proj.scheme, QuarkW4A4MXFP4
    ):
        return "not Quark MXFP4"
    # set only on some load paths
    if getattr(proj, "dequantized_bf16", False):
        return "MXFP4 weights dequantized to bf16 (AITER_FORCE_A8W4)"
    w, ws = proj.weight, proj.weight_scale
    asm = asm_layout(proj)
    if asm and bitwise:
        return (
            "on the ASM FP4 GEMM, which the kernels do not match bit for bit "
            "(--fp4-gemm-backend triton)"
        )
    if not asm:
        # The kernels shuffle their own copy of a row-major weight; one another
        # path already preshuffled would be shuffled twice.
        for marker in ("is_shuffled", "aiter_bpreshuffled"):
            if getattr(w, marker, False) or getattr(ws, marker, False):
                return f"weights already preshuffled ({marker})"
    if w.dtype != torch.uint8 or w.shape != (n, k // 2):
        return f"weight {tuple(w.shape)} {w.dtype}, want MXFP4 {n} x {k}"
    if ws is None or ws.shape != (n, k // 32) or not ws.is_contiguous():
        return "weight scales not e8m0 [N, K / 32]"
    if asm and not w.is_contiguous():
        return "ASM-layout weight not contiguous"
    if asm and not _unswizzles(n, k // 32):
        return "ASM-layout scales not in the tile order unswizzle_scales inverts"
    return None


def mxfp4_operands(proj, name: str, idx: int, copies: dict, *, in_place: bool = True):
    """``proj``'s weight in the (16, 16)-shuffled order and its plain e8m0 scales:
    an ASM-layout weight in place (copied as it is without ``in_place``) with its
    scales unswizzled, any other weight copied shuffled with its scales in place.
    The copies go in ``copies``."""
    if asm_layout(proj):
        w, ws = proj.weight.data, unswizzle_scales(proj.weight_scale.data)
        copies[f"unswizzled {name} scales", idx] = ws
        if not in_place:
            w = w.clone()
            copies[f"copied {name}", idx] = w
        return w, ws
    w = D.shuffle_w(proj.weight.data.contiguous())
    copies[f"shuffled {name}", idx] = w
    return w, proj.weight_scale


def ffn_reason(layer, *, tp: int, bitwise: bool = False) -> Optional[str]:
    """Why K2 cannot run this layer's FFN share at TP ``tp`` (with o_proj folded
    in at TP1), else None; ``bitwise`` as in ``mxfp4_reason``."""
    inter = D.FGeo(tp).inter
    projs = [
        ("gate_up_proj", layer.mlp.gate_up_proj, 2 * inter, D.HIDDEN),
        ("down_proj", layer.mlp.down_proj, D.HIDDEN, inter),
    ]
    if tp == 1:
        projs.append(("o_proj", o_proj(layer), D.HIDDEN, D.CORE))
    for name, proj, n, k in projs:
        why = mxfp4_reason(proj, n, k, bitwise=bitwise)
        if why is not None:
            return f"{name}: {why}"
    return None


def gdn_reason(layer, *, tp: int, bitwise: bool = False) -> Optional[str]:
    """Why K1 cannot run this GDN layer's share at TP ``tp`` (with out_proj
    folded in at TP2), else None; ``bitwise`` as in ``mxfp4_reason``."""
    a = layer.linear_attn
    geo = K.Geo(tp)
    if a.attn_tp_size != tp:
        return f"GDN heads split {a.attn_tp_size} ways at TP{tp}"
    projs = [
        ("in_proj_qkvz", a.in_proj_qkvz, geo.qkvz, K.HIDDEN),
        ("in_proj_ba", a.in_proj_ba, geo.ba, K.HIDDEN),
    ]
    if tp == 2:
        projs.append(("out_proj", a.out_proj, K.HIDDEN, geo.val))
    for name, proj, n, k in projs:
        why = mxfp4_reason(proj, n, k, bitwise=bitwise)
        if why is not None:
            return f"{name}: {why}"
    conv = a.conv1d.weight
    if conv.dtype != torch.bfloat16 or conv.numel() != geo.conv * K.CONV_W:
        return f"conv1d weight {tuple(conv.shape)} {conv.dtype}"
    if not conv.is_contiguous() or a.conv1d.bias is not None:
        return "conv1d weight not contiguous or with bias"
    if a.A_log.dtype != torch.float32 or a.dt_bias.dtype != torch.bfloat16:
        return f"A_log {a.A_log.dtype} / dt_bias {a.dt_bias.dtype}"
    if a.A_log.numel() != geo.nv or a.dt_bias.numel() != geo.nv:
        return f"A_log / dt_bias of {a.A_log.numel()} heads, want {geo.nv}"
    gated_norm = (
        a.norm.weight.dtype == torch.bfloat16
        and a.norm.weight.numel() == K.HD
        and a.norm.activation in ("silu", "swish")
        and a.norm.group_size is None
        and a.norm.norm_before_gate
    )
    if not gated_norm:
        return "gated norm variant"
    heads = (a.num_k_heads, a.num_v_heads, a.head_k_dim, a.head_v_dim)
    if heads != (K.NK, K.NV, K.HD, K.HD):
        return f"heads {heads}"
    return None


def state_reason(cache, *, tp: int) -> Optional[str]:
    """Why K1 cannot read and write a GDN layer's mamba pool cache, else None."""
    geo = K.Geo(tp)
    t = cache.temporal
    if t.dtype != torch.float32 or t.shape[1:] != (geo.nv, K.HD, K.HD):
        return f"SSM state {tuple(t.shape)} {t.dtype}"
    if not t[0].is_contiguous():
        return "SSM state slot not dense"
    if cache.replayssm_d is not None:
        return "ReplaySSM ring"
    c = cache.conv[0]
    if c.dtype != torch.bfloat16 or c.shape[1:] != (geo.conv, K.SL):
        return f"conv state {tuple(c.shape)} {c.dtype}"
    return None


def gdn_state(cache) -> DenseGdnState:
    return DenseGdnState(conv_state=cache.conv[0], rstate=cache.temporal)


def ffn_weights(layer, idx: int, *, tp: int, copies: dict) -> DenseFfnWeights:
    """Layer ``idx``'s K2 operands: gate_up, down and (TP1) o_proj / out_proj as
    ``mxfp4_operands`` gives them."""
    mlp = layer.mlp
    w13, w13s = mxfp4_operands(mlp.gate_up_proj, "gate_up", idx, copies)
    w2, w2s = mxfp4_operands(mlp.down_proj, "down", idx, copies)
    w_o = w_os = None
    if tp == 1:
        w_o, w_os = mxfp4_operands(o_proj(layer), "o_proj", idx, copies)
    return DenseFfnWeights(
        w13=w13,
        w13s=w13s,
        w2=w2,
        w2s=w2s,
        ln_w=layer.post_attention_layernorm.weight,
        w_o=w_o,
        w_os=w_os,
    )


def gdn_weights(layer, idx: int, *, tp: int, copies: dict) -> DenseGdnWeights:
    """GDN layer ``idx``'s K1 operands: in_proj_qkvz, in_proj_ba and (TP2)
    out_proj as ``mxfp4_operands`` gives them."""
    a = layer.linear_attn
    # finalize_fused_in_proj may pack in_proj_qkvz with in_proj_ba after the KV
    # pool is sized, freeing its weight as loaded; read in place, that weight
    # would stay alive outside the pool's budget
    w_qkvz, w_qkvzs = mxfp4_operands(
        a.in_proj_qkvz, "in_proj_qkvz", idx, copies, in_place=False
    )
    w_ba, w_bas = mxfp4_operands(a.in_proj_ba, "in_proj_ba", idx, copies)
    w_o = w_os = None
    if tp == 2:
        w_o, w_os = mxfp4_operands(a.out_proj, "out_proj", idx, copies)
    return DenseGdnWeights(
        w_qkvz=w_qkvz,
        w_qkvzs=w_qkvzs,
        w_ba=w_ba,
        w_bas=w_bas,
        conv_w=a.conv1d.weight,
        a_log=a.A_log,
        dt_bias=a.dt_bias,
        norm_w=a.norm.weight,
        ln_w=layer.input_layernorm.weight,
        w_o=w_o,
        w_os=w_os,
    )
