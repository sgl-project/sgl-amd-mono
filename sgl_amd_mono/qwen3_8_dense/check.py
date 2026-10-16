# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang PR #43558 (Apache-2.0):
# python/sglang/srt/layers/qwen3_8_dense_mono.py
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each covered layer also runs SGLang's
path on the kernel's inputs, on the caches as they were, and checks the kernel's
outputs and the state rows it writes against it, each against its
``spec.NUMERICS`` bound. Every check is logged; a miss is logged as a violation.
TP1 checks K1 and K2; TP2 checks K1's core and state."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/attention/fla/fused_recurrent.py": (
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
    "kernels/ops/mamba/causal_conv1d_triton.py": ("causal_conv1d_update",),
    "srt/models/qwen3_5.py": (
        "linear_attn",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "_forward_input_proj",
        "fix_query_key_value_ordering",
        "conv1d",
        "activation",
        "num_v_heads",
        "attn_tp_size",
        "head_v_dim",
        "head_k_dim",
        "A_log",
        "dt_bias",
        "norm",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
}


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    return {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _rows(idx: torch.Tensor) -> torch.Tensor:
    # padded rows carry slot -1
    return idx[idx >= 0].long()


def _stock_gdn(layer, h, residual, conv_state, ssm_state, idx, *, normed: bool):
    """SGLang's GDN decode up to the gated norm; ``normed`` inputs skip the
    input layernorm (TP2)."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    a = layer.linear_attn
    if normed:
        x, res = h, None
    elif residual is None:
        x, res = layer.input_layernorm(h), h
    else:
        x, res = layer.input_layernorm(h, residual)
    qkvz, ba = a._forward_input_proj(x)
    q, k, v, z, b, ga = a.fix_query_key_value_ordering(qkvz, ba)
    mixed = torch.cat([q, k, v.reshape(v.size(0), -1)], dim=-1)
    mixed = causal_conv1d_update(
        mixed,
        conv_state,
        a.conv1d.weight.view(a.conv1d.weight.size(0), -1),
        a.conv1d.bias,
        a.activation,
        conv_state_indices=idx,
    )
    o = mixed.new_empty(h.size(0), 1, a.num_v_heads // a.attn_tp_size, a.head_v_dim)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=ga.contiguous(),
        b=b.contiguous(),
        A_log=a.A_log,
        dt_bias=a.dt_bias,
        scale=a.head_k_dim**-0.5,
        initial_state=ssm_state,
        out=o,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    core = a.norm(o.view(-1, a.head_v_dim), z.reshape(-1, a.head_v_dim))
    return core.view(h.size(0), -1), res


def reference_gdn(layer, h, residual, state, idx, *, tp: int) -> dict:
    """SGLang's GDN outputs on ``h``; the state rows are restored after."""
    rows = _rows(idx)
    saved = (state.conv_state[rows].clone(), state.rstate[rows].clone())
    core, res = _stock_gdn(
        layer, h, residual, state.conv_state, state.rstate, idx, normed=tp > 1
    )
    ref = {
        "core": core.clone(),
        "res": None if res is None else res.clone(),
        "conv": state.conv_state[rows].clone(),
        "state": state.rstate[rows].clone(),
    }
    state.conv_state[rows], state.rstate[rows] = saved
    return ref


def reference_ffn(layer, core, residual) -> dict:
    """SGLang's TP1 o_proj, residual add, post-norm and MLP on ``core``."""
    from .gfx950.weights import o_proj

    o, _ = o_proj(layer)(core)
    x, res = layer.post_attention_layernorm(o, residual)
    return {"out": layer.mlp(x).clone(), "res": res.clone()}


def reference_mlp(layer, x) -> dict:
    """SGLang's TP2 MLP share on the post-normed ``x``, before the all-reduce."""
    return {"out": layer.mlp(x).clone(), "res": None}


def compare_gdn(i: int, ref: dict, core, res, state, idx) -> None:
    rows = _rows(idx)
    outputs = {
        "core": _metrics(core, ref["core"]),
        "conv": _metrics(state.conv_state[rows], ref["conv"]),
        "state": _metrics(state.rstate[rows], ref["state"]),
    }
    if ref["res"] is not None:
        outputs["res"] = _metrics(res, ref["res"])
    _report(i, "gdn", core.shape[0], outputs)


def compare_ffn(i: int, ref: dict, out, res) -> None:
    outputs = {"out": _metrics(out, ref["out"])}
    if ref["res"] is not None:
        outputs["res"] = _metrics(res, ref["res"])
    _report(i, "ffn", out.shape[0], outputs)


def _report(i: int, kind: str, rows: int, outputs: dict) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen3_8_dense.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return
    values = {
        f"{kind}.{out}.{metric}": v
        for out, ms in outputs.items()
        for metric, v in ms.items()
    }
    logger.info(
        "monokernel check layer %d (%s, %d rows): %s",
        i,
        kind,
        rows,
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                i,
                name,
                v,
                bound,
            )
