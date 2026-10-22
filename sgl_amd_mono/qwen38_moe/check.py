# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): next to each launch, SGLang's own layer
code runs on the launch's input, and the launch's outputs are checked against it.

- K1: SGLang's input norm and GDN up to out_proj's input, on the state rows as
  they were (restored after): the core, the residual, and the conv and SSM state
  rows of the step's requests.
- K2: SGLang's o_proj, TP reduce, post-attention norm and MoE block with its TP
  reduce, on K2's own input: the layer output and the residual. The MoE block
  runs one row at a time: in a batch of 8, SGLang's router picks other experts
  for some rows than for the same row alone.

Each value is checked against its ``spec.NUMERICS`` bound on TP rank 0. Every
check is logged; a miss is logged as a violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/models/qwen3_5.py": (
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "linear_attn",
        "out_proj",
        "o_proj",
        "layer_id",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
}


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    # equal also covers the graph warmup's zero tensors, where the cosine is 0
    return {
        "cosine": 1.0 if torch.equal(a, b) else F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _live(idx: torch.Tensor) -> torch.Tensor:
    return idx[idx > 0].long()


def k1_reference(layer, h, residual, conv, temporal, idx, forward_batch) -> dict:
    """SGLang's input norm and GDN on K1's input; the state rows are restored."""
    from sgl_amd_mono.qwen38_moe.spec import _proj_input

    rows = _live(idx)
    saved = conv[rows].clone(), temporal[rows].clone()
    if residual is None:
        res, x = h.clone(), layer.input_layernorm(h.clone())
    else:
        x, res = layer.input_layernorm(h.clone(), residual.clone())
    with _proj_input(layer.linear_attn, "out_proj"):
        core = layer.linear_attn(x, forward_batch)
    ref = {
        "core": core.clone(),
        "res_out": res.clone(),
        "conv": conv[rows].clone(),
        "state": temporal[rows].clone(),
    }
    conv[rows], temporal[rows] = saved
    return ref


def k1_outputs(core, res_out, conv, temporal, idx) -> dict:
    rows = _live(idx)
    return {
        "core": core,
        "res_out": res_out,
        "conv": conv[rows],
        "state": temporal[rows],
    }


def k2_reference(layer, o_proj, core, residual, forward_batch) -> dict:
    """SGLang's o_proj, post-attention norm and MoE block on K2's input. Both
    projections leave their TP sum to the layer's stage boundary: reduce here."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    a, _ = o_proj(core)
    a = tensor_model_parallel_all_reduce(a)
    x, res = layer.post_attention_layernorm(a, residual.clone())
    out = torch.cat(
        [
            tensor_model_parallel_all_reduce(layer.mlp(x[i : i + 1], forward_batch))
            for i in range(x.shape[0])
        ]
    )
    return {"out": out.clone(), "res_out": res.clone()}


def compare(kind: str, layer_id: int, rows: int, ref: dict, got: dict) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen38_moe.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return
    values = {
        f"{kind}.{name}.{metric}": v
        for name in ref
        for metric, v in _metrics(got[name], ref[name]).items()
    }
    logger.info(
        "monokernel check layer %d (%s, %d rows): %s",
        layer_id,
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
                layer_id,
                name,
                v,
                bound,
            )
