# SPDX-License-Identifier: Apache-2.0
"""K1 and K2 for one decode step.

K1 (GDN layers): input RMSNorm, block-FP8 GDN projections, conv, gated delta.
K2: local output projection, tensor-parallel all-reduce, post-norm, router and
block-FP8 MoE, then the MoE all-reduce. Full-attention layers write the KV
cache with SGLang's aiter attention (``o_proj`` replaced by identity) and then
run the same K2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_amd_mono.qwen35_a17b_fp8.check import _metrics, log_check
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.gdn import gdn_core
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.gemv import block_fp8_gemv
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.moe import route_and_moe

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_req_to_token_pool",
    ),
}


class _IdentityOut(nn.Module):
    """The attention core, flat. ``self_attention`` unpacks ``o_proj`` as a pair."""

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        return x.reshape(x.shape[0], -1), None


def _add_norm(hidden, residual, norm):
    if residual is None:
        # copy first: a fused norm may rewrite ``hidden`` in place
        residual = hidden.clone()
        out = norm(hidden)
        if isinstance(out, tuple):
            out = out[0]
        return out, residual
    out = norm(hidden, residual)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, residual


def _allreduce(x: torch.Tensor) -> torch.Tensor:
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x)


def _mamba(layer_id: int, forward_batch):
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_req_to_token_pool,
    )

    backend = get_attn_backend()
    linear = getattr(backend, "linear_attn_backend", backend)
    meta = linear.forward_metadata
    idx = meta.mamba_cache_indices
    cache = get_req_to_token_pool().mamba2_layer_cache(layer_id)
    return cache.conv[0], cache.temporal, idx


def _slots(idx: torch.Tensor, rows: int) -> torch.Tensor:
    use = idx[:rows].to(torch.long)
    return use[use >= 0].unique()


def _checking(enabled: bool) -> bool:
    return enabled and not torch.cuda.is_current_stream_capturing()


def run_k1(runtime, layer, hidden, residual, forward_batch, rows: int, check: bool):
    runtime.enter(0)
    y, residual = _add_norm(hidden, residual, layer.input_layernorm)
    mod = layer.linear_attn
    conv, ssm, idx = _mamba(layer.layer_id, forward_batch)
    ref_out = ref_conv = ref_ssm = slots = None
    if _checking(check):
        slots = _slots(idx, rows)
        saved_conv = conv.index_select(0, slots).clone() if slots.numel() else None
        saved_ssm = ssm.index_select(0, slots).clone() if slots.numel() else None
        ref_out = mod(y, forward_batch)
        if slots.numel():
            ref_conv = conv.index_select(0, slots).clone()
            ref_ssm = ssm.index_select(0, slots).clone()
            conv.index_copy_(0, slots, saved_conv)
            ssm.index_copy_(0, slots, saved_ssm)
    core = gdn_core(mod, y, conv, ssm, idx, rows)
    if ref_out is not None:
        proj = block_fp8_gemv(core, mod.out_proj.weight, mod.out_proj.weight_scale_inv)
        values = {f"k1.proj.{k}": v for k, v in _metrics(proj, ref_out).items()}
        if slots is not None and slots.numel():
            values.update(
                {
                    f"k1.conv.{k}": v
                    for k, v in _metrics(conv.index_select(0, slots), ref_conv).items()
                }
            )
            values.update(
                {
                    f"k1.state.{k}": v
                    for k, v in _metrics(ssm.index_select(0, slots), ref_ssm).items()
                }
            )
        log_check(layer.layer_id, "k1", rows, values)
    return core, residual


def run_attn(layer, hidden, residual, positions, forward_batch):
    y, residual = _add_norm(hidden, residual, layer.input_layernorm)
    original = layer.o_proj
    layer.o_proj = _IdentityOut()
    try:
        core = layer.self_attention(positions, y, forward_batch)
    finally:
        layer.o_proj = original
    if isinstance(core, tuple):
        core = core[0]
    return core, residual


def run_k2(runtime, layer, core, residual, forward_batch, rows: int, proj, check: bool):
    runtime.enter(1)
    local = core[:rows]
    if local.shape[0] != rows:
        raise RuntimeError(
            f"layer {layer.layer_id} core has {local.shape[0]} rows, the step has {rows}"
        )
    partial = block_fp8_gemv(local, proj.weight, proj.weight_scale_inv)
    if _checking(check):
        ref, _ = proj(local)
        values = {f"k2.proj.{k}": v for k, v in _metrics(partial, ref).items()}
    else:
        values = None
    full = _allreduce(partial)
    y, residual = _add_norm(full, residual, layer.post_attention_layernorm)
    # reduce_results is false: both sides are this rank's partial, before the all-reduce
    ref_in = y.detach().clone() if values is not None else None
    local = route_and_moe(layer.mlp, y)
    if values is not None:
        ref_moe = layer.mlp(ref_in, forward_batch)
        if isinstance(ref_moe, tuple):
            ref_moe = ref_moe[0]
        values.update({f"k2.moe.{k}": v for k, v in _metrics(local, ref_moe).items()})
        log_check(layer.layer_id, "k2", rows, values)
    hidden = _allreduce(local)
    return hidden, residual
