# SPDX-License-Identifier: Apache-2.0
"""K1 and K2 for one decode step.

K1 (GDN layers): the previous layer's MoE all-reduce, residual add and input
RMSNorm with the FP8 quant of its output, the block-FP8 GDN projections, conv,
gated delta and the gated norm, which also quantizes the core for
``out_proj``. K2: local output projection, then its tensor-parallel all-reduce,
residual add and post-norm, then router and block-FP8 MoE, whose partial sum
the next layer (or the final norm) reduces. Full-attention layers take the
same reduced, normed and quantized input into ``qkv_proj``, then SGLang's q/k
norm, rope and aiter attention, which writes the KV cache. K2 then applies the
output gate with the FP8 quant of ``o_proj``'s input.

Each all-reduce, add and Gemma RMSNorm (and quant) is one aiter custom
all-reduce kernel with ``1 + w`` taken in fp32, as SGLang's unfused Gemma norm
does; without the custom all-reduce it falls back to the all-reduce and the
separate norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sgl_amd_mono.qwen35_a17b_fp8.check import _metrics, log_check
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.gdn import gdn_core
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.gemv import block_fp8_gemv_wg
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.layout import BLOCK
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.moe import route_and_moe
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.norm import add_norm_quant, gate_quant

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/distributed/parallel_state.py": ("get_tp_group", "ca_comm"),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_req_to_token_pool",
    ),
    "srt/models/qwen3_5.py": ("_prepare_qkv_gate", "qkv_proj", "attn"),
}

# Inputs up to this size take the one-stage fused all-reduce, larger ones the two-stage
ONE_STAGE_BYTES = 128 * 1024


class _QuantInputLinear(nn.Module):
    """``qkv_proj`` on the FP8 pair the input norm already made. SGLang unpacks a
    linear's output as a pair."""

    def __init__(self, linear, xq):
        super().__init__()
        self.args = (xq, linear.weight, linear.weight_scale_inv)

    def forward(self, x):
        xq, weight, scale = self.args
        return block_fp8_gemv_wg(x, xq, weight, scale), None


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


def _custom_allreduce(method: str):
    from sglang.srt.distributed.parallel_state import get_tp_group

    comm = get_tp_group().ca_comm
    if comm is None or getattr(comm, "disabled", True):
        return None
    return getattr(comm, method, None)


def _one_stage(x: torch.Tensor) -> bool:
    return x.numel() * x.element_size() <= ONE_STAGE_BYTES


def _reduce_norm(partial, residual, norm):
    """``(y, residual)``: all-reduce ``partial``, add ``residual``, Gemma RMSNorm."""
    fused = _custom_allreduce("custom_fused_ar_rms")
    if fused is not None:
        out = fused(
            partial,
            residual,
            norm.weight,
            norm.variance_epsilon,
            _one_stage(partial),
            gemma_norm=True,
        )
        if out is not None:
            return out
    return _add_norm(_allreduce(partial), residual, norm)


def _reduce_norm_quant(partial, residual, norm):
    """``(y, residual, (q, scale))``: ``_reduce_norm`` and the 1x128 FP8 quant of
    ``y``, with the column-major scale the preshuffled GEMM reads."""
    fused = _custom_allreduce("custom_fused_ar_rms_per_group_quant")
    if fused is not None:
        out = fused(
            partial,
            residual,
            norm.weight,
            norm.variance_epsilon,
            BLOCK,
            _one_stage(partial),
            emit_bf16=True,
            transpose_scale=True,
            gemma_norm=True,
        )
        if out is not None:
            q, residual, scale, y = out
            return y, residual, (q, scale)
    return add_norm_quant(_allreduce(partial), residual, norm)


def _norm_quant(hidden, residual, norm):
    """The input norm and quant of a layer: ``hidden`` is the embedding when
    ``residual`` is None, else the previous layer's MoE partial."""
    if residual is None:
        return add_norm_quant(hidden, None, norm)
    return _reduce_norm_quant(hidden, residual, norm)


def final_norm(partial, residual, norm):
    """The last layer's MoE all-reduce, residual add and the final norm."""
    return _reduce_norm(partial, residual, norm)[0]


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


def run_k1(layer, hidden, residual, forward_batch, rows: int, check: bool):
    """The GDN core ``(bf16, (fp8, scale))`` and the residual."""
    y, residual, y_q = _norm_quant(hidden, residual, layer.input_layernorm)
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
    core, core_q = gdn_core(mod, y, y_q, conv, ssm, idx, rows)
    if ref_out is not None:
        proj = block_fp8_gemv_wg(
            core, core_q, mod.out_proj.weight, mod.out_proj.weight_scale_inv
        )
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
    return (core, core_q), residual


def run_attn(layer, hidden, residual, positions, forward_batch):
    """The attention core with its output gate ``(core, gate)``, and the residual."""
    y, residual, y_q = _norm_quant(hidden, residual, layer.input_layernorm)
    qkv_proj = layer.qkv_proj
    layer.qkv_proj = _QuantInputLinear(qkv_proj, y_q)
    try:
        q, k, v, gate = layer._prepare_qkv_gate(positions, y, forward_batch)
    finally:
        layer.qkv_proj = qkv_proj
    core = layer.attn(q, k, v, forward_batch)
    return (core.reshape(core.shape[0], -1), gate), residual


def run_k2(layer, core, residual, forward_batch, rows: int, proj, check: bool):
    """This rank's MoE partial and the residual. ``core`` is K1's
    ``(bf16, (fp8, scale))`` or the attention's ``(core, gate)``."""
    core, extra = core
    local = core[:rows]
    if local.shape[0] != rows:
        raise RuntimeError(
            f"layer {layer.layer_id} core has {local.shape[0]} rows, the step has {rows}"
        )
    if isinstance(extra, torch.Tensor):
        local, local_q = gate_quant(local, extra[:rows])
    else:
        local_q = extra
    partial = block_fp8_gemv_wg(local, local_q, proj.weight, proj.weight_scale_inv)
    if _checking(check):
        ref, _ = proj(local)
        values = {f"k2.proj.{k}": v for k, v in _metrics(partial, ref).items()}
    else:
        values = None
    y, residual = _reduce_norm(partial, residual, layer.post_attention_layernorm)
    # reduce_results is false: both sides are this rank's partial, before the all-reduce
    ref_in = y.detach().clone() if values is not None else None
    local = route_and_moe(layer.mlp, y)
    if values is not None:
        ref_moe = layer.mlp(ref_in, forward_batch)
        if isinstance(ref_moe, tuple):
            ref_moe = ref_moe[0]
        values.update({f"k2.moe.{k}": v for k, v in _metrics(local, ref_moe).items()})
        log_check(layer.layer_id, "k2", rows, values)
    return local, residual
