# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B-A17B block-FP8 on gfx950, TP4.

Decode steps of 1 to 8 rows run two new launches. A GDN layer's K1 is the input
norm, the block-FP8 projections, the conv and the gated delta. K2 is the output
projection, the tensor-parallel all-reduce, the post-norm and the block-FP8 MoE.
A full-attention layer writes the KV cache with SGLang's aiter attention, with
``o_proj`` replaced by identity, and then runs the same K2. Wider steps, and
anything this spec does not declare, keep the standard forward.
"""

from __future__ import annotations

import importlib.util
import os

import torch
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.qwen35_a17b_fp8.gfx950.layout import (
    BA_N,
    BLOCK,
    CONV_DIM,
    EXPERTS,
    HIDDEN,
    LAYERS,
    O_K,
    QKVZ_N,
    ROWS,
    TP,
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/monokernels/spec.py": (
        "DeviceLimits",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "k1.conv.cosine": 0.999,
        "k1.state.cosine": 0.999,
        "k1.proj.cosine": 0.99,
        "k1.proj.rel": 0.05,
        "k2.proj.cosine": 0.99,
        "k2.proj.rel": 0.05,
        "k2.moe.cosine": 0.99,
        "k2.moe.rel": 0.05,
    },
)

_TEXT = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": EXPERTS,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "rms_norm_eps": 1e-6,
    "vocab_size": 248320,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "full_attention_interval": 4,
}
_FULL = frozenset({"full_attention", "attention"})
_LINEAR = frozenset({"linear_attention"})
_FP8 = torch.float8_e4m3fn


def _text(config):
    return getattr(config, "text_config", config)


def _quant(config) -> dict:
    raw = getattr(config, "quantization_config", None) or getattr(
        _text(config), "quantization_config", None
    )
    return dict(raw or {})


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    del cfg
    text = _text(hf_config)
    reasons = [
        f"needs Qwen3.5-397B-A17B's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in _TEXT.items()
        if getattr(text, key, None) != want
    ]
    # Some loaders leave layer_types empty until the modules exist. The model
    # check then reads the real GDN and full-attention layers.
    kinds = list(getattr(text, "layer_types", None) or [])
    if kinds and (
        len(kinds) != LAYERS
        or any(
            (kinds[i] not in _FULL) if i % 4 == 3 else (kinds[i] not in _LINEAR)
            for i in range(len(kinds))
        )
    ):
        reasons.append(
            "needs 60 layers, linear attention except every 4th layer "
            f"(full attention), got {kinds[:8]}"
        )
    rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
    section = list(rope.get("mrope_section") or [])
    if section != [11, 11, 10] or rope.get("partial_rotary_factor") != 0.25:
        reasons.append(
            "needs mRoPE sections [11, 11, 10] and partial_rotary_factor 0.25 "
            f"(rotary dim 64), got {section} and {rope.get('partial_rotary_factor')!r}"
        )
    if rope.get("rope_theta") != 10000000:
        reasons.append(f"needs rope_theta 10000000, got {rope.get('rope_theta')!r}")
    quant = _quant(hf_config)
    method = quant.get("quant_method")
    block = list(quant.get("weight_block_size") or [])
    if method != "fp8" or quant.get("activation_scheme") != "dynamic" or block != [BLOCK, BLOCK]:
        reasons.append(
            "needs dynamic block-FP8 weights with weight_block_size [128, 128], "
            f"got quant_method {method!r}, activation_scheme "
            f"{quant.get('activation_scheme')!r}, weight_block_size {block}"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _shape(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _dense_block(module, n, k, what) -> list[str]:
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale_inv", None)
    if weight is None or getattr(weight, "dtype", None) != _FP8 or _shape(weight) != (n, k):
        return [f"{what} weight is not block-FP8 [{n}, {k}]"]
    if scale is None or getattr(scale, "dtype", None) != torch.float32 or _shape(scale) != (
        n // BLOCK,
        k // BLOCK,
    ):
        return [f"{what} scale is not a fp32 {BLOCK}x{BLOCK} block scale"]
    return []


def _floating(module, shape, what) -> list[str]:
    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", None)
    if weight is None or dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return [f"{what} is not an unquantized floating weight"]
    if shape is not None and _shape(weight) != shape:
        return [f"{what} weight is {_shape(weight)}, TP4 wants {shape}"]
    return []


def _expert_block(weight, scale, what, *, k_last: int) -> list[str]:
    shape = _shape(weight)
    if (
        getattr(weight, "dtype", None) != _FP8
        or len(shape) != 3
        or shape[2] != k_last
        or shape[1] % BLOCK
        or shape[0] < EXPERTS
    ):
        return [f"{what} weight is not block-FP8 [E, N, {k_last}]"]
    want = (shape[0], shape[1] // BLOCK, shape[2] // BLOCK)
    if scale is None or getattr(scale, "dtype", None) != torch.float32 or _shape(scale) != want:
        return [f"{what} scale is not a fp32 {BLOCK}x{BLOCK} block scale {want}"]
    return []


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    del hf_config, copies
    inner = getattr(model, "model", None)
    layers = list(getattr(inner, "layers", []) or [])
    if len(layers) != LAYERS:
        return [f"needs {LAYERS} decoder layers, got {len(layers)}"]
    reasons = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not getattr(mlp, "enable_shared_expert_fusion", False):
            reasons.append(f"layer {i}: the shared expert is not fused into the block-FP8 MoE")
            continue
        if getattr(mlp, "reduce_results", True):
            reasons.append(f"layer {i}: the MoE all-reduce is inside the block, K2 owns it")
        experts = getattr(mlp, "experts", None)
        reasons += _expert_block(
            getattr(experts, "w13_weight", None),
            getattr(experts, "w13_weight_scale_inv", None),
            f"layer {i} w13",
            k_last=HIDDEN,
        )
        w2 = getattr(experts, "w2_weight", None)
        w2_scale = getattr(experts, "w2_weight_scale_inv", None)
        w2_shape = _shape(w2)
        if (
            getattr(w2, "dtype", None) != _FP8
            or len(w2_shape) != 3
            or w2_shape[1] != HIDDEN
            or w2_shape[2] % BLOCK
        ):
            reasons.append(f"layer {i} w2 weight is not block-FP8 [E, {HIDDEN}, N]")
        elif (
            w2_scale is None
            or getattr(w2_scale, "dtype", None) != torch.float32
            or _shape(w2_scale) != (w2_shape[0], HIDDEN // BLOCK, w2_shape[2] // BLOCK)
        ):
            reasons.append(f"layer {i} w2 scale is not a fp32 {BLOCK}x{BLOCK} block scale")
        reasons += _floating(getattr(mlp, "gate", None), (EXPERTS, HIDDEN), f"layer {i} router")
        if hasattr(layer, "linear_attn"):
            if i % 4 == 3:
                reasons.append(f"layer {i}: a full-attention layer is registered as GDN")
            attn = layer.linear_attn
            reasons += _dense_block(attn.in_proj_qkvz, QKVZ_N, HIDDEN, f"layer {i} in_proj_qkvz")
            reasons += _floating(attn.in_proj_ba, (BA_N, HIDDEN), f"layer {i} in_proj_ba")
            conv = _shape(getattr(getattr(attn, "conv1d", None), "weight", None))
            if conv not in ((CONV_DIM, 1, 4), (CONV_DIM, 4)):
                reasons.append(f"layer {i} conv1d weight is {conv}, TP4 wants {(CONV_DIM, 1, 4)}")
            reasons += _dense_block(attn.out_proj, HIDDEN, O_K, f"layer {i} out_proj")
        elif hasattr(layer, "o_proj"):
            if i % 4 != 3:
                reasons.append(f"layer {i}: a GDN layer is registered as full attention")
            qkv = getattr(layer, "qkv_proj", None)
            qkv_weight = getattr(qkv, "weight", None)
            qkv_n = _shape(qkv_weight)[:1]
            if len(qkv_n) != 1 or qkv_n[0] % BLOCK:
                reasons.append(f"layer {i} qkv N is not a multiple of {BLOCK}")
            else:
                reasons += _dense_block(qkv, qkv_n[0], HIDDEN, f"layer {i} qkv")
            reasons += _dense_block(layer.o_proj, HIDDEN, O_K, f"layer {i} o_proj")
        else:
            reasons.append(f"layer {i}: neither a GDN layer nor a full-attention layer")
    return reasons


class _Built:
    """One rank's barriers and the decode forward."""

    def __init__(self, model, runtime):
        self.model = model
        self.runtime = runtime
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        """``Qwen3_5MoeForConditionalGeneration.forward`` for a routed decode step."""
        from sgl_amd_mono.qwen35_a17b_fp8.gfx950.forward import run_attn, run_k1, run_k2

        lm = self.model
        contains = getattr(forward_batch, "contains_image_inputs", None)
        if (
            get_embedding
            or pp_proxy_tensors is not None
            or plan.key.rows > ROWS[-1]
            or (contains is not None and contains())
        ):
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, get_embedding, pp_proxy_tensors
            )
        if getattr(lm, "is_mrope_enabled", False):
            positions = forward_batch.mrope_positions
        inner = lm.model
        self.runtime.advance()
        hidden = inner.embed_tokens(input_ids)
        rows = plan.key.rows
        if hidden.shape[0] != rows:
            raise RuntimeError(
                f"decode step has {hidden.shape[0]} rows, the routed width is {rows}"
            )
        residual = None
        start = getattr(inner, "start_layer", 0)
        end = getattr(inner, "end_layer", len(inner.layers))
        for i in range(start, end):
            layer = inner.layers[i]
            if hasattr(layer, "linear_attn"):
                core, residual = run_k1(
                    self.runtime, layer, hidden, residual, forward_batch, rows, self.check
                )
                proj = layer.linear_attn.out_proj
            else:
                core, residual = run_attn(layer, hidden, residual, positions, forward_batch)
                proj = layer.o_proj
            hidden, residual = run_k2(
                self.runtime, layer, core, residual, forward_batch, rows, proj, self.check
            )
        if residual is None:
            hidden = inner.norm(hidden)
        else:
            hidden = inner.norm(hidden, residual)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)


def _build(model, hf_config, ctx):
    del hf_config
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen35_a17b_fp8.gfx950.barrier import PhaseRuntime

    if ctx.isa != "gfx950":
        raise ValueError(f"Qwen3.5 block-FP8 monokernel is gfx950 only, got {ctx.isa}")
    parallel = get_parallel()
    if int(parallel.tp_size) != TP:
        raise ValueError(f"Qwen3.5 block-FP8 monokernel is TP{TP}, got TP{parallel.tp_size}")
    device = model.model.embed_tokens.weight.device
    runtime = PhaseRuntime(
        int(parallel.tp_rank), parallel.tp_group.cpu_group, device, ctx.deadline_s
    )
    return _Built(model, runtime)


SPEC = MonoSpec(
    name="qwen35_a17b_fp8_tp4",
    devices={"gfx950": DeviceLimits(widths=ROWS, tp_sizes=(TP,), resident_blocks=BLOCKS)},
    features=frozenset(),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={"moe_runner_backend": frozenset({"auto", "aiter"})},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,
    config_disable_reasons=_config_disable_reasons,
)


def _install_block_fp8_linear() -> None:
    # Prefill uses SGLang's block-FP8 linear, whose gfx950 small-M helper imports
    # symbols that module no longer defines. Install the stand-in before that import.
    from sgl_amd_mono.qwen35_a17b_fp8.gfx950.gemv import _allow_block_fp8_linear

    _allow_block_fp8_linear()


_install_block_fp8_linear()
