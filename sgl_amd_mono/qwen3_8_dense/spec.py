# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang PR #43558 (Apache-2.0):
# python/sglang/srt/layers/qwen3_8_dense_mono.py
"""Qwen3.8-27B dense decode on gfx950. TP1: every layer's FFN share runs in K2
with the o_proj / out_proj folded in, every GDN layer's share before it in K1,
fused across the layer boundary. TP2: K1 runs each GDN share with out_proj, K2
each MLP share, both before SGLang's all-reduce; the rest is SGLang's."""

from __future__ import annotations

import importlib.util
import logging
import os

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        BuildContext,
        DeviceLimits,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        BuildContext,
        DeviceLimits,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS

from . import HF_CONFIG

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "BuildContext",
        "DeviceLimits",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"', '"tp_size"'),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype",),
    "srt/arg_groups/*.py": ('"enable_linear_replayssm"',),
    "srt/layers/layernorm.py": ("_use_aiter", "_has_rocm_triton_gemma_rms_norm"),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "final_norm"),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "req_to_token_pool",
        "_track_mamba_state_decode",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/mem_cache/memory_pool.py": ("mamba2_layer_cache",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/model_executor/forward_batch_info.py": (
        "input_embeds",
        "mrope_positions",
    ),
    "srt/models/qwen2_moe.py": ("Qwen2MoeMLP",),
    "srt/models/qwen3_5.py": (
        "pp_group",
        "embed_tokens",
        "layers",
        "norm",
        "linear_attn",
        "layer_id",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "_prepare_qkv_gate",
        "attn",
        "attn_output_gate",
        "self_attention",
        "attn_boundary",
        "ffn_boundary",
        "flashinfer_mnnvl_cutedsl_fusion",
        "residual_batch",
    ),
    "kernels/ops/elementwise/elementwise.py": ("fused_sigmoid_mul",),
    "srt/models/qwen3_vl.py": (
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
        "is_mrope_enabled",
        "capture_aux_hidden_states",
        "logits_processor",
        "lm_head",
        "model",
    ),
}

ROWS = tuple(range(1, 9))
TP_SIZES = (1, 2)
# Measured on MI355X (PR #43558): wider steps lose to SGLang's path
TP_MAX_ROWS = {1: 4, 2: 8}
# K1 and K2 match SGLang's Triton path bit for bit (exact by construction)
NUMERICS = Numerics(
    NumericsKind.BITWISE,
    {
        "gdn.core.max_abs": 0.0,
        "gdn.res.max_abs": 0.0,
        "gdn.conv.max_abs": 0.0,
        "gdn.state.max_abs": 0.0,
        "ffn.out.max_abs": 0.0,
        "ffn.res.max_abs": 0.0,
    },
)


def _text(config):
    return getattr(config, "text_config", config)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    reasons = []
    text = _text(hf_config)
    for key, want in HF_CONFIG.items():
        got = getattr(text, key, None)
        if got != want:
            reasons.append(f"needs Qwen3.8-27B's {key} = {want!r}, got {got!r}")
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal != torch.float32 or dtypes.conv != torch.bfloat16:
        reasons.append(
            f"needs f32 SSM / bf16 conv state, got {dtypes.temporal} / {dtypes.conv}"
        )
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "quark":
        reasons.append(f"needs Quark MXFP4 weights, got {quant.get('quant_method')}")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    from sglang.srt.layers.layernorm import (
        _has_rocm_triton_gemma_rms_norm,
        _use_aiter,
    )
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
    from sglang.srt.runtime_context import get_parallel

    from .gfx950 import weights as W

    tp = get_parallel().tp_size
    m = model.model
    if model.capture_aux_hidden_states:
        return ["captures aux hidden states (speculative decoding)"]
    if m.pp_group.world_size != 1:
        return ["pipeline parallelism"]
    # K1 / K2 fold in the Gemma RMSNorm of SGLang's ROCm Triton kernel
    if not (_use_aiter and _has_rocm_triton_gemma_rms_norm):
        return ["Gemma RMSNorm not on the ROCm Triton kernel (SGLANG_USE_AITER=1)"]
    for i, layer in enumerate(m.layers):
        if not isinstance(layer.mlp, Qwen2MoeMLP):
            return [f"layer {i}: MLP {type(layer.mlp).__name__}"]
        why = W.gdn_reason(layer, tp=tp) if W.is_gdn(layer) else None
        if why is None:
            why = W.ffn_reason(layer, tp=tp)
        if why is not None:
            return [f"layer {i}: {why}"]
    return []


def _attention_core(layer, positions, x, forward_batch):
    """SGLang's attention up to (not including) o_proj, which K2 folds in."""
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    q, k, v, gate = layer._prepare_qkv_gate(
        positions=positions, hidden_states=x, forward_batch=forward_batch
    )
    o = layer.attn(q, k, v, forward_batch)
    if layer.attn_output_gate:
        o = fused_sigmoid_mul(o, gate, inplace=True)
    return o.contiguous()


class _Built:
    """The built monokernel: ``forward`` replaces the model's for covered steps."""

    def __init__(self, model, rt, ffn: dict, gdn: dict, *, tp: int, check: bool):
        self.model, self.rt, self.ffn, self.gdn = model, rt, ffn, gdn
        self.tp, self.cap, self.check = tp, TP_MAX_ROWS[tp], check
        # GDN layer -> its mamba pool cache, bound at the first launch: the pool
        # is allocated after the framework builds
        self.states: dict = {}

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        lm = self.model
        if (
            get_embedding
            or plan.key.rows > self.cap
            or forward_batch.input_embeds is not None
        ):
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, get_embedding, pp_proxy_tensors
            )
        self.rt.advance_step()
        if lm.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        step = self._tp1 if self.tp == 1 else self._tp2
        hidden = step(input_ids, positions, forward_batch)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)

    def _checking(self) -> bool:
        # the reference runs eagerly; captured graphs replay the launches alone
        return self.check and not torch.cuda.is_current_stream_capturing()

    def _linear_attn(self):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        lab = get_attn_backend().linear_attn_backend
        return lab, lab.forward_metadata.mamba_cache_indices

    def _state(self, i, layer, lab):
        from .gfx950 import weights as W

        cache = lab.req_to_token_pool.mamba2_layer_cache(layer.linear_attn.layer_id)
        state = self.states.get(i)
        if (
            state is None
            or state.rstate.data_ptr() != cache.temporal.data_ptr()
            or state.conv_state.data_ptr() != cache.conv[0].data_ptr()
        ):
            why = W.state_reason(cache, tp=self.tp)
            if why is not None:
                raise RuntimeError(f"Qwen3.8-27B monokernel: layer {i}: {why}")
            state = self.states[i] = W.gdn_state(cache)
        return state

    def _gdn(self, i, layer, lab, idx, h, residual, forward_batch):
        a = layer.linear_attn
        state = self._state(i, layer, lab)
        ref = None
        if self._checking():
            from . import check

            ref = check.reference_gdn(layer, h, residual, state, idx, tp=self.tp)
        core, res, out = self.rt.gdn(self.gdn[i], state, idx, h, residual, layer=i)
        if ref is not None:
            check.compare_gdn(i, ref, core, res, state, idx)
        lab._track_mamba_state_decode(
            forward_batch, state.conv_state, state.rstate, idx, a.layer_id
        )
        return core, res, out

    def _tp1(self, input_ids, positions, forward_batch):
        """Each layer's GDN share or attention core, then K2 from o_proj on."""
        m = self.model.model
        h = m.embed_tokens(input_ids)
        lab, idx = self._linear_attn()
        residual = None
        for i, layer in enumerate(m.layers):
            if i in self.gdn:
                core, res, _ = self._gdn(i, layer, lab, idx, h, residual, forward_batch)
            else:
                if residual is None:
                    x, res = layer.input_layernorm(h), h
                else:
                    x, res = layer.input_layernorm(h, residual)
                core = _attention_core(layer, positions, x, forward_batch)
                res = res.contiguous()
            ref = None
            if self._checking():
                from . import check

                ref = check.reference_ffn(layer, core, res)
            h, residual = self.rt.ffn(self.ffn[i], core, res, layer=i)
            if ref is not None:
                check.compare_ffn(i, ref, h, residual)
        h, _ = m.norm(h, residual)
        return h

    def _tp2(self, input_ids, positions, forward_batch):
        """SGLang's layer loop (``Qwen3_5ForCausalLM.forward`` and the decoder
        layers' ``forward``) with each GDN share and MLP share in K1 / K2; the
        norms and all-reduces stay SGLang's, and tuple inputs (fused quant
        paths) take SGLang's module."""
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        m = self.model.model
        fb = forward_batch
        h = m.embed_tokens(input_ids)
        residual_batch.start(fb)
        lab = idx = None
        for i, layer in enumerate(m.layers):
            h = layer.attn_boundary.prepare(h, fb, capture_gathered=None)
            if i not in self.gdn:
                h = layer.self_attention(
                    positions=positions, hidden_states=h, forward_batch=fb
                )
            elif isinstance(h, torch.Tensor):
                if lab is None:
                    lab, idx = self._linear_attn()
                _, _, h = self._gdn(i, layer, lab, idx, h, None, fb)
            else:
                h = layer.linear_attn(h, fb)
            h = layer.attn_boundary.finish(h, fb)
            h = layer.ffn_boundary.prepare(h, fb)
            with layer.ffn_boundary.exit(fb) as ffn_exit:
                if isinstance(h, torch.Tensor):
                    h = self._ffn_partial(i, layer, h)
                else:
                    h = layer.mlp(h)
            h = ffn_exit.finish(h)
        return residual_batch.final_norm(
            h,
            fb,
            m.norm,
            finalize_norm=m.flashinfer_mnnvl_cutedsl_fusion,
            skip_empty=True,
        )

    def _ffn_partial(self, i, layer, x):
        ref = None
        if self._checking():
            from . import check

            ref = check.reference_mlp(layer, x)
        out = self.rt.ffn_partial(self.ffn[i], x, layer=i)
        if ref is not None:
            check.compare_ffn(i, ref, out, None)
        return out


def _build(model, hf_config, ctx: BuildContext) -> _Built:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.common import guard

    from .gfx950 import runner as R
    from .gfx950 import weights as W

    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    layers = model.model.layers
    device = layers[0].input_layernorm.weight.device
    gdn_layers = [i for i, layer in enumerate(layers) if W.is_gdn(layer)]
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    cap = TP_MAX_ROWS[tp]
    R.ensure_writable_flydsl_cache()
    rt = R.DenseRuntime(
        tp=tp,
        max_rows=cap,
        eps=layers[0].input_layernorm.variance_epsilon,
        norm_eps=layers[gdn_layers[0]].linear_attn.norm.eps,
        fault_site=fault_site,
        fault_after=fault_after,
        device=device,
    )
    rt.set_deadline(ctx.deadline_s)
    ffn = {
        i: W.ffn_weights(layer, i, tp=tp, copies=ctx.copies)
        for i, layer in enumerate(layers)
    }
    gdn = {i: W.gdn_weights(layers[i], i, tp=tp, copies=ctx.copies) for i in gdn_layers}
    widths = sorted(r for r in ctx.widths if r <= cap)
    # Each rank starts at a different width so the ranks fill FlyDSL's disk cache
    # in parallel instead of all compiling the same kernel.
    start = len(widths) * rank // tp
    for w in widths[start:] + widths[:start]:
        rt.precompile([w])
    # The widths are per ISA, so the framework's routing line lists them all;
    # wider steps than the cap take the model's forward.
    logger.info(
        "%s: TP%d runs decode steps of up to %d rows, wider ones the model's forward",
        SPEC.name,
        tp,
        cap,
    )
    check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
    return _Built(model, rt, ffn, gdn, tp=tp, check=check)


SPEC = MonoSpec(
    name="qwen3_8_dense_hip",
    devices={
        "gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    features=frozenset(),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        server_args={"enable_linear_replayssm": frozenset({False})},
        env={"SGLANG_USE_AITER": True},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
