# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's spec on a server with ``--enable-linear-replayssm-spec``
(``spec.SPEC`` hands its checks and build here): MTP verify steps of more than
``wide.NARROW`` rows run ``wide.Wide.backbone``; narrower steps run SGLang's forward.
The wide KDA stage is SGLang's ReplaySSM ring verify, which the narrow launches'
verify scratch excludes, so one server runs one of the two.

HiCache: the launch reads the committed KDA state and writes only the verify window
and the ring, as SGLang's verify does, and leaves the DSA KV and index cache to
SGLang's attention; a verify step's batch has no HiCache load in flight (loads are
an extend batch's, waited for before its draft)."""

from __future__ import annotations

import importlib.util

import torch

from sgl_amd_mono.glm53_flash import HF_CONFIG, LINEAR_ATTN_CONFIG, wide

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype",),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
}


def _text(config):
    return getattr(config, "text_config", config)


def config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    text = _text(hf_config)
    reasons = [
        f"needs GLM-5.3-Flash's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    linear = getattr(text, "linear_attn_config", None) or {}
    reasons += [
        f"needs GLM-5.3-Flash's linear_attn_config {key} = {want!r}"
        for key, want in LINEAR_ATTN_CONFIG.items()
        if linear.get(key) != want
    ]
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal not in (torch.float32, torch.bfloat16) or (
        dtypes.conv != torch.bfloat16
    ):
        reasons.append(
            f"needs the KDA state in f32 or bf16 and its conv state in bf16, got "
            f"{dtypes.temporal} and {dtypes.conv}"
        )
    # the target's verify steps of the NextN draft (NEXTN resolves to EAGLE)
    if cfg.speculative_algorithm != "EAGLE":
        reasons.append(
            f"the ReplaySSM ring verify serves MTP (EAGLE), got "
            f"{cfg.speculative_algorithm!r}"
        )
    tokens = cfg.speculative_num_draft_tokens
    if (
        cfg.speculative_eagle_topk != 1
        or tokens != cfg.speculative_num_steps + 1
        or tokens not in wide.SUPPORTED_L
    ):
        reasons.append(
            f"MTP verifies one draft chain of {list(wide.SUPPORTED_L)} tokens: topk 1, "
            "draft tokens = steps + 1"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53_flash.gfx950.weights import kda_weights

    reasons = []
    if get_parallel().tp_size != wide.TP:
        reasons.append(f"the wide launch is built for TP{wide.TP}")
    if model.is_mrope_enabled:
        reasons.append("multimodal rotary positions")
    if getattr(model.model, "layers_to_capture", None):
        reasons.append("auxiliary hidden-state capture (EAGLE3 / DFLASH)")
    layers = model.model.layers
    covered = [i for i, ly in enumerate(layers) if wide.is_covered(ly)]
    if not covered:
        reasons.append("no KDA layer with the routed MoE")
    for i in covered:
        try:
            kda_weights(layers[i], i, wide.TP, {})
        except ValueError as e:
            reasons.append(f"layer {i}: {e}")
        why = wide.moe_reason(layers[i].mlp)
        if why:
            reasons.append(f"layer {i}: the launch's top-k does not fit: {why}")
    return reasons


class _Built:
    def __init__(self, lm, runner):
        self.lm, self.wide = lm, runner

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        lm = self.lm
        if (
            plan.key.rows <= wide.NARROW
            or plan.key.tokens_per_request != self.wide.L
            or input_embeds is not None
            or getattr(forward_batch, "ragged_verify_layout", None) is not None
        ):
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            try:
                hidden = self.wide.backbone(input_ids, positions, forward_batch)
            except BaseException as e:
                import traceback

                wide._say(
                    f"wide backbone raised {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()[-2000:]}"
                )
                raise
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)


def build(model, hf_config, ctx):
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    from sglang.srt.runtime_context import get_parallel, get_spec

    p = get_parallel()
    L = int(get_spec().speculative_num_draft_tokens)
    device = model.model.layers[0].input_layernorm.weight.device
    state_bf16 = mamba2_state_dtype(hf_config).temporal == torch.bfloat16
    w = wide.Wide(
        model, p.tp_rank, p.tp_group.cpu_group, device, L, state_bf16, ctx.copies
    )
    w.warmup()
    torch.cuda.empty_cache()
    return _Built(model, w)
