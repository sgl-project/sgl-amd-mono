# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 on gfx950: the monokernel spec (RFC #43949).

A routed step is a speculative verify step (DSpark or DFLASH) of L draft tokens a
request (L = 5, 6 or 8). One request's step runs Kimi-K3's launches: K1 on the KDA
layers, the MLA projection front on the DSA layers and the layer launch (o_proj to
the MoE output) on every MoE layer. A step of R >= 2 requests runs
``wide.Wide.backbone``: one persistent launch a KDA layer, SGLang's routed experts
between launches, and SGLang's own path for layer 0, DSA attention and each run's
exit. Eligible DSA projection-front and post-attention stages use fused launches.

The indexer, sparse MLA and decode context parallel (DCP) combine stay on
SGLang's path. The fused projection front uses the `_mono_pre` handoff. Under DCP the
attention hands o_proj the same TP-local 12-head core as without it.
"""

from __future__ import annotations

import msgspec

try:
    from sglang.kernels.plugin_kernels.amd.spec import Feature, Numerics
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import Feature, Numerics

from sgl_amd_mono.kimi_k3 import spec as k3
from sgl_amd_mono.kimi_k31 import wide

SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "Feature",
        "Numerics",
        "KV_OFFLOAD",
    ),
    "srt/arg_groups/fields/*.py": (
        "enable_lmcache",
        "enable_flexkv",
        "disaggregation_decode_enable_offload_kvcache",
    ),
    "srt/environ.py": ("envs", "SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT"),
    "srt/models/kimi_k3.py": ("KimiK31ForConditionalGeneration",),
}


NUMERICS = Numerics(
    k3.NUMERICS.kind,
    {
        **k3.NUMERICS.bounds,
        # Measured with the (since removed) entry check on the real model (DFLASH,
        # DCP8, rows 12 .. 288): a run's first launch's prefix after the MoE tail vs
        # SGLang's whole MoE on the same input, median 3.6e-3; and a run's exit stream
        # vs the gathered-prefix tail, median 3.2e-3. bf16 rounding in another order.
        "layer.mid.rel": 0.006,
        "run.stream.rel": 0.006,
    },
)


def _offload_disable_reasons(
    *, lmcache: bool, flexkv: bool, decode_offload: bool, host_only_checkpoint: bool
) -> list[str]:
    # Feature.KV_OFFLOAD also admits these; only HiCache has run under the launches.
    flags = {
        "--enable-lmcache": lmcache,
        "--enable-flexkv": flexkv,
        "--disaggregation-decode-enable-offload-kvcache": decode_offload,
    }
    reasons = [
        f"{flag}: only HiCache's KV offload is supported"
        for flag, on in flags.items()
        if on
    ]
    if host_only_checkpoint:
        # SGLang then rings ReplaySSM state by request row; the launches index it by slot.
        reasons.append(
            "SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT: the launches index the ReplaySSM ring "
            "by mamba slot"
        )
    return reasons


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.environ import envs

    # Kimi-K3's checks also take the DCP sizes, attention CP and the wide draft lengths
    reasons = k3._config_disable_reasons(cfg=cfg, hf_config=hf_config)
    reasons += _offload_disable_reasons(
        lmcache=cfg.enable_lmcache,
        flexkv=cfg.enable_flexkv,
        decode_offload=cfg.disaggregation_decode_enable_offload_kvcache,
        host_only_checkpoint=envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.get(),
    )
    if not getattr(k3._text_config(hf_config), "dsa_config", None):
        reasons.append("no dsa_config: a Kimi-K3 checkpoint, served by kimi_k3")
    if not k3.WIDE:
        reasons.append(
            "SGLANG_AMD_MONO_K3_WIDE=0: kimi_k31 requires the wide launch for verify steps"
        )
    tokens = getattr(cfg, "speculative_num_draft_tokens", None)
    if tokens not in wide.SUPPORTED_L:
        reasons.append(
            f"--speculative-num-draft-tokens {tokens} is not one of {list(wide.SUPPORTED_L)}"
        )
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    reasons = k3._model_disable_reasons(model, hf_config, copies)
    layers = k3._backbone(model).layers
    covered = [i for i, layer in enumerate(layers) if wide._is_covered(layer)]
    own: list[str] = []
    if not covered:
        own.append("no KDA layer with a latent MoE")
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

    k3_checked = k3.WIDE and k3._wide_will_run()
    if not k3_checked:
        for i in covered:
            attn = layers[i].self_attn
            try:
                k1_mod._validate_in_proj_operands(attn)
            except ValueError as e:
                own.append(f"layer {i}: {e}")
            if own and i > covered[0]:
                break
    return reasons + own


_Built = k3._WideBuilt


def _build(model, hf_config, ctx):
    return k3._build(model, hf_config, ctx, use_wide=True)


SPEC = msgspec.structs.replace(
    k3.SPEC,
    name="kimi_k31_hip",
    devices={
        "gfx950": msgspec.structs.replace(
            k3.SPEC.devices["gfx950"],
            widths=tuple(range(1, k3.MAX_ROWS + 1)) + wide.wide_widths(),
        )
    },
    # DCP shards the DSA layers' KV, which run on SGLang's path.
    # HiCache moves KDA states between steps; a launch reads the device state pool by slot.
    features=k3.SPEC.features | {Feature.CONTEXT_PARALLEL, Feature.KV_OFFLOAD},
    requires=msgspec.structs.replace(
        k3.SPEC.requires,
        attention_backends=frozenset({"dsa"}),
        server_args={
            **k3.SPEC.requires.server_args,
            # DSpark's and DFLASH's target steps are uniform verify steps of L rows a request
            "speculative_algorithm": frozenset({"DSPARK", "DFLASH"}),
            # K1 and the wide KDA stage read and write the ReplaySSM ring
            "enable_linear_replayssm_spec": frozenset({True}),
        },
        hf_config={"model_type": "kimi_k31"},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    config_disable_reasons=_config_disable_reasons,
)
