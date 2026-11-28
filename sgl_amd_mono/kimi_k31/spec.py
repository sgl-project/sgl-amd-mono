# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 on gfx950: the monokernel spec (RFC #43949).

A routed step is a speculative verify step (DSpark or DFLASH) of L draft tokens a
request (L = 6 or 8). One request's step runs Kimi-K3's launches: K1 on the KDA
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
import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import Feature, Numerics
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import Feature, Numerics

from sgl_amd_mono.kimi_k3 import spec as k3
from sgl_amd_mono.kimi_k31 import wide

SGLANG_READS = {
    # the KDA state dtype SGLang allocates, which the wide launch reads as fp32
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype", "temporal"),
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
    "srt/models/kimi_k3.py": (
        "KimiK31ForConditionalGeneration",
        "_qkvgbfa_layer",
        "o_proj",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

# The decode context parallel sizes the launches were checked at.
DCP_SIZES = (1, 8)

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

    reasons = k3._config_disable_reasons(cfg=cfg, hf_config=hf_config)
    reasons += _offload_disable_reasons(
        lmcache=cfg.enable_lmcache,
        flexkv=cfg.enable_flexkv,
        decode_offload=cfg.disaggregation_decode_enable_offload_kvcache,
        host_only_checkpoint=envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.get(),
    )
    if not getattr(k3._text_config(hf_config), "dsa_config", None):
        reasons.append("no dsa_config: a Kimi-K3 checkpoint, served by kimi_k3")
    dcp = getattr(cfg, "dcp_size", 1) or 1
    if dcp not in DCP_SIZES:
        reasons.append(f"--dcp-size {dcp} is not one of {list(DCP_SIZES)}")
    if (getattr(cfg, "attn_cp_size", 1) or 1) > 1 or getattr(
        cfg, "enable_prefill_cp", False
    ):
        # CONTEXT_PARALLEL is accepted for DCP; attention CP reshapes the decode rows
        reasons.append("attention context parallel is on; only --dcp-size is accepted")
    # The wide KDA stage reads the recurrent state as fp32 (4 B a slot element);
    # Kimi-K3's K1 also takes bf16, the wide launch does not.
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    state = mamba2_state_dtype(None).temporal
    if state != torch.float32:
        reasons.append(f"the KDA state is {state}; the wide launch reads float32")
    tokens = getattr(cfg, "speculative_num_draft_tokens", None)
    if tokens not in wide.SUPPORTED_L:
        reasons.append(
            f"--speculative-num-draft-tokens {tokens} is not one of {list(wide.SUPPORTED_L)}"
        )
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Kimi-K3's per-layer checks, and what the wide launch reads on top: every
    KDA layer's merged in_proj, and its in_proj and o_proj weights as bf16."""
    reasons = k3._model_disable_reasons(model, hf_config, copies)
    layers = k3._backbone(model).layers
    covered = [i for i, layer in enumerate(layers) if wide._is_covered(layer)]
    if not covered:
        reasons.append("no KDA layer with a latent MoE")
    unmerged = [
        i
        for i in covered
        if getattr(layers[i].self_attn, "_qkvgbfa_layer", None) is None
    ]
    if unmerged:
        reasons.append(f"layers {unmerged}: the KDA in_proj is not merged")

    # Kimi-K3's checks take a Quark FP8 o_proj and in-projection (K1 and the layer
    # launch read them); the wide launch reads both as bf16 only.
    def _o_proj_not_bf16(attn) -> bool:
        # a qweight-only o_proj has no weight; K3's checks already report it
        weight = getattr(attn.o_proj, "weight", None)
        return weight is not None and weight.dtype != torch.bfloat16

    not_bf16 = [
        i
        for i in covered
        if i not in unmerged
        and (
            layers[i].self_attn._qkvgbfa_layer.weight.dtype != torch.bfloat16
            or _o_proj_not_bf16(layers[i].self_attn)
        )
    ]
    if not_bf16:
        reasons.append(
            f"layers {not_bf16}: the KDA in_proj or o_proj is not bf16, which the "
            "wide launch reads"
        )
    return reasons


class _Built(k3._Built):
    """Kimi-K3's launches for the narrow widths, and the wide layer launch for the
    verify steps above them."""

    def __init__(self, narrow, wide_runner) -> None:
        self.__dict__.update(narrow.__dict__)
        self._wide = wide_runner
        self._shim = k3._LanguageModelShim(self)

    def _backbone(self, input_ids, forward_batch, input_embeds):
        if self._plan.key.rows <= wide.NARROW:
            return super()._backbone(input_ids, forward_batch, input_embeds)
        try:
            return self._wide.backbone(self, input_ids, forward_batch, input_embeds)
        except BaseException as e:
            # surface it: the caller's handling must not hide a wide-path failure
            import traceback

            wide._say(
                f"wide backbone raised {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            )
            raise


def _build(model, hf_config, ctx):
    from sglang.srt.runtime_context import get_parallel

    narrow_widths = tuple(w for w in ctx.widths if w <= wide.NARROW)
    narrow = k3.SPEC.build(
        model, hf_config, msgspec.structs.replace(ctx, widths=narrow_widths)
    )
    if not any(w > wide.NARROW for w in ctx.widths):
        return narrow

    def mem():
        return (
            torch.cuda.mem_get_info()[0],
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )

    def used(a, b):
        return "device {:.2f} / torch allocated {:.2f} / reserved {:.2f} GiB".format(
            *((x - y) / 2**30 for x, y in ((a[0], b[0]), (b[1], a[1]), (b[2], a[2])))
        )

    p = get_parallel()
    device = k3._backbone(model).layers[0].input_layernorm.weight.device
    m0 = mem()
    w = wide.Wide(
        model=model,
        rank=p.tp_rank,
        group=p.tp_group.cpu_group,
        device=device,
        L=wide.tokens(),
        max_rows=max(ctx.widths),
    )
    m1 = mem()
    w.warmup()
    m2 = mem()
    torch.cuda.empty_cache()
    wide._say(
        f"memory: wide layers {used(m0, m1)}; warmup {used(m1, m2)}; after empty_cache {used(m0, mem())}"
    )
    return _Built(narrow, w)


SPEC = msgspec.structs.replace(
    k3.SPEC,
    name="kimi_k31_hip",
    devices={
        "gfx950": msgspec.structs.replace(
            k3.SPEC.devices["gfx950"],
            widths=k3.SPEC.devices["gfx950"].widths + wide.wide_widths(),
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
