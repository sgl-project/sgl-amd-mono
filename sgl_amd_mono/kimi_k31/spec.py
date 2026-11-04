# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 on gfx950: the monokernel spec (RFC #43949).

A routed step of up to 8 rows (a decode step, or a verify step of one request)
runs Kimi-K3's launches: K1 on the KDA layers, the MLA projection front on the
DSA layers and the layer launch (o_proj to the MoE output) on every MoE layer.
A speculative verify step (DSpark or DFLASH) of R >= 2 requests x L draft tokens
(L = 6 or 8) above 8 rows runs ``wide.Wide.backbone``: one persistent launch a
KDA layer, SGLang's routed experts between launches, and SGLang's own path for
layer 0, the DSA layers and each run's exit. ``SGLANG_AMD_MONO_K31_WIDE=0``
leaves those steps on SGLang's path.

The DSA layers keep SGLang's attention up to o_proj, so the indexer, the sparse
MLA and the decode context parallel (DCP) combine stay on SGLang's path. Under
DCP the attention hands o_proj the same TP-local 12-head core as without it.
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
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "Feature",
        "Numerics",
    ),
    "srt/models/kimi_k3.py": ("KimiK31ForConditionalGeneration", "_qkvgbfa_layer"),
    "srt/runtime_context.py": ("get_parallel",),
}

# The decode context parallel sizes the launches were checked at.
DCP_SIZES = (1, 8)

NUMERICS = Numerics(
    k3.NUMERICS.kind,
    {
        **k3.NUMERICS.bounds,
        # SGLANG_AMD_MONO_K31_WIDE_ENTRY_CHECK on the real model (DFLASH, DCP8, rows
        # 12 .. 288): a run's first launch's prefix after the MoE tail vs SGLang's
        # whole MoE on the same input, median 3.6e-3; and a run's exit stream vs the
        # gathered-prefix tail, median 3.2e-3. bf16 rounding in another order.
        "layer.mid.rel": 0.006,
        "run.stream.rel": 0.006,
    },
)


def _wide_config_disable_reasons(cfg) -> list[str]:
    reasons = []
    if getattr(cfg, "speculative_algorithm", None) not in ("DSPARK", "DFLASH"):
        reasons.append(
            "the wide launch serves DSpark and DFLASH verify steps; set "
            "SGLANG_AMD_MONO_K31_WIDE=0 to serve decode on the Kimi-K3 launches"
        )
    tokens = getattr(cfg, "speculative_num_draft_tokens", None)
    if tokens not in wide.SUPPORTED_L:
        reasons.append(
            f"--speculative-num-draft-tokens {tokens} is not one of "
            f"{list(wide.SUPPORTED_L)} (the wide launch; SGLANG_AMD_MONO_K31_WIDE=0 "
            "serves up to 8 rows only)"
        )
    if not getattr(cfg, "enable_linear_replayssm_spec", False):
        reasons.append(
            "the wide launch reads and writes the ReplaySSM ring: pass "
            "--enable-linear-replayssm-spec, or set SGLANG_AMD_MONO_K31_WIDE=0"
        )
    return reasons


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = k3._config_disable_reasons(cfg, hf_config)
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
    if wide.enabled():
        reasons += _wide_config_disable_reasons(cfg)
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Kimi-K3's per-layer checks, and what the wide launch reads on top: every
    KDA layer's merged in_proj."""
    reasons = k3._model_disable_reasons(model, hf_config, copies)
    if not wide.enabled():
        return reasons
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
    """Kimi-K3's launches at the narrow widths, then the wide launches of every
    bucket, compiled by running each once (``warmup``) before graph capture."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k31 import wide_check

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

    # an eager check of the wide layer launch on this model (off unless asked)
    wide_check.install(model)
    p = get_parallel()
    device = k3._backbone(model).layers[0].input_layernorm.weight.device
    m0 = mem()
    w = wide.Wide(model, p.tp_rank, p.tp_group.cpu_group, device, L=wide.tokens())
    m1 = mem()
    w.warmup()
    m2 = mem()
    torch.cuda.empty_cache()
    wide._say(
        f"memory: wide layers {used(m0, m1)}; warmup {used(m1, m2)}; after empty_cache {used(m0, mem())}"
    )
    return _Built(narrow, w)


def _devices():
    devices = dict(k3.SPEC.devices)
    if wide.enabled():
        limits = devices["gfx950"]
        devices["gfx950"] = msgspec.structs.replace(
            limits, widths=tuple(limits.widths) + wide.wide_widths()
        )
    return devices


SPEC = msgspec.structs.replace(
    k3.SPEC,
    name="kimi_k31_hip",
    devices=_devices(),
    # DCP (CONTEXT_PARALLEL) shards the DSA layers' KV, which run on SGLang's path
    features=k3.SPEC.features | {Feature.CONTEXT_PARALLEL},
    requires=msgspec.structs.replace(
        k3.SPEC.requires,
        attention_backends=frozenset({"dsa"}),
        # DSpark's and DFLASH's target steps are uniform verify steps of L rows a request
        server_args={
            **k3.SPEC.requires.server_args,
            "speculative_algorithm": frozenset({None, "DSPARK", "DFLASH"}),
        },
        hf_config={"model_type": "kimi_k31"},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    config_disable_reasons=_config_disable_reasons,
)
