# SPDX-License-Identifier: Apache-2.0
"""The monokernel spec of Qwen3.5-397B-A17B (``Qwen3_5MoeForConditionalGeneration``).

SGLang registers one spec per architecture. Dynamic 128x128 block-FP8 is ``fp8``
(TP4 and TP8 decode). Quark MXFP4 with per-channel FP8 projections is ``mxfp4``
(TP4 decode and verify). This spec selects the checkpoint's own path. Device
limits and features are the union; a path that does not accept the server's TP
or a feature still refuses in ``config_disable_reasons``.
"""

from __future__ import annotations

from sglang.srt.monokernels.spec import DeviceLimits, MonoSpec, Requirements

from sgl_amd_mono.qwen35_a17b.config import is_block_fp8, is_quark_mxfp4, quant_of
from sgl_amd_mono.qwen35_a17b.fp8 import spec as fp8
from sgl_amd_mono.qwen35_a17b.mxfp4 import spec as mxfp4

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/monokernels/spec.py": ("DeviceLimits", "MonoSpec", "Requirements"),
    "srt/monokernels/guards.py": ("features_in_use",),
}


def _child(hf_config):
    if is_block_fp8(hf_config):
        return fp8
    if is_quark_mxfp4(hf_config):
        return mxfp4
    return None


def _unsupported(hf_config) -> list[str]:
    method = quant_of(hf_config).get("quant_method")
    return [
        "Qwen3.5 family serves dynamic 128x128 block-FP8 or Quark MXFP4 AttnFP8; "
        f"quant_method {method!r} is not one of them"
    ]


def _child_limit_reasons(cfg, spec: MonoSpec) -> list[str]:
    """TP and features the family advertises for both paths, narrowed to ``spec``."""
    if cfg is None:
        return []
    reasons = []
    tp = getattr(cfg, "tp_size", None)
    if tp is not None:
        allowed = spec.devices["gfx950"].tp_sizes
        if tp not in allowed:
            reasons.append(
                f"tp_size {tp!r} is not one of {sorted(allowed)} for {spec.name}"
            )
    # A real server config has pp_size. The unit fixtures that only set the fields
    # a child spec reads do not, and the guard already checked the family's union.
    if hasattr(cfg, "pp_size"):
        from sglang.srt.monokernels.guards import features_in_use

        reasons += [
            f"{feature.value} is in use and {spec.name} does not accept it"
            for feature in sorted(
                features_in_use(cfg) - spec.features, key=lambda item: item.name
            )
        ]
    return reasons


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    child = _child(hf_config)
    if child is None:
        return _unsupported(hf_config)
    reasons = _child_limit_reasons(cfg, child.SPEC)
    if child is fp8:
        fp8._install_block_fp8_linear()
    if child.SPEC.config_disable_reasons is not None:
        reasons += child.SPEC.config_disable_reasons(cfg, hf_config)
    return reasons


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    child = _child(hf_config)
    if child is None:
        return _unsupported(hf_config)
    if child is fp8:
        fp8._install_block_fp8_linear()
    return child.SPEC.model_disable_reasons(model, hf_config, copies)


def _build(model, hf_config, ctx):
    child = _child(hf_config)
    if child is None:
        method = quant_of(hf_config).get("quant_method")
        raise ValueError(
            f"Qwen3.5 family has no monokernel for quant_method {method!r}"
        )
    if child is fp8:
        fp8._install_block_fp8_linear()
    return child.SPEC.build(model, hf_config, ctx)


def _devices() -> dict[str, DeviceLimits]:
    limits = [module.SPEC.devices["gfx950"] for module in (fp8, mxfp4)]
    return {
        "gfx950": DeviceLimits(
            widths=tuple(sorted({w for item in limits for w in item.widths})),
            tp_sizes=tuple(sorted({tp for item in limits for tp in item.tp_sizes})),
            resident_blocks=max(item.resident_blocks for item in limits),
        )
    }


SPEC = MonoSpec(
    name="qwen35_a17b",
    devices=_devices(),
    features=fp8.SPEC.features | mxfp4.SPEC.features,
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={"moe_runner_backend": frozenset({"auto", "aiter"})},
    ),
    numerics=fp8.SPEC.numerics,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,
    config_disable_reasons=_config_disable_reasons,
)
