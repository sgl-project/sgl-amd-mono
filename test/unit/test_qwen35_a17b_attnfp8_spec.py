"""The Qwen3.5 AttnFP8 gfx950 monokernel spec on CPU: registry entry and the
verify-step config it accepts."""

from types import SimpleNamespace

import torch
from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.qwen35_a17b_attnfp8 import HF_CONFIG
from sgl_amd_mono.qwen35_a17b_attnfp8 import spec as qwen

QWEN = SimpleNamespace(**HF_CONFIG)


def _cfg(**kw):
    base = dict(
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        enable_linear_replayssm=False,
        enforce_shared_experts_fusion=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_plugin_registers_the_attnfp8_spec():
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        isa="gfx950",
    )
    assert found is qwen.SPEC, why
    assert found.devices["gfx950"].tp_sizes == (4,)
    assert found.devices["gfx950"].widths == tuple(range(1, 9))
    causal, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["Qwen3_5MoeForCausalLM"],
        isa="gfx950",
    )
    assert causal is not qwen.SPEC, why


def test_config_accepts_the_eagle_verify_step(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(
        mamba_utils,
        "mamba2_state_dtype",
        lambda config: SimpleNamespace(temporal=torch.bfloat16, conv=torch.bfloat16),
    )
    assert qwen.SPEC.config_disable_reasons(_cfg(), QWEN) == []
    reasons = qwen.SPEC.config_disable_reasons(_cfg(speculative_eagle_topk=2), QWEN)
    assert reasons and "topk" in reasons[0]
