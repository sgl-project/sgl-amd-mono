"""The Qwen3.5 AttnFP8 gfx950 monokernel spec on CPU: the family registry
entry and the verify-step config it accepts."""

from types import SimpleNamespace

import torch
from sglang.srt.monokernels import spec as registry
from sglang.srt.monokernels.spec import Feature

import sgl_amd_mono
from sgl_amd_mono.qwen35_a17b import spec as family
from sgl_amd_mono.qwen35_a17b.mxfp4 import HF_CONFIG
from sgl_amd_mono.qwen35_a17b.mxfp4 import spec as qwen

QWEN = SimpleNamespace(**HF_CONFIG)
ARCH = ["Qwen3_5MoeForConditionalGeneration"]


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


def _quark():
    return SimpleNamespace(
        text_config=QWEN,
        quantization_config={
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "group_size": 32, "scale_format": "e8m0"}
            },
        },
    )


def _mamba(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(
        mamba_utils,
        "mamba2_state_dtype",
        lambda config: SimpleNamespace(temporal=torch.bfloat16, conv=torch.bfloat16),
    )


def test_the_plugin_registers_the_family_spec():
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=ARCH,
        isa="gfx950",
    )
    assert found is family.SPEC, why
    assert found is not qwen.SPEC
    assert found.devices["gfx950"].tp_sizes == (4, 8)
    assert qwen.SPEC.devices["gfx950"].tp_sizes == (4,)
    assert found.devices["gfx950"].widths == tuple(range(1, 9))
    assert Feature.SPECULATIVE_DECODING in found.features
    assert Feature.SPECULATIVE_DECODING in qwen.SPEC.features
    causal, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["Qwen3_5MoeForCausalLM"],
        isa="gfx950",
    )
    assert causal is not family.SPEC, why


def test_config_accepts_the_eagle_verify_step(monkeypatch):
    _mamba(monkeypatch)
    assert qwen.SPEC.config_disable_reasons(_cfg(), QWEN) == []
    reasons = qwen.SPEC.config_disable_reasons(_cfg(speculative_eagle_topk=2), QWEN)
    assert reasons and "topk" in reasons[0]


def test_family_serves_quark_mxfp4_at_tp4_and_refuses_tp8(monkeypatch):
    _mamba(monkeypatch)
    hf = _quark()
    assert family.SPEC.config_disable_reasons(_cfg(), hf) == []
    assert family.SPEC.config_disable_reasons(_cfg(tp_size=4), hf) == []
    refused = family.SPEC.config_disable_reasons(_cfg(tp_size=8), hf)
    assert refused and "tp_size" in refused[0], refused


def _server(**kw):
    base = dict(
        pp_size=1,
        nnodes=1,
        enable_dp_attention=False,
        attn_dp_size=1,
        ep_size=1,
        enable_eplb=False,
        attn_cp_size=1,
        dcp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm=None,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_lora=False,
        lora_paths=None,
        disaggregation_mode="null",
        enable_return_routed_experts=False,
        enable_return_indexer_topk=False,
        enable_hierarchical_cache=False,
        disaggregation_decode_enable_offload_kvcache=False,
        enable_lmcache=False,
        enable_flexkv=False,
        enable_memory_saver=False,
        enable_weights_cpu_backup=False,
        cpu_offload_gb=0,
        offload_group_size=0,
        enable_torch_compile=False,
        tp_size=4,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_family_refuses_speculative_decoding_for_block_fp8():
    hf = SimpleNamespace(
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    plain = " | ".join(family.SPEC.config_disable_reasons(_server(), hf))
    assert "speculative decoding" not in plain, plain
    eagle = " | ".join(
        family.SPEC.config_disable_reasons(_server(speculative_algorithm="EAGLE"), hf)
    )
    assert "speculative decoding" in eagle, eagle
    assert "qwen35_a17b_fp8" in eagle, eagle
