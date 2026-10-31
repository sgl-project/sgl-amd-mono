"""The DeepSeek-V4.1 gfx950 monokernel spec on CPU (RFC #43949).

The layers its forward covers on DeepSeek-V4.1-Flash, its registry entry, its per-output bounds
and its model refusal reasons; SGLang checks its declared requirements, as for every spec."""

from types import SimpleNamespace

import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.deepseek_v41 import CANDIDATES, HF_CONFIG, draft
from sgl_amd_mono.deepseek_v41 import spec as dsv41

# DeepSeek-V4.1-Flash's layer structure (its config.json)
FLASH = SimpleNamespace(
    num_hidden_layers=40,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    engram_layer_ids=[1, 14],
    **HF_CONFIG,
    model_type="deepseek_v41",
    architectures=["DeepseekV4ForCausalLM"],
)


def test_the_forward_covers_every_flash_layer():
    # the first layer, the Engram layers, the index sources and DSpark's target layers (37-39) too
    assert dsv41._covered_layers(FLASH) == list(range(40))


def test_other_layers_have_no_kernels():
    hashed = SimpleNamespace(**{**vars(FLASH), "num_hash_layers": 4})
    assert dsv41._covered_layers(hashed) == list(range(4, 40))  # hash-routed MoE
    ratio4 = SimpleNamespace(**{**vars(FLASH), "compress_ratios": [4] * 43})
    assert dsv41._covered_layers(ratio4) == []  # another compress ratio


def test_the_plugin_registers_both_specs():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    for arch, spec in (
        ("DeepseekV4ForCausalLM", dsv41.SPEC),
        ("DeepseekV4ForCausalLMDSpark", draft.SPEC),
    ):
        found, why = registry.find_spec(
            backend=sgl_amd_mono.BACKEND,
            architectures=[arch],
            isa="gfx950",
            model_type="deepseek_v41",
        )
        assert found is spec, why


def test_kv_records_are_bit_exact():
    # the normal path (a prefill, a batch past the largest width) reads the KV records K1 writes
    for spec in (dsv41.SPEC, draft.SPEC):
        assert spec.numerics.bound("k1.kv.max_abs") == 0.0


def _layer(i, **attn):
    attn = {
        "compressor": None,
        "indexer": None,
        "q_head_norm": False,
        "fuse_wqa_wkv": True,
        "n_local_heads": 16,
        "compress_ratio": 0 if i < 2 else 2 if i < 20 else 1,
        **{lin: SimpleNamespace() for lin in ("wqkv_a", "wq_b", "wo_a", "wo_b")},
        **attn,
    }
    experts = SimpleNamespace(w13_weight=SimpleNamespace(is_shuffled=True))
    return SimpleNamespace(
        hc_boundary_fused=True,
        engram=None,
        self_attn=SimpleNamespace(**attn),
        mlp=SimpleNamespace(is_nextn=False, is_hash=False, experts=experts),
    )


def test_model_reasons(monkeypatch):
    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_parallel", lambda: SimpleNamespace(tp_size=4))
    layers = [_layer(i) for i in range(40)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    seen = {
        ("shared expert fp8", i, p, k): None
        for i in range(40)
        for p in dsv41._SHARED
        for k in ("weight", "scale")
    }
    assert dsv41.SPEC.model_disable_reasons(model, FLASH, seen) == []
    # wo_b shuffled in place (layer 5), a compressor at ratio 0 (layer 1), no load hook (layer 7)
    layers[5] = _layer(5, wo_b=SimpleNamespace(aiter_bpreshuffled=True))
    layers[1] = _layer(1, compress_ratio=0, compressor=SimpleNamespace())
    seen = {k: v for k, v in seen.items() if k[1] != 7}
    reasons = dsv41.SPEC.model_disable_reasons(model, FLASH, seen)
    assert len(reasons) == 3, reasons
    assert reasons[0].startswith("layer 1: a compressor or indexer at compress ratio 0")
    assert reasons[1].startswith("layer 5: an FP8 linear method reordered")
    assert "(layers [7])" in reasons[2]
    # the forward runs every layer, so one without kernels refuses the model
    hashed = SimpleNamespace(**{**vars(FLASH), "num_hash_layers": 2})
    reasons = dsv41.SPEC.model_disable_reasons(model, hashed, seen)
    assert reasons[0].startswith("layers [0, 1]: no kernels"), reasons
    # K2i's indexers: the candidate blocks REINDEX layers score, SGLang's decode-row query path
    layers[1], layers[5] = _layer(1), _layer(5)
    seen |= {
        ("shared expert fp8", 7, p, k): None
        for p in dsv41._SHARED
        for k in ("weight", "scale")
    }
    rope = SimpleNamespace(dtype=torch.complex64)
    ix = SimpleNamespace(
        is_candidate_source=True,
        uses_candidates=False,
        weights_proj_hip_max_tokens=64,
        **CANDIDATES,
    )
    layers[20] = _layer(20, indexer=ix, freqs_cis=rope)
    assert dsv41.SPEC.model_disable_reasons(model, FLASH, seen) == []
    other_blocks = SimpleNamespace(**{**vars(ix), "candidate_topk_blocks": 1024})
    layers[20] = _layer(20, indexer=other_blocks, freqs_cis=rope)
    no_gemv = SimpleNamespace(**{**vars(ix), "weights_proj_hip_max_tokens": -1})
    layers[24] = _layer(24, indexer=no_gemv, freqs_cis=rope)
    reasons = dsv41.SPEC.model_disable_reasons(model, FLASH, seen)
    assert [r.split(":")[0] for r in reasons] == ["layer 20", "layer 24"], reasons


def test_pd_runs_on_a_mori_decode_server_only():
    def reasons(mode, backend="mori"):
        cfg = SimpleNamespace(
            disaggregation_mode=mode, disaggregation_transfer_backend=backend
        )
        return [
            r for r in dsv41._config_disable_reasons(cfg, FLASH) if "FlyDSL" not in r
        ]

    pd = registry.Feature.PD_DISAGGREGATION
    assert pd in dsv41.SPEC.features and pd in draft.SPEC.features
    assert reasons("null") == reasons(None) == []
    assert reasons("decode") == []
    assert reasons("prefill") == ["PD monokernels support the decode role only"]
    assert reasons("decode", "mooncake") == [
        "PD monokernels require the mori transfer backend"
    ]


def _unified(monkeypatch, triton: bool, fp8: bool):
    from sglang.srt.layers.attention.dsv4 import unified_kv_gate as gate

    monkeypatch.setattr(gate, "is_unified_kv_triton", lambda: triton)
    monkeypatch.setattr(gate, "is_unified_kv_fp8", lambda: fp8)


def test_unified_kv_config_reasons(monkeypatch):
    cfg = SimpleNamespace(
        disaggregation_mode="null", disaggregation_transfer_backend="mori"
    )

    def kv_reasons():
        reasons = dsv41._config_disable_reasons(cfg, FLASH)
        return [r for r in reasons if "unified" in r]

    _unified(monkeypatch, triton=False, fp8=False)
    assert kv_reasons() == []
    # the FP8 two-pool rows K1 writes and K2 reads
    _unified(monkeypatch, triton=True, fp8=True)
    assert kv_reasons() == []
    _unified(monkeypatch, triton=True, fp8=False)
    assert kv_reasons() and "SGLANG_DSV4_UNIFIED_KV_FP8" in kv_reasons()[0]
    # an SGLang without the pool's compressed row base (PENDING_POOL_READS)
    from sglang.srt.mem_cache import deepseek_v4_memory_pool as pools

    monkeypatch.delattr(pools.DeepSeekV4TokenToKVPool, "get_unified_compress_row_base")
    _unified(monkeypatch, triton=True, fp8=True)
    assert kv_reasons() and "get_unified_compress_row_base" in kv_reasons()[0]


def test_unified_kv_row_layout():
    # K1 / K2 address SGLang's FP8 unified rows by these sizes
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import layout

    from sgl_amd_mono.deepseek_v41.gfx950.attention import plan

    assert plan.UNI_ROW == layout.DSV4_FP8_NOPE_ROW_BYTES
    assert plan.NOPE // layout.DSV4_FP8_QUANT_TILE == 7
    # the doubled codes end inside the row, 16 B aligned for K2's one load
    assert plan.NOPE % 16 == 0 and plan.NOPE + 16 <= plan.UNI_ROW
