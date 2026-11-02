"""The MiniMax-M3 gfx950 monokernel spec on CPU (RFC #43949).

Checks its registry entries, the configurations it refuses before load and the verify
widths it serves on the EAGLE3 recipe. The layer views it refuses after load are pinned
per TP size in test_minimax_m3_weights.py. SGLang checks the declared requirements, the
same way for every spec."""

from types import SimpleNamespace
from unittest import mock

import pytest

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry
try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

import sgl_amd_mono
from sgl_amd_mono.minimax import spec as family
from sgl_amd_mono.minimax_m3 import spec as m3

# MiniMax-M3's text config (amd/MiniMax-M3-MXFP4 config.json, text_config)
TEXT = SimpleNamespace(
    hidden_size=6144,
    num_hidden_layers=60,
    num_local_experts=128,
    num_experts_per_tok=4,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    scoring_func="sigmoid",
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    rms_norm_eps=1e-6,
    sparse_attention_config={
        "sparse_topk_blocks": 16,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_block_size": 128,
        "sparse_score_type": "max",
    },
)
VL = SimpleNamespace(
    text_config=TEXT, architectures=["MiniMaxM3SparseForConditionalGeneration"]
)
# the EAGLE3 recipe's server arguments the spec reads
CFG = SimpleNamespace(
    tp_size=4,
    speculative_algorithm="EAGLE3",
    speculative_eagle_topk=1,
    speculative_num_steps=3,
    speculative_num_draft_tokens=4,
    enable_hierarchical_cache=True,
    hicache_io_backend="kernel",
    hicache_mem_layout="layer_first",
    enable_lmcache=False,
    enable_flexkv=False,
    disaggregation_decode_enable_offload_kvcache=False,
)


def _replace(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def _config_reasons(cfg=CFG, hf_config=VL):
    with (
        mock.patch.object(m3, "_has_flydsl", return_value=True),
        mock.patch.object(m3, "_pending_env_reasons", return_value=[]),
    ):
        return m3.config_disable_reasons(cfg, hf_config)


def test_the_plugin_registers_both_architectures():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    for arch in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ):
        found, why = registry.find_spec(
            backend=sgl_amd_mono.BACKEND, architectures=[arch], isa="gfx950"
        )
        # the architecture's spec, which MiniMax-M3.1 shares, hands M3 to this one
        assert found is family.SPEC, why
    assert family._model(VL) is m3


def test_the_recipe_is_accepted():
    assert _config_reasons() == []
    assert _config_reasons(hf_config=TEXT) == []  # the text-only checkpoint
    plain = _replace(CFG, speculative_algorithm=None, enable_hierarchical_cache=False)
    assert _config_reasons(cfg=plain) == []


@pytest.mark.parametrize(
    "cfg, text",
    [
        (_replace(CFG, speculative_eagle_topk=2), TEXT),
        (_replace(CFG, speculative_num_draft_tokens=3), TEXT),
        (_replace(CFG, hicache_io_backend="direct"), TEXT),
        (_replace(CFG, hicache_mem_layout="page_first"), TEXT),
        (_replace(CFG, enable_lmcache=True), TEXT),
        (CFG, _replace(TEXT, hidden_size=5120)),
        (CFG, _replace(TEXT, swiglu_alpha=1.5)),
        (CFG, _replace(TEXT, num_hidden_layers=200)),
        (
            CFG,
            _replace(
                TEXT,
                sparse_attention_config={
                    **TEXT.sparse_attention_config,
                    "sparse_local_block": 2,
                },
            ),
        ),
    ],
    ids=[
        "tree draft",
        "draft tokens",
        "hicache io",
        "hicache layout",
        "lmcache",
        "hidden size",
        "swiglu alpha",
        "mailbox layers",
        "selection",
    ],
)
def test_each_unsupported_setting_is_one_reason(cfg, text):
    assert len(_config_reasons(cfg=cfg, hf_config=_replace(VL, text_config=text))) == 1


def test_an_sglang_without_the_shuffle_kv_is_refused_with_a_reason():
    # without sgl-project/sglang#41998 / #41708's switches: a refusal naming them, not a crash
    from sglang.srt.environ import envs

    present = {n: getattr(envs, n, None) for n in m3.PENDING_ENV}
    with mock.patch("sglang.srt.environ.envs", SimpleNamespace()):
        missing = m3._pending_env_reasons()
    assert len(missing) == len(m3.PENDING_ENV)
    assert all("sglang#41998" in r for r in missing)
    if all(v is not None for v in present.values()):
        # an SGLang with them: the defaults are what the kernels need
        assert m3._pending_env_reasons() == []


def test_the_recipe_routes_verify_batches_of_one_to_four():
    # 4 rows a request: batch sizes 1-4 (4 to 16 rows) join the configured ones (8: the normal path)
    assert m3.SPEC.widths("gfx950") == tuple(range(1, 17))
    assert covered_capture_sizes(
        widths=m3.SPEC.widths("gfx950"), tokens_per_request=4, sizes=[1, 2, 4, 8]
    ) == [1, 2, 3, 4, 8]


def test_tp8_defaults_the_unpadded_expert_layout(monkeypatch):
    # at TP8 the kernels need SGLANG_AITER_K3_OPT; unset, the spec turns it on for the workers
    monkeypatch.delenv(m3.K3_OPT_ENV, raising=False)
    assert _config_reasons(cfg=_replace(CFG, tp_size=8)) == []
    assert m3.os.environ[m3.K3_OPT_ENV] == "1"


def test_tp8_with_the_expert_layout_switched_off_is_one_reason(monkeypatch):
    monkeypatch.setenv(m3.K3_OPT_ENV, "0")
    reasons = _config_reasons(cfg=_replace(CFG, tp_size=8))
    assert len(reasons) == 1 and m3.K3_OPT_ENV in reasons[0]


@pytest.mark.parametrize("tp", [2, 4])
def test_tp2_and_tp4_leave_the_expert_layout_alone(monkeypatch, tp):
    monkeypatch.delenv(m3.K3_OPT_ENV, raising=False)
    assert _config_reasons(cfg=_replace(CFG, tp_size=tp)) == []
    assert m3.K3_OPT_ENV not in m3.os.environ
