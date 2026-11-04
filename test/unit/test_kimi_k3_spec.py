"""The Kimi-K3 gfx950 monokernel spec on CPU (RFC #43949): the layers its forward
covers on the Kimi-K3 layer structure, its registry entry, its per-output bounds
and its config refusal reasons. SGLang checks its declared requirements, the same
way for every spec."""

from types import SimpleNamespace

import pytest

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import HF_CONFIG, LINEAR_ATTN_CONFIG, MAX_BLOCKS, MAX_ROWS
from sgl_amd_mono.kimi_k3 import spec as k3

# Kimi-K3's layer structure (its config.json): 93 layers, a dense layer 0, then
# the KDA / full-attention cadence. Every layer past the dense one runs a MoE.
FULL_ATTN = list(range(4, 93, 4))
KDA = [i for i in range(1, 93) if i not in FULL_ATTN]
TEXT = SimpleNamespace(
    num_hidden_layers=93,
    attn_res_block_size=12,
    linear_attn_config={**LINEAR_ATTN_CONFIG, "full_attn_layers": FULL_ATTN + [93],
                        "kda_layers": KDA},
    **HF_CONFIG,
)  # fmt: skip
K3 = SimpleNamespace(
    model_type="kimi_k3",
    architectures=["KimiK3ForConditionalGeneration"],
    text_config=TEXT,
)


def test_the_forward_covers_every_moe_layer():
    # the dense layer 0 keeps the model's own forward; every other layer is a
    # MoE layer and takes the launch
    assert k3._covered_layers(K3) == list(range(1, 93))


def test_coverage_is_not_an_attention_kind():
    # The launch starts after the attention, so both kinds are covered. A spec
    # that declared only the KDA layers would still serve and still be correct,
    # just slower -- which is why this is pinned.
    covered = set(k3._covered_layers(K3))
    assert set(KDA) <= covered and set(FULL_ATTN) <= covered
    assert len(covered) == 92


def test_the_dense_prefix_and_the_moe_cadence_are_read_from_the_config():
    dense = SimpleNamespace(**{**vars(TEXT), "first_k_dense_replace": 4})
    assert k3._covered_layers(SimpleNamespace(text_config=dense)) == list(range(4, 93))
    every_other = SimpleNamespace(**{**vars(TEXT), "moe_layer_freq": 2})
    assert k3._covered_layers(SimpleNamespace(text_config=every_other)) == list(
        range(2, 93, 2)
    )


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["KimiK3ForConditionalGeneration"],
        isa="gfx950",
    )
    assert found is k3.SPEC, why


def test_the_spec_is_tp8_and_at_most_eight_rows():
    limits = k3.SPEC.devices["gfx950"]
    assert limits.tp_sizes == (8,)
    assert limits.widths == tuple(range(1, MAX_ROWS + 1))
    # With DSpark the runtime serves only the widths that hold whole requests,
    # so block size 7 (8 tokens a request) leaves batch size 1.
    assert [w // 8 for w in limits.widths if w % 8 == 0] == [1]


def test_the_shapes_are_checked_on_the_text_config():
    # The architecture names the VLM wrapper, whose top-level config carries
    # none of the shapes, so requires.hf_config cannot reach them.
    assert "hidden_size" not in k3.SPEC.requires.hf_config
    assert k3.SPEC.requires.hf_config == {"model_type": "kimi_k3"}
    wrong = SimpleNamespace(**{**vars(TEXT), "num_experts_per_token": 8})
    reasons = k3._config_disable_reasons(None, SimpleNamespace(text_config=wrong))
    assert any("num_experts_per_token" in r for r in reasons), reasons


def test_a_block_size_past_the_launch_is_refused():
    # The bank rows a layer's seam reads grow with the layer index; at block
    # size 12 the last layer reads exactly MAX_BLOCKS of them.
    tight = SimpleNamespace(**{**vars(TEXT), "attn_res_block_size": 8})
    reasons = k3._config_disable_reasons(None, SimpleNamespace(text_config=tight))
    assert any("attention-residual blocks" in r for r in reasons), reasons
    ok = k3._config_disable_reasons(None, K3)
    assert not any("attention-residual blocks" in r for r in ok), ok


def test_the_bounds_name_the_launches_and_their_outputs():
    for name in (
        "k2.prefix.cosine",
        "k2.x.cosine",
        "k2.out.cosine",
        "moe.out.cosine",
        "k1.ssm.cosine",
        "k1.ssm_bf16.cosine",
        "k1.ring_v.cosine",
        "k1.ring_k.cosine",
        "k1.ring_g.cosine",
        "k1.ring_beta.cosine",
    ):
        assert 0.0 < k3.SPEC.numerics.bound(name) <= 1.0
    with pytest.raises(KeyError):
        k3.SPEC.numerics.bound("k2.nothing.cosine")


def test_k1_builds_for_the_ssm_pools_dtype():
    """K1 reads and writes SGLang's SSM pool in place, so its build has to take
    the dtype the pool was allocated in: bf16 or fp32, and nothing else."""
    import torch

    from sgl_amd_mono.kimi_k3.gfx950 import k1

    for dtype in (torch.float32, torch.bfloat16):
        params = SimpleNamespace(dtype=SimpleNamespace(temporal=dtype))
        text = SimpleNamespace(mamba2_cache_params=params)
        assert k1.ssm_dtype(SimpleNamespace(text_config=text)) is dtype
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            use_full_rank_gate=True,
            _qkvgbfa_layer=object(),
            qkv_conv1d=SimpleNamespace(weight=torch.zeros(4, 1, 4)),
        )
    )
    assert k1._unsupported_reasons(layer, torch.float32) == []
    assert k1._unsupported_reasons(layer, torch.bfloat16) == []
    reasons = k1._unsupported_reasons(layer, torch.float16)
    assert any("--mamba-ssm-dtype" in r for r in reasons), reasons


def test_k1_takes_the_replayssm_ring_only_in_its_ring_build():
    """Under --enable-linear-replayssm-spec the pool has the ReplaySSM rings and
    no intermediate_ssm. The ring build takes a step only with rings of the
    layout it writes and long enough for the step; the snapshot build never."""
    pytest.importorskip("flydsl")
    pytest.importorskip("aiter")
    import torch

    from sgl_amd_mono.kimi_k3.gfx950 import k1
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    qlen, layers, slots = 6, 2, 3

    def batch(ring):
        key = kda.KdaPreBuild(
            tokens=qlen,
            qlen=qlen,
            nblocks=1,
            delta=False,
            state_len=qlen + kda.CONV_W - 1,
            ring=ring,
        )
        return SimpleNamespace(layers=[SimpleNamespace(key=key)])

    def step(ring_len=16, g_dtype=torch.float32):
        head = (layers, slots, kda.NH, ring_len)
        return {
            "rows": torch.zeros(1, dtype=torch.long),
            "slots": torch.zeros(1, dtype=torch.int32),
            "num_draft": qlen,
            "ring": True,
            "inter_ssm": None,
            "rings": (
                torch.empty(*head, kda.HD, dtype=torch.bfloat16),
                torch.empty(*head, kda.HD, dtype=torch.bfloat16),
                torch.empty(*head, kda.HD, dtype=g_dtype),
                torch.empty(*head, dtype=torch.float32),
            ),
            "committed_ssm": torch.empty(layers, slots, kda.NH, kda.HD, kda.HD),
            "committed_conv": torch.empty(
                layers, slots, kda.CONV_W - 1, 3 * kda.PROJ, dtype=torch.bfloat16
            ),
        }

    assert k1._step_shapes_ok(batch(True), step())
    assert not k1._step_shapes_ok(batch(False), step())
    assert not k1._step_shapes_ok(batch(True), step(ring_len=qlen - 1))
    assert not k1._step_shapes_ok(batch(True), step(g_dtype=torch.bfloat16))


def test_the_build_needs_a_real_model():
    with pytest.raises((AttributeError, TypeError)):
        k3.SPEC.build(None, K3, None)


def test_max_blocks_matches_the_reference_config():
    # layer 92 at block size 12 reads 8 blocks: cdiv(92, 12) = 8, and 92 is not
    # a block-write layer. The launch is built for exactly that many.
    last = max(k3._covered_layers(K3))
    block = TEXT.attn_res_block_size
    assert -(-last // block) + int(last % block == 0) == MAX_BLOCKS


def _fake_layer(*, gate=True, kda=False):
    """A covered layer as model_disable_reasons reads it, everything passing."""
    import torch

    def lin(*shape):
        return SimpleNamespace(
            weight=torch.zeros(*shape, dtype=torch.bfloat16), bias=None
        )

    attn = SimpleNamespace(
        o_proj=lin(HF_CONFIG["hidden_size"], 1536),
        forward_pre_o_proj=lambda **kw: None,
    )
    if not kda:  # only the MLA layers carry the gate
        attn.use_output_gate = gate
    experts = SimpleNamespace(
        w13_weight=SimpleNamespace(is_shuffled=True),
        w2_weight=SimpleNamespace(is_shuffled=True),
    )
    up = SimpleNamespace(
        weight=torch.zeros(HF_CONFIG["hidden_size"], 8, dtype=torch.bfloat16)
    )
    return SimpleNamespace(
        self_attn=attn,
        mlp=SimpleNamespace(
            use_latent_moe=True,
            routed_expert_norm=object(),
            routed_expert_up_proj=up,
            experts=experts,
        ),
        use_attn_residuals=True,
        _sp_moe=False,
        all_reduce_fusion=False,
    )


def _reasons(monkeypatch, layer):
    import sglang.srt.utils.common as common

    monkeypatch.setattr(common, "require_mlp_sync", lambda: False)
    model = SimpleNamespace(model=SimpleNamespace(layers={i: layer for i in range(93)}))
    return k3._model_disable_reasons(model, K3, {})


def test_a_covered_layer_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer()) == []
    assert _reasons(monkeypatch, _fake_layer(kda=True)) == []


def test_an_mla_layer_without_the_output_gate_is_refused(monkeypatch):
    # hasattr(attn, "forward_pre_o_proj") is true for both attention classes,
    # so it cannot carry this: for MLA the split rides on the gate's wrapper.
    reasons = _reasons(monkeypatch, _fake_layer(gate=False))
    assert any("output gate" in r for r in reasons), reasons


def test_a4w4_experts_are_the_plain_layout(monkeypatch):
    # Under AITER_SITUV2_A4W4, Mxfp4MoEMethod shuffles with shuffle_weight /
    # shuffle_scale, not the a16w4 pair the launch reads with guint=False. The
    # bytes agree for Kimi-K3's TP8 shapes; this fails if aiter changes either.
    pytest.importorskip("aiter.ops.shuffle")
    import torch

    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import LAT, RI

    monkeypatch.setenv("AITER_SITUV2_A8W4", "0")
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    layer = _fake_layer()
    layer.mlp.experts = SimpleNamespace(
        w13_weight=torch.zeros(1, 2 * RI, LAT // 2, dtype=torch.uint8),
        w2_weight=torch.zeros(1, LAT, RI // 2, dtype=torch.uint8),
    )
    layer.mlp.experts.w13_weight.is_shuffled = True
    layer.mlp.experts.w2_weight.is_shuffled = True
    assert k3._a4w4_layout_is_plain(layer.mlp.experts)
    assert _reasons(monkeypatch, layer) == []


def test_the_guard_flags_sit_past_every_width(monkeypatch):
    """Every width reads the guard flags at the same offset.

    The runner allocates one peer buffer for all the widths and puts the flags
    after the data. A launch that derived the offset from its own width wrote
    its flags inside the data region of a wider launch. Two widths in one run
    then disagreed on where the flags are, and the narrower one read live data
    as an error word.
    """
    from sgl_amd_mono.kimi_k3.gfx950 import layer
    from sgl_amd_mono.kimi_k3.gfx950.stages import moe

    off = layer.flags_offset()
    for width in range(1, MAX_ROWS + 1):
        assert layer.peer_bytes(width) <= off, width
        assert moe.peer_bytes(width) <= off, width
    # the widest launch ends exactly there: a larger offset wastes a peer page
    assert off == max(layer.peer_bytes(MAX_ROWS), moe.peer_bytes(MAX_ROWS))
