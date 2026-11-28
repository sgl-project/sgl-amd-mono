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
    activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
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


def test_the_spec_is_tp8_and_covers_narrow_and_wide_rows():
    from sgl_amd_mono.kimi_k31 import wide

    limits = k3.SPEC.devices["gfx950"]
    assert limits.tp_sizes == (8,)
    assert limits.widths[:MAX_ROWS] == tuple(range(1, MAX_ROWS + 1))
    if k3.WIDE:
        assert limits.widths[MAX_ROWS:] == wide.wide_widths()
    else:
        assert limits.widths == tuple(range(1, MAX_ROWS + 1))


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
    for name in ("k2.prefix.cosine", "k2.x.cosine", "k2.out.cosine", "moe.out.cosine"):
        assert 0.0 < k3.SPEC.numerics.bound(name) <= 1.0
    with pytest.raises(KeyError):
        k3.SPEC.numerics.bound("k2.nothing.cosine")


def test_the_build_needs_a_real_model():
    with pytest.raises((AttributeError, TypeError)):
        k3.SPEC.build(None, K3, None)


def test_max_blocks_matches_the_reference_config():
    # layer 92 at block size 12 reads 8 blocks: cdiv(92, 12) = 8, and 92 is not
    # a block-write layer. The launch is built for exactly that many.
    last = max(k3._covered_layers(K3))
    block = TEXT.attn_res_block_size
    assert -(-last // block) + int(last % block == 0) == MAX_BLOCKS


def _fake_layer(*, gate=True, kda=False, o_proj=None, ri=384, act="situ"):
    """A covered layer as model_disable_reasons reads it, everything passing."""
    import torch

    def lin(*shape):
        return SimpleNamespace(
            weight=torch.zeros(*shape, dtype=torch.bfloat16), bias=None
        )

    attn = SimpleNamespace(
        o_proj=o_proj or lin(HF_CONFIG["hidden_size"], 1536),
        forward_pre_o_proj=lambda **kw: None,
    )
    if not kda:  # only the MLA layers carry the gate
        attn.use_output_gate = gate
    e, lat = HF_CONFIG["num_experts"], HF_CONFIG["routed_expert_hidden_size"]
    experts = SimpleNamespace(
        w13_weight=SimpleNamespace(is_shuffled=True, shape=(e, 2 * ri, lat // 2)),
        w2_weight=SimpleNamespace(is_shuffled=True, shape=(e, lat, ri // 2)),
        moe_runner_config=SimpleNamespace(activation=act),
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
    from sglang.srt.layers import dp_attention
    from sglang.srt.utils import common

    # where spec.py imports it from (common: SGLang before sgl-project/sglang#48781)
    source = dp_attention if hasattr(dp_attention, "require_mlp_sync") else common
    monkeypatch.setattr(source, "require_mlp_sync", lambda: False)
    model = SimpleNamespace(model=SimpleNamespace(layers={i: layer for i in range(93)}))
    return k3._model_disable_reasons(model, K3, {})


def test_require_mlp_sync_is_read_from_utils_common_on_an_older_sglang(monkeypatch):
    # Before sgl-project/sglang#48781 SGLang defines it in srt/utils/common.py and
    # not in dp_attention; the checks must still run there, and still refuse.
    from sglang.srt.layers import dp_attention
    from sglang.srt.utils import common

    monkeypatch.delattr(dp_attention, "require_mlp_sync", raising=False)
    model = SimpleNamespace(
        model=SimpleNamespace(layers={i: _fake_layer() for i in range(93)})
    )
    for on in (False, True):
        monkeypatch.setattr(common, "require_mlp_sync", lambda on=on: on, raising=False)
        reasons = k3._model_disable_reasons(model, K3, {})
        assert reasons == (["mlp-sync padding is on"] if on else []), reasons


def test_a_covered_layer_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer()) == []
    assert _reasons(monkeypatch, _fake_layer(kda=True)) == []


def test_an_mla_layer_without_the_output_gate_is_refused(monkeypatch):
    # hasattr(attn, "forward_pre_o_proj") is true for both attention classes,
    # so it cannot carry this: for MLA the split rides on the gate's wrapper.
    reasons = _reasons(monkeypatch, _fake_layer(gate=False))
    assert any("output gate" in r for r in reasons), reasons


# o_proj as SGLang leaves a Quark per-channel FP8 one: the [out, in]
# storage transposed for its GEMM, an fp32 scale a row.
def _fp8_o_proj(*, quark=True, dynamic=True):
    import torch
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    hidden = HF_CONFIG["hidden_size"]
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": dynamic}
    )
    return SimpleNamespace(
        weight=torch.zeros(hidden, 1536, dtype=torch.float8_e4m3fn).t(),
        weight_scale=torch.ones(hidden, 1, dtype=torch.float32),
        scheme=scheme if quark else object(),
        bias=None,
    )


def test_a_quark_fp8_o_proj_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer(o_proj=_fp8_o_proj())) == []


def test_a_static_input_fp8_o_proj_is_refused(monkeypatch):
    """A Quark FP8 o_proj with a static input scale must not be served: the launch
    skips that scale's clipping and rounding of the activation."""
    reasons = _reasons(monkeypatch, _fake_layer(o_proj=_fp8_o_proj(dynamic=False)))
    assert any("quantizes its input statically" in r for r in reasons), reasons


def test_an_fp8_o_proj_outside_quark_is_refused(monkeypatch):
    reasons = _reasons(monkeypatch, _fake_layer(o_proj=_fp8_o_proj(quark=False)))
    assert any("Quark per-channel FP8" in r for r in reasons), reasons


@pytest.mark.parametrize("shuffled", [True, False])
def test_the_fp8_o_proj_layout_is_the_one_sglang_chose(monkeypatch, shuffled):
    # SGLang preshuffles the FP8 weight exactly when its own predicate says so;
    # the launch must read the same layout, or it reads the bytes in a wrong order.
    import sglang.srt.layers.quantization.fp8_utils as fp8_utils

    from sgl_amd_mono.kimi_k3.gfx950.stages import oproj

    monkeypatch.setattr(fp8_utils, "use_aiter_bpreshuffle_gemm", lambda n: shuffled)
    fmt, storage, scale = k3._o_proj_operands(_fp8_o_proj())
    assert fmt == (oproj.O_FP8_SHUFFLED if shuffled else oproj.O_FP8)
    assert storage.is_contiguous() and tuple(storage.shape) == (
        HF_CONFIG["hidden_size"],
        1536,
    )
    assert tuple(scale.shape) == (HF_CONFIG["hidden_size"],)


@pytest.mark.parametrize("shuffled", [True, False])
def test_the_fp8_kda_in_proj_layout_is_the_one_sglang_chose(monkeypatch, shuffled):
    # SGLang's merged Quark FP8 KDA in-projection is fused_qkvg_proj's own weight,
    # padded to 6336 rows, with f_b kept as a bf16 copy. K1 must read the layout
    # SGLang stored, and that copy of f_b rather than the FP8 f_b_proj.
    import sglang.srt.layers.quantization.fp8_utils as fp8_utils
    import torch
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    from sgl_amd_mono.kimi_k3.gfx950 import k1
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    monkeypatch.setattr(fp8_utils, "use_aiter_bpreshuffle_gemm", lambda n: shuffled)
    rows = 6336
    attn = SimpleNamespace(
        use_full_rank_gate=True,
        _qkvgbfa_in_qkvg_proj=True,
        fused_qkvg_proj=SimpleNamespace(
            weight=torch.zeros(rows, kda.HIDDEN, dtype=torch.float8_e4m3fn).t(),
            weight_scale=torch.ones(rows, 1, dtype=torch.float32),
            scheme=QuarkW8A8Fp8(
                {"qscheme": "per_channel"},
                {"qscheme": "per_channel", "is_dynamic": True},
            ),
        ),
        _bfa_f_b_w=torch.zeros(kda.PROJ, kda.HD, dtype=torch.bfloat16),
    )
    fmt, w_in, scale, w_in_fab, w_fb = k1._in_proj_operands(attn)
    assert w_in_fab is None
    assert fmt == (kda.W_FP8_SHUFFLED if shuffled else kda.W_FP8)
    assert tuple(w_in.shape) == (rows, kda.HIDDEN) and w_in.is_contiguous()
    assert tuple(scale.shape) == (rows,)
    assert w_fb is attn._bfa_f_b_w


def test_k1_takes_a_bf16_or_fp32_state_only():
    # K1 builds for the state SGLang allocates, which --mamba-ssm-dtype reaches
    # through SGLANG_MAMBA_SSM_DTYPE; an env-only setting must pick the same variant.
    from sglang.srt.environ import envs

    from sgl_amd_mono.kimi_k3.gfx950 import k1

    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        assert k1._state_config() == (True, [])
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("float32"):
        assert k1._state_config() == (False, [])
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("float16"):
        assert "float32 or bfloat16" in k1._state_config()[1][0]


def test_padded_experts_are_refused(monkeypatch):
    # Without SGLANG_OPT_HIP_K3_AITER the intermediate is padded to 512.
    reasons = _reasons(monkeypatch, _fake_layer(ri=512))
    assert any("padded intermediate" in r for r in reasons), reasons


def test_the_separated_a4w4_expert_layout_is_not_refused(monkeypatch):
    # AITER_SITUV2_A4W4's preshuffle is byte for byte the launch's gate_up=False one.
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    monkeypatch.setenv("AITER_SITUV2_A8W4", "0")
    monkeypatch.setattr(k3, "_k3_aiter_opt", lambda: True)
    reasons = k3._config_disable_reasons(None, K3)
    assert not any("interleaved" in r or "K3_AITER" in r for r in reasons), reasons
    # and the launch is built for that separated layout, A8W4 for the interleaved one
    moe = SimpleNamespace(
        experts=SimpleNamespace(moe_runner_config=SimpleNamespace(activation="situ"))
    )
    assert k3._gu_interleaved(moe) is False
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")
    assert k3._gu_interleaved(moe) is True


def test_a_situ_the_launch_does_not_compute_is_refused(monkeypatch):
    """K2 computes SiTU with fixed (beta, linear_beta); a config with other values,
    here under the A4W4 layout, must not reach the launch."""
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    monkeypatch.setenv("AITER_SITUV2_A8W4", "0")
    monkeypatch.setattr(k3, "_k3_aiter_opt", lambda: True)
    text = SimpleNamespace(**{**vars(TEXT), "activation_situ_beta": 2.0})
    hf = SimpleNamespace(**{**vars(K3), "text_config": text})
    reasons = k3._config_disable_reasons(None, hf)
    assert any("SiTU (beta, linear_beta)" in r for r in reasons), reasons
    assert not any("SiTU" in r for r in k3._config_disable_reasons(None, K3))


def test_the_k3_aiter_switch_is_read_under_either_name(monkeypatch):
    """The launch needs the K3 AITER switch whether SGLang reads it through its Envs
    field or, before that field existed, through SGLANG_AITER_K3_OPT."""
    from types import SimpleNamespace

    import sglang.srt.environ as environ

    def blocked():
        return any("K3_AITER" in r for r in k3._config_disable_reasons(None, K3))

    monkeypatch.delenv("SGLANG_AITER_K3_OPT", raising=False)
    monkeypatch.setattr(environ, "envs", SimpleNamespace())
    assert blocked()
    monkeypatch.setenv("SGLANG_AITER_K3_OPT", "1")
    assert not blocked()

    monkeypatch.delenv("SGLANG_AITER_K3_OPT")
    for on in (False, True):
        field = SimpleNamespace(get=lambda on=on: on)
        monkeypatch.setattr(
            environ, "envs", SimpleNamespace(SGLANG_OPT_HIP_K3_AITER=field)
        )
        assert blocked() is not on


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


def test_experts_with_another_activation_are_refused(monkeypatch):
    """K2 computes SiTU; a SiLU MoE with otherwise valid shapes must not be served,
    whichever expert layout AITER's switches select."""
    reasons = _reasons(monkeypatch, _fake_layer(act="silu"))
    assert any("not SiTU" in r for r in reasons), reasons


def test_k1_declines_a_request_scoped_replayssm_ring(monkeypatch):
    """A request-indexed ReplaySSM ring (sgl-project/sglang#44718) must fall back to
    SGLang before K1 stages anything: K1 writes the ring by slot."""
    import torch
    from sglang.srt.model_executor import forward_context

    from sgl_amd_mono.kimi_k3.gfx950 import k1

    def run(request_ring):
        cache = SimpleNamespace(
            intermediate_ssm=None,
            intermediate_conv_window=[torch.zeros(1)],
            replayssm_rawv=torch.zeros(1),
        )
        pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=cache, replayssm_request_ring=request_ring
            )
        )
        backend = SimpleNamespace(req_to_token_pool=pool, forward_metadata=object())
        monkeypatch.setattr(
            forward_context,
            "get_attn_backend",
            lambda: SimpleNamespace(linear_attn_backend=backend),
        )
        batch = SimpleNamespace(idx=None, layers=[])
        fb = SimpleNamespace(spec_info=object())
        try:
            out = k1.prepare(batch, fb)
        except Exception:  # past the guard the stub runs out; staging had begun
            out = "staged"
        return out, batch.idx

    assert run(request_ring=True) == (None, None)
    assert run(request_ring=False) != (None, None)


@pytest.mark.parametrize("b_dtype", ["fp8", "bf16"])
def test_split_quark_fp8_in_proj_and_caching(monkeypatch, b_dtype):
    import sglang.srt.layers.quantization.fp8_utils as fp8_utils
    import torch
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    from sgl_amd_mono.kimi_k3.gfx950 import k1
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda
    from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt

    raw = torch.randint(0, 256, (32, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
    sh = wfmt.preshuffle_fp8(raw)
    assert torch.equal(wfmt.unshuffle_fp8(sh).view(torch.uint8), raw.view(torch.uint8))

    monkeypatch.setattr(fp8_utils, "use_aiter_bpreshuffle_gemm", lambda n: n >= 128)
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"},
        {"qscheme": "per_channel", "is_dynamic": True},
    )

    torch.manual_seed(7)

    def make_fp8(rows, cols):
        raw_q = (torch.randn(rows, cols) * 0.5).clamp(-448, 448).to(torch.float8_e4m3fn)
        sc = (torch.rand(rows, 1) * 0.05 + 0.01).to(torch.float32)
        return SimpleNamespace(
            weight=raw_q.t(),
            weight_scale=sc,
            scheme=scheme,
            _raw=raw_q,
        )

    b_proj = (
        make_fp8(kda.NH, kda.HIDDEN)
        if b_dtype == "fp8"
        else SimpleNamespace(
            weight=torch.randn(kda.NH, kda.HIDDEN, dtype=torch.bfloat16).t(),
            weight_scale=None,
            scheme=scheme,
        )
    )
    attn = SimpleNamespace(
        use_full_rank_gate=True,
        _qkvgbfa_in_qkvg_proj=False,
        _qkvgbfa_layer=None,
        _bfa_f_b_w=None,
        fused_qkvg_proj=make_fp8(kda.FA_0, kda.HIDDEN),
        f_a_proj=make_fp8(kda.HD, kda.HIDDEN),
        b_proj=b_proj,
        f_b_proj=make_fp8(kda.PROJ, kda.HD),
    )
    k1._validate_in_proj_operands(attn)
    assert not hasattr(attn, "_sgl_amd_mono_k1_ops")
    ops1 = k1._in_proj_operands(attn)
    ops2 = k1._in_proj_operands(attn)
    assert ops1 is ops2
    fmt, w_in, scale, w_in_fab, w_fb = ops1
    assert fmt == kda.W_FP8_SHUFFLED
    assert tuple(w_in.shape) == (kda.FA_0, kda.HIDDEN)
    assert tuple(w_in_fab.shape) == (kda.NPROJ - kda.FA_0, kda.HIDDEN)
    assert tuple(scale.shape) == (kda.NPROJ,)
    assert w_fb.dtype == torch.bfloat16 and tuple(w_fb.shape) == (kda.PROJ, kda.HD)
    unshuf_fab = wfmt.unshuffle_fp8(w_in_fab).float() * scale[kda.FA_0 :, None]
    ref_fa = wfmt.unshuffle_fp8(attn.f_a_proj._raw).float() * attn.f_a_proj.weight_scale
    assert torch.allclose(unshuf_fab[: kda.HD], ref_fa, atol=1e-4, rtol=1e-3)
    assert torch.count_nonzero(scale[kda.FA_0 + kda.HD + kda.NH :]) == 0
    assert torch.count_nonzero(unshuf_fab[kda.HD + kda.NH :]) == 0
    # b_proj's rows of the tail: Quark FP8 as stored, or bf16 re-quantized a row at a time
    got_b = unshuf_fab[kda.HD : kda.HD + kda.NH]
    s_b = scale[kda.FA_0 + kda.HD : kda.FA_0 + kda.HD + kda.NH]
    if b_dtype == "fp8":
        ref_b = attn.b_proj._raw.float() * attn.b_proj.weight_scale
        assert torch.equal(s_b, attn.b_proj.weight_scale.view(-1))
        assert torch.allclose(got_b, ref_b, atol=0, rtol=1e-6)
    else:
        ref_b = attn.b_proj.weight.t().float()
        amax = ref_b.abs().amax(dim=1)
        assert torch.allclose(s_b, amax / 448.0, rtol=1e-6)
        # E4M3 keeps 3 mantissa bits: within 1/16 of a value, or a subnormal step
        bound = ref_b.abs() / 16 + (amax / 448.0 * 2.0**-9)[:, None]
        assert bool(((got_b - ref_b).abs() <= bound).all())
    # f_b: the Quark FP8 weight dequantized once to bf16
    ref_fb = wfmt.unshuffle_fp8(attn.f_b_proj._raw).float() * attn.f_b_proj.weight_scale
    assert torch.equal(w_fb, ref_fb.to(torch.bfloat16))


_NO_OFFLOAD = dict(
    enable_lmcache=False,
    enable_flexkv=False,
    disaggregation_decode_enable_offload_kvcache=False,
)


def test_kv_offload_is_accepted_for_hicache_only(monkeypatch):
    """The spec declares KV_OFFLOAD for HiCache; the other offload paths it admits are
    refused, one reason each, through Kimi-K3's own config hook."""
    monkeypatch.setattr(k3, "WIDE", True)
    cfg = dict(
        dcp_size=8,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        cuda_graph_max_bs_decode=16,
        max_running_requests=16,
        enable_linear_replayssm_spec=True,
        **_NO_OFFLOAD,
    )
    assert k3.Feature.KV_OFFLOAD in k3.SPEC.features
    base = k3._config_disable_reasons(SimpleNamespace(**cfg), K3)
    assert not any("HiCache" in r for r in base), base
    # HiCache itself adds no refusal
    hicache = SimpleNamespace(**cfg, enable_hierarchical_cache=True)
    assert k3._config_disable_reasons(hicache, K3) == base
    for flag in _NO_OFFLOAD:
        reasons = k3._config_disable_reasons(SimpleNamespace(**{**cfg, flag: True}), K3)
        extra = [r for r in reasons if r not in base]
        assert len(extra) == 1 and "only HiCache" in extra[0], (flag, extra)


def test_the_host_only_ring_refusal_follows_the_widths_this_draft_length_serves(
    monkeypatch,
):
    """Kimi-K3 refuses a host-only ring only when a step can reach a wide width at its L:
    with the 128,256,384 override, 12 requests of 8 tokens never do (the first is 15)."""
    monkeypatch.setattr(k3, "WIDE", True)
    monkeypatch.setattr(k3, "_host_only_mamba", lambda: True)
    cfg = SimpleNamespace(
        dcp_size=8,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=8,
        cuda_graph_max_bs_decode=12,
        max_running_requests=12,
        enable_linear_replayssm_spec=True,
        **_NO_OFFLOAD,
    )
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    reasons = k3._config_disable_reasons(cfg, K3)
    assert any("request-scoped ReplaySSM" in r for r in reasons), reasons
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", "128,256,384")
    reasons = k3._config_disable_reasons(cfg, K3)
    assert not any("request-scoped ReplaySSM" in r for r in reasons), reasons
    # an eager server routes up to its running cap: 16 requests reach 120 rows
    eager = SimpleNamespace(**{**vars(cfg), "max_running_requests": 16})
    reasons = k3._config_disable_reasons(eager, K3)
    assert any("request-scoped ReplaySSM" in r for r in reasons), reasons


def test_kimi_k31_says_the_host_only_ring_refusal_once(monkeypatch):
    """Kimi-K3's hook refuses a host-only ring on a wide ReplaySSM server and Kimi-K3.1's
    on any server; with both hooks firing the refusal is listed once."""
    from sgl_amd_mono.kimi_k31 import spec as k31

    monkeypatch.setattr(k3, "WIDE", True)
    monkeypatch.setattr(k3, "_host_only_mamba", lambda: True)
    monkeypatch.setenv("SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT", "1")
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    cfg = SimpleNamespace(
        dcp_size=8,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        cuda_graph_max_bs_decode=16,
        max_running_requests=16,
        enable_linear_replayssm_spec=True,
        **_NO_OFFLOAD,
    )
    assert any("request-scoped" in r for r in k3._config_disable_reasons(cfg, K3))
    reasons = k31._config_disable_reasons(cfg, K3)
    ring = [r for r in reasons if "SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT" in r]
    assert len(ring) == 1 and "mamba slot" in ring[0], reasons


def test_wide_config_guards_draft_tokens_and_request_scoped_ring(monkeypatch):
    monkeypatch.setattr(k3, "WIDE", True)
    bad_decode = SimpleNamespace(
        dcp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm=None,
        speculative_num_draft_tokens=None,
        cuda_graph_max_bs_decode=16,
        max_running_requests=16,
        enable_linear_replayssm_spec=False,
        **_NO_OFFLOAD,
    )
    reasons = k3._config_disable_reasons(bad_decode, K3)
    assert any("SGLANG_AMD_MONO_K3_WIDE=0" in r for r in reasons), reasons
    # an unsupported draft length past MAX_ROWS rows is refused even with one request
    ten = SimpleNamespace(
        **{
            **vars(bad_decode),
            "speculative_algorithm": "DSPARK",
            "speculative_num_draft_tokens": 10,
            "cuda_graph_max_bs_decode": 1,
            "max_running_requests": 1,
        }
    )
    reasons_ten = k3._config_disable_reasons(ten, K3)
    assert any("SGLANG_AMD_MONO_K3_WIDE=0" in r for r in reasons_ten), reasons_ten

    narrow_decode = SimpleNamespace(
        **{**vars(bad_decode), "cuda_graph_max_bs_decode": 4, "max_running_requests": 4}
    )
    reasons_narrow = k3._config_disable_reasons(narrow_decode, K3)
    assert not any("SGLANG_AMD_MONO_K3_WIDE=0" in r for r in reasons_narrow), (
        reasons_narrow
    )

    monkeypatch.setattr(k3, "_host_only_mamba", lambda: True)
    ring_cfg = SimpleNamespace(
        dcp_size=8,
        attn_cp_size=1,
        enable_prefill_cp=False,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        cuda_graph_max_bs_decode=16,
        max_running_requests=16,
        enable_linear_replayssm_spec=True,
        **_NO_OFFLOAD,
    )
    reasons_ring = k3._config_disable_reasons(ring_cfg, K3)
    assert any("request-scoped ReplaySSM" in r for r in reasons_ring), reasons_ring

    narrow_ring = SimpleNamespace(
        **{**vars(ring_cfg), "cuda_graph_max_bs_decode": 1, "max_running_requests": 1}
    )
    reasons_narrow_ring = k3._config_disable_reasons(narrow_ring, K3)
    assert not any("request-scoped ReplaySSM" in r for r in reasons_narrow_ring)

    # _wide_will_run reads get_exec().graph.cuda_graph_max_bs_decode cleanly.
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_spec",
        lambda: SimpleNamespace(
            speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_max_bs_decode=16)),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_schedule",
        lambda: SimpleNamespace(max_running_requests=16),
    )
    assert k3._wide_will_run() is True
