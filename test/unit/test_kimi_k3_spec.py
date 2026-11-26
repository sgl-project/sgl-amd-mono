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
from sgl_amd_mono.kimi_k3 import (
    HF_CONFIG,
    LINEAR_ATTN_CONFIG,
    MAX_BLOCKS,
    MAX_ROWS,
    SHARED_GATE_UP,
)
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
HIDDEN = HF_CONFIG["hidden_size"]


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


FRONT_ROWS = (
    HF_CONFIG["num_experts"],
    HF_CONFIG["routed_expert_hidden_size"],
    SHARED_GATE_UP,
)


def _front(dtype=None, rows=FRONT_ROWS):
    """A MoE front's three weights, as SGLang or the load hook holds them."""
    import torch

    return tuple(
        torch.zeros(n, HIDDEN, dtype=torch.bfloat16).to(dtype or torch.bfloat16)
        for n in rows
    )


def _fake_layer(*, gate=True, kda=False, front_dtype=None, front_rows=FRONT_ROWS):
    """A covered layer as model_disable_reasons reads it, everything passing."""
    import torch

    def lin(*shape):
        return SimpleNamespace(
            weight=torch.zeros(*shape, dtype=torch.bfloat16), bias=None
        )

    w_gate, w_ld, w_sgu = _front(front_dtype, front_rows)
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
            gate=SimpleNamespace(weight=w_gate),
            routed_expert_down_proj=SimpleNamespace(weight=w_ld),
            shared_experts=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=w_sgu)),
        ),
        use_attn_residuals=True,
        _sp_moe=False,
        all_reduce_fusion=False,
    )


def _reasons(monkeypatch, layer, copies=None):
    from sglang.srt.layers import dp_attention
    from sglang.srt.utils import common

    # where spec.py imports it from (common: SGLang before sgl-project/sglang#48781)
    source = dp_attention if hasattr(dp_attention, "require_mlp_sync") else common
    monkeypatch.setattr(source, "require_mlp_sync", lambda: False)
    model = SimpleNamespace(model=SimpleNamespace(layers={i: layer for i in range(93)}))
    return k3._model_disable_reasons(model, K3, {} if copies is None else copies)


def _first_layer(reasons: list) -> list:
    """The first covered layer's reasons; the checks stop one layer later."""
    return [r for r in reasons if r.startswith("layer 1: ")]


def _kept(front, layers=range(1, 93)) -> dict:
    """The load hook's copies: one front shared by every covered layer."""
    return {(k3.FRONT_COPY, i, w): t for i in layers for w, t in zip(k3.FRONT, front)}


def test_a_covered_layer_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer()) == []
    assert _reasons(monkeypatch, _fake_layer(kda=True)) == []


def test_an_mla_layer_without_the_output_gate_is_refused(monkeypatch):
    # hasattr(attn, "forward_pre_o_proj") is true for both attention classes,
    # so it cannot carry this: for MLA the split rides on the gate's wrapper.
    reasons = _reasons(monkeypatch, _fake_layer(gate=False))
    assert any("output gate" in r for r in reasons), reasons


def test_an_fp8_front_without_bf16_copies_is_refused(monkeypatch):
    # SGLang's static FP8 front leaves E4M3 views in the module weights; the
    # launch would read their bytes as bf16 without an error.
    import torch

    reasons = _first_layer(
        _reasons(monkeypatch, _fake_layer(front_dtype=torch.float8_e4m3fn))
    )
    assert len(reasons) == 3, reasons
    assert all(f"{k3.BF16_FRONT_ENV}=1" in r for r in reasons), reasons


def test_an_fp8_front_with_the_copies_kept_at_load_passes(monkeypatch):
    import torch

    layer = _fake_layer(front_dtype=torch.float8_e4m3fn)
    assert _reasons(monkeypatch, layer, _kept(_front())) == []


@pytest.mark.parametrize(
    "front_rows, bad",
    [
        ((FRONT_ROWS[0] - 1, *FRONT_ROWS[1:]), "router gate"),
        ((FRONT_ROWS[0], FRONT_ROWS[1], 2 * SHARED_GATE_UP), "shared gate_up"),
    ],
)
def test_a_front_of_the_wrong_shape_is_refused(monkeypatch, front_rows, bad):
    layer = _fake_layer(front_rows=front_rows)
    reasons = _first_layer(_reasons(monkeypatch, layer))
    assert len(reasons) == 1 and bad in reasons[0], reasons


def test_a_non_contiguous_front_is_refused(monkeypatch):
    import torch

    layer = _fake_layer()
    rows = HF_CONFIG["routed_expert_hidden_size"]
    layer.mlp.routed_expert_down_proj.weight = torch.zeros(
        HIDDEN, rows, dtype=torch.bfloat16
    ).t()
    reasons = _first_layer(_reasons(monkeypatch, layer))
    assert len(reasons) == 1 and "latent down" in reasons[0], reasons


@pytest.mark.parametrize(
    "front_rows, bad",
    [
        ((FRONT_ROWS[0] - 1, *FRONT_ROWS[1:]), "router gate"),
        ((FRONT_ROWS[0], FRONT_ROWS[1], 2 * SHARED_GATE_UP), "shared gate_up"),
    ],
)
def test_a_kept_front_of_the_wrong_shape_is_refused(monkeypatch, front_rows, bad):
    import torch

    layer = _fake_layer(front_dtype=torch.float8_e4m3fn)
    reasons = _first_layer(_reasons(monkeypatch, layer, _kept(_front(rows=front_rows))))
    assert len(reasons) == 1 and bad in reasons[0], reasons


def test_a_missing_front_weight_is_refused_not_raised(monkeypatch):
    layer = _fake_layer()
    layer.mlp.shared_experts = None
    reasons = _first_layer(_reasons(monkeypatch, layer))
    assert reasons == ["layer 1: no shared gate_up projection"], reasons


def test_a_bf16_front_is_read_in_place_even_with_copies():
    layer = _fake_layer()
    own = k3._module_front(layer.mlp)
    front = k3._front_weights(layer.mlp, _kept(_front()), 1)
    assert all(f is o for f, o in zip(front, own))


def _parallel(monkeypatch, rank: int, size: int = 8):
    import sglang.srt.runtime_context as runtime_context

    p = SimpleNamespace(tp_rank=rank, tp_size=size)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: p)


def _ckpt(i: int, w: str) -> str:
    return f"language_model.model.layers.{i}.block_sparse_moe.{w}.weight"


def test_the_load_hook_keeps_this_ranks_front(monkeypatch):
    import torch

    _parallel(monkeypatch, rank=3)
    model = SimpleNamespace(config=K3)
    gate, down = torch.randn(4, 2), torch.randn(6, 2)
    shared_gate = torch.arange(16.0).view(16, 1).expand(16, 2).contiguous()
    shared_up = shared_gate + 100
    copies = {}
    for w, t in (
        ("shared_experts.up_proj", shared_up),
        ("gate", gate),
        ("routed_expert_down_proj", down),
        ("shared_experts.gate_proj", shared_gate),
    ):
        k3._load_tensor(copies, model, _ckpt(5, w), t)

    assert sorted(w for _, _, w in copies) == sorted(k3.FRONT)
    kept_gate, kept_down, gate_up = k3._kept_front(copies, 5)
    assert torch.equal(kept_gate, gate) and kept_gate.data_ptr() != gate.data_ptr()
    assert torch.equal(kept_down, down)
    # rank 3 of 8 holds rows 6 and 7 of each half, gate before up
    assert gate_up[:, 0].tolist() == [6.0, 7.0, 106.0, 107.0]


@pytest.mark.parametrize(
    "name",
    [
        _ckpt(0, "gate"),  # the dense layer 0 is not covered
        _ckpt(5, "shared_experts.down_proj"),
        _ckpt(5, "routed_expert_up_proj"),
        "language_model.model.layers.5.self_attn.o_proj.weight",
        "vision_tower.encoder.blocks.5.mlp.fc0.weight",
    ],
)
def test_the_load_hook_ignores_what_the_launch_reads_in_place(monkeypatch, name):
    import torch

    _parallel(monkeypatch, rank=0)
    copies = {}
    k3._load_tensor(copies, SimpleNamespace(config=K3), name, torch.zeros(8, 2))
    assert copies == {}


def test_a_text_only_checkpoint_name_is_kept(monkeypatch):
    import torch

    _parallel(monkeypatch, rank=0)
    copies = {}
    name = "model.layers.7.block_sparse_moe.gate.weight"
    k3._load_tensor(copies, SimpleNamespace(config=K3), name, torch.zeros(4, 2))
    assert list(copies) == [(k3.FRONT_COPY, 7, "gate")]


def test_the_build_uploads_the_copies_only_for_an_fp8_front():
    import torch

    copies = _kept(_front(), layers=(1, 2))
    fp8 = _fake_layer(front_dtype=torch.float8_e4m3fn).mlp
    placed = k3._place_front(fp8, copies, 1, torch.device("cpu"))
    assert all(
        _is_same(p, copies[k3.FRONT_COPY, 1, w]) for p, w in zip(placed, k3.FRONT)
    )

    bf16 = _fake_layer().mlp
    placed = k3._place_front(bf16, copies, 2, torch.device("cpu"))
    assert all(p is o for p, o in zip(placed, k3._module_front(bf16)))
    assert not any(i == 2 for _, i, _ in copies), "layer 2's copies are dropped"
    assert any(i == 1 for _, i, _ in copies)


def _is_same(a, b) -> bool:
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape


def test_the_front_is_kept_at_load_only_when_asked(monkeypatch):
    # a load hook closes the load paths that bypass it (dummy, layered, ...)
    monkeypatch.delenv(k3.BF16_FRONT_ENV, raising=False)
    assert k3._load_hook() is None
    monkeypatch.setenv(k3.BF16_FRONT_ENV, "1")
    assert k3._load_hook() is k3._load_tensor


def test_kimi_k31_keeps_no_front_at_load():
    from sgl_amd_mono.kimi_k31 import spec as k31

    # K3.1's wide launch reads SGLang's front in place
    assert k31.SPEC.load_tensor is None


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
