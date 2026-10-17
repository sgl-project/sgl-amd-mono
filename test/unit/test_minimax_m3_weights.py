# SPDX-License-Identifier: Apache-2.0
"""What ``weights.py`` refuses a MiniMax-M3 layer for, at each supported TP size.

The kernels read these tensors through raw pointers, so every shape the loader
accepts is a shape the kernels will dereference: a view that is one head or one
scale column off does not fault, it produces plausible garbage. The checks are
pure-Python arithmetic over a rank's ``Dims``, so they can be pinned on CPU
against a fake layer -- which is the only way to pin them at all, since a real
TP2 expert tensor is 19 GB.

The expectations below are literals rather than expressions over ``Dims`` on
purpose: they were compared value-for-value against the kernels' in-tree
ancestor (``srt/models/minimax_m3_mono/weights.py``, where TP was fixed at
import) at TP 2, 4 and 8, and a derivation that drifts would drift here too.
"""

import importlib.util
import sys
import types

import pytest

from sgl_amd_mono.minimax_m3.gfx950.config import HIDDEN, N_ROUTED, TP_SIZES, Dims

MODULE = "sgl_amd_mono.minimax_m3.gfx950.weights"
E = N_ROUTED + 1  # the 128 routed experts and the fused shared one

# tp -> the shapes this rank's checks accept
EXPECTED = {
    2: dict(qkv_rows=4992, o_k=4096, inter=1536, s2_cols=48),
    4: dict(qkv_rows=2560, o_k=2048, inter=768, s2_cols=24),
    8: dict(qkv_rows=1536, o_k=1024, inter=384, s2_cols=16),
}


# --------------------------------------------------------------- fake tensors
class FakeTensor:
    """Shape, dtype and identity -- all ``weights.py`` reads off a weight."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.device = tuple(shape), dtype, "cpu"

    @property
    def data(self):
        return self

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return id(self)

    # the quantization arithmetic: the loader's own checks run before it, and
    # the numerics are aiter's, so the fake only has to let the calls through
    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def abs(self):
        return self

    def amax(self, dim=None):
        return self

    def clamp_(self, *args, **kwargs):
        return self

    def __truediv__(self, other):
        return self

    def __getitem__(self, key):
        return self


def _stub_torch():
    """Only what ``weights.py`` touches; the dtypes just have to be distinct."""
    t = types.ModuleType("torch")
    for name in ("bfloat16", "uint8", "float32", "float8_e4m3fn"):
        setattr(t, name, f"torch.{name}")
    t.Tensor = FakeTensor
    t.finfo = lambda dtype: types.SimpleNamespace(tiny=1e-38)
    t.ones = lambda n, dtype=None, device=None: FakeTensor((n,), dtype)
    t.tensor = lambda v, dtype=None, device=None: FakeTensor((len(v),), dtype)
    return t


@pytest.fixture
def weights(monkeypatch):
    """``weights.py`` with torch and aiter's shuffle stubbed, re-imported so the
    stubs are what it binds. Torch is stubbed only where it is absent, so the
    same test runs inside the ROCm image against the real one."""
    if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
        monkeypatch.setitem(sys.modules, "torch", _stub_torch())
    shuffle = types.ModuleType("aiter.ops.shuffle")
    shuffle.shuffle_weight = lambda w, layout=None: w
    for name in ("aiter", "aiter.ops"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "aiter.ops.shuffle", shuffle)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    return importlib.import_module(MODULE)


# ------------------------------------------------------------------- fixtures
class GemmaRMSNorm:
    """``_gemma_norm_weight`` matches on the type name: the kernels apply w + 1."""

    def __init__(self, n, dtype):
        self.weight = FakeTensor((n,), dtype)


def build_layer(tp, torch, **tweak):
    """A sparse MoE layer shaped as the rank's ``Dims`` says, or off by ``tweak``."""
    dims = Dims(tp=tp)
    v = dict(
        EXPECTED[tp],
        num_heads=dims.local_q_heads,
        num_kv_heads=dims.local_kv_heads,
        top_k=5,
    )
    v.update(tweak)
    bf16, u8, f32 = torch.bfloat16, torch.uint8, torch.float32

    w13 = FakeTensor((E, 2 * v["inter"], HIDDEN // 2), u8)
    w2 = FakeTensor((E, HIDDEN, v["inter"] // 2), u8)
    w13.is_shuffled = w2.is_shuffled = True
    experts = types.SimpleNamespace(
        moe_ep_rank=0,
        num_local_experts=E,
        num_fused_shared_experts=1,
        top_k=v["top_k"],  # FusedMoE counts the fused shared expert in its top_k
        intermediate_size_per_partition=v["inter"],
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=FakeTensor((E, 2 * v["inter"], HIDDEN // 32), u8),
        w2_weight_scale=FakeTensor((E, HIDDEN, v["s2_cols"]), u8),
    )
    attn = types.SimpleNamespace(
        is_sparse_attention_layer=True,
        disable_index_value=True,
        num_heads=v["num_heads"],
        num_kv_heads=v["num_kv_heads"],
        num_idx_heads=dims.local_index_heads,
        head_dim=128,
        idx_head_dim=128,
        rotary_dim=64,
        use_gemma_norm=True,
        qk_norm_type="per_head",
        _fused_qkv_index=types.SimpleNamespace(
            weight=FakeTensor((v["qkv_rows"], HIDDEN), bf16)
        ),
        o_proj=types.SimpleNamespace(weight=FakeTensor((HIDDEN, v["o_k"]), bf16)),
        q_norm=GemmaRMSNorm(128, bf16),
        k_norm=GemmaRMSNorm(128, bf16),
        index_q_norm=GemmaRMSNorm(128, bf16),
        index_k_norm=GemmaRMSNorm(128, bf16),
        rotary_emb=types.SimpleNamespace(cos_sin_cache=FakeTensor((1 << 16, 64), f32)),
        # the radix attention the cache scales hang off; None means "one"
        attn=types.SimpleNamespace(k_scale=None, v_scale=None),
    )
    return types.SimpleNamespace(
        is_layer_sparse=True,
        self_attn=attn,
        mlp=types.SimpleNamespace(
            experts=experts,
            gate=types.SimpleNamespace(weight=FakeTensor((N_ROUTED, HIDDEN), f32)),
            e_score_correction_bias=FakeTensor((N_ROUTED,), f32),
        ),
        input_layernorm=GemmaRMSNorm(HIDDEN, bf16),
        post_attention_layernorm=GemmaRMSNorm(HIDDEN, bf16),
    )


def load(weights, tp, **tweak):
    torch = sys.modules["torch"]
    layer = build_layer(tp, torch, **tweak)
    heads = Dims(tp=tp).index_heads
    return layer, weights.SparseMoeLayer.from_layer(layer, 7, heads, {})


# ---------------------------------------------------------------------- tests
@pytest.mark.parametrize("tp", TP_SIZES)
def test_a_matching_layer_is_accepted(tp, weights):
    layer, got = load(weights, tp)
    experts = layer.mlp.experts
    assert got.layer_id == 7
    assert got.attn is layer.self_attn
    # the MXFP4 expert tensors and the norms are read in place, not copied
    assert (got.w13, got.s13, got.w2, got.s2) == (
        experts.w13_weight, experts.w13_weight_scale,
        experts.w2_weight, experts.w2_weight_scale,
    )  # fmt: skip
    assert got.g_in is layer.input_layernorm.weight
    assert got.g_post is layer.post_attention_layernorm.weight


@pytest.mark.parametrize("tp", TP_SIZES)
def test_the_rank_shapes_a_layer_must_match(tp, weights):
    """One head or one expert column off is refused rather than misread."""
    want = EXPECTED[tp]
    for tweak, where in (
        ({"num_heads": Dims(tp=tp).local_q_heads + 1}, "heads"),
        ({"num_kv_heads": Dims(tp=tp).local_kv_heads + 1}, "heads"),
        ({"qkv_rows": want["qkv_rows"] + 128}, "qkv"),
        ({"o_k": want["o_k"] + 128}, "o_proj"),
        ({"inter": want["inter"] // 2}, "expert intermediate"),
    ):
        with pytest.raises(ValueError, match=where):
            load(weights, tp, **tweak)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_the_w2_scale_is_checked_at_its_padded_width(tp, weights):
    """aiter's e8m0_shuffle stores w2's scale padded to a multiple of 8 columns,
    which only moves TP8 (12 -> 16). A loader reading the unpadded width would
    accept a tensor the kernels then stride wrongly, so the TP8 case must refuse
    -- and TP2/TP4, where the two widths coincide, must not."""
    dims = Dims(tp=tp)
    assert EXPECTED[tp]["s2_cols"] == dims.padded_scale_cols
    if dims.padded_scale_cols == dims.scale_cols:
        load(weights, tp, s2_cols=dims.scale_cols)
        return
    with pytest.raises(ValueError, match="expert scales"):
        load(weights, tp, s2_cols=dims.scale_cols)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_the_refusal_names_the_rank_width_it_wanted(tp, weights):
    """A refusal is reported per rank through the spec's disable reasons, so it
    has to say which TP's width it was reading."""
    want = EXPECTED[tp]
    with pytest.raises(ValueError, match=f"built for {want['inter']}"):
        load(weights, tp, inter=want["inter"] // 2)
    with pytest.raises(ValueError, match=rf"want \({HIDDEN}, {want['o_k']}\)"):
        load(weights, tp, o_k=want["o_k"] + 128)


def test_context_parallel_index_heads_are_refused(weights):
    """No caller builds cp=True; the kernels' cp branches are unreachable."""
    from sgl_amd_mono.minimax_m3.gfx950.config import TOTAL_INDEX_HEADS, IndexHeads

    dims = Dims(tp=4)
    heads = IndexHeads(dims=dims, count=TOTAL_INDEX_HEADS, cp=True)
    with pytest.raises(ValueError, match="context parallelism"):
        weights.SparseMoeLayer.from_layer(None, 7, heads, {})


def test_sparse_selection_pins_the_models_block_counts(weights):
    """TP-independent: the selection is the model's, not a rank's share."""
    cfg = dict(
        sparse_topk_blocks=16,
        sparse_init_block=0,
        sparse_local_block=1,
        sparse_block_size=128,
        sparse_score_type="max",
    )
    assert weights.sparse_selection(
        types.SimpleNamespace(sparse_attention_config=cfg)
    ) == (16, 0, 1)
    for bad in (
        dict(cfg, sparse_topk_blocks=8),
        dict(cfg, sparse_block_size=64),
        dict(cfg, sparse_score_type="sum"),
        {},
    ):
        with pytest.raises(ValueError):
            weights.sparse_selection(types.SimpleNamespace(sparse_attention_config=bad))
