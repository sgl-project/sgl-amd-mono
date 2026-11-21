"""The Qwen3.8-Flash-Next TP2 decode's launch operands on CPU: the MXFP4 copies'
layouts against the wide kernels' addressing (``tiled.Fp4Weights``, shuffled), the
experts' padding as SGLang loads it, the fused HC down matrix and the router, and
the operands and copies of every layer."""

import collections

import pytest
import torch
from qwen38_flash_next_decode_fakes import decode_mxfp4, fake_model, meta

from sgl_amd_mono.qwen38_flash_next.decode import weights as W
from sgl_amd_mono.qwen38_flash_next.decode.wide import layer as K
from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L


def _distinct_bytes(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)


def _data_byte(row, piece, cols):
    """``tiled.Fp4Weights._data_word`` (shuffled) in bytes: a 16-row group's
    16-byte K pieces as [piece][row % 16][16 B]; piece p holds K 32 p .. 32 p + 32."""
    return (row // 16) * (16 * cols // 2) + piece * 256 + (row % 16) * 16


def _scale_byte(row, kb, scale_cols):
    """``tiled.Fp4Weights._scale_byte`` (shuffled): e8m0_shuffle's byte of (row,
    K block ``kb``)."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, f = kb // 8, (kb // 4) % 2, kb % 4
    return ((((r32 * (scale_cols // 8) + c8) * 4 + f) * 16 + b) * 2 + d) * 2 + a


def test_shuffle_16x16_is_the_kernels_data_addressing():
    """Each 16 bytes the GEMM reads for (row, piece) are the row's 32 fp4 of that
    piece in the row-major weight."""
    n, k = 64, 512  # k fp4 a row
    q = _distinct_bytes(n, k // 2)
    flat = W.shuffle_16x16(q).reshape(-1)
    for row in range(n):
        for piece in range(k // 32):
            at = _data_byte(row, piece, k)
            assert torch.equal(flat[at : at + 16], q[row, 16 * piece : 16 * piece + 16])


def test_swizzle_scales_is_the_kernels_scale_addressing():
    n, cols = 64, 16
    s = _distinct_bytes(n, cols, seed=1)
    flat = W.swizzle_scales(s).reshape(-1)
    got = torch.stack(
        [flat[_scale_byte(r, c, cols)] for r in range(n) for c in range(cols)]
    )
    assert torch.equal(got, s.reshape(-1))


def test_shuffle_and_swizzle_reject_ragged_shapes():
    with pytest.raises(ValueError, match="packed FP4 weight"):
        W.shuffle_16x16(torch.zeros(24, 64, dtype=torch.uint8))
    with pytest.raises(ValueError, match="e8m0 scales"):
        W.swizzle_scales(torch.zeros(32, 12, dtype=torch.uint8))


def test_unshuffle_and_unswizzle_invert_the_copy_layouts():
    q, s = _distinct_bytes(64, 256, seed=4), _distinct_bytes(64, 16, seed=5)
    assert not torch.equal(W.shuffle_16x16(q), q)
    assert not torch.equal(W.swizzle_scales(s), s)
    assert torch.equal(W.unshuffle_16x16(W.shuffle_16x16(q)), q)
    assert torch.equal(W.unswizzle_scales(W.swizzle_scales(s)), s)


def test_mxfp4_copy_layouts_and_validation():
    w = torch.zeros(64, 512, dtype=torch.bfloat16)
    q, s = _distinct_bytes(64, 256, seed=2), _distinct_bytes(16, 64, seed=3)

    def quantize(x):
        assert x is w
        return q, s.t()  # scales handed back column-major, as AITER may

    wq, ws = W.mxfp4_copy(w, quantize)
    assert torch.equal(wq, W.shuffle_16x16(q))
    assert torch.equal(ws, W.swizzle_scales(s.t().contiguous()))
    with pytest.raises(ValueError, match="gave"):
        W.mxfp4_copy(w, lambda x: (q[:, :128], s.t()))
    with pytest.raises(ValueError, match="expected 2-D bf16"):
        W.mxfp4_copy(w.float(), quantize)


def test_mxfp4_dequant_reads_the_copy():
    q = _distinct_bytes(32, 128, seed=6)
    g = torch.Generator().manual_seed(7)
    s = torch.randint(120, 131, (32, 8), generator=g, dtype=torch.uint8)
    wq, ws = W.mxfp4_copy(torch.zeros(32, 256, dtype=torch.bfloat16), lambda x: (q, s))
    got = W.mxfp4_dequant(wq, ws)
    assert got.dtype == torch.bfloat16 and got.shape == (32, 256)
    assert torch.equal(got.float(), decode_mxfp4(q, s))


def test_the_gdn_copies_need_no_scale_padding():
    """Both GDN linears' K / 32 is a multiple of 8, so the copy's [N, K / 32] scales
    are the kernels' padded scale rows (``Fp4Weights.scale_cols``)."""
    for k in (L.HIDDEN, *(L.shard(tp).VALUE_DIM for tp in L.TPS)):
        assert W.scale_cols(k) == k // 32


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"), reason="SGLang sizes w13 by this dtype"
)
def test_sglang_pads_the_experts_to_512():
    from sglang.srt.layers.moe.utils import get_moe_weight_sizes

    w13_rows, w2_bytes, padded = get_moe_weight_sizes(L.shard(2).RI, True, True, True)
    assert (w13_rows, w2_bytes, padded) == (1024, 256, True)
    w13 = meta(513, w13_rows, 1280, dtype=torch.uint8)
    w2 = meta(513, 2560, w2_bytes, dtype=torch.uint8)
    assert W.moe_rip(w13, w2, 320) == 512
    K.LayerBuild(kind="gdn", tokens=1, tp=2, rip=512)


def test_moe_rip_at_the_kernels_own_padding():
    w13 = meta(513, 768, 1280, dtype=torch.uint8)
    w2 = meta(513, 2560, 192, dtype=torch.uint8)
    assert W.moe_rip(w13, w2, 320) == 384


@pytest.mark.parametrize(
    "w13_shape, w2_shape, dtype, match",
    [
        ((513, 1024, 1280), (513, 2560, 192), torch.uint8, "to 512, w2 to 384"),
        ((513, 512, 1280), (513, 2560, 128), torch.uint8, "padded to 256"),
        ((513, 832, 1280), (513, 2560, 208), torch.uint8, "padded to 416"),
        ((513, 1024, 2560), (513, 2560, 256), torch.uint8, "not 2560 wide"),
        ((513, 1024, 1280), (513, 2560, 256), torch.bfloat16, "not packed FP4"),
    ],
)
def test_moe_rip_rejects(w13_shape, w2_shape, dtype, match):
    with pytest.raises(ValueError, match=match):
        W.moe_rip(meta(*w13_shape, dtype=dtype), meta(*w2_shape, dtype=dtype), 320)


def test_fuse_hc_down_inject():
    g = torch.Generator().manual_seed(0)
    down = torch.randn(320, 10240, generator=g).bfloat16()
    inject = torch.randn(4, 10240, generator=g).bfloat16()
    fused = W.fuse_hc_down_inject(down, inject)
    assert fused.shape == (L.DOWN_ROWS, 10240) == (336, 10240)
    assert fused.dtype == torch.bfloat16
    assert torch.equal(fused[:320], down) and torch.equal(fused[320:324], inject)
    assert not fused[324:].any()
    fused[0, 0] += 1  # a copy: SGLang's weight unchanged
    assert not torch.equal(fused[:320], down)
    with pytest.raises(ValueError, match="expected"):
        W.fuse_hc_down_inject(torch.zeros(10240, 320), torch.zeros(4, 10240))


def test_router_gate_takes_the_shared_expert_s_gate_last():
    g = torch.Generator().manual_seed(1)
    gate = torch.randn(512, 2560, generator=g).bfloat16()
    shared = torch.randn(1, 2560, generator=g).bfloat16()
    out = W.router_gate(gate, shared)
    assert out.shape == (L.E_ALL, L.HIDDEN) and out.is_contiguous()
    assert torch.equal(out[:512], gate) and torch.equal(out[512], shared[0])
    with pytest.raises(ValueError, match="router"):
        W.router_gate(gate[:256], shared)


def _meta_quantize(w):
    n, k = w.shape
    return meta(n, k // 2, dtype=torch.uint8), meta(n, k // 32, dtype=torch.uint8)


def test_build_layers_operands_and_copies():
    model = fake_model(rip=512)
    copies = {}
    layers = W.build_layers(model, copies, _meta_quantize)
    assert [w.kind for w in layers] == [
        "qsa" if i % 4 == 3 else "gdn" for i in range(48)
    ]
    assert [w.idx for w in layers] == list(range(48))
    assert [i for i, w in enumerate(layers) if w.ple] == [1]
    kinds = collections.Counter(key[0] for key in copies)
    # 2 fused HC down matrices and a router a layer; GDN in_proj_qkvz and out_proj
    # with their scales; dt_bias held in f32 by the fake load
    assert kinds == {"hc_down": 96, "router": 48, "mxfp4": 36 * 4, "bf16": 36}
    gdn, qsa = layers[0], layers[3]
    names = set(K.ABI.names)
    assert set(gdn.args) <= names and set(qsa.args) <= names
    assert gdn.args["w_qkvz"].shape == (8192, 1280)
    assert gdn.args["w_qkvz_s"].shape == (8192, 80)
    assert gdn.args["w_out"].shape == (2560, 1536)
    assert gdn.args["w_out_s"].shape == (2560, 96)
    assert gdn.args["dt_bias"].dtype == torch.bfloat16
    assert gdn.gdn_eps == 1e-6 and qsa.gdn_eps is None
    assert gdn.hc_eps == gdn.mlp_hc_eps == 1e-6
    sg = model.model.layers[3]
    assert qsa.args["w_out"] is sg.o_proj.weight
    assert qsa.args["w_qkv"] is sg.qkv_proj.weight
    assert qsa.args["w_ix"] is sg.indexer.index_qk_proj.weight
    assert not {"w_qkvz", "w_conv", "qkv_out"} & set(qsa.args)
    assert not {"w_qkv", "w_ix"} & set(gdn.args)
    assert gdn.args["a_di"].shape == (336, 10240)
    assert copies[("hc_down", 3, "mlp")] is qsa.args["m_di"]
    assert copies[("router", 5)] is layers[5].args["w_gate"]
    assert layers[5].args["w_gate"].shape == (513, 2560)
    experts = model.model.layers[5].mlp.experts
    assert layers[5].args["w13"] is experts.w13_weight
    assert layers[5].args["w2s"] is experts.w2_weight_scale


def test_build_layers_refuses_a_changed_sglang_tensor():
    model = fake_model(rip=512)
    target = model.model.layers[0].linear_attn.out_proj.weight

    def quantize(w):
        if w is target:
            w.add_(0)  # an in-place write SGLang's module would not see coming
        return _meta_quantize(w)

    with pytest.raises(RuntimeError, match="layer 0 out_proj"):
        W.build_layers(model, {}, quantize)
