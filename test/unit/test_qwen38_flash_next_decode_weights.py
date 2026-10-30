"""The Qwen3.8-Flash-Next launch operands on CPU: the per-rank TP2 shapes, the
MXFP4 copies' layouts against the kernels' addressing, the activation rounding the
kernels apply, the experts' padding as SGLang loads it, the fused HC down matrix,
and the operands and copies of every layer."""

import collections

import pytest
import torch
from qwen38_flash_next_decode_fakes import decode_mxfp4, fake_model, meta

from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import layout as KL
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import mx_layout as QL
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import weights as W


def test_tp2_rank_shapes():
    lay = KL.Layout(2)
    assert (lay.QKVZ, lay.BA, lay.CONV, lay.CORE, lay.NV) == (8192, 48, 5120, 3072, 24)
    assert (lay.RI, lay.RIP) == (320, 384)
    with pytest.raises(ValueError, match="tensor parallel size 4"):
        KL.Layout(4)


def _distinct_bytes(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)


def test_shuffle_16x16_is_the_kernels_tile_addressing():
    """Every lane's 16 B the FP4 GEMVs read (``fp4_tile_dword``) are the 32 fp4 of
    its row at its K offset in the unshuffled weight."""
    n, k = 64, 512  # k fp4 a row
    q = _distinct_bytes(n, k // 2)
    flat = W.shuffle_16x16(q).reshape(-1)
    for rg in range(n // 16):
        for st in range(k // QL.FP4_STEP):
            for lane in range(64):
                at = 4 * QL.fp4_tile_dword(0, rg, k, st, lane)
                row, col = 16 * rg + lane % 16, (128 * st + 32 * (lane // 16)) // 2
                assert torch.equal(flat[at : at + 16], q[row, col : col + 16])


def test_swizzle_scales_is_the_kernels_scale_index():
    n, cols = 64, 16
    s = _distinct_bytes(n, cols, seed=1)
    flat = W.swizzle_scales(s).reshape(-1)
    got = torch.stack(
        [flat[QL.scale_index(r, c, cols)] for r in range(n) for c in range(cols)]
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


def test_mxfp4_fake_quant_is_the_kernels_activation_rounding():
    """1 x 32 blocks; code ceil(log2(amax / 6)); nearest e2m1, ties to even, at most 6."""
    x = torch.zeros(2, 64)
    # block 0: amax 6 -> scale 1; ties 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 2.5 -> 2,
    # 3.5 -> 4, 5 -> 4
    x[0, :8] = torch.tensor([6.0, 0.25, 0.75, 1.25, 2.5, -3.5, 5.0, 1.75])
    # block 1: amax 0.375 -> scale 2^-4 (exact), kept apart from block 0's scale
    x[0, 32:34] = torch.tensor([0.375, -0.1])
    # block 2: amax 6.5 -> ceil -> scale 2 (a floor or round-to-nearest code keeps 1)
    x[1, :3] = torch.tensor([6.5, 1.5, -0.4])
    y = W.mxfp4_fake_quant(x.bfloat16())
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert y[0, :8].float().tolist() == [6.0, 0.0, 1.0, 1.0, 2.0, -4.0, 4.0, 2.0]
    assert y[0, 32:34].float().tolist() == [0.375, -0.09375]  # -0.1 / 2^-4 -> -1.5
    assert y[1, :3].float().tolist() == [6.0, 2.0, 0.0]
    assert not y[0, 8:32].any() and not y[1, 32:].any()


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"), reason="SGLang sizes w13 by this dtype"
)
def test_sglang_pads_the_experts_to_512_and_the_launch_reads_384():
    from sglang.srt.layers.moe.utils import get_moe_weight_sizes

    lay = KL.Layout(2)
    w13_rows, w2_bytes, padded = get_moe_weight_sizes(lay.RI, True, True, True)
    assert (w13_rows, w2_bytes, padded) == (1024, 256, True)
    w13 = meta(513, w13_rows, 1280, dtype=torch.uint8)
    w2 = meta(513, 2560, w2_bytes, dtype=torch.uint8)
    assert W.moe_rip(w13, w2, lay.RI) == 512 > lay.RIP


def test_moe_rip_at_the_kernels_own_padding():
    w13 = meta(513, 768, 1280, dtype=torch.uint8)
    w2 = meta(513, 2560, 192, dtype=torch.uint8)
    assert W.moe_rip(w13, w2, 320) == 384


@pytest.mark.parametrize(
    "w13_shape, w2_shape, dtype, match",
    [
        ((513, 1024, 1280), (513, 2560, 192), torch.uint8, "to 512, w2 to 384"),
        ((513, 512, 1280), (513, 2560, 128), torch.uint8, "intermediate 256 for 320"),
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
    assert fused.shape == (336, 10240) and fused.dtype == torch.bfloat16
    assert torch.equal(fused[:320], down) and torch.equal(fused[320:324], inject)
    assert not fused[324:].any()
    fused[0, 0] += 1  # a copy: SGLang's weight unchanged
    assert not torch.equal(fused[:320], down)
    with pytest.raises(ValueError, match="expected"):
        W.fuse_hc_down_inject(torch.zeros(10240, 320), torch.zeros(4, 10240))


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
    kinds = collections.Counter(key[0] for key in copies)
    # 2 fused HC down matrices a layer; GDN in_proj_qkvz and out_proj with their
    # scales; dt_bias held in f32 by the fake load
    assert kinds == {"hc_down": 96, "mxfp4": 36 * 4, "bf16": 36}
    gdn, qsa = layers[0], layers[3]
    assert gdn.gdn.qkvz.weight.shape == (8192, 1280)
    assert gdn.gdn.qkvz.scale.shape == (8192, 80)
    assert gdn.gdn.out.weight.shape == (2560, 1536)
    assert gdn.gdn.out.scale.shape == (2560, 96)
    assert gdn.gdn.dt_bias.dtype == torch.bfloat16 and gdn.gdn.norm_eps == 1e-6
    assert gdn.o_proj is None and qsa.gdn is None
    assert qsa.o_proj is model.model.layers[3].o_proj.weight
    assert gdn.attn_hc.down.shape == (336, 10240) and gdn.attn_hc.eps == 1e-6
    assert copies[("hc_down", 3, "mlp")] is qsa.mlp_hc.down
    experts = model.model.layers[5].mlp.experts
    assert layers[5].moe.w13 is experts.w13_weight
    assert layers[5].moe.w2s is experts.w2_weight_scale


def test_build_layers_refuses_a_changed_sglang_tensor():
    model = fake_model(rip=512)
    target = model.model.layers[0].linear_attn.out_proj.weight

    def quantize(w):
        if w is target:
            w.add_(0)  # an in-place write SGLang's module would not see coming
        return _meta_quantize(w)

    with pytest.raises(RuntimeError, match="layer 0 out_proj"):
        W.build_layers(model, {}, quantize)
