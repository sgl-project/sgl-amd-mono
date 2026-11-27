# SPDX-License-Identifier: Apache-2.0
"""Derived-property and negative-branch unit tests for wide ``dsa_post``."""

from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt
from sgl_amd_mono.kimi_k31 import wide
from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp
from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm
from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import BK


def _gemms(b: int, L: int):
    return tuple(Gemm(*(int(v) for v in c.split(":"))) for c in wide.cfg_rows(b, L))


@pytest.mark.parametrize("L", wide.SUPPORTED_L)
@pytest.mark.parametrize("timeline", [False, True])
def test_dsa_post_plan_and_tile_invariants_across_buckets(
    L: int, timeline: bool
) -> None:
    """Every bucket satisfies dsa_post K-split, CTA placement, and nvb-invariant peer offsets."""
    for b in wide.buckets(L):
        _, _, go, gf, gs = _gemms(b, L)
        assert dp.KO % go.ks == 0 and (dp.KO // go.ks) % BK == 0
        assert go.bm % go.ks == 0 and dp.SL % go.bn == 0
        ref_peer = None
        for nvb in (0, 1, 4, 8):
            pl, _ = dp.plan(b, go, gf, gs, wide.NF, nvb, timeline=timeline)
            if ref_peer is None:
                ref_peer = pl.peer_offsets
            else:
                assert pl.peer_offsets == ref_peer, (L, b, nvb)
            assert pl.scratch_bytes() > 0
            if timeline:
                assert "tl" in pl.scratch_offsets[0]
        if b <= mf.SORT_MAX_ROWS:
            n_s = ((b + gs.bm - 1) // gs.bm) * ((dp.H + gs.bn - 1) // gs.bn)
            assert BLOCKS - b - 1 - mf.SORT_HELPERS >= max(b, n_s), (L, b, gs)


def test_dsa_launch_args_zero_copy_and_partial_bucket_isolation() -> None:
    """Full contiguous inputs pass zero-copy; partial/strided inputs copy and zero-pad tail rows."""
    b, rows, rank = 16, 12, 3
    w = object.__new__(wide.Wide)
    w.rank = rank
    w.cols = slice(rank * wide.SL, (rank + 1) * wide.SL)
    w.epoch = torch.zeros(4, dtype=torch.int32)
    w.peers = {b: SimpleNamespace(addresses=torch.zeros(8, dtype=torch.int64))}
    w.peers_dsa = {b: SimpleNamespace(addresses=torch.zeros(8, dtype=torch.int64))}

    z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt)
    pk = SimpleNamespace(
        o_fmt=wfmt.BF16,
        w=dict(
            wo=z(wide.H, wide.KO),
            wo_s=None,
            cw_m=z(wide.H, dt=torch.float32),
            w_ln=z(wide.H),
            wf=z(wide.NF, wide.H),
            bias=z(wide.E, dt=torch.float32),
            w_sd=z(wide.H, wide.SI),
        ),
    )
    bufs = dict(
        core=torch.ones(b, wide.KO, dtype=torch.bfloat16),
        pout=[z(b, wide.H), z(b, wide.H)],
        bank=z(b, 8, wide.SL),
        acc_o=z(b * wide.H),
        lat=z(b, wide.LAT),
        ids=z(b, 16, dt=torch.int32),
        w=z(b, 16, dt=torch.float32),
        lat_p=[z(b, wide.LAT), z(b, wide.LAT)],
        sh=[z(b, wide.H), z(b, wide.H)],
        logits=z(b, wide.E, dt=torch.float32),
        cnt=z(128, dt=torch.int32),
        cnt_dsa=z(64, dt=torch.int32),
        scratch=z(512, dt=torch.uint8),
        scratch_dsa=z(256, dt=torch.uint8),
        sort=z(256, dt=torch.uint8),
    )

    col0 = 2 * rank * wide.SL
    core_full = torch.randn(b, wide.KO, dtype=torch.bfloat16)
    pfx_full = torch.randn(b, wide.H, dtype=torch.bfloat16)
    args = w._dsa_launch_args(pk, b, bufs, core_full, pfx_full, cur=1, slot=2)
    assert args[0] == core_full.data_ptr()
    assert args[3] == pfx_full.data_ptr() + col0
    assert args[24] == b * wide.H * 2 - col0

    core_part = torch.randn(rows, wide.KO, dtype=torch.bfloat16)
    pfx_strided = torch.randn(rows, wide.H * 2, dtype=torch.bfloat16)[:, : wide.H]
    args2 = w._dsa_launch_args(pk, b, bufs, core_part, pfx_strided, cur=1, slot=2)
    assert args2[0] == bufs["core"].data_ptr()
    assert torch.equal(bufs["core"][:rows], core_part)
    assert torch.count_nonzero(bufs["core"][rows:]) == 0
    assert args2[3] == bufs["pout"][0].data_ptr() + col0
    assert args2[24] == rows * wide.H * 2 - col0


def _fake_dsa_layer(
    *,
    block_write=False,
    ar_fusion=False,
    output_gate=True,
    has_pre_o_proj=True,
    has_mlp_res_norm=True,
    has_mlp_res_proj=True,
    wo_dtype=torch.bfloat16,
    wo_shape=(wide.H, wide.KO),
    wo_contig=True,
    w_ln_dtype=torch.bfloat16,
    w_sd_dtype=torch.bfloat16,
    w_sd_shape=(wide.H, wide.SI),
    w_sd_contig=True,
    res_eps=1e-5,
    ln_eps=1e-5,
    res_proj_shape=(1, wide.H),
    w_ln_shape=(wide.H,),
    w_ln_contig=True,
    act_kind="situ",
    situ_beta=4.0,
    situ_linear_beta=25.0,
):
    wo = torch.empty(wo_shape, dtype=wo_dtype)
    if not wo_contig:
        wo = wo.T.contiguous().T
    w_sd = torch.empty(w_sd_shape, dtype=w_sd_dtype)
    if not w_sd_contig:
        w_sd = w_sd.T.contiguous().T
    attn = SimpleNamespace(
        use_output_gate=output_gate,
        o_proj=SimpleNamespace(weight=wo, bias=None),
    )
    if has_pre_o_proj:
        attn.forward_pre_o_proj = lambda *a, **k: None
    from sglang.srt.layers.activation import SiluAndMul, SituAndMul

    act_fn = (
        SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)
        if act_kind == "situ"
        else SiluAndMul()
    )
    mlp = SimpleNamespace(
        shared_experts=SimpleNamespace(
            act_fn=act_fn, down_proj=SimpleNamespace(weight=w_sd, bias=None)
        )
    )
    return SimpleNamespace(
        self_attn=attn,
        mlp=mlp,
        is_block_write_layer=block_write,
        all_reduce_fusion=ar_fusion,
        mlp_res_norm=(
            SimpleNamespace(
                weight=torch.empty(wide.H, dtype=torch.bfloat16),
                variance_epsilon=res_eps,
            )
            if has_mlp_res_norm
            else None
        ),
        mlp_res_proj=(
            SimpleNamespace(weight=torch.empty(res_proj_shape, dtype=torch.bfloat16))
            if has_mlp_res_proj
            else None
        ),
        post_attention_layernorm=SimpleNamespace(
            weight=(
                torch.empty(w_ln_shape, dtype=w_ln_dtype)
                if w_ln_contig
                else torch.empty(w_ln_shape[0] * 2, dtype=w_ln_dtype)[::2]
            ),
            variance_epsilon=ln_eps,
        ),
    )


def test_dsa_fusable_negative_gates(monkeypatch) -> None:
    """Every safety gate in _dsa_fusable rejects incompatible layers instead of mis-launching."""
    monkeypatch.setattr(wide, "_tail_fusable", lambda layer: True)
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_DSA_POST", raising=False)
    assert wide._dsa_fusable(_fake_dsa_layer()) is True

    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_DSA_POST", "0")
    assert wide._dsa_fusable(_fake_dsa_layer()) is False
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_DSA_POST", "1")

    for bad_kwargs in (
        {"block_write": True},
        {"ar_fusion": True},
        {"output_gate": False},
        {"has_mlp_res_norm": False},
        {"has_mlp_res_proj": False},
        {"wo_dtype": torch.float8_e4m3fn},
        {"wo_shape": (wide.H, 1024)},
        {"wo_contig": False},
        {"w_ln_dtype": torch.float32},
        {"w_sd_dtype": torch.float8_e4m3fn},
        {"w_sd_shape": (wide.H, 512)},
        {"w_sd_contig": False},
        {"res_eps": 1e-6, "ln_eps": 1e-5},
        {"res_proj_shape": (1, 1024)},
        {"w_ln_shape": (1024,)},
        {"w_ln_contig": False},
        {"act_kind": "silu"},
        {"situ_beta": 1.0},
        {"situ_linear_beta": None},
        {"situ_linear_beta": 10.0},
    ):
        assert wide._dsa_fusable(_fake_dsa_layer(**bad_kwargs)) is False, bad_kwargs


def test_dsa_post_builder_validation_and_fp8_aux16(monkeypatch) -> None:
    """dsa_post rejects invalid tile/sort/nvb invariants and accepts Quark FP8 + wide aux16."""
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    _, _, go, gf, gs = _gemms(16, 8)
    launch, pl = dp.dsa_post(16, go, gf, gs, wide.NF, nvb_m=1, launches=92)
    launch2, pl2 = dp.dsa_post(16, go, gf, gs, wide.NF, nvb_m=1, launches=92)
    assert launch2 is launch and pl2 is pl

    with pytest.raises(AssertionError):
        dp.dsa_post(16, go, gf, gs, wide.NF, nvb_m=9, nbk=8, launches=92)
    with pytest.raises(AssertionError):
        dp.dsa_post(32, go, Gemm(32, 128, 2, 4, 7), gs, wide.NF, nvb_m=1, launches=24)
    for gf_uneven in (Gemm(64, 128, 2, 4, 56), Gemm(32, 128, 2, 4, 56)):
        with pytest.raises(AssertionError):
            dp.plan(64, go, gf_uneven, gs, wide.NF, nvb_m=1)
    for go_bad in (Gemm(16, 128, 1, 8, 3), Gemm(16, 192, 1, 8, 4)):
        with pytest.raises(AssertionError):
            dp.plan(16, go_bad, gf, gs, wide.NF, nvb_m=1)

    _, _, go2, gf2, gs2 = _gemms(288, 6)
    launch_fp8, _ = dp.dsa_post(
        288,
        go2,
        gf2,
        gs2,
        wide.NF,
        nvb_m=3,
        launches=23,
        sort="aux16",
        o_fmt=wfmt.FP8_SHUFFLED,
    )
    assert callable(launch_fp8)
    with pytest.raises(AssertionError):
        dp.dsa_post(288, go2, gf2, gs2, wide.NF, nvb_m=3, launches=23, sort="sgl")

    layer = _fake_dsa_layer()
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )
    layer.self_attn.o_proj = SimpleNamespace(
        weight=torch.zeros((wide.H, wide.KO), dtype=torch.float8_e4m3fn).t(),
        weight_scale=torch.ones(wide.H, dtype=torch.float32),
        scheme=scheme,
        bias=None,
    )
    monkeypatch.setattr(wide, "_tail_fusable", lambda _l: True)
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_DSA_POST", "1")
    assert wide._dsa_fusable(layer) is True
