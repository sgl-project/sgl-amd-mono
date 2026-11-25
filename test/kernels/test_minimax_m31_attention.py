# SPDX-License-Identifier: Apache-2.0
"""The M3.1 attention launches against SGLang's kernels on one GPU, on the NVFP4 pool as
SGLang lays it out (one [layers, pages, row] slab, strided per-layer views):

- the front (``front.py``) against the input norm, the fused qkv + index GEMM and the
  qk-norm / RoPE / NVFP4 store: the residual bitwise, Q and index Q no further from an
  f64 reference than SGLang's, the pool rows as close, the padding slot unwritten;
- the verify indexer (``index.py``) against ``verify_topk_idx``: the same blocks a row
  with the score scratch poisoned first (a block not scored in the launch would rank
  first), and the arrival counters back at 0 (a launch whose idle CTAs arrived left
  them counting and picked on stale scores);
- the verify sparse attention (``sparse.py``) against ``flash_decode_nvfp4_gfx950``."""

import pytest
import torch

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("gfx950 only", allow_module_level=True)

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import (  # noqa: E402
    Nvfp4Cache,
)

from sgl_amd_mono.minimax_m31 import front, index  # noqa: E402
from sgl_amd_mono.minimax_m31.runner import (  # noqa: E402
    FrontRunner,
    IndexRunner,
    SparseRunner,
)

H, HD, PAGE, EPS = 6144, 128, 128, 1e-6
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DEV = torch.device("cuda")


def _rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def _decode(payload, scales):
    lut = torch.tensor(E2M1 + [-v for v in E2M1], device=payload.device)
    vals = torch.stack([lut[(payload & 0xF).long()], lut[(payload >> 4).long()]], -1)
    sc = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return vals.flatten(-2) * sc


class Pool:
    """Main K / V and index K of one layer of a two-layer slab, SGLang's page rows."""

    def __init__(self, pages, g):
        main = torch.zeros(2, pages, 2 * 8192 + 2 * 1024, dtype=torch.uint8, device=DEV)
        idx = torch.zeros(2, pages, 8192 + 1024, dtype=torch.uint8, device=DEV)
        main.random_(0, 256, generator=g)
        idx.random_(0, 256, generator=g)
        r, i = main[1], idx[1]
        self.k = Nvfp4Cache(
            r[:, :8192].view(pages, 1, PAGE, 64),
            r[:, 16384:17408].view(pages, 1, PAGE, 8),
        )
        self.v = Nvfp4Cache(
            r[:, 8192:16384].view(pages, 1, PAGE, 64),
            r[:, 17408:].view(pages, 1, PAGE, 8),
        )
        self.ik = Nvfp4Cache(
            i[:, :8192].view(pages, PAGE, 64), i[:, 8192:].view(pages, PAGE, 8)
        )
        for c in (self.k, self.v, self.ik):
            c.scales.copy_(
                (torch.rand(c.scales.shape, device=DEV, generator=g) * 0.5 + 0.05)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )


def _cos_sin(max_pos, theta=1e7, rot=64):
    inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
    f = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([f.cos(), f.sin()], -1).float().to(DEV)


@pytest.mark.parametrize("s", [1, 8, 16])
def test_front_matches_sglang(s):
    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from sglang.kernels.ops.attention.minimax_m3_qk_norm_rope import (
        sparse_qk_index_gemma_rmsnorm_rope_cache,
    )
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_add_rmsnorm_fp8

    g = torch.Generator(device=DEV).manual_seed(s)
    x = (torch.randn(s, H, device=DEV, generator=g) * 0.6).bfloat16()
    res = (torch.randn(s, H, device=DEV, generator=g) * 2.0).bfloat16()
    norm_w = (torch.randn(H, device=DEV, generator=g) * 0.2).bfloat16()
    wf = torch.randn(front.COLS, H, device=DEV, generator=g) * 0.02
    ws = wf.abs().amax(1, keepdim=True) / 448.0
    w8 = (wf / ws).to(torch.float8_e4m3fn)
    norms = [
        (torch.randn(HD, device=DEV, generator=g) * 0.3).bfloat16() for _ in range(4)
    ]
    cs = _cos_sin(1 << 16)
    pos = torch.randint(0, 1 << 16, (s,), device=DEV, generator=g)
    loc = torch.randperm(64 * PAGE - 1, device=DEV, generator=g)[:s].add(1).long()
    loc[-1] = 0 if s > 1 else loc[-1]  # a CUDA-graph padding row
    pools = [Pool(64, torch.Generator(device=DEV).manual_seed(7)) for _ in range(2)]
    # SGLang
    (x8, xsc, _), res_ref = gemma_add_rmsnorm_fp8(x, res, norm_w, EPS, keep_bf16=True)
    y = aiter.gemm_a8w8_bpreshuffle(
        x8, shuffle_weight(w8, (16, 16)), xsc, ws, None, torch.bfloat16
    )
    p = pools[0]
    q_ref, _, iq_ref, _ = sparse_qk_index_gemma_rmsnorm_rope_cache(
        y[:, :2048],
        y[:, 2048:2176],
        y[:, 2176:2304],
        y[:, 2304:2432],
        y[:, 2432:],
        p.k,
        p.v,
        p.ik,
        loc,
        *norms,
        pos,
        cs,
        EPS,
        HD,
        64,
        True,
    )
    # the launch
    from types import SimpleNamespace

    w = SimpleNamespace(
        input_norm=norm_w,
        wqkv=shuffle_weight(w8, (16, 16)),
        wqkv_scale=ws.reshape(-1).contiguous(),
        q_norm=norms[0],
        k_norm=norms[1],
        iq_norm=norms[2],
        ik_norm=norms[3],
    )
    p1 = pools[1]
    slot0 = (p1.k.payload[0, 0, 0].clone(), p1.v.payload[0, 0, 0].clone())
    res_new, q, iq = FrontRunner(DEV, (s,))(
        x, res, w, pos, loc, (p1.k, p1.v, p1.ik), cs
    )
    torch.cuda.synchronize()
    assert torch.equal(res_new, res_ref)
    # f64: the exact projection of the exact norm, then the heads' norms and RoPE
    s64 = x.double() + res.double()
    xn = (
        s64
        * torch.rsqrt(s64.pow(2).mean(-1, keepdim=True) + EPS)
        * (1 + norm_w.double())
    )
    heads = (xn @ (w8.double() * ws.double()).T).view(s, 20, HD)
    nw = torch.stack(
        [norms[0].double()] * 16
        + [
            norms[1].double(),
            torch.zeros(HD, dtype=torch.float64, device=DEV),
            norms[2].double(),
            norms[3].double(),
        ]
    )
    nrm = heads * torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + EPS) * (1 + nw)
    c, sn = cs[pos].double()[:, None, :32], cs[pos].double()[:, None, 32:]
    a, b = nrm[..., :32].clone(), nrm[..., 32:64].clone()
    ex = nrm.clone()
    ex[..., :32], ex[..., 32:64] = a * c - b * sn, b * c + a * sn
    assert (
        _rel(q, ex[:, :16].reshape(s, -1))
        <= _rel(q_ref, ex[:, :16].reshape(s, -1)) * 1.05
    )
    assert _rel(iq, ex[:, 18]) <= _rel(iq_ref, ex[:, 18]) * 1.10
    valid = loc != 0
    for name, head, rows in (
        ("k", 16, loc[valid]),
        ("v", 17, loc[valid]),
        ("ik", 19, loc),
    ):
        got_c, ref_c = getattr(p1, name), getattr(p, name)
        sel = (
            (rows // PAGE, 0, rows % PAGE)
            if name != "ik"
            else (rows // PAGE, rows % PAGE)
        )
        got = _decode(got_c.payload[sel], got_c.scales[sel])
        ref = _decode(ref_c.payload[sel], ref_c.scales[sel])
        exr = (heads if head == 17 else ex)[
            valid if name != "ik" else slice(None), head
        ]
        assert _rel(got, exr) <= _rel(ref, exr) * 1.10 + 0.005, name
    # the padding row's main K / V slot keeps its bytes
    if s > 1:
        assert torch.equal(p1.k.payload[0, 0, 0], slot0[0])
        assert torch.equal(p1.v.payload[0, 0, 0], slot0[1])


def _verify_case(ctx, reqs, g, pool):
    """A verify step's rows (8 drafts a request) over random pool pages."""
    rows, maxkv = 8 * reqs, 1 << 19
    r2t = torch.zeros(4, maxkv, dtype=torch.int32, device=DEV)
    for r in range(reqs):
        n = -(-(ctx[r] + 8) // PAGE)
        pages = (
            torch.randperm(pool.k.payload.shape[0] - 1, device=DEV, generator=g)[:n] + 1
        )
        r2t[r + 1, : n * PAGE] = (
            (pages[:, None] * PAGE + torch.arange(PAGE, device=DEV)).flatten().int()
        )
    seq = torch.tensor(
        [ctx[r] + j + 1 for r in range(reqs) for j in range(8)],
        dtype=torch.int64,
        device=DEV,
    )
    slot = torch.tensor(
        [r + 1 for r in range(reqs) for _ in range(8)], dtype=torch.int64, device=DEV
    )
    iq = (torch.randn(rows, HD, device=DEV, generator=g) * 0.5).bfloat16()
    q = (torch.randn(rows, 16 * HD, device=DEV, generator=g) * 0.5).bfloat16()
    return r2t, seq, slot, iq, q


# contexts a request: under 16 blocks, long, past 1024 blocks (two top-k passes)
CASES = [
    ((1500,), 1),
    ((73728,), 1),
    ((200000,), 1),
    ((150000, 1673), 2),
    ((300000, 90000), 2),
]


@pytest.fixture(scope="module")
def pool():
    return Pool(3000, torch.Generator(device=DEV).manual_seed(3))


@pytest.mark.parametrize("ctx,reqs", CASES)
def test_indexer_matches_sglang(ctx, reqs, pool):
    from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_idx import (
        verify_topk_idx,
    )

    g = torch.Generator(device=DEV).manual_seed(sum(ctx))
    r2t, seq, slot, iq, _ = _verify_case(ctx, reqs, g, pool)
    ref = verify_topk_idx(
        q=iq.view(-1, 1, HD),
        k_cache=pool.ik,
        req_to_token=r2t,
        seq_lens=seq,
        max_seqlen=int(seq.max()),
        slot_ids=slot,
        num_draft=8,
        block_size=128,
        topk=16,
        init_blocks=0,
        local_blocks=1,
        sm_scale=None,
        q_scale=None,
        k_scale=None,
        topk_idx_out=None,
        page_size=PAGE,
    )
    runner = IndexRunner(DEV)
    for _ in range(3):
        scores = runner.scratch.view(torch.int32)[
            index.SCORE : index.SCORE + 2 * index.DRAFT * index.MAX_BLOCKS
        ]
        scores.fill_(0x7F7F7F7F)  # ~3.4e38: a block left unscored would rank first
        got = runner(iq, pool.ik, r2t, seq, slot)
        torch.cuda.synchronize()
        assert torch.equal(got.sort(-1).values, ref.sort(-1).values)
        cnt = runner.scratch.view(torch.int32)[
            index.CNT : index.CNT + 2 * index.CNT_STRIDE : index.CNT_STRIDE
        ]
        assert cnt.tolist() == [0, 0]


@pytest.mark.parametrize("ctx,reqs", CASES)
def test_sparse_attention_matches_sglang(ctx, reqs, pool):
    from sglang.kernels.ops.attention.minimax_sparse.decode.nvfp4_decode_gfx950 import (
        flash_decode_nvfp4_gfx950,
    )

    g = torch.Generator(device=DEV).manual_seed(sum(ctx) + 1)
    r2t, seq, slot, iq, q = _verify_case(ctx, reqs, g, pool)
    topk = IndexRunner(DEV)(iq, pool.ik, r2t, seq, slot)
    ref = flash_decode_nvfp4_gfx950(
        q.view(-1, 16, HD), pool.k, pool.v, r2t, seq, slot, 128, topk, HD**-0.5, 1.0
    ).reshape(q.shape[0], -1)
    got = SparseRunner(DEV)(q, pool.k, pool.v, r2t, seq, slot, topk)
    torch.cuda.synchronize()
    assert _rel(got, ref) < 1e-3
