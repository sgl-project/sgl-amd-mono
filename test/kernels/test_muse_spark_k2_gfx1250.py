"""The Muse Spark K2 launch on one gfx1250 GPU (a TP group of one: the peer push
and the rank-ordered sum go to itself) against the math of the ROCm model's ops
it replaces: the fused output norm-gate, o_proj (one bf16 rounding; the
all-reduce sums one partial), the attention boundary (weightless post norm,
gated residual in fp32, pre-FFN norm), the router's top-8 on the fp32 weight and
the norm's fp32 output with the sigmoid scores, and the pooler with the pre-expert
norm; and that it compiles for every served width at TP 8.

The 8-rank test (later): ``torchrun --nproc-per-node 8``, every rank building
``K2Build(tokens, npes=8)``, a gloo group for ``PeerBuffer(sym_bytes(8) +
guard.FLAG_BYTES, group, rank, 8, dev)``, the launch given ``peers.local`` as
``sym`` and ``peers.addresses.data_ptr()`` as ``peers``; each rank's stream and
x against a reference whose o_proj output is ``torch.distributed.all_reduce`` of
the ranks' bf16 partials summed in fp32 and rounded once (the old
``test_muse_spark_peers.py`` pattern), every rank agreeing on ids and scores.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)

LAYER = 5
# the launch orders the scores as fp32 does (``test_k2_routes_ties_on_every_score_bit``); against this reference
# two candidates may still swap at the top-8 boundary when their scores are closer than the fp32 sums' order
# moves them (the launch sums the router's products in its own order, and its norm sums come from part sums)
TIE_TOL = 1e-3


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _bf(t):
    return t.to(torch.bfloat16).float()


def _rms(x, eps, w=None):
    """RMSNorm in fp32, the weight applied after the scale, one bf16 rounding."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    return y.to(torch.bfloat16)


def _reference(o, g, stream, w):
    """The ops' math: ``_rms_norm_gate_fwd_kernel`` (norm x sigmoid, one rounding),
    the o_proj GEMM rounded once, ``_fused_norm_residual_triton_kernel`` with a
    weightless post norm and bf16 round-trips, ``_joint_epilogue`` (sigmoid, bias,
    repeated argmax with the pick masked: a tie to the lowest id; the pooled row
    rounded before its norm)."""
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        POST_EPS,
        RMS_EPS,
        ROUTE_EPS,
        ROUTE_SCALE,
        ROUTER_MULT,
        TOP_K,
    )

    rows = o.shape[0]
    bf = torch.bfloat16
    of = o.float().reshape(rows, -1, HEAD_DIM)
    rstd = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    og = (of * rstd * torch.sigmoid(g.float().reshape(rows, -1, HEAD_DIM))).to(bf)
    ao = (og.reshape(rows, -1).float() @ w["w_o"].float().T).to(bf)  # TP 1: no sum
    nb = _rms(ao, POST_EPS).float()
    new = w["alpha"] * stream + w["beta"] * nb
    x = _rms(new, RMS_EPS, w["pre_w"])
    # the model's router reads the pre-FFN norm's fp32 output (before its bf16 rounding) and the fp32 weight
    x32 = (
        new
        * torch.rsqrt(new.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        * w["pre_w"].float()
    )
    logits = x32 @ w["router"]
    sc = torch.sigmoid(logits * ROUTER_MULT)
    biased = sc + w["bias"]
    order = torch.argsort(
        biased, dim=-1, descending=True, stable=True
    )  # a tie: the lower id first
    ids = order[:, :TOP_K]
    picks = sc.gather(1, ids)
    scores = picks / (picks.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    pooled = (x.float() @ w["w_pool"].float().T).to(bf)
    xn = _rms(pooled, RMS_EPS, w["xn_w"])
    return new, ids.to(torch.int32), scores, xn, biased


def _close(got, ref, rtol, atol, what):
    got, ref = got.float(), ref.float()
    bad = ((got - ref).abs() > rtol * ref.abs() + atol).float().mean().item()
    assert bad == 0.0, f"{what}: {bad:.2%} of the elements off by more than {rtol} relative + {atol}"  # fmt: skip


def _check_routing(ids, scores, ids_ref, scores_ref, biased):
    """Every token's picks are the reference's, unless a pick differs from one
    within TIE_TOL of the boundary; the scores of the shared picks agree."""
    from sgl_amd_mono.muse_spark.gfx1250.config import EXPERTS, TOP_K

    for t in range(ids.shape[0]):
        got = ids[t].tolist()
        ref = ids_ref[t].tolist()
        assert len(set(got)) == TOP_K and all(0 <= e < EXPERTS for e in got), (t, got)
        extra = sorted(set(got) - set(ref))
        if extra:
            bound = biased[t, ref].min().item()  # the weakest reference pick
            worst = biased[t, extra].min().item()
            assert worst >= bound - TIE_TOL, f"token {t}: picks {got} vs {ref}"
        ref_at = {e: s for e, s in zip(ref, scores_ref[t].tolist())}
        for e, s in zip(got, scores[t].tolist()):
            if e in ref_at:
                assert abs(s - ref_at[e]) <= 1e-3 + 1e-2 * abs(ref_at[e]), (
                    t,
                    e,
                    s,
                    ref_at[e],
                )


def _xq_ref(xn):
    """K3's MX FP8 blocks of the bf16 expert input (XQW code words then XSW scale
    words a token), as ``ffn.stage_xq_cols`` would stage them."""
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import XQW, XSW

    rows = xn.shape[0]
    xb = xn.float().reshape(rows, -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    inv448 = torch.tensor([0x3B124925], dtype=torch.int32).view(torch.float32).item()
    wb = (amax * inv448).view(torch.int32)
    e = (((wb >> 23) & 0xFF) + ((wb & 0x7FFFFF) != 0).to(torch.int32)).clamp(0, 254)
    q = (xb * torch.ldexp(torch.ones_like(amax), 127 - e)).to(torch.float8_e4m3fn)
    codes = q.view(torch.uint8).reshape(rows, -1).contiguous().view(torch.int32)
    scales = e.reshape(rows, -1).to(torch.uint8).contiguous().view(torch.int32)
    assert codes.shape[1] == XQW and scales.shape[1] == XSW
    return torch.cat([codes, scales], 1)


@pytest.mark.parametrize(
    "rows,xq,pshard",
    [
        (8, True, False),
        (16, True, False),
        (8, False, False),
        (8, True, True),
        (16, True, True),
        (12, True, False),
        (12, True, True),
    ],
)
def test_k2_matches_the_ops_math(rows, xq, pshard):
    import os

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import XQW, XSW
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import (
        k2_scratch_layout,
        sym_bytes,
    )

    torch.manual_seed(100 + rows)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    o = torch.randn(rows, Q_K, device=dev).to(bf)
    g = torch.randn(rows, Q_K, device=dev).to(bf)
    stream = torch.randn(rows, HIDDEN, device=dev)
    router = (
        torch.randn(HIDDEN, EXPERTS, device=dev) * 0.02
    )  # the fp32 router [HIDDEN, EXPERTS]
    w_pool = (torch.randn(MOE_HIDDEN, HIDDEN, device=dev) * 0.02).to(bf)
    w = {
        "w_o": (torch.randn(HIDDEN, Q_K, device=dev) * 0.02).to(bf),
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "pre_w": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "router": router.contiguous(),
        "bias": torch.randn(EXPERTS, device=dev) * 0.01,
        "w_pool": w_pool,
        "xn_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
    }
    ids = torch.full((MAX_TOKENS, TOP_K), -7, dtype=torch.int32, device=dev)
    scores = torch.full((MAX_TOKENS, TOP_K), float("nan"), device=dev)
    xn = torch.full((MAX_TOKENS, MOE_HIDDEN), float("nan"), device=dev, dtype=bf)
    lay = k2_scratch_layout()
    scratch = torch.zeros(scratch_bytes(lay), dtype=torch.uint8, device=dev)
    peers = PeerBuffer(sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    peers.bytes.zero_()
    epoch = _epoch(dev)
    # (pshard on a group of one: the rank's row slice is the whole pooler, pushed through ``poolr``)
    launch = build_k2(K2Build(tokens=rows, npes=1, xq_pre=xq, pshard=pshard))
    new_ref, ids_ref, scores_ref, xn_ref, biased = _reference(o, g, stream, w)
    got_stream = stream.clone()

    def run():
        launch(
            o.data_ptr(), g.data_ptr(), got_stream.data_ptr(), w["w_o"].data_ptr(),
            w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
            w["router"].data_ptr(), 0, w["bias"].data_ptr(),
            w["w_pool"].data_ptr(), w["xn_w"].data_ptr(), ids.data_ptr(), scores.data_ptr(),
            xn.data_ptr(), scratch.data_ptr(), peers.local, peers.addresses.data_ptr(), 0,
            epoch.data_ptr(), LAYER,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1

    try:
        for step in range(2):  # the second step's tags differ from the first's
            got_stream.copy_(stream)
            run()
        torch.cuda.synchronize()
        assert epoch[list(guard.ERROR_WORDS)].eq(0).all(), "a wait failed"
        for what, got_, ref_ in (
            ("stream", got_stream, new_ref),
            ("expert input", xn[:rows], xn_ref),
        ):
            err = (got_.float() - ref_.float()).abs()
            cos = (
                torch.nn.functional.cosine_similarity(
                    got_.float(), ref_.float(), dim=-1
                )
                .min()
                .item()
            )
            same_ = (got_.float() == ref_.float()).float().mean().item()
            print(
                f"{what} S={rows}: max abs err {err.max().item():.4g}, min cosine {cos:.6f}, bit-equal {same_:.2%}"
            )
        _close(got_stream, new_ref, 1 / 64, 1e-2, "stream")
        _close(xn[:rows], xn_ref, 1 / 32, 2e-2, "expert input")
        _check_routing(ids[:rows], scores[:rows], ids_ref, scores_ref, biased)
        same = sum(
            len(set(ids[t].tolist()) & set(ids_ref[t].tolist())) for t in range(rows)
        )
        print(f"routing S={rows}: {same} / {rows * TOP_K} picks equal the reference's")
        assert (
            (ids[rows:] == -7).all()
            and scores[rows:].isnan().all()
            and xn[rows:].isnan().all()
        )
        if xq:
            off, _ = lay["k2_xinq"]
            got_q = (
                scratch[off : off + rows * (XQW + XSW) * 4]
                .view(torch.int32)
                .reshape(rows, -1)
            )
            assert torch.equal(got_q, _xq_ref(xn[:rows])), (
                "the expert input's MX FP8 blocks differ"
            )
        if os.environ.get("K2_TIME"):
            ts = []
            for _ in range(20):
                a_, b_ = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a_.record()
                run()
                b_.record()
                b_.synchronize()
                ts.append(a_.elapsed_time(b_) * 1e3)
            print(
                f"K2 S={rows} npes=1: {sorted(ts)[10]:.1f} us a launch (median; host-timed)"
            )
    finally:
        peers.close()


def _score_with_low_bits(value: float, low: int) -> float:
    """``value`` as fp32 with the low 8 bits of its pattern set to ``low``."""
    bits = torch.tensor([value], dtype=torch.float32).view(torch.int32).item()
    return (
        torch.tensor([(bits & ~0xFF) | low], dtype=torch.int32)
        .view(torch.float32)
        .item()
    )


# (case, the experts well above the tie group, {expert of the group: low 8 bits of its score}, the group's picks in
# fp32's order, whether the launch's place order is fp32's too)
_TIES = [
    # a pair across the 8th place: the higher expert has the larger score
    ("pair", [3, 40, 77, 101, 150, 200, 250], {60: 0x20, 180: 0x25}, [180], True),
    # three experts for two places: the two largest scores, in score order
    (
        "triple",
        [3, 40, 77, 101, 150, 200],
        {10: 0x30, 20: 0x32, 30: 0x31},
        [20, 30],
        True,
    ),
    # the same score to the last bit: the lower expert
    ("equal", [3, 40, 77, 101, 150, 200, 250], {60: 0x20, 180: 0x20}, [60], True),
    # the group fills its places exactly (nothing past the 8th in it): the same picks; their places within the
    # group follow the keys (the lower expert first), which only moves the order of fp32 sums
    ("inside", [3, 40, 77, 101, 150, 200], {10: 0x30, 20: 0x32}, [20, 10], False),
]


@pytest.mark.parametrize("rows", [3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
def test_k2_every_plain_decode_width(rows):
    """The widths between the ones above, on the served build (the staged expert input)."""
    test_k2_matches_the_ops_math(rows, True, False)


@pytest.mark.parametrize("rows", [1, 4, 8, 12])
@pytest.mark.parametrize("case", [c[0] for c in _TIES])
def test_k2_routes_ties_on_every_score_bit(rows, case):
    """Scores that agree in their top 24 bits around the 8th place are still ordered as fp32 orders them (a tie
    to the lower expert): the router weight is zero, so every sigmoid is 0.5 and the biased scores are the
    crafted ones exactly, in the launch and in the reference."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        EXPERTS,
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import (
        k2_scratch_layout,
        sym_bytes,
    )

    _, above, group, group_picks, ordered = next(c for c in _TIES if c[0] == case)
    torch.manual_seed(300 + rows)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    o = torch.randn(rows, Q_K, device=dev).to(bf)
    g = torch.randn(rows, Q_K, device=dev).to(bf)
    stream = torch.randn(rows, HIDDEN, device=dev)
    router = torch.zeros(HIDDEN, EXPERTS, device=dev)
    score = torch.full((EXPERTS,), 0.1)
    score[above] = torch.tensor([1.5 - 0.01 * i for i in range(len(above))])
    for e, low in group.items():
        score[e] = _score_with_low_bits(0.8, low)
    bias = (score - 0.5).to(dev)
    assert all(
        (bias[e] + 0.5).item() == score[e].item() for e in group
    )  # the group's scores come out exactly
    w = {
        "w_o": (torch.randn(HIDDEN, Q_K, device=dev) * 0.02).to(bf),
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "pre_w": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "router": router,
        "bias": bias.contiguous(),
        "w_pool": (torch.randn(MOE_HIDDEN, HIDDEN, device=dev) * 0.02).to(bf),
        "xn_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
    }
    ids = torch.full((MAX_TOKENS, TOP_K), -7, dtype=torch.int32, device=dev)
    scores = torch.full((MAX_TOKENS, TOP_K), float("nan"), device=dev)
    xn = torch.full((MAX_TOKENS, MOE_HIDDEN), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(k2_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    peers.bytes.zero_()
    epoch = _epoch(dev)
    launch = build_k2(K2Build(tokens=rows, npes=1, xq_pre=True, pshard=True))
    _, ids_ref, scores_ref, _, _ = _reference(o, g, stream, w)
    want = above + group_picks
    assert all(ids_ref[t].tolist() == want for t in range(rows)), (
        ids_ref[0].tolist(),
        want,
    )
    got_stream = stream.clone()
    try:
        for step in range(2):
            got_stream.copy_(stream)
            launch(
                o.data_ptr(), g.data_ptr(), got_stream.data_ptr(), w["w_o"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
                w["router"].data_ptr(), 0, w["bias"].data_ptr(),
                w["w_pool"].data_ptr(), w["xn_w"].data_ptr(), ids.data_ptr(), scores.data_ptr(),
                xn.data_ptr(), scratch.data_ptr(), peers.local, peers.addresses.data_ptr(), 0,
                epoch.data_ptr(), LAYER,
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        assert epoch[list(guard.ERROR_WORDS)].eq(0).all(), "a wait failed"
        for t in range(rows):
            got = ids[t].tolist()
            assert sorted(got) == sorted(want), (
                f"{case}, token {t}: picks {got}, fp32's order gives {want}"
            )
            if ordered:
                assert got == want, (
                    f"{case}, token {t}: places {got}, fp32's order gives {want}"
                )
        if ordered:
            _close(scores[:rows], scores_ref, 1e-2, 1e-3, "scores")
    finally:
        peers.close()


@pytest.mark.parametrize("rows", [8, 16])
def test_k2_compiles_at_tp8(rows):
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    compile_only(build_k2(K2Build(tokens=rows, npes=8, xq_pre=True)))
    compile_only(build_k2(K2Build(tokens=rows, npes=8, xq_pre=True, pshard=True)))


def test_a_fault_site_compiles():
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    compile_only(build_k2(K2Build(tokens=8, fault_site=1, fault_after=1)))
