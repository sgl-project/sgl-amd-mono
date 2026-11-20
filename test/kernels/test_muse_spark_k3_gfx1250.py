"""The Muse Spark K3 launch on one gfx1250 GPU (a TP / EP group of one) against
the math of the normal gfx1250 expert path it replaces (aiter's A8W4 TDM grouped
MoE + Muse's finalize): the experts' input quantized to MX FP8 (E4M3 values, an
E8M0 scale a 32-value block by aiter's RoundUp rule), gate / up on the E2M1 weights
in fp32, silu(gate) up in fp32 re-quantized the same way, the down GEMM rounded to
bf16 a (token, slot), the finalize (per-expert post norm with input scale, the
score-weighted slot sum, one rounding), the unpooler (one rounding) and the FFN
boundary, with the last-layer rule as a case. The experts go through SGLang's own
``prepare_gfx1250_a8w4_weights`` (the layout the kernel reads in place); picks
outside the rank's experts are skipped and one expert is picked by every token."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EXPERT_ROWS = 8  # the test's expert table: a prefix of the rank's 32; picks past it are other ranks'
LOCAL_START = 64
LAYER = 7


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _fp4_weight(experts, rows, k, device, gen):
    """Random packed E2M1 [experts, rows, k / 2] (the low nibble the even k) with
    E8M0 scales [experts, rows, k / 32]: the checkpoint layout; and the decoded
    values [experts, rows, k] (fp32)."""
    packed = torch.randint(0, 256, (experts, rows, k // 2), device=device, dtype=torch.uint8, generator=gen)  # fmt: skip
    scales = torch.randint(118, 128, (experts, rows, k // 32), device=device, dtype=torch.uint8, generator=gen)  # fmt: skip
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(experts, rows, k)
    mag = E2M1.to(device)[(codes & 7).long()]
    val = torch.where(codes & 8 != 0, -mag, mag)
    scale = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), scales.to(torch.int32) - 127
    )
    dec = (val.reshape(experts, rows, k // 32, 32) * scale[..., None]).reshape(
        experts, rows, k
    )
    return packed, scales, dec


def mx8_dequant(x):
    """x (fp32 [..., N]) through aiter's gfx1250 MX FP8 quantization and back:
    e = the exponent of amax * fp32(1 / 448), plus one when its mantissa is non-zero;
    E4M3 = RNE(x 2^(127 - e))."""
    shape = x.shape
    xb = x.float().reshape(*shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    inv448 = torch.tensor([0x3B124925], dtype=torch.int32).view(torch.float32).item()
    wb = (amax * inv448).view(torch.int32)
    e = ((wb >> 23) & 0xFF) + ((wb & 0x7FFFFF) != 0).to(torch.int32)
    e = e.clamp(0, 254)
    q = (
        (xb * torch.ldexp(torch.ones_like(amax), 127 - e))
        .to(torch.float8_e4m3fn)
        .float()
    )
    return (q * torch.ldexp(torch.ones_like(amax), e - 127)).reshape(shape)


def _bf(t):
    return t.to(torch.bfloat16).float()


def _experts_math(xin, ids, local_start, dec13, dec2, rows_of=None):
    """The normal gfx1250 path's expert rows y [S, TOP_K, MOE_HIDDEN] (zeros where the
    pick is another rank's)."""
    from sgl_amd_mono.muse_spark.gfx1250.config import INTER, LOCAL_EXPERTS, TOP_K

    S = xin.shape[0]
    y = torch.zeros(S, TOP_K, dec2.shape[1], device=xin.device)
    xq = mx8_dequant(xin.float())
    for t in range(S):
        for s in range(TOP_K):
            e = int(ids[t, s]) - local_start
            if not 0 <= e < LOCAL_EXPERTS:
                continue
            gu = xq[t] @ dec13[e].T
            gate, up = gu[:INTER], gu[INTER:]
            mid = mx8_dequant(gate * torch.sigmoid(gate) * up)
            y[t, s] = _bf(mid @ dec2[e].T)
    return y


def _reference(xin, ids, scores, local_start, dec13, dec2, w, stream, last):
    """The experts + the finalize (one rank: the all-reduce is the identity) + the
    unpooler and the boundary with ``pre_bf16 = last``."""
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        LOCAL_EXPERTS,
        POST_EPS,
        RMS_EPS,
        TOP_K,
    )

    y = _experts_math(xin, ids, local_start, dec13, dec2)
    rows = xin.shape[0]
    out = torch.zeros(rows, dec2.shape[1], device=xin.device)
    for t in range(rows):
        for s in range(TOP_K):
            e = int(ids[t, s]) - local_start
            if not 0 <= e < LOCAL_EXPERTS:
                continue
            v = _bf(y[t, s] * w["pen_w"].float())
            sc = float(scores[t, s]) / torch.sqrt(v.pow(2).mean() + POST_EPS)
            out[t] += sc * v
    part = _bf(out)
    branch = _bf(part @ w["w_unpool"].float().T)
    post_input = _bf(branch * w["pfw"].float())
    nb = _bf(
        post_input * torch.rsqrt(post_input.pow(2).mean(-1, keepdim=True) + POST_EPS)
    )
    new = w["alpha"] * stream + w["beta"] * nb
    pre = _bf(new) if last else new
    attn_in = (pre * torch.rsqrt(pre.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w["npw"].float()).to(torch.bfloat16)  # fmt: skip
    return new, attn_in, nb


def _close(got, ref, rtol, atol, what):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    bad = (err > rtol * ref.abs() + atol).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    print(f"{what}: max abs err {err.max().item():.4g}, max rel err {(err / ref.abs().clamp(min=1e-3)).max().item():.4g}, min cosine {cos.min().item():.6f}")  # fmt: skip
    assert bad == 0.0, f"{what}: {bad:.2%} of the elements off by more than {rtol} relative + {atol}"  # fmt: skip
    assert cos.min().item() >= 0.996, (
        f"{what}: a row's cosine is {cos.min().item():.4f}"
    )


def _picks(rows, device, gen):
    """ids int32 [rows, TOP_K]: slot 0 is the rank's expert 0 for every token (the
    widest column set), the rest a random choice of the other table experts and of
    ids outside the rank, one of them past the rank's table."""
    from sgl_amd_mono.muse_spark.gfx1250.config import LOCAL_EXPERTS, TOP_K

    ids = torch.empty(rows, TOP_K, dtype=torch.int64, device=device)
    for t in range(rows):
        # K3_TEST_SPAN: the ids the other slots draw from (default: the table and seven ids past it; EXPERT_ROWS - 1:
        # every pick on this rank, the heaviest plan a row count has)
        span = int(os.environ.get("K3_TEST_SPAN", EXPERT_ROWS + 7))
        others = torch.randperm(span, device=device, generator=gen)[: TOP_K - 1] + 1
        ids[t, 0] = 0
        ids[t, 1:] = others
    if "K3_TEST_SPAN" not in os.environ:
        ids[0, 1] = LOCAL_EXPERTS + 3  # past the rank's table
    ids = torch.where(ids >= EXPERT_ROWS, ids + LOCAL_EXPERTS, ids)
    return (ids + LOCAL_START).to(torch.int32)


def gfx1250_experts(w13, s13, w2, s2, local_start):
    """The experts through SGLang's own gfx1250 load-time shuffle (the tensors the
    kernel reads in place): (w13, w2, s13, s2) as ``experts.gfx1250_a8w4`` holds them. An SGLang without
    ``prepare_gfx1250_a8w4_weights`` (one whose MoE backend shuffles the stored tensors in place): aiter's two
    shuffle functions with the arguments both pass."""
    try:
        from sglang.srt.rocm_muse_spark.mxfp4_moe import prepare_gfx1250_a8w4_weights
    except ImportError:
        from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

        e = w13.shape[0]
        return (
            moe_shuffle_weight(
                w13.clone().view(torch.uint8).contiguous(),
                experts_cnt=e,
                is_guinterleave=True,
                gate_up=True,
            ).contiguous(),
            moe_shuffle_weight(
                w2.clone().view(torch.uint8).contiguous(), experts_cnt=e
            ).contiguous(),
            moe_shuffle_scale(
                s13.clone().view(torch.uint8).contiguous(),
                experts_cnt=e,
                is_guinterleave=True,
                gate_up=True,
            ),
            moe_shuffle_scale(s2.clone().view(torch.uint8).contiguous(), experts_cnt=e),
        )

    P = torch.nn.Parameter
    ex = SimpleNamespace(
        num_local_experts=w13.shape[0],
        local_expert_start_for_a8w4=local_start,
        w13_weight=P(w13.clone(), requires_grad=False),
        w2_weight=P(w2.clone(), requires_grad=False),
        w13_weight_scale=P(s13.clone(), requires_grad=False),
        w2_weight_scale=P(s2.clone(), requires_grad=False),
    )
    prepare_gfx1250_a8w4_weights(ex)
    a = ex.gfx1250_a8w4
    return a[0], a[1], a[2], a[3]


def k3_case(rows, last, seed, npes=1, experts=EXPERT_ROWS, ids=None):
    """The tensors of a one-rank K3 case (``experts``: the table's rows; ``ids``: the picks, else ``_picks``)."""
    from sgl_amd_mono.muse_spark.gfx1250.config import HIDDEN, INTER, MOE_HIDDEN, TOP_K

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    bf = torch.bfloat16
    w13, s13, dec13 = _fp4_weight(experts, 2 * INTER, MOE_HIDDEN, dev, gen)
    w2, s2, dec2 = _fp4_weight(experts, MOE_HIDDEN, INTER, dev, gen)
    g13, g2, gs13, gs2 = gfx1250_experts(w13, s13, w2, s2, LOCAL_START)
    xin = (torch.randn(rows, MOE_HIDDEN, device=dev) * 0.5).to(bf)
    ids = _picks(rows, dev, gen) if ids is None else ids
    scores = torch.rand(rows, TOP_K, device=dev) * 0.3
    stream = torch.randn(rows, HIDDEN, device=dev)
    w = {
        "pen_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
        "pfw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "npw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "w_unpool": (torch.randn(HIDDEN, MOE_HIDDEN, device=dev) * 0.02).to(bf),
    }
    return SimpleNamespace(
        xin=xin, ids=ids, scores=scores, stream=stream, w=w, dec13=dec13, dec2=dec2,
        w13=g13, s13=gs13, w2=g2, s2=gs2,
    )  # fmt: skip


def _time_launches(run, n=20):
    """Mean device time of one ``run()`` (us), each timed by its own events."""
    ts = []
    for _ in range(n):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        run()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


@pytest.mark.parametrize(
    "rows,last,xq",
    [
        (8, False, False),
        (16, True, False),
        (8, True, False),
        (16, False, False),
        (8, False, True),
        (16, True, True),
        (1, False, True),
        (2, True, True),
        (4, False, True),
        (1, True, True),
        (12, False, True),
        (12, True, True),
        (12, False, False),
    ],
)
def test_k3_matches_the_gfx1250_expert_math(rows, last, xq):
    _needs_xq(xq)
    _check_k3(k3_case(rows, last, 300 + rows), rows, last, xq)


@pytest.mark.parametrize("rows", [3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
def test_k3_every_plain_decode_width(rows):
    """The widths between the ones above, on the staged expert input (the last-layer form at the odd ones)."""
    _check_k3(k3_case(rows, rows % 2 == 1, 300 + rows), rows, rows % 2 == 1, True)


def _needs_xq(xq):
    """The ring build (``SGLANG_AMD_MONO_K3_WAVE=2``, the server's form) reads K2's staged expert input: it has no
    build without it."""
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import K3_WAVE

    if K3_WAVE == 2 and not xq:
        pytest.skip(
            "the ring build takes the expert input as K2's MX FP8 blocks (xq_pre)"
        )


@pytest.mark.parametrize(
    "rows,experts",
    [
        (16, 13),
        (16, 17),
        (16, 24),
        (16, 25),
        (16, 32),
        (8, 13),
        (8, 20),
        (4, 14),
        (2, 16),
        (12, 13),
        (12, 20),
        (12, 24),
        (12, 25),
    ],
)
def test_k3_many_experts_a_rank(rows, experts):
    """Rows whose picks cover ``experts`` experts of this rank, a few columns each: more tiles than one round of
    items takes (12 with every wave of the grid on one), so a wave runs two or three items."""
    from sgl_amd_mono.muse_spark.gfx1250.config import TOP_K

    dev = torch.device("cuda")
    ids = torch.tensor(
        [[(t * TOP_K + j) % experts for j in range(TOP_K)] for t in range(rows)],
        device=dev,
    )
    assert ids.unique().numel() == experts
    k = k3_case(
        rows,
        False,
        700 + experts,
        experts=experts,
        ids=(ids + LOCAL_START).to(torch.int32),
    )
    _check_k3(k, rows, False, True)


def _check_k3(k, rows, last, xq):
    """A K3 launch (two steps) of the case ``k`` against the reference math."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.config import HIDDEN, MAX_TOKENS
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import (
        k3_scratch_layout,
        k3_sym_layout,
    )

    dev = torch.device("cuda")
    w = k.w
    attn_out = torch.full(
        (MAX_TOKENS, HIDDEN), float("nan"), device=dev, dtype=torch.bfloat16
    )
    scratch = torch.zeros(
        scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(
        scratch_bytes(k3_sym_layout(1)) + guard.FLAG_BYTES, None, 0, 1, dev
    )
    peers.bytes.zero_()
    epoch = _epoch(dev)
    launch = build_k3(
        K3Build(tokens=rows, npes=1, xq_pre=xq)
    )  # the rule from the launch's ``last``
    x_arg = k.xin
    if xq:  # the expert input as K2's MX FP8 blocks (``xin`` carries them in a standalone build)
        from test_muse_spark_k2_gfx1250 import _xq_ref

        x_arg = _xq_ref(k.xin)
    new_ref, attn_ref, nb_ref = _reference(k.xin, k.ids, k.scores, LOCAL_START, k.dec13, k.dec2, w, k.stream, last)  # fmt: skip
    got = k.stream.clone()

    def run():
        launch(
            x_arg.data_ptr(), k.ids.data_ptr(), k.scores.data_ptr(), LOCAL_START,
            k.w13.data_ptr(), k.s13.data_ptr(), k.w2.data_ptr(), k.s2.data_ptr(),
            w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
            w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
            w["w_unpool"].data_ptr(), attn_out.data_ptr(), scratch.data_ptr(),
            peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, int(last),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1

    try:
        for step in range(2):  # the second step's tags differ from the first's
            got.copy_(k.stream)
            run()
        torch.cuda.synchronize()
        assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
            "a wait failed"
        )
        _close(got, new_ref, 1 / 32, 2e-2, f"stream S={rows}")
        # the FFN branch itself (the normed unpooled rows), where beta does not hide it
        cols = w["beta"] > 0.1
        nb_got = (got - w["alpha"] * k.stream)[:, cols] / w["beta"][cols]
        _close(nb_got, nb_ref[:, cols], 1 / 16, 3e-2, f"branch S={rows}")
        # (the stream's 2e-2 times the norm's scale: a bf16 ulp of the branch also lands on near-zero elements)
        _close(attn_out[:rows], attn_ref, 1 / 32, 3e-2, f"attention input S={rows}")
        assert attn_out[rows:].isnan().all(), "rows past the step were written"
        if os.environ.get("K3_TIME"):
            med, best = _time_launches(run)
            print(f"K3 S={rows} npes=1 last={int(last)}: {med:.1f} us a launch (median; best {best:.1f}; incl. the epoch bump)")  # fmt: skip
    finally:
        peers.close()


@pytest.mark.parametrize("rows", [8, 16])
def test_k3_compiles_at_tp8(rows):
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import K3_WAVE

    compile_only(build_k3(K3Build(tokens=rows, npes=8, xq_pre=K3_WAVE == 2)))


def test_a_fault_site_compiles():
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import K3_WAVE

    compile_only(
        build_k3(K3Build(tokens=8, fault_site=1, fault_after=1, xq_pre=K3_WAVE == 2))
    )


@pytest.mark.parametrize("rows", [8, 16])
def test_k3_partial_matches_sglang_normal_path(rows):
    """K3's finalized partial rows (``k3_fbuf``: the experts + the per-expert post norm +
    the score-weighted slot sum of this rank, one bf16 rounding) against SGLang's own
    gfx1250 expert path on the same in-place weights: ``mxfp4_moe.run_gfx1250_a8w4``
    (aiter's A8W4 TDM grouped MoE, Muse's finalize)."""
    from sglang.srt.rocm_muse_spark import mxfp4_moe

    if not hasattr(mxfp4_moe, "run_gfx1250_a8w4"):
        pytest.skip("this SGLang has no run_gfx1250_a8w4 (sgl-project/sglang#50707)")
    _needs_xq(False)

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
        POST_EPS,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import (
        k3_scratch_layout,
        k3_sym_layout,
    )

    dev = torch.device("cuda")
    k = k3_case(rows, False, 500 + rows)
    w = k.w
    mask = torch.zeros(256, dtype=torch.int32, device=dev)
    mask[LOCAL_START : LOCAL_START + EXPERT_ROWS] = 1
    ex = SimpleNamespace(
        num_local_experts=EXPERT_ROWS,
        gfx1250_a8w4=(k.w13, k.w2, k.s13, k.s2, mask),
        expert_hidden_size=MOE_HIDDEN,
    )
    normal = mxfp4_moe.run_gfx1250_a8w4(
        k.xin, k.scores, k.ids, ex, LOCAL_START, w["pen_w"], POST_EPS, 0.0
    )
    torch.cuda.synchronize()
    lay = k3_scratch_layout()
    attn_out = torch.zeros((MAX_TOKENS, HIDDEN), device=dev, dtype=torch.bfloat16)
    scratch = torch.zeros(scratch_bytes(lay), dtype=torch.uint8, device=dev)
    peers = PeerBuffer(
        scratch_bytes(k3_sym_layout(1)) + guard.FLAG_BYTES, None, 0, 1, dev
    )
    peers.bytes.zero_()
    epoch = _epoch(dev)
    got = k.stream.clone()
    try:
        build_k3(K3Build(tokens=rows, npes=1))(
            k.xin.data_ptr(), k.ids.data_ptr(), k.scores.data_ptr(), LOCAL_START,
            k.w13.data_ptr(), k.s13.data_ptr(), k.w2.data_ptr(), k.s2.data_ptr(),
            w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
            w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
            w["w_unpool"].data_ptr(), attn_out.data_ptr(), scratch.data_ptr(),
            peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, 0,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        torch.cuda.synchronize()
        off, _ = lay["k3_fbuf"]
        fbuf = (
            scratch[off : off + rows * MOE_HIDDEN * 2]
            .view(torch.bfloat16)
            .reshape(rows, MOE_HIDDEN)
        )
        _close(
            fbuf,
            normal.reshape(rows, MOE_HIDDEN),
            1 / 16,
            2e-2,
            f"finalized partial vs SGLang S={rows}",
        )
        same = (fbuf == normal.reshape(rows, MOE_HIDDEN)).float().mean().item()
        print(
            f"finalized partial vs SGLang S={rows}: {same:.2%} of the bf16 values identical"
        )
    finally:
        peers.close()
