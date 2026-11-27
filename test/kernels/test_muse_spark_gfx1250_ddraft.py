"""The DFlash2 draft's D-draft launch (its five layers, each D-pre -> D-attn -> D-post ->
D-MLP in one launch) on one gfx1250 GPU (a TP group of one) against the same chain
in torch with the kernels' rounding points: bf16 at h, hc, qkv, the normed / rotated
q and k, P, o, the rank partials, attn_out, g, u, a and out; fp8 K / V in the cache;
the GEMVs' fp32 sums as exact as fp64 makes them."""

import struct

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx125" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx1250 GPU",
)

PAGES = 2048  # cache pages of a layer's K / V planes
MAX_CTX = 4096  # req_to_token's row (tokens)
MAX_POS = 8192
EPS = 1e-5


def _epoch(device, deadline_s=30.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


GEMV_DT = [torch.float64]


def _gemv(x, w):
    """bf16 x [S, K] @ bf16 w [N, K]^T, the fp32 sums as exact as fp64 (``GEMV_DT``)."""
    return (x.to(GEMV_DT[0]) @ w.to(GEMV_DT[0]).T).float()


def _conv32(x, delta, base, D):
    rows = x.shape[0]
    xb = x.reshape(rows, D.GROUPS, D.GROUP)
    c = base.float().reshape(1, D.TAPS, D.GROUPS, D.GROUP) + delta.reshape(
        rows, D.TAPS, D.GROUPS, 1
    )
    pos = torch.arange(rows, device=x.device) % D.BLOCK
    prev = torch.cat([torch.zeros_like(xb[:1]), xb[:-1]], 0) * (pos >= 1).view(-1, 1, 1)
    return (c[:, 0] * xb + c[:, 1] * prev).reshape(rows, D.H)


def _rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(
        torch.bfloat16
    )


def _pre(res2, w, slots, positions, cos_sin, D):
    """D-pre from its residual: (coef fp32 [S, 2, TAPS, GROUPS], q bf16, k bf16, v bf16)."""
    bf = torch.bfloat16
    S = res2.shape[0]
    h = _rms(res2, w["w_in"])
    coef = _gemv(h, w["w_kpa"]).reshape(S, 2, D.TAPS, D.GROUPS)
    hc = _conv32(h.float(), coef[:, 0], w["base_a"][0], D).to(bf)
    heads = _gemv(hc, w["w_qkv"]).to(bf).float().reshape(S, 10, 128)
    inv = torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + EPS)
    wn = torch.stack([w["q_norm"]] * 8 + [w["k_norm"]]).float()  # [9, 128]
    n = (heads[:, :9] * inv[:, :9] * wn[None]).to(bf).float()
    cs = cos_sin[positions.long()].float()
    cos, sin = cs[:, None, :64], cs[:, None, 64:]
    x1, x2 = n[..., :64], n[..., 64:]
    rot = torch.cat([(x1 * cos - x2 * sin).to(bf), (x2 * cos + x1 * sin).to(bf)], -1)
    return coef, rot[:, :8].reshape(S, 1024), rot[:, 8], heads[:, 9].to(bf)


def _fp8(t):
    return t.float().clamp(-448, 448).to(torch.float8_e4m3fn)


def _attn(q, kc, vc, ctx, r2t, rp, scale):
    """Every block row over the whole prefix and the whole block, P bf16, o bf16."""
    S = q.shape[0]
    o = torch.zeros(S, 1024, device=q.device)
    for r in range(S // 8):
        keys = int(ctx[r]) + 8
        sl = r2t[int(rp[r]), :keys].long()
        K, V = kc[sl, 0].float(), vc[sl, 0].float()
        for t in range(r * 8, r * 8 + 8):
            qh = q[t].float().reshape(8, 128)
            p = torch.softmax(scale * (qh @ K.T), -1).to(torch.bfloat16).float()
            o[t] = (p @ V).reshape(-1)
    return o.to(torch.bfloat16)


def _post(o, coef, w, D):
    d = _gemv(o, w["w_o"]).to(torch.bfloat16).float()
    return _conv32(d, coef[:, 1], w["base_a"][1], D).to(torch.bfloat16)


def _mlp(res2, attn_out, w, D):
    """D-MLP: (out, res_out)."""
    bf = torch.bfloat16
    S = res2.shape[0]
    res3 = (res2.float() + attn_out.float()).to(bf)
    h = _rms(res3, w["w_post"])
    coef = _gemv(h, w["w_kp"]).reshape(S, 2, D.TAPS, D.GROUPS)
    hc = _conv32(h.float(), coef[:, 0], w["base"][0], D).to(bf)
    gu = _gemv(hc, w["w_gu"])
    g, u = gu[:, : D.I].to(bf).float(), gu[:, D.I :].to(bf).float()
    a = (torch.nn.functional.silu(g).to(bf).float() * u).to(bf)
    dd = _gemv(a, w["w_down"]).to(bf).float()
    return _conv32(dd, coef[:, 1], w["base"][1], D).to(bf), res3


def _layer(x, res, first, w, kc, vc, ctx, r2t, rp, slots, positions, cos_sin, scale):
    """One draft layer in torch: (out, res_out); writes the block's K / V into kc / vc."""
    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp as D

    res2 = x if first else (res.float() + x.float()).to(torch.bfloat16)
    coef, q, k, v = _pre(res2, w, slots, positions, cos_sin, D)
    kc[slots.long(), 0] = _fp8(k)
    vc[slots.long(), 0] = _fp8(v)
    o = _attn(q, kc, vc, ctx, r2t, rp, scale)
    return _mlp(res2, _post(o, coef, w, D), w, D)


def _metrics(got, ref, rtol, atol):
    got, ref = got.float(), ref.float()
    bad = ((got - ref).abs() > rtol * ref.abs() + atol).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).min().item()
    return bad, cos


def _close(got, ref, rtol, atol, what, bad_max, cos_min):
    bad, cos = _metrics(got, ref, rtol, atol)
    assert bad < bad_max and cos >= cos_min, (
        f"{what}: {bad:.3%} off, least row cosine {cos:.5f}"
    )


@pytest.mark.parametrize("rows", [8, 16])
def test_ddraft_matches_the_draft_layers(rows):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn as DA
    from sgl_amd_mono.muse_spark.gfx1250 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx1250 import draft_layer as DL
    from sgl_amd_mono.muse_spark.gfx1250 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes

    dev = torch.device("cuda")
    torch.manual_seed(700 + rows)
    bf = torch.bfloat16
    S, R = rows, rows // 8
    L = DL.LAYERS

    def rnd(*shape, s=1.0, mean=0.0):
        return (torch.randn(*shape, device=dev) * s + mean).to(bf)

    ws = []
    for _ in range(L):
        ws.append({
            "w_in": rnd(D.H, s=0.1, mean=1.0), "w_kpa": rnd(D.COEF_ROWS, D.H, s=0.01),
            "base_a": rnd(2, D.TAPS, D.H, s=0.05, mean=1.0), "w_qkv": rnd(DH.QKV_ROWS, D.H, s=0.02),
            "q_norm": rnd(128, s=0.1, mean=1.0), "k_norm": rnd(128, s=0.1, mean=1.0),
            "w_o": rnd(D.H, 1024, s=0.03), "w_post": rnd(D.H, s=0.1, mean=1.0),
            "w_kp": rnd(D.COEF_ROWS, D.H, s=0.01), "base": rnd(2, D.TAPS, D.H, s=0.05, mean=1.0),
            "w_gu": rnd(D.GU_ROWS, D.H, s=0.02), "w_down": rnd(D.H, D.I, s=0.02),
        })  # fmt: skip
    wtab = torch.tensor([ws[i][nm].data_ptr() for i in range(L) for nm in DL.WEIGHTS], dtype=torch.int64, device=dev)  # fmt: skip
    ang = (
        torch.arange(MAX_POS, device=dev)[:, None]
        * (10000.0 ** (-torch.arange(64, device=dev) / 64))[None, :]
    )
    cos_sin = torch.cat([torch.cos(ang), torch.sin(ang)], 1).contiguous()
    # the prefix keys of each layer's cache (random fp8), the block's slots after them
    ctx = [int(x) for x in torch.randint(100, 1500, (R,))]
    perm = torch.randperm(PAGES - 1, device=dev) + 1
    r2t = torch.zeros(R + 2, MAX_CTX, dtype=torch.int32, device=dev)
    rp = torch.randperm(R + 2, device=dev)[:R].to(torch.int64)
    slots, positions = [], []
    at = 0
    for r in range(R):
        keys = ctx[r] + 8
        n = (keys + 15) // 16
        sl = (
            perm[at : at + n][:, None] * 16 + torch.arange(16, device=dev)[None, :]
        ).reshape(-1)[:keys]
        r2t[int(rp[r]), :keys] = sl.to(torch.int32)
        slots.append(sl[ctx[r] :])
        positions.append(torch.arange(ctx[r], keys, device=dev))
        at += n
    slots = torch.cat(slots).to(torch.int32)
    positions = torch.cat(positions).to(torch.int32)
    kvs = [
        (torch.randn(PAGES * 16, 1, 128, device=dev) * 1.5).to(torch.float8_e4m3fn)
        for _ in range(2 * L)
    ]
    kvs_ref = [t.clone() for t in kvs]
    x = rnd(S, D.H)
    scale = 128**-0.5
    # the reference chain (the drift check of the five layers)
    xr, rr = x, x
    for i in range(L):
        xr, rr = _layer(xr, rr, i == 0, ws[i], kvs_ref[2 * i], kvs_ref[2 * i + 1], ctx, r2t, rp, slots, positions, cos_sin, scale)  # fmt: skip
    # the launch
    lens = torch.tensor(ctx, dtype=torch.int32, device=dev)
    nan = float("nan")
    res_pre = torch.full((S, D.H), nan, device=dev, dtype=bf)
    q = torch.full((S, 1024), nan, device=dev, dtype=bf)
    coef = torch.full((S, D.COEF_ROWS), nan, device=dev)
    o = torch.full((S, 1024), nan, device=dev, dtype=bf)
    attn_out = torch.full((S, D.H), nan, device=dev, dtype=bf)
    res_out = torch.full((S, D.H), nan, device=dev, dtype=bf)
    out = torch.full((S, D.H), nan, device=dev, dtype=bf)

    def zeros(nbytes):
        return torch.zeros(nbytes, dtype=torch.uint8, device=dev)

    s_pre = zeros(scratch_bytes(DH.dp_scratch_layout()))
    s_attn = zeros(scratch_bytes(DA.dattn_scratch_layout(S, 8)))
    s_post = zeros(scratch_bytes(DH.do_scratch_layout()))
    s_mlp = zeros(scratch_bytes(D.dm_scratch_layout()))
    flags = zeros(scratch_bytes(DL.dl_flags_layout()))
    p_post = PeerBuffer(DH.do_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    p_mlp = PeerBuffer(D.dm_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    p_post.bytes.zero_()
    p_mlp.bytes.zero_()
    epoch = _epoch(dev)
    launch = DL.build_ddraft(DL.DDraftBuild(tokens=S, npes=1))
    sb = struct.unpack("i", struct.pack("f", scale))[0]
    try:
        launch(
            wtab.data_ptr(), *[t.data_ptr() for t in kvs], x.data_ptr(), cos_sin.data_ptr(), positions.data_ptr(), 0,
            slots.data_ptr(), 0, res_pre.data_ptr(), q.data_ptr(), coef.data_ptr(), s_pre.data_ptr(), o.data_ptr(),
            r2t.data_ptr(), MAX_CTX, r2t.numel() * 4, rp.data_ptr(), 1, lens.data_ptr(), 0, sb, s_attn.data_ptr(),
            attn_out.data_ptr(), s_post.data_ptr(), p_post.local, p_post.addresses.data_ptr(), res_out.data_ptr(),
            out.data_ptr(), s_mlp.data_ptr(), p_mlp.local, p_mlp.addresses.data_ptr(), flags.data_ptr(), 0,
            epoch.data_ptr(), stream=torch.cuda.current_stream(),
        )  # fmt: skip
        torch.cuda.synchronize()
        assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
            "a wait failed"
        )
        assert not out.isnan().any() and not res_out.isnan().any(), "NaN in the outputs"
        # the last layer, stage by stage from the launch's own inputs to each stage
        w4, kc, vc = ws[L - 1], kvs[2 * L - 2], kvs[2 * L - 1]
        coef_ref, q_ref, k_ref, v_ref = _pre(res_pre, w4, slots, positions, cos_sin, D)
        _close(
            coef,
            coef_ref.reshape(S, -1),
            1 / 64,
            1e-3,
            "D-pre coef",
            bad_max=0.001,
            cos_min=0.99999,
        )
        _close(q, q_ref, 1 / 32, 1e-2, "D-pre q", bad_max=0.002, cos_min=0.9999)
        _close(
            kc[slots.long(), 0],
            _fp8(k_ref),
            1 / 8,
            5e-2,
            "D-pre k (fp8)",
            bad_max=0.002,
            cos_min=0.999,
        )
        _close(
            vc[slots.long(), 0],
            _fp8(v_ref),
            1 / 8,
            5e-2,
            "D-pre v (fp8)",
            bad_max=0.002,
            cos_min=0.999,
        )
        # (in the launch the attention's o rows go to the o box: tagged bf16 pairs)
        ob_off, ob_bytes = DL.dl_flags_layout()["dl_obox"]
        o = (
            flags[ob_off : ob_off + ob_bytes]
            .view(torch.int32)
            .reshape(-1, 2)[:, 0]
            .contiguous()
        )
        o = o.view(bf)[: S * 1024].reshape(S, 1024)
        _close(
            o,
            _attn(q, kc, vc, ctx, r2t, rp, scale),
            1 / 32,
            2e-2,
            "D-attn o",
            bad_max=0.001,
            cos_min=0.9999,
        )
        _close(attn_out, _post(o, coef.reshape(S, 2, D.TAPS, D.GROUPS), w4, D), 1 / 32, 2e-2, "D-post attn_out", bad_max=0.002, cos_min=0.9999)  # fmt: skip
        out_ref, res_ref = _mlp(res_pre, attn_out, w4, D)
        assert torch.equal(res_out, res_ref), (
            "D-MLP res_out is not bf16(res' + attn_out)"
        )
        _close(out, out_ref, 1 / 32, 2e-2, "D-MLP out", bad_max=0.002, cos_min=0.9999)
        # the five layers end to end: this random-weight chain amplifies each layer's bf16
        # rounding flips (the torch chain with fp32 GEMV sums drifts from the fp64 one by as
        # much), so only a loose bound
        _close(
            out,
            xr,
            1 / 8,
            1e-1,
            "the draft's out after five layers",
            bad_max=0.25,
            cos_min=0.998,
        )
    finally:
        p_post.close()
        p_mlp.close()


@pytest.mark.parametrize("rows", [8, 16])
def test_widths_compile(rows):
    from sgl_amd_mono.muse_spark.gfx1250 import draft_layer as DL
    from sgl_amd_mono.muse_spark.gfx1250.runner import compile_only

    compile_only(DL.build_ddraft(DL.DDraftBuild(tokens=rows, npes=8)))
