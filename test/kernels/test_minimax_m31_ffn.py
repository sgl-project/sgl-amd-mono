# SPDX-License-Identifier: Apache-2.0
"""The M3.1 FFN launch against native ops on one GPU (TP=1 build): the seam residual,
Gemma post-attention norm, FP32 router, top-4, the routed A16W4 experts against AITER on
SGLang's post-load MXFP4 layout (AITER (16, 16) shuffle, shuffled E8M0 group-32 scales,
separated gate / up), and the MXFP8 shared expert against torch. Adapted from vLLM's
tests/models/minimax_m3/amd_specialized_decode/test_ops.py (Apache-2.0)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("gfx950 only", allow_module_level=True)

from sgl_amd_mono.minimax_m31 import a16w8, ffn  # noqa: E402
from sgl_amd_mono.minimax_m31.runner import FfnRunner  # noqa: E402

R = ffn.EXPERTS
H, INTER = ffn.HIDDEN, ffn.INTER


def _metrics(got, want):
    got, want = got.float().flatten(), want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    rel = ((got - want).norm() / want.norm()).item()
    return cos, rel


@pytest.fixture(scope="module")
def weights():
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scales(*shape):
        # e8m0 codes around 2^-7: gate / up pre-activations of a few units
        return torch.randint(
            119, 123, shape, dtype=torch.uint8, device=dev, generator=g
        )

    def kernel_format(w, s):
        # what SGLang's process_requantized_weights leaves for aiter's fused MoE
        rows = w.shape[1]
        s = e8m0_shuffle(s.view(R * rows, -1)).view(R, rows, -1)
        w = shuffle_weight(w.contiguous(), (16, 16)).view(torch.float4_e2m1fn_x2)
        w.is_shuffled = True
        return w, s

    w13, s13 = kernel_format(codes(R, 2 * INTER, H // 2), scales(R, 2 * INTER, H // 32))
    w2, s2 = kernel_format(codes(R, H, INTER // 2), scales(R, H, INTER // 32))

    def mxfp8(rows, k):
        q = torch.randn(rows, k, device=dev, generator=g).mul(2).to(torch.float8_e4m3fn)
        return q, scales(rows, k // 32)

    gate_up, gate_up_scale = mxfp8(2 * INTER, H)
    down, down_scale = mxfp8(H, INTER)
    w13s, s13s = a16w8.layout(gate_up, gate_up_scale)
    w2s, s2s = a16w8.layout(down, down_scale)
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        w13s=w13s,
        s13s=s13s,
        w2s=w2s,
        s2s=s2s,
        shared=SimpleNamespace(
            gate_up=gate_up,
            gate_up_scale=gate_up_scale,
            down=down,
            down_scale=down_scale,
        ),
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(ffn.EXPERTS, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(ffn.EXPERTS, device=dev, generator=g).mul(0.01),
    )


def _native_experts(x, w, ids, wts):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    return fused_moe(
        x,
        w.w13,
        w.w2,
        wts,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w.s13,
        w2_scale=w.s2,
        gate_mode="separated",
        swiglu_limit=ffn.LIMIT,
    )


def _native_shared(x, w):
    """The shared expert on its MXFP8 values, MID in bf16 as the launch keeps it."""

    def values(q, scale):
        return q.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)

    h = x.float() @ values(w.gate_up, w.gate_up_scale).T
    gate = h[:, :INTER].clamp(max=ffn.LIMIT)
    up = h[:, INTER:].clamp(-ffn.LIMIT, ffn.LIMIT)
    mid = (gate * torch.sigmoid(gate * ffn.ALPHA) * (up + ffn.BETA)).bfloat16()
    return mid.float() @ values(w.down, w.down_scale).T


def _native_norm(part, residual, weight):
    total = part.float() + residual.float()
    var = total.pow(2).mean(-1, keepdim=True)
    x = total * torch.rsqrt(var + ffn.EPS) * (1.0 + weight.float())
    return x.bfloat16(), total.bfloat16()


def _native_route(logits, bias):
    scores = logits.sigmoid()
    ids = (scores + bias).topk(ffn.TOPK, dim=-1).indices
    wts = scores.gather(1, ids)
    wts = wts / wts.sum(-1, keepdim=True) * ffn.ROUTED_SCALE
    return ids.int(), wts


def _clear_choices(logits, bias, margin):
    top = (logits.sigmoid() + bias).topk(ffn.TOPK + 1, dim=-1).values
    return (top[:, ffn.TOPK - 1] - top[:, ffn.TOPK]) > margin


def _pairs(region, n):
    return region[: n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _runner(weights, s):
    dev = torch.device("cuda")
    buf = torch.zeros(ffn.peer_bytes(1), dtype=torch.uint8, device=dev)
    peers = torch.tensor([buf.data_ptr()], dtype=torch.int64, device=dev)
    runner = FfnRunner(1, 0, buf.data_ptr(), peers, dev, (s,))
    runner.warm(weights)
    runner.peer_bufs = [buf]
    return runner


def _run(weights, s, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    part = torch.randn(s, H, device="cuda", generator=g).bfloat16()
    residual = torch.randn(s, H, device="cuda", generator=g).mul(4).bfloat16()
    runner = _runner(weights, s)
    slots = torch.arange(s, dtype=torch.int64, device="cuda")
    out, res_out = runner(part, residual, weights, slots)
    torch.cuda.synchronize()
    err = runner.peer_bufs[0][: 4 * ffn.ERROR_WORDS].view(torch.int32).tolist()
    assert err[0] == 0, err
    x = runner.region(s, "xn")[: s * H * 2].view(torch.bfloat16).view(s, H)
    parts = _pairs(runner.region(s, "logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s, ffn.EXPERTS, ffn.ROUTER_PARTS)
    route = _pairs(runner.region(s, "route"), s * 2 * ffn.TOPK).view(s, 2 * ffn.TOPK)
    return SimpleNamespace(
        part=part,
        residual=residual,
        out=out,
        res_out=res_out,
        x=x,
        logits=parts.sum(-1),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
    )


@pytest.mark.parametrize("s", [1, 3, 8, 16])
def test_ffn_stages_match_native(weights, s):
    r = _run(weights, s)
    want_x, want_res = _native_norm(r.part, r.residual, weights.post_norm)
    torch.testing.assert_close(r.res_out, want_res, rtol=0, atol=0)
    cos, rel = _metrics(r.x, want_x)
    assert cos > 0.99999 and rel < 4e-3, (cos, rel)

    want_logits = r.x.float() @ weights.router.T
    torch.testing.assert_close(r.logits, want_logits, rtol=1e-4, atol=1e-4)

    ids, wts = _native_route(r.logits, weights.router_bias)
    clear = _clear_choices(r.logits, weights.router_bias, 1e-4)
    assert clear.any()
    got_ids, order = r.ids.sort(-1)
    want_ids, want_order = ids.sort(-1)
    assert torch.equal(got_ids[clear], want_ids[clear])
    torch.testing.assert_close(
        r.wts.gather(1, order)[clear],
        wts.gather(1, want_order)[clear],
        rtol=1e-5,
        atol=1e-6,
    )

    want = _native_experts(r.x, weights, r.ids, r.wts).float()
    want = want + _native_shared(r.x, weights.shared)
    cos, rel = _metrics(r.out, want)
    assert cos > 0.9999 and rel < 1e-2, (cos, rel)
