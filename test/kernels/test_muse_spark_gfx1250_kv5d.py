"""The Muse Spark K1 and K-attn launches on SGLang's bf16 ``vectorized_5d`` KV caches with 64-key pages
(--kv-cache-dtype bf16 --page-size 64, SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d), one gfx1250 GPU:
K [page][head][dim / 8][key][8], V [page][head][key / 8][dim][8].

K1 must write exactly the bf16 K / V it hands out (``k_out`` / ``v_out``) at the tokens' slots and nothing else;
K-attn must match the reference attention over the keys the page table maps (as ``test_muse_spark_gfx1250_kattn``
checks the fp8 NHD caches)."""

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

PAGE = 64
PAGES = 4096  # cache pages of the test's K / V (262k keys)
MAX_BLOCKS = 4096  # the page table's row


def pack5d(k, v):
    """Logical [slots, heads, dim] K / V (slots a multiple of PAGE) as the vectorized_5d caches."""
    slots, heads, dim = k.shape
    pages = slots // PAGE
    k5 = k.view(pages, PAGE, heads, dim // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    v5 = v.view(pages, PAGE // 8, 8, heads, dim).permute(0, 3, 1, 4, 2).contiguous()
    return k5, v5


def unpack5d(k5, v5):
    """The vectorized_5d caches as logical [slots, heads, dim] K / V."""
    pages, heads, dc, page, x = k5.shape
    k = k5.permute(0, 3, 1, 2, 4).reshape(pages * page, heads, dc * x)
    v = v5.permute(0, 2, 4, 1, 3).reshape(pages * page, heads, v5.shape[3])
    return k, v


def test_pack5d_is_sglangs_layout():
    """``pack5d`` against the index formulas of SGLang's reshape_and_cache_shuffle_5d (CPU)."""
    heads, dim, pages = 2, 64, 3
    k = torch.arange(pages * PAGE * heads * dim, dtype=torch.float32).view(
        pages * PAGE, heads, dim
    )
    v = -k
    k5, v5 = pack5d(k, v)
    X = 8
    for slot in (0, 1, 7, 8, 63, 64, 100, 191):
        for h in range(heads):
            for d in (0, 1, 7, 8, 33, 63):
                blk, sp = slot // PAGE, slot % PAGE
                kt = (
                    blk * heads * dim * PAGE
                    + h * dim * PAGE
                    + (d // X) * PAGE * X
                    + sp * X
                    + d % X
                )
                vt = (
                    blk * heads * dim * PAGE
                    + h * dim * PAGE
                    + (sp // X) * dim * X
                    + d * X
                    + sp % X
                )
                assert (
                    k5.reshape(-1)[kt] == k[slot, h, d]
                    and v5.reshape(-1)[vt] == v[slot, h, d]
                )
    k2, v2 = unpack5d(k5, v5)
    assert (k2 == k).all() and (v2 == v).all()


@pytest.mark.parametrize(
    "rows,rope,swa,slot64",
    [(8, True, False, True), (16, False, True, False), (8, True, True, True)],
)
def test_k1_writes_the_5d_cache(rows, rope, swa, slot64):
    import test_muse_spark_gfx1250_k1 as K1T

    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        HIDDEN,
        KV_K,
        LOCAL_KV_HEADS,
        MAX_TOKENS,
        Q_K,
        QKVG_ROWS,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layer import K1Build, build_k1
    from sgl_amd_mono.muse_spark.gfx1250.layout import k1_scratch_layout, scratch_bytes

    torch.manual_seed(100 + rows)
    dev, bf = torch.device("cuda"), torch.bfloat16
    SLOTS = PAGES * PAGE
    x = torch.randn(rows, HIDDEN, device=dev).to(bf)
    w = (torch.randn(QKVG_ROWS, HIDDEN, device=dev) * 0.02).to(bf)
    cos_sin = K1T._cos_sin(dev)
    positions = torch.randint(0, K1T.POSITIONS, (rows,), device=dev, dtype=torch.int64)
    full = torch.randperm(SLOTS, device=dev)[:rows]
    slots = full if slot64 else full.to(torch.int32)
    swa_map = torch.randperm(SLOTS, device=dev, dtype=torch.int64)
    k5 = torch.zeros(
        PAGES, LOCAL_KV_HEADS, HEAD_DIM // 8, PAGE, 8, device=dev, dtype=bf
    )
    v5 = torch.zeros(
        PAGES, LOCAL_KV_HEADS, PAGE // 8, HEAD_DIM, 8, device=dev, dtype=bf
    )
    q = torch.full((MAX_TOKENS, Q_K), float("nan"), device=dev, dtype=bf)
    k = torch.full((MAX_TOKENS, KV_K), float("nan"), device=dev, dtype=bf)
    v = torch.full_like(k, float("nan"))
    g = torch.full_like(q, float("nan"))
    scratch = torch.zeros(
        scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=dev
    )
    epoch = K1T._epoch(dev)
    launch = build_k1(K1Build(tokens=rows, kv5d=True))
    for step in range(2):
        launch(
            x.data_ptr(), w.data_ptr(), cos_sin.data_ptr(), positions.data_ptr(),
            slots.data_ptr(), swa_map.data_ptr(), k5.data_ptr(), v5.data_ptr(),
            q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), scratch.data_ptr(),
            epoch.data_ptr(), 3, int(rope), int(swa), int(slot64), 1,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1
    torch.cuda.synchronize()
    q_ref, k_ref, v_ref, g_ref = K1T._reference(x, w, cos_sin, positions, rope)
    K1T._close(q[:rows], q_ref, 1 / 64, "q")
    K1T._close(k[:rows], k_ref, 1 / 64, "k")
    K1T._close(v[:rows], v_ref, 1 / 128, "v")
    kl, vl = unpack5d(k5, v5)
    at = swa_map[full] if swa else full
    # the cache holds the launch's own bf16 outputs, bit for bit
    assert (
        kl[at].reshape(rows, -1).view(torch.int16) == k[:rows].view(torch.int16)
    ).all(), "K cache != k_out"
    assert (
        vl[at].reshape(rows, -1).view(torch.int16) == v[:rows].view(torch.int16)
    ).all(), "V cache != v_out"
    untouched = torch.ones(SLOTS, dtype=torch.bool, device=dev)
    untouched[at] = False
    assert (kl[untouched].float() == 0).all() and (vl[untouched].float() == 0).all(), (
        "a write outside the slots"
    )


def _reference(q, kl, vl, table, seq_lens, ql, window, scale):
    """``test_muse_spark_gfx1250_kattn._reference`` over logical bf16 K / V and 64-key pages."""
    from sgl_amd_mono.muse_spark.gfx1250.config import HEAD_DIM, LOCAL_KV_HEADS, Q_K

    rows, _ = q.shape
    R = rows // ql
    out = torch.zeros(rows, Q_K, device=q.device)
    for r in range(R):
        keys = int(seq_lens[r])
        ctx = keys - ql
        slots = (
            table[r, : (keys + PAGE - 1) // PAGE].long()[:, None] * PAGE
            + torch.arange(PAGE, device=q.device)[None, :]
        )
        slots = slots.reshape(-1)[:keys]
        K, V = kl[slots].float(), vl[slots].float()
        for h in range(LOCAL_KV_HEADS):
            for qpos in range(ql):
                t = r * ql + qpos
                qh = q[t].float().reshape(-1, HEAD_DIM)[h * 8 : (h + 1) * 8]
                s = scale * (qh @ K[:, h].T)
                kidx = torch.arange(keys, device=q.device)
                ok = kidx < ctx + qpos + 1
                if window:
                    ok &= (ctx + qpos - kidx) < window
                s = torch.where(ok[None, :], s, torch.full_like(s, float("-inf")))
                p = torch.softmax(s, dim=-1).to(torch.bfloat16).float()
                out[t, (h * 8) * HEAD_DIM : (h + 1) * 8 * HEAD_DIM] = (
                    p @ V[:, h]
                ).reshape(-1)
    return out


@pytest.mark.parametrize(
    "ql,R,window,part,lens,kind",
    [
        (8, 1, 0, 0, [300], ""),
        (8, 1, 0, 0, [8192], "full"),
        (8, 1, 2047, 0, [8192], "sliding"),
        (8, 2, 2047, 0, [3000, 2200], "sliding"),
        (8, 2, 0, 0, [8192, 8192], "full"),
        (8, 4, 0, 1024, [5000, 17, 700, 2049], ""),
        (8, 2, 2047, 0, [95000, 60000], ""),
        (8, 2, 0, 0, [95000, 60000], "full"),
        (8, 1, 0, 0, [65], ""),  # one key into the second page
    ],
)
def test_kattn_reads_the_5d_cache(ql, R, window, part, lens, kind):
    import test_muse_spark_gfx1250_kattn as KT

    from sgl_amd_mono.muse_spark.gfx1250.config import (
        HEAD_DIM,
        LOCAL_KV_HEADS,
        Q_K,
        SOFTMAX_SCALE,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layer import KAttnBuild, build_kattn
    from sgl_amd_mono.muse_spark.gfx1250.layout import (
        ATTN_CHUNK,
        kattn_scratch_layout,
        scratch_bytes,
    )

    part = -(-part // ATTN_CHUNK) * ATTN_CHUNK
    torch.manual_seed(11 * ql + R + len(kind))
    dev, bf = torch.device("cuda"), torch.bfloat16
    S = ql * R
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    q = (torch.randn(S, Q_K, device=dev) * 5.47).to(bf)
    kl = (torch.randn(PAGES * PAGE, LOCAL_KV_HEADS, HEAD_DIM, device=dev) * 2).to(bf)
    vl = torch.randn(PAGES * PAGE, LOCAL_KV_HEADS, HEAD_DIM, device=dev).to(bf)
    k5, v5 = pack5d(kl, vl)
    perm = torch.randperm(PAGES - 1, device=dev) + 1
    table = torch.zeros(R, MAX_BLOCKS, dtype=torch.int32, device=dev)
    at = 0
    for r in range(R):
        n = (lens[r] + PAGE - 1) // PAGE
        assert at + n <= PAGES - 1, "the test's pages do not cover these lengths"
        table[r, :n] = perm[at : at + n].to(torch.int32)
        at += n
    o = torch.full((S, Q_K), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(kattn_scratch_layout(S, ql)), dtype=torch.uint8, device=dev
    )
    epoch = KT._epoch(dev)
    launch = build_kattn(KAttnBuild(tokens=S, q_len=ql, kind=kind, kv5d=True))
    scale_bits = struct.unpack("i", struct.pack("f", SOFTMAX_SCALE))[0]
    prefix_lens = (seq_lens - ql).contiguous()
    for step in range(2):
        launch(
            q.data_ptr(), o.data_ptr(), k5.data_ptr(), v5.data_ptr(),
            table.data_ptr(), MAX_BLOCKS, table.numel() * 4, prefix_lens.data_ptr(), 0,
            ql, window, part, scale_bits, scratch.data_ptr(), epoch.data_ptr(), 11,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1
    torch.cuda.synchronize()
    ref = _reference(q, kl, vl, table, seq_lens, ql, window, SOFTMAX_SCALE)
    got = o.float()
    assert not got.isnan().any(), "NaN in the attention output"
    err = (got - ref).abs()
    tol = ref.abs() * (1 / 32) + 2e-2
    bad = (err > tol).float().mean().item()
    assert bad == 0.0, f"{bad:.2%} of the outputs off (max err {err.max().item():.4f})"
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).min().item()
    assert cos > 0.999, f"least row cosine {cos}"
