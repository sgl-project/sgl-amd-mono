"""The Muse Spark K-attn launch on one gfx950 GPU against the verify attention
of the ROCm model's kernels over an fp8 page-16 NHD cache: causal rows of a
request, the 2047-key sliding window, several partitions a request and their
merge, contexts up to ~95k keys; and that it compiles for the served widths."""

import struct

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

PAGES = 16384  # cache pages of the test's K / V planes (262k keys)
MAX_BLOCKS = 16384  # the page table's row


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _reference(q, k_cache, v_cache, table, seq_lens, ql, window, scale):
    """Per request and head: causal, windowed softmax attention over the keys the
    page table maps, K / V exact in fp32, P rounded to bf16 as the kernels."""
    from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, Q_K

    rows, _ = q.shape
    R = rows // ql
    out = torch.zeros(rows, Q_K, device=q.device)
    for r in range(R):
        keys = int(seq_lens[r])
        ctx = keys - ql
        slots = table[r, : (keys + 15) // 16].long()[:, None] * 16 + torch.arange(16, device=q.device)[None, :]
        slots = slots.reshape(-1)[:keys]
        K = k_cache[slots].float()  # [keys, KV_HEADS, 64]
        V = v_cache[slots].float()
        for h in range(LOCAL_KV_HEADS):
            for qpos in range(ql):
                t = r * ql + qpos
                qh = q[t].float().reshape(-1, HEAD_DIM)[h * 8 : (h + 1) * 8]  # [8, 64]
                s = scale * (qh @ K[:, h].T)  # [8, keys]
                kidx = torch.arange(keys, device=q.device)
                ok = kidx < ctx + qpos + 1
                if window:
                    ok &= (ctx + qpos - kidx) < window
                s = torch.where(ok[None, :], s, torch.full_like(s, float("-inf")))
                p = torch.softmax(s, dim=-1).to(torch.bfloat16).float()
                o = p @ V[:, h]  # [8, 64]
                out[t, (h * 8) * HEAD_DIM : (h + 1) * 8 * HEAD_DIM] = o.reshape(-1)
    return out


@pytest.mark.parametrize(
    "ql,R,window,part,lens",
    [
        (8, 1, 0, 256, [300]),
        (8, 2, 2047, 256, [3000, 2200]),
        (8, 4, 0, 1024, [5000, 17, 700, 2049]),
        (8, 8, 2047, 0, None),  # the kernel sizes the partition
        (8, 8, 0, 0, [20000, 300, 8, 95000, 4096, 2048, 2047, 9]),  # C8 at the AgentX context (a request holds >= its 8 rows)
        (8, 2, 2047, 0, [95000, 60000]),
        (1, 16, 2047, 256, None),
    ],
)
def test_kattn_matches_the_verify_attention(ql, R, window, part, lens):
    from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, Q_K, SOFTMAX_SCALE
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn
    from sgl_amd_mono.muse_spark.gfx950.layout import kattn_scratch_layout, scratch_bytes

    torch.manual_seed(7 * ql + R)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    S = ql * R
    if lens is None:
        lens = [int(x) for x in torch.randint(ql, 2600, (R,))]  # keys include the step's ql rows
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    q = (torch.randn(S, Q_K, device=dev) * 5.47).to(bf)  # K1 hands q scaled by qk_scale_factor
    k_cache = (torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=dev) * 2).to(torch.float8_e4m3fn)
    v_cache = (torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=dev)).to(torch.float8_e4m3fn)
    # distinct pages a request, the rest of the rows zero
    perm = torch.randperm(PAGES - 1, device=dev) + 1
    table = torch.zeros(R, MAX_BLOCKS, dtype=torch.int32, device=dev)
    at = 0
    for r in range(R):
        n = (lens[r] + 15) // 16
        assert at + n <= PAGES - 1, "the test's pages do not cover these lengths"
        table[r, :n] = perm[at : at + n].to(torch.int32)
        at += n
    scale = SOFTMAX_SCALE
    o = torch.full((S, Q_K), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(scratch_bytes(kattn_scratch_layout(S, ql)), dtype=torch.uint8, device=dev)
    epoch = _epoch(dev)
    launch = build_kattn(KAttnBuild(tokens=S, q_len=ql))
    scale_bits = struct.unpack("i", struct.pack("f", scale))[0]
    prefix_lens = (seq_lens - ql).contiguous()  # the keys before the step's rows
    for step in range(2):
        launch(
            q.data_ptr(), o.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
            table.data_ptr(), MAX_BLOCKS, table.numel() * 4, prefix_lens.data_ptr(), 0,
            ql, window, part, scale_bits, scratch.data_ptr(), epoch.data_ptr(), 11,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1
    torch.cuda.synchronize()
    ref = _reference(q, k_cache, v_cache, table, seq_lens, ql, window, scale)
    got = o.float()
    assert not got.isnan().any(), "NaN in the attention output"
    err = (got - ref).abs()
    tol = ref.abs() * (1 / 32) + 2e-2
    bad = (err > tol).float().mean().item()
    assert bad == 0.0, f"{bad:.2%} of the outputs off (max err {err.max().item():.4f})"
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).min().item()
    assert cos > 0.999, f"least row cosine {cos}"


@pytest.mark.parametrize("rows,ql", [(8, 8), (16, 8), (32, 8), (64, 8), (16, 1)])
def test_widths_compile(rows, ql):
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_kattn(KAttnBuild(tokens=rows, q_len=ql)))
