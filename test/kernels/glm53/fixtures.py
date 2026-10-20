"""Nonzero shuffled-FP8 fixtures shared by CP and cross-version GLM checks."""

import hashlib
import math

import torch

from sgl_amd_mono.glm53.gfx950.packing import pack_bf16, pack_fp8, pack_ptpc_fp8
from sgl_amd_mono.glm53.gfx950.runner import (
    INDEX_PAGE_BYTES,
    Glm53Caches,
    Glm53Step,
    Glm53Weights,
)

HIDDEN, INDEX_DIM, TOPK = 6144, 128, 2048


def make_weights(tp, device, ties=False, seed=719):
    g = torch.Generator(device=device).manual_seed(seed)
    heads, inter = 64 // tp, 2048 // tp

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    def fp8(rows, k, *, packed=False, scale_k=128, experts=None):
        shape = (rows, k) if experts is None else (experts, rows, k)
        q = rnd(*shape, scale=0.2).to(torch.float8_e4m3fn)
        scales = torch.full(
            (*shape[:-2], math.ceil(rows / 128), math.ceil(k / scale_k)),
            k**-0.5,
            dtype=torch.float32,
            device=device,
        )
        return (pack_fp8(q) if packed else pack_ptpc_fp8(q)), scales

    qa, sqa = fp8(2624, HIDDEN)
    qb, sqb = fp8(heads * 256, 2048)
    uk, suk = fp8(heads * 512, 192, packed=True, scale_k=64)
    uv, suv = fp8(heads * 256, 512, packed=True)
    wo, so = fp8(HIDDEN, heads * 256)
    ug, sug = fp8(2 * inter, HIDDEN, experts=257)
    dn, sdn = fp8(HIDDEN, inter, experts=257)
    ik, sik = fp8(INDEX_DIM, HIDDEN)
    iq, siq = fp8(32 * INDEX_DIM, 2048)
    iw = pack_bf16(rnd(32, HIDDEN, scale=0.0 if ties else HIDDEN**-0.5).bfloat16())
    keep = (
        ik,
        sik,
        iw,
        iq,
        siq,
        torch.ones(128, device=device),
        torch.zeros(128, device=device),
    )
    index = torch.tensor(
        [t.data_ptr() for t in keep] + [0], dtype=torch.int64, device=device
    )
    return Glm53Weights(
        g_in=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        g_q=torch.ones(2048, dtype=torch.bfloat16, device=device),
        g_kv=torch.ones(512, dtype=torch.bfloat16, device=device),
        g_post=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        w_qkv_a=qa,
        s_qkv_a=sqa,
        w_q_b=qb,
        s_q_b=sqb,
        w_uk=uk,
        s_uk=suk,
        w_uv=uv,
        s_uv=suv,
        w_o=wo,
        s_o=so,
        w_r=pack_bf16(rnd(256, HIDDEN, scale=0.02).bfloat16()),
        bias=rnd(256, scale=0.05),
        w_ug=ug,
        s_ug=sug,
        w_dn=dn,
        s_dn=sdn,
        shuffled=True,
        index=index,
        keep=keep,
    )


def fixture(
    rows,
    capacity,
    device,
    pad_rows,
    full_context=False,
    seed=720,
    slot_offset=0,
    request_width=1,
):
    g = torch.Generator(device=device).manual_seed(seed)
    pages_per_request = capacity // 64
    assert rows % request_width == 0
    requests = rows // request_width
    pages = requests * pages_per_request + 1
    table = torch.zeros(
        requests, pages_per_request + 3, dtype=torch.int32, device=device
    )
    table[:, :pages_per_request] = (
        torch.randperm(pages - 1, generator=g, device=device).reshape(requests, -1)
        + 1
        + slot_offset // 64
    )
    req = torch.arange(requests - 1, -1, -1, dtype=torch.int32, device=device)
    req = req.repeat_interleave(request_width)
    lengths = ([max(1, capacity - 3), 65, 2049, 1, 4097, 127, 2048, 129] * 2)[:rows]
    if full_context:
        lengths = [capacity - 3 - row for row in range(rows)]
    if request_width > 1:
        lengths = [
            capacity - request_width - 2 - row // request_width + row % request_width
            for row in range(rows)
        ]
    lengths = [min(n, capacity) for n in lengths]
    if pad_rows:
        lengths[-pad_rows:] = [0] * pad_rows
    pos = torch.tensor(
        [max(0, n - 1) for n in lengths], dtype=torch.int64, device=device
    )
    slots = table[req.long(), pos // 64].long() * 64 + pos % 64
    for row, n in enumerate(lengths):
        if not n:
            slots[row] = -1
    counts = torch.tensor(
        [min(n, TOPK) for n in lengths], dtype=torch.int32, device=device
    )
    indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0).int()]
    )
    h = torch.randn(rows, HIDDEN, generator=g, device=device).bfloat16()
    angles = (
        torch.arange(capacity, device=device).float()[:, None]
        * (8000000 ** (-torch.arange(0, 64, 2, device=device).float() / 64))[None, :]
    )
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kv = (torch.randn(pages * 64, 576, generator=g, device=device) * 0.1).bfloat16()
    values = (torch.randn(pages, 64, 128, generator=g, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    shuffled = (
        values.view(torch.uint8)
        .reshape(pages, 4, 16, 8, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    scales = torch.ones(pages, 64, dtype=torch.float32, device=device)
    idx = torch.cat([shuffled.reshape(pages, -1), scales.view(torch.uint8)], dim=1)
    idx = idx.reshape(pages, 64, 132)
    if slot_offset:
        # Translate identical logical data past the 32-bit buffer byte limit.
        # The unused prefix remains covered by check_untouched().
        kv_storage = torch.zeros(
            slot_offset + kv.shape[0], 576, dtype=kv.dtype, device=device
        )
        kv_storage[slot_offset:].copy_(kv)
        kv = kv_storage
        idx_storage = torch.zeros(
            slot_offset // 64 + pages, 64, 132, dtype=idx.dtype, device=device
        )
        idx_storage[slot_offset // 64 :].copy_(idx)
        idx = idx_storage
    step = Glm53Step(
        pos,
        slots,
        indptr,
        req,
        table,
        torch.full((rows * TOPK + 1,), -7, dtype=torch.int32, device=device),
    )
    return h, cos, sin, Glm53Caches(kv, idx), step, lengths


def check_untouched(before, after, written_slots):
    valid = written_slots[written_slots >= 0].long().unique()
    mask = torch.ones(before.kv.shape[0], dtype=torch.bool, device=valid.device)
    mask[valid] = False
    assert torch.equal(before.kv[mask], after.kv[mask]), "untouched KV rows changed"
    changed = (before.idx != after.idx).reshape(-1)
    d = torch.arange(128, device=valid.device)
    for slot in valid.tolist():
        page, row = slot // 64, slot % 64
        offset = (
            page * INDEX_PAGE_BYTES
            + (row // 16) * 2048
            + (d // 16) * 256
            + (row % 16) * 16
            + d % 16
        )
        changed[offset] = False
        scale = page * INDEX_PAGE_BYTES + 8192 + row * 4
        changed[scale : scale + 4] = False
    assert not changed.any(), "untouched index-cache bytes changed"


def tensor_digest(tensor):
    """Hash full state on the test server without saving large cache tensors."""
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for chunk in raw.split(16 * 1024 * 1024):
        digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()
