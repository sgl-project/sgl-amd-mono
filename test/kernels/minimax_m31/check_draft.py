"""Check the M3.1 DSpark draft launch (minimax_m31/drafter) against an f32 reference of
SGLang's five draft layers (torchrun, TP4): random MXFP8 weights at the real shapes, random
bf16 K / V caches, 1 and 2 requests at prefixes inside and past the 4096-token window. The
final hidden states and every layer's new K / V rows within the bounds, no other cache row
written, graph replays bitwise, every rank alike; past the window, no key outside every
row's window read, and the key at row k's window start read by rows 0 .. k only.

    torchrun --nproc-per-node 4 test/kernels/minimax_m31/check_draft.py [--time]
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

H, L, HD, ROT, W, Q = 6144, 5, 128, 64, 4096, 7
EPS = 1e-6
FULL = {
    "q_proj": (8192, H),
    "k_proj": (512, H),
    "v_proj": (512, H),
    "o_proj": (H, 8192),
    "gate_proj": (12288, H),
    "up_proj": (12288, H),
    "down_proj": (H, 12288),
}
# measured: out rel 0.084-0.088, min cosine 0.9957, new K / V rows rel ~0.08 (e4m3
# activations and K / V against the f32 reference); a wrong page is past 0.3
OUT_REL, MIN_COS, KV_REL = 0.12, 0.99, 0.12
# a planted key: e4m3-exact, its score dominates the heads it lines up with
K_EDGE, V_EDGE = 32.0, 32.0


def mx_quant(w):
    """f32 [N, K] -> (e4m3 bytes, e8m0 bytes [N, K / 32])."""
    n, k = w.shape
    blk = w.view(n, k // 32, 32)
    am = blk.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(am / 448.0)).clamp(-127, 127)
    q = (blk / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(n, k).view(torch.uint8), (e + 127).to(torch.uint8)


def mx_dequant(q, s):
    n, k = q.shape
    vals = q.view(torch.float8_e4m3fn).float().view(n, k // 32, 32)
    return (vals * torch.exp2(s.float() - 127)[..., None]).view(n, k)


def gemma(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1 + w.float())


def rope(x, cs):
    """NeoX over the first ROT dims; cs [rows, ROT] (cos | sin)."""
    c, s = cs[:, None, : ROT // 2].float(), cs[:, None, ROT // 2 :].float()
    a, b = x[..., : ROT // 2].clone(), x[..., ROT // 2 : ROT].clone()
    out = x.clone()
    out[..., : ROT // 2], out[..., ROT // 2 : ROT] = a * c - b * s, b * c + a * s
    return out


def reference(x, pos, prefix, nreq, r2t, caches, full, norms, fin, cos_sin, rank):
    """The draft layers in f32: (final hidden [rows, H], this rank's new K / V rows a
    layer). Every rank holds the same prefix caches, so one rank's stands for all four
    KV heads."""
    rows = x.shape[0]
    hid, res, kv_rows = x.float(), None, []
    for i in range(L):
        n = norms[i]
        res = hid if res is None else hid + res
        xn = gemma(res, n[0])
        q = (xn @ full[i, "q_proj"].T).view(rows, 64, HD)
        k = (xn @ full[i, "k_proj"].T).view(rows, 4, HD)
        v = (xn @ full[i, "v_proj"].T).view(rows, 4, HD)
        q = rope(gemma(q, n[2]), cos_sin[pos])
        k = rope(gemma(k, n[3]), cos_sin[pos])
        kv_rows.append((k[:, rank], v[:, rank]))
        att = torch.zeros(rows, 64, HD, device=x.device)
        for r in range(nreq):
            s0 = r2t[r + 1]
            kb, vb = k[r * Q : (r + 1) * Q], v[r * Q : (r + 1) * Q]
            for j in range(Q):
                # the row's window of the prefix, then the whole block (the draft block
                # attends both ways)
                t = r * Q + j
                lo = max(0, prefix + j - (W - 1))
                kp = s0[lo:prefix].long()
                kk = torch.cat(
                    [caches[i][0][kp, 0].float()[:, None].expand(-1, 4, -1), kb]
                )
                vv = torch.cat(
                    [caches[i][1][kp, 0].float()[:, None].expand(-1, 4, -1), vb]
                )
                kh, vh = kk[:, torch.arange(64) // 16], vv[:, torch.arange(64) // 16]
                sc = torch.einsum("hd,nhd->hn", q[t], kh) / HD**0.5
                att[t] = torch.einsum("hn,nhd->hd", sc.softmax(-1), vh)
        res = att.view(rows, -1) @ full[i, "o_proj"].T + res
        xn = gemma(res, n[1])
        gg = (xn @ full[i, "gate_proj"].T).clamp(max=7.0)
        uu = (xn @ full[i, "up_proj"].T).clamp(-7.0, 7.0)
        hid = (gg * torch.sigmoid(1.702 * gg) * (uu + 1)) @ full[i, "down_proj"].T
    return gemma(hid + res, fin), kv_rows


def restore(caches, saved):
    for (k, v), (k0, v0) in zip(caches, saved):
        k.copy_(k0)
        v.copy_(v0)


def window_edges(
    runner, args, caches, saved, out, new_rows, r2t, nreq, prefix, slots, gs
):
    """The window past a long prefix, against the unchanged step (``out``, every layer's
    ``new_rows``) -> what went wrong. Layer 1's new K / V rows of a row hold layer 0's
    attention of that row alone, so they tell the rows' layer-0 windows apart."""
    bad = []
    first = prefix - (W - 1)  # row 0's window start; row j's is first + j
    # garbage in every key before row 0's window start: no row may read it
    restore(caches, saved)
    for r in range(nreq):
        old = r2t[r + 1, :first].long()
        for k, v in caches:
            k[old] = 400.0
            v[old] = -400.0
    got = runner.forward(*args)
    torch.cuda.synchronize()
    same = torch.equal(got, out) and all(
        torch.equal(k[slots], k1) and torch.equal(v[slots], v1)
        for (k, v), (k1, v1) in zip(caches, new_rows)
    )
    if not same:
        bad.append("a key before every row's window was read")
    # a dominant key at row 0's window start + j, in layer 0: rows 0 .. j read it
    sign = torch.randint(0, 2, (2, HD), generator=gs, device=out.device).float() * 2 - 1
    for j in (0, 3, Q - 1):
        restore(caches, saved)
        for r in range(nreq):
            s = r2t[r + 1, first + j].long()
            caches[0][0][s, 0] = (K_EDGE * sign[0]).bfloat16()
            caches[0][1][s, 0] = (V_EDGE * sign[1]).bfloat16()
        runner.forward(*args)
        torch.cuda.synchronize()
        k1, v1 = caches[1][0][slots], caches[1][1][slots]
        moved = ((k1 != new_rows[1][0]) | (v1 != new_rows[1][1])).flatten(1).any(1)
        moved = moved.view(nreq, Q).cpu()
        if not torch.equal(moved, (torch.arange(Q) <= j).expand(nreq, Q)):
            bad.append(
                f"the key at row 0's window start + {j} reached rows {moved.int().tolist()}"
            )
    restore(caches, saved)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="900,5000,70000")
    ap.add_argument("--reqs", default="1,2")
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert tp == 4, "the draft launch is TP4"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.minimax_m31.drafter import runner as R

    g = torch.Generator().manual_seed(1)
    # full weights (alike on every rank), quantized, then this rank's shards
    full, weights = {}, []
    for i in range(L):
        wl = {}
        for p, (n, k) in FULL.items():
            q, s = mx_quant(
                torch.randn(n, k, generator=g) * (0.01 if p == "down_proj" else 0.02)
            )
            full[i, p] = mx_dequant(q, s).to(dev)
            wl[p, "weight"] = R.shard(q, p, "weight", rank, tp).contiguous()
            wl[p, "weight_scale_inv"] = R.shard(
                s, p, "weight_scale_inv", rank, tp
            ).contiguous()
        weights.append(wl)

    def norm(n):
        return (torch.randn(n, generator=g) * 0.1).bfloat16().to(dev)

    norms = [[norm(H), norm(H), norm(HD), norm(HD)] for _ in range(L)]
    fin = norm(H)
    inv = 1.0 / (1e7 ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT))
    f = torch.arange(1 << 17, dtype=torch.float64)[:, None] * inv[None, :]
    cos_sin = torch.cat([f.cos(), f.sin()], -1).float().to(dev)

    def ns(**kw):
        return SimpleNamespace(**kw)

    layers = [
        ns(
            decoder_layer=ns(
                input_layernorm=ns(weight=n[0]),
                post_attention_layernorm=ns(weight=n[1]),
                self_attn=ns(q_norm=ns(weight=n[2]), k_norm=ns(weight=n[3])),
            ),
            self_attn=ns(attn=ns(layer_id=i), rotary_emb=ns(cos_sin_cache=cos_sin)),
        )
        for i, n in enumerate(norms)
    ]
    runner = R.DraftRunner(
        ns(layers=layers, norm=ns(weight=fin)), weights, rank, cpu, dev
    )
    # pages of 128 slots, page 0 the padding page; every rank's caches alike
    npages = 4096
    nslots = npages * 128
    gk = torch.Generator(device=dev).manual_seed(5)
    caches = [
        tuple(
            torch.randn(nslots, 1, HD, generator=gk, device=dev).bfloat16()
            for _ in "kv"
        )
        for _ in range(L)
    ]
    pool = ns(get_kv_buffer=lambda i: caches[i])
    r2t = torch.zeros(4, 1 << 17, dtype=torch.int32, device=dev)
    reqs = [int(v) for v in a.reqs.split(",")]
    cases = [(n, p) for n in reqs for p in map(int, a.prefix.split(","))]
    fails = []
    for nreq, prefix in cases:
        gs = torch.Generator(device=dev).manual_seed(prefix + nreq)
        rows = nreq * Q
        # every request its own pages, as SGLang's allocator hands them out
        n = -(-(prefix + Q) // 128)
        perm = torch.randperm(npages - 1, generator=gs, device=dev) + 1
        for r in range(nreq):
            pages = perm[r * n : (r + 1) * n]
            r2t[r + 1, : n * 128] = (
                (pages[:, None] * 128 + torch.arange(128, device=dev)).flatten().int()
            )
        rpi = torch.arange(1, nreq + 1, dtype=torch.int64, device=dev)
        seq = torch.full((nreq,), prefix, dtype=torch.int64, device=dev)
        pos = (
            torch.cat([torch.arange(prefix, prefix + Q) for _ in range(nreq)])
            .long()
            .to(dev)
        )
        slots = torch.cat([r2t[r + 1, prefix : prefix + Q] for r in range(nreq)]).long()
        x = torch.randn(rows, H, generator=gs, device=dev).bfloat16()
        dist.broadcast(x, 0)
        before = [(k.clone(), v.clone()) for k, v in caches]
        backend = ns(token_to_kv_pool=pool, req_to_token_pool=ns(req_to_token=r2t))
        fb = ns(req_pool_indices=rpi, seq_lens=seq, out_cache_loc=slots)
        out = runner.forward(x, pos, fb, backend).clone()
        torch.cuda.synchronize()
        new_rows = [(k[slots].clone(), v[slots].clone()) for k, v in caches]
        ref, kv_rows = reference(
            x, pos, prefix, nreq, r2t, before, full, norms, fin, cos_sin, rank
        )
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(out.float(), ref, -1).min().item()
        kv_rel, stray = 0.0, 0
        for (k, v), (k0, v0), (kr, vr) in zip(caches, before, kv_rows):
            for c, c0, want in ((k, k0, kr), (v, v0, vr)):
                kv_rel = max(
                    kv_rel, ((c[slots, 0].float() - want).norm() / want.norm()).item()
                )
                changed = (c != c0).view(nslots, -1).any(1)
                changed[slots] = False
                stray += int(changed.sum())
        # graph replays write the same rows: bitwise the eager output
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = runner.forward(x, pos, fb, backend)
        bad_replay = 0
        for _ in range(a.replays):
            graph.replay()
            torch.cuda.synchronize()
            bad_replay += int(not torch.equal(replayed, out))
        every = [torch.empty_like(out) for _ in range(tp)]
        dist.all_gather(every, out)
        alike = all(torch.equal(every[0], e) for e in every)
        edges = []
        if prefix >= W:
            edges = window_edges(runner, (x, pos, fb, backend), caches, before, out, new_rows, r2t, nreq, prefix, slots, gs)  # fmt: skip
        ok = rel <= OUT_REL and cos >= MIN_COS and kv_rel <= KV_REL
        ok = ok and stray == 0 and bad_replay == 0 and alike and not edges
        line = (
            f"reqs={nreq} prefix={prefix:6d}: {'PASS' if ok else 'FAIL'} | out rel {rel:.4f} "
            f"min cos {cos:.5f} | new K / V rel {kv_rel:.4f} | other rows written {stray} | "
            f"replays differing {bad_replay} | ranks alike {alike}"
            + (f" | window {'; '.join(edges) or 'kept'}" if prefix >= W else "")
        )
        if a.time:
            st, en = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            st.record()
            for _ in range(20):
                graph.replay()
            en.record()
            torch.cuda.synchronize()
            line += f" | {st.elapsed_time(en) * 1e3 / 20:.1f} us a launch"
        if rank == 0:
            print(line, flush=True)
        if not ok:
            fails.append((nreq, prefix))
        del graph
        restore(caches, before)
    health = runner.health.tolist()
    if any(health):
        fails.append(("health", health))
    dist.barrier()
    if rank == 0:
        print("ALL PASS" if not fails else f"FAIL at {fails}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
