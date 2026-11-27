"""K2 and K3 on eight gfx1250 GPUs (a TP 8 / EP 8 group, one process a GPU, a gloo
group for the peer buffer).

K3 (EP 8): each rank holds its 32 experts as SGLang keeps them on gfx1250
(``prepare_gfx1250_a8w4_weights``, read in place; the test fills each rank's first
EXPERTS_USED and routes only to those), the same ids / scores / x on every rank;
every rank's stream and next attention input against the reference: each rank's
partial (its picks, slot order, one bf16 rounding) summed in rank order in fp32,
one rounding, then the unpooler and the boundary.

K2 (TP 8): each rank its own attention output, o_gate and o_proj K shard; the
reference's o_proj output is the ranks' bf16 partials summed in fp32 in rank
order, one rounding; then the boundary, router and pooler (whole, or ``pshard``:
each rank its contiguous row slice of the pooler, the pooled rows exchanged).

``python test/kernels/test_muse_spark_tp8_gfx1250.py k3|k2 [rows] [pshard] [trace]``
(also collected by pytest when eight GPUs are present)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 8
E_RANK = 32  # experts a rank
EXPERTS_USED = 4  # experts a rank the test routes to (the table's prefix)
K3_LAYER = 9
K2_LAYER = 5


def _peers(nbytes, rank, dev):
    import torch.distributed as dist

    from sgl_amd_mono.common.peer import PeerBuffer

    pb = PeerBuffer(nbytes, dist.group.WORLD, rank, NP, dev)
    pb.bytes.zero_()
    torch.cuda.synchronize()
    dist.barrier()
    return pb


def _stamps(scratch, region, layer):
    from sgl_amd_mono.common.arch import BLOCKS
    from sgl_amd_mono.muse_spark.gfx1250.device import TRACE_LAYERS, TRACE_SLOTS

    off, nbytes = region
    t = scratch[off : off + nbytes].view(torch.int64)[
        : TRACE_LAYERS * BLOCKS * TRACE_SLOTS
    ]
    return t.reshape(TRACE_LAYERS, BLOCKS, TRACE_SLOTS)[layer % TRACE_LAYERS].cpu()


def _bench(launch_once, n=30, fenced=False):
    """Back-to-back launches (the ranks pace each other through the mailboxes): the
    mean launch time (us). ``fenced``: a host barrier between launches (a standalone K2
    has no later stage that keeps a fast rank's next pushes behind a slow rank's reads;
    in a layer, K3 does), the mean of each launch's own event time."""
    import torch.distributed as dist

    if fenced:
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            dist.barrier()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            launch_once()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        return sorted(ts)[n // 2]

    for _ in range(5):
        launch_once()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        launch_once()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / n


def _k3_worker(rank, rows, trace, q):
    import test_muse_spark_k3_gfx1250 as T
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx1250.config import (  # fmt: skip
        HIDDEN,
        INTER,
        MAX_TOKENS,
        MOE_HIDDEN,
        POST_EPS,
        RMS_EPS,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.ffn import (
        k3_scratch_layout,
        k3_sym_layout,
    )

    dev = torch.device("cuda", rank)
    bf = torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(
        900 + rows
    )  # the same tables on every rank
    torch.manual_seed(900 + rows)
    U = EXPERTS_USED
    # every rank's used experts (NP U of them), table index r U + k = global expert r E_RANK + k
    w13, s13, dec13 = T._fp4_weight(NP * U, 2 * INTER, MOE_HIDDEN, dev, gen)
    w2, s2, dec2 = T._fp4_weight(NP * U, MOE_HIDDEN, INTER, dev, gen)
    xin = (torch.randn(rows, MOE_HIDDEN, device=dev) * 0.5).to(bf)
    tab = torch.empty(rows, TOP_K, dtype=torch.int64, device=dev)  # table indices
    for t in range(rows):
        tab[t, 0] = 0  # every token's slot 0: rank 0's expert 0 (its widest column set)
        tab[t, 1:] = (
            torch.randperm(NP * U - 1, device=dev, generator=gen)[: TOP_K - 1] + 1
        )
    ids = ((tab // U) * E_RANK + tab % U).to(torch.int32)
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
    lo = rank * U
    g13, g2, gs13, gs2 = T.gfx1250_experts(
        w13[lo : lo + U],
        s13[lo : lo + U],
        w2[lo : lo + U],
        s2[lo : lo + U],
        rank * E_RANK,
    )
    # the reference: each rank's partial (its picks, slot order, one rounding), summed in rank order
    y = torch.zeros(rows, TOP_K, MOE_HIDDEN, device=dev)
    xq = T.mx8_dequant(xin.float())
    for t in range(rows):
        for s in range(TOP_K):
            e = int(tab[t, s])
            gu = xq[t] @ dec13[e].T
            gate, up = gu[:INTER], gu[INTER:]
            y[t, s] = T._bf(T.mx8_dequant(gate * torch.sigmoid(gate) * up) @ dec2[e].T)
    total = torch.zeros(rows, MOE_HIDDEN, device=dev)
    for r in range(NP):
        part = torch.zeros(rows, MOE_HIDDEN, device=dev)
        for t in range(rows):
            for s in range(TOP_K):
                if int(tab[t, s]) // U != r:
                    continue
                v = T._bf(y[t, s] * w["pen_w"].float())
                part[t] += (
                    float(scores[t, s]) / torch.sqrt(v.pow(2).mean() + POST_EPS) * v
                )
        total += T._bf(part)
    branch = T._bf(T._bf(total) @ w["w_unpool"].float().T)
    post_input = T._bf(branch * w["pfw"].float())
    nb = T._bf(
        post_input * torch.rsqrt(post_input.pow(2).mean(-1, keepdim=True) + POST_EPS)
    )
    new_ref = w["alpha"] * stream + w["beta"] * nb
    attn_ref = (new_ref * torch.rsqrt(new_ref.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w["npw"].float()).to(bf)  # fmt: skip

    lay = k3_scratch_layout()
    attn_out = torch.full((MAX_TOKENS, HIDDEN), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(scratch_bytes(lay), dtype=torch.uint8, device=dev)
    peers = _peers(scratch_bytes(k3_sym_layout(NP)) + guard.FLAG_BYTES, rank, dev)
    epoch = T._epoch(dev, 5.0)
    launch = build_k3(K3Build(tokens=rows, npes=NP, trace=trace))
    got = stream.clone()

    def once():
        launch(
            xin.data_ptr(), ids.data_ptr(), scores.data_ptr(), rank * E_RANK,
            g13.data_ptr(), gs13.data_ptr(), g2.data_ptr(), gs2.data_ptr(),
            w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
            w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
            w["w_unpool"].data_ptr(), attn_out.data_ptr(), scratch.data_ptr(),
            peers.local, peers.addresses.data_ptr(), rank, epoch.data_ptr(), K3_LAYER, 0,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1

    try:
        for _ in range(2):  # the second step's tags differ from the first's
            got.copy_(stream)
            torch.cuda.synchronize()
            dist.barrier()
            once()
            torch.cuda.synchronize()
        errs = epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
        res = _compare(
            rank,
            errs,
            [("stream", got, new_ref), ("attention input", attn_out[:rows], attn_ref)],
        )
        if trace:
            res["launch_us"] = _bench(once)
            res["stamps"] = _stamps(scratch, lay["k3_trace"], K3_LAYER).tolist()
        q.put(res)
    finally:
        peers.close()


def _k2_worker(rank, rows, pshard, trace, q):
    import test_muse_spark_k2_gfx1250 as T
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.muse_spark.gfx1250.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx1250.config import (  # fmt: skip
        EXPERTS,
        HEAD_DIM,
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        RMS_EPS,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx1250.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx1250.stages.post_attn import (
        k2_scratch_layout,
        sym_bytes,
    )

    dev = torch.device("cuda", rank)
    bf = torch.bfloat16
    torch.manual_seed(100 + rows)  # the same tensors on every rank
    os_ = [torch.randn(rows, Q_K, device=dev).to(bf) for _ in range(NP)]
    gs_ = [torch.randn(rows, Q_K, device=dev).to(bf) for _ in range(NP)]
    wos = [(torch.randn(HIDDEN, Q_K, device=dev) * 0.02).to(bf) for _ in range(NP)]
    stream = torch.randn(rows, HIDDEN, device=dev)
    router = torch.randn(HIDDEN, EXPERTS, device=dev) * 0.02
    w_pool = (torch.randn(MOE_HIDDEN, HIDDEN, device=dev) * 0.02).to(bf)
    w = {
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "pre_w": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "router": router.contiguous(),
        "bias": torch.randn(EXPERTS, device=dev) * 0.01,
        "w_pool": w_pool,
        "xn_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
    }
    # the reference: the ranks' bf16 o_proj partials summed in rank order in fp32, one rounding
    ao = torch.zeros(rows, HIDDEN, device=dev)
    for r in range(NP):
        of = os_[r].float().reshape(rows, -1, HEAD_DIM)
        og = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + RMS_EPS) * torch.sigmoid(gs_[r].float().reshape(rows, -1, HEAD_DIM))).to(bf)  # fmt: skip
        ao += (og.reshape(rows, -1).float() @ wos[r].float().T).to(bf).float()
    new_ref, ids_ref, scores_ref, xn_ref, biased = _k2_reference_from_ao(
        T, ao.to(bf), stream, w
    )

    lay = k2_scratch_layout()
    ids = torch.full((MAX_TOKENS, TOP_K), -7, dtype=torch.int32, device=dev)
    scores = torch.full((MAX_TOKENS, TOP_K), float("nan"), device=dev)
    xn = torch.full((MAX_TOKENS, MOE_HIDDEN), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(scratch_bytes(lay), dtype=torch.uint8, device=dev)
    peers = _peers(sym_bytes(NP) + guard.FLAG_BYTES, rank, dev)
    epoch = T._epoch(dev, 5.0)
    launch = build_k2(
        K2Build(tokens=rows, npes=NP, xq_pre=True, pshard=pshard, trace=trace)
    )
    pool_rows = MOE_HIDDEN // NP
    pool_ptr = (
        w["w_pool"][rank * pool_rows : (rank + 1) * pool_rows]
        if pshard
        else w["w_pool"]
    ).data_ptr()
    got = stream.clone()

    def once():
        launch(
            os_[rank].data_ptr(), gs_[rank].data_ptr(), got.data_ptr(), wos[rank].data_ptr(),
            w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
            w["router"].data_ptr(), 0, w["bias"].data_ptr(),
            pool_ptr, w["xn_w"].data_ptr(), ids.data_ptr(), scores.data_ptr(),
            xn.data_ptr(), scratch.data_ptr(), peers.local, peers.addresses.data_ptr(), rank,
            epoch.data_ptr(), K2_LAYER,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1

    try:
        for _ in range(2):
            got.copy_(stream)
            torch.cuda.synchronize()
            dist.barrier()
            once()
            torch.cuda.synchronize()
        errs = [int(x) for x in epoch[list(guard.ERROR_WORDS)].tolist()]
        res = _compare(
            rank, errs, [("stream", got, new_ref), ("expert input", xn[:rows], xn_ref)]
        )
        try:
            T._check_routing(ids[:rows], scores[:rows], ids_ref, scores_ref, biased)
            res["routing"] = "ok"
        except AssertionError as ex:
            res["routing"] = f"FAILED {ex}"
        if trace:
            res["launch_us"] = _bench(once, fenced=True)
            res["stamps"] = _stamps(scratch, lay["k2_trace"], K2_LAYER).tolist()
        q.put(res)
    finally:
        peers.close()


def _k2_reference_from_ao(T, ao, stream, w):
    """``test_muse_spark_k2_gfx1250._reference`` from the all-reduced attention output on."""
    from sgl_amd_mono.muse_spark.gfx1250.config import (  # fmt: skip
        POST_EPS,
        RMS_EPS,
        ROUTE_EPS,
        ROUTE_SCALE,
        ROUTER_MULT,
        TOP_K,
    )

    bf = torch.bfloat16
    nb = T._rms(ao, POST_EPS).float()
    new = w["alpha"] * stream + w["beta"] * nb
    x = T._rms(new, RMS_EPS, w["pre_w"])
    # the model's router reads the pre-FFN norm's fp32 output (before its bf16 rounding) and the fp32 weight
    x32 = (
        new
        * torch.rsqrt(new.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        * w["pre_w"].float()
    )
    logits = x32 @ w["router"]
    sc = torch.sigmoid(logits * ROUTER_MULT)
    biased = sc + w["bias"]
    order = torch.argsort(biased, dim=-1, descending=True, stable=True)
    ids = order[:, :TOP_K]
    picks = sc.gather(1, ids)
    scores = picks / (picks.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    pooled = (x.float() @ w["w_pool"].float().T).to(bf)
    xn = T._rms(pooled, RMS_EPS, w["xn_w"])
    return new, ids.to(torch.int32), scores, xn, biased


def _compare(rank, errs, pairs):
    out = {"rank": rank, "errs": errs, "checks": []}
    for what, got, ref in pairs:
        g, r = got.float(), ref.float()
        cos = torch.nn.functional.cosine_similarity(g, r, dim=-1).min().item()
        err = (g - r).abs()
        bad = (err > r.abs() / 32 + 2e-2).float().mean().item()
        out["checks"].append((what, err.max().item(), cos, bad))
    return out


def _worker(rank, which, rows, pshard, trace, port, q):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    try:
        if which == "k3":
            _k3_worker(rank, rows, trace, q)
        else:
            _k2_worker(rank, rows, pshard, trace, q)
    except Exception as ex:  # report, do not hang the others' queue reads
        import traceback

        q.put({"rank": rank, "error": f"{ex}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def run(which, rows, pshard=False, trace=False):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + (os.getpid() % 2000)
    procs = [
        ctx.Process(target=_worker, args=(r, which, rows, pshard, trace, port, q))
        for r in range(NP)
    ]
    for p in procs:
        p.start()
    res = [q.get(timeout=1500) for _ in range(NP)]
    for p in procs:
        p.join(timeout=120)
    return sorted(res, key=lambda d: d["rank"])


def _assert_ok(res):
    for d in res:
        assert "error" not in d, d.get("error")
        assert all(e == 0 for e in d["errs"]), (
            f"rank {d['rank']}: a wait failed {d['errs']}"
        )
        for what, err, cos, bad in d["checks"]:
            assert bad == 0.0 and cos >= 0.996, (
                f"rank {d['rank']} {what}: cosine {cos:.5f}, {bad:.2%} off"
            )
        assert d.get("routing", "ok") == "ok", d["routing"]


eight = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= NP),
    reason="needs eight GPUs",
)


@eight
@pytest.mark.parametrize("rows", [8, 16, 12])
def test_k3_tp8(rows):
    _assert_ok(run("k3", rows))


@eight
@pytest.mark.parametrize("rows,pshard", [(8, False), (8, True), (16, True), (12, True)])
def test_k2_tp8(rows, pshard):
    _assert_ok(run("k2", rows, pshard))


STAGES = {
    "k3": {
        0: "entry",
        2: "plan",
        3: "ug",
        12: "dn mids",
        4: "dn",
        5: "fin",
        8: "unp flags",
        6: "unpool",
        7: "end",
    },
    "k2": {
        0: "entry",
        2: "o",
        3: "boundary",
        4: "x in",
        5: "passes",
        6: "stream",
        7: "end",
    },
}
SPANS = {
    "k3": [
        ("plan", 0, 2),
        ("ug", 2, 3),
        ("dn", 3, 4),
        ("fin", 4, 5),
        ("unpool", 5, 6),
        ("bnd", 6, 7),
        ("CTA", 0, 7),
    ],
    "k2": [
        ("o", 0, 2),
        ("bnd part", 2, 3),
        ("wait x", 3, 4),
        ("passes", 4, 5),
        ("res", 5, 6),
        ("route/xn", 6, 7),
        ("CTA", 0, 7),
    ],
}


def report(which, res):
    for d in res:
        if "error" in d:
            print(f"rank {d['rank']}: ERROR {d['error']}")
            continue
        line = f"rank {d['rank']}: errors {d['errs']} " + " ".join(
            f"{w}: max err {e:.3g} cos {c:.6f} off {b:.2%}"
            for w, e, c, b in d["checks"]
        )
        if "routing" in d:
            line += f" routing {d['routing'][:80]}"
        print(line)
        if "stamps" in d:
            t = torch.tensor(d["stamps"], dtype=torch.float64) / 100.0  # us
            cells = []
            for nm, a, b in SPANS[which]:
                m = (t[:, a] > 0) & (t[:, b] > 0)
                v = (t[:, b] - t[:, a])[m]
                if v.numel():
                    cells.append(f"{nm} {v.median().item():.1f}/{v.max().item():.1f}")
            print(
                f"   launch {d['launch_us']:.1f} us (back to back) | per-CTA spans median/max us: "
                + " | ".join(cells)
            )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "k3"
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    pshard = "pshard" in sys.argv[3:]
    trace = "trace" in sys.argv[3:]
    report(which, run(which, rows, pshard, trace))
