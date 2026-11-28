# Adapted from vllm-project/vllm#67678 at ce476ce305 (Apache-2.0):
# tests/models/qwen3_5/amd_specialized_decode/test_layer.py
"""Check the wide launches (``wide.layer``: out_proj or o_proj, its TP reduce,
the post-attention norm, the router, the experts with the fused shared expert,
the MoE TP reduce and the next layer's input norm) on 8 gfx950 GPUs (torchrun),
on random TP8 shards in SGLang's layouts, against float64 references with the
stock rounding points (``wide_reference``):

- compile: every wide key the spec can make (``attn`` on the gated core,
  ``gdn_back`` with a bf16 or an MXFP4 out_proj, at 17 and 32 rows; the decode
  step's ``gdn`` keys and the launches before them, at 32 rows) compiles on
  every rank;
- chain: one step of three launches (``gdn_back``, ``attn``, ``gdn_back``) at 17
  and 32 rows, with a bf16 and an MXFP4 out_proj; each launch's MoE input rows,
  routes, route weights, residual and normed rows against the reference fed the
  previous launch's outputs (relative Frobenius errors);
- front: one decode step of two ``gdn`` launches at 32 rows (the step's first
  layer, then the next layer from the first launch's MXFP4 rows), bf16
  projections and an f32 state, or MXFP4 projections and a bf16 state, with
  pad rows: each launch's core and conv and SSM state rows against K1 on 8-row
  chunks of its input rows, every other state slot kept, and its tail as in
  the chain;
- fault: rank 3 stalls at its first wait (``stop=False``) in ``gdn_back`` at 32
  rows, with a deadline of 0.5 s and 5 s on the peers: rank 3 sets its deadline
  word and its flag reaches every rank.

The MoE is checked on the kernel's own MXFP4 input rows (``x4``): the
reference's input differs from the kernel's in a few bf16 ulps, which flip MXFP4
codes; routes are compared as top-k sets within a tie margin
(``wide_reference.route``).
"""

import datetime
import os
import sys
import time
from dataclasses import replace

import torch
import torch.distributed as dist
import wide_reference as R
from inputs import BF, CORE, MAXT, epoch, errors, gen, k1_state, k1_weights

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import grid
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W
from sgl_amd_mono.qwen38_a95b_mxfp4.spec import _compile_only

WIDTHS = (17, 32)
FORMATS = ("bf16", "mxfp4")
STEP = ("gdn_back", "attn", "gdn_back")
EPS = 1e-6

rank = int(os.environ["RANK"])
assert int(os.environ["WORLD_SIZE"]) == L.TP, f"needs {L.TP} ranks"
torch.cuda.set_device(rank)
DEV = torch.device("cuda", rank)
dist.init_process_group("gloo", timeout=datetime.timedelta(hours=1))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def all_ok(flag: bool) -> bool:
    t = torch.tensor([0 if flag else 1])
    dist.all_reduce(t)
    return int(t) == 0


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def allsum(part):
    """Every rank's bf16 partial, summed in rank order -> bf16."""
    parts = [torch.empty_like(part.cpu()) for _ in range(L.TP)]
    dist.all_gather(parts, part.cpu().contiguous())
    tot = parts[0].double()
    for p in parts[1:]:
        tot = tot + p.double()
    return R.bf(tot).to(DEV)


def key(kind, rows, fmt, **kw):
    """The spec's wide key (``spec._wide_keys``) of a layer kind."""
    if kind == "attn":
        return W.LayerBuild(kind="attn", tokens=rows, tp=L.TP, og=False, rip=L.RI, **kw)
    return W.LayerBuild(
        kind="gdn_back",
        tokens=rows,
        tp=L.TP,
        out=fmt,
        out_shuffled=fmt == "mxfp4",
        rip=L.RI,
        **kw,
    )


def compile_keys(keys):
    """Every rank compiles every key before any rank launches one: a rank still
    compiling would leave the others waiting in the kernel."""
    keys = list(keys)
    start = len(keys) * rank // L.TP  # each rank at its own share of the disk cache
    ok = True
    try:
        for k in keys[start:] + keys[:start]:
            _compile_only(W.build(k))
        torch.cuda.synchronize()
    except Exception as e:  # reported, then every rank stops
        print(f"[rank {rank}] compile: {type(e).__name__}: {e}", flush=True)
        ok = False
    return all_ok(ok)


def check_compile():
    q4 = W.q4_of("mxfp4")
    keys = {key(kind, s, f) for kind in set(STEP) for s in WIDTHS for f in FORMATS}
    # the decode keys: the GDN fronts, and the launches before an MXFP4 front
    keys |= {front_key(32, qq, n) for qq in (False, True) for n in (0, 1)}
    keys |= {
        replace(front_key(32, True, 1), next_q4=q4),
        key("attn", 32, "bf16", next_q4=q4),
    }
    keys = sorted(keys, key=repr)
    t0 = time.time()
    ok = compile_keys(keys)
    log(
        f"compile: {len(keys)} wide keys in {time.time() - t0:.0f} s: {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        raise SystemExit("a rank failed to compile")
    return []


class Side:
    """A peer buffer, a guard block, the barrier slots of one step's launches and
    a scratch of ``rows``, as the spec's ``_Wide``."""

    def __init__(self, rows, launches, seconds=30.0):
        self.peer = PeerBuffer(W.peer_buffer_bytes(), dist.group.WORLD, rank, L.TP, DEV)
        self.peer.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()  # no peer writes into a buffer before its owner zeroed it
        self.epoch = epoch(DEV, seconds)
        self.epoch[:1].add_(1)
        self.rows = rows
        self.layout = W.scratch_layout(rows, L.RI)
        self.scratch = torch.zeros(
            W.scratch_bytes(rows, L.RI), dtype=torch.uint8, device=DEV
        )
        self.sync = torch.zeros(grid.sync_words(), dtype=torch.int32, device=DEV)
        base = self.sync.data_ptr()
        slot = [base + 4 * grid.SLOT_WORDS * k for k in range(grid.SLOTS)]
        self.slots = [(slot[a], slot[b]) for a, b in grid.launch_slots(launches)]
        self.folds = base + 4 * grid.FOLD_BASE

    def view(self, name, cols, dtype=BF):
        off = self.layout[name][0]
        n = self.rows * cols * torch.tensor([], dtype=dtype).element_size()
        return self.scratch[off : off + n].view(dtype).view(self.rows, cols)

    def route(self):
        """(ids [rows, SLOTS], weights [rows, SLOTS]) of the last launch."""
        off, n = self.layout["route"][0], self.rows * L.SLOTS
        words = self.scratch[off : off + 8 * n]
        return words[: 4 * n].view(torch.int32).view(-1, L.SLOTS), words[4 * n :].view(
            torch.float32
        ).view(-1, L.SLOTS)

    def run(self, n, k, layer, core, w, n_post, n_next, front=None):
        """Launch ``n`` of the step: model layer ``layer`` from its out_proj /
        o_proj input ``core``, or for ``gdn`` from the scratch rows with the GDN
        arguments ``front``."""
        args = dict.fromkeys(W.ABI.names, 0) | {
            "scratch": self.scratch,
            "attn" if k.kind == "attn" else "core_out": 0 if core is None else core,
            "n_post": n_post,
            "n_next": n_next,
            "w_out": w["out"],
            "w_out_s": w["out_s"],
            "w_gate": w["w_gate"],
            "w13": w["w13"],
            "w13s": w["w13s"],
            "w2": w["w2"],
            "w2s": w["w2s"],
            "peers": self.peer.addresses,
            "rank": rank,
            "epoch": self.epoch,
            "layer": layer,
            "sync": self.slots[n][0],
            "sync_next": self.slots[n][1],
            "folds": self.folds,
        }
        W.run(k, args | (front or {}))

    def close(self):
        self.peer.close()


def weights(shared, mine, kind, fmt):
    """One layer's tail weights (kernel layouts) and their float64 references."""

    def rn(*shape, scale=1.0, g=mine):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(BF)

    cols = L.ATTN_CORE if kind == "attn" else L.VALUE_DIM
    w = {}
    if kind == "attn" or fmt == "bf16":
        w["out"] = w["out_r"] = rn(L.HIDDEN, cols, scale=0.02)
        w["out_s"] = 0
    else:
        codes, sc = R.mxfp4(L.HIDDEN, cols, mine, lo=119, hi=123, device=DEV)
        w["out"], w["out_s"] = R.shuffle_data(codes), R.shuffle_scale(sc)
        w["out_r"] = R.dequant_mx(codes, sc)
    e, rip = L.E_ALL, L.RI
    w13, w13s = R.mxfp4(e * 2 * rip, L.HIDDEN, mine, lo=119, hi=123, device=DEV)
    w2, w2s = R.mxfp4(e * L.HIDDEN, rip, mine, lo=119, hi=123, device=DEV)
    return w | {
        "w_gate": rn(e, L.HIDDEN, scale=0.02, g=shared),
        "w13": R.shuffle_data(w13).view(e, 2 * rip, L.HIDDEN // 2),
        "w13s": R.shuffle_scale(w13s).view(e, 2 * rip, -1),
        "w2": R.shuffle_data(w2).view(e, L.HIDDEN, rip // 2),
        "w2s": R.shuffle_scale(w2s).view(e, L.HIDDEN, -1),
        "r13": w13.view(e, 2 * rip, -1),
        "r13s": w13s.view(e, 2 * rip, -1),
        "r2": w2.view(e, L.HIDDEN, -1),
        "r2s": w2s.view(e, L.HIDDEN, -1),
        "fmt": "bf16" if kind == "attn" else fmt,
    }


def tail_notes(side, tag, res, core, w, n_post, n_next, q4_next=False):
    """(name, relative error, bound) of a launch's MoE input rows, routes, route
    weights, residual and normed rows, against the reference fed the launch's
    residual ``res`` and out_proj / o_proj input ``core``. ``q4_next``: the
    closing combine quantized its rows for the next in_proj_qkvz into the MoE
    input rows' buffer: they are checked as that."""
    rows = side.rows
    z1 = res.double() + allsum(R.linear(core.double(), w["out_r"], w["fmt"]))
    res1 = R.bf(z1)
    xn = R.gemma_norm(z1, n_post, EPS)
    kxd = R.dequant_mx(
        side.view("x4", L.HIDDEN // 2, torch.uint8),
        side.view("x4s", L.HIDDEN // 32, torch.int32),
    )
    notes = []
    if not q4_next:
        notes.append((f"{tag} moe input (MXFP4)", rel(kxd, R.quant_act(xn)), 2e-2))
    kids, kws = side.route()
    y, ids, ws = R.moe(
        xn,
        w["w_gate"],
        w["r13"],
        w["r13s"],
        w["r2"],
        w["r2s"],
        L.EXPERTS,
        L.TOPK,
        L.RI,
        L.RI,
        prefer=kids[:, : L.TOPK],
        xd=None if q4_next else kxd,
    )
    same = sum(set(kids[r].tolist()) == set(ids[r].tolist()) for r in range(rows))
    notes.append((f"{tag} routes", 1 - same / rows, 0.0))
    notes.append((f"{tag} route weights", rel(kws, ws), 1e-2))
    z2 = res1 + allsum(R.bf(y))
    res2 = R.bf(z2)
    notes.append((f"{tag} residual", rel(side.view("res", L.HIDDEN), res2), 1e-2))
    x2 = R.gemma_norm(z2, n_next, EPS)
    notes.append((f"{tag} x", rel(side.view("x", L.HIDDEN), x2), 1e-2))
    if q4_next:
        notes.append((f"{tag} next qkvz rows (MXFP4)", rel(kxd, R.quant_act(x2)), 2e-2))
    return notes


def check_chain(rows, fmt):
    shared, mine = gen(DEV, 1234), gen(DEV, 1000 + rank)
    norms = [
        (torch.randn(L.HIDDEN, generator=shared, device=DEV) * 0.1).to(BF)
        for _ in range(2 * len(STEP))
    ]
    keys = [key(kind, rows, fmt) for kind in STEP]
    side = Side(rows, len(STEP))
    notes = []
    try:
        side.view("res", L.HIDDEN).copy_(
            (torch.randn(rows, L.HIDDEN, generator=shared, device=DEV)).to(BF)
        )
        for n, (kind, k) in enumerate(zip(STEP, keys)):
            w = weights(shared, mine, kind, fmt)
            cols = L.ATTN_CORE if kind == "attn" else L.VALUE_DIM
            core = (torch.randn(rows, cols, generator=mine, device=DEV)).to(BF)
            n_post, n_next = norms[2 * n], norms[2 * n + 1]
            res = side.view("res", L.HIDDEN).clone()
            side.run(n, k, n, core, w, n_post, n_next)
            torch.cuda.synchronize()
            words = errors(side.epoch)
            if any(words.values()):
                notes.append((f"L{n} guard error words {words}", 1.0, 0.0))
                break
            notes += tail_notes(side, f"L{n} {kind}", res, core, w, n_post, n_next)
    finally:
        side.close()
    bad = [f"{name}: {v:.2e} > {tol:.0e}" for name, v, tol in notes if not v <= tol]
    for b in bad:
        print(f"[rank {rank}] chain {rows} rows {fmt}: {b}", flush=True)
    ok = all_ok(not bad)
    worst = max((v for name, v, _ in notes if "routes" not in name), default=0.0)
    log(
        f"chain ({rows} rows, {fmt} out_proj): rank 0 largest error {worst:.2e}: "
        f"{'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"chain ({rows} rows, {fmt})"]


def front_key(rows, qq, n):
    """The spec's decode key of GDN layer ``n`` (0: the step's first) of a step
    of GDN layers; ``qq``: MXFP4 in_proj_qkvz and out_proj and a bf16 state."""
    fmt, q4 = ("mxfp4", W.q4_of("mxfp4")) if qq else ("bf16", "none")
    return W.LayerBuild(
        kind="gdn",
        tokens=rows,
        tp=L.TP,
        first=n == 0,
        first_q4=q4 if n == 0 else "none",
        next_q4=q4 if n == 0 else "none",
        qkvz=fmt,
        qkvz_shuffled=qq,
        out=fmt,
        out_shuffled=qq,
        state_bf16=qq,
        rip=L.RI,
    )


def check_front(qq, rows=32, slots=40):
    """Two ``gdn`` launches, the step's first GDN layer and the next, against K1
    on 8-row chunks of the same input rows and state: the core, the conv and SSM
    state rows, the other slots kept; then each launch's tail (``tail_notes``)
    on its own core. Layer 0's experts add zero (the MoE output of a ``gdn``
    launch is checked on layer 1)."""
    fmt = "mxfp4" if qq else "bf16"
    keys = [front_key(rows, qq, n) for n in range(2)]
    k1 = gdn.K1Build(
        tokens=MAXT, first=True, eps=EPS, norm_eps=EPS, state_bf16=qq, qq=qq
    )
    try:
        _compile_only(gdn.build(k1))
        k1_ok = True
    except Exception as e:  # reported, then every rank stops
        print(f"[rank {rank}] compile K1: {type(e).__name__}: {e}", flush=True)
        k1_ok = False
    if not (compile_keys(keys) and all_ok(k1_ok)):
        raise SystemExit("a rank failed to compile the front keys")
    shared, mine = gen(DEV, 4321), gen(DEV, 2000 + rank)
    # layer n: input norm norms[2n], post-attention norm norms[2n + 1]
    norms = [
        (torch.randn(L.HIDDEN, generator=shared, device=DEV) * 0.1).to(BF)
        for _ in range(2 * len(keys) + 1)
    ]
    idx = (torch.randperm(slots - 1, generator=shared, device=DEV)[:rows] + 1).int()
    idx[[5, 17, 31]] = torch.tensor([-1, 0, -1], dtype=torch.int32, device=DEV)
    live = idx[idx > 0].long()
    keep = torch.ones(slots, dtype=torch.bool, device=DEV)
    keep[live] = False
    k1_scratch = torch.zeros(gdn.scratch_bytes(k1), dtype=torch.uint8, device=DEV)
    k1_epoch = epoch(DEV)
    side = Side(rows, len(keys))
    notes = []
    try:
        res = torch.randn(rows, L.HIDDEN, generator=shared, device=DEV).to(BF)
        for n, k in enumerate(keys):
            n_in, n_post, n_next = norms[2 * n : 2 * n + 3]
            w1 = k1_weights(mine, qq) | {"ln_w": n_in}
            conv, state = k1_state(mine, slots, bf16=qq)
            ref_conv, ref_state = conv.clone(), state.clone()
            ref_core = torch.empty(rows, CORE, dtype=BF, device=DEV)
            ref_res = torch.empty_like(res)
            for c in range(0, rows, MAXT):
                k1_epoch[:1].add_(1)
                gdn.gdn_pre(
                    k1,
                    **w1,
                    hidden=res[c : c + MAXT],
                    residual=None,
                    res_out=ref_res[c : c + MAXT],
                    conv_state=ref_conv,
                    rstate=ref_state,
                    st_idx=idx[c : c + MAXT],
                    core=ref_core[c : c + MAXT],
                    scratch=k1_scratch,
                    epoch=k1_epoch,
                    layer=0,
                )
            w = weights(shared, mine, "gdn_back", fmt)
            if n == 0:
                # The next input norm reads the unrounded residual sum, and K1
                # below reads only the stored bf16 residual: with no expert
                # output the sum is bf16-exact, so both norm the same rows.
                w |= {"w2": torch.zeros_like(w["w2"]), "r2": torch.zeros_like(w["r2"])}
            front = {
                "emb": res if n == 0 else 0,
                "n_in": n_in,
                "w_qkvz": w1["w_qkvz"],
                "w_qkvz_s": w1.get("w_qkvzs", 0),
                "w_ba": w1["w_ba"],
                "w_conv": w1["conv_w"],
                "a_log": w1["a_log"],
                "dt_bias": w1["dt_bias"],
                "gdn_norm": w1["norm_w"],
                "conv_state": conv,
                "conv_stride": conv.stride(0),
                "ssm": state,
                "ssm_stride": state.stride(0),
                "st_idx": idx,
            }
            conv0, state0 = conv.clone(), state.clone()
            side.run(n, k, n, None, w, n_post, n_next, front)
            torch.cuda.synchronize()
            words = [errors(side.epoch), errors(k1_epoch)]
            if any(any(e.values()) for e in words):
                notes.append((f"L{n} guard error words {words}", 1.0, 0.0))
                break
            tag = f"front L{n}"
            if qq:
                core = R.dequant_mx(
                    side.view("core", L.VALUE_DIM // 2, torch.uint8),
                    side.view("core_s", L.VALUE_DIM // 32, torch.int32),
                )
                want, tol = R.quant_act(ref_core.double()), 2e-2
            else:
                core, want, tol = side.view("core", L.VALUE_DIM).clone(), ref_core, 1e-2
            notes.append((f"{tag} core", rel(core[idx > 0], want[idx > 0]), tol))
            notes.append((f"{tag} conv", rel(conv[live], ref_conv[live]), 1e-2))
            notes.append((f"{tag} state", rel(state[live], ref_state[live]), 1e-2))
            kept = torch.equal(conv[keep], conv0[keep]) and torch.equal(
                state[keep], state0[keep]
            )
            notes.append((f"{tag} other slots changed", 0.0 if kept else 1.0, 0.0))
            notes += tail_notes(
                side, tag, res, core, w, n_post, n_next, k.next_q4 != "none"
            )
            res = side.view("res", L.HIDDEN).clone()
    finally:
        side.close()
    bad = [f"{name}: {v:.2e} > {tol:.0e}" for name, v, tol in notes if not v <= tol]
    for b in bad:
        print(f"[rank {rank}] front {fmt}: {b}", flush=True)
    ok = all_ok(not bad)
    worst = max((v for name, v, _ in notes if "routes" not in name), default=0.0)
    log(
        f"front ({rows} rows, {fmt} GDN projections): rank 0 largest error "
        f"{worst:.2e}: {'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"front ({rows} rows, {fmt})"]


def check_fault(rows=32, fmt="mxfp4"):
    fault = dict(fault_site=1) if rank == 3 else {}
    k = key("gdn_back", rows, fmt, stop=False, **fault)
    if not compile_keys([k]):
        raise SystemExit("a rank failed to compile the fault key")
    shared, mine = gen(DEV, 77), gen(DEV, 78 + rank)
    w = weights(shared, mine, "gdn_back", fmt)
    core = (torch.randn(rows, L.VALUE_DIM, generator=mine, device=DEV)).to(BF)
    norm = torch.zeros(L.HIDDEN, dtype=BF, device=DEV)
    # the first wait reads the peers, so a peer that launched first can time out
    # on rank 3 before rank 3's stall does: the peers wait longer
    side = Side(rows, 2, seconds=0.5 if rank == 3 else 5.0)
    try:
        t0 = time.time()
        side.run(0, k, 0, core, w, norm, norm)
        torch.cuda.synchronize()
        dt = time.time() - t0
        words = errors(side.epoch)
        flags = side.peer.bytes.view(torch.int32)[
            k2.FLAGS_OFF // 4 : k2.FLAGS_OFF // 4 + L.TP
        ].tolist()
    finally:
        side.close()
    bad = []
    if rank == 3 and words[guard.ERR_DEADLINE] != 2:
        bad.append(f"rank 3 error words {words}, want deadline word 2")
    if flags[3] != 1:
        bad.append(f"rank 3's flag not set here: {flags}")
    if bad:
        print(f"[rank {rank}] fault: {bad} in {dt:.2f} s", flush=True)
    ok = all_ok(not bad)
    log(
        f"fault (gdn_back at {rows} rows): rank 3 stalled, returned in {dt:.2f} s: "
        f"{'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"fault (gdn_back at {rows} rows)"]


def main():
    fails = check_compile()
    for rows in WIDTHS:
        for fmt in FORMATS:
            dist.barrier()
            fails += check_chain(rows, fmt)
    for qq in (False, True):
        dist.barrier()
        fails += check_front(qq)
    dist.barrier()
    fails += check_fault()
    for f in fails:
        log(f"FAIL {f}")
    log("PASS" if not fails else f"FAIL ({len(fails)})")
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
