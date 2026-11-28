# Adapted from vllm-project/vllm#67678 at ce476ce305 (Apache-2.0):
# tests/models/qwen3_5/amd_specialized_decode/test_layer.py
"""Check the split GDN launch (``wide.layer``, the ``gdn`` kind at 16 rows:
input norm, in_proj_qkvz and in_proj_ba, the conv and the recurrence with the
gated norm) on 8 gfx950 GPUs (torchrun), on random TP8 shards in SGLang's
layouts:

- compile: every split key the spec can make (layer 0 or not, a bf16 or an
  MXFP4 in_proj_qkvz, an f32 or a bf16 SSM state) compiles on every rank;
- front: two decode steps of two split launches (the step's first GDN layer
  from the embedding, then the next from its input and residual), bf16
  projections and an f32 state, or an MXFP4 in_proj_qkvz and a bf16 state, with
  pad rows: each launch's core, residual, and conv and SSM state rows against K1
  on 8-row chunks of the same input rows and state, a zero core on the pad rows,
  every other state slot kept. Each step has new input rows, so a launch that
  reads the previous step's hand-offs fails;
- fault: rank 3 stalls at its first wait (``stop=False``) with a deadline of
  0.5 s: rank 3 sets its deadline word and its flag reaches every rank.
"""

import datetime
import os
import sys
import time

import torch
import torch.distributed as dist
from inputs import BF, CORE, MAXT, epoch, errors, gen, k1_state, k1_weights, rnd

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import grid
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.wide import layer as W
from sgl_amd_mono.qwen38_a95b_mxfp4.spec import _compile_only

ROWS = L.SPLIT_MAX
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


def split_key(qq, first, state_bf16, **kw):
    """The spec's split key (``spec._split_keys``) of a GDN layer; ``qq``: an
    MXFP4 in_proj_qkvz."""
    return W.LayerBuild(
        kind="gdn",
        tokens=ROWS,
        tp=L.TP,
        rip=L.RI,
        first=first,
        first_q4=W.q4_of("mxfp4") if qq else "none",
        qkvz="mxfp4" if qq else "bf16",
        qkvz_shuffled=qq,
        eps=EPS,
        gdn_eps=EPS,
        state_bf16=state_bf16,
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
    keys = [
        split_key(qq, first, sb)
        for qq in (False, True)
        for first in (True, False)
        for sb in (False, True)
    ]
    t0 = time.time()
    ok = compile_keys(keys)
    log(
        f"compile: {len(keys)} split keys in {time.time() - t0:.0f} s: {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        raise SystemExit("a rank failed to compile")
    return []


class Side:
    """A peer buffer, a guard block, the barrier slots of a step of ``launches``
    split launches, their scratch and core output, as the spec's ``_Split``."""

    def __init__(self, launches, seconds=30.0):
        self.peer = PeerBuffer(W.peer_buffer_bytes(), dist.group.WORLD, rank, L.TP, DEV)
        self.peer.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()  # no peer writes into a buffer before its owner zeroed it
        self.epoch = epoch(DEV, seconds)
        self.scratch = torch.zeros(
            W.scratch_bytes(ROWS, L.RI), dtype=torch.uint8, device=DEV
        )
        off = W.scratch_layout(ROWS, L.RI)["res"][0]
        self.res = self.scratch[off : off + ROWS * L.HIDDEN * 2].view(BF)
        self.res = self.res.view(ROWS, L.HIDDEN)
        self.core = torch.empty(ROWS, L.VALUE_DIM, dtype=BF, device=DEV)
        self.sync = torch.zeros(grid.sync_words(), dtype=torch.int32, device=DEV)
        base = self.sync.data_ptr()
        slot = [base + 4 * grid.SLOT_WORDS * k for k in range(grid.SLOTS)]
        self.slots = [(slot[a], slot[b]) for a, b in grid.launch_slots(launches)]
        self.folds = base + 4 * grid.FOLD_BASE

    def run(self, n, k, w, emb, res_in, conv, state, idx):
        """Launch ``n`` of the step (layer ``n``) on the K1 weights ``w``, from
        the input rows ``emb`` and the residual ``res_in`` (None: the first)."""
        args = dict.fromkeys(W.ABI.names, 0) | {
            "scratch": self.scratch,
            "emb": emb,
            "res_in": 0 if res_in is None else res_in,
            "core_out": self.core,
            "n_in": w["ln_w"],
            "w_qkvz": w["w_qkvz"],
            "w_qkvz_s": w.get("w_qkvzs", 0),
            "w_ba": w["w_ba"],
            "w_conv": w["conv_w"],
            "a_log": w["a_log"],
            "dt_bias": w["dt_bias"],
            "gdn_norm": w["norm_w"],
            "conv_state": conv,
            "conv_stride": conv.stride(0),
            "ssm": state,
            "ssm_stride": state.stride(0),
            "st_idx": idx,
            "peers": self.peer.addresses,
            "rank": rank,
            "epoch": self.epoch,
            "layer": n,
            "sync": self.slots[n][0],
            "sync_next": self.slots[n][1],
            "folds": self.folds,
        }
        W.run(k, args)

    def close(self):
        self.peer.close()


def check_front(qq, slots=40, steps=2):
    """``steps`` decode steps of two split launches against K1 on 8-row chunks
    of the same input rows and state."""
    fmt = "mxfp4" if qq else "bf16"
    keys = [split_key(qq, n == 0, qq) for n in range(2)]
    k1s = [
        gdn.K1Build(
            tokens=MAXT, first=n == 0, eps=EPS, norm_eps=EPS, state_bf16=qq, qq=qq
        )
        for n in range(2)
    ]
    try:
        for k in k1s:
            _compile_only(gdn.build(k))
        k1_ok = True
    except Exception as e:  # reported, then every rank stops
        print(f"[rank {rank}] compile K1: {type(e).__name__}: {e}", flush=True)
        k1_ok = False
    if not (compile_keys(keys) and all_ok(k1_ok)):
        raise SystemExit("a rank failed to compile the front keys")
    shared, mine = gen(DEV, 4321), gen(DEV, 2000 + rank)
    idx = (torch.randperm(slots - 1, generator=shared, device=DEV)[:ROWS] + 1).int()
    idx[[5, 13]] = torch.tensor([-1, 0], dtype=torch.int32, device=DEV)
    pad = idx <= 0
    live = idx[~pad].long()
    keep = torch.ones(slots, dtype=torch.bool, device=DEV)
    keep[live] = False
    w = [k1_weights(mine, qq) for _ in keys]
    st = [k1_state(mine, slots, bf16=qq) for _ in keys]
    k1_scratch = torch.zeros(
        max(gdn.scratch_bytes(k) for k in k1s), dtype=torch.uint8, device=DEV
    )
    k1_epoch = epoch(DEV)
    side = Side(len(keys))
    notes = []
    try:
        for step in range(steps):
            side.epoch[:1].add_(1)
            h, res = rnd(shared, ROWS, L.HIDDEN), None
            for n, k in enumerate(keys):
                conv, state = st[n]
                ref_conv, ref_state = conv.clone(), state.clone()
                ref_core = torch.empty(ROWS, CORE, dtype=BF, device=DEV)
                ref_res = torch.empty_like(h)
                for c in range(0, ROWS, MAXT):
                    k1_epoch[:1].add_(1)
                    gdn.gdn_pre(
                        k1s[n],
                        **w[n],
                        hidden=h[c : c + MAXT],
                        residual=None if res is None else res[c : c + MAXT],
                        res_out=ref_res[c : c + MAXT],
                        conv_state=ref_conv,
                        rstate=ref_state,
                        st_idx=idx[c : c + MAXT],
                        core=ref_core[c : c + MAXT],
                        scratch=k1_scratch,
                        epoch=k1_epoch,
                        layer=n,
                    )
                conv0, state0 = conv.clone(), state.clone()
                side.run(n, k, w[n], h, res, conv, state, idx)
                torch.cuda.synchronize()
                words = [errors(side.epoch), errors(k1_epoch)]
                tag = f"step {step} L{n}"
                if any(any(e.values()) for e in words):
                    notes.append((f"{tag} guard error words {words}", 1.0, 0.0))
                    break
                core = side.core
                notes.append((f"{tag} core", rel(core[~pad], ref_core[~pad]), 1e-2))
                notes.append((f"{tag} pad core", float(core[pad].abs().max()), 0.0))
                notes.append((f"{tag} res", rel(side.res, ref_res), 1e-2))
                notes.append((f"{tag} conv", rel(conv[live], ref_conv[live]), 1e-2))
                notes.append((f"{tag} state", rel(state[live], ref_state[live]), 1e-2))
                kept = torch.equal(conv[keep], conv0[keep]) and torch.equal(
                    state[keep], state0[keep]
                )
                notes.append((f"{tag} other slots changed", 0.0 if kept else 1.0, 0.0))
                # the next layer's input: a new layer output, this launch's residual
                # (a copy: the next launch rewrites the scratch it reads)
                h, res = rnd(shared, ROWS, L.HIDDEN), side.res.clone()
    finally:
        side.close()
    bad = [f"{name}: {v:.2e} > {tol:.0e}" for name, v, tol in notes if not v <= tol]
    for b in bad:
        print(f"[rank {rank}] front {fmt}: {b}", flush=True)
    ok = all_ok(not bad)
    worst = max((v for _, v, _ in notes), default=0.0)
    log(
        f"front ({ROWS} rows, {fmt} in_proj_qkvz, {steps} steps): rank 0 largest "
        f"error {worst:.2e}: {'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"front ({ROWS} rows, {fmt})"]


def check_fault(slots=40):
    fault = dict(fault_site=1) if rank == 3 else {}
    k = split_key(True, False, True, stop=False, **fault)
    if not compile_keys([k]):
        raise SystemExit("a rank failed to compile the fault key")
    shared, mine = gen(DEV, 77), gen(DEV, 78 + rank)
    w = k1_weights(mine, qq=True)
    conv, state = k1_state(mine, slots, bf16=True)
    idx = (torch.randperm(slots - 1, generator=shared, device=DEV)[:ROWS] + 1).int()
    h, res = rnd(shared, ROWS, L.HIDDEN), rnd(shared, ROWS, L.HIDDEN)
    # the split launch reads no peer, so only rank 3 stalls: the peers wait longer
    side = Side(2, seconds=0.5 if rank == 3 else 5.0)
    try:
        side.epoch[:1].add_(1)
        t0 = time.time()
        side.run(0, k, w, h, res, conv, state, idx)
        torch.cuda.synchronize()
        dt = time.time() - t0
        words = errors(side.epoch)
        dist.barrier()  # rank 3 has flagged every peer before a rank reads its flags
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
        f"fault (split at {ROWS} rows): rank 3 stalled, returned in {dt:.2f} s: "
        f"{'OK' if ok else 'FAIL'}"
    )
    return [] if ok else [f"fault (split at {ROWS} rows)"]


def main():
    fails = check_compile()
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
