"""The GLM-5.3 layer launches under decode context parallelism (DCP, the TP group:
``Glm53Build.dcp == tp``) against the TP launches on the same inputs, under torchrun
at TP 4 or 8: ``torchrun --nproc-per-node 4 test/kernels/glm53/check_dcp.py --rows
2,4,8``. A step is ``--kinds`` (an indexer launch, then launches that read its lists)
on random weights in SGLang's layouts, every rank's q_b / W_UK / W_UV / o_proj shard
its own; the experts' down projections are zero, so a layer adds its attention only.
The DCP rank r keeps the MLA rows of slots s with s % tp == r, at s // tp (SGLang's
owner rule), and the whole index cache. For each width and rowset, on every rank:

  - the step's output is within ``--tol`` (relative, and cosine > 0.999) of TP's: the
    partials' bf16 rounding before the LSE merge is the only difference;
  - the indexer launch's list of each row is this rank's cut of TP's list, in order,
    as cache rows, and its count follows the lists (``W * 2048 + 1 + s``);
  - the first launch writes TP's MLA rows at this rank's slots and TP's index rows,
    bitwise; a later launch's MLA rows differ from TP's only at this rank's new rows;
  - every rank ends with the same output, and a graph of the step replays it bitwise;
  - with a rank's counts zeroed after the indexer launch (a negative control) the
    output misses its keys and fails the bound.

64 heads a rank attend (a q_pe staging bug above 32 heads gave NaN)."""

import argparse
import os
import sys

import msgspec
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as cl  # noqa: E402

from sgl_amd_mono.glm53.gfx950.config import (  # noqa: E402
    GLM5_GLOBAL_HEADS,
    HIDDEN,
    KV_LORA,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
)
from sgl_amd_mono.glm53.gfx950.packing import pack_fp8  # noqa: E402
from sgl_amd_mono.glm53.gfx950.runner import (  # noqa: E402
    KV_ROW,
    TOPK,
    Glm53Build,
    Glm53Caches,
    Glm53LayerOp,
    Glm53Runtime,
)


def rank_weights(layer: int, kind: str, rank: int, tp: int, device):
    """check_layer.py's weights of a ``kind`` layer, with this rank's own attention
    shard (q_b, W_UK, W_UV, o_proj) and zero expert down projections."""
    w = cl.layer_weights(layer, rank, tp, device, kind)
    g = cl._gen(1000 * layer + 500 + rank, device)
    heads = GLM5_GLOBAL_HEADS // tp
    qb = heads * (NOPE_DIM + PE_DIM)
    zero = torch.zeros(w.w_dn.shape, dtype=torch.bfloat16, device=device)
    return msgspec.structs.replace(
        w,
        w_q_b=cl._fp8(g, (qb, Q_LORA), device),
        s_q_b=cl._scales(g, (qb // 128, Q_LORA // 128), Q_LORA, device),
        w_uk=pack_fp8(cl._fp8(g, (heads * KV_LORA, NOPE_DIM), device).contiguous()),
        w_uv=pack_fp8(cl._fp8(g, (heads * V_DIM, KV_LORA), device).contiguous()),
        w_o=cl._fp8(g, (HIDDEN, heads * V_DIM), device),
        s_o=cl._scales(g, (HIDDEN // 128, heads * V_DIM // 128), heads * V_DIM, device),
        w_dn=zero.to(w.w_dn.dtype),
    )


class Step:
    """One step of ``kinds`` (layer ids from 3) at DCP size ``dcp``, inputs chained."""

    def __init__(self, kinds, dcp, positions, requests, rank, tp, group, device, args):
        W = len(positions)
        live = [p for p in positions if p is not None]
        pages = cl.Pages(max(requests) + 1, max(live) + W + 1, device)
        self.caches = [
            cl.layer_caches(3 + i, pages, cl.has_index(k), device, args.kv == "fp8")
            for i, k in enumerate(kinds)
        ]
        self.cos, self.sin = cl.rope(max(live) + W + 1, device)
        st, topk0 = cl.make_step(positions, requests, pages, device)
        self.W, self.drop = W, False
        if dcp > 1:
            topk0 = torch.zeros(W * TOPK + 1 + W, dtype=torch.int32, device=device)
            st = msgspec.structs.replace(st, topk=topk0.clone())
            self.caches = [
                Glm53Caches(
                    kv=c.kv.view(-1, tp, KV_ROW)[:, rank].contiguous(), idx=c.idx
                )
                for c in self.caches
            ]
        self.st, self.topk0 = st, topk0
        self.runtimes = {}
        for kind in sorted(set(kinds)):
            index = cl.has_index(kind)
            key = Glm53Build(
                rows=W,
                tp=tp,
                launches_per_step=kinds.count(kind),
                with_indexer=index,
                index_max_seq=cl.MAX_SEQ if index else 0,
                fp8_shuffled=True,
                kv_fp8=args.kv == "fp8",
                stop=True,
                dcp=dcp,
            )
            self.runtimes[kind] = Glm53Runtime(key, rank, group, device)
            self.runtimes[kind].set_deadline(args.deadline)
        self.ops, ordinal = [], {k: 0 for k in self.runtimes}
        for i, kind in enumerate(kinds):
            w = rank_weights(3 + i, kind, rank, tp, device)
            self.ops.append(Glm53LayerOp(self.runtimes[kind], w, ordinal[kind]))
            ordinal[kind] += 1
        g = cl._gen(3, device)
        self.h = torch.randn(W, HIDDEN, generator=g, device=device).bfloat16()

    def run(self) -> torch.Tensor:
        self.st.topk.copy_(self.topk0)
        h = self.h
        for i, op in enumerate(self.ops):
            h = op.launch(h, self.caches[i], self.cos, self.sin, self.st)
            if i == 0 and self.drop:
                self.st.topk[self.W * TOPK + 1 :].zero_()
        for rt in self.runtimes.values():
            rt.advance_step()
        return h

    def errors(self) -> int:
        from sgl_amd_mono.common import guard

        return sum(
            int(rt.epoch[i]) != 0
            for rt in self.runtimes.values()
            for i in guard.ERROR_WORDS
        )

    def close(self) -> None:
        for rt in self.runtimes.values():
            rt.close()


def _replays(step: Step, replays: int) -> bool:
    """A graph of the step, replayed, gives the eager step's output each time."""
    want = step.run().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step.run()
    ok = True
    for _ in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        ok &= torch.equal(out, want)
    del graph
    return ok


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm()).item()


def check(args, rank: int, tp: int, group, device) -> bool:
    kinds = args.kinds.split(",")
    assert cl.has_index(kinds[0]), "the shared launches read the first launch's lists"
    ok = True
    for W in map(int, args.rows.split(",")):
        for rowset in args.rowsets.split(","):
            positions, requests = cl.rows_of(W, rowset)
            live = [s for s, p in enumerate(positions) if p is not None]
            pages = cl.Pages(
                max(requests) + 1,
                max(p for p in positions if p is not None) + W + 1,
                "cpu",
            )
            slots = [
                -1 if p is None else pages.slot(r, p)
                for p, r in zip(positions, requests)
            ]
            out, caches, lists = {}, {}, {}
            for name, dcp, drop in (
                ("tp", 1, False),
                ("dcp", tp, False),
                ("drop", tp, True),
            ):
                step = Step(
                    kinds, dcp, positions, requests, rank, tp, group, device, args
                )
                step.drop = drop and rank == 0
                out[name] = step.run().clone()
                torch.cuda.synchronize()
                errors = step.errors()
                caches[name] = [
                    (c.kv.clone(), None if c.idx is None else c.idx.clone())
                    for c in step.caches
                ]
                lists[name] = step.st.topk.clone()
                if name == "tp":
                    h_in, indptr = step.h.clone(), step.st.indptr.tolist()
                replay_ok = True
                if name == "dcp" and args.replays:
                    replay_ok = _replays(step, args.replays)
                step.close()
                del step
                torch.cuda.synchronize()
                dist.barrier(group=group)
                if errors:
                    print(
                        f"FAIL rank={rank} {name} rows={W} {rowset}: error words set",
                        flush=True,
                    )
                    ok = False
                if not replay_ok:
                    print(
                        f"FAIL rank={rank} rows={W} {rowset}: a replay differs",
                        flush=True,
                    )
                    ok = False
            ref = (out["tp"] - h_in)[live].float()
            got = (out["dcp"] - h_in)[live].float()
            rel = _rel(ref, got)
            cos = torch.nn.functional.cosine_similarity(
                ref.flatten(), got.flatten(), dim=0
            ).item()
            neg = _rel(ref, (out["drop"] - h_in)[live].float())
            lists_ok = True
            for s in live:
                tp_list = lists["tp"][indptr[s] : indptr[s + 1]]
                want = (tp_list[tp_list % tp == rank] // tp).tolist()
                n = int(lists["dcp"][W * TOPK + 1 + s])
                lists_ok &= lists["dcp"][indptr[s] : indptr[s] + n].tolist() == want
            own_new = {sl // tp for sl in slots if sl >= 0 and sl % tp == rank}
            cache_ok = True
            for li, ((kv, idx), (tkv, tidx)) in enumerate(
                zip(caches["dcp"], caches["tp"])
            ):
                rows = tkv.view(-1, tp, KV_ROW)[:, rank].contiguous()
                bad = set(
                    (kv.view(torch.uint8) != rows.view(torch.uint8))
                    .any(1)
                    .nonzero()
                    .flatten()
                    .tolist()
                )
                cache_ok &= not bad if li == 0 else bad <= own_new
                if li == 0 and idx is not None:
                    cache_ok &= torch.equal(idx, tidx)
            same = [None] * tp
            dist.all_gather_object(same, out["dcp"].cpu(), group=group)
            ranks_ok = all(torch.equal(same[0], o) for o in same)
            passed = (
                rel < args.tol and cos > 0.999 and lists_ok and cache_ok and ranks_ok
            )
            neg_fails = neg > 10 * args.tol
            print(
                f"{'PASS' if passed and neg_fails else 'FAIL'} rank={rank} rows={W} "
                f"rowset={rowset} rel={rel:.3e} cos={cos:.6f} lists={lists_ok} "
                f"caches={cache_ok} ranks_equal={ranks_ok} negative_rel={neg:.3e}",
                flush=True,
            )
            ok &= passed and neg_fails
    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="index,shared,shared")
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--rowsets", default="p0,p1,v4", help="comma-separated: p0, p1, v4")
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--deadline", type=float, default=10.0)
    ap.add_argument("--replays", type=int, default=2, help="graph replays of a step")
    ap.add_argument("--tol", type=float, default=0.02, help="relative bound to TP")
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    ok = check(args, rank, tp, group, device)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
