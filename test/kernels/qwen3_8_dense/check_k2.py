"""Check K2 (``dense_ffn``) bit for bit against SGLang's stock decode ops on one
gfx950 GPU, on random MXFP4 weights (shuffled with ``dense_ffn.shuffle_w``).

``--tp 1``: o_proj folded in, then the residual add + post_attention_layernorm,
gate_up, SiLU-mul and down, against dynamic_mxfp4_quant + the Triton
``gemm_afp4wfp4``, the Gemma norm and SiluAndMul. ``--tp 2``: each rank's down
partial before the all-reduce, from the stock norm's output, its half of gate_up's
rows and down's K columns. Both also run the chain check: rows whose exact sums
need more than 24 bits, where the same GEMMs summed in K quarters differ from
stock and K2 must still match it.

``--stock``: the stock properties K2 relies on: at K2's K sizes the stock MXFP4
GEMM's output is bf16(fp32(exact sum)), and its split counts are the ones K2
follows at every width.

Widths 1..``max_tokens(tp)`` (past the serving caps), each with its own weights
and Mailbox layer tag, over many input draws: a norm reduced in another order
than the stock kernel's still matches on most rows, and shows on about one row
in 640. Each launch's guard words must stay clean."""

import argparse
import sys

import torch
from sglang.kernels.ops.layernorm import minimax_m3_rmsnorm as M
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4 as q

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen3_8_dense.gfx950 import dense_ffn as D

BF = torch.bfloat16
EPS = 1e-6
DRAWS = 512
# Mailbox tag slots: the layer index a launch passes as ``layer``
LAYER_TAGS = (0, 17, 63)
# A wait's deadline: far past any launch, so only a hang trips it
DEADLINE_S = 10.0
# K2's GEMM K sizes: gate_up, down at TP1, down at TP2
STOCK_KS = (5120, 17408, 8704)


def widths(tp, arg=None):
    if arg:
        return [int(s) for s in arg.split(",")]
    return list(range(1, D.max_tokens(tp) + 1))


def guard_epoch(deadline_s=DEADLINE_S):
    """A launch's epoch buffer: word 0 the step's epoch, then the guard block
    with its deadline set (``guard.py``)."""
    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device="cuda")
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def next_step(epoch):
    epoch[:1].add_(1)


def guard_errors(epoch):
    return [int(epoch[i]) for i in guard.ERROR_WORDS]


def mxfp4(g, n, k):
    """Random MXFP4 [n, k]: fp4 pairs and e8m0 scales 2^-7 .. 2^-1."""
    w = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    ws = torch.randint(
        120, 127, (n, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    return w, ws


def rnd(g, *shape, scale=1.0, dtype=BF):
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)


def linear(x, w, ws):
    """The stock Quark MXFP4 linear on ROCm."""
    xq, xs = q.dynamic_mxfp4_quant(x)
    y = torch.empty(x.size(0), w.size(0), dtype=BF, device=x.device)
    q.gemm_afp4wfp4(xq, w, xs, ws, BF, y)
    return y


def k_cols(w, ws, k, tp, rank):
    """Row-parallel: the rank's input columns of an MXFP4 [n, k]."""
    n = k // tp
    return (
        w[:, rank * n // 2 : (rank + 1) * n // 2].contiguous(),
        ws[:, rank * n // 32 : (rank + 1) * n // 32].contiguous(),
    )


def gemma_norm(x, residual, w):
    """GemmaRMSNorm as the stock path runs it on ROCm with aiter -> x, residual."""
    if residual is None:
        return M.gemma_rmsnorm(x, w, EPS), x
    return M.gemma_fused_add_rmsnorm(x, residual, w, EPS)


_FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def dequant(wq, ws):
    """MXFP4 [n, k / 2] (low nibble the even column), e8m0 [n, k / 32] -> fp64."""
    lut = torch.tensor(_FP4, dtype=torch.float64, device=wq.device)
    v = torch.stack([lut[(wq & 0xF).long()], lut[(wq >> 4).long()]], -1)
    s = torch.pow(2.0, ws.double() - 127.0).repeat_interleave(32, dim=1)
    return v.reshape(wq.size(0), -1) * s


def report(fails, name, bad, draws, errors=(0, 0, 0)):
    print(f"{name}: {len(bad)} of {draws} draws differ, guard words {list(errors)}",
          flush=True)  # fmt: skip
    if bad or any(errors):
        fails.append(name)


def finish(fails):
    if fails:
        print(f"FAILED: {', '.join(fails)}", flush=True)
        sys.exit(1)
    print("all checks passed", flush=True)


def ffn_tp1(fails, sizes, draws):
    act = SiluAndMul()
    for s in sizes:
        g = torch.Generator(device="cuda").manual_seed(s)
        layer = LAYER_TAGS[s % len(LAYER_TAGS)]
        w_o, w_os = mxfp4(g, D.HIDDEN, D.CORE)
        w13, w13s = mxfp4(g, 2 * D.INTER, D.HIDDEN)
        w2, w2s = mxfp4(g, D.HIDDEN, D.INTER)
        ln_w = rnd(g, D.HIDDEN, scale=0.1)
        weights = dict(
            w_o=D.shuffle_w(w_o),
            w_os=w_os,
            ln_w=ln_w,
            w13=D.shuffle_w(w13),
            w13s=w13s,
            w2=D.shuffle_w(w2),
            w2s=w2s,
        )
        key = D.FfnBuild(tokens=s, eps=EPS, shuffled=True, oproj=True)
        out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
        res_out = torch.empty_like(out)
        epoch = guard_epoch()
        # one scratch for every launch: the epoch tags must keep it valid
        scratch = torch.zeros(D.scratch_bytes(key), dtype=torch.uint8, device="cuda")
        bad = []
        for draw in range(draws):
            core = rnd(g, s, D.CORE, scale=0.5)
            residual = rnd(g, s, D.HIDDEN, scale=2.0)
            x, res_ref = gemma_norm(linear(core, w_o, w_os), residual, ln_w)
            out_ref = linear(act(linear(x, w13, w13s)), w2, w2s)
            next_step(epoch)
            D.dense_ffn(
                key,
                hidden=core,
                residual=residual,
                out=out,
                res_out=res_out,
                scratch=scratch,
                epoch=epoch,
                layer=layer,
                **weights,
            )
            if not (torch.equal(res_out, res_ref) and torch.equal(out, out_ref)):
                bad.append(draw)
        report(fails, f"K2 TP1 width {s} layer {layer}", bad, draws,
               guard_errors(epoch))  # fmt: skip


def ffn_tp2(fails, sizes, draws):
    act = SiluAndMul()
    inter = D.FGeo(2).inter
    for rank in range(2):
        for s in sizes:
            g = torch.Generator(device="cuda").manual_seed(200 + s)
            layer = LAYER_TAGS[s % len(LAYER_TAGS)]
            w13, w13s = mxfp4(g, 2 * D.INTER, D.HIDDEN)
            w2, w2s = mxfp4(g, D.HIDDEN, D.INTER)
            ln_w = rnd(g, D.HIDDEN, scale=0.1)
            # column-parallel gate_up: the rank's half of gate, then of up
            rows = [
                slice(o + rank * inter, o + (rank + 1) * inter) for o in (0, D.INTER)
            ]
            w13 = torch.cat([w13[r] for r in rows]).contiguous()
            w13s = torch.cat([w13s[r] for r in rows]).contiguous()
            w2, w2s = k_cols(w2, w2s, D.INTER, 2, rank)
            w13_sh, w2_sh = D.shuffle_w(w13), D.shuffle_w(w2)
            key = D.FfnBuild(tokens=s, shuffled=True, tp=2, normed=True)
            out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
            epoch = guard_epoch()
            scratch = torch.zeros(
                D.scratch_bytes(key), dtype=torch.uint8, device="cuda"
            )
            bad = []
            for draw in range(draws):
                h = rnd(g, s, D.HIDDEN, scale=0.5)
                x, _ = gemma_norm(h, rnd(g, s, D.HIDDEN, scale=2.0), ln_w)
                out_ref = linear(act(linear(x, w13, w13s)), w2, w2s)
                next_step(epoch)
                D.dense_ffn(
                    key,
                    hidden=x,
                    residual=None,
                    ln_w=None,
                    w13=w13_sh,
                    w13s=w13s,
                    w2=w2_sh,
                    w2s=w2s,
                    out=out,
                    res_out=None,
                    scratch=scratch,
                    epoch=epoch,
                    layer=layer,
                )
                if not torch.equal(out, out_ref):
                    bad.append(draw)
            report(fails, f"K2 TP2 rank {rank} width {s} layer {layer}", bad, draws,
                   guard_errors(epoch))  # fmt: skip


def linear_parts(x, w, ws, splits=None, block_k=128):
    """The stock GEMM's fp32 split partials [n, M, N] (``splits``: forced to
    that many equal K splits), or its bf16 output [M, N] for one split."""
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    xq, xs = q.dynamic_mxfp4_quant(x)
    cfg = None
    if splits is not None:
        cfg = dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=1,
            num_warps=4,
            num_stages=2,
            waves_per_eu=2,
            matrix_instr_nonkdim=16,
            cache_modifier=None,
            NUM_KSPLIT=splits,
        )
    return gemm_afp4wfp4(xq, w, xs, ws, BF, config=cfg, skip_reduce=True)


def linear_quartered(x, w, ws):
    """The GEMM as four K quarters, each chained from zero, summed in order."""
    p = linear_parts(x, w, ws, splits=4)
    assert p.dim() == 3 and p.size(0) == 4
    return (((p[0] + p[1]) + p[2]) + p[3]).to(BF)


def stock_accumulation(fails):
    """On typical inputs at K2's K sizes the stock MXFP4 GEMM's output is
    bf16(fp32(exact sum)), so a K2 mismatch is a K2 bug, not a rounding corner."""
    g = torch.Generator(device="cuda").manual_seed(7)
    for k in STOCK_KS:
        w, ws = mxfp4(g, 512, k)
        wd = dequant(w, ws)
        bad = []
        for draw in range(8):
            x = torch.randn(8, k, generator=g, device="cuda")
            if draw % 2:  # heavy-tailed rows
                x = x * torch.randn(8, k, generator=g, device="cuda").mul(2).exp()
            x = x.to(BF)
            xq, xs = q.dynamic_mxfp4_quant(x)
            exact = dequant(xq.view(torch.uint8), xs.view(torch.uint8)) @ wd.T
            if (linear(x, w, ws) != exact.float().to(BF)).any():
                bad.append(draw)
        report(fails, f"stock GEMM K {k} is its exact sum rounded", bad, 8)


def stock_splits(fails):
    """K2's split counts are the stock GEMM's at every width (from the partials
    its config yields)."""
    g = torch.Generator(device="cuda").manual_seed(3)
    for tp in D.TP_SIZES:
        inter = D.FGeo(tp).inter
        shapes = [("gate_up", 2 * inter, D.HIDDEN, D.gu_splits)]
        shapes.append(("down", D.HIDDEN, inter, lambda s, tp=tp: D.d_splits(s, tp)))
        if tp == 1:
            shapes.append(("o_proj", D.HIDDEN, D.CORE, lambda s: D.O_SPLITS))
        for name, n, k, want in shapes:
            w, ws = mxfp4(g, n, k)
            got, wanted = [], []
            for s in widths(tp):
                p = linear_parts(rnd(g, s, k), w, ws)
                got.append(p.size(0) if p.dim() == 3 else 1)
                wanted.append(want(s))
            print(f"stock {name} TP{tp} splits by width: {got}, K2 {wanted}",
                  flush=True)  # fmt: skip
            if got != wanted:
                fails.append(f"stock {name} TP{tp} splits")


def cancel_blocks(w, ws, rows, blocks, g):
    """Rows ``rows`` of an MXFP4 [n, k] zeroed but for ``blocks``: (block, kind),
    kind "big" (random fp4, scale 2^30), "neg" (the previous big block negated)
    or "small" (random, scale 2^0)."""
    k_blocks = ws.size(1)
    for r in rows:
        w[r] = 0
        ws[r] = 127
        prev = None
        for b, kind in blocks:
            assert 0 <= b < k_blocks
            cols = slice(16 * b, 16 * b + 16)
            if kind == "neg":
                w[r, cols] = w[r, 16 * prev : 16 * prev + 16] ^ 0x88
                ws[r, b] = ws[r, prev]
                continue
            w[r, cols] = torch.randint(
                0, 256, (16,), dtype=torch.uint8, device="cuda", generator=g
            )
            ws[r, b] = 157 if kind == "big" else 127
            prev = b


def copy_block(t, dst, src):
    """Columns 32 dst .. of ``t`` (bf16 [.., k]) := columns 32 src .."""
    t[..., 32 * dst : 32 * dst + 32] = t[..., 32 * src : 32 * src + 32]


def chain_weights(g, hidden, inter):
    """Gate rows (seen through down rows reading only their columns) and down
    rows each get a cancelling pair across a quarter boundary inside a split
    (P1) and across the split boundary (P2)."""
    w13, w13s = mxfp4(g, 2 * inter, hidden)
    w2, w2s = mxfp4(g, hidden, inter)
    for w, ws, rows, k in ((w13, w13s, (0, 4), hidden), (w2, w2s, (0, 4), inter)):
        nb = k // 32
        qb, sb = nb // 4, nb // 2
        p1 = [(qb - 1, "big"), (qb, "neg"), (qb + 1, "small")]
        p2 = [(1, "small"), (sb - 1, "big"), (sb, "neg"), (nb - 1, "small")]
        cancel_blocks(w, ws, range(rows[0], rows[0] + 4), p1, g)
        cancel_blocks(w, ws, range(rows[1], rows[1] + 4), p2, g)
    # down's big blocks read equal H blocks: gate / up rows copied
    nb = inter // 32
    for dst, src in ((nb // 4, nb // 4 - 1), (nb // 2, nb // 2 - 1)):
        for base in (0, inter):
            w13[base + 32 * dst : base + 32 * dst + 32] = w13[
                base + 32 * src : base + 32 * src + 32
            ]
            w13s[base + 32 * dst : base + 32 * dst + 32] = w13s[
                base + 32 * src : base + 32 * src + 32
            ]
    # down rows 8 .. 11 read the crafted gate rows' columns (H block 0) only
    w2[8:12, 16:] = 0
    return w13, w13s, w2, w2s


def stock_chains(fails, tp, sizes):
    """K2 reproduces the stock GEMMs' arithmetic, not only their exact sums."""
    act = SiluAndMul()
    inter = D.FGeo(tp).inter
    g = torch.Generator(device="cuda").manual_seed(40 + tp)
    w13, w13s, w2, w2s = chain_weights(g, D.HIDDEN, inter)
    w13_sh, w2_sh = D.shuffle_w(w13), D.shuffle_w(w2)
    nb = D.HIDDEN // 32
    for s in sizes:
        h = rnd(g, s, D.HIDDEN, scale=0.5)
        r = rnd(g, s, D.HIDDEN, scale=2.0)
        ln_w = rnd(g, D.HIDDEN, scale=0.1)
        # the gate rows' big blocks read equal x blocks
        for t in (h, r, ln_w):
            copy_block(t, nb // 4, nb // 4 - 1)
            copy_block(t, nb // 2, nb // 2 - 1)
        x, res_ref = gemma_norm(h, r, ln_w)
        ref = linear(act(linear(x, w13, w13s)), w2, w2s)
        quart = linear_quartered(act(linear_quartered(x, w13, w13s)), w2, w2s)
        epoch = guard_epoch()
        next_step(epoch)
        out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
        if tp == 1:
            key = D.FfnBuild(tokens=s, eps=EPS, shuffled=True)
            args = dict(hidden=h, residual=r, ln_w=ln_w)
            res_out = torch.empty_like(out)
        else:
            key = D.FfnBuild(tokens=s, shuffled=True, tp=2, normed=True)
            args = dict(hidden=x, residual=None, ln_w=None)
            res_out = None
        scratch = torch.zeros(D.scratch_bytes(key), dtype=torch.uint8, device="cuda")
        D.dense_ffn(
            key,
            **args,
            w13=w13_sh,
            w13s=w13s,
            w2=w2_sh,
            w2s=w2s,
            out=out,
            res_out=res_out,
            scratch=scratch,
            epoch=epoch,
            layer=0,
        )
        quartered_differs = not torch.equal(quart, ref)
        exact = torch.equal(out, ref) and (tp == 2 or torch.equal(res_out, res_ref))
        errors = guard_errors(epoch)
        print(f"K2 chains TP{tp} width {s}: quartered differs {quartered_differs}, "
              f"K2 exact {exact}, guard words {errors}", flush=True)  # fmt: skip
        if not (quartered_differs and exact) or any(errors):
            fails.append(f"K2 chains TP{tp} width {s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, choices=D.TP_SIZES)
    p.add_argument("--stock", action="store_true")
    p.add_argument("--draws", type=int, help="per width (and rank at TP2)")
    p.add_argument("--widths", help="comma-separated, default 1..max_tokens(tp)")
    args = p.parse_args()
    assert args.tp or args.stock, "pass --tp or --stock"
    fails = []
    if args.stock:
        stock_accumulation(fails)
        stock_splits(fails)
    if args.tp == 1:
        ffn_tp1(fails, widths(1, args.widths), args.draws or DRAWS)
    elif args.tp == 2:
        ffn_tp2(fails, widths(2, args.widths), args.draws or DRAWS // 2)
    if args.tp:
        stock_chains(fails, args.tp, widths(args.tp, args.widths))
    finish(fails)


if __name__ == "__main__":
    main()
