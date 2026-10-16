"""Check K1 (``dense_gdn_pre``) bit for bit against SGLang's stock decode ops on
one gfx950 GPU, on random MXFP4 weights (shuffled with ``dense_ffn.shuffle_w``).

The residual add + input_layernorm -> in_proj_qkvz / in_proj_ba -> conv1d -> the
gated delta rule -> the gated norm, against the Gemma norm, the MXFP4 linears,
``causal_conv1d_update``, ``fused_recurrent_gated_delta_rule_packed_decode`` and
RMSNormGated. The core, the residual and both states must match, on the first
layer (no residual) and with a pad row (slot -1); other slots stay untouched.

``--tp 1``: rank 0 from the layer input. ``--tp 2`` (from the stock input norm's
output, as the runtime runs it there): each rank's half of the heads, its
column-parallel slices of the same layer, with out_proj folded in (its partial,
from the rank's K columns), then rank 0 without it.

Widths 1..``max_tokens(tp)`` (past the serving caps), each with its own weights
and Mailbox layer tag, over many input draws. Each launch's guard words must
stay clean."""

import argparse

import torch
from check_k2 import (
    BF,
    DRAWS,
    EPS,
    LAYER_TAGS,
    finish,
    gemma_norm,
    guard_epoch,
    guard_errors,
    k_cols,
    linear,
    mxfp4,
    next_step,
    report,
    rnd,
    widths,
)
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

from sgl_amd_mono.qwen3_8_dense.gfx950 import dense_ffn as D
from sgl_amd_mono.qwen3_8_dense.gfx950 import dense_gdn as K

# slots 2 .. 17 at width 16, the rest untouched
SLOTS = 20


def gdn_weights(g):
    w = {
        "ln_w": rnd(g, K.HIDDEN, scale=0.1),
        "conv_w": rnd(g, K.CONV, 1, K.CONV_W, scale=0.4),
        "a_log": rnd(g, K.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": rnd(g, K.NV, scale=0.5),
        "norm_w": (1 + rnd(g, K.HD, scale=0.1, dtype=torch.float32)).to(BF),
    }
    w["w_qkvz"], w["w_qkvzs"] = mxfp4(g, K.QKVZ, K.HIDDEN)
    w["w_ba"], w["w_bas"] = mxfp4(g, K.BA, K.HIDDEN)
    w["w_o"], w["w_os"] = mxfp4(g, K.HIDDEN, K.VAL)
    gn = RMSNormGated(
        K.HD, eps=EPS, group_size=None, norm_before_gate=True, dtype=BF
    ).cuda()
    gn.weight.data.copy_(w["norm_w"])
    w["gated_norm"] = gn
    return w


def rank_slice(w, tp, rank):
    """The rank's column-parallel share of a GDN layer: each of q | k | v | z,
    b | a, the conv's q | k | v and the heads' A_log / dt_bias cut in tp."""
    if tp == 1:
        return w
    geo = K.Geo(tp)

    def rows(t, parts):
        return torch.cat(
            [t[o + rank * n : o + (rank + 1) * n] for o, n in parts]
        ).contiguous()

    qkvz = [(0, geo.key), (K.KEY, geo.key), (2 * K.KEY, geo.val), (K.CONV, geo.val)]
    heads = slice(rank * geo.nv, (rank + 1) * geo.nv)
    out = dict(w)
    out["w_qkvz"] = rows(w["w_qkvz"], qkvz)
    out["w_qkvzs"] = rows(w["w_qkvzs"], qkvz)
    out["w_ba"] = rows(w["w_ba"], [(0, geo.nv), (K.NV, geo.nv)])
    out["w_bas"] = rows(w["w_bas"], [(0, geo.nv), (K.NV, geo.nv)])
    out["conv_w"] = rows(w["conv_w"], qkvz[:3])
    out["a_log"] = w["a_log"][heads].contiguous()
    out["dt_bias"] = w["dt_bias"][heads].contiguous()
    out["w_o"], out["w_os"] = k_cols(w["w_o"], w["w_os"], K.VAL, tp, rank)
    return out


def stock_gdn(geo, w, h, residual, conv, ssm, idx):
    """The rank's heads (``geo``) -> core, residual out; ``conv`` / ``ssm``
    updated in place."""
    x, res = gemma_norm(h, residual, w["ln_w"])
    qkvz = linear(x, w["w_qkvz"], w["w_qkvzs"])
    ba = linear(x, w["w_ba"], w["w_bas"])
    q, k, v, z = qkvz.split([geo.key, geo.key, geo.val, geo.val], dim=-1)
    b, a = ba.split([geo.nv, geo.nv], dim=-1)
    mixed = causal_conv1d_update(
        torch.cat([q, k, v], dim=-1),
        conv,
        w["conv_w"].view(geo.conv, K.CONV_W),
        None,
        "silu",
        conv_state_indices=idx,
    )
    o = torch.empty(h.size(0), 1, geo.nv, K.HD, dtype=BF, device="cuda")
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        scale=K.HD**-0.5,
        initial_state=ssm,
        out=o,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    core = w["gated_norm"](o.view(-1, K.HD), z.reshape(-1, K.HD))
    return core.view(h.size(0), geo.val), res


def gdn(fails, tp, rank, normed, sizes, draws, oproj=False):
    geo = K.Geo(tp)
    for s in sizes:
        first = s in (1, 5) and not normed
        g = torch.Generator(device="cuda").manual_seed(100 + s)
        layer = LAYER_TAGS[s % len(LAYER_TAGS)]
        w = rank_slice(gdn_weights(g), tp, rank)
        w_qkvz, w_ba = D.shuffle_w(w["w_qkvz"]), D.shuffle_w(w["w_ba"])
        key = K.K1Build(
            tokens=s,
            first=first,
            eps=EPS,
            norm_eps=EPS,
            tp=tp,
            normed=normed,
            oproj=oproj,
        )
        fold = {}
        if oproj:
            out = torch.empty(s, K.HIDDEN, dtype=BF, device="cuda")
            fold = dict(w_o=D.shuffle_w(w["w_o"]), w_os=w["w_os"], out=out)
        core = torch.empty(s, geo.val, dtype=BF, device="cuda")
        res_out = torch.empty(s, K.HIDDEN, dtype=BF, device="cuda")
        epoch = guard_epoch()
        scratch = torch.zeros(K.scratch_bytes(key), dtype=torch.uint8, device="cuda")
        idx = torch.arange(2, 2 + s, dtype=torch.int32, device="cuda")
        if s > 1:
            idx[1] = -1  # a pad row
        bad = []
        for draw in range(draws):
            h = rnd(g, s, K.HIDDEN, scale=0.5)
            residual = None if first else rnd(g, s, K.HIDDEN, scale=2.0)
            # the mamba pool's layout: conv (slots, dim, 3) dim-first, SSM fp32
            conv = rnd(g, SLOTS, geo.conv, K.SL)
            ssm = rnd(g, SLOTS, geo.nv, K.HD, K.HD, scale=0.05, dtype=torch.float32)
            conv_ref, ssm_ref = conv.clone(), ssm.clone()
            core_ref, res_ref = stock_gdn(geo, w, h, residual, conv_ref, ssm_ref, idx)
            core.fill_(7.0)
            next_step(epoch)
            if normed:
                x, _ = gemma_norm(h, residual, w["ln_w"])
                front = dict(hidden=x)
            else:
                front = dict(
                    hidden=h, residual=residual, res_out=res_out, ln_w=w["ln_w"]
                )
            K.dense_gdn_pre(
                key,
                **front,
                w_qkvz=w_qkvz,
                w_qkvzs=w["w_qkvzs"],
                w_ba=w_ba,
                w_bas=w["w_bas"],
                conv_w=w["conv_w"],
                conv_state=conv,
                a_log=w["a_log"],
                dt_bias=w["dt_bias"],
                norm_w=w["norm_w"],
                rstate=ssm,
                st_idx=idx,
                core=core,
                scratch=scratch,
                epoch=epoch,
                layer=layer,
                **fold,
            )
            exact = (
                (normed or torch.equal(res_out, res_ref))
                and torch.equal(core, core_ref)
                and torch.equal(conv, conv_ref)
                and torch.equal(ssm, ssm_ref)
            )
            if oproj:
                o_ref = linear(core_ref, w["w_o"], w["w_os"])
                exact = exact and torch.equal(out, o_ref)
            if s > 1:
                exact = exact and not core[1].any()
            if not exact:
                bad.append(draw)
        name = (
            f"K1 TP{tp} rank {rank} width {s} layer {layer} first {first} "
            f"normed {normed} oproj {oproj}"
        )
        report(fails, name, bad, draws, guard_errors(epoch))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, choices=K.TP_SIZES, required=True)
    p.add_argument("--draws", type=int, default=DRAWS, help="per width and rank")
    p.add_argument("--widths", help="comma-separated, default 1..max_tokens(tp)")
    args = p.parse_args()
    sizes = widths(args.tp, args.widths)
    fails = []
    if args.tp == 1:
        gdn(fails, 1, 0, False, sizes, args.draws)
    else:
        for rank in range(2):
            gdn(fails, 2, rank, True, sizes, args.draws, oproj=True)
        # out_proj left to the stock path
        gdn(fails, 2, 0, True, sizes, args.draws // 4)
    finish(fails)


if __name__ == "__main__":
    main()
