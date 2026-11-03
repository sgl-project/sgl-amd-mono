# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the Qwen3.8-Flash-Next TP2 decode launches on random weights in SGLang's
post-load layouts under torchrun at TP2: ``torchrun --nproc-per-node 2
test/kernels/qwen38_flash_next_decode/check_blocks.py --kind gdn``. Kinds:

- ``front``: the HC front against SGLang's ``GatedResidual.mix`` (the block input
  and the injection logits), from the previous block's sums of squares or its own,
  at every width.
- ``gdn``: the GDN block (in_proj_qkvz from the MXFP4 copy ``weights.mxfp4_copy``
  makes) against AITER's conv update and fused gated delta rule and SGLang's gated
  RMSNorm: the core, the conv and recurrent state rows it writes, and the pad rows
  (state index <= 0: no state written, a zero core).
- ``blocks``: out_proj (MXFP4) and o_proj (bf16, the output gate) and the MoE (the
  experts padded as SGLang pads them, and at the kernels' own padding), each with
  the TP all-reduce and the HC combine into the streams, against torch; the router's
  picks a top-10 of the logits; the sums of squares the next front reads.
- ``stall``: rank 0 launches alone, built to keep running after a failed wait
  (``OprojBuild.stop``): its wait fails at the deadline, sets the ``deadline``
  error word and flags every rank (``common/guard.py``); the flag alone then stops
  the next launch's waits at once.
- ``steps``: a GDN and a QSA layer built by ``weights.build_layers`` from SGLang
  modules, run by ``runner.Launcher`` at every width: no wait fails, the ranks
  agree, a step reruns bitwise, and a graph of the step replays it bitwise.

Every rank exits 0 when its checks pass."""

import argparse
import dataclasses
import os
import sys
import time
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import blocks, front, runner
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import weights as W
from sgl_amd_mono.qwen38_flash_next.decode.gfx950.layout import (
    DOWN_ROWS,
    HC,
    HD,
    HH,
    HIDDEN,
    LORA,
    MAX_TOKENS,
    TOPK,
    E,
    Layout,
)

bf = torch.bfloat16
TP = 2
LAY = Layout(TP)
WIDTHS = tuple(range(1, MAX_TOKENS + 1))
COS_MIN = 0.9999
DEADLINE_S = 60.0
STALL_DEADLINE_S = 5.0  # SGLang's least
ERRORS = list(guard.ERROR_WORDS)


def _cos(a, b) -> float:
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def _rel_l2(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _randn(g, *shape, scale=1.0, dtype=bf):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(dtype)


def _u8(g, *shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=g, device=g.device).to(torch.uint8)


def _sum_ranks(t: torch.Tensor, group) -> torch.Tensor:
    """The ranks' bf16 partials summed in f32, as the launches' all-reduce."""
    s = t.float().cpu()
    dist.all_reduce(s, group=group)
    return s.to(t.device).to(bf)


def _agree(t: torch.Tensor, group, what) -> None:
    outs = [None] * TP
    dist.all_gather_object(outs, t.cpu(), group=group)
    assert all(torch.equal(outs[0], o) for o in outs), (what, "ranks disagree")


def _epoch(device, value=1, deadline_s=DEADLINE_S):
    """An epoch buffer (the guard block) at step epoch ``value``."""
    epoch = runner.epoch_block(deadline_s, device)
    epoch[0] = value
    return epoch


def _msq(state):
    """A block's sums of squares of ``state``: [NRED, S, 4] f32 by 128 columns."""
    s = state.size(0)
    sq = state.float().view(s, HC, blocks.NRED, -1).square().sum(-1)
    return sq.permute(2, 0, 1).contiguous()


def _gated_residual(g, device):
    """SGLang's ``GatedResidual`` as Qwen3.8-Flash-Next builds it, random weights."""
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HIDDEN,
        hc_lowrank=LORA,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    # bf16 as SGLang's loader builds it (the model under the bf16 default dtype)
    hc = GatedResidual(cfg).to(device=device, dtype=bf)
    with torch.no_grad():
        hc.hc_norm.weight.copy_(_randn(g, HH, scale=0.1))
        hc.input_mix_weight_down.weight.copy_(_randn(g, LORA, HH, scale=0.01))
        hc.input_mix_weight_up.weight.copy_(_randn(g, HH, LORA, scale=0.05))
        hc.block_inject_weight.weight.copy_(_randn(g, HC, HH, scale=0.01))
    return hc


def _front_operands(hc):
    return {
        "hc_w": hc.hc_norm.weight,
        "w_down": W.fuse_hc_down_inject(
            hc.input_mix_weight_down.weight, hc.block_inject_weight.weight
        ),
        "w_up": hc.input_mix_weight_up.weight,
    }


@torch.inference_mode()
def check_front(rank, group, device) -> bool:
    g = torch.Generator(device=device).manual_seed(rank)
    hc = _gated_residual(g, device)
    ops = _front_operands(hc)
    assert ops["w_down"].shape == (DOWN_ROWS, HH)
    for s in WIDTHS:
        for self_msq in (False, True):
            state = _randn(g, s, HH, scale=2.0)
            want_x, (_, normed, _) = hc.mix(state)
            want_inj = F.linear(normed, hc.block_inject_weight.weight)
            key = front.FrontBuild(tokens=s, tp=TP, self_msq=self_msq)
            x = torch.full((s, HIDDEN), float("nan"), dtype=bf, device=device)
            inj = torch.full((s, HC), float("nan"), dtype=bf, device=device)
            scratch = torch.zeros(
                front.scratch_bytes(key), dtype=torch.uint8, device=device
            )
            epoch = _epoch(device)
            front.hc_front(
                key,
                state=state,
                msq_in=None if self_msq else _msq(state),
                **ops,
                inj_out=inj,
                bin_out=x,
                scratch=scratch,
                epoch=epoch,
                layer=0,
            )
            torch.cuda.synchronize()
            assert not epoch[ERRORS].any(), ("front", s, self_msq)
            for got, want, what in ((x, want_x, "x"), (inj, want_inj, "inj")):
                assert _cos(got, want) > COS_MIN, ("front", what, s, self_msq)
                torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
    print(f"rank {rank}: front ok at widths {WIDTHS}", flush=True)
    return True


def _gdn_case(s: int, n_real: int, device, slots: int = 20):
    """One GDN layer's operands as the launch reads them, its input rows and its
    states in SGLang's pool layout (the conv window first, a dense recurrent slot);
    the live rows on slots 3 .., the pad rows on slot 0."""
    g = torch.Generator(device=device).manual_seed(0)
    w_qkvz = _randn(g, LAY.QKVZ, HIDDEN, scale=0.05)
    qkvz = W.Fp4(*W.mxfp4_copy(w_qkvz))
    w = {
        "x": _randn(g, s, HIDDEN),
        "qkvz": qkvz,
        "w_qkvz_ref": W.mxfp4_dequant(qkvz.weight, qkvz.scale),
        "w_ba": _randn(g, LAY.BA, HIDDEN, scale=0.05),
        "conv_w": _randn(g, LAY.CONV, 1, 4, scale=0.4),
        "a_log": _randn(g, LAY.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _randn(g, LAY.NV, scale=0.5),
        "norm_w": (1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
        "rstate": _randn(g, slots, LAY.NV, HD, HD, scale=0.05, dtype=torch.float32),
        "conv_pool": _randn(g, slots, 3, LAY.CONV),
    }
    idx = torch.zeros(s, dtype=torch.int32, device=device)
    idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device=device)
    w["st_idx"] = idx
    return w


def _gdn_reference(w, n: int):
    """The GDN decode of the first ``n`` rows: in_proj_qkvz on MXFP4 inputs and the
    copy's values, in_proj_ba in bf16, AITER's conv update and fused gated delta
    rule, SGLang's gated RMSNorm (norm(o) x sigmoid(z)) -> (conv pool, recurrent
    state, core [n, CORE])."""
    from aiter.ops.triton.conv.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token,
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule,
    )
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm

    x = w["x"][:n]
    qkvz = F.linear(W.mxfp4_fake_quant(x), w["w_qkvz_ref"])
    ba = F.linear(x, w["w_ba"])
    norm = RMSNorm(HD, eps=1e-6, norm_before_gate=True, activation="sigmoid")
    norm = norm.to(device=x.device, dtype=bf)
    norm.weight.data.copy_(w["norm_w"])
    conv_pool, rstate = w["conv_pool"].clone(), w["rstate"].clone()
    z = torch.empty(n, LAY.NV, HD, dtype=bf, device=x.device)
    core = torch.empty(n, LAY.NV, HD, dtype=bf, device=x.device)
    idx = w["st_idx"][:n]
    mixed, b, a = fused_reshape_causal_conv1d_update_single_token(
        qkvz.contiguous(),
        n,
        LAY.NK,
        LAY.NV,
        HD,
        HD,
        ba.contiguous(),
        z,
        core,
        conv_pool.transpose(-1, -2),
        w["conv_w"].view(-1, 4),
        None,
        "silu",
        conv_state_indices=idx,
        validate_data=True,
        qkvz_layout="flat",
    )
    fused_rearrange_sigmoid_gated_delta_rule(
        A_log=w["a_log"],
        a=a,
        b=b,
        dt_bias=w["dt_bias"],
        qkv=mixed,
        key_dim=LAY.KEY,
        value_dim=LAY.VAL,
        head_k_dim=HD,
        head_v_dim=HD,
        initial_state=rstate,
        inplace_final_state=True,
        cu_seqlens=torch.arange(n + 1, dtype=torch.int32, device=x.device),
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core.reshape(-1),
    )
    return conv_pool, rstate, norm(core, z).reshape(n, LAY.CORE)


@torch.inference_mode()
def check_gdn(rank, group, device) -> bool:
    # (1, 1) twice: a width again after others compiled
    for s, n_real in ((1, 1), (5, 4), (8, 8), (1, 1), (9, 9), (16, 15), (16, 16)):
        w = _gdn_case(s, n_real, device)
        conv_ref, rstate_ref, core_ref = _gdn_reference(w, n_real)
        key = runner.gdn_key(s, TP, 1e-6)
        conv_pool, rstate = w["conv_pool"].clone(), w["rstate"].clone()
        core = torch.full((s, LAY.CORE), float("nan"), dtype=bf, device=device)
        scratch = torch.zeros(
            blocks.scratch_bytes(blocks.gdn_scratch_layout(key)),
            dtype=torch.uint8,
            device=device,
        )
        epoch = _epoch(device)
        blocks.gdn_block(
            key,
            x=w["x"],
            w_qkvz=w["qkvz"].weight,
            w_qkvzs=w["qkvz"].scale,
            w_ba=w["w_ba"],
            conv_w=w["conv_w"],
            conv_state=conv_pool.transpose(-1, -2),
            a_log=w["a_log"],
            dt_bias=w["dt_bias"],
            norm_w=w["norm_w"],
            rstate=rstate,
            st_idx=w["st_idx"],
            core=core,
            scratch=scratch,
            epoch=epoch,
            layer=0,
        )
        torch.cuda.synchronize()
        what = ("gdn", s, n_real)
        assert not epoch[ERRORS].any(), what
        live = w["st_idx"][:n_real].long()
        errs = {
            "conv": _rel_l2(conv_pool[live], conv_ref[live]),
            "state": _rel_l2(rstate[live], rstate_ref[live]),
        }
        assert errs["conv"] < 5e-3 and errs["state"] < 5e-3, (what, errs)
        assert _cos(core[:n_real], core_ref) > COS_MIN, what
        # the pad rows: a zero core, no state written (slot 0 and every other slot)
        assert not core[n_real:].any(), what
        others = torch.ones(conv_pool.size(0), dtype=torch.bool, device=device)
        others[live] = False
        assert torch.equal(conv_pool[others], w["conv_pool"][others]), what
        assert torch.equal(rstate[others], w["rstate"][others]), what
    print(f"rank {rank}: gdn ok", flush=True)
    return True


def _peers(rank, group, device):
    from sgl_amd_mono.common.peer import PeerBuffer

    peers = PeerBuffer(runner.peer_bytes(TP), group, rank, TP, device)
    peers.bytes.zero_()
    return peers, {"peers": peers.addresses + guard.FLAG_BYTES, "rank": rank}


def moe_weights(rank, rip, device):
    """One MoE layer as SGLang holds this rank's shard after loading the Quark MXFP4
    checkpoint: the intermediate (and its scales) zero-padded from RI to ``rip`` a
    TP chunk (``_pad_tp_chunks``), the scales through ``e8m0_shuffle`` and the
    weights through ``shuffle_weight(layout=(16, 16))``; the shared expert is expert
    512. ``raw``: the unpadded rank chunks."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
        _pad_tp_chunks,
    )

    ri, ne = LAY.RI, E + 1
    gc = torch.Generator(device=device).manual_seed(7)  # replicated
    gr = torch.Generator(device=device).manual_seed(100 + rank)  # this rank's chunk
    raw = {
        "w1": _u8(gr, ne, ri, HIDDEN // 2),
        "w3": _u8(gr, ne, ri, HIDDEN // 2),
        "w2": _u8(gr, ne, HIDDEN, ri // 2),
        "w1s": _u8(gr, ne, ri, HIDDEN // 32, lo=118, hi=123),
        "w3s": _u8(gr, ne, ri, HIDDEN // 32, lo=118, hi=123),
        "w2s": _u8(gr, ne, HIDDEN, ri // 32, lo=118, hi=123),
    }

    def pad(t, dim, width):
        return _pad_tp_chunks(t, dim=dim, tp_size=1, padded=width)

    w13 = torch.cat([pad(raw["w1"], 1, rip), pad(raw["w3"], 1, rip)], dim=1)
    w13s = torch.cat([pad(raw["w1s"], 1, rip), pad(raw["w3s"], 1, rip)], dim=1)
    w2 = pad(raw["w2"], 2, rip // 2)
    w2s = pad(raw["w2s"], 2, rip // 32)

    def shuffle_scales(s):
        s0, s1, _ = s.shape
        return e8m0_shuffle(s.view(s0 * s1, -1)).view(s0, s1, -1)

    raw["w13"] = torch.cat([raw["w1"], raw["w3"]], dim=1)
    raw["w13s"] = torch.cat([raw["w1s"], raw["w3s"]], dim=1)
    return {
        "raw": raw,
        "intermediate_pad": rip - ri,
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "w_sg": _randn(gc, 1, HIDDEN, scale=0.02),
        "w13": shuffle_weight(w13.contiguous(), (16, 16)),
        "w2": shuffle_weight(w2.contiguous(), (16, 16)),
        "w13s": shuffle_scales(w13s),
        "w2s": shuffle_scales(w2s),
    }


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mxfp4_dequant(q, scales):
    """Row-major packed e2m1 bytes (low nibble first) times their e8m0 -> f32."""
    lut = torch.tensor(_FP4 + tuple(-v for v in _FP4), device=q.device)
    v = lut[torch.stack([q & 0xF, q >> 4], -1).flatten(-2).long()]
    return v * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _route(x, mw, ids=None):
    """The bf16 router's softmax top-10 (or ``ids``) renormalized, then the fused
    shared expert as pick 10 (expert 512) weighted by the sigmoid of its gate."""
    probs = torch.softmax(F.linear(x, mw["w_gate"]).float(), dim=-1)
    if ids is None:
        ids = torch.topk(probs, TOPK, dim=-1, sorted=True)[1]
    wts = probs.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    sg = torch.sigmoid(F.linear(x, mw["w_sg"]).float())
    wts = torch.cat([wts, sg], dim=-1)
    return wts, torch.cat([ids, torch.full_like(ids[:, :1], E)], dim=-1)


def _check_route(scratch, region, x, mw):
    """The launch's picks: a top-10 of the bf16 logits up to ties, their weights the
    renormalized softmax. -> the picks, for the references to use."""
    s = x.shape[0]
    words = scratch[region[0] : region[0] + s * 2 * TOPK * 8].view(torch.int32)
    words = words.view(s, 2 * TOPK, 2)[:, :, 0]
    ids, wts = words[:, :TOPK].long(), words[:, TOPK:].contiguous().view(torch.float32)
    logits = F.linear(x, mw["w_gate"]).float()
    tenth = torch.topk(logits, TOPK, dim=-1)[0][:, -1:]
    slack = 2 * 2.0**-7 * tenth.abs()
    assert (logits.gather(-1, ids) >= tenth - slack).all(), "picks not a top-10"
    assert all(len(set(r.tolist())) == TOPK for r in ids), "repeated pick"
    torch.testing.assert_close(
        wts, _route(x, mw, ids)[0][:, :TOPK], rtol=1e-2, atol=1e-3
    )
    return ids


def _moe_exact(x, mw, ids, group):
    """The experts in f32 on the dequantized weights, nothing else quantized."""
    raw, ri = mw["raw"], LAY.RI
    wts, ids = _route(x, mw, ids)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            gu = _mxfp4_dequant(raw["w13"][e], raw["w13s"][e]) @ x[t].float()
            h = F.silu(gu[:ri]) * gu[ri:]
            out[t] += wts[t, k] * (_mxfp4_dequant(raw["w2"][e], raw["w2s"][e]) @ h)
    return _sum_ranks(out.to(bf), group)


def _moe_stock(x, mw, ids, group):
    """AITER's ``fused_moe`` on the same weights and picks, called as SGLang's AITER
    MoE runner calls it."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    wts, ids = _route(x, mw, ids)
    w13 = mw["w13"].view(torch.float4_e2m1fn_x2)
    w2 = mw["w2"].view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = w2.is_shuffled = True
    out = fused_moe(
        hidden_states=x,
        w1=w13,
        w2=w2,
        topk_weight=wts.float(),
        topk_ids=ids.to(torch.int32),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        w1_scale=mw["w13s"],
        w2_scale=mw["w2s"],
        hidden_pad=0,
        intermediate_pad=mw["intermediate_pad"],
        gate_mode=GateMode.SEPARATED.value,
    )
    return _sum_ranks(out, group)


@torch.inference_mode()
def check_blocks(rank, group, device) -> bool:
    peers, launch = _peers(rank, group, device)
    epoch = _epoch(device, 0)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    gc = torch.Generator(device=device).manual_seed(11)
    w_out = _randn(gr, HIDDEN, LAY.CORE, scale=0.05)
    out_fp4 = W.Fp4(*W.mxfp4_copy(w_out))
    w_out_ref = W.mxfp4_dequant(out_fp4.weight, out_fp4.scale)
    w_o = _randn(gr, HIDDEN, LAY.CORE, scale=0.02)
    experts = {rip: moe_weights(rank, rip, device) for rip in (512, 384)}

    def run(block, key, state, inj, **kw):
        s = key.tokens
        out = torch.full_like(state, float("nan"))
        msq = torch.full((blocks.NRED, s, HC), float("nan"), device=device)
        block(key, out=out, state=state, inj=inj, msq_out=msq, **kw)
        torch.cuda.synchronize()
        return out.unflatten(-1, (HC, HIDDEN)), msq

    def combine(block, key, what, layer, **kw):
        """``block`` into a zero state with zero injection logits (each stream the
        all-reduced block output), then into a random state against the HC combine
        of that output and the sums of squares the next front reads, on mailbox
        slots ``layer`` and ``layer + 2``. -> the block output."""
        s = key.tokens
        zero = torch.zeros(s, HH, dtype=bf, device=device)
        got, _ = run(block, key, zero, zero.new_zeros(s, HC), layer=layer, **kw)
        blk = got[:, 0].contiguous()
        assert all(torch.equal(got[:, k], blk) for k in range(HC)), what
        _agree(blk, group, what)
        state, inj = _randn(gc, s, HH), _randn(gc, s, HC, scale=2.0)
        got, msq = run(block, key, state, inj, layer=layer + 2, **kw)
        m = 2.0 * torch.sigmoid(inj.float() / HC)
        want = (
            state.float().unflatten(-1, (HC, HIDDEN))
            + blk.float()[:, None] * m[..., None]
        )
        torch.testing.assert_close(
            got, want.to(bf), atol=1e-2, rtol=1e-2, msg=str(what)
        )
        torch.testing.assert_close(msq, _msq(got.flatten(-2)), rtol=1e-4, atol=0)
        _agree(got, group, what)
        return blk

    # each width twice: both epoch parities
    for s in [w for w in WIDTHS for _ in range(2)]:
        epoch[:1].add_(1)
        for slot, gdn in enumerate((True, False)):
            core = _randn(gr, s, LAY.CORE)
            gate = None if gdn else _randn(gr, s, LAY.CORE, scale=2.0)
            out = combine(
                blocks.oproj_block,
                runner.oproj_key(s, TP, gdn),
                ("oproj", gdn, s),
                core=core,
                gate=gate,
                w_o=out_fp4.weight if gdn else w_o,
                w_os=out_fp4.scale if gdn else None,
                **launch,
                epoch=epoch,
                layer=blocks.OPROJ_SLOT + 4 * slot,
            )
            if gdn:
                part = F.linear(W.mxfp4_fake_quant(core), w_out_ref)
            else:
                part = F.linear(
                    (core.float() * torch.sigmoid(gate.float())).to(bf), w_o
                )
            err = _rel_l2(out, _sum_ranks(part, group))
            assert err < 1e-2, ("oproj", gdn, s, err)

        for slot, (rip, mw) in enumerate(experts.items()):
            key = runner.moe_key(s, TP, rip)
            x = _randn(gc, s, HIDDEN)
            layout = blocks.moe_scratch_layout(key)
            scratch = torch.zeros(
                blocks.scratch_bytes(layout), dtype=torch.uint8, device=device
            )
            out = combine(
                blocks.moe_block,
                key,
                ("moe", rip, s),
                x=x,
                w_gate=mw["w_gate"],
                w_sg=mw["w_sg"],
                w13=mw["w13"],
                w13s=mw["w13s"],
                w2=mw["w2"],
                w2s=mw["w2s"],
                scratch=scratch,
                **launch,
                epoch=epoch,
                layer=blocks.MOE_SLOT + 4 * slot,
            )
            ids = _check_route(scratch, layout["route"], x.clone(), mw)
            exact = _moe_exact(x.clone(), mw, ids, group)
            err, err_stock = (
                _rel_l2(out, exact),
                _rel_l2(_moe_stock(x.clone(), mw, ids, group), exact),
            )
            assert err < 1.02 * err_stock, ("moe", rip, s, err, err_stock)
    flags = peers.bytes[: 4 * TP].view(torch.int32)
    assert not epoch[ERRORS].any() and not flags.any()
    peers.close()
    print(f"rank {rank}: oproj and moe ok at widths {WIDTHS}", flush=True)
    return True


@torch.inference_mode()
def check_stall(rank, group, device) -> bool:
    peers, launch = _peers(rank, group, device)
    flags = peers.bytes[: 4 * TP].view(torch.int32)
    epoch = _epoch(device, deadline_s=STALL_DEADLINE_S)
    key = dataclasses.replace(runner.oproj_key(1, TP, False), stop=False)
    g = torch.Generator(device=device).manual_seed(rank)
    state = _randn(g, 1, HH)

    def launch_alone() -> float:
        """The o_proj block on this rank only: its peer never sends its partial."""
        t0 = time.monotonic()
        blocks.oproj_block(
            key,
            core=_randn(g, 1, LAY.CORE),
            gate=_randn(g, 1, LAY.CORE),
            w_o=_randn(g, HIDDEN, LAY.CORE, scale=0.02),
            w_os=None,
            state=state,
            inj=_randn(g, 1, HC),
            out=torch.empty_like(state),
            msq_out=torch.empty(blocks.NRED, 1, HC, device=device),
            **launch,
            epoch=epoch,
            layer=blocks.OPROJ_SLOT,
        )
        torch.cuda.synchronize()
        return time.monotonic() - t0

    if rank == 0:
        runner.compile_build(blocks.build_oproj, key, torch.cuda.current_stream())
        took = launch_alone()
        assert STALL_DEADLINE_S * 0.9 < took < STALL_DEADLINE_S * 6, took
        # 1 + the site of the wait that timed out
        assert epoch[guard.ERR_DEADLINE] > 1, epoch.tolist()
        assert epoch[guard.ERR_FAULT] == 0, epoch.tolist()
        assert flags.tolist() == [1, 0], "this rank flags itself"
        # with this GPU's words cleared, the flag alone stops the next launch's waits
        epoch[ERRORS] = 0
        assert launch_alone() < STALL_DEADLINE_S / 2
        assert epoch[ERRORS].tolist() == [0, 1, 0], epoch.tolist()
    dist.barrier(group=group)
    if rank == 1:
        assert flags.tolist() == [1, 0], "rank 0 flags its peer"
    peers.close()
    print(f"rank {rank}: stall ok", flush=True)
    return True


def _module_layers(rank, device):
    """A GDN layer and a QSA layer as SGLang's modules hold them at TP2, random; the
    experts padded as SGLang pads them."""
    g = torch.Generator(device=device).manual_seed(5)

    def mlp():
        mw = moe_weights(rank, 512, device)
        return NS(
            gate=NS(weight=mw["w_gate"]),
            shared_expert_gate=NS(weight=mw["w_sg"]),
            experts=NS(
                w13_weight=mw["w13"],
                w2_weight=mw["w2"],
                w13_weight_scale=mw["w13s"],
                w2_weight_scale=mw["w2s"],
            ),
        )

    gdn = NS(
        linear_attn=NS(
            in_proj_qkvz=NS(weight=_randn(g, LAY.QKVZ, HIDDEN, scale=0.05)),
            in_proj_ba=NS(weight=_randn(g, LAY.BA, HIDDEN, scale=0.05)),
            conv1d=NS(weight=_randn(g, LAY.CONV, 1, 4, scale=0.4)),
            out_proj=NS(weight=_randn(g, HIDDEN, LAY.CORE, scale=0.05)),
            A_log=_randn(g, LAY.NV, scale=0.5, dtype=torch.float32),
            dt_bias=_randn(g, LAY.NV, scale=0.5, dtype=torch.float32),
            norm=NS(
                weight=(1 + _randn(g, HD, scale=0.1, dtype=torch.float32)).to(bf),
                eps=1e-6,
            ),
        ),
        attn_hyper_connection=_gated_residual(g, device),
        mlp_hyper_connection=_gated_residual(g, device),
        mlp=mlp(),
    )
    qsa = NS(
        o_proj=NS(weight=_randn(g, HIDDEN, LAY.CORE, scale=0.02)),
        attn_hyper_connection=_gated_residual(g, device),
        mlp_hyper_connection=_gated_residual(g, device),
        mlp=mlp(),
    )
    return NS(model=NS(layers=[gdn, qsa]))


# not inference_mode: building the copies reads the weights' version counters, which
# a server's loaded weights have and inference tensors do not
@torch.no_grad()
def check_steps(rank, group, device) -> bool:
    model = _module_layers(rank, device)
    copies = {}
    layers = W.build_layers(model, copies)
    experts = model.model.layers[0].mlp.experts
    rip = W.moe_rip(experts.w13_weight, experts.w2_weight, LAY.RI)
    la = runner.Launcher(WIDTHS, rank, TP, group, device, rip, DEADLINE_S)
    g = torch.Generator(device=device).manual_seed(rank)
    pool = (
        _randn(g, 20, 3, LAY.CONV),
        _randn(g, 20, LAY.NV, HD, HD, scale=0.05, dtype=torch.float32),
    )
    # the QSA core, alike on every rank
    gq = torch.Generator(device=device).manual_seed(3)

    for s in WIDTHS:
        ins = {
            "state": _randn(gq, s, HH),
            "core": _randn(gq, s, LAY.CORE),
            "gate": _randn(gq, s, LAY.CORE, scale=2.0),
            "slots": torch.arange(3, 3 + s, dtype=torch.int32, device=device),
        }
        held = tuple(t.clone() for t in pool)

        def step():
            la.begin_step()
            state, msq = ins["state"], None
            for w in layers:
                x, inj = la.hc_front(w.idx, w.attn_hc, state, msq, la.attn_scratch)
                if w.gdn is not None:
                    core, gate = (
                        la.gdn(
                            w.idx,
                            w.gdn,
                            x,
                            pool[0].transpose(-1, -2),
                            pool[1],
                            ins["slots"],
                        ),
                        None,
                    )
                else:
                    core, gate = ins["core"], ins["gate"]
                mid, msq = la.oproj(w, core, gate, state, inj)
                x, inj = la.hc_front(w.idx, w.mlp_hc, mid, msq, la.mlp_scratch)
                state, msq = la.moe(w.idx, w.moe, x, mid, inj)
            return state

        def rerun():
            for t, h in zip(pool, held):
                t.copy_(h)
            out = step().clone()
            torch.cuda.synchronize()
            return out, tuple(t.clone() for t in pool)

        first, first_pool = rerun()
        assert not la.epoch[ERRORS].any(), ("steps", s, "a wait failed")
        assert torch.isfinite(first.float()).all(), ("steps", s)
        _agree(first, group, ("steps", s))
        again, again_pool = rerun()
        assert torch.equal(again, first), ("steps", s, "a rerun differs")
        assert all(torch.equal(a, b) for a, b in zip(again_pool, first_pool))
        for t, h in zip(pool, held):
            t.copy_(h)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        for _ in range(3):
            for t, h in zip(pool, held):
                t.copy_(h)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, first), ("steps", s, "a replay differs")
            assert all(torch.equal(a, b) for a, b in zip(pool, first_pool))
        assert not la.epoch[ERRORS].any(), ("steps", s, "a wait failed")
    print(f"rank {rank}: steps ok at widths {WIDTHS}; {len(copies)} copies", flush=True)
    return True


KINDS = {
    "front": check_front,
    "gdn": check_gdn,
    "blocks": check_blocks,
    "stall": check_stall,
    "steps": check_steps,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KINDS), required=True)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert tp == TP, f"run at TP{TP}"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    return 0 if KINDS[args.kind](rank, group, device) else 1


if __name__ == "__main__":
    sys.exit(main())
