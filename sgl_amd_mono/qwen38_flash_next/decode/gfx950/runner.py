# SPDX-License-Identifier: Apache-2.0
"""The launches of one decoder layer, from ``weights.Layer`` operands to the
kernels' arguments, and the per-rank state every launch reads (``Launcher``).

A routed decode step of S rows runs, per layer:

    attention HC mix  -> front.hc_front
    GDN linear_attn   -> blocks.gdn_block (up to out_proj's input), or
    QSA               -> SGLang's attention, up to o_proj's input
    out_proj / o_proj -> blocks.oproj_block (+ the TP all-reduce + the attention
                         HC's combine)
    MLP HC mix        -> front.hc_front
    MoE               -> blocks.moe_block (+ the TP all-reduce + the MLP HC's
                         combine)

Every launch is ``BLOCKS`` resident CTAs; the in-kernel waits tag their mailboxes
with the step's epoch, bumped once a step inside the step's graph
(``begin_step``), so no buffer is zeroed between steps. Every wait runs under the
plugin's guard (``common/guard.py``): past SGLang's deadline, or once a rank
failed, the launch stops.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import blocks, front
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import layout as KL
from sgl_amd_mono.qwen38_flash_next.decode.gfx950.weights import Gdn, Hc, Layer, Moe

DEFAULT_EPS = 1e-6
NO_FAULT = (0, 0)  # (site, after) of ``guard.fault_injection``


def front_key(s: int, tp: int, eps: float, self_msq: bool) -> front.FrontBuild:
    return front.FrontBuild(tokens=s, tp=tp, self_msq=self_msq, eps=eps)


def gdn_key(s: int, tp: int, eps: float) -> blocks.GdnBuild:
    return blocks.GdnBuild(tokens=s, tp=tp, norm_eps=eps, shuffled=True)


def oproj_key(s: int, tp: int, gdn: bool, fault=NO_FAULT) -> blocks.OprojBuild:
    """GDN: out_proj from its MXFP4 copy; QSA: o_proj bf16 with the output gate."""
    return blocks.OprojBuild(
        tokens=s,
        tp=tp,
        oq=gdn,
        og=not gdn,
        shuffled=True,
        fault_site=fault[0],
        fault_after=fault[1],
    )


def moe_key(s: int, tp: int, rip: int, fault=NO_FAULT) -> blocks.MoeBuild:
    """Experts padded to ``rip`` (the weights' stride), w2's K read to the
    kernels' own padding (``Layout.RIP``)."""
    return blocks.MoeBuild(
        tokens=s,
        tp=tp,
        rip=rip,
        kd=KL.Layout(tp).RIP,
        fault_site=fault[0],
        fault_after=fault[1],
    )


def build_keys(
    widths: Iterable[int],
    tp: int,
    rip: int,
    hc_eps: Iterable[float] = (DEFAULT_EPS,),
    gdn_eps: Iterable[float] = (DEFAULT_EPS,),
    fault=NO_FAULT,
) -> list[tuple[Callable[[Any], Any], Any]]:
    """Every (build, key) the ``Launcher`` runs at ``widths``, sorted so that every
    rank compiles in one order. ``fault``: the ``Launcher``'s; the injection stalls
    a wait of the launches that wait on their peers."""
    keys = set()
    for s in widths:
        keys.add((blocks.build_moe, moe_key(s, tp, rip, fault)))
        keys |= {
            (front.build, front_key(s, tp, e, m)) for e in hc_eps for m in (False, True)
        }
        keys |= {(blocks.build_gdn, gdn_key(s, tp, e)) for e in gdn_eps}
        keys |= {
            (blocks.build_oproj, oproj_key(s, tp, g, fault)) for g in (False, True)
        }
    return sorted(keys, key=lambda bk: repr(bk[1]))


_ABI = {
    front.FrontBuild: front.ABI,
    blocks.GdnBuild: blocks.GDN_ABI,
    blocks.OprojBuild: blocks.OPROJ_ABI,
    blocks.MoeBuild: blocks.MOE_ABI,
}


def compile_build(build, key, stream) -> None:
    """Compile one launch's kernel without running it: FlyDSL's ``COMPILE_ONLY``
    returns before the launch."""
    launch = build(key)
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*_ABI[type(key)].zeros(), stream=stream)
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def _buf(n: int, dev) -> torch.Tensor:
    return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)


def scratch_bytes(s: int, tp: int, rip: int) -> tuple[int, int]:
    """(attention half, MLP half) scratch bytes at width ``s``: each half's largest
    launch (its HC front, then its GDN or MoE block)."""
    fronts = max(
        front.scratch_bytes(front_key(s, tp, DEFAULT_EPS, m)) for m in (False, True)
    )
    gdn = blocks.scratch_bytes(blocks.gdn_scratch_layout(gdn_key(s, tp, DEFAULT_EPS)))
    moe = blocks.scratch_bytes(blocks.moe_scratch_layout(moe_key(s, tp, rip)))
    return max(gdn, fronts), max(moe, fronts)


def peer_bytes(tp: int) -> int:
    """The guard's flags (``blocks.FLAGS``), then both parities' peer regions."""
    return guard.FLAG_BYTES + blocks.peer_bytes(KL.MAX_TOKENS, tp)


def epoch_block(deadline_s: float, device) -> torch.Tensor:
    """A launch's epoch buffer: the guard block (``common/guard.py``) with every
    wait's deadline, its word 0 the step epoch."""
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return epoch


class Launcher:
    """One rank's launch state: two scratch buffers a width (a layer's attention
    half and MLP half, so a launch never reuses the scratch the previous one may
    still poll), the epoch buffer and the peer buffer (its construction is
    collective over ``group``). ``deadline_s``: every wait's deadline."""

    def __init__(
        self,
        widths: Iterable[int],
        rank: int,
        tp: int,
        group,
        device: torch.device,
        rip: int,
        deadline_s: float,
        make_peers=None,
    ):
        if make_peers is None:
            from sgl_amd_mono.common.peer import PeerBuffer as make_peers
        self.rank, self.tp, self.rip = rank, tp, rip
        self.fault = guard.fault_injection(rank) or NO_FAULT
        halves = {s: scratch_bytes(s, tp, rip) for s in widths}
        self.attn_scratch = {s: _buf(a, device) for s, (a, _) in halves.items()}
        self.mlp_scratch = {s: _buf(m, device) for s, (_, m) in halves.items()}
        self.epoch = epoch_block(deadline_s, device)
        self.peers = make_peers(peer_bytes(tp), group, rank, tp, device)
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + guard.FLAG_BYTES

    def begin_step(self) -> None:
        self.epoch[:1].add_(1)

    def _common(self, slot: int) -> dict:
        return {"epoch": self.epoch, "layer": slot}

    def hc_front(
        self, idx: int, hc: Hc, state: torch.Tensor, msq: torch.Tensor | None, scratch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``GatedResidual.mix(state)`` -> (block input [S, H], injection logits
        [S, HC]); ``msq``: the state's sums of squares the previous block wrote,
        None to compute them (the first layer, after a PLE)."""
        s = state.size(0)
        x = state.new_empty(s, KL.HIDDEN)
        inj = state.new_empty(s, KL.HC)
        front.hc_front(
            front_key(s, self.tp, hc.eps, msq is None),
            state=state,
            msq_in=msq,
            hc_w=hc.norm,
            w_down=hc.down,
            w_up=hc.up,
            inj_out=inj,
            bin_out=x,
            scratch=scratch[s],
            **self._common(blocks.HC_SLOT + idx),
        )
        return x, inj

    def gdn(
        self,
        idx: int,
        g: Gdn,
        x: torch.Tensor,
        conv: torch.Tensor,
        ssm: torch.Tensor,
        st_idx: torch.Tensor,
    ) -> torch.Tensor:
        """``linear_attn`` up to out_proj's input -> core [S, CORE]; updates the conv
        state ``conv`` [slots, CONV, 3] (a view, strides read) and the recurrent
        state ``ssm`` [slots, NV, 128, 128] f32 in place at the slots ``st_idx``
        (int32, <= 0: a pad row, nothing written)."""
        s = x.size(0)
        core = x.new_empty(s, KL.Layout(self.tp).CORE)
        blocks.gdn_block(
            gdn_key(s, self.tp, g.norm_eps),
            x=x,
            w_qkvz=g.qkvz.weight,
            w_qkvzs=g.qkvz.scale,
            w_ba=g.ba,
            conv_w=g.conv,
            conv_state=conv,
            a_log=g.a_log,
            dt_bias=g.dt_bias,
            norm_w=g.norm,
            rstate=ssm,
            st_idx=st_idx,
            core=core,
            scratch=self.attn_scratch[s],
            **self._common(idx),
        )
        return core

    def oproj(
        self,
        w: Layer,
        core: torch.Tensor,
        gate: torch.Tensor | None,
        state: torch.Tensor,
        inj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """out_proj / o_proj, its all-reduce and the attention HC's combine -> (the
        next state, its sums of squares)."""
        out, msq = _block_out(state)
        gdn = w.gdn is not None
        blocks.oproj_block(
            oproj_key(core.size(0), self.tp, gdn, self.fault),
            core=core,
            gate=gate,
            w_o=w.gdn.out.weight if gdn else w.o_proj,
            w_os=w.gdn.out.scale if gdn else None,
            state=state,
            inj=inj,
            out=out,
            msq_out=msq,
            peers=self.peer_table,
            rank=self.rank,
            **self._common(blocks.OPROJ_SLOT + w.idx),
        )
        return out, msq

    def moe(
        self, idx: int, m: Moe, x: torch.Tensor, state: torch.Tensor, inj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The MoE, its all-reduce and the MLP HC's combine -> (the next state, its
        sums of squares)."""
        out, msq = _block_out(state)
        blocks.moe_block(
            moe_key(x.size(0), self.tp, self.rip, self.fault),
            x=x,
            w_gate=m.gate,
            w_sg=m.sgate,
            w13=m.w13,
            w13s=m.w13s,
            w2=m.w2,
            w2s=m.w2s,
            state=state,
            inj=inj,
            out=out,
            msq_out=msq,
            scratch=self.mlp_scratch[x.size(0)],
            peers=self.peer_table,
            rank=self.rank,
            **self._common(blocks.MOE_SLOT + idx),
        )
        return out, msq


def _block_out(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A block's outputs: the next state and its sums of squares."""
    s = state.size(0)
    msq = torch.empty(blocks.NRED, s, KL.HC, dtype=torch.float32, device=state.device)
    return torch.empty_like(state), msq
