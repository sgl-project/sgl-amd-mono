# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# vllm/models/qwen4_exp/amd/specialized_decode/runtime/layers.py
"""One rank's launch state and the launches of every layer of a step, from the
layers' operands (``weights.Layer``) to the wide kernels' arguments.

The state every launch reads: the scratch every width shares (the launches of a
step run in order), the block input ``x`` and qsa_front's projections of it, the
two barrier slots the launches of a step alternate (``wide.common.grid``), the
fold counters, the step epoch (bumped once a step; a CUDA graph captures the
bump, and the TP exchange tags carry it), the health words and the peer
mailboxes of the TP exchange (``common.peer.PeerBuffer``).

Every launch is built in SGLang's setting: the conv state as SGLang lays it out
(``conv_dim_first``), the waits' deadlines from SGLang's
(``BuildContext.deadline_s``, D: a grid barrier waits D + D/8 and an exchange
line a peer has not written D + 2 D/8, at most 0.625 of the watchdog;
``wide.common.grid.deadlines``) and stop mode, where a wait past its deadline
raises this rank's flag in every other rank's peer buffer (not its own, so its
GPU's first fault, the one the runtime reports, is at a wait's page) and faults
the launch at its wait kind's page (``wide.common.grid``); a rank whose peer
raised its flag faults at the flag's page. The peer buffer's first ``PEER_HEAD``
bytes hold those flags: zeroed at allocation (``peer._alloc_uncached``) and
written by nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_flash_next.decode.weights import Layer
from sgl_amd_mono.qwen38_flash_next.decode.wide import layer as K
from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L
from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import PEER_HEAD, health_words
from sgl_amd_mono.qwen38_flash_next.decode.wide.common.grid import (
    FOLD_BASE,
    SLOT_WORDS,
    sync_words,
)
from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import exchange
from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import qsa as qsa_

NO_FAULT = (0, 0)  # (site, after) of ``guard.fault_injection``


def compile_all(keys: Iterable[K.LayerBuild]) -> None:
    for key in keys:
        K.compile_only(key)


class Launcher:
    """Per-rank launch state and the launches of every layer of a step. Its
    construction is collective over ``group`` (the peer buffer's handshake)."""

    def __init__(
        self,
        layers: list[Layer],
        widths: Iterable[int],
        tp: int,
        rank: int,
        group,
        device: torch.device,
        rip: int,
        deadline_s: float,
        make_peers=None,
    ):
        if make_peers is None:
            from sgl_amd_mono.common.peer import PeerBuffer as make_peers
        self.layers, self.widths = layers, tuple(sorted(widths))
        self.tp, self.rank, self.rip, self.deadline_s = tp, rank, rip, deadline_s
        self.launches = sum(1 if ly.kind == "gdn" else 2 for ly in layers)
        if self.launches % 2:
            raise ValueError(
                f"{self.launches} launches a step: the barrier slots need an even count"
            )
        # exact bf16 sums in rank order, as SGLang's decode all-reduce at these sizes
        self.codec = "bf16"
        self.fault = guard.fault_injection(rank) or NO_FAULT
        top = max(self.widths)
        # zeroed: the MoE's split-unit marks are compared against step tags
        self.scratch = torch.zeros(
            K.scratch_bytes(top), dtype=torch.uint8, device=device
        )
        self.x = torch.empty(top, L.HIDDEN, dtype=torch.bfloat16, device=device)
        # qsa_front's projections of x: qkv_proj's and index_qk_proj's outputs
        self.qkv = torch.empty(top, qsa_.QKV, dtype=torch.bfloat16, device=device)
        self.ixqk = torch.empty(top, qsa_.IX, dtype=torch.bfloat16, device=device)
        # the step's GDN slots as int32, per width (a graph records the copy)
        self.st_idx = {
            w: torch.zeros(w, dtype=torch.int32, device=device) for w in self.widths
        }
        self.sync = torch.zeros(sync_words(), dtype=torch.int32, device=device)
        self.health = torch.zeros(
            health_words(L.BLOCKS), dtype=torch.int32, device=device
        )
        self.epoch = torch.zeros(4, dtype=torch.int32, device=device)
        self.peers = make_peers(
            PEER_HEAD + exchange.mailbox_bytes(tp), group, rank, tp, device
        )
        self.peer_table = torch.zeros(4, dtype=torch.int64, device=device)
        self.peer_table[:tp] = self.peers.addresses + PEER_HEAD
        base = self.sync.data_ptr()
        self.slots = (base, base + 4 * SLOT_WORDS)
        self.folds = base + 4 * FOLD_BASE
        self.args = [self._static_args(ly) for ly in layers]
        self._bound: dict[int, tuple[int, int]] = {}

    def _static_args(self, layer: Layer) -> dict:
        """The arguments every step of a layer shares; the GDN states at
        ``bind``. Each ABI name the kind does not read is 0."""
        a = dict.fromkeys(K.ABI.names, 0) | layer.args
        a |= {
            "x": self.x,
            "scratch": self.scratch,
            "peers": self.peer_table,
            "rank": self.rank,
            "epoch": self.epoch,
            "layer": layer.idx,
            "folds": self.folds,
            "health": self.health,
        }
        if layer.kind == "qsa":
            a |= {"qkv_out": self.qkv, "ix_out": self.ixqk}
        return a

    def bound(self, i: int, conv: torch.Tensor, ssm: torch.Tensor) -> bool:
        """Whether layer i's launch arguments hold these GDN states."""
        return self._bound.get(i) == (conv.data_ptr(), ssm.data_ptr())

    def bind(self, i: int, conv: torch.Tensor, ssm: torch.Tensor) -> None:
        """Layer i's GDN states (``spec.gdn_state_reasons`` holds): conv [slots,
        CONV_DIM, 3] bf16 and the recurrent state [slots, NV, DV, DK] f32, slots
        their strides apart."""
        self.args[i] |= {
            "conv_state": conv,
            "conv_stride": conv.stride(0),
            "ssm": ssm,
            "ssm_stride": ssm.stride(0),
        }
        self._bound[i] = (conv.data_ptr(), ssm.data_ptr())

    def begin_step(self) -> None:
        self.epoch[:1].add_(1)

    def keys(self, i: int, rows: int, stats: bool) -> list[K.LayerBuild]:
        """The builds layer i's launches use at ``rows`` (``stats``: its input
        state did not come from a launch)."""
        layer = self.layers[i]
        common = dict(
            tokens=rows,
            tp=self.tp,
            rip=self.rip,
            hc_eps=layer.hc_eps,
            allreduce=self.codec,
            wait_s=self.deadline_s,
            stop=True,
            fault_site=self.fault[0],
            fault_after=self.fault[1],
        )
        if layer.kind == "gdn":
            return [
                K.LayerBuild(
                    kind="gdn",
                    stats=stats,
                    qkvz_shuffled=True,
                    out_shuffled=True,
                    gdn_eps=layer.gdn_eps,
                    conv_dim_first=True,
                    **common,
                )
            ]
        return [
            K.LayerBuild(kind="qsa_front", stats=stats, **common),
            K.LayerBuild(kind="qsa_back", **common),
        ]

    def step_keys(self, rows: int) -> list[K.LayerBuild]:
        """Every layer's builds at ``rows``, in step order: the first layer and a
        layer with a PLE take ``stats``."""
        out = []
        for i, layer in enumerate(self.layers):
            out += self.keys(i, rows, i == 0 or layer.ple)
        return out

    def all_keys(self) -> list[K.LayerBuild]:
        """Every build a step of any width uses, sorted so that every rank
        compiles in one order."""
        return sorted({k for w in self.widths for k in self.step_keys(w)}, key=repr)

    def launch(self, key: K.LayerBuild, i: int, seq: int, dynamic: dict) -> int:
        """Launch ``key`` for layer i as the step's launch ``seq`` (which picks
        its barrier slot); returns the next ``seq``."""
        a = self.args[i] | dynamic
        a["sync"] = self.slots[seq % 2]
        a["sync_next"] = self.slots[(seq + 1) % 2]
        K.run(key, a)
        return seq + 1
