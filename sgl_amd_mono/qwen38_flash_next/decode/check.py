# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 on the TP2 decode path (eager decode only): each layer of
a routed step also runs SGLang's decoder layer on the same input streams and GDN
states, and the launches' outputs are compared with it:

    <kind>.streams  the layer's output streams (kind gdn or qsa)
    <kind>.delta    the same less the input streams: the layer's own contribution
    gdn.conv        the conv state rows the step wrote
    gdn.state       the recurrent state rows the step wrote
    gdn.update      the same less the state before the step
    gdn.track       after the step, the tracked rows' track slots against their
                    working slots (the radix cache's track-save copies them)
    qsa.mix         qsa_front's block input x vs ``GatedResidual.mix``
    qsa.qkv         qsa_front's qkv_proj output, which SGLang's attention is fed,
                    vs SGLang's qkv_proj of that x
    qsa.ix          likewise the indexer's index_qk_proj

Each metric against its bound (cosine: at least; max_abs: at most) in
``spec.BOUNDS``, or ``spec.CHECK_MXFP4_BOUNDS`` with the MXFP4 reference. Rank 0
logs a line a step (each metric's worst value and its layer) and a line a
violation. The reference runs SGLang's layer without its PLE addition (the input
streams hold it); the GDN state rows it wrote are restored before the launch. A
QSA reference writes the step's KV and index cache rows, which the launch path
writes again for the same tokens.

The launches run the GDN in_proj_qkvz and out_proj on their MXFP4 copies decoded
to bf16 (bf16 activations) where SGLang runs the checkpoint's bf16 weights; both
run the experts as MXFP4, each its own implementation.
``SGLANG_AMD_MONO_CHECK_REF=mxfp4`` gives the reference's two GDN projections the
launches' operands, so what is left of a GDN layer's difference is the kernels'
own.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": ("set_written", "take_output"),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/models/qwen3_5.py": (
        "linear_attn",
        "in_proj_ba",
        "out_proj",
        "qkv_proj",
        "_forward_input_proj",
    ),
    "srt/models/qwen4_exp.py": (
        "ple",
        "mix",
        "attn_hyper_connection",
        "indexer",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": ("index_qk_proj",),
    "srt/layers/attention/mamba/mamba2_metadata.py": (
        "mamba_cache_indices",
        "mamba_track_indices",
    ),
    "srt/model_executor/forward_batch_info.py": ("mamba_track_mask",),
}

REF_ENV = "SGLANG_AMD_MONO_CHECK_REF"
REFS = ("bf16", "mxfp4")


def reference_mode() -> str:
    """What the reference's GDN projections run (``SGLANG_AMD_MONO_CHECK_REF``)."""
    ref = os.environ.get(REF_ENV, "bf16")
    if ref not in REFS:
        raise ValueError(f"{REF_ENV}={ref!r}: one of {REFS}")
    return ref


@contextmanager
def _instance_attr(obj, name: str, value):
    """``obj.name = value`` (an instance attribute over the class's) for the block."""
    had = name in obj.__dict__
    old = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


@contextmanager
def mxfp4_reference(attn, args: dict):
    """SGLang's GDN module ``attn`` for one reference call with its in_proj_qkvz
    and out_proj computing what the launch does with the layer's MXFP4 copies
    (``args``: the launch's ``w_qkvz`` / ``w_out`` and their scales). Overrides
    ``_forward_input_proj`` and ``out_proj.forward`` on the instances only."""
    from sgl_amd_mono.qwen38_flash_next.decode.weights import mxfp4_dequant

    w_qkvz = mxfp4_dequant(args["w_qkvz"], args["w_qkvz_s"])
    w_out = mxfp4_dequant(args["w_out"], args["w_out_s"])

    def input_proj(hidden_states):
        # the fused all-reduce + norm path hands a tuple: its first entry the rows
        x = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        return F.linear(x, w_qkvz), attn.in_proj_ba(x)[0]

    def out_forward(x, *args, **kwargs):
        return F.linear(x, w_out), None

    with (
        _instance_attr(attn.out_proj, "forward", out_forward),
        _instance_attr(attn, "_forward_input_proj", input_proj),
    ):
        yield


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # f64: an f32 sum over a step's 192 x 10240 streams is not exact to 1e-6
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    if a.numel() == 0:
        return 1.0
    return F.cosine_similarity(a, b, dim=0, eps=1e-30).item()


def worse(name: str, v: float, than: float) -> bool:
    """Whether ``v`` is worse than ``than`` for metric ``name``: a lower cosine, a
    higher max_abs; a NaN is worse than anything."""
    return not (v >= than if name.endswith(".cosine") else v <= than)


def violations(
    values: dict[str, tuple[float, int]], bounds: dict[str, float]
) -> list[str]:
    """``values``: metric -> (worst value, its layer); the metrics past their bound."""
    out = []
    for name, bound in bounds.items():
        if name not in values:
            continue
        v, layer = values[name]
        if worse(name, v, bound):
            out.append(f"layer {layer} {name} = {v:.6g}, bound {bound:g}")
    return out


class StepCheck:
    """The checks of one routed eager step."""

    def __init__(self, positions, forward_batch, rows: int, ref: str, launcher):
        from sglang.srt.runtime_context import get_parallel

        self.fb, self.rows, self.ref_mode = forward_batch, rows, ref
        self.rank0 = get_parallel().tp_rank == 0
        self.positions = positions
        self.args = launcher.args
        self.worst: dict[str, tuple[float, int]] = {}
        self.states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.ref = None

    def _note(self, name: str, v: float, layer: int) -> None:
        """Keep a metric's worst value (``worse``); a NaN sticks."""
        old = self.worst.get(name)
        if old is None or (old[0] == old[0] and worse(name, v, old[0])):
            self.worst[name] = (v, layer)

    def before(self, layer, w, positions, state: torch.Tensor, slots, gdn) -> None:
        """SGLang's layer on ``state``; its outputs kept, the GDN rows restored.
        ``w``: the layer's ``weights.Layer``; ``gdn``: its (conv, recurrent)
        states, None for a QSA layer."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            set_written,
            take_output,
        )

        regions = {}
        if gdn is not None:
            self.states[w.idx] = gdn
            sel = slots[slots > 0].long()
            regions = {"conv": (gdn[0], sel), "state": (gdn[1], sel)}
        saved = {n: t[i].clone() for n, (t, i) in regions.items()}
        ple, layer.ple = layer.ple, None
        hooks = (
            mxfp4_reference(layer.linear_attn, self.args[w.idx])
            if gdn is not None and self.ref_mode == "mxfp4"
            else nullcontext()
        )
        try:
            hidden = set_written(state.clone(), self.fb)
            with hooks:
                out = layer(
                    positions=positions, hidden_states=hidden, forward_batch=self.fb
                )
            streams = take_output(out, self.fb).clone()
        finally:
            layer.ple = ple
        self.ref = {
            "in": state.clone(),
            "streams": streams,
            "regions": regions,
            "saved": saved,
        }
        for n, (t, i) in regions.items():
            self.ref[n] = t[i].clone()
            t[i] = saved[n]

    def front(self, i: int, layer, state, x, qkv, ixqk) -> None:
        """After QSA layer i's qsa_front: its block input and the projections
        SGLang's attention is fed, against SGLang's on the same inputs."""
        self._note(
            "qsa.mix.cosine", cosine(x, layer.attn_hyper_connection.mix(state)[0]), i
        )
        self._note("qsa.qkv.cosine", cosine(qkv, layer.qkv_proj(x)[0]), i)
        self._note("qsa.ix.cosine", cosine(ixqk, layer.indexer.index_qk_proj(x)[0]), i)

    def after(self, w, state: torch.Tensor) -> None:
        ref, kind = self.ref, w.kind
        self._note(f"{kind}.streams.cosine", cosine(state, ref["streams"]), w.idx)
        delta = state.float() - ref["in"].float()
        self._note(
            f"{kind}.delta.cosine",
            cosine(delta, ref["streams"].float() - ref["in"].float()),
            w.idx,
        )
        for n, (t, i) in ref["regions"].items():
            got = t[i]
            self._note(f"gdn.{n}.cosine", cosine(got, ref[n]), w.idx)
            if n == "state":
                before = ref["saved"][n].float()
                update = cosine(got.float() - before, ref[n].float() - before)
                self._note("gdn.update.cosine", update, w.idx)
        self.ref = None

    def track(self, forward_batch, md) -> None:
        """``gdn.track``: in each GDN layer's states, the step's tracked rows' track
        slots against their working slots."""
        mask = getattr(forward_batch, "mamba_track_mask", None)
        if mask is None:
            return
        src = md.mamba_cache_indices[: self.rows].long()
        dst = md.mamba_track_indices[: self.rows].long()
        keep = mask[: self.rows] & (src > 0) & (dst >= 0)
        if not bool(keep.any()):
            return
        src, dst = src[keep], dst[keep]
        for i, pools in sorted(self.states.items()):
            diff = torch.stack(
                [(t[dst].float() - t[src].float()).abs().max() for t in pools]
            )
            self._note("gdn.track.max_abs", diff.max().item(), i)

    def finish(self) -> None:
        from sgl_amd_mono.qwen38_flash_next.decode.spec import check_bounds

        if not self.rank0:
            return
        pos = self.positions[: self.rows]
        logger.info(
            "monokernel check step (%d rows, positions %d..%d): %s",
            self.rows,
            int(pos.min()),
            int(pos.max()),
            " ".join(
                f"{k}={v:.6g}@{layer}" for k, (v, layer) in sorted(self.worst.items())
            ),
        )
        for v in violations(self.worst, check_bounds(self.ref_mode)):
            logger.error("monokernel check violation: %s", v)
