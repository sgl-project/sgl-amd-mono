# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 on the TP2 decode path (eager decode only): each layer of
a routed step also runs SGLang's decoder layer on the same input streams and GDN
states, and the launches' outputs are compared with it:

    front.mix       each HC front's block input vs ``GatedResidual.mix``
    <kind>.streams  the layer's output streams (kind gdn or qsa)
    <kind>.delta    the same less the input streams: the layer's own contribution
    gdn.core        out_proj's input (this rank's heads)
    gdn.conv        the conv state rows the step wrote
    gdn.state       the recurrent state rows the step wrote
    gdn.update      the same less the state before the step
    gdn.track       after the step, the tracked rows' track slots against their
                    working slots (the radix cache's track-save copies them)

Each metric against its bound (cosine: at least; max_abs: at most) in
``spec.BOUNDS``, or ``spec.CHECK_MXFP4_BOUNDS`` with the MXFP4 reference. Rank 0
logs a line a step (each metric's worst value and its layer) and a line a
violation. The reference runs SGLang's layer without its PLE addition (the input
streams hold it); the GDN state rows it wrote are restored before the launch. A
QSA reference writes the step's KV and index cache rows, which the launch path
writes again for the same tokens.

The launches run the GDN in_proj_qkvz and out_proj as MXFP4 x MXFP4 (the private
copies, on MXFP4-quantized inputs) where SGLang runs the checkpoint's bf16 weights;
both run the experts as MXFP4, each its own implementation.
``SGLANG_AMD_MONO_CHECK_REF=mxfp4`` gives the reference's two GDN projections the
launches' arithmetic, so what is left of a GDN layer's difference is the
kernels' own.
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
        "_forward_input_proj",
    ),
    "srt/models/qwen4_exp.py": ("ple", "mix"),
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
def gdn_reference(attn, gdn, ref: str):
    """SGLang's GDN module ``attn`` for one reference call: yields a dict that
    receives out_proj's input as ``"core"``; with ``ref == "mxfp4"`` its
    in_proj_qkvz and out_proj compute what the launches do with ``gdn``'s
    (``weights.Gdn``) MXFP4 copies. Overrides ``_forward_input_proj`` and
    ``out_proj.forward`` on the instances only."""
    from sgl_amd_mono.qwen38_flash_next.decode.gfx950.weights import (
        mxfp4_dequant,
        mxfp4_fake_quant,
    )

    rec: dict = {}
    out_proj = attn.out_proj
    original_out = out_proj.forward
    if ref == "mxfp4":
        w_qkvz = mxfp4_dequant(gdn.qkvz.weight, gdn.qkvz.scale)
        w_out = mxfp4_dequant(gdn.out.weight, gdn.out.scale)

    def input_proj(hidden_states):
        # the fused all-reduce + norm path hands a tuple: its first entry the rows
        x = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        return F.linear(mxfp4_fake_quant(x), w_qkvz), attn.in_proj_ba(x)[0]

    def out_forward(x, *args, **kwargs):
        rec["core"] = x.detach().clone()
        if ref == "mxfp4":
            return F.linear(mxfp4_fake_quant(x), w_out), None
        return original_out(x, *args, **kwargs)

    with _instance_attr(out_proj, "forward", out_forward):
        if ref == "mxfp4":
            with _instance_attr(attn, "_forward_input_proj", input_proj):
                yield rec
        else:
            yield rec


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # f64: an f32 sum over a step's 16 x 40960 streams is not exact to 1e-6
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

    def __init__(self, positions, forward_batch, rows: int, ref: str = "bf16"):
        from sglang.srt.runtime_context import get_parallel

        self.fb = forward_batch
        self.rows = rows
        self.rank0 = get_parallel().tp_rank == 0
        self.positions = positions
        self.ref_mode = ref
        self.worst: dict[str, tuple[float, int]] = {}
        self.ref = None
        self.kcore = None

    def _note(self, name: str, v: float, layer: int) -> None:
        """Keep a metric's worst value (``worse``); a NaN sticks."""
        old = self.worst.get(name)
        if old is None or (old[0] == old[0] and worse(name, v, old[0])):
            self.worst[name] = (v, layer)

    def front(self, w, hc_module, state: torch.Tensor, x: torch.Tensor) -> None:
        ref = hc_module.mix(state)[0]
        self._note("front.mix.cosine", cosine(x, ref), w.idx)

    def before(self, layer, w, positions, state: torch.Tensor, slots, gdn) -> None:
        """SGLang's layer on ``state``; its outputs kept, the GDN rows restored."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            set_written,
            take_output,
        )

        regions = {}
        if gdn is not None:
            sel = slots[slots > 0].long()
            regions = {"conv": (gdn[0], sel), "state": (gdn[1], sel)}
        saved = {n: t[i].clone() for n, (t, i) in regions.items()}
        ple, layer.ple = layer.ple, None
        hooks = (
            nullcontext({})
            if w.gdn is None
            else gdn_reference(layer.linear_attn, w.gdn, self.ref_mode)
        )
        try:
            hidden = set_written(state.clone(), self.fb)
            with hooks as rec:
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
            "core": rec.get("core"),
        }
        self.kcore = None
        for n, (t, i) in regions.items():
            self.ref[n] = t[i].clone()
            t[i] = saved[n]

    def core(self, core: torch.Tensor) -> None:
        """The GDN launch's output (out_proj's input) of the layer being checked."""
        self.kcore = core

    def after(self, w, state: torch.Tensor) -> None:
        ref, kind = self.ref, w.kind
        if self.kcore is not None and ref["core"] is not None:
            self._note("gdn.core.cosine", cosine(self.kcore, ref["core"]), w.idx)
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
        self.kcore = None

    def track(self, states: dict, forward_batch, md) -> None:
        """``gdn.track``: in each GDN layer's pools (``states``: layer -> (conv,
        recurrent)), the step's tracked rows' track slots against their working
        slots."""
        mask = getattr(forward_batch, "mamba_track_mask", None)
        if mask is None:
            return
        src = md.mamba_cache_indices[: self.rows].long()
        dst = md.mamba_track_indices[: self.rows].long()
        keep = mask[: self.rows] & (src > 0) & (dst >= 0)
        if not bool(keep.any()):
            return
        src, dst = src[keep], dst[keep]
        for i, pools in sorted(states.items()):
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
