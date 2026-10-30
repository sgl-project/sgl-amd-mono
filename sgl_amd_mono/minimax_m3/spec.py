# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 on gfx950: the monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at build: a routed decode or
verify step runs :meth:`_Built.forward` here, which takes and returns what
``MiniMaxM3SparseForCausalLM.forward`` does -- the embedding, every layer, the
final norm and SGLang's logits processor, with SGLang's own modules wherever
SGLang owns the state.

Each sparse MoE layer is one launch, K1 fused in front of K4::

    K1 part   (ar, res) -> h, q; K / V / index cache insert; indexer scores
    K4 part   top-k, attention .. FFN, both all-reduces in-kernel -> (ar, h_mid)

``(ar, res)`` is the (all-reduced MLP output, residual) pair a decoder layer
hands its input norm, so a layer computes the same values either way. A launch
reads the previous layer's ``(ar, h_mid)`` while it writes its own, so both
alternate between two buffers by layer parity, baked into each layer's
arguments at build: these buffers are never reallocated. The embedding, the
leading dense-MLP layers and the final norm run the model's own modules.

Per step, inside the graph where there is one, the first routed layer derives
the row tables the kernels read from SGLang's decode buffers: a [tokens, blocks]
int32 block table (page-128 block ids from ``req_to_token``) and int32 sequence
lengths; positions and the slot mapping are read as the int64 buffers they are.
Sparse K / V are page16 SHUFFLE planes (the AITER sparse attention layout
prefill writes too), index-K is NHD. The caches and the request table are bound
at build.

**This spec requires sgl-project/sglang#41998.** Its kernels read the page-16
SHUFFLE sparse K / V that PR writes, and they cannot produce that layout
themselves: routing serves only decode and verify steps, so prefill always runs
the standard path and writes the context the kernels later read. Stock SGLang's
``MiniMaxSparseKVPool`` allocates a buffer of exactly the right shape and dtype
-- ``sparse_kv_views`` is a zero-copy ``.view()`` over it -- but fills it in NHD
order, because the SHUFFLE order comes only from
``aiter.reshape_and_cache(..., asm_layout=True)``, which #41998 introduces.

``SGLANG_AMD_MONO_CHECK=1`` runs SGLang's modules on a copy of every sparse layer's
input in eager steps and logs both outputs' differences on the live rows
(``sgl_amd_mono/minimax/check.py``); the bounds in ``NUMERICS`` come from these runs.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import TYPE_CHECKING

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

if TYPE_CHECKING:  # annotations only, under ``from __future__ import annotations``
    from sglang.srt.monokernels.spec import BuildContext, Plan

# The interface version a spec declares (``MonoSpec.api_version``), read from the module
# MonoSpec came from rather than written as a literal, so the spec states the version of
# the SGLang it loads under; omitted if that SGLang does not version the interface.
MONOKERNEL_API_VERSION = getattr(
    sys.modules[MonoSpec.__module__], "MONOKERNEL_API_VERSION", None
)

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.minimax_m3 import HF_CONFIG
from sgl_amd_mono.minimax_m3.gfx950.config import (
    LAYER_SLOTS,
    SPARSE_BLOCK,
    TP_SIZES,
    WIDTHS,
    Dims,
)

logger = logging.getLogger(__name__)

# The SGLang names this glue reads, by file under the sglang package (``f(p)``:
# a parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
#
# The three ``minimax_sparse_backend`` / ``memory_pool`` names marked below do
# NOT resolve against stock SGLang main: they arrive with #41998. Until that
# merges, the contract test fails on them, which is the honest signal -- this
# spec cannot run on a SGLang that cannot write the KV layout it reads.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        # MONOKERNEL_API_VERSION is read optionally (see above), so it is not pinned here
        "BuildContext",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Plan",
        "Requirements",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/eplb/expert_distribution.py": ("get_global_expert_distribution_recorder",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "stream_of",
        "final_norm",
    ),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList",),
    # check mode's reference forward
    "srt/layers/layer_boundary/residual/stream.py": (
        "ResidualStream",
        "from_handoff",
        "export",
        "write",
        "record",
    ),
    # Requirements.env keys. Not imports, so the contract test would not
    # otherwise guard them -- but an env var this spec requires and SGLang does
    # not define would make the requirement unsatisfiable at startup, which is
    # worth catching here rather than there.
    # the server arguments ``SPEC.requires`` and config_disable_reasons read
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "page_size",
        "moe_runner_backend",
        "disable_shared_experts_fusion",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "enable_hierarchical_cache",
        "hicache_io_backend",
        "hicache_mem_layout",
        "enable_lmcache",
        "enable_flexkv",
        "disaggregation_decode_enable_offload_kvcache",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
    ),
    # Requirements.env keys and ``envs``. #41998's switches are not here: they are read
    # by name (PENDING_ENV), so an SGLang without them is refused with a reason.
    "srt/environ.py": (
        "SGLANG_USE_AITER",
        "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ",
        "SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
        "envs",
    ),
}


# Check mode's bounds: ``minimax/check.py`` logs, per sparse layer, each output's largest
# difference over the largest reference value (rel) and least row cosine on the live rows,
# the MLP output on the rows routed alike; test/e2e/test_minimax_m3_server.py holds every
# launch to these. Measured at TP4 (#8).
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.residual.rel": 0.1,
        "layer.residual.cosine": 0.995,
        "layer.mlp_out.rel": 0.5,
        "layer.mlp_out.cosine": 0.4,
    },
)


# TP8: a rank's MoE width is 3072 / 8 = 384, which aiter pads to 512 unless
# SGLANG_AITER_K3_OPT is on (quark_w4a4_mxfp4_moe.py, ``pad_for_aiter``), and the kernels read
# the unpadded 384-wide experts. SGLang reads the switch once, when each TP worker imports that
# module; the workers are spawned after the server arguments resolve, which is when SGLang runs
# config_disable_reasons, so the switch is defaulted on here rather than asked of every user.
# At TP2 / TP4 the width is a multiple of 256 and nothing is padded either way, so it is left
# alone there: it also picks the model path's MoE kernels.
K3_OPT_ENV = "SGLANG_AITER_K3_OPT"


def _tp8_expert_layout_reasons(cfg) -> list[str]:
    if cfg.tp_size != 8:
        return []
    if K3_OPT_ENV not in os.environ:
        os.environ[K3_OPT_ENV] = "1"
        logger.info(
            "minimax_m3 mono at TP8: %s defaulted to 1 (unpadded experts)", K3_OPT_ENV
        )
    if os.environ[K3_OPT_ENV].lower() not in ("1", "true"):
        return [
            f"TP8 needs {K3_OPT_ENV}=1 (or unset): the kernels read the unpadded "
            "384-wide experts, which aiter pads to 512 without it"
        ]
    return []


def _has_flydsl() -> bool:
    return importlib.util.find_spec("flydsl") is not None


# switches of sgl-project/sglang#41998 (on #41708) for the page16 SHUFFLE K / V, not in main yet:
# read by name, so an SGLang without them is refused with a reason; pin in SGLANG_READS once merged
PENDING_ENV = {
    # prefill writes the page16 SHUFFLE K / V the kernels read
    "SGLANG_M3_USE_AITER_SPARSE_PA": True,
    "SGLANG_M3_USE_AITER_PREFILL_INDEXER": True,
    "SGLANG_M3_USE_AITER_INDEXER": True,
}


def _pending_env_reasons() -> list[str]:
    from sglang.srt.environ import envs

    reasons = []
    for name, want in PENDING_ENV.items():
        var = getattr(envs, name, None)
        if var is None:
            reasons.append(
                f"needs SGLang with {name} (the page16 SHUFFLE K / V of "
                "sgl-project/sglang#41998 and #41708)"
            )
        elif var.get() != want:
            reasons.append(f"{name} {var.get()!r} is not {want!r}")
    return reasons


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _sparse_ids(layers) -> list[int]:
    """The MoE layers, by the model's own definition rather than ours.

    ``MiniMaxM3DecoderLayer`` sets ``is_layer_sparse`` from ``moe_layer_freq``;
    reading the attribute rather than recomputing it keeps the two from drifting
    apart, and the layer objects are what the launches actually replace.
    """
    return [i for i, layer in enumerate(layers) if layer.is_layer_sparse]


# --------------------------------------------------------------- refusals
def config_disable_reasons(cfg, hf_config) -> list[str]:
    """Checks that cannot be stated declaratively in ``Requirements``."""
    t = _text_config(hf_config)
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in HF_CONFIG.items()
        if getattr(t, k, None) != v
    ]
    if t.num_hidden_layers > LAYER_SLOTS:
        reasons.append(
            f"{t.num_hidden_layers} layers, the mailbox epochs hold {LAYER_SLOTS}"
        )
    from sgl_amd_mono.minimax_m3.gfx950.weights import sparse_selection

    try:
        sparse_selection(t)
    except ValueError as e:
        reasons.append(str(e))
    if cfg.speculative_algorithm is not None and (
        cfg.speculative_eagle_topk != 1
        or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        reasons.append(
            "speculative decoding: the kernels verify one linear draft chain "
            "(--speculative-eagle-topk 1, draft tokens = steps + 1)"
        )
    if cfg.enable_hierarchical_cache and (
        cfg.hicache_io_backend != "kernel" or cfg.hicache_mem_layout != "layer_first"
    ):
        reasons.append(
            "HiCache keeps the page16 SHUFFLE K / V only with --hicache-io-backend "
            "kernel and --hicache-mem-layout layer_first"
        )
    if cfg.enable_lmcache or cfg.enable_flexkv or cfg.disaggregation_decode_enable_offload_kvcache:  # fmt: skip
        reasons.append("KV offload other than HiCache")
    if not _has_flydsl():
        reasons.append("needs FlyDSL")
    reasons += _tp8_expert_layout_reasons(cfg)
    return reasons + _pending_env_reasons()


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Checks against the loaded model, run on every rank and reported together."""
    from sglang.srt.runtime_context import get_parallel

    reasons = []
    backbone = model.model
    if backbone.start_layer != 0 or backbone.end_layer != len(backbone.layers):
        reasons.append(
            f"pipelined: layers {backbone.start_layer}..{backbone.end_layer} of "
            f"{len(backbone.layers)}"
        )
    sparse = _sparse_ids(backbone.layers)
    if not sparse or sparse != list(range(sparse[0], len(backbone.layers))):
        reasons.append(f"MoE layers {sparse} are not a trailing run of the stack")
    elif sparse[0] > 0:
        # the first launch takes the previous layer's MLP output with its
        # residual add still pending: the kernel applies the add with its input
        # norm
        update = backbone.layers[sparse[0] - 1].ffn_boundary.declaration.update
        if not update.is_plain_add:
            reasons.append(
                f"layer {sparse[0] - 1} hands on a {type(update).__name__} residual "
                "update; the kernels apply a plain add"
            )
    par = get_parallel()
    if not (par.attn_tp_size == par.tp_size and par.moe_tp_size == par.tp_size):
        reasons.append(
            f"attention TP {par.attn_tp_size} / MoE TP {par.moe_tp_size} against TP "
            f"{par.tp_size}: the kernels shard attention and experts over TP alike"
        )
    if par.moe_ep_size != 1:
        reasons.append(f"MoE EP {par.moe_ep_size}: the kernels shard experts over TP")
    return reasons


# ------------------------------------------------------------------ build
class _Built:
    """One rank's built kernels and the forward that runs them."""

    def __init__(self, model, hf_config, ctx: BuildContext) -> None:
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.minimax_m3.gfx950.packed import Packed

        par = get_parallel()
        self.dims = Dims(tp=par.tp_size)
        self.lm, self.model = model, model.model
        self.sparse_ids = _sparse_ids(self.model.layers)
        self.first_layer = self.sparse_ids[0]
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

        # Only the widths this TP can actually run. DeviceLimits.widths is
        # declared per ISA, so the framework hands us the union; a step outside
        # this TP's range runs the model's normal forward below. Building the
        # out-of-range widths would JIT kernels that can never be reached --
        # minutes each.
        low, high = self.dims.width_range
        self.min_rows, self.max_rows = low, high
        widths = tuple(w for w in ctx.widths if low <= w <= high)
        if not widths:
            raise ValueError(
                f"TP {self.dims.tp} runs widths {low}..{high}; this server's steps "
                f"are {sorted(ctx.widths)}"
            )
        self.packed = Packed(model, hf_config, self.dims, ctx)
        self.kernels = {w: self.packed.build_kernel(w) for w in widths}
        logger.info(
            "minimax_m3 mono: TP %d, widths %s built of %s declared (this TP runs "
            "%d..%d); %d sparse layers (%d..%d)",
            self.dims.tp, list(widths), list(ctx.widths), low, high,
            len(self.sparse_ids), self.first_layer, len(self.model.layers) - 1,
        )  # fmt: skip

    # ------------------------------------------------------------- forward
    def forward(
        self,
        plan: Plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """``MiniMaxM3SparseForCausalLM.forward`` for a routed step."""
        lm = self.lm
        rows = plan.key.rows
        if not (self.min_rows <= rows <= self.max_rows) or rows not in self.kernels:
            # Outside this TP's width range. Going through the class reaches the
            # model's own forward: the runtime installed its dispatch on the
            # *instance*, so ``lm.forward`` would recurse.
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        hidden = self._backbone(plan, input_ids, positions, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            hidden, aux = hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``MiniMaxM3Model.forward``: the embedding, the leading dense layers on
        SGLang's path and the sparse ones as launches, then the final norm.

        Mirrors the model's own loop, including the expert-distribution
        recorder's per-layer context on the layers that still run SGLang's
        modules. The launches are deliberately not wrapped: they never enter
        SGLang's MoE modules, so there is nothing for the recorder to observe.
        The model swaps that context for a ``nullcontext`` under the piecewise
        prefill graph backend; this forward runs only on routed decode and
        verify steps, so that branch has no counterpart here.

        Pipeline parallelism and two-batch overlap are absent here because the
        spec does not list those features, so the framework refuses them before
        a server starts; the model's branches for them have no counterpart.
        """
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.layers.layer_boundary.residual.batch import (
            final_norm,
            start,
        )

        # The KV caches and the request table are reachable only from inside a
        # forward (SGLang publishes the attention backend per call), and they
        # are sized after the model loads, so they are bound on the first
        # routed step rather than at build. Idempotent, and before any launch.
        self.packed.bind()

        model = self.model
        hidden = (
            model.get_input_embeddings()(input_ids)
            if input_embeds is None
            else input_embeds
        )
        start(forward_batch)

        aux = AuxHiddenStateList()
        kernel = self.kernels[plan.key.rows]
        # Seeded from the graph key, which every rank holds alike, so a replay
        # derives the row tables it captured. The tables themselves are filled
        # once, by the step's first routed layer.
        memo: dict = {
            "query_len": plan.key.tokens_per_request,
            "is_verify": plan.key.forward_mode == "target_verify",
        }
        for i, layer in enumerate(model.layers):
            capture = aux if getattr(layer, "_is_layer_to_capture", False) else None
            if i >= self.first_layer:
                hidden = self._launch(
                    kernel, layer, i, positions, hidden, forward_batch, memo, capture
                )
                continue
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden = layer(
                    positions=positions,
                    forward_batch=forward_batch,
                    hidden_states=hidden,
                    captured_last_layer_outputs=capture,
                )
        hidden = final_norm(hidden, forward_batch, model.norm, skip_empty=True)
        return hidden if len(aux) == 0 else (hidden, aux)

    def _launch(
        self, kernel, layer, layer_id, positions, hidden, forward_batch, memo, capture
    ):
        """One sparse layer as a launch, in the decoder layer's own terms.

        Takes the previous layer's output with its residual add still pending
        (the kernel applies it with the input norm) and returns its own MLP
        output, already all-reduced, the same way.
        """
        import torch
        from sglang.srt.layers.layer_boundary.residual.batch import stream_of

        stream = stream_of(forward_batch)
        hidden, residual = stream.export(hidden)
        check = self.check and not torch.cuda.is_current_stream_capturing()
        ref = (
            self._reference(layer_id, positions, forward_batch, hidden, residual)
            if check
            else None
        )
        output, residual = kernel.launch(
            layer_id, positions, hidden, residual, forward_batch, memo
        )
        if ref is not None:
            from sgl_amd_mono.minimax.check import log_check
            from sgl_amd_mono.minimax_m3.gfx950.config import TOP_K

            live = forward_batch.out_cache_loc[: kernel.rows] > 0
            log_check(layer, layer_id, live, (output, residual), ref, TOP_K)
        if capture is not None:
            # the collector copies this reused kernel buffer
            capture.capture(kernel.capture_input())
        stream.write(residual)
        return stream.record(output, layer.ffn_boundary.declaration.update)

    def _reference(self, lid, positions, forward_batch, ar, residual):
        """Check mode: SGLang's layer ``lid`` on a copy of the launch's input pair.

        Runs on a residual stream of its own and returns (MLP output, residual) as the
        launch does. It writes the same KV slots, which the launch then writes again."""
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        layer = self.model.layers[lid]
        update = self.model.layers[lid - 1].ffn_boundary.declaration.update
        hidden, stream = ResidualStream.from_handoff(
            ar.clone(), residual.clone(), update
        )
        original = forward_batch.residual_stream
        forward_batch.residual_stream = stream
        try:
            out = layer(
                positions=positions, forward_batch=forward_batch, hidden_states=hidden
            )
            return stream.export(out)
        finally:
            forward_batch.residual_stream = original


def build(model, hf_config, ctx: BuildContext) -> _Built:
    return _Built(model, hf_config, ctx)


SPEC = MonoSpec(
    name="minimax_m3",
    devices={
        # widths is per ISA, so this is the UNION over TP 2 / 4 / 8. The usable
        # range is per TP (config.WIDTH_RANGE) and _Built routes the rest to the
        # model's forward -- see the class comment there.
        "gfx950": DeviceLimits(
            widths=WIDTHS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS
        ),
    },
    features=frozenset(
        {
            # one linear draft chain; config_disable_reasons refuses the rest,
            # and RAGGED_VERIFY stays off: a verify step's rows are even here
            Feature.SPECULATIVE_DECODING,
            # HiCache, with the layer-first kernel backend that keeps the
            # page16 SHUFFLE K / V (config_disable_reasons refuses the others)
            Feature.KV_OFFLOAD,
            # ``_launch`` hands the collector K1's residual after the input add,
            # the hidden state SGLang's attention boundary hands EAGLE3 -- the
            # same tensor #8 captured. test/e2e/test_minimax_m3_server.py's
            # verify test serves an EAGLE3 draft through it.
            Feature.EAGLE3_HIDDEN_STATES,
        }
    ),
    requires=Requirements(
        # The attention weights the kernels quantize to private FP8 copies are
        # read as bf16 (``weights.py:68-71``), which refuses anything else --
        # but from inside ``build``, as a bare ValueError after the model has
        # loaded. Declaring it here moves that refusal to startup validation,
        # where the framework collects every rank's reasons and reports them
        # together.
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            # the kernels index 128-token pages and write fp8_e4m3 K / V
            "page_size": frozenset({SPARSE_BLOCK}),
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # they run the shared expert as the fused expert of the MXFP4 MoE
            "disable_shared_experts_fusion": frozenset({False}),
            "speculative_algorithm": frozenset({None, "EAGLE3"}),
            # HiSparse maps the logical KV slots to device rows; the kernels write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # the kernels select blocks in every layer, so there is no indexer
            # pass to skip: FREQ buckets the sparse layers and runs the indexer
            # only in the first of each bucket
            "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ": 1,
            "SGLANG_USE_AITER": True,  # they read aiter's MXFP4 MoE layout
            # the index-K cache the kernels score is FP8
            "SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE": True,
            # benchmark routing overrides: the kernels route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
            # #41998's page16 SHUFFLE K / V switches are checked by name (PENDING_ENV)
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=model_disable_reasons,
    config_disable_reasons=config_disable_reasons,
    build=build,
    **(
        {}
        if MONOKERNEL_API_VERSION is None
        else {"api_version": MONOKERNEL_API_VERSION}
    ),
)
