# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash on gfx950: the FFN monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed decode or DSpark
verify step runs ``_Built.forward``, which takes and returns what the model's forward
does. The embedding, the dense layers and each sparse layer's input norm and attention
run SGLang's modules; the rest of every sparse layer is one launch (``ffn.py``), from
the attention output (o_proj on SGLang's rowwise-FP8 copy of its weight) to the
reduced MoE output, whose residual add the next layer's input norm applies. SGLang's
final norm and logits processor close the step.

The launch reads SGLang's post-load MXFP4 routed experts in place, and the shared
expert in the checkpoint's own MXFP8: the load hook keeps this rank's shard, and the
build lays it out for the launch (``a16w8``). ``SGLANG_AMD_MONO_CHECK=1`` runs SGLang's
layer on a copy of every sparse layer's input in eager steps and logs both outputs'
differences on the live rows; the bounds in ``NUMERICS`` come from these runs.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager

import torch

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

from sgl_amd_mono.minimax_m31 import ffn

logger = logging.getLogger(__name__)

TP = 4
WIDTHS = tuple(range(1, ffn.MAX_TOKENS + 1))
BLOCKS = 256

# The SGLang names this glue reads, by file under the sglang package
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/models/minimax_m3.py": (
        "is_layer_sparse",
        '"_is_layer_to_capture"',
        "_norm_quant",
        "attn_boundary",
        "input_layernorm",
        "_fused_qkv_index",
        "_can_use_rocm_sparse_qk_index_norm_rope_static",
        "q_norm",
        "k_norm",
        "index_q_norm",
        "index_k_norm",
        "variance_epsilon",
        "rotary_emb",
        "_get_sparse_kv_pool",
        "_mark_sparse_kv_cached_by_fusion",
        "forward_core",
        "self_attn",
        "mlp",
        "experts",
        "gate",
        "e_score_correction_bias",
        "post_attention_layernorm",
        "layer_id",
        "o_proj",
        "disable_index_value",
        "embed_tokens",
        "norm",
        "layers",
        "end_layer",
        "layers_to_capture",
        "finalize_norm",
    ),
    "srt/models/minimax_common/amd/minimax_m3_hip.py": (
        "attn_quant_format",
        "nvfp4_layer_caches",
    ),
    "srt/mem_cache/minimax_m3_nvfp4_pool.py": ("MiniMaxM3Nvfp4KVPool",),
    "srt/model_executor/forward_context.py": ("get_forward_context", "attn_backend"),
    "srt/layers/attention/minimax_sparse_backend.py": (
        "sparse",
        "_linear_verify_meta",
        "kv_is_nvfp4",
        "speculative_num_draft_tokens",
        "use_dense_sparse_decode",
        "_use_msa_decode",
        "hisparse_coordinator",
        "indexer_cp",
        "fp8_attn_gemm",
        "index_cache_enabled",
        "score_type",
        "block_size_k",
        "page_size",
        "topk_blocks",
        "init_blocks",
        "local_blocks",
        "req_to_token",
        "_max_seqlen_k",
        "seq_lens",
        "req_pool_indices",
    ),
    "srt/layers/attention/minimax_sparse_ops/minimax_sparse.py": (
        "minimax_sparse_decode",
    ),
    "srt/layers/radix_attention.py": (
        "q_scale_float",
        "k_scale_float",
        "v_scale_float",
        "idx_q_scale_float",
        "idx_k_scale_float",
        "idx_v_scale_float",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache", "is_neox_style"),
    "srt/distributed/parallel_state.py": ("get_tp_group", "rank_in_group", "cpu_group"),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/environ.py": ("envs",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "stream_of",
        "final_norm",
        "final_norm(capture)",
        "final_norm(finalize_norm)",
    ),
    "srt/layers/layer_boundary/residual/stream.py": (
        "ResidualStream",
        "write",
        "record",
        "residual",
        "snapshot",
        "from_handoff",
        "export",
    ),
    "srt/layers/layer_boundary/residual/add_norm.py": ("PLAIN_ADD",),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList", "capture"),
    "srt/layers/quantization/fp8.py": ("_ptpc_weight", "_ptpc_scale"),
    "srt/models/minimax_m3_vl.py": (
        "logits_processor",
        "lm_head",
        "forward(get_embedding)",
    ),
}

# The checkpoint names of the shared experts' MXFP8 tensors
_CKPT_SHARED = re.compile(
    r"(?:language_model\.)?model\.layers\.(?P<i>\d+)\.block_sparse_moe\.shared_experts\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)"
)


def _text(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def is_m31(hf_config) -> bool:
    """MiniMax-M3.1-Flash's checkpoint: NVFP4 experts next to MXFP8 dense layers."""
    q = getattr(hf_config, "quantization_config", None) or getattr(
        _text(hf_config), "quantization_config", None
    )
    return isinstance(q, dict) and str(q.get("moe_quant_algo", "")).upper() == "NVFP4"


def config_disable_reasons(cfg, hf_config) -> list[str]:
    """What the shared MiniMax spec's requirements cannot state for M3.1."""
    import importlib.util

    from sglang.srt.environ import envs

    t = _text(hf_config)
    reasons = []
    for key, want in (
        ("hidden_size", ffn.HIDDEN),
        ("num_local_experts", ffn.EXPERTS),
        ("num_experts_per_tok", ffn.TOPK),
        ("routed_scaling_factor", ffn.ROUTED_SCALE),
        ("scoring_func", "sigmoid"),
        ("hidden_act", "swigluoai"),
    ):
        if getattr(t, key, None) != want:
            reasons.append(
                f"config {key}={getattr(t, key, None)!r}, built for {want!r}"
            )
    if getattr(t, "intermediate_size", None) != TP * ffn.INTER:
        reasons.append(f"intermediate_size {getattr(t, 'intermediate_size', None)}")
    if cfg.speculative_algorithm not in (None, "DSPARK"):
        reasons.append(
            f"speculative algorithm {cfg.speculative_algorithm}: needs DSPARK or none"
        )
    if cfg.moe_runner_backend not in ("auto", "aiter"):
        reasons.append(f"--moe-runner-backend {cfg.moe_runner_backend}: needs aiter")
    if getattr(cfg, "enable_hisparse", False):
        reasons.append("HiSparse")
    if getattr(cfg, "expert_distribution_recorder_mode", None) is not None:
        reasons.append("the launches record no expert distribution")
    for name in (
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ):
        var = getattr(envs, name, None)
        if var is not None and var.get():
            reasons.append(f"{name}: the launches route with the model's own router")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each sparse layer's MXFP8 shared expert, by its
    checkpoint name (``...layers.{i}.block_sparse_moe.shared_experts.{gate,up,down}_proj
    .{weight,weight_scale_inv}``): the rows of gate / up, the columns of down."""
    from sglang.srt.runtime_context import get_parallel

    m = _CKPT_SHARED.fullmatch(name)
    if m is None:
        return
    p = get_parallel()
    dim = 1 if m["proj"] == "down" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    copies["shared expert mxfp8", int(m["i"]), m["proj"], m["kind"]] = (
        shard.contiguous()
    )


def _sparse_ids(model) -> list[int]:
    return [i for i, layer in enumerate(model.model.layers) if layer.is_layer_sparse]


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    reasons = []
    sparse = _sparse_ids(model)
    if not sparse or sparse != list(range(sparse[0], len(model.model.layers))):
        return [f"sparse layers {sparse}: the forward needs them after the dense ones"]
    for i in sparse:
        layer = model.model.layers[i]
        experts = layer.mlp.experts
        w13, w2 = (
            getattr(experts, "w13_weight", None),
            getattr(experts, "w2_weight", None),
        )
        if (
            w13 is None
            or w13.dtype != torch.uint8
            or tuple(w13.shape) != (ffn.EXPERTS, 2 * ffn.INTER, ffn.HIDDEN // 2)
            or not getattr(w13, "is_shuffled", False)
            or tuple(w2.shape) != (ffn.EXPERTS, ffn.HIDDEN, ffn.INTER // 2)
            or not getattr(w2, "is_shuffled", False)
        ):
            reasons.append(f"layer {i}: routed experts are not SGLang's shuffled MXFP4")
            break
        if tuple(layer.mlp.gate.weight.shape) != (ffn.EXPERTS, ffn.HIDDEN):
            reasons.append(f"layer {i}: router {tuple(layer.mlp.gate.weight.shape)}")
            break
        if layer.mlp.e_score_correction_bias is None:
            reasons.append(f"layer {i}: no routing bias")
            break
        if not getattr(layer.self_attn, "disable_index_value", False):
            reasons.append(f"layer {i}: index values add to o_proj's output")
            break
        wo, wos = _o_proj_rowwise(layer)
        if (
            wo is None
            or wo.dtype != torch.float8_e4m3fn
            or tuple(wo.shape) != (ffn.HIDDEN, ffn.ATTN_K)
            or wos is None
            or tuple(wos.shape) != (ffn.HIDDEN, 1)
        ):
            reasons.append(
                f"layer {i}: o_proj has no rowwise-FP8 copy (MXFP8 on aiter)"
            )
            break
        missing = [
            (proj, kind)
            for proj in ("gate", "up", "down")
            for kind in ("weight", "weight_scale_inv")
            if ("shared expert mxfp8", i, proj, kind) not in copies
        ]
        if missing:
            reasons.append(f"layer {i}: no checkpoint shared expert {missing}")
            break
    return reasons


def _o_proj_rowwise(layer):
    """The rowwise-FP8 o_proj SGLang keeps for decode-sized steps on gfx950: the
    (16, 16)-shuffled weight and its per-row scale."""
    w = layer.self_attn.o_proj.weight
    return getattr(w, "_ptpc_weight", None), getattr(w, "_ptpc_scale", None)


@contextmanager
def _o_proj_input(o_proj):
    """``o_proj`` returns its input while the attention runs: the launch applies it."""
    o_proj.forward = lambda x: (x, None)
    try:
        yield
    finally:
        del o_proj.forward


def _shared_mxfp8(copies: dict, i: int):
    """Layer i's shared expert in the launch's MXFP8 layout (``a16w8.layout``), from the
    checkpoint values the load hook kept: (w13, s13, w2, s2), gate / up separated."""
    from sgl_amd_mono.minimax_m31 import a16w8

    def take(proj):
        return (
            copies.pop(("shared expert mxfp8", i, proj, "weight")),
            copies.pop(("shared expert mxfp8", i, proj, "weight_scale_inv")),
        )

    (gw, gs), (uw, us), (dw, ds) = take("gate"), take("up"), take("down")
    return (
        *a16w8.layout(torch.cat([gw, uw]), torch.cat([gs, us])),
        *a16w8.layout(dw, ds),
    )


def front_disable_reasons(layer) -> list[str]:
    """Why a sparse layer's attention front cannot run as one launch (``front.py``):
    it replaces the input norm, the fused qkv + index projection and SGLang's fused
    qk-norm / RoPE / NVFP4 store, whose static gate it inherits."""
    from sgl_amd_mono.minimax_m31 import front

    attn, norm = layer.self_attn, layer.input_layernorm
    proj = getattr(attn, "_fused_qkv_index", None)
    w = getattr(getattr(proj, "weight", None), "_ptpc_weight", None)
    ws = getattr(getattr(proj, "weight", None), "_ptpc_scale", None)
    if (
        w is None
        or w.dtype != torch.float8_e4m3fn
        or tuple(w.shape) != (front.COLS, ffn.HIDDEN)
        or ws is None
        or ws.numel() != front.COLS
    ):
        return ["no rowwise-FP8 copy of the fused qkv + index projection"]
    if type(norm).__name__ != "GemmaRMSNorm" or norm.variance_epsilon != ffn.EPS:
        return [f"input norm {type(norm).__name__}"]
    if not getattr(attn, "_can_use_rocm_sparse_qk_index_norm_rope_static", False):
        return ["SGLang's fused qk-norm / RoPE store does not apply"]
    if attn.q_norm.variance_epsilon != ffn.EPS or not attn.rotary_emb.is_neox_style:
        return ["head norm eps or RoPE style"]
    if (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads) != (
        front.Q_HEADS,
        1,
        1,
    ) or (attn.head_dim, attn.rotary_dim) != (front.HEAD, 64):
        return ["attention head layout"]
    scales = (
        attn.attn.q_scale_float,
        attn.attn.k_scale_float,
        attn.attn.v_scale_float,
        attn.attn.idx_q_scale_float,
        attn.attn.idx_k_scale_float,
    )
    if any(v not in (None, 1.0) for v in scales):
        return [
            f"Q / K / V / index Q / index K scales {scales}: the launches take unit ones"
        ]
    return []


def _front_weights(layer):
    from types import SimpleNamespace

    attn = layer.self_attn
    w = attn._fused_qkv_index.weight
    return SimpleNamespace(
        input_norm=layer.input_layernorm.weight.data,
        wqkv=w._ptpc_weight,
        wqkv_scale=w._ptpc_scale.reshape(-1).contiguous(),
        q_norm=attn.q_norm.weight.data,
        k_norm=attn.k_norm.weight.data,
        iq_norm=attn.index_q_norm.weight.data,
        ik_norm=attn.index_k_norm.weight.data,
    )


class _Built:
    """One rank's FFN launcher, every sparse layer's weights, and the forward."""

    def __init__(
        self,
        lm,
        runner,
        peer,
        weights: dict,
        sparse: list[int],
        front=None,
        index=None,
        attend=None,
    ):
        self.lm, self.model = lm, lm.model
        self.runner, self.peer, self.weights = runner, peer, weights
        self.first, self.sparse = sparse[0], sparse
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"
        # (FrontRunner, {layer: weights}) for the sparse layers past the first
        self.front = front
        self._front_caches: dict[int, tuple] = {}
        # the verify indexer launch (IndexRunner) under the front, or None
        self.index = index
        # the verify sparse attention launch (SparseRunner) under the indexer, or None
        self.attend = attend

    def _front_pool(self, layer, lid):
        """Layer lid's write views of SGLang's NVFP4 pool, or None off it."""
        caches = self._front_caches.get(lid)
        if caches is None:
            from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
            from sglang.srt.models.minimax_common.amd.minimax_m3_hip import (
                nvfp4_layer_caches,
            )

            pool = layer.self_attn._get_sparse_kv_pool()
            if not isinstance(pool, MiniMaxM3Nvfp4KVPool):
                logger.info("M3.1 attention front off: the KV pool is %s", type(pool))
                self.front = None
                return None
            caches = nvfp4_layer_caches(pool, layer.self_attn.attn.layer_id)
            self._front_caches[lid] = caches
        return caches

    def _front(self, layer, lid, out, res_out, positions, forward_batch, aux):
        """Layer lid's attention from the previous FFN launch's (output, residual):
        the front launch, then SGLang's sparse attention on its Q and index Q, the
        pool rows already stored. Returns (attention output, residual entering)."""
        runner, weights = self.front
        attn = layer.self_attn
        caches = self._front_caches[lid]
        res_new, q, iq = runner(
            out,
            res_out,
            weights[lid],
            positions,
            forward_batch.out_cache_loc,
            caches,
            attn.rotary_emb.cos_sin_cache,
        )
        if getattr(layer, "_is_layer_to_capture", False):
            aux.capture(res_new)
        sparse = self._verify_backend(forward_batch, q.shape[0])
        if sparse is not None:
            return self._attend(layer, q, iq, caches, sparse), res_new
        attn._mark_sparse_kv_cached_by_fusion(forward_batch, attn.attn.layer_id)
        # K, V and index K: the attention reads the pool rows, not these
        unused = q[:, : attn.head_dim]
        inner = (q, unused, unused, iq, unused, None, forward_batch)
        with _o_proj_input(attn.o_proj):
            out_attn = attn.forward_core((None, forward_batch, inner))
        return out_attn, res_new

    def _verify_backend(self, forward_batch, rows):
        """SGLang's sparse attention backend when this step's top-k runs as the
        plugin's indexer launch (``index.py``): a linear DSpark verify of 8-row
        requests whose top-k SGLang computes per layer with the max score."""
        if self.index is None or not forward_batch.forward_mode.is_target_verify():
            return None
        from sglang.srt.model_executor.forward_context import get_forward_context

        backend = get_forward_context().attn_backend
        sparse = getattr(backend, "sparse", backend)
        meta = getattr(sparse, "_linear_verify_meta", None)
        if (
            meta is None
            or not getattr(sparse, "kv_is_nvfp4", False)
            or sparse.speculative_num_draft_tokens != 8
            or rows % 8
            or not 1 <= rows // 8 <= 2
            or meta.seq_lens.shape[0] != rows
            or sparse.use_dense_sparse_decode
            or sparse._use_msa_decode
            or sparse.hisparse_coordinator is not None
            or sparse.indexer_cp is not None
            or sparse.fp8_attn_gemm
            or sparse.index_cache_enabled
            or sparse.score_type != "max"
            or (sparse.block_size_k, sparse.page_size, sparse.topk_blocks)
            != (128, 128, 16)
            or (sparse.init_blocks, sparse.local_blocks) != (0, 1)
        ):
            return None
        return sparse

    def _attend(self, layer, q, iq, caches, sparse):
        """Layer's sparse attention on the step's Q and index Q: the indexer launch's
        top-k, then the sparse attention launch, or SGLang's NVFP4 sparse decode on it
        (``cached_topk_idx`` skips SGLang's own scoring and top-k)."""
        from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
            minimax_sparse_decode,
        )

        a = layer.self_attn.attn
        meta = sparse._linear_verify_meta
        rows = q.shape[0]
        k, v, ik = caches
        topk = self.index(
            iq, ik, sparse.req_to_token, meta.seq_lens, meta.req_pool_indices
        )
        if self.attend is not None:
            return self.attend(
                q, k, v, sparse.req_to_token, meta.seq_lens, meta.req_pool_indices, topk
            )
        _, o = minimax_sparse_decode(
            q.view(rows, -1, 128),
            None,
            k,
            v,
            iq.view(rows, 1, 128),
            None,
            ik,
            None,
            sparse.req_to_token,
            meta.req_pool_indices,
            meta.seq_lens,
            sparse._max_seqlen_k,
            1,
            sparse.block_size_k,
            sparse.topk_blocks,
            sparse.init_blocks,
            sparse.local_blocks,
            score_type=sparse.score_type,
            disable_index_value=True,
            page_size=sparse.page_size,
            q_scale=a.q_scale_float,
            k_scale=a.k_scale_float,
            v_scale=a.v_scale_float,
            idx_q_scale=a.idx_q_scale_float,
            idx_k_scale=a.idx_k_scale_float,
            idx_v_scale=a.idx_v_scale_float,
            cached_topk_idx=topk,
            packed_queries=sparse.speculative_num_draft_tokens,
        )
        return o.reshape(rows, -1)

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        lm = self.lm
        hidden, aux = self._backbone(plan, input_ids, positions, forward_batch)
        if kwargs.get("get_embedding", False):
            return hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, plan, input_ids, positions, forward_batch):
        """``MiniMaxM3Model.forward`` with each sparse layer's post-attention work as
        one launch."""
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
        from sglang.srt.layers.layer_boundary.residual.batch import (
            final_norm,
            start,
            stream_of,
        )

        m, rows = self.model, plan.key.rows
        hidden = m.embed_tokens(input_ids)
        start(forward_batch)
        aux = AuxHiddenStateList()
        for i in range(self.first):
            layer = m.layers[i]
            hidden = layer(
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden,
                captured_last_layer_outputs=aux
                if getattr(layer, "_is_layer_to_capture", False)
                else None,  # fmt: skip
            )
        slots = forward_batch.out_cache_loc
        stream = stream_of(forward_batch)
        check = self.check and not torch.cuda.is_current_stream_capturing()
        # the previous sparse layer's (output, residual), while the stream holds
        # neither: the front launch applies the residual add
        held = None
        for lid in self.sparse:
            layer = m.layers[lid]
            front = (
                held is not None
                and not check
                and self.front is not None
                and self._front_pool(layer, lid) is not None
            )
            if front:
                attn, entering = self._front(
                    layer, lid, *held, positions, forward_batch, aux
                )
            else:
                if held is not None:
                    stream.write(held[1])
                    hidden = stream.record(held[0], PLAIN_ADD)
                ref = self._reference(layer, positions, forward_batch, hidden) if check else None  # fmt: skip
                call = {}
                if getattr(layer, "_norm_quant", None) is not None:
                    call["quant_format"] = layer._norm_quant.attn_quant_format(rows)
                x = layer.attn_boundary.prepare(
                    hidden,
                    forward_batch,
                    capture_gathered=aux
                    if getattr(layer, "_is_layer_to_capture", False)
                    else None,  # fmt: skip
                    **call,
                )
                with _o_proj_input(layer.self_attn.o_proj):
                    attn = layer.self_attn(
                        positions=positions,
                        hidden_states=x,
                        forward_batch=forward_batch,
                    )
                entering = stream.residual
            out, residual = self.runner(
                attn.contiguous(),
                entering.contiguous(),
                self.weights[lid],
                slots,
            )
            if not front and ref is not None:
                from sgl_amd_mono.minimax.check import log_check

                log_check(layer, lid, slots[:rows] > 0, (out, residual), ref, ffn.TOPK)
            held = (out, residual)
        # the final norm applies the last residual add
        stream.write(held[1])
        hidden = stream.record(held[0], PLAIN_ADD)
        hidden = final_norm(
            hidden,
            forward_batch,
            m.norm,
            # the DSpark draft reads the last layer's updated residual
            capture=aux.capture if m.end_layer in m.layers_to_capture else None,
            finalize_norm=m.finalize_norm,
            skip_empty=True,
        )
        return hidden, (aux if len(aux) else None)

    def _reference(self, layer, positions, forward_batch, hidden):
        """Check mode: SGLang's layer on a copy of the residual entering it, on a
        stream of its own; returns (MLP output, residual after attention) as the launch
        does. It writes the same KV slots, which the launch's layer writes again."""
        from sglang.srt.layers.layer_boundary.residual.batch import stream_of
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        entering = stream_of(forward_batch).snapshot(hidden)
        hidden, stream = ResidualStream.from_handoff(entering, None, None)
        original = forward_batch.residual_stream
        forward_batch.residual_stream = stream
        try:
            out = layer(
                positions=positions, forward_batch=forward_batch, hidden_states=hidden
            )
            return stream.export(out)
        finally:
            forward_batch.residual_stream = original


def build(model, hf_config, ctx):
    from types import SimpleNamespace

    from sglang.srt.distributed import get_tp_group

    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.minimax_m31.runner import FfnRunner

    group = get_tp_group()
    rank, device = (
        group.rank_in_group,
        torch.device("cuda", torch.cuda.current_device()),
    )
    peer = PeerBuffer(ffn.peer_bytes(TP), group.cpu_group, rank, TP, device)
    runner = FfnRunner(TP, rank, peer.local, peer.addresses, device, tuple(ctx.widths))
    runner.set_deadline(ctx.deadline_s)
    sparse = _sparse_ids(model)
    weights = {}
    for i in sparse:
        layer = model.model.layers[i]
        moe, experts = layer.mlp, layer.mlp.experts
        w13s, s13s, w2s, s2s = _shared_mxfp8(ctx.copies, i)
        wo, wos = _o_proj_rowwise(layer)
        weights[i] = SimpleNamespace(
            post_norm=layer.post_attention_layernorm.weight.data,
            router=moe.gate.weight.data.float().contiguous(),
            router_bias=moe.e_score_correction_bias.data.float().contiguous(),
            w13=experts.w13_weight.data.view(torch.float4_e2m1fn_x2),
            s13=experts.w13_weight_scale.data,
            w2=experts.w2_weight.data.view(torch.float4_e2m1fn_x2),
            s2=experts.w2_weight_scale.data,
            w13s=w13s,
            s13s=s13s,
            w2s=w2s,
            s2s=s2s,
            wo=wo,
            wos=wos.contiguous(),
        )
    # every width compiles alike; its residency is checked against the grid
    runner.warm(
        weights[sparse[0]],
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    logger.info(
        "MiniMax-M3.1 FFN monokernels: %d sparse layers, rows %s",
        len(sparse),
        tuple(ctx.widths),
    )
    front = None
    if os.environ.get("SGLANG_AMD_MONO_M31_FRONT", "1") == "1":
        from sgl_amd_mono.minimax_m31.runner import FrontRunner

        reasons = {i: front_disable_reasons(model.model.layers[i]) for i in sparse[1:]}
        off = {i: r for i, r in reasons.items() if r}
        if off:
            logger.info("M3.1 attention front off: %s", next(iter(off.items())))
        else:
            front = (
                FrontRunner(device, tuple(ctx.widths)),
                {i: _front_weights(model.model.layers[i]) for i in sparse[1:]},
            )
            logger.info(
                "M3.1 attention front launch: layers %d .. %d", sparse[1], sparse[-1]
            )
    index = None
    if front is not None and os.environ.get("SGLANG_AMD_MONO_M31_INDEX", "1") == "1":
        from sgl_amd_mono.minimax_m31.runner import IndexRunner

        index = IndexRunner(device)
        logger.info("M3.1 verify indexer launch under the front")
    attend = None
    if index is not None and os.environ.get("SGLANG_AMD_MONO_M31_SPARSE", "1") == "1":
        from sgl_amd_mono.minimax_m31.runner import SparseRunner

        attend = SparseRunner(device)
        logger.info("M3.1 verify sparse attention launch under the indexer")
    return _Built(model, runner, peer, weights, sparse, front, index, attend)


# check mode, 2 runs of 16 GSM8K prompts (49,476 and 44,004 launches); worst case:
# residual, every row: rel 0.042, cosine 0.99831
# MLP output, rows routed alike (94%): rel 0.28, cosine 0.66
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.residual.rel": 0.08,
        "layer.residual.cosine": 0.995,
        "layer.mlp_out.rel": 0.5,
        "layer.mlp_out.cosine": 0.5,
    },
)

SPEC = MonoSpec(
    name="minimax_m31_ffn_hip",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(TP,), resident_blocks=BLOCKS)
    },
    # each row's FFN is independent and reads no KV: any verify layout, any KV tier
    features=frozenset(
        {
            Feature.SPECULATIVE_DECODING,
            Feature.RAGGED_VERIFY,
            Feature.EAGLE3_HIDDEN_STATES,
            Feature.KV_OFFLOAD,
        }
    ),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}), env={"SGLANG_USE_AITER": True}
    ),
    numerics=NUMERICS,
    model_disable_reasons=model_disable_reasons,
    build=build,
    api_version=1,  # the spec module's MONOKERNEL_API_VERSION
    config_disable_reasons=config_disable_reasons,
    load_tensor=load_tensor,
)
