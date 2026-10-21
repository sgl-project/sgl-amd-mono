# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

Every covered layer is one launch for the whole decoder layer (``gfx950/kernel.py``):
the input norm, q_a | kv_a, the q and kv norms, the cache writes, q_b, the indexer
(on the layers that run it) and its exact top-2048, the sparse MLA, o_proj and its
TP reduce, the post-attention norm, the router, the routed and shared FP8 experts
and the MoE TP reduce, adding both outputs to the hidden stream.

The forward is ``DeepseekV2ForCausalLM.forward`` for a routed decode step: SGLang's
embedding, dense layers 0 to 2, final norm and logits processor around the 75 MoE
layers' launches. The first covered layer takes the stream layer 2 wrote and,
being a layer that reuses the previous top-k, layer 2's indices. A pad row
(SGLang's padding slot 0, or a width's row past the step) writes no cache and
attends no key.

The kernels read SGLang's weights in place, except the copies logged at startup:
W_UK and W_UV packed from SGLang's per-tensor FP8 w_kc and w_vc, the router and the
indexer's weights_proj packed for the MFMA tiles, the indexer's k LayerNorm in f32,
and one bf16 rotary table.
"""

from __future__ import annotations

import importlib.util
import os

import msgspec
import torch
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.glm53 import HF_CONFIG

# A8W8 padding buckets. Native FP4 uses exact widths through 12; both formats
# retain the 16-row cohort bucket for 13-16 logical tokens.
WIDTHS = (2, 4, 6, 8, 12, 16)
ROWS = tuple(range(1, max(WIDTHS) + 1))
TP_SIZES = (4, 8)
# TP4 uses the bounded row/cohort schedule through sixteen total query tokens.
# Retain the existing TP8 support range; this port is validated at TP4.
TP_MAX_ROWS = {4: 16, 8: 8}
# SGLang's DSA cache on ROCm: 64-token pages
PAGE_SIZE = 64
TOPK = 2048


def _width(rows: int, tp: int = 4, expert_mxfp4: bool = False) -> int:
    if expert_mxfp4 and tp == 4 and rows <= 12:
        return rows
    widths = (2, 4, 8) if tp == 8 else WIDTHS
    return next(w for w in widths if w >= rows)


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "page_size",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
        "speculative_eagle_topk",
    ),
    "srt/distributed/parallel_state.py": ("cpu_group",),
    "srt/environ.py": (
        "SGLANG_DSA_FUSE_TOPK",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/layers/attention/dsa/dsa_indexer.py": ("use_gfx950_fused_indexer",),
    "srt/layers/attention/dsa_backend.py": ("forward_metadata", "real_page_table"),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStatePacker", "capture", "finalize"),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("maybe_input_scattered",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "fold",
        "take_output",
        "set_written",
        "complete_output",
        "final_norm",
    ),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_index_k_with_scale_buffer"),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "seq_lens",
        "forward_mode",
        "is_idle",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/models/deepseek_v2.py": (
        "model",
        "embed_tokens",
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "self_attn",
        "input_layernorm",
        "indexer",
        "gemm_output_zero_allocator_size",
        "next_full_attention_layer_id",
        "layers_to_capture",
        "logits_processor",
        "lm_head",
        "forward(prev_topk_indices)",
        "forward(captured_last_layer_outputs)",
    ),
    "srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "process_model_config",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("BumpAllocator",),
}

# Each covered layer against SGLang's normal layer on the same input and caches
# (SGLANG_AMD_MONO_CHECK): the stream it writes and the cache rows.
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.streams.cosine": 0.95,
        "layer.kv.cosine": 0.995,
        "layer.index.cosine": 0.999,
    },
)


def _covered_layers(config) -> list[int]:
    """The MoE layers."""
    return list(range(config.first_k_dense_replace, config.num_hidden_layers))


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = [
        f"needs GLM-5.3's {key} = {want!r}, got {getattr(hf_config, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(hf_config, key, None) != want
    ]
    if getattr(hf_config, "moe_layer_freq", 1) != 1:
        reasons.append("needs every layer after the dense ones to be MoE")
    if (getattr(cfg, "speculative_eagle_topk", None) or 1) != 1:
        reasons.append("GLM mono-kernel needs a linear speculative chain (topk=1)")
    # SGLang scales the softmax by YaRN's mscale; the kernel's scale is fixed
    rope = (
        getattr(hf_config, "rope_parameters", None)
        or getattr(hf_config, "rope_scaling", None)
        or {}
    )
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default":
        reasons.append(f"RoPE {rope_type!r}: the kernel's softmax scale is unscaled")
    if cfg.page_size not in (None, PAGE_SIZE):
        reasons.append(f"--page-size {cfg.page_size}: the DSA caches need {PAGE_SIZE}")
    quant = getattr(hf_config, "quantization_config", None) or {}
    block_fp8 = quant.get("quant_method") == "fp8" and list(
        quant.get("weight_block_size") or ()
    ) == [128, 128]
    # Quark supports per-layer formats. Check actual loaded projections and
    # expert tensors below; its name alone does not imply a compatible layout.
    if not block_fp8 and quant.get("quant_method") != "quark":
        reasons.append(
            "needs FP8 block-128 or Quark with block-FP8 attention, got "
            f"{quant.get('quant_method')!r} {quant.get('weight_block_size')!r}"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _where(ids: list) -> str:
    return f"layer {ids[0]}" if len(ids) == 1 else f"{len(ids)} layers from {ids[0]}"


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sgl_amd_mono.glm53.gfx950.weights import (
        experts_reason,
        projection_format_reasons,
        unshuffled_projections,
    )

    reasons = []
    covered = _covered_layers(hf_config)
    row_major, unrotated = {}, []
    for i in covered:
        layer = model.model.layers[i]
        why = experts_reason(layer.mlp)
        if why is not None:
            reasons.append(f"layer {i}: {why}")
        reasons.extend(f"layer {i}: {why}" for why in projection_format_reasons(layer))
        for name in unshuffled_projections(layer):
            row_major.setdefault(name, []).append(i)
        indexer = layer.self_attn.indexer
        if indexer is not None and not indexer.use_gfx950_fused_indexer:
            unrotated.append(i)
    reasons += [
        f"{name} of {_where(ids)}: needs AITER's (16, 16) FP8 layout, which SGLang "
        "applies for its AITER block-FP8 GEMM"
        for name, ids in row_major.items()
    ]
    if unrotated:
        reasons.append(
            f"the indexer of {_where(unrotated)}: needs SGLang's gfx950 fused DSA "
            "indexer, which caches rotated index keys"
        )
    first = model.model.layers[covered[0]]
    prev = model.model.layers[covered[0] - 1]
    if first.self_attn.indexer is None and prev.self_attn.indexer is None:
        reasons.append(
            f"layer {covered[0]} reuses the top-k of layer {covered[0] - 1}, which "
            "does not compute it"
        )
    return reasons


class _Layer:
    """One covered layer: its module, weights and per-width ops; its caches bind at
    its first launch and rebind if they move."""

    def __init__(self, i: int, module, weights):
        self.i, self.module, self.w = i, module, weights
        self.ops: dict = {}  # width -> Glm53LayerOp
        self.caches = None  # Glm53Caches


class _Built:
    """One rank's monokernels: the runtimes per (width, kind), each covered layer's
    ops, the step's rows per width, and the forward."""

    def __init__(self, lm, layers: dict, runtimes: dict, cos, sin, tp: int, device):
        from sgl_amd_mono.glm53.gfx950.config import HIDDEN
        from sgl_amd_mono.glm53.gfx950.runner import Glm53Step

        self.lm, self.model = lm, lm.model
        self._layers, self.runtimes = layers, runtimes
        self.first, self.last = min(layers), max(layers)
        self.tp = tp
        self.moe_activation_dtype = (
            "fp4"
            if getattr(getattr(layers[self.first], "w", None), "expert_mxfp4", False)
            else "fp8"
        )
        self.max_rows = TP_MAX_ROWS.get(tp, max(WIDTHS))
        self.check_layers = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        self.cos, self.sin = cos, sin
        self.topk_cols = torch.arange(TOPK, dtype=torch.int64, device=device)
        self.steps, self.pad_h = {}, {}
        i32, i64 = torch.int32, torch.int64
        for w in {w for w, _ in runtimes}:
            self.steps[w] = Glm53Step(
                positions=torch.zeros(w, dtype=i64, device=device),
                slots=torch.full((w,), -1, dtype=i64, device=device),
                indptr=torch.zeros(w + 1, dtype=i32, device=device),
                req_ids=torch.zeros(w, dtype=i32, device=device),
                page_table=torch.zeros(w, 1, dtype=i32, device=device),
                # the last entry takes the copies of a row's unused keys
                topk=torch.zeros(w * TOPK + 1, dtype=i32, device=device),
            )
            self.pad_h[w] = torch.zeros(w, HIDDEN, dtype=torch.bfloat16, device=device)

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        *,
        indexer_cp: bool = False,
        moe_activation_dtype: str | None = None,
    ):
        """Run a decode step with an explicit, capture-time CP choice.

        Both variants are built at initialization. A captured graph retains its
        choice; callers selecting both modes must keep distinct graph instances.
        All TP ranks must pass the same Python bool.
        ``moe_activation_dtype=None`` follows the loaded expert weights: FP8
        for A8W8 and FP4 for A4W4. An explicit value must match that format;
        forward does not convert weights or build a different quantization path.
        """
        if type(indexer_cp) is not bool:
            raise TypeError("indexer_cp must be a Python bool")
        dtype = getattr(self, "moe_activation_dtype", "fp8")
        if moe_activation_dtype is not None:
            if type(moe_activation_dtype) is not str or moe_activation_dtype not in (
                "fp8",
                "fp4",
            ):
                raise ValueError("moe_activation_dtype must be fp8 or fp4")
            if moe_activation_dtype != dtype:
                raise ValueError(
                    f"loaded experts require {dtype} activations; only A8W8 and A4W4 are supported"
                )
        lm = self.lm
        if plan.key.rows > self.max_rows:
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden, aux = self._backbone(
                plan, input_ids, positions, forward_batch, input_embeds, indexer_cp
            )
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(
        self, plan, input_ids, positions, forward_batch, input_embeds, indexer_cp
    ):
        """``DeepseekV2Model.forward``: the embedding and dense layers on SGLang's
        path, the covered ones as launches, then the final norm."""
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
        from sglang.srt.layers.layer_boundary.residual.batch import (
            complete_output,
            final_norm,
            start,
        )
        from sglang.srt.utils.common import BumpAllocator

        m = self.model
        capture_layers = getattr(m, "layers_to_capture", ())
        aux = AuxHiddenStatePacker(len(capture_layers))
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        start(forward_batch)
        zero = BumpAllocator(
            buffer_size=(m.end_layer - m.start_layer) * 2,
            dtype=torch.float32,
            device=hidden.device,
        )
        gemm_zero = None
        if m.gemm_output_zero_allocator_size > 0:
            gemm_zero = BumpAllocator(
                buffer_size=m.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=hidden.device,
            )
        topk_indices = None
        for i in range(m.start_layer, m.end_layer):
            L = self._layers.get(i)
            if L is not None:
                hidden = self._layer(
                    L,
                    plan,
                    positions,
                    hidden,
                    forward_batch,
                    topk_indices,
                    indexer_cp,
                    aux if i in capture_layers else None,
                )
                topk_indices = None
                continue
            hidden, topk_indices = m.layers[i](
                positions,
                hidden,
                forward_batch,
                zero,
                gemm_zero,
                None,
                prev_topk_indices=topk_indices,
                captured_last_layer_outputs=aux if i in capture_layers else None,
                next_full_attention_layer_id=m.next_full_attention_layer_id.get(i),
            )
        hidden = complete_output(hidden, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden = final_norm(hidden, forward_batch, m.norm)
        return hidden, aux.finalize() if len(aux) else None

    def _layer(
        self,
        L: _Layer,
        plan,
        positions,
        hidden,
        forward_batch,
        topk_indices,
        indexer_cp,
        aux=None,
    ):
        """The stream the previous layer wrote (the first covered layer) or the
        previous launch's output in; the launch's output, or the written stream
        after the last covered layer."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            fold,
            set_written,
            take_output,
        )

        S = plan.key.rows
        W = _width(S, self.tp, getattr(L.w, "expert_mxfp4", False))
        step = self._step(plan, positions, forward_batch, S, W)
        if L.i == self.first:
            # layer 2's output still owes its residual add (to the next input norm)
            h = take_output(fold(hidden, forward_batch), forward_batch)
            if S < W:
                pad = self.pad_h[W]
                pad[:S].copy_(h)
                pad[S:].zero_()
                h = pad
            if L.w.index is None:
                self._take_topk(topk_indices, step, S, W)
        else:
            h = hidden
        if aux is not None:
            # SGLang captures the input residual before this layer's norm.
            # Copy now: subsequent launches reuse the same output arenas.
            aux.capture(h[:S])
        self._bind(L)
        checking = self.check_layers and not torch.cuda.is_current_stream_capturing()
        if checking:
            from sgl_amd_mono.glm53 import check

            if L.i == self.first:
                plan.state["ref_topk"] = topk_indices
            prev = None if L.w.index is not None else plan.state.get("ref_topk")
            ref = check.reference(L, positions, h[:S].clone(), forward_batch, S, prev)
            if ref["topk"] is not None:
                plan.state["ref_topk"] = ref["topk"]
        out = L.ops[W].forward(
            h, L.caches, self.cos, self.sin, step, indexer_cp=indexer_cp
        )
        if checking:
            check.compare(L, ref, out[:S], positions, forward_batch, S)
        if L.i != self.last:
            return out
        self._advance_step(W, indexer_cp)
        return set_written(out[:S], forward_batch)

    def _advance_step(self, width: int, indexer_cp: bool) -> None:
        """Only runtimes launched by this forward consume an epoch."""
        for kind in ("index_cp" if indexer_cp else "index", "shared"):
            self.runtimes[width, kind].advance_step()

    def _take_topk(self, topk_indices, step, S: int, W: int) -> None:
        """Layer 2's top-k (SGLang's [S, 2048] KV slots, -1 past each row's keys)
        into the compact buffer the next launch reads (row s at indptr[s])."""
        if topk_indices is None:
            raise RuntimeError("GLM-5.3 monokernel: no top-k from the layer before")
        start = step.indptr[:S].to(torch.int64).unsqueeze(1)
        count = (step.indptr[1 : S + 1] - step.indptr[:S]).to(torch.int64).unsqueeze(1)
        dst = torch.where(self.topk_cols < count, start + self.topk_cols, W * TOPK)
        step.topk.scatter_(
            0, dst.reshape(-1), topk_indices[:S].reshape(-1).to(torch.int32)
        )

    @staticmethod
    def _bind(L: _Layer) -> None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        from sgl_amd_mono.glm53.gfx950.runner import (
            INDEX_PAGE_TOKENS,
            KV_ROW,
            Glm53Caches,
        )

        pool = get_token_to_kv_pool()
        kv = pool.get_key_buffer(L.i)
        # a layer that reuses the previous top-k keeps no index cache
        idx = pool.get_index_k_with_scale_buffer(L.i) if L.w.index is not None else None
        if (
            L.caches is not None
            and L.caches.kv.data_ptr() == kv.data_ptr()
            and (idx is None or L.caches.idx.data_ptr() == idx.data_ptr())
        ):
            return
        if (
            kv.dtype != torch.bfloat16
            or kv.shape[-1] != KV_ROW
            or not kv.is_contiguous()
        ):
            raise RuntimeError(
                f"GLM-5.3 monokernel: layer {L.i}: MLA cache {kv.dtype} "
                f"{tuple(kv.shape)}; needs contiguous bf16 [slots, 1, {KV_ROW}]"
            )
        if idx is not None and (idx.dtype != torch.uint8 or not idx.is_contiguous()):
            raise RuntimeError(
                f"GLM-5.3 monokernel: layer {L.i}: index cache {idx.dtype} "
                f"{tuple(idx.shape)}; needs contiguous uint8"
            )
        L.caches = Glm53Caches(
            kv=kv.view(kv.shape[0], KV_ROW),
            idx=None if idx is None else idx.view(idx.shape[0], INDEX_PAGE_TOKENS, -1),
        )

    def _step(self, plan, positions, forward_batch, S: int, W: int):
        """The step's rows for every launch (``Glm53Step``): built once a step, in
        the graph when captured."""
        if "step" not in plan.state:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            st = self.steps[W]
            loc = forward_batch.out_cache_loc[:S].to(torch.int64)
            valid = loc != 0
            st.slots[:S] = torch.where(valid, loc, -1)
            st.slots[S:] = -1
            st.positions[:S] = positions[:S].to(torch.int64)
            st.positions[S:] = 0
            # Verify has several query tokens per request. seq_lens is per
            # request; each query's causal extent is its absolute position + 1.
            seq = positions[:S].to(torch.int32) + 1
            count = torch.where(valid, seq.clamp(max=TOPK), 0)
            # indptr[0] stays 0 from its allocation
            st.indptr[1 : S + 1] = torch.cumsum(count, 0, dtype=torch.int32)
            st.indptr[S + 1 :] = st.indptr[S]
            st.req_ids[:S] = torch.arange(S, dtype=torch.int32, device=loc.device)
            st.req_ids[S:] = 0
            md = get_attn_backend().forward_metadata
            page_table = md.real_page_table
            if page_table.shape[0] < S:
                requests = plan.key.batch_size
                if page_table.shape[0] < requests:
                    raise ValueError("GLM page table does not cover every request")
                st.req_ids[:S] = torch.arange(
                    requests, dtype=torch.int32, device=loc.device
                ).repeat_interleave(plan.key.tokens_per_request)
            plan.state["step"] = msgspec.structs.replace(st, page_table=page_table[:S])
        return plan.state["step"]


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel, process_model_config

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53.gfx950.layout import INDEX_KEYS_PER_TASK
    from sgl_amd_mono.glm53.gfx950.runner import (
        Glm53Build,
        Glm53LayerOp,
        Glm53Runtime,
    )
    from sgl_amd_mono.glm53.gfx950.weights import layer_weights, rope_tables

    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    device = model.model.layers[0].input_layernorm.weight.device
    copies = ctx.copies
    covered = _covered_layers(hf_config)
    layers = {
        i: _Layer(
            i,
            model.model.layers[i],
            layer_weights(model.model.layers[i], i, tp, copies),
        )
        for i in covered
    }
    cos, sin = rope_tables([model.model.layers[i] for i in covered], copies)
    shuffled = {L.w.shuffled for L in layers.values()}
    if len(shuffled) != 1:
        raise ValueError("GLM-5.3 monokernel: the layers' FP8 layouts differ")
    fp8_shuffled = shuffled.pop()
    kinds = {
        "index": [i for i in covered if layers[i].w.index is not None],
        "shared": [i for i in covered if layers[i].w.index is None],
    }
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    context_len = process_model_config().context_len
    # the copied table is SGLang's at build; a longer served context reads past it
    if cos.shape[0] < context_len:
        raise ValueError(
            f"GLM-5.3 monokernel: the rotary table holds {cos.shape[0]} positions, "
            f"the server's context is {context_len}"
        )
    # the keys an indexer row scores at most: the served context's, in whole tasks
    tasks = -(-context_len // INDEX_KEYS_PER_TASK)
    max_seq = tasks * INDEX_KEYS_PER_TASK
    formats = {getattr(L.w, "expert_mxfp4", False) for L in layers.values()}
    if len(formats) != 1:
        raise ValueError("covered GLM layers must use one expert weight format")
    expert_mxfp4 = formats.pop()
    cap = TP_MAX_ROWS.get(tp, max(WIDTHS))
    widths = sorted({_width(r, tp, expert_mxfp4) for r in ctx.widths if r <= cap})
    if not widths:
        raise ValueError(
            f"GLM-5.3 monokernel: TP{tp} runs at most {cap} rows on the kernels, and "
            f"this server's decode graphs route {sorted(ctx.widths)}"
        )
    runtimes = {}
    # each runtime's peer handshake is collective: every rank builds them in one order
    for w in widths:
        for kind, ids in kinds.items():
            for cp in (False, True) if kind == "index" else (False,):
                key = Glm53Build(
                    rows=w,
                    tp=tp,
                    launches_per_step=len(ids),
                    with_indexer=kind == "index",
                    # an unpaged launch keeps index_max_seq words in LDS
                    index_max_seq=max_seq if kind == "index" else 0,
                    fp8_shuffled=fp8_shuffled,
                    fault_site=fault_site,
                    fault_after=fault_after,
                    indexer_cp=cp,
                    expert_mxfp4=expert_mxfp4,
                    moe_activation_dtype="fp4" if expert_mxfp4 else "fp8",
                )
                rt = Glm53Runtime(key, rank, p.tp_group.cpu_group, device)
                rt.set_deadline(ctx.deadline_s)
                runtimes[w, "index_cp" if cp else kind] = rt
            for n, i in enumerate(ids):
                layers[i].ops[w] = Glm53LayerOp(
                    runtimes[w, kind],
                    layers[i].w,
                    n,
                    cp_runtime=runtimes[w, "index_cp"] if kind == "index" else None,
                )
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    start = len(widths) * rank // tp
    for w in widths[start:] + widths[:start]:
        for kind in ("index", "index_cp", "shared"):
            runtimes[w, kind].precompile()
    return _Built(model, layers, runtimes, cos, sin, tp, device)


SPEC = MonoSpec(
    name="glm53_hip",
    devices={
        "gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            "kv_cache_dtype": frozenset({"auto", "bfloat16"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # HiSparse maps the logical KV slots to device rows; the kernels write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # layer 3 reads layer 2's top-k as KV slots; unfused top-k hands out positions
            "SGLANG_DSA_FUSE_TOPK": True,
            # benchmark routing overrides: the kernels route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
