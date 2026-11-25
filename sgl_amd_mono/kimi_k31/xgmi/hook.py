# SPDX-License-Identifier: Apache-2.0
"""Opt-in runtime hook wiring XGMI row-sharded prefill fusion into Kimi-K3.1."""

from __future__ import annotations

import os
from typing import Optional

import msgspec
import torch
import torch.distributed as dist

from sgl_amd_mono.kimi_k31.xgmi.heap import (
    PREFILL_AG_HALF_BYTES,
    PREFILL_MAX_ROWS,
    TP,
    XgmiConfig,
    XgmiHeap,
    get_global_heap,
)
from sgl_amd_mono.kimi_k31.xgmi.kernels import fp4_pack_allgather, row_allgather
from sgl_amd_mono.kimi_k31.xgmi.prefill import (
    _quant_routed_input,
    prefill_reduce_scatter_32cu,
)

SGLANG_READS = {
    "srt/layers/layer_boundary/residual/attn_bank.py": ("AttnBankState", "_FfnReadout"),
    "srt/layers/layer_boundary/residual/attn_bank_hip.py": ("RowShardRead",),
    "srt/models/kimi_k3.py": (
        "KimiK3ForConditionalGeneration",
        "KimiK3LinearForCausalLM",
        "KimiK3MoE",
        "_k3_bf16_gemm",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

# Stays strictly above SGLANG_ROCM_K3_COLUMN_TAIL_MAX_TOKENS (384) so DSA layers
# during wide verify steps (up to 64x6=384 rows) never enter row-sharded prefill.
DEFAULT_HOOK_PREFILL_MIN = 512
DEFAULT_HOOK_PREFILL_MAX = 32768


class XgmiHookSummary(msgspec.Struct, frozen=True):
    hooked_layers: int
    rebound_banks: int
    heap_bytes: int
    heap_mib: float
    prefill_range: tuple[int, int]
    use_fp4: bool


def xgmi_enabled() -> bool:
    return os.environ.get("SGLANG_AMD_MONO_K31_XGMI", "0") == "1"


def _resolve_config() -> XgmiConfig:
    fp4_env = os.environ.get("SGLANG_AMD_MONO_K31_XGMI_FP4")
    use_fp4 = (
        fp4_env == "1"
        if fp4_env is not None
        else os.environ.get("AITER_SITUV2_A4W4", "0") == "1"
    )
    return XgmiConfig(
        prefill_min_rows=int(
            os.environ.get(
                "SGLANG_AMD_MONO_K31_XGMI_PREFILL_MIN", str(DEFAULT_HOOK_PREFILL_MIN)
            )
        ),
        prefill_max_rows=int(
            os.environ.get(
                "SGLANG_AMD_MONO_K31_XGMI_PREFILL_MAX", str(DEFAULT_HOOK_PREFILL_MAX)
            )
        ),
        use_fp4=use_fp4,
    )


def _unwrap_backbone(model):
    from sglang.srt.models.kimi_k3 import (
        KimiK3ForConditionalGeneration,
        KimiK3LinearForCausalLM,
    )

    if isinstance(model, KimiK3ForConditionalGeneration):
        model = model.language_model
    if isinstance(model, KimiK3LinearForCausalLM):
        model = model.model
    return model


def _find_attn_bank_state(layer):
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )

    boundary = layer.ffn_boundary
    if boundary is None:
        return None
    for edges in boundary.plan.edges.values():
        read = edges.incoming.need.read
        if isinstance(read, _FfnReadout) and isinstance(read.state, AttnBankState):
            return read.state
    decl_read = boundary.declaration.read
    if isinstance(decl_read, _FfnReadout) and isinstance(
        decl_read.state, AttnBankState
    ):
        return decl_read.state
    return None


def _gather_normed_all(
    heap: XgmiHeap,
    normed: torch.Tensor,
    need_normed_all: bool,
    overlap_normed: bool,
) -> tuple[Optional[torch.Tensor], tuple[torch.Tensor, ...]]:
    if not need_normed_all:
        return None, ()
    if not overlap_normed:
        return row_allgather(heap=heap, x_shard=normed), ()
    # Allocate out_all/tmp on main_stream and keep inputs live until wait_stream(comm_stream)
    # to avoid cross-stream CUDACachingAllocator pools and record_stream fragmentation.
    normed_c = normed.contiguous()
    rows, cols = normed_c.shape
    out_all = normed_c.new_empty(rows * TP, cols)
    max_rows_per_rank = PREFILL_AG_HALF_BYTES // (TP * cols * 2)
    tmp = (
        normed_c.new_empty(max_rows_per_rank * TP, cols)
        if rows > max_rows_per_rank
        else None
    )
    main_stream = torch.cuda.current_stream(normed_c.device)
    comm_stream = heap.streams.comm
    comm_stream.wait_stream(main_stream)
    try:
        with torch.cuda.stream(comm_stream):
            row_allgather(heap=heap, x_shard=normed_c, out_all=out_all, tmp=tmp)
    except BaseException:
        main_stream.wait_stream(comm_stream)
        raise
    keep = (normed_c, tmp) if tmp is not None else (normed_c,)
    return out_all, keep


def _make_shard_front(
    mlp,
    heap: XgmiHeap,
    config: XgmiConfig,
    takes_layer,
    gemm_fn,
    orig_shard_front,
    *,
    need_normed_all: bool,
):
    router_cols = mlp._front_sizes[1]
    rd_weight = mlp._front_w[mlp._front_sizes[0] :]

    def _xgmi_shard_front(normed: torch.Tensor, num_tokens: int) -> bool:
        if not takes_layer(num_tokens):
            return (
                False
                if orig_shard_front is None
                else orig_shard_front(normed, num_tokens)
            )
        if normed.dtype != torch.bfloat16:
            raise TypeError(
                f"XGMI row-sharded front requires bfloat16 activations, got {normed.dtype}"
            )
        if (mlp._row_front_shared is None) != need_normed_all:
            raise RuntimeError(
                "mlp._row_front_shared mutated after TP-verified XGMI hook installation"
            )

        overlap_normed = need_normed_all and num_tokens <= PREFILL_MAX_ROWS
        with heap.overlap_guard():
            normed_all, comm_keep = _gather_normed_all(
                heap=heap,
                normed=normed,
                need_normed_all=need_normed_all,
                overlap_normed=overlap_normed,
            )

            try:
                fused = gemm_fn(normed, rd_weight, out_dtype=torch.float32)
                router_logits = fused[:, :router_cols].contiguous()
                routed_input = fused[:, router_cols:].to(normed.dtype).contiguous()
                topk_out = mlp.topk(normed, router_logits)
                q_data, q_scale = _quant_routed_input(
                    routed_input, use_fp4=config.use_fp4
                )
            finally:
                if overlap_normed:
                    torch.cuda.current_stream(normed.device).wait_stream(
                        heap.streams.comm
                    )
                comm_keep = ()

            q_all, s_all, ids_all, w_all = fp4_pack_allgather(
                heap=heap,
                q_data=q_data,
                q_scale=q_scale,
                topk_ids=topk_out.topk_ids.to(torch.int32),
                topk_weights=topk_out.topk_weights.to(torch.float32),
            )
        mlp._row_front_out = (
            (q_all[:num_tokens], s_all[:num_tokens]),
            ids_all[:num_tokens],
            w_all[:num_tokens],
            normed,
            normed_all,
        )
        return True

    return _xgmi_shard_front


def _make_row_sharded_tail(mlp, heap: XgmiHeap, takes_layer):
    orig_rs_tail = mlp._row_sharded_tail

    def _xgmi_row_sharded_tail(
        latent_rows: torch.Tensor,
        shared_rows: torch.Tensor,
        num_tokens: int,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        if not takes_layer(num_tokens):
            return orig_rs_tail(latent_rows, shared_rows, num_tokens, prefix_sum)
        if latent_rows.dtype != torch.bfloat16 or shared_rows.dtype != torch.bfloat16:
            raise TypeError(
                "XGMI row-sharded tail requires bfloat16 partials, "
                f"got latent={latent_rows.dtype}, shared={shared_rows.dtype}"
            )
        if shared_rows.shape[0] > num_tokens:
            latent_rows[num_tokens:].zero_()
            shared_rows[num_tokens:].zero_()
        latent = prefill_reduce_scatter_32cu(heap=heap, partial=latent_rows)
        shared = prefill_reduce_scatter_32cu(heap=heap, partial=shared_rows)
        return mlp._row_tail(latent, shared, num_tokens, prefix_sum)

    return _xgmi_row_sharded_tail


def _layer_plan(layer):
    """Inspect a MoE layer without mutating it; return (attn_state, orig_read, tp_plan_entry)."""
    mlp = layer.mlp
    bf16_ok = (
        layer.post_attention_layernorm.weight.dtype == torch.bfloat16
        and mlp._front_sizes is not None
        and len(mlp._front_sizes) == 3
        and mlp._front_w is not None
        and mlp._front_w.dtype == torch.bfloat16
        and mlp.routed_expert_up_proj is not None
        and mlp.routed_expert_up_proj.weight.dtype == torch.bfloat16
    )
    attn_state = _find_attn_bank_state(layer) if bf16_ok else None
    orig_read = attn_state.ffn_row_shard if attn_state is not None else None
    rebound_ok = orig_read is not None
    entry = (
        rebound_ok,
        bool(mlp._row_front_on),
        mlp._row_front_shared is None,
        mlp._a8w4_moe_min_tokens,
        mlp._a8w4_moe_max_tokens,
    )
    return (attn_state if rebound_ok else None), orig_read, entry


def _snapshot_still_matches(layer, attn_state, orig_read, plan_entry) -> bool:
    """Verify live layer fields and readout still match the TP-agreed snapshot."""
    rebound_ok, row_front_on, need_normed_all, min_tok, max_tok = plan_entry
    mlp = layer.mlp
    if (
        bool(mlp._row_front_on) != row_front_on
        or (mlp._row_front_shared is None) != need_normed_all
        or mlp._a8w4_moe_min_tokens != min_tok
        or mlp._a8w4_moe_max_tokens != max_tok
    ):
        return False
    if rebound_ok:
        return (
            attn_state is not None
            and attn_state.ffn_row_shard is orig_read
            and orig_read is not None
            and layer.post_attention_layernorm.weight.dtype == torch.bfloat16
            and mlp._front_sizes is not None
            and len(mlp._front_sizes) == 3
            and mlp._front_w is not None
            and mlp._front_w.dtype == torch.bfloat16
            and mlp.routed_expert_up_proj is not None
            and mlp.routed_expert_up_proj.weight.dtype == torch.bfloat16
        )
    return True


def _prepare_moe_layer_hook(
    layer,
    *,
    attn_state,
    orig_read,
    plan_entry,
    heap: XgmiHeap,
    cfg: XgmiConfig,
    gemm_fn,
    row_shard_cls,
):
    """Build one MoE layer's replacement hooks without mutating the model.

    None when ``plan_entry`` marks the layer ineligible; the tuple is committed only
    after every TP rank passes the installation vote.
    """
    rebound_ok, row_front_on, need_normed_all, min_tok, max_tok = plan_entry
    if not rebound_ok:
        return None

    mlp = layer.mlp

    def takes_layer(num_tokens: int) -> bool:
        if (
            bool(mlp._row_front_on) != row_front_on
            or mlp._a8w4_moe_min_tokens != min_tok
            or mlp._a8w4_moe_max_tokens != max_tok
        ):
            raise RuntimeError(
                "mlp._row_front_on mutated after TP-verified XGMI hook installation"
            )
        return (
            row_front_on
            and min_tok is not None
            and not torch.cuda.is_current_stream_capturing()
            and max(cfg.prefill_min_rows, min_tok)
            <= num_tokens
            <= min(cfg.prefill_max_rows, max_tok)
        )

    orig_elig = mlp._row_sharded_tail_eligible

    def _tail_eligible(num_tokens: int, has_prefix: bool) -> bool:
        if has_prefix and takes_layer(num_tokens):
            return True
        return orig_elig(num_tokens, has_prefix)

    tail_fn = _make_row_sharded_tail(mlp=mlp, heap=heap, takes_layer=takes_layer)

    def _takes_combined(num_tokens: int) -> bool:
        return takes_layer(num_tokens) or bool(orig_read.takes(num_tokens))

    shard_front_fn = _make_shard_front(
        mlp=mlp,
        heap=heap,
        config=cfg,
        takes_layer=takes_layer,
        gemm_fn=gemm_fn,
        orig_shard_front=orig_read.shard_front,
        need_normed_all=need_normed_all,
    )
    row_shard = row_shard_cls(
        _takes_combined,
        shard_front=shard_front_fn,
    )
    return (mlp, attn_state, _tail_eligible, tail_fn, shard_front_fn, row_shard)


def _collect_moe_plan(*, model, moe_cls) -> tuple[list, list]:
    moe_targets = []
    plan_entries = []
    for layer in _unwrap_backbone(model).layers:
        if not isinstance(layer.mlp, moe_cls) or not layer.mlp.use_latent_moe:
            continue
        attn_state, orig_read, entry = _layer_plan(layer)
        moe_targets.append((layer, attn_state, orig_read))
        plan_entries.append(entry)
    return moe_targets, plan_entries


def _prepare_all_hooks(
    *,
    moe_targets: list,
    plan_entries: list,
    heap: XgmiHeap,
    cfg: XgmiConfig,
    gemm_fn,
    row_shard_cls,
) -> Optional[list]:
    """Every eligible layer's prepared hooks, or None if a layer drifted from its plan."""
    prepared = []
    for (layer, attn_state, orig_read), entry in zip(moe_targets, plan_entries):
        if not _snapshot_still_matches(
            layer, attn_state=attn_state, orig_read=orig_read, plan_entry=entry
        ):
            return None
        item = _prepare_moe_layer_hook(
            layer,
            attn_state=attn_state,
            orig_read=orig_read,
            plan_entry=entry,
            heap=heap,
            cfg=cfg,
            gemm_fn=gemm_fn,
            row_shard_cls=row_shard_cls,
        )
        if item is not None:
            prepared.append(item)
    return prepared


def _vote_install(
    *,
    heap: XgmiHeap,
    local_valid: bool,
    plans_match: bool,
    local_error: Optional[Exception],
) -> None:
    if heap.cpu_group is not None and dist.is_initialized():
        ok = torch.tensor([int(local_valid)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=heap.cpu_group)
        if not ok.item():
            reason = (
                "TP ranks disagree on XGMI hook eligibility or _row_front_shared configuration"
                if not plans_match
                else "MoE layer dispatch fields or ffn_row_shard mutated during TP-agreed XGMI hook installation"
            )
            raise RuntimeError(reason) from local_error
    elif not local_valid:
        raise RuntimeError(
            "MoE layer dispatch fields or ffn_row_shard mutated during XGMI hook installation"
        ) from local_error


def install_xgmi_hooks(
    model,
    *,
    cpu_group=None,
    rank: Optional[int] = None,
    device: Optional[torch.device] = None,
    heap: Optional[XgmiHeap] = None,
    config: Optional[XgmiConfig] = None,
) -> XgmiHookSummary:
    """Install opt-in XGMI row-sharded prefill hooks on a Kimi-K3.1 model."""
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MoE, _k3_bf16_gemm

    if heap is None:
        heap = get_global_heap(cpu_group=cpu_group, rank=rank, device=device)

    cfg: Optional[XgmiConfig] = None
    moe_targets = []
    plan_entries = []
    local_plan = None
    local_error: Optional[Exception] = None
    try:
        cfg = _resolve_config() if config is None else config
        moe_targets, plan_entries = _collect_moe_plan(model=model, moe_cls=KimiK3MoE)
        local_plan = (
            cfg.prefill_min_rows,
            cfg.prefill_max_rows,
            cfg.use_fp4,
            tuple(plan_entries),
        )
    except Exception as err:
        local_error = err

    plans_match = local_plan is not None
    if heap.cpu_group is not None and dist.is_initialized():
        all_plans = [None] * heap.world_size
        dist.all_gather_object(all_plans, local_plan, group=heap.cpu_group)
        plans_match = local_plan is not None and all(
            plan == local_plan for plan in all_plans
        )

    # Prepare every replacement hook without mutating the model; convert any local
    # validation or preparation error into a failed TP vote so peers never strand.
    prepared = []
    local_valid = plans_match
    if local_valid:
        assert cfg is not None
        try:
            prepared_or_none = _prepare_all_hooks(
                moe_targets=moe_targets,
                plan_entries=plan_entries,
                heap=heap,
                cfg=cfg,
                gemm_fn=_k3_bf16_gemm,
                row_shard_cls=RowShardRead,
            )
            local_valid = prepared_or_none is not None
            prepared = prepared_or_none or []
        except Exception as err:
            local_error = err
            local_valid = False

    _vote_install(
        heap=heap,
        local_valid=local_valid,
        plans_match=plans_match,
        local_error=local_error,
    )

    # Commit prepared replacements atomically on the quiescent model.
    for mlp, attn_state, tail_elig, tail_fn, shard_front_fn, row_shard in prepared:
        mlp._row_sharded_tail_eligible = tail_elig
        mlp._row_sharded_tail = tail_fn
        mlp.shard_front = shard_front_fn
        attn_state.ffn_row_shard = row_shard
    hooked = len(moe_targets)
    rebound_banks = len(prepared)

    assert cfg is not None
    model._xgmi_heap = heap
    return XgmiHookSummary(
        hooked_layers=hooked,
        rebound_banks=rebound_banks,
        heap_bytes=heap.heap_bytes,
        heap_mib=heap.heap_mib,
        prefill_range=(cfg.prefill_min_rows, cfg.prefill_max_rows),
        use_fp4=cfg.use_fp4,
    )


def maybe_install_xgmi_hooks(model, *, log_fn=None) -> Optional[XgmiHookSummary]:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    enabled = int(xgmi_enabled())
    tp_group = parallel.tp_group.cpu_group
    if dist.is_initialized() and tp_group is not None:
        vote = torch.tensor([enabled, -enabled], dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=tp_group)
        if vote[0].item() == 0 and vote[1].item() == -1:
            raise RuntimeError("SGLANG_AMD_MONO_K31_XGMI disagrees across TP ranks")
        enabled = int(vote[0].item())
    if not enabled:
        return None
    heap = get_global_heap(
        cpu_group=tp_group,
        rank=parallel.tp_rank,
    )
    summary = install_xgmi_hooks(model, heap=heap)
    if log_fn is not None:
        log_fn(
            f"xgmi: hooked {summary.hooked_layers} MoE layers "
            f"(rebound {summary.rebound_banks} banks), "
            f"heap {summary.heap_mib:.1f} MiB, prefill {summary.prefill_range}, fp4={summary.use_fp4}"
        )
    return summary
