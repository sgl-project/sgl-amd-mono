# SPDX-License-Identifier: Apache-2.0
"""Wide-row verify steps (R requests x L rows, R >= 2, L = the server's draft tokens,
6 or 8) on the layer launch (``gfx950/layer.py``), eager v0.

A step walks the layers as SGLang's backbone does, except for every run of covered
layers (KDA attention + latent MoE: all but layer 0, which has a dense MLP, and the
DSA layers), which go launch by launch:

    enter    after a DSA layer (MoE on the merged front): SGLang runs that layer up
             to its MoE's TP partials and the first launch runs its tail (t1 on);
             else the stream after the previous layer, folded to the full prefix.
             The bank -> this rank's column slices
    layer i  launch i (MoE tail of i - 1 unless it starts the run; restart on a
             block-write layer; DSpark capture of i - 1 when asked; reads the conv
             state from SGLang's cache by slot and writes the verify windows for
             SGLang's accept) -> a written bank row gathered into SGLang's bank
             -> SGLang's routed experts on the launch's latent and router logits
    leave    the last layer's MoE tail on SGLang (``_column_split_tail``), this rank's
             prefix columns added into its shared columns before the tail's all-reduce
             -> the stream the next standard layer reads

The column-sharded state lives only inside a run, so the standard layers see exactly
what they see today. Rows are padded to a bucket (padded requests have no KDA
state: slot -1).
"""

import contextlib
import os

import torch

# The SGLang names this module reads (test/unit/test_contract.py).
SGLANG_READS = {
    "srt/runtime_context.py": ("get_spec", "get_parallel", '"dcp_enabled"'),
    "srt/arg_groups/fields/parallel.py": ("dcp_replicate_q_proj",),
    "srt/models/kimi_k3.py": (
        "KimiK3MoE",
        "AuxHiddenStateList",
        "_cdiv",
        # the routed experts the launch's top-k / sort feed (_routed, _experts)
        "_forward_routed",
        "_route_quant_fuse_eligible",
        # DSA layers stop before o_proj for dsa_post
        "forward_pre_o_proj",
        "use_output_gate",
        "g_proj",
        '"_mono_pre"',
        "mlp_res_norm",
        "mlp_res_proj",
        "prev_valid_blocks",
    ),
    "srt/layers/layer_boundary/__init__.py": (
        "AttentionInputs",
        "get_attn_tp_context",
    ),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "set_attn_inputs",
        "clear_attn_inputs",
    ),
    "srt/models/deepseek_v2.py": (
        "prepare_qkv_latent",
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj",
        "q_b_proj_qrep_weight",
        "w_kc_qrep",
        "use_deep_gemm_bmm",
        "use_dsa",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "num_local_heads",
        "dcp_size",
        "qk_head_dim",
        "indexer",
    ),
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py": (
        "is_dcp_q_replicate_phase",
        "_split_q_nope_pe",
    ),
    "srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py": (
        "forward_absorb_rocm_prepare",
        '"_mono_pre"',
        "q_replicate_active",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": ("wk", "weights_proj", "wq_b"),
    "srt/layers/attention/dsa/indexer_prologue_hip.py": ('"_mono_pre"',),
    "srt/layers/activation.py": ("SituAndMul", "beta", "linear_beta"),
    # the new dev_amd moved KimiK3MoE's ROCm paths into kimi_k3_amd
    # NextAttnRead.max_tokens: the row cap of SGLang's column tail, which
    # _column_exit must not exceed
    "srt/models/kimi_k3_amd/moe.py": (
        "max_tokens",
        "_vector_aligned",
        "_forward_shared_casting_routed",
        "_forward_routed_a8w4",
    ),
    # _launch_topk_ok: when SGLang's top-k post-processing is a no-op
    "srt/environ.py": (
        "envs",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/layers/moe/__init__.py": ("get_moe_a2a_backend",),
    "srt/layers/moe/topk.py": (
        "TopKOutputFormat",
        "is_expert_remapping_enabled",
        "build_precomputed_topk_output",
        "num_shared_topk_slots",
        "waterfill_balancer",
        "enable_waterfill",
    ),
    # _experts: the routed experts on the launch's top-k / sort
    "srt/layers/zero_copy_context.py": ("set_moe_output",),
    "srt/layers/moe/route_quant_handoff.py": ("stage", "clear"),
    "srt/layers/attn_residual_hip.py": ("gather_covers",),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "written_residual",
        "final_norm",
        "fold",
        "set_written",
        "start",
    ),
    # re-exported by srt/distributed/__init__.py
    "srt/distributed/communication_op.py": (
        "tensor_model_parallel_all_reduce",
        "tensor_model_parallel_all_gather",
    ),
    "srt/utils/common.py": ("BumpAllocator",),
    # the layer's place in the state pool
    "srt/layers/attention/hybrid_linear_attn_backend.py": ("state_layer",),
    "srt/mem_cache/kv/router.py": ("view_layer_id",),
}

TP, H, SL, E, NF, LAT, KO, SI = 8, 7168, 896, 896, 2096, 3584, 1536, 768
REQ_BUCKETS = (
    2,
    4,
    8,
    16,
    32,
    48,
)  # requests a bucket holds; its rows are L x requests
SUPPORTED_L = (6, 8)
BUCKETS = tuple(8 * r for r in REQ_BUCKETS)
CFGS = {  # 8-token bucket -> (up, in_proj, o_proj, front, shared down): bm:bn:wm:wn:ks; an L-token
    # bucket of the same requests takes the same configs
    16: (
        "16:128:1:8:8",
        "16:128:1:8:4",
        "16:128:1:8:4",
        "16:128:1:8:8",
        "16:128:1:8:1",
    ),
    32: (
        "32:128:2:4:8",
        "32:128:2:4:4",
        "32:128:2:4:2",
        "32:128:2:4:8",
        "32:128:2:4:1",
    ),
    64: ("64:128:2:4:8", "64:128:2:4:4", "64:64:2:4:2", "64:128:2:4:8", "64:64:2:4:1"),
    128: (
        "128:32:4:2:8",
        "128:128:2:4:4",
        "64:64:2:4:1",
        "128:128:2:4:8",
        "64:64:2:4:1",
    ),
    256: (
        "128:32:4:2:8",
        "128:128:2:4:4",
        "128:64:2:4:1",
        "128:128:2:4:8",
        "128:64:2:4:1",
    ),
    384: (
        "128:32:4:2:4",
        "192:128:4:2:2",
        "192:64:4:2:1",
        "128:128:2:4:4",
        "128:64:2:4:1",
    ),
}
# bucket rows -> MoE front config, where the CFGS one is a poor fit: 17 column tiles x the row tiles x ks must stay
# within BLOCKS (one round), and a row tile past m is wasted work
FRONT_ROWS = {192: "64:128:2:4:4", 256: "96:128:2:4:4", 288: "96:128:2:4:4"}
# bucket rows -> in_proj config, by the same rule (50 column tiles x row tiles x ks within BLOCKS, no padded row tile)
INP_ROWS = {96: "96:128:2:4:4", 192: "192:128:4:2:4", 256: "128:128:2:4:2"}
# bucket rows -> latent up / shared down configs, by the same rule (o_proj: padded rows measured no cost, kept)
UP_ROWS = {
    192: "96:64:2:4:8",
    256: "128:64:4:2:8",
    288: "96:64:2:4:4",
    384: "96:128:2:4:8",
}
SDOWN_ROWS = {288: "96:128:2:4:1", 384: "192:64:4:2:1"}
NARROW = 8  # rows up to this run Kimi-K3's launches (the wide launch starts at 2 requests above 8 rows)


def tokens() -> int:
    """Rows a request takes in this server's verify steps (DSpark / DFLASH draft tokens)."""
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1
    return int(spec.speculative_num_draft_tokens or 1)


def req_buckets() -> tuple:
    # SGLANG_AMD_MONO_K31_WIDE_BUCKETS lists rows at 8 tokens a request
    env = os.environ.get("SGLANG_AMD_MONO_K31_WIDE_BUCKETS")
    return tuple(sorted(int(x) // 8 for x in env.split(","))) if env else REQ_BUCKETS


def buckets(L: int = 8) -> tuple:
    return tuple(L * r for r in req_buckets())


def cfg_rows(b: int, L: int) -> str:
    """The CFGS entry of an L-token bucket of b rows."""
    assert b % L == 0 and b // L in REQ_BUCKETS, (b, L)
    up, inp, oproj, front, sdown = CFGS[8 * (b // L)]
    return (
        UP_ROWS.get(b, up),
        INP_ROWS.get(b, inp),
        oproj,
        FRONT_ROWS.get(b, front),
        SDOWN_ROWS.get(b, sdown),
    )


def wide_widths(max_reqs: int = 48) -> tuple:
    # only widths a warmed bucket holds: a graph capture must not meet a cold one; rows
    # up to a bucket left out below the smallest one stay on SGLang's own path. The
    # server's draft tokens are not known at import: every supported L's widths (the
    # binding keeps those a multiple of the server's)
    rs = req_buckets()
    lo = max((x for x in REQ_BUCKETS if x < rs[0]), default=1)
    top = min(max_reqs, rs[-1])
    widths = {
        L * r
        for L in SUPPORTED_L
        for r in range(max(2, lo + 1), top + 1)
        if L * r > NARROW
    }
    # the binding serves a width at every L it is a multiple of: drop one another L
    # reads as more requests than the largest bucket holds (8-token 312 .. 384 rows
    # would be 52 .. 64 requests at 6 tokens)
    return tuple(
        sorted(
            w for w in widths if all(w // L <= top for L in SUPPORTED_L if w % L == 0)
        )
    )


def _say(msg):
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[k31 wide] {msg}", flush=True)


def _launch_sort(b):
    """The sort kind bucket b's launches also make (with the output zeroing), or None
    (``SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT=1``, needs ``..._LAUNCH_TOPK=1``; m <= 64):
    "aux16" (aiter's BM16 aux sort, handed over by ``aiter.presorted_moe.register_presorted_moe``) when aiter
    has that API, else "sgl" (SGLang's BM32 small sort + mxfp8 quant, ``presort.injected``)."""
    from sgl_amd_mono.kimi_k31 import presort
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

    if not (
        os.environ.get("SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT", "1") == "1"
        and os.environ.get("SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK", "1") == "1"
        and b <= mf.SORT_MAX_ROWS
    ):
        return None
    return presort.sort_kind()


def _sort_views(buf, m, kind="sgl"):
    """Tensor views of the launch's sort buffer (``moe_front.sort_layout``): as SGLang's small sort returns them
    ("sgl"), or under ``register_presorted_moe``'s key names ("aux16")."""
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

    ly = mf.sort_layout(m, kind)
    o = ly["offs"]

    def at(k, n):
        return buf[o[k] : o[k] + n]

    if kind == "aux16":
        maxp, nblk = ly["maxp"], ly["nblk"]
        return dict(
            kind=kind,
            sorted_token_ids=at("ids", 4 * maxp).view(torch.int32),
            sorted_weights=at("w", 4 * maxp).view(torch.float32),
            sorted_expert_ids=at("eids", 4 * nblk).view(torch.int32),
            num_valid_ids=at("nv", 8).view(torch.int32),
            m_indices=at("mi", 4 * maxp).view(torch.int32),
            reverse_sorted=at("dest", 4 * m * mf.TOPK).view(torch.int32),
        )
    maxp, nblk, srows = ly["maxp"], ly["nblk"], ly["srows"]
    return dict(
        kind=kind,
        ids=at("ids", 4 * maxp).view(torch.int32),
        w=at("w", 4 * maxp).view(torch.float32),
        eids=at("eids", 4 * nblk).view(torch.int32),
        nv=at("nv", 8).view(torch.int32),
        q=at("q", m * LAT).view(torch.float8_e4m3fn).view(m, LAT),
        qs=at("qs", srows * mf.SCALE_COLS).view(srows, mf.SCALE_COLS),
    )


def _launch_topk_ok(moe) -> bool:
    """Whether the routed experts may take the launch's own top-16 (ids, w) instead of SGLang's
    ``self.topk``: the launch's route stage is sigmoid + bias, ungrouped top-16, renormalized (10.18), so only a
    STANDARD output whose post-processing is a no-op (no padding mask / EPLB remap / waterfill) qualifies."""
    if os.environ.get("SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK", "1") != "1":
        return False
    ok = getattr(moe, "_k31_launch_topk_ok", None)
    if ok is None:
        # SGLang's precomputed_topk_postprocess_is_noop is CUDA-only. On HIP, select_experts' post-processing
        # (_post_process_topk_ids) is a no-op for K3's call (no num_token_non_padded, no shared slots) unless
        # EPLB remapping, megamoe or a routing simulation is on, so check exactly those.
        bad = []
        try:
            from sglang.srt.environ import envs
            from sglang.srt.layers.moe import get_moe_a2a_backend
            from sglang.srt.layers.moe.topk import (
                TopKOutputFormat,
                is_expert_remapping_enabled,
            )

            cfg = moe.topk.topk_config
            checks = {
                "output_format": cfg.output_format in (None, TopKOutputFormat.STANDARD),
                "sigmoid": cfg.scoring_func == "sigmoid",
                "bias": cfg.correction_bias is not None,
                "ungrouped": (cfg.num_expert_group or 1) == 1
                and (cfg.topk_group or 1) == 1,
                "top16": cfg.top_k == 16,
                "no_shared_slots": cfg.num_shared_topk_slots == 0,
                "no_waterfill": moe.topk.waterfill_balancer is None
                and not moe.topk.enable_waterfill,
                "no_eplb_remap": not is_expert_remapping_enabled(),
                "no_megamoe": not get_moe_a2a_backend().is_megamoe(),
                "no_sim": not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
                and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get(),
            }
            bad = [k for k, v in checks.items() if not v]
        except Exception as e:
            bad = [f"{type(e).__name__}: {e}"]
        ok = not bad
        moe._k31_launch_topk_ok = ok
        _say(
            f"launch top-k {'on' if ok else 'off'} (layer {moe.layer_idx})"
            + ("" if ok else f": {', '.join(bad)}")
        )
    return ok


def _routed(moe, lat, logits, ids, w, out, sortv=None):
    """SGLang's ``KimiK3MoE._forward_routed(lat, logits, lat, out)``, with the launch's (ids, w) in place of
    ``self.topk`` when ``SGLANG_AMD_MONO_K31_WIDE_LAUNCH_TOPK=1`` (saves aiter's topk kernel a layer)."""
    if not _launch_topk_ok(moe):
        return moe._forward_routed(lat, logits, lat, out)
    if hasattr(moe, "_forward_routed_a8w4") and moe._forward_routed_a8w4(
        lat, logits, lat, out
    ):
        return
    _experts(moe, lat, ids, w, out, sortv)


def _experts(moe, lat, ids, w, out, sortv):
    """SGLang's routed experts on the launch's (ids, w) -> out; with ``sortv``, on the launch's sort + quant."""
    from sglang.srt.layers.moe.route_quant_handoff import clear, stage
    from sglang.srt.layers.moe.topk import build_precomputed_topk_output
    from sglang.srt.layers.zero_copy_context import set_moe_output

    if moe._route_quant_fuse_eligible:
        stage(lat, layer=moe.experts)
    try:
        topk_output = build_precomputed_topk_output(
            w, ids, moe.topk.topk_config, moe.layer_idx
        )
        with contextlib.ExitStack() as es:
            inj = None
            if sortv is not None:
                from sgl_amd_mono.kimi_k31 import presort

                cm = (
                    presort.registered(sortv, ids, w, out)
                    if sortv["kind"] == "aux16"
                    else presort.injected(sortv, ids.shape[0], ids, out)
                )
                inj = es.enter_context(cm)
            with set_moe_output(out):
                y = moe.experts(lat, topk_output)
            if inj is not None and not getattr(moe, "_k31_sort_said", False):
                moe._k31_sort_said = True
                _say(
                    f"launch sort {sortv['kind']} {'on' if inj['used'] else 'NOT used'} "
                    f"(layer {moe.layer_idx})"
                )
    finally:
        clear()
    if y.data_ptr() != out.data_ptr():
        out.copy_(y)


class _Layer:
    """A covered layer's weights as the launch reads them (views where SGLang's
    layout already fits; the front rows and the folded up projection are copies)."""

    def __init__(self, backbone, i, rank, tails):
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        bf = torch.bfloat16
        layer = backbone.layers[i]
        attn, moe = layer.self_attn, layer.mlp
        self.i, self.layer, self.moe = i, layer, moe
        self.t1 = (i - 1) in tails  # the previous layer's MoE tail runs in this launch
        self.write = bool(layer.is_block_write_layer)
        self.nvb = layer.prev_valid_blocks
        r = rank
        self.w = dict(
            w_ip=attn._qkvgbfa_layer.weight,
            cw_a=(
                layer.self_attention_res_norm.weight.float()
                * layer.self_attention_res_proj.weight.view(H).float()
            ).contiguous(),
            w_in=layer.input_layernorm.weight,
            wo=attn.o_proj.weight,
            cw_m=(
                layer.mlp_res_norm.weight.float()
                * layer.mlp_res_proj.weight.view(H).float()
            ).contiguous(),
            w_ln=layer.post_attention_layernorm.weight,
            bias=moe.gate.e_score_correction_bias.float().contiguous(),
            w_sd=moe.shared_experts.down_proj.weight,
        )
        self.w["wf"], self.front_rows = _front_weight(moe, r)
        self.up_norm = False
        if self.t1:
            prev = backbone.layers[i - 1].mlp
            up = prev.routed_expert_up_proj.weight[r * SL : (r + 1) * SL]
            if up.is_contiguous():
                # SGLang's rows as is; the launch scales the gathered latent by the norm weight
                self.w["w_up"], self.up_norm = up, True
                self.w["bias"] = torch.cat(
                    [self.w["bias"], prev.routed_expert_norm.weight.float()]
                )
            else:
                self.w["w_up"] = pa.fold_up(
                    prev.routed_expert_up_proj.weight, prev.routed_expert_norm.weight, r
                )
        self.kda_w = dict(
            w_fb=attn.f_b_proj.weight,
            conv_w=attn.qkv_conv1d.weight.squeeze(1),
            a_log=attn.A_log.detach().reshape(-1).contiguous(),
            dt_bias=attn.dt_bias,
            on_w=attn.o_norm.weight.data.to(bf).contiguous(),
        )
        self.eps = layer.input_layernorm.variance_epsilon
        self.onorm_eps = float(attn.o_norm.eps)
        self.mamba_idx = None
        mods = (layer, backbone.layers[i - 1]) if self.t1 else (layer,)
        self.shared = {
            t.untyped_storage().data_ptr() for mod in mods for t in mod.parameters()
        }


def _front_weight(moe, rank):
    """(wf, front_rows) for the layer launch's MoE front: SGLang's merged front weight
    (``_merge_front_weights``: router, latent down and shared gate_up as views of one
    bf16 buffer) with the rows this rank's slices start at, else a copy of this rank's
    [router | latent down | shared gate_up] rows (front_rows None)."""
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

    parts = (
        moe.gate.weight[rank * mf.E_R : (rank + 1) * mf.E_R],
        moe.routed_expert_down_proj.weight[rank * mf.L_R : (rank + 1) * mf.L_R],
        moe.shared_experts.gate_up_proj.weight,
    )
    st = {p.untyped_storage().data_ptr() for p in parts}
    if len(st) == 1 and all(
        p.dtype == torch.bfloat16 and p.stride() == (H, 1) for p in parts
    ):
        storage = parts[0].untyped_storage()
        base = torch.empty(0, dtype=torch.bfloat16, device=parts[0].device).set_(
            storage, 0, (storage.nbytes() // (2 * H), H)
        )
        return base, tuple((p.data_ptr() - base.data_ptr()) // (2 * H) for p in parts)
    return torch.cat([p.to(torch.bfloat16) for p in parts]).contiguous(), None


def _is_covered(layer) -> bool:
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    return hasattr(layer.self_attn, "qkv_conv1d") and isinstance(layer.mlp, KimiK3MoE)


def _tail_fusable(layer) -> bool:
    """A standard layer (DSA) whose MoE tail the next run's first launch can take:
    a stream-writing latent MoE on the merged front."""
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    mlp = layer.mlp
    return (
        isinstance(mlp, KimiK3MoE)
        and layer._ffn_writes_stream
        and mlp._eligible_for_fused_front
        and not getattr(layer, "_sp_moe", False)
        and not getattr(mlp, "_use_mega_moe", False)
    )


def _dsa_fusable(layer) -> bool:
    """Whether a non-KDA (DSA) layer can run its post-attention chain in ``dsa_post``."""
    if os.environ.get("SGLANG_AMD_MONO_K31_WIDE_DSA_POST", "1") != "1":
        return False
    attn = layer.self_attn
    if (
        not _tail_fusable(layer)
        or getattr(layer, "is_block_write_layer", False)
        or getattr(layer, "all_reduce_fusion", False)
        or not getattr(attn, "use_output_gate", False)
        or not hasattr(attn, "forward_pre_o_proj")
        or getattr(layer, "mlp_res_norm", None) is None
        or getattr(layer, "mlp_res_proj", None) is None
    ):
        return False
    wo = getattr(getattr(attn, "o_proj", None), "weight", None)
    ln = getattr(layer, "post_attention_layernorm", None)
    w_ln = getattr(ln, "weight", None)
    res_norm = layer.mlp_res_norm
    res_proj_w = getattr(layer.mlp_res_proj, "weight", None)
    from sglang.srt.layers.activation import SituAndMul

    shared = getattr(layer.mlp, "shared_experts", None)
    act_fn = getattr(shared, "act_fn", None)
    w_sd = getattr(getattr(shared, "down_proj", None), "weight", None)
    return (
        wo is not None
        and getattr(attn.o_proj, "bias", None) is None
        and wo.dtype == torch.bfloat16
        and tuple(wo.shape) == (H, KO)
        and wo.is_contiguous()
        and w_ln is not None
        and w_ln.dtype == torch.bfloat16
        and tuple(w_ln.shape) == (H,)
        and w_ln.is_contiguous()
        and res_proj_w is not None
        and res_proj_w.numel() == H
        and res_proj_w.is_contiguous()  # read as .view(H)
        and getattr(ln, "variance_epsilon", None) is not None
        and getattr(res_norm, "variance_epsilon", None) == ln.variance_epsilon
        and isinstance(getattr(res_norm, "weight", None), torch.Tensor)
        and tuple(res_norm.weight.shape) == (H,)
        and isinstance(act_fn, SituAndMul)
        and act_fn.beta == 4.0
        and act_fn.linear_beta == 25.0
        and w_sd is not None
        and getattr(shared.down_proj, "bias", None) is None
        and w_sd.dtype == torch.bfloat16
        and tuple(w_sd.shape) == (H, SI)
        and w_sd.is_contiguous()
    )


def _pre_o_proj(layer, x, positions, fb, zero_allocator):
    """Run ``layer.self_attn`` up to the gated bf16 ``[rows, KO]`` input of ``o_proj``."""
    from sglang.srt.layers.layer_boundary import (
        AttentionInputs,
        get_attn_tp_context,
    )

    attn = layer.self_attn
    latent = getattr(attn, "prepare_qkv_latent", None)
    if latent is not None:
        get_attn_tp_context().set_attn_inputs(AttentionInputs(x, fb, latent))
    try:
        return attn.forward_pre_o_proj(
            hidden_states=x,
            positions=positions,
            forward_batch=fb,
            zero_allocator=zero_allocator,
        )
    finally:
        if latent is not None:
            get_attn_tp_context().clear_attn_inputs()


# SGLang's q-replicate predicate in forward_absorb_rocm_prepare, conjunct by conjunct: the
# one the plugin evaluates at every step before it hands a replicated-Q front over (the
# wide launch in _run_mla_pre, the narrow one in _mla_covered)
_QREP_PREDICATE = (
    "self.is_dcp_q_replicate_phase(forward_batch)",
    "not self.use_deep_gemm_bmm",
    "self.w_kc_qrep is not None",
    "self.q_b_proj_qrep_weight is not None",
)


def _head_count(node, assigned: dict, *, qrep: bool):
    """A ``pre["q"].view`` head-count expression with ``q_replicate_active = qrep``, as
    ``(coefficient, power of self.num_local_heads, power of self.dcp_size)``. Reads int
    literals, those two attributes, ``*``, a conditional on the bare ``q_replicate_active``
    and names in ``assigned``; None for anything else."""
    import ast

    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value, 0, 0
    if isinstance(node, ast.Name):
        if node.id not in assigned:
            return None
        return _head_count(assigned[node.id], {}, qrep=qrep)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return {"num_local_heads": (1, 1, 0), "dcp_size": (1, 0, 1)}.get(node.attr)
    if (
        isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.Name)
        and node.test.id == "q_replicate_active"
    ):
        return _head_count(node.body if qrep else node.orelse, assigned, qrep=qrep)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a = _head_count(node.left, assigned, qrep=qrep)
        b = _head_count(node.right, assigned, qrep=qrep)
        if a is None or b is None:
            return None
        return a[0] * b[0], a[1] + b[1], a[2] + b[2]
    return None


def _mono_pre_q_view(tree):
    """``(head expression, assignments before it)`` of SGLang's ``_mono_pre`` handoff, or
    None unless the function's source has exactly the shape below. This is a source-shape
    check: anything else reads as unsupported, and the layer keeps SGLang's own front.

    - ``q_replicate_active`` is not a parameter; it is assigned once, before the view, to
      ``_QREP_PREDICATE``'s conjunction (any order), whose ``forward_batch`` is the
      function's parameter, never rebound.
    - ``pre = self.__dict__.get("_mono_pre")`` is its one assignment; every other use of
      ``pre`` is ``pre is (not) None``, a read of ``pre["<key>"]`` or ``pre.get("<key>")``
      of another key than "q"; ``"_mono_pre"`` appears only there.
    - ``pre["q"]`` is read once, as ``name = pre["q"].view(-1 or pre["q_lora"].shape[0],
      heads, self.qk_head_dim)``, a statement directly in ``if self.q_lora_rank is not
      None and pre is not None`` (or ``if pre is not None``), itself a statement of the
      function body.
    - Before that ``if``: only imports, a docstring, the two assignments above and names
      set to constants. In its branch before the view: only names the head count reads.
      After the view: only name assignments (in ifs too), never using ``name``.
    - The statement right after the ``if`` chain assigns or returns
      ``self._split_q_nope_pe(name, ...)``; every later use of ``name`` is such a call.
      What the split's outputs go through afterwards is not checked.
    - No other way to bind a name (``:=``, ``match``, lambdas, nested functions or
      classes, comprehensions, ``global`` / ``nonlocal``, ``except ... as``, ``import
      ... as``, an import of the names above), no rebinding of ``self``, no reflective
      call (``setattr``, ``exec``, ``vars``, ...), and no ``assert``, ``raise`` or
      ``try``."""
    import ast

    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    # no construct that binds or shadows a name where a Name store would not show it
    hidden = (ast.NamedExpr, ast.Match, ast.Lambda, ast.ClassDef, ast.Global)
    hidden += (ast.Nonlocal, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    # nor a statement that stops the step on one path (SGLang's handoff has none)
    hidden += (ast.Assert, ast.Raise, ast.Try)
    # nor reflection that writes attributes or names out of sight (setattr, exec, ...)
    reflective = {"setattr", "delattr", "exec", "eval", "compile", "vars", "globals"}
    reflective |= {"locals", "__import__", "__setattr__", "__delattr__"}
    if any(
        isinstance(n, hidden)
        or (
            isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) in reflective
        )
        or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn)
        or (isinstance(n, ast.ExceptHandler) and n.name is not None)
        or (
            isinstance(n, (ast.Import, ast.ImportFrom))
            and any(x.asname is not None or x.name == "*" for x in n.names)
        )
        for n in ast.walk(fn)
    ):
        return None
    imported = {
        x.name.split(".")[0]
        for n in ast.walk(fn)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for x in n.names
    }
    # the handoff dict is named once, where pre is read from it
    if (
        sum(
            isinstance(n, ast.Constant) and n.value == "_mono_pre" for n in ast.walk(fn)
        )
        != 1
        or {"self", "pre", "forward_batch", "q_replicate_active"} & imported
    ):
        return None
    a = fn.args
    params = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    # the predicate reads the step's own batch: a parameter never rebound
    if {"q_replicate_active", "pre"} & params or "forward_batch" not in params:
        return None
    if any(
        isinstance(n, ast.Name)
        and n.id in ("forward_batch", "self")
        and not isinstance(n.ctx, ast.Load)
        for n in ast.walk(fn)
    ):
        return None

    def dump(e: str) -> str:
        return ast.dump(ast.parse(e, mode="eval").body)

    parent = {id(c): n for n in ast.walk(fn) for c in ast.iter_child_nodes(n)}
    stores = {"pre": [], "q_replicate_active": []}
    q_reads = []
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Name) and n.id in stores):
            continue
        if not isinstance(n.ctx, ast.Load):
            stores[n.id].append(n)
            continue
        if n.id != "pre":
            continue
        up = parent.get(id(n))
        if isinstance(up, ast.Compare) and ast.dump(up) in (
            dump("pre is not None"),
            dump("pre is None"),
        ):
            continue
        if (
            isinstance(up, ast.Subscript)
            and isinstance(up.ctx, ast.Load)
            and isinstance(up.slice, ast.Constant)
            and isinstance(up.slice.value, str)
        ):
            if up.slice.value == "q":
                q_reads.append(up)
            continue
        call = parent.get(id(up))
        if not (
            isinstance(up, ast.Attribute)
            and up.attr == "get"
            and isinstance(call, ast.Call)
            and call.func is up
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and call.args[0].value != "q"
        ):
            return None
    if len(q_reads) != 1 or any(len(v) != 1 for v in stores.values()):
        return None
    view = parent.get(id(parent.get(id(q_reads[0]))))
    if not (
        isinstance(view, ast.Call)
        and isinstance(view.func, ast.Attribute)
        and view.func.attr == "view"
        and view.func.value is q_reads[0]
        and len(view.args) == 3
        and not view.keywords
        and ast.dump(view.args[0]) in (dump("-1"), dump('pre["q_lora"].shape[0]'))
        and ast.dump(view.args[2]) == dump("self.qk_head_dim")
    ):
        return None
    top = {}
    for s in fn.body:
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and s.lineno < view.lineno
        ):
            top[s.targets[0].id] = s
    pred = top.get("q_replicate_active")
    if (
        pred is None
        or pred.targets[0] is not stores["q_replicate_active"][0]
        or top.get("pre") is None
        or top["pre"].targets[0] is not stores["pre"][0]
        or ast.dump(top["pre"].value) != dump('self.__dict__.get("_mono_pre")')
        or not isinstance(pred.value, ast.BoolOp)
        or not isinstance(pred.value.op, ast.And)
        or sorted(ast.dump(v) for v in pred.value.values)
        != sorted(dump(e) for e in _QREP_PREDICATE)
    ):
        return None
    stmt = parent.get(id(view))
    guard = parent.get(id(stmt))
    if not (
        isinstance(stmt, ast.Assign)
        and stmt.value is view
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(guard, ast.If)
        and stmt in guard.body
        and guard in fn.body
    ):
        return None
    # before the guard only imports, a docstring and the assignments below: no way out early
    for s in fn.body[: fn.body.index(guard)]:
        doc = isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
        # a name set to a constant (SGLang: `q_lora = None`), or pre / the predicate as
        # matched above; nothing that could change what the head count reads
        named = (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and (isinstance(s.value, ast.Constant) or s in (pred, top["pre"]))
        )
        if not (doc or named or isinstance(s, (ast.Import, ast.ImportFrom))):
            return None
    # SGLang's guard (or the bare ``pre`` test): its branch runs in both q-replicate states
    if ast.dump(guard.test) not in (
        dump("self.q_lora_rank is not None and pre is not None"),
        dump("pre is not None"),
    ):
        return None
    at = guard.body.index(stmt)
    name = stmt.targets[0].id
    if name in imported:
        return None

    def plain(
        s,
    ) -> bool:  # name assignments, and ifs of them (SGLang: the indexer's top-k)
        if isinstance(s, ast.If):
            return all(plain(x) for x in (*s.body, *s.orelse))
        return isinstance(s, ast.Assign) and all(
            isinstance(t, ast.Name) for t in s.targets
        )

    exits = (ast.Return, ast.Break, ast.Continue, ast.Yield, ast.YieldFrom, ast.Await)
    rest = guard.body[at + 1 :]
    if not all(plain(s) for s in rest) or any(
        (isinstance(n, ast.Name) and n.id == name) or isinstance(n, exits)
        for s in rest
        for n in ast.walk(s)
    ):
        return None  # the viewed q is used again, or the branch does more than assign
    # the next statement after the if chain is `... = self._split_q_nope_pe(name, ...)` or
    # `return self._split_q_nope_pe(name, ...)`: the split runs whenever the view did
    nxt = fn.body.index(guard) + 1
    after = fn.body[nxt] if nxt < len(fn.body) else None
    head = after.value if isinstance(after, (ast.Assign, ast.Return)) else None
    if not (
        isinstance(head, ast.Call)
        and ast.dump(head.func) == dump("self._split_q_nope_pe")
        and head.args
        and isinstance(head.args[0], ast.Name)
        and head.args[0].id == name
    ):
        return None
    split = 0
    for n in ast.walk(fn):
        # past the if / elif chain the viewed q only goes to SGLang's split, as is
        if isinstance(n, ast.Name) and n.id == name and n.lineno > guard.end_lineno:
            call = parent.get(id(n))
            if not (
                isinstance(n.ctx, ast.Load)
                and isinstance(call, ast.Call)
                and call.args
                and call.args[0] is n
                and ast.dump(call.func) == dump("self._split_q_nope_pe")
            ):
                return None
            split += 1
    if not split:
        return None
    assigned: dict = {}
    for s in guard.body[:at]:
        # before the view only head-count names (SGLang: `num_q_heads = ...`)
        if not (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and all(_head_count(s.value, {}, qrep=q) is not None for q in (False, True))
        ):
            return None
        assigned[s.targets[0].id] = s.value
    return view.args[1], assigned


def _sglang_mono_pre_caps(attn) -> tuple[bool, bool]:
    """Return ``(supports_local, supports_qrep)`` for SGLang's ``_mono_pre`` handoff in
    ``forward_absorb_rocm_prepare``: whether it views ``pre["q"]`` with the local heads
    without q-replicate, and whether that view takes ``num_local_heads * dcp_size`` heads
    under it (sgl-project/sglang#53130), read from the view's head expression."""
    cached = getattr(attn, "_k31_mono_pre_caps", None)
    if cached is not None:
        return cached
    fn = getattr(attn, "forward_absorb_rocm_prepare", None)
    local_ok, qrep_ok = False, False
    if fn is not None:
        try:
            import ast
            import inspect
            import textwrap

            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            if any(
                isinstance(n, ast.Constant) and n.value == "_mono_pre"
                for n in ast.walk(tree)
            ):
                view = _mono_pre_q_view(tree)
                if view is not None:
                    head, assigned = view
                    local_ok = _head_count(head, assigned, qrep=False) == (1, 1, 0)
                    qrep_ok = local_ok and (
                        _head_count(head, assigned, qrep=True) == (1, 1, 1)
                    )
        except (OSError, TypeError, SyntaxError):
            local_ok, qrep_ok = False, False
    caps = (local_ok, qrep_ok)
    attn._k31_mono_pre_caps = caps
    return caps


def _sglang_mono_pre_supports_local(attn) -> bool:
    return _sglang_mono_pre_caps(attn)[0]


def _sglang_mono_pre_supports_qrep(attn) -> bool:
    return _sglang_mono_pre_caps(attn)[1]


def _mla_max_rows(qb_rows: int) -> int:
    from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

    env_cap = os.environ.get("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS")
    if env_cap is not None:
        max_rows = int(env_cap)
        if not 1 <= max_rows <= mp.MAX_ROWS:
            raise ValueError(
                f"SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS must be in [1, {mp.MAX_ROWS}]"
            )
        return max_rows
    return mp.MAX_ROWS_QREP if qb_rows == mp.QB_QREP else mp.MAX_ROWS_LOCAL


def _mla_verify_qb(layer, par) -> int | None:
    """Target-verify ``q_b`` output width (``QB_LOCAL`` or ``QB_QREP``) that
    ``mla_pre_wide`` can execute for ``layer``, or ``None`` if it must fall back."""
    from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

    attn = layer.self_attn
    dcp_size = getattr(attn, "dcp_size", 1)
    qrep_verify = bool(
        getattr(par, "dcp_enabled", False)
        and getattr(par, "dcp_replicate_q_proj", False)
        and dcp_size > 1
        and not getattr(attn, "use_deep_gemm_bmm", False)
    )
    if qrep_verify:
        if (
            _sglang_mono_pre_supports_qrep(attn)
            and getattr(attn, "num_local_heads", 0)
            * dcp_size
            * getattr(attn, "qk_head_dim", 0)
            == mp.QB_QREP
        ):
            return mp.QB_QREP
        return None
    if (
        _sglang_mono_pre_supports_local(attn)
        and getattr(attn, "num_local_heads", 0) * getattr(attn, "qk_head_dim", 0)
        == mp.QB_LOCAL
    ):
        return mp.QB_LOCAL
    return None


def _valid_qb_weight(w, qb_rows: int) -> bool:
    from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

    return (
        isinstance(w, torch.Tensor)
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and tuple(w.shape) == (qb_rows, mp.QL)
    )


def _mla_pre_fusable(layer) -> bool:
    """Whether a DSA layer's attention projection front can run in ``mla_pre_wide``."""
    if os.environ.get("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE", "1") != "1":
        return False
    from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

    attn = layer.self_attn
    idx = getattr(attn, "indexer", None)
    fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    if (
        idx is None
        or fused is None
        or not _sglang_mono_pre_supports_local(attn)
        or not getattr(attn, "use_dsa", False)
        or not getattr(attn, "use_output_gate", False)
        or getattr(attn, "q_lora_rank", None) != mp.QL
        or getattr(attn, "kv_lora_rank", None) != mp.KV
        or getattr(attn, "qk_rope_head_dim", None) != mp.ROPE
        or getattr(attn, "qk_head_dim", None) != 192
    ):
        return False
    mods = (
        (fused, (mp.LATW, H)),
        (getattr(attn, "g_proj", None), (mp.GATE, H)),
        (getattr(idx, "wk", None), (mp.IK, H)),
        (getattr(idx, "weights_proj", None), (mp.IW, H)),
        (getattr(idx, "wq_b", None), (mp.IQ, mp.QL)),
        (getattr(attn, "q_b_proj", None), (mp.QB_LOCAL, mp.QL)),
    )
    for mod, shp in mods:
        w = getattr(mod, "weight", None)
        if (
            not isinstance(w, torch.Tensor)
            or getattr(mod, "bias", None) is not None
            or w.dtype != torch.bfloat16
            or tuple(w.shape) != shp
            or not w.is_contiguous()
        ):
            return False
    for ln_name, dim in (("q_a_layernorm", mp.QL), ("kv_a_layernorm", mp.KV)):
        ln = getattr(attn, ln_name, None)
        nw = getattr(ln, "weight", None)
        if (
            not isinstance(nw, torch.Tensor)
            or nw.dtype != torch.bfloat16
            or tuple(nw.shape) != (dim,)
            or not nw.is_contiguous()
            or getattr(ln, "variance_epsilon", None) is None
        ):
            return False
    return True


class _DSALayer:
    """A fusable DSA layer's weights as ``dsa_post`` (and optionally ``mla_pre_wide``) reads them."""

    def __init__(self, backbone, i, rank):
        layer = backbone.layers[i]
        attn, moe = layer.self_attn, layer.mlp
        self.i, self.layer, self.moe = i, layer, moe
        self.nvb = layer.prev_valid_blocks
        self.eps = layer.post_attention_layernorm.variance_epsilon
        self.mla_pre = _mla_pre_fusable(layer)
        self.mla_pre_qb = None
        self.q_eps = (
            float(attn.q_a_layernorm.variance_epsilon) if self.mla_pre else 1e-6
        )
        self.kv_eps = (
            float(attn.kv_a_layernorm.variance_epsilon) if self.mla_pre else 1e-6
        )
        self.w = dict(
            wo=attn.o_proj.weight,
            cw_m=(
                layer.mlp_res_norm.weight.float()
                * layer.mlp_res_proj.weight.view(H).float()
            ).contiguous(),
            w_ln=layer.post_attention_layernorm.weight,
            bias=moe.gate.e_score_correction_bias.float().contiguous(),
            w_sd=moe.shared_experts.down_proj.weight,
        )
        self.w["wf"], self.front_rows = _front_weight(moe, rank)
        self.shared = {t.untyped_storage().data_ptr() for t in layer.parameters()}


class Wide:
    def __init__(self, model, rank, group, device, L=8):
        from sgl_amd_mono.common import guard

        assert L in SUPPORTED_L, L
        self.L = L
        lm = getattr(model, "language_model", model)
        self.bb = bb = lm.model
        self.rank, self.group, self.device = rank, group, device
        idx = [
            i for i in range(bb.start_layer, bb.end_layer) if _is_covered(bb.layers[i])
        ]
        cov = set(idx)
        # standard layers whose MoE tail runs in the next run's first launch (t1 on):
        # SGLang runs them up to the routed / shared partials only
        self.entry = {
            i - 1
            for i in idx
            if (i - 1) not in cov
            and i - 1 >= bb.start_layer
            and _tail_fusable(bb.layers[i - 1])
        }
        local_dsa = {
            i
            for i in range(bb.start_layer, bb.end_layer)
            if i not in cov and _dsa_fusable(bb.layers[i])
        }
        if group is not None and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, local_dsa, group=group)
            dsa_idx = sorted(set.intersection(*gathered))
        else:
            dsa_idx = sorted(local_dsa)
        a0 = torch.cuda.memory_allocated()
        self.layers = {i: _Layer(bb, i, rank, cov | self.entry) for i in idx}
        self.dsa_layers = {i: _DSALayer(bb, i, rank) for i in dsa_idx}
        a1 = torch.cuda.memory_allocated()
        for i in dsa_idx:
            if i not in self.entry and (i + 1) in self.layers:
                raise ValueError(f"non-entry DSA layer {i} precedes KDA layer {i + 1}")
        self.slots = {
            i: n for n, i in enumerate(idx)
        }  # a KDA launch's slot in the step
        self.slots_dsa = {
            i: n for n, i in enumerate(dsa_idx)
        }  # a DSA launch's slot in its own peer ring
        if any(pk.mla_pre for pk in self.dsa_layers.values()):
            from sglang.srt.runtime_context import get_parallel

            par = get_parallel()
            for pk in self.dsa_layers.values():
                pk.mla_pre_qb = _mla_verify_qb(pk.layer, par) if pk.mla_pre else None
        caps = set(bb.dspark_layers_to_capture or ())
        self.peers, self.bufs, self.half = {}, {}, {}
        self.peers_dsa, self.half_dsa = {}, {}
        self.buckets = buckets(L)
        self.tabs = {}  # (bucket, layer) -> KDA table: every pointer in it is fixed once built
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        # long enough for a capture variant's first compile on one rank while the
        # others already wait (every other variant is compiled by warmup())
        ticks = guard.deadline_ticks(
            float(os.environ.get("SGLANG_AMD_MONO_K31_WIDE_DEADLINE_S", "900"))
        )
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.cols = slice(rank * SL, (rank + 1) * SL)
        self.kda_ready = False
        n_mla = sum(1 for pk in self.dsa_layers.values() if pk.mla_pre_qb is not None)
        _say(
            f"{len(idx)} covered layers ({idx[0]} .. {idx[-1]}), {len(dsa_idx)} dsa_post layers"
            + (f" ({dsa_idx[0]} .. {dsa_idx[-1]})" if dsa_idx else "")
            + f", {n_mla} mla_pre_wide layers"
            + f", {L} tokens a request, buckets {self.buckets}, captures {sorted(caps)}, "
            f"block-write {[i for i in idx if self.layers[i].write]}, tails into a run's first launch "
            f"{sorted(self.entry)}"
        )
        own, seen = {}, set()
        for ly in (*self.layers.values(), *self.dsa_layers.values()):
            for k, t in (*ly.w.items(), *getattr(ly, "kda_w", {}).items()):
                st = t.untyped_storage()
                if st.data_ptr() in ly.shared or st.data_ptr() in seen:
                    continue
                seen.add(st.data_ptr())
                own[k] = own.get(k, 0) + st.nbytes()
        _say(f"layers: torch allocated {(a1 - a0) / 2**30:.2f} GiB")
        _say(
            "weight copies: "
            + ", ".join(
                f"{k} {v / 2**20:.0f} MiB"
                for k, v in sorted(own.items(), key=lambda x: -x[1])
            )
        )

    # ------------------------------------------------------------ per bucket
    def _mla_bucket_specs(self, b: int) -> dict[int, int]:
        """Map eligible ``qb_rows -> m_mla`` launch width for wide bucket ``b``.

        A bucket can use the MLA launch only when the smallest served width it
        holds (``wide_widths`` at this ``L``, past the previous bucket) does not
        exceed ``max_rows``. The first bucket starts at the served lower cut, not at
        0: with a bucket override or a small cap, no step reaches a launch below it.
        """
        buckets = getattr(self, "buckets", ())
        if b in buckets:
            idx = buckets.index(b)
            prev_b = buckets[idx - 1] if idx > 0 else 0
        else:
            prev_b = b - 1
        reach = min(
            (w for w in wide_widths() if w % self.L == 0 and prev_b < w <= b),
            default=None,
        )
        specs: dict[int, int] = {}
        if reach is None:
            return specs
        for pk in self.dsa_layers.values():
            qb = pk.mla_pre_qb
            if qb is None:
                continue
            lim = _mla_max_rows(qb)
            if reach <= lim:
                specs[qb] = min(b, lim)
        return specs

    def _cfg(self, b):
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay
        from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

        return lay.LayerCfg(
            *(Gemm(*(int(v) for v in c.split(":"))) for c in cfg_rows(b, self.L))
        )

    def _variant(self, pk, b, L, captures):
        cap = (pk.i - 1) in captures
        return dict(
            nvb_a=pk.nvb,
            nvb_m=pk.nvb + int(pk.write),
            write_a=pk.write,
            capture=cap,
            t1=pk.t1,
            restart=pk.write,
        )

    def _ensure(self, b, L, captures):
        """Peer buffer and activation buffers of bucket b (collective on first use:
        every rank reaches it on the same step)."""
        if b in self.bufs:
            return self.bufs[b]
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay

        cfg = self._cfg(b)
        half, half_dsa, scratch, scratch_dsa = 0, 0, 0, 0
        for pk in self.layers.values():
            v = self._variant(pk, b, L, captures)
            for cap in (
                False,
                True,
            ):  # sized for a capture launch too: the captured layers come later
                pl = lay.plan(b, cfg, L, v["nvb_a"], v["nvb_m"], NF, cap)[0]
                half, scratch = (
                    max(half, pl.peer_offsets[1]),
                    max(scratch, pl.scratch_bytes()),
                )
        for pk in self.dsa_layers.values():
            pl = dp.plan(b, cfg.oproj, cfg.front, cfg.sdown, NF, pk.nvb)[0]
            half_dsa = max(half_dsa, pl.peer_offsets[1])
            scratch_dsa = max(scratch_dsa, pl.scratch_bytes())
        from sgl_amd_mono.common import guard

        self.half[b] = (
            half  # one layout for every variant: same halves, the guard's flags after both
        )
        need = 2 * half + guard.FLAG_BYTES
        self.peers[b] = PeerBuffer(need, self.group, self.rank, TP, self.device)
        self.peers[b].bytes.zero_()
        if self.dsa_layers:
            # Dedicated peer ring for dsa_post so DSA mailbox flags never alias KDA raw BF16 payloads.
            self.half_dsa[b] = half_dsa
            self.peers_dsa[b] = PeerBuffer(
                2 * half_dsa + guard.FLAG_BYTES, self.group, self.rank, TP, self.device
            )
            self.peers_dsa[b].bytes.zero_()
        dev, bf = self.device, torch.bfloat16

        def z(*shape, dt=bf):
            return torch.zeros(*shape, dtype=dt, device=dev)

        from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1

        # the routed latent and shared partials of a ping share one buffer: the column exit
        # reduces the pair in one collective, as SGLang's fused front does
        flat = [z(b * (LAT + H)), z(b * (LAT + H))]
        # the launch's hand-off mailboxes uncached (as the K3 launches): a consumer reads a line at its home memory,
        # not through the producer XCD's L2. A one-rank peer buffer, never freed while the process lives.
        scr_pb = PeerBuffer(scratch, self.group, self.rank, 1, dev)
        scr = scr_pb.bytes
        scr.zero_()
        scr_dsa_pb = (
            PeerBuffer(max(scratch_dsa, 256), self.group, self.rank, 1, dev)
            if self.dsa_layers
            else None
        )
        scr_dsa = scr_dsa_pb.bytes if scr_dsa_pb is not None else None
        if scr_dsa is not None:
            scr_dsa.zero_()
        bufs = dict(
            cfg=cfg,
            mid=z(b, SL),
            bank=z(b, 8, SL),
            pout=[z(b, H), z(b, H)],
            lat=z(b, LAT),
            flat=flat,
            lat_p=[f[: b * LAT].view(b, LAT) for f in flat],
            sh=[f[b * LAT :].view(b, H) for f in flat],
            ids=z(b, 16, dt=torch.int32),
            w=z(b, 16, dt=torch.float32),
            logits=z(b, E, dt=torch.float32),
            cap=z(b, H),
            core=z(b, k1.PROJ),
            ip_buf=z(b, k1.NPROJ),
            acc_u=z(b, SL),
            acc_o=z(b * H),
            cnt=torch.zeros(lay.counters(b, cfg, NF), dtype=torch.int32, device=dev),
            cnt_dsa=(
                torch.zeros(
                    dp.counters(b, cfg.oproj, cfg.front, NF),
                    dtype=torch.int32,
                    device=dev,
                )
                if self.dsa_layers
                else None
            ),
            scratch=scr,
            scratch_pb=scr_pb,
            scratch_dsa=scr_dsa,
            scratch_dsa_pb=scr_dsa_pb,
            st_idx=torch.full((b // L, L), -1, dtype=torch.int32, device=dev),
            num_acc=torch.ones(b // L, dtype=torch.int32, device=dev),
            win_row=torch.full((b // L,), -1, dtype=torch.int32, device=dev),
        )
        # the launch's sort + quant (sort builds); a placeholder pointer otherwise
        from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

        kind = _launch_sort(b)
        if kind:
            bufs["sort"] = torch.zeros(
                mf.sort_layout(b, kind)["nbytes"], dtype=torch.uint8, device=dev
            )
            mf.sort_init(bufs["sort"], b, kind)
            bufs["sortv"] = _sort_views(bufs["sort"], b, kind)
        else:
            bufs["sort"] = torch.zeros(256, dtype=torch.uint8, device=dev)
            bufs["sortv"] = None
        from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

        mla_specs = self._mla_bucket_specs(b)
        if mla_specs:
            max_m = max(mla_specs.values())
            max_qb = max(mla_specs)
            bufs["mla_specs"] = mla_specs
            mla_state = {}
            for qb, m_mla in mla_specs.items():
                scr_pb = PeerBuffer(
                    mp.scratch_layout(m_mla, qb)["nbytes"], None, 0, 1, dev
                )
                scr_pb.bytes.zero_()
                mla_state[qb] = (
                    scr_pb,
                    scr_pb.bytes,
                    torch.zeros(mp.counters(qb), dtype=torch.int32, device=dev),
                )
            bufs["mla_state"] = mla_state
            bufs["mla_x"] = z(max_m, H)
            bufs["mla_out"] = dict(
                q=z(max_m * max_qb),
                latent=z(max_m, mp.KV + mp.ROPE),
                k_nope=z(max_m, mp.KV),
                q_lora=z(max_m, mp.QL),
                gate=z(max_m, mp.GATE),
                idx_q=z(max_m, mp.IQ),
                idx_k=z(max_m, mp.IK),
                idx_w=z(max_m, mp.IW),
            )
        self.bufs[b] = bufs
        _say(f"bucket {b}: peer {need / 1e6:.1f} MB, scratch {scratch / 1e6:.1f} MB")
        return bufs

    def _launch_args(self, pk, b, bufs, tab, ping, slot, prefix=None):
        """``prefix``: SGLang's prefix [rows, H] a run's first launch reads (else the
        previous launch's ``pout``); the launch reads / writes this rank's columns."""
        w, zt = pk.w, bufs["mid"]
        prev, cur = ping, 1 - ping
        col0 = 2 * self.cols.start
        pre = bufs["pout"][prev] if prefix is None else prefix
        return (
            bufs["lat_p"][prev].data_ptr(),
            bufs["sh"][prev].data_ptr(),
            w.get("w_up", zt).data_ptr(),
            pre.data_ptr() + col0,
            bufs["mid"].data_ptr(),
            bufs["pout"][cur].data_ptr() + col0,
            bufs["bank"].data_ptr(),
            w["cw_a"].data_ptr(),
            w["w_in"].data_ptr(),
            w["w_ip"].data_ptr(),
            bufs["ip_buf"].data_ptr(),
            bufs["acc_u"].data_ptr(),
            tab.data_ptr(),
            bufs["core"].data_ptr(),
            w["wo"].data_ptr(),
            w["cw_m"].data_ptr(),
            w["w_ln"].data_ptr(),
            w["wf"].data_ptr(),
            bufs["acc_o"].data_ptr(),
            w["bias"].data_ptr(),
            bufs["lat"].data_ptr(),
            bufs["ids"].data_ptr(),
            bufs["w"].data_ptr(),
            w["w_sd"].data_ptr(),
            bufs["sh"][cur].data_ptr(),
            bufs["cap"].data_ptr(),
            H,
            bufs["logits"].data_ptr(),
            bufs["cnt"].data_ptr(),
            bufs["scratch"].data_ptr(),
            self.peers[b].addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            slot,
            pre.shape[0] * H * 2 - col0,
            bufs["sort"].data_ptr(),
        )

    def _build_launch(self, pk, b, L, v, cfg):
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay

        return lay.kda_layer(
            b,
            cfg,
            L,
            NF,
            nvb_a=v["nvb_a"],
            nvb_m=v["nvb_m"],
            write_a=v["write_a"],
            nbk=8,
            launches=len(self.layers),
            eps=pk.eps,
            eps_n=pk.onorm_eps,
            capture=v["capture"],
            t1=v["t1"],
            restart=v["restart"],
            half_bytes=self.half[b],
            front_rows=pk.front_rows,
            up_norm=pk.up_norm,
            sort=_launch_sort(b),
        )[0]

    def _dsa_launch_args(self, pk, b, bufs, core, prefix, cur, slot):
        w = pk.w
        col0 = 2 * self.cols.start
        rows = core.shape[0]
        assert core.shape == (rows, KO), (core.shape, rows, KO)
        assert bufs["core"].shape == (b, KO), bufs["core"].shape
        assert bufs["pout"][cur].shape == (b, H), bufs["pout"][cur].shape
        assert core.dtype == torch.bfloat16 and prefix.dtype == torch.bfloat16, (
            core.dtype,
            prefix.dtype,
        )
        if rows == b and core.stride() == (KO, 1):
            core_ptr = core.data_ptr()
        else:
            bufs["core"][:rows].copy_(core)
            if rows < b:
                bufs["core"][rows:].zero_()
            core_ptr = bufs["core"].data_ptr()
        if prefix.stride() == (H, 1):
            pre = prefix
            pfx_bytes = pre.shape[0] * H * 2 - col0
        else:
            # attn_boundary.prepare already materialized the previous layer's stream on this stream.
            pre = bufs["pout"][1 - cur]
            pre[:rows, self.cols] = prefix[:, self.cols]
            pfx_bytes = rows * H * 2 - col0
        return (
            core_ptr,
            w["wo"].data_ptr(),
            pre.data_ptr() + col0,
            bufs["pout"][cur].data_ptr() + col0,
            bufs["bank"].data_ptr(),
            w["cw_m"].data_ptr(),
            w["w_ln"].data_ptr(),
            w["wf"].data_ptr(),
            bufs["acc_o"].data_ptr(),
            w["bias"].data_ptr(),
            bufs["lat"].data_ptr(),
            bufs["ids"].data_ptr(),
            bufs["w"].data_ptr(),
            w["w_sd"].data_ptr(),
            bufs["sh"][cur].data_ptr(),
            bufs["lat_p"][cur].data_ptr(),
            bufs["logits"].data_ptr(),
            bufs["cnt_dsa"].data_ptr(),
            bufs["scratch_dsa"].data_ptr(),
            self.peers_dsa[b].addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            slot,
            pfx_bytes,
            bufs["sort"].data_ptr(),
        )

    def _dsa_variant(self, pk, b):
        return dict(
            nvb_m=pk.nvb,
            nbk=8,
            launches=len(self.dsa_layers),
            eps=pk.eps,
            half_bytes=self.half_dsa[b],
            front_rows=pk.front_rows,
            sort=_launch_sort(b),
        )

    def _build_dsa_launch(self, pk, b, cfg):
        from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp

        return dp.dsa_post(
            b,
            cfg.oproj,
            cfg.front,
            cfg.sdown,
            NF,
            **self._dsa_variant(pk, b),
        )[0]

    def warmup(self, buckets=None):
        """Compile every non-capture variant of every bucket by launching it once on
        the real weights and dummy activations, with no KDA state (every slot -1).
        Every rank runs the same launches in the same order, before any step."""
        import time

        L = self.L
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        t0 = time.time()
        dev, bf, f32 = self.device, torch.bfloat16, torch.float32
        dummy = dict(
            conv_state=torch.zeros(
                1, k1.CONV_W - 1, 3 * k1.PROJ, dtype=bf, device=dev
            ).mT,
            conv_win=torch.zeros(
                1, L, k1.CONV_W - 1, 3 * k1.PROJ, dtype=bf, device=dev
            ),
            rstate=torch.zeros(1, k1.NH, k1.HD, k1.HD, dtype=f32, device=dev),
            ring=(
                torch.zeros(1, k1.NH, 16, k1.HD, dtype=bf, device=dev),
                torch.zeros(1, k1.NH, 16, k1.HD, dtype=bf, device=dev),
                torch.zeros(1, k1.NH, 16, k1.HD, dtype=f32, device=dev),
                torch.zeros(1, k1.NH, 16, dtype=f32, device=dev),
            ),
        )
        n = 0
        for b in buckets or self.buckets:
            bufs = self._ensure(b, L, set())
            st_idx = torch.full((b // L, L), -1, dtype=torch.int32, device=dev)
            num_acc = torch.ones(b // L, dtype=torch.int32, device=dev)
            win_row = torch.full((b // L,), -1, dtype=torch.int32, device=dev)
            seen = set()
            for pk in self.layers.values():
                v = self._variant(pk, b, L, set())
                key = tuple(sorted(v.items()))
                if key in seen:
                    continue
                seen.add(key)
                tab = pa.kda_table(
                    dev,
                    **pk.kda_w,
                    **dummy,
                    st_idx=st_idx,
                    num_acc=num_acc,
                    win_row=win_row,
                    core_out=bufs["core"],
                )
                self.epoch[0] += 1
                _run_compiled(
                    self._build_launch(pk=pk, b=b, L=L, v=v, cfg=bufs["cfg"]),
                    *self._launch_args(pk, b, bufs, tab, 0, self.slots[pk.i]),
                    torch.cuda.current_stream(),
                )
                n += 1
            seen_dsa, need_dsa = set(), set()
            for pk in self.dsa_layers.values():
                bkey = (pk.nvb, pk.front_rows, pk.eps)
                if bkey not in seen_dsa:
                    seen_dsa.add(bkey)
                    need_dsa.add(pk.i)
            if self.group is not None and torch.distributed.is_initialized():
                gathered = [None] * torch.distributed.get_world_size(self.group)
                torch.distributed.all_gather_object(
                    gathered, need_dsa, group=self.group
                )
                warm_ids = sorted(set().union(*gathered))
            else:
                warm_ids = sorted(need_dsa)
            for idx in warm_ids:
                pk = self.dsa_layers[idx]
                self.epoch[0] += 1
                _run_compiled(
                    self._build_dsa_launch(pk, b, bufs["cfg"]),
                    *self._dsa_launch_args(
                        pk,
                        b,
                        bufs,
                        bufs["core"],
                        bufs["pout"][0],
                        1,
                        self.slots_dsa[pk.i],
                    ),
                    torch.cuda.current_stream(),
                )
                n += 1
            from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

            if "mla_out" in bufs:
                # Keep self.epoch untouched during rank-local MLA warmup so all
                # TP ranks stay aligned for subsequent collective launches; zero
                # this bucket's private scratch afterwards to clear warmup tags.
                mla_specs = bufs["mla_specs"]
                mla_state = bufs["mla_state"]
                warm_ep = self.epoch.clone()
                warm_ep[0] = 0
                dummy_qrep = None
                seen_mla = set()
                warmed_qbs = set()
                for pk in self.dsa_layers.values():
                    qb_rows = pk.mla_pre_qb
                    if qb_rows not in mla_specs:
                        continue
                    m_mla = mla_specs[qb_rows]
                    mkey = (m_mla, qb_rows, pk.q_eps, pk.kv_eps)
                    if mkey in seen_mla:
                        continue
                    seen_mla.add(mkey)
                    attn = pk.layer.self_attn
                    idx_m = attn.indexer
                    if qb_rows == mp.QB_QREP:
                        cand = getattr(attn, "q_b_proj_qrep_weight", None)
                        if _valid_qb_weight(cand, mp.QB_QREP):
                            w_qb = cand
                        else:
                            if dummy_qrep is None:
                                dummy_qrep = torch.zeros(
                                    mp.QB_QREP, mp.QL, dtype=bf, device=dev
                                )
                            w_qb = dummy_qrep
                    else:
                        w_qb = attn.q_b_proj.weight
                    mout = bufs["mla_out"]
                    _, scr_mla, cnt_mla = mla_state[qb_rows]
                    warm_ep[0] += 1
                    _run_compiled(
                        mp.mla_pre_wide(
                            m_mla,
                            qb_rows,
                            launches=len(self.dsa_layers),
                            q_eps=pk.q_eps,
                            kv_eps=pk.kv_eps,
                        )[0],
                        bufs["mla_x"].data_ptr(),
                        attn.fused_qkv_a_proj_with_mqa.weight.data_ptr(),
                        attn.g_proj.weight.data_ptr(),
                        idx_m.wk.weight.data_ptr(),
                        idx_m.weights_proj.weight.data_ptr(),
                        attn.q_a_layernorm.weight.data_ptr(),
                        attn.kv_a_layernorm.weight.data_ptr(),
                        idx_m.wq_b.weight.data_ptr(),
                        w_qb.data_ptr(),
                        mout["q"].data_ptr(),
                        mout["latent"].data_ptr(),
                        mout["k_nope"].data_ptr(),
                        mout["q_lora"].data_ptr(),
                        mout["gate"].data_ptr(),
                        mout["idx_q"].data_ptr(),
                        mout["idx_k"].data_ptr(),
                        mout["idx_w"].data_ptr(),
                        cnt_mla.data_ptr(),
                        scr_mla.data_ptr(),
                        warm_ep.data_ptr(),
                        self.slots_dsa[pk.i],
                        torch.cuda.current_stream(),
                    )
                    warmed_qbs.add(qb_rows)
                    n += 1
                for qb_rows in warmed_qbs:
                    mla_state[qb_rows][1].zero_()
            torch.cuda.synchronize()
        _say(f"warmup: {n} launches compiled and run in {time.time() - t0:.0f} s")

    # ------------------------------------------------------------ KDA state
    def _kda_step(self, fb, R, L, bufs):
        """The step's KDA state (K1's prepare, for R requests). Device work only:
        a graph replays it."""
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend().linear_attn_backend
        pool, meta = backend.req_to_token_pool, backend.forward_metadata
        cache = pool.mamba_pool.mamba_cache
        assert getattr(cache, "replayssm_rawv", None) is not None, (
            "the wide launch runs on the ReplaySSM ring"
        )
        if not self.kda_ready:
            self.kda_ready = True
            for pk in self.layers.values():
                pk.mamba_idx = backend.state_layer(
                    pk.layer.self_attn.attn.layer_id
                ).view_layer_id
        slots = meta.mamba_cache_indices[:R]
        vrows = select_verify_intermediate_state_indices(
            backend.verify_intermediate_state_indices,
            fb.req_pool_indices,
            slots >= 0,
            pool.size,
        )[:R]
        bufs["st_idx"].fill_(-1)
        bufs["st_idx"][:R] = slots.to(torch.int32).view(R, 1)
        bufs["win_row"].fill_(-1)
        bufs["win_row"][:R] = vrows.to(torch.int32)
        return dict(cache=cache)

    def _kda_table(self, pk, ks, b, bufs):
        """The launch reads the conv state from SGLang's cache by slot (``st_idx``)
        and writes the verify windows at ``win_row`` (K1's ``finish`` for R requests)."""
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        tab = self.tabs.get((b, pk.i))
        if tab is None:
            # a host-to-device copy: the eager warmup forwards before a capture build it
            assert not torch.cuda.is_current_stream_capturing(), (
                f"KDA table ({b}, {pk.i}) first built in a capture"
            )
            c, i = ks["cache"], pk.mamba_idx
            conv, win = c.conv[0][i], c.intermediate_conv_window[0][i]
            assert conv.shape[1:] == win.shape[2:] and conv.shape[1] == 3, (
                tuple(conv.shape),
                tuple(win.shape),
            )
            ring = (
                c.replayssm_rawk[i],
                c.replayssm_rawv[i],
                c.replayssm_g[i],
                c.replayssm_beta[i],
            )
            tab = self.tabs[(b, pk.i)] = pa.kda_table(
                self.device,
                **pk.kda_w,
                conv_state=conv.mT,
                rstate=c.temporal[i],
                st_idx=bufs["st_idx"],
                num_acc=bufs["num_acc"],
                core_out=bufs["core"],
                conv_win=win,
                win_row=bufs["win_row"],
                ring=ring,
            )
        return tab

    # ------------------------------------------------------------ the step
    def _tail_front(
        self, layer, hidden, positions, fb, zero_allocator, bufs, rows, ping
    ):
        """A standard layer whose MoE tail the next launch takes: SGLang's layer up to
        the MoE's TP partials (routed latent -> lat_p, shared -> sh) and this rank's
        prefix the tail adds (the launch reads its columns), the next launch's T1 inputs."""
        from sglang.srt.layers.layer_boundary.residual.batch import written_residual

        h = layer.attn_boundary.prepare(hidden, fb, capture_gathered=None)
        h = layer._run_self_attn(h, positions, fb, zero_allocator)
        h = layer.attn_boundary.finish(h, fb)
        x = layer.ffn_boundary.prepare(h, fb)
        prefix = written_residual(fb)
        assert (
            x.shape == (rows, H) and prefix is not None and prefix.shape == (rows, H)
        ), (x.shape, prefix)
        from sglang.srt.models.kimi_k3_amd.moe import _vector_aligned

        moe = layer.mlp
        # as SGLang's fused front: the routed input's bf16 cast rides the shared SiTU launch
        defer = moe._defer_routed_cast_eligible
        gate_up, logits, routed = moe._fused_front(x, cast_routed=not defer)
        if defer and not _vector_aligned(routed):
            defer, routed = False, routed.to(x.dtype)
        sh = bufs["sh"][ping][:rows]
        if defer:
            routed = moe._forward_shared_casting_routed(gate_up, sh, routed)
        else:
            moe._forward_shared(gate_up, sh)
        moe._forward_routed(x, logits, routed, bufs["lat_p"][ping][:rows])
        return prefix

    def _column_exit(self, moe, b, rows, bufs, ping):
        """A run's exit as SGLang's column tail (up to its row cap): one all-reduce of the
        [latent | shared] pair, this rank's stream columns (its prefix columns added
        there), and the stream's all-gather that also stages the next layer's attention
        read. None, before anything runs, when SGLang's kernels do not take the batch."""
        import sglang.kernels.ops.attention.k3_attn_res_read_hip as hip
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        from sglang.srt.layers.attn_residual_hip import gather_covers

        read = moe.next_attn_read
        if (
            read is None
            # _column_tail_and_next_read's own cap: past it, it would decline after the all-reduce
            or rows > read.max_tokens
            or not moe._fused_latent_tail_eligible
        ):
            return None
        prefix = bufs["pout"][ping][:rows]  # this rank's columns written, the others 0
        if not (
            hip.tail_covered(bufs["sh"][ping][:rows], prefix)
            and gather_covers(read.bank.require(), rows, SL, read.writes_block)
        ):
            return None
        red = tensor_model_parallel_all_reduce(bufs["flat"][ping])
        stream = moe._column_tail_and_next_read(
            red[: b * LAT].view(b, LAT)[:rows], red[b * LAT :].view(b, H)[:rows], prefix
        )
        assert stream is not None
        return stream

    def _exit_stream(self, moe, b, rows, bufs, ping):
        """Leave column-sharded state to a full stream for the next standard layer or final norm."""
        stream = self._column_exit(moe, b, rows, bufs, ping)
        if stream is None:
            # this rank's prefix columns ride in its columns of the shared partial, so
            # the tail's all-reduce carries them (in place: no launch reads sh after)
            sh = bufs["sh"][ping][:rows]
            sh[:, self.cols] += bufs["pout"][ping][:rows, self.cols]
            stream = moe._column_split_tail(
                bufs["lat_p"][ping][:rows], sh, None, cols=self.cols
            )
        return stream

    def _run_mla_pre(self, pk, bufs, h, fb):
        """Run ``mla_pre_wide`` for a fusable DSA layer; return ``(attn_pre, idx_pre)``
        or ``None`` if this step should fall back to SGLang's standard projection front."""
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mp

        qb_want = pk.mla_pre_qb
        mla_specs = bufs.get("mla_specs")
        mla_state = bufs.get("mla_state")
        if (
            qb_want is None
            or not mla_specs
            or not mla_state
            or qb_want not in mla_specs
            or qb_want not in mla_state
            or "mla_out" not in bufs
            or h.ndim != 2
            or h.shape[0] == 0
            or h.shape[1] != H
            or h.dtype != torch.bfloat16
            or h.device != bufs["mla_x"].device
        ):
            return None
        m_mla = mla_specs[qb_want]
        rows = h.shape[0]
        if rows > m_mla:
            return None
        attn = pk.layer.self_attn
        q_replicate_active = (
            attn.is_dcp_q_replicate_phase(fb)
            and not attn.use_deep_gemm_bmm
            and attn.w_kc_qrep is not None
            and attn.q_b_proj_qrep_weight is not None
        )
        qb_rows = (
            attn.num_local_heads * attn.dcp_size * attn.qk_head_dim
            if q_replicate_active
            else attn.num_local_heads * attn.qk_head_dim
        )
        if qb_rows != qb_want:
            return None
        w_qb = attn.q_b_proj_qrep_weight if q_replicate_active else attn.q_b_proj.weight
        if not _valid_qb_weight(w_qb, qb_rows):
            return None
        if rows == m_mla and h.stride() == (H, 1):
            x_ptr = h.data_ptr()
        else:
            bufs["mla_x"][:rows].copy_(h)
            if rows < m_mla:
                bufs["mla_x"][rows:m_mla].zero_()
            x_ptr = bufs["mla_x"].data_ptr()
        idx_m = attn.indexer
        mout = bufs["mla_out"]
        _, scr_mla, cnt_mla = mla_state[qb_rows]
        _run_compiled(
            mp.mla_pre_wide(
                m_mla,
                qb_rows,
                launches=len(self.dsa_layers),
                q_eps=pk.q_eps,
                kv_eps=pk.kv_eps,
            )[0],
            x_ptr,
            attn.fused_qkv_a_proj_with_mqa.weight.data_ptr(),
            attn.g_proj.weight.data_ptr(),
            idx_m.wk.weight.data_ptr(),
            idx_m.weights_proj.weight.data_ptr(),
            attn.q_a_layernorm.weight.data_ptr(),
            attn.kv_a_layernorm.weight.data_ptr(),
            idx_m.wq_b.weight.data_ptr(),
            w_qb.data_ptr(),
            mout["q"].data_ptr(),
            mout["latent"].data_ptr(),
            mout["k_nope"].data_ptr(),
            mout["q_lora"].data_ptr(),
            mout["gate"].data_ptr(),
            mout["idx_q"].data_ptr(),
            mout["idx_k"].data_ptr(),
            mout["idx_w"].data_ptr(),
            cnt_mla.data_ptr(),
            scr_mla.data_ptr(),
            self.epoch.data_ptr(),
            self.slots_dsa[pk.i],
            torch.cuda.current_stream(),
        )
        q_view = mout["q"][: m_mla * qb_rows].view(m_mla, qb_rows)[:rows]
        attn_pre = {
            "q": q_view,
            "latent": mout["latent"][:rows],
            "k_nope": mout["k_nope"][:rows],
            "q_lora": mout["q_lora"][:rows],
            "gate": mout["gate"][:rows],
            "q_nope_out": None,
        }
        idx_pre = {
            "q": mout["idx_q"][:rows],
            "k": mout["idx_k"][:rows],
            "w": mout["idx_w"][:rows],
        }
        return attn_pre, idx_pre

    def _dsa_step(self, pk, hidden, positions, fb, zero_allocator, b, bufs, rows, ping):
        """One fusable DSA layer: optional ``mla_pre_wide`` + attention up to ``forward_pre_o_proj``,
        ``dsa_post`` launch, then SGLang's routed experts onto ``lat_p[cur]``."""
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from sglang.srt.layers.layer_boundary.residual.batch import written_residual

        layer = pk.layer
        h = layer.attn_boundary.prepare(hidden, fb, capture_gathered=None)
        prefix = written_residual(fb)
        assert prefix is not None and prefix.shape == (rows, H), (prefix, rows)
        pre_pair = self._run_mla_pre(pk, bufs, h, fb)
        if pre_pair is not None:
            attn = layer.self_attn
            attn.__dict__["_mono_pre"], attn.indexer.__dict__["_mono_pre"] = pre_pair
            try:
                core = _pre_o_proj(layer, h, positions, fb, zero_allocator)
            finally:
                attn.__dict__.pop("_mono_pre", None)
                attn.indexer.__dict__.pop("_mono_pre", None)
        else:
            core = _pre_o_proj(layer, h, positions, fb, zero_allocator)
        assert core.shape == (rows, KO), (core.shape, rows, KO)
        cur = 1 - ping
        _run_compiled(
            self._build_dsa_launch(pk, b, bufs["cfg"]),
            *self._dsa_launch_args(
                pk, b, bufs, core, prefix, cur, self.slots_dsa[pk.i]
            ),
            torch.cuda.current_stream(),
        )
        _routed(
            pk.moe,
            bufs["lat"][:rows],
            bufs["logits"][:rows],
            bufs["ids"][:rows],
            bufs["w"][:rows],
            bufs["lat_p"][cur][:rows],
            sortv=bufs["sortv"] if rows == b else None,
        )
        return cur

    def backbone(self, built, input_ids, fb, input_embeds):
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        from sglang.srt.layers.layer_boundary.residual.batch import (
            final_norm,
            fold,
            set_written,
            start,
        )
        from sglang.srt.models.kimi_k3 import AuxHiddenStateList, _cdiv
        from sglang.srt.utils.common import BumpAllocator

        m = self.bb
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        rows = hidden.shape[0]
        L = int(fb.spec_info.draft_token_num)
        assert L == self.L and rows % L == 0, (rows, L, self.L)
        if not getattr(self, "_said", False):
            self._said = True
            _say(f"first wide step: rows {rows}, L {L}")
        R = rows // L
        b = next(x for x in self.buckets if x >= rows)
        captures = set(m.dspark_layers_to_capture or ())
        bufs = self._ensure(b, L, captures)
        cfg = bufs["cfg"]
        start(fb)
        zero_allocator = BumpAllocator(
            buffer_size=2 * (m.end_layer - m.start_layer),
            dtype=torch.float32,
            device=hidden.device,
        )
        m.attn_bank.open(hidden, _cdiv(m.end_layer, m.config.attn_res_block_size), None)
        aux = AuxHiddenStateList()
        ks = self._kda_step(fb, R, L, bufs)
        self.epoch[0].add_(1)
        cols = self.cols
        i = m.start_layer
        ping = 0
        entry_prefix = None  # SGLang's prefix the next run's first launch reads (kept alive until it is enqueued)
        bank_synced_nvb = 0
        while i < m.end_layer:
            if i in self.dsa_layers:
                res = m.attn_bank.require()
                nvb = res.num_valid_blocks
                if nvb > bank_synced_nvb:
                    bufs["bank"][:rows, bank_synced_nvb:nvb] = res.block_residual[
                        :rows, bank_synced_nvb:nvb, cols
                    ]
                    bank_synced_nvb = nvb
                pk = self.dsa_layers[i]
                assert nvb == pk.nvb, (i, nvb, pk.nvb)
                ping = self._dsa_step(
                    pk,
                    hidden,
                    built._positions,
                    fb,
                    zero_allocator,
                    b,
                    bufs,
                    rows,
                    ping,
                )
                # When i is in self.entry, the next KDA launch (i+1, t1=True) consumes
                # pout[ping], sh[ping], and lat_p[ping] directly via entry_prefix=None
                # (and performs DSpark capture of layer i if i is in captures).
                entry_prefix = None
                if i not in self.entry:
                    hidden = set_written(
                        self._exit_stream(pk.moe, b, rows, bufs, ping), fb
                    )
                    if i in captures:
                        aux.append(m._dspark_capture_stream(i, hidden, fb))
                i += 1
                continue
            if i in self.entry:
                prefix = self._tail_front(
                    m.layers[i],
                    hidden,
                    built._positions,
                    fb,
                    zero_allocator,
                    bufs,
                    rows,
                    ping,
                )
                if prefix.stride() == (H, 1):
                    entry_prefix = prefix
                else:
                    bufs["pout"][ping][:rows, cols] = prefix[:, cols]
                i += 1
                continue
            if i not in self.layers:
                hidden = m.layers[i](
                    positions=built._positions,
                    hidden_states=hidden,
                    forward_batch=fb,
                    zero_allocator=zero_allocator,
                    capture_gathered=None,
                )
                if i in captures and (i + 1) not in self.layers:
                    aux.append(m._dspark_capture_stream(i, hidden, fb))
                i += 1
                continue
            # a run of covered layers
            res = m.attn_bank.require()
            if not self.layers[
                i
            ].t1:  # else its first launch runs the previous layer's MoE tail
                full = fold(hidden, fb)
                bufs["mid"][:rows] = full[:, cols]
            # copy only the bank rows bufs["bank"] does not hold yet: the launches write theirs into it
            # themselves (then gathered into SGLang's bank)
            nvb = res.num_valid_blocks
            if nvb > bank_synced_nvb:
                bufs["bank"][:rows, bank_synced_nvb:nvb] = res.block_residual[
                    :rows, bank_synced_nvb:nvb, cols
                ]
                bank_synced_nvb = nvb
            run = []
            while i in self.layers:
                run.append(i)
                i += 1
            for n, li in enumerate(run):
                pk = self.layers[li]
                v = self._variant(pk, b, L, captures)
                assert v["t1"] == (n > 0 or (li - 1) in self.entry)
                assert res.num_valid_blocks == pk.nvb, (
                    li,
                    res.num_valid_blocks,
                    pk.nvb,
                )
                tab = self._kda_table(pk, ks, b, bufs)
                cur = 1 - ping
                pre = entry_prefix if n == 0 else None
                _run_compiled(
                    self._build_launch(pk=pk, b=b, L=L, v=v, cfg=cfg),
                    *self._launch_args(pk, b, bufs, tab, ping, self.slots[li], pre),
                    torch.cuda.current_stream(),
                )
                if n == 0:
                    entry_prefix = None
                if v["capture"]:
                    aux.append(bufs["cap"][:rows].clone())
                if pk.write:
                    res.write(
                        tensor_model_parallel_all_gather(
                            bufs["bank"][:rows, pk.nvb].contiguous(), dim=-1
                        )
                    )
                    bank_synced_nvb = max(bank_synced_nvb, pk.nvb + 1)
                lat_p = bufs["lat_p"][cur]
                _routed(
                    pk.moe,
                    bufs["lat"][:rows],
                    bufs["logits"][:rows],
                    bufs["ids"][:rows],
                    bufs["w"][:rows],
                    lat_p[:rows],
                    # the launch sorted all b rows: usable only when the step fills its bucket (pad = M = rows)
                    sortv=bufs["sortv"] if rows == b else None,
                )
                ping = cur
            # leave: the last layer's MoE tail on SGLang
            last = self.layers[run[-1]]
            hidden = set_written(self._exit_stream(last.moe, b, rows, bufs, ping), fb)
            if run[-1] in captures:
                aux.append(m._dspark_capture_stream(run[-1], hidden, fb))
        hidden = final_norm(hidden, fb, m._final_read, capture=None, skip_empty=True)
        m.attn_bank.close()
        if m.dspark_layers_to_capture is not None:
            return hidden, aux
        return hidden
