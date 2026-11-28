# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 on gfx950: the monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed decode or
DSpark verify step runs ``_Built.forward`` here, which takes and returns what
``KimiK3ForConditionalGeneration.forward`` does.

One launch a layer. It starts at the attention's TP-local core output, which
is o_proj's input, and ends at the layer's MoE output. It runs, in order:

1. o_proj and the attention-TP all-reduce;
2. the MLP attention-residual seam;
3. the router and its top-16;
4. the latent down and the shared gate_up projections;
5. the MXFP4 experts;
6. the shared down and the latent up projections;
7. the MoE all-reduce.

Both reduces go over a peer buffer. Neither leaves the kernel.

The attention itself stays on SGLang's path. That is where this contract
differs from DeepSeek-V4.1's: the model runs the attention up to
``forward_pre_o_proj``, and the launch owns everything after it. The KDA and
the full-attention layers reach that point the same way. At TP8 both stop at a
``CORE``-wide output. So the launch covers every MoE layer, not one attention
kind.

The kernels read SGLang's weights in place. The dense weights are read as bf16
in their checkpoint layout. o_proj may instead be a Quark per-channel FP8
weight, read as SGLang stored it for its FP8 GEMM (``_o_proj_operands``). The
experts are read in whichever MXFP4 layout AITER's MoE was given: gate / up
interleaved (A8W4, the A16W4 default) or separated (``AITER_SITUV2_A4W4``),
whose preshuffle is byte for byte the one the launch was written for. There is
no prepared-weight copy, so the KV cache budget pays nothing.

Two further differences from DeepSeek-V4.1 are deliberate. Each is argued in
the module that holds it. The mailbox epoch is bumped by the host, not advanced
on the device (``gfx950/tag.py``). And the build compiles by running each
launch once on real weights, not through FlyDSL's compile-only mode
(``_warmup``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

import msgspec
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

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k3 import (
    CORE,
    HF_CONFIG,
    LINEAR_ATTN_CONFIG,
    MAX_BLOCKS,
    MAX_ROWS,
)

logger = logging.getLogger(__name__)

TP_SIZE = 8
# K2's OPT_* bits (gfx950/layer.py: sentinel all-reduces, preloads, tagged x rows,
# router partials, batched polls), each bit for bit the launch without it; unset
# is layer.K2_OPT, all of them.
K2_OPT = os.environ.get("SGLANG_AMD_MONO_K3_K2_OPT")
# The MLA (DSA) layers' projection front as one launch (gfx950/attention/mla_pre.py):
# AttnRes, fused_qkv_a, the q / kv norms, q_b, g_proj and the indexer's wk,
# weights_proj and wq_b, handed to SGLang's MLA forward and indexer.
MLA_PRE = os.environ.get("SGLANG_AMD_MONO_K3_MLA_PRE", "1") == "1"
# SGLANG_AMD_MONO_K3_K1=0 builds no K1: every KDA layer keeps SGLang's attention.
K1 = os.environ.get("SGLANG_AMD_MONO_K3_K1", "1") == "1"
# SGLANG_AMD_MONO_K3_WIDE=0 keeps only the <=8-row narrow launches.
WIDE = os.environ.get("SGLANG_AMD_MONO_K3_WIDE", "1") == "1"
DCP_SIZES = (1, 8)

# The SGLang names this glue reads, by file under the sglang package (``f(p)``:
# a parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    # the MXFP4 expert layout the launch reads, as AITER's MoE was given it
    "srt/layers/quantization/mxfp4.py": ("_aiter_situ_uses_gu_interleaved_weights",),
    # the K3 AITER switch the expert layout depends on (its renamed Envs field is
    # optional: SGLANG_AITER_K3_OPT is read where it does not exist yet) and the
    # host-only Mamba checkpoint switch (required)
    "srt/environ.py": ("envs", "SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT"),
    # the SiTU parameters K2 hard-codes
    "srt/configs/kimi_linear.py": (
        "activation_situ_beta",
        "activation_situ_linear_beta",
    ),
    # a Quark per-channel FP8 projection, and whether SGLang preshuffled it; scheme is
    # set on every Quark linear, and a non-Quark linear has none (a guarded read)
    "srt/layers/quantization/quark/quark.py": ("scheme",),
    "srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py": (
        "QuarkW8A8Fp8",
        "weight_qscheme",
        "is_static_input_scheme",
        "weight_scale",
    ),
    "srt/layers/quantization/fp8_utils.py": (
        "use_aiter_bpreshuffle_gemm",
        "use_aiter_bpreshuffle_gemm(output_size)",
    ),
    # o_proj's bias, which the launch has no term for
    "srt/layers/linear.py": ("bias",),
    # the experts' activation, which the launch computes as SiTU
    "srt/layers/moe/fused_moe_triton/layer.py": ("moe_runner_config",),
    "srt/layers/moe/moe_runner/base.py": ("activation",),
    "srt/utils/common.py": (
        "BumpAllocator",
        "get_bool_env_var",
    ),
    # mlp-sync padding, which the launch has no handling for
    "srt/layers/dp_attention.py": ("require_mlp_sync",),
    # the model's own forward, which _Built reproduces for a routed step
    "srt/managers/mm_utils.py": ("general_mm_embed_routine",),
    "srt/managers/schedule_batch.py": ("Modality",),
    "srt/layers/layer_boundary/__init__.py": (
        "AttentionInputs",
        "get_attn_tp_context",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "final_norm",
        "set_written",
        "start",
        "written_residual",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_exec",
        "get_spec",
        "get_schedule",
        '"dcp_enabled"',
    ),
    "srt/arg_groups/fields/exec_.py": (
        "cuda_graph_max_bs_decode",
        "enable_linear_replayssm_spec",
    ),
    "srt/arg_groups/fields/schedule.py": ("max_running_requests",),
    "srt/arg_groups/fields/spec.py": (
        "speculative_algorithm",
        "speculative_num_draft_tokens",
    ),
    "srt/arg_groups/fields/parallel.py": (
        "dcp_size",
        "attn_cp_size",
        "enable_prefill_cp",
        "dcp_replicate_q_proj",
    ),
    # the KV offload paths Feature.KV_OFFLOAD admits besides HiCache
    "srt/arg_groups/fields/*.py": (
        "enable_lmcache",
        "enable_flexkv",
        "disaggregation_decode_enable_offload_kvcache",
    ),
    "srt/models/deepseek_v2.py": (
        "q_b_proj_qrep_weight",
        "w_kc_qrep",
        "use_deep_gemm_bmm",
        "num_local_heads",
        "dcp_size",
        "qk_head_dim",
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
    # the attention stops here and the launch runs o_proj; both attention
    # classes expose it. The rest are what the forward and the checks read.
    "srt/models/kimi_k3.py": (
        "AuxHiddenStateList",
        "_cdiv",
        "forward_pre_o_proj",
        "use_latent_moe",
        "routed_expert_norm",
        "routed_expert_up_proj",
        "use_attn_residuals",
        "_sp_moe",
        "all_reduce_fusion",
        "experts",
        "o_proj",
        "layers",
        "language_model",
    ),
}


def _text_config(hf_config):
    """Kimi-K3's shapes live on the text config; a text-only checkpoint has
    none and is its own text config."""
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_config(hf_config) -> dict:
    return getattr(_text_config(hf_config), "linear_attn_config", None) or {}


def _backbone(model):
    """The K3 language backbone, under the VLM wrapper or on its own."""
    return getattr(model, "language_model", model).model


def _covered_layers(hf_config) -> list[int]:
    """The layers the launch can run: every MoE layer. Which attention a layer
    runs does not matter -- the launch starts after it."""
    t = _text_config(hf_config)
    dense = int(getattr(t, "first_k_dense_replace", 0) or 0)
    freq = int(getattr(t, "moe_layer_freq", 1) or 1)
    return [i for i in range(dense, int(t.num_hidden_layers)) if i % freq == 0]


# SGLang's Kimi-K3 AITER switch, which keeps the experts at their native
# 384-wide intermediate; an SGLang without the Envs field reads the old name.
def _k3_aiter_opt() -> bool:
    from sglang.srt.environ import envs

    field = getattr(envs, "SGLANG_OPT_HIP_K3_AITER", None)
    if field is not None:
        return bool(field.get())
    from sglang.srt.utils.common import get_bool_env_var

    return get_bool_env_var("SGLANG_AITER_K3_OPT")


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the shapes, which live on the text
    config and so are out of ``requires.hf_config``'s reach; SGLang's K3 AITER
    switch; and FlyDSL."""
    t = _text_config(hf_config)
    linear = _linear_attn_config(hf_config)
    reasons = [
        f"text config {k} {getattr(t, k, None)!r} is not {v!r}"
        for k, v in HF_CONFIG.items()
        if getattr(t, k, None) != v
    ]
    reasons += [
        f"linear_attn_config {k} {linear.get(k)!r} is not {v!r}"
        for k, v in LINEAR_ATTN_CONFIG.items()
        if linear.get(k) != v
    ]
    # The bank rows a layer's MLP seam reads grow with the layer index; the
    # launch is built for MAX_BLOCKS of them.
    block = getattr(t, "attn_res_block_size", None)
    if not block:
        reasons.append("needs the attention-residual stream (attn_res_block_size)")
    else:
        last = max(_covered_layers(hf_config), default=0)
        blocks = -(-last // block) + int(last % block == 0)
        if blocks > MAX_BLOCKS:
            reasons.append(
                f"attn_res_block_size {block} gives layer {last} {blocks} "
                f"attention-residual blocks, the launch is built for {MAX_BLOCKS}"
            )
    if not _k3_aiter_opt():
        reasons.append(
            "needs SGLANG_OPT_HIP_K3_AITER=1 (SGLANG_AITER_K3_OPT=1 before the "
            "rename): without it the experts are padded past their native "
            "384-wide intermediate"
        )
    # K2 computes SiTU with fixed (beta, linear_beta); SGLang reads them from the config
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import BETA, LBETA

    situ = (t.activation_situ_beta, t.activation_situ_linear_beta)
    if situ != (BETA, LBETA):
        reasons.append(
            f"SiTU (beta, linear_beta) is {situ}; the launch computes ({BETA}, {LBETA})"
        )
    if cfg is not None:
        reasons += _offload_disable_reasons(cfg)
        dcp = cfg.dcp_size or 1
        if dcp not in DCP_SIZES:
            reasons.append(f"--dcp-size {dcp} is not one of {list(DCP_SIZES)}")
        if (cfg.attn_cp_size or 1) > 1 or cfg.enable_prefill_cp:
            reasons.append(
                "attention context parallel is on; only --dcp-size is accepted"
            )
    if WIDE and cfg is not None:
        from sgl_amd_mono.kimi_k31 import wide

        tokens, wide_runs = _wide_step_shape(
            cfg.speculative_algorithm,
            cfg.speculative_num_draft_tokens,
            cfg.cuda_graph_max_bs_decode,
            cfg.max_running_requests,
        )
        if wide_runs and tokens not in wide.SUPPORTED_L:
            reasons.append(
                f"wide launch requires --speculative-num-draft-tokens in "
                f"{list(wide.SUPPORTED_L)} (got {tokens}); set SGLANG_AMD_MONO_K3_WIDE=0 "
                "for narrow-only <=8-row serving"
            )
        if wide_runs and cfg.enable_linear_replayssm_spec and _host_only_mamba():
            reasons.append(HOST_ONLY_RING_REASON)
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


# A Quark per-channel FP8 linear -> (in AITER's (16, 16) preshuffle, the [out, in]
# storage, its fp32 row scale); raises ValueError naming the misfit.
def _quark_fp8_storage(linear, name: str):
    from sglang.srt.layers.quantization.fp8_utils import use_aiter_bpreshuffle_gemm
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    scheme = getattr(linear, "scheme", None)
    if not (
        isinstance(scheme, QuarkW8A8Fp8) and scheme.weight_qscheme == "per_channel"
    ):
        raise ValueError(f"an FP8 {name} is read only as Quark per-channel FP8")
    if getattr(linear, "bias", None) is not None:
        raise ValueError(f"the FP8 {name} has a bias, which the launch does not add")
    # The launch multiplies the bf16 activation and applies only the weight scale;
    # a static input scale would be clipped and rounded by SGLang and not by it.
    if scheme.is_static_input_scheme:
        raise ValueError(f"the FP8 {name} quantizes its input statically")
    if linear.weight.dim() != 2:
        raise ValueError(f"the FP8 {name} weight is not 2-D")
    # SGLang stores the weight transposed for its FP8 GEMM
    storage = linear.weight.t()
    if storage.dtype != torch.float8_e4m3fn:
        raise ValueError(f"the FP8 {name} is {storage.dtype}, not E4M3 (float8_e4m3fn)")
    if not storage.is_contiguous():
        raise ValueError(f"the FP8 {name} is not SGLang's transposed contiguous weight")
    rows = storage.shape[0]
    scale = linear.weight_scale
    if not (
        isinstance(scale, torch.Tensor)
        and scale.dtype == torch.float32
        and scale.numel() == rows
        and scale.is_contiguous()
    ):
        raise ValueError(
            f"the FP8 {name} has no fp32 scale for each of its {rows} rows"
        )
    # preshuffled exactly when SGLang's own GEMM chose that at load
    return use_aiter_bpreshuffle_gemm(rows), storage, scale.view(rows)


# o_proj as the launch reads it: (``O_*`` format, the [7168, 1536] storage,
# its fp32 scale a row or None). Raises ValueError naming what does not fit.
def _o_proj_operands(o_proj):
    from sgl_amd_mono.kimi_k3.gfx950.stages import oproj

    hidden = HF_CONFIG["hidden_size"]
    # GPTQ / AWQ-style linears keep qweight and have no weight
    weight = getattr(o_proj, "weight", None)
    if weight is None:
        raise ValueError("o_proj has no weight")
    if o_proj.bias is not None:
        raise ValueError("o_proj has a bias")
    if weight.dtype == torch.bfloat16:
        if tuple(weight.shape) != (hidden, CORE) or not weight.is_contiguous():
            raise ValueError(f"o_proj is not a contiguous bf16 [{hidden}, {CORE}]")
        return oproj.O_BF16, weight, None
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"o_proj is {weight.dtype}, not bf16 or Quark FP8 E4M3")
    shuffled, storage, scale = _quark_fp8_storage(linear=o_proj, name="o_proj")
    if tuple(storage.shape) != (hidden, CORE):
        raise ValueError(
            f"the FP8 o_proj is {tuple(storage.shape)}, not [{hidden}, {CORE}]"
        )
    return (oproj.O_FP8_SHUFFLED if shuffled else oproj.O_FP8), storage, scale


def _offload_disable_reasons(cfg) -> list[str]:
    # Feature.KV_OFFLOAD also admits these; only HiCache has run under the wide launches.
    flags = {
        "--enable-lmcache": cfg.enable_lmcache,
        "--enable-flexkv": cfg.enable_flexkv,
        "--disaggregation-decode-enable-offload-kvcache": (
            cfg.disaggregation_decode_enable_offload_kvcache
        ),
    }
    return [
        f"{flag}: only HiCache's KV offload is supported"
        for flag, on in flags.items()
        if on
    ]


HOST_ONLY_RING_REASON = (
    "wide launch does not support request-scoped ReplaySSM rings "
    "(SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT=1 with --enable-linear-replayssm-spec)"
)


def _host_only_mamba() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.get())


def _wide_step_shape(
    spec_algo, draft_tokens, graph_max_bs, max_running_reqs
) -> tuple[int, bool]:
    """(tokens a request, whether a step goes wide): with no cap known, yes; at a supported
    draft length, when the smallest width it serves fits the cap's rows; at any other,
    when a step passes MAX_ROWS rows (so the draft-length check refuses it)."""
    from sgl_amd_mono.kimi_k31 import wide

    tokens = int(draft_tokens or 1) if spec_algo is not None else 1
    # the larger cap: an eager server routes up to max_running_reqs, a graph one up to
    # the graph cap (and an unset one is 0)
    max_bs = max(int(graph_max_bs or 0), int(max_running_reqs or 0))
    if max_bs == 0:
        return tokens, True
    if tokens not in wide.SUPPORTED_L:
        return tokens, max_bs * tokens > MAX_ROWS
    mine = wide.widths_at(tokens)
    return tokens, bool(mine) and mine[0] <= max_bs * tokens


def _wide_will_run() -> bool:
    """Whether the current server's steps reach a width the wide launch serves
    (``_wide_step_shape``)."""
    from sglang.srt.runtime_context import get_exec, get_schedule, get_spec

    try:
        spec = get_spec()
        _, wide_runs = _wide_step_shape(
            spec.speculative_algorithm,
            spec.speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_max_bs_decode,
            get_schedule().max_running_requests,
        )
    except ValueError:
        return False
    return wide_runs


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Every per-layer check the config cannot answer. The kernels read each
    weight as a raw address, so a wrong dtype, shape or layout is read as if it
    were the expected one."""
    try:
        from sglang.srt.layers.dp_attention import require_mlp_sync
    except ImportError:  # SGLang before sgl-project/sglang#48781
        from sglang.srt.utils.common import require_mlp_sync

    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import LAT, RI, E

    hidden = HF_CONFIG["hidden_size"]
    covered = _covered_layers(hf_config)
    layers = _backbone(model).layers
    reasons = []
    if require_mlp_sync():
        # mlp-sync pads extend batches to a multiple of attn_tp_size; the
        # launch has no padded-row handling.
        reasons.append("mlp-sync padding is on")
    for i in covered:
        layer = layers[i]
        attn, moe = layer.self_attn, layer.mlp
        experts = getattr(moe, "experts", None)
        up = getattr(moe, "routed_expert_up_proj", None)
        try:
            _o_proj_operands(attn.o_proj)
            o_proj_reason = None
        except ValueError as e:
            o_proj_reason = str(e)
        checks = (
            (moe.use_latent_moe, "not a latent MoE layer"),
            (moe.routed_expert_norm is not None, "no latent RMSNorm"),
            (
                up is not None
                and up.weight.shape[0] == hidden
                and up.weight.is_contiguous(),
                f"the latent up projection is not a contiguous [{hidden}, latent]",
            ),
            (
                experts is not None
                and getattr(experts.w13_weight, "is_shuffled", False)
                and getattr(experts.w2_weight, "is_shuffled", False),
                "the experts are not in aiter's preshuffled MXFP4 layout",
            ),
            # The launch indexes an expert at its native 384-wide intermediate;
            # a padded one (512) would be read at the wrong strides.
            (
                experts is None
                or (
                    tuple(experts.w13_weight.shape) == (E, 2 * RI, LAT // 2)
                    and tuple(experts.w2_weight.shape) == (E, LAT, RI // 2)
                ),
                f"the experts are not [{E}, {2 * RI}, {LAT // 2}] / "
                f"[{E}, {LAT}, {RI // 2}] packed MXFP4 (a padded intermediate)",
            ),
            # K2 computes SiTU for routed and shared experts alike; another
            # activation would also read as AITER's separated A4W4 layout.
            (
                experts is not None and experts.moe_runner_config.activation == "situ",
                "the experts' activation is not SiTU, which the launch computes",
            ),
            (layer.use_attn_residuals, "no attention-residual stream"),
            # SP-MoE reduce-scatters o_proj's output and the fused all-reduce
            # completes it out of a symmetric buffer; the launch does neither.
            (not layer._sp_moe, "SP-MoE is on"),
            (not layer.all_reduce_fusion, "the fused attention all-reduce is on"),
            (
                hasattr(attn, "forward_pre_o_proj"),
                "the attention cannot stop before o_proj",
            ),
            # The MLA cores reach o_proj from several places, so stopping
            # before it rides on the gated o_proj wrapper, which only exists
            # with the output gate. The KDA layers have no such attribute and
            # stop before o_proj on their own.
            (
                getattr(attn, "use_output_gate", True),
                "the MLA output gate is off, and the split before o_proj "
                "rides on its wrapper (mla_use_output_gate)",
            ),
            (o_proj_reason is None, o_proj_reason),
        )
        reasons += [f"layer {i}: {why}" for ok, why in checks if not ok]
        if reasons and i > covered[0]:
            break  # one layer's reasons say it for all
    if WIDE and _wide_will_run():
        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        _, state_reasons = k1_mod._state_config()
        reasons += state_reasons
        kda_layers = [i for i in covered if k1_mod._is_kda_layer(layers[i])]
        for i in kda_layers:
            attn = layers[i].self_attn
            try:
                k1_mod._validate_in_proj_operands(attn)
            except ValueError as e:
                reasons.append(f"layer {i}: {e}")
                break
    return reasons


def _build_narrow(model, hf_config, ctx, narrow_widths):
    """Pack every covered layer and compile the narrow (<= MAX_ROWS) widths."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950.runner import K3MonoRunner

    backbone = _backbone(model)
    p = get_parallel()
    device = backbone.layers[0].input_layernorm.weight.device
    # the peer handshake is collective: every rank reaches it here
    runner = K3MonoRunner(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
    runner.set_deadline(ctx.deadline_s)
    layers, batches = {}, {}
    for rows in narrow_widths:
        for i in _covered_layers(hf_config):
            packed = _weights(backbone.layers[i], i, rows, runner)
            layers.setdefault(rows, {})[i] = packed
        # FlyDSL compiles a launcher on its first call, and the first real call
        # is inside a CUDA graph capture, where compiling aborts the process.
        # A compile-only call cannot stand in: with zero arguments the buffer
        # loads fold to a pattern LLVM cannot select. So run each launch once,
        # on this layer's real weights and throwaway activations. Every rank
        # runs the same launches in the same order, so the peer waits pair up
        # exactly as they do on a served step.
        #
        # K1's conv rows come from its batch, so that is built first.
        batches[rows] = k1_mod.build_batch(layers[rows], device)
        _warmup(runner, layers[rows], rows)
    return _Built(model, runner, layers, batches)


def _build(model, hf_config, ctx, *, use_wide: bool | None = None):
    """Pack every covered layer and compile the narrow and (when reached) wide widths."""
    from sglang.srt.runtime_context import get_parallel

    narrow_widths = tuple(w for w in ctx.widths if w <= MAX_ROWS)
    narrow = _build_narrow(model, hf_config, ctx, narrow_widths)
    wide_on = WIDE if use_wide is None else use_wide
    if not (wide_on and any(w > MAX_ROWS for w in ctx.widths)):
        return narrow
    from sgl_amd_mono.kimi_k31 import wide

    L = wide.tokens()
    served = _wide_build_rows(ctx.widths, L)
    if not served:
        # every routed width past MAX_ROWS is another draft length's: no wide launch to
        # build, and those steps run SGLang's own backbone
        return _WideBuilt(narrow, None)

    def mem():
        return (
            torch.cuda.mem_get_info()[0],
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )

    def used(a, b):
        return "device {:.2f} / torch allocated {:.2f} / reserved {:.2f} GiB".format(
            *((x - y) / 2**30 for x, y in ((a[0], b[0]), (b[1], a[1]), (b[2], a[2])))
        )

    p = get_parallel()
    device = _backbone(model).layers[0].input_layernorm.weight.device
    m0 = mem()
    w = wide.Wide(
        model,
        p.tp_rank,
        p.tp_group.cpu_group,
        device,
        L=L,
        max_rows=max(served),
    )
    m1 = mem()
    w.warmup()
    m2 = mem()
    torch.cuda.empty_cache()
    wide._say(
        f"memory: wide layers {used(m0, m1)}; warmup {used(m1, m2)}; "
        f"after empty_cache {used(m0, mem())}"
    )
    return _WideBuilt(narrow, w)


def _wide_build_rows(widths, L: int) -> list:
    """The routed widths the wide launch serves at L draft tokens (the binding also routes
    another L's widths that are multiples of L; those run SGLang's own backbone)."""
    from sgl_amd_mono.kimi_k31 import wide

    mine = set(wide.widths_at(L))
    return [w for w in widths if w in mine]


def _warmup(runner, packed_layers: dict, rows: int) -> None:
    """Launch every layer once at this width, before any graph capture."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    device = runner.device
    core = torch.zeros(rows, CORE, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    banks: dict[int, torch.Tensor] = {}

    def bank(nb: int):
        # One row past the nb blocks the launch reads: a block-write layer's
        # launch (K1 or mla_pre, key.write_idx == nb) stores its updated prefix
        # in block nb, as a served step does into the bank's spare capacity.
        # An nb-row bank put the last token's write one HIDDEN row past the end
        # and zeroed whatever the allocator had placed there (seen: a later
        # layer's indexer k_norm at 8 draft tokens, which silently broke that
        # layer's DSA top-k).
        if nb not in banks:
            banks[nb] = torch.zeros(
                rows, nb + 1, HIDDEN, dtype=torch.bfloat16, device=device
            )
        return banks[nb]

    max_nhq = max(
        (p.mla[0].nhq for p in packed_layers.values() if p.mla is not None), default=0
    )
    mla_out = _mla_outputs(rows, device, nhq=max_nhq) if max_nhq else None
    dummy_qrep_qb = None
    dummy_qrep_wkc = None
    for packed in packed_layers.values():
        # K1 first: on a served step it runs before this layer's K2, so the
        # waits pair up in the same order here as there.
        if packed.k1 is not None:
            k1_mod.warmup(
                runner, packed.k1, runner.prefix[:rows], bank(packed.k1.key.nblocks)
            )
        if packed.mla is not None:
            from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

            key, w = packed.mla
            w_warm = w
            if key.nhq > mp.NHQ:
                attn = packed.module.self_attn
                qb_shp = (key.nhq * mp.QHD, mp.QL)
                cand_qb = getattr(attn, "q_b_proj_qrep_weight", None)
                if _valid_bf16_tensor(cand_qb, qb_shp):
                    w_qb = cand_qb
                else:
                    if dummy_qrep_qb is None or tuple(dummy_qrep_qb.shape) != qb_shp:
                        dummy_qrep_qb = torch.zeros(
                            qb_shp, dtype=torch.bfloat16, device=device
                        )
                    w_qb = dummy_qrep_qb
                w_warm = {**w, "w_qb": w_qb}
                if key.qn:
                    wkc_shp = (key.nhq, mp.QN, mp.KV)
                    cand_kc = getattr(attn, "w_kc_qrep", None)
                    if _valid_bf16_tensor(cand_kc, wkc_shp):
                        w_kc = cand_kc
                    else:
                        if (
                            dummy_qrep_wkc is None
                            or tuple(dummy_qrep_wkc.shape) != wkc_shp
                        ):
                            dummy_qrep_wkc = torch.zeros(
                                wkc_shp, dtype=torch.bfloat16, device=device
                            )
                        w_kc = dummy_qrep_wkc
                    w_warm["w_kc"] = w_kc
            out_warm = _mla_out_view(mla_out, key.nhq)
            mp.mla_pre_launch(
                key, prefix=runner.prefix[:rows], blocks=bank(key.nblocks), w=w_warm,
                out=out_warm, scratch=runner.scratch,
                layer=packed.i, epoch=runner.epoch,
            )  # fmt: skip
        _k2_launch(
            runner, packed, core, runner.prefix[:rows], bank(packed.key.nblocks), out
        )
    torch.cuda.synchronize()


# Bounds against SGLang's standard path on the same GPU. They come from what
# test/kernels/kimi_k3 measures at TP8, with a little room. Each number below
# is the worse of the two prefix modes, and for the MoE the worst of widths
# 1, 4 and 8:
#
#   k2.prefix  cos 1.000000            k2.x  cos 1.000000
#   k2.out     cos 0.999994 rel 0.0036
#   moe.out    cos 0.999847 rel 0.0175
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # The updated attention-residual prefix: o_proj, the TP reduce and
        # the add. The reduce sums the same bf16 partials in the same rank
        # order as the standard path, so the result is exact. A bound below
        # 1.0 would let a reordered reduce through.
        "k2.prefix.cosine": 1.0 - 1e-7,
        # The seam: slice, gate, mix, then the FFN input norm. Also exact.
        "k2.x.cosine": 1.0 - 1e-7,
        # The same two on a Quark FP8 o_proj (check_k2 --o-proj fp8*). Not exact:
        # the launch widens the weight to bf16 and sums in MFMA order, SGLang's
        # FP8 GEMM quantizes the activation, and no reference sums in the
        # launch's order, so a sum near a bf16 rounding boundary can land one ulp
        # apart. Measured on 8x MI355X against check_k2's torch reference (the
        # E4M3 weight in fp32 times the unquantized activation, row-scaled): prefix
        # 1 - 1.19e-7 accumulating (one fp32 ulp of the cosine), exactly 1 resetting.
        "k2_fp8.prefix.cosine": 1.0 - 1e-6,
        "k2_fp8.x.cosine": 1.0 - 1e-6,
        # The layer output: the MoE on top of the seam. The expert sums land in
        # another order, so this is bf16 rounding, not exact.
        "k2.out.cosine": 0.99999,
        "k2.out.rel": 0.005,
        # The MoE launch, which runs in no forward: the kernel checks use it as
        # the reference for the MoE half of the layer launch. Looser than
        # k2.out because the check feeds it a fresh input at every width, and
        # one row has the least averaging.
        "moe.out.cosine": 0.9998,
        "moe.out.rel": 0.02,
        # SGLANG_AMD_MONO_CHECK, which scores K1 against SGLang's own
        # attention on the real weights at TP8, over 28 steps of one request:
        #
        #   k1.core    cos 0.99999678  rel 0.0026
        #   k1.bank    rel 0 (bit for bit); cos 0.99999982
        #   k1.ssm     cos 0.99999976  rel 0.00033
        #   k1.window  cos 0.99999988  rel 0.00032
        #
        # The gated core output: the in-projection, f_b, the conv and the
        # recurrence. Its reference sums in another order, so this is bf16
        # rounding.
        "k1.core.cosine": 0.99999,
        "k1.core.rel": 0.004,
        # The bank row the launch writes in place of AttnResidual.write. The
        # same values in the same order, so it is exact: rel is what holds it
        # to that, and DSpark's capture reads the bank, so a row that drifts
        # changes the draft's input. The cosine bound is the looser of the two
        # on purpose. Cosine of two 4608-wide bf16 rows carries its own
        # rounding, measured down to 0.99999982 on rows that are bit for bit.
        "k1.bank.cosine": 1.0 - 1e-6,
        "k1.bank.rel": 0.0,
        # The recurrent state staged for the step's verify rows.
        "k1.ssm.cosine": 1.0 - 1e-6,
        "k1.ssm.rel": 0.001,
        # The conv window handed back to SGLang's speculative state, which the
        # accept-time commit reads. Nothing in the serving path reports a wrong
        # one: it shows up as a lower acceptance length and a GSM8K score a few
        # points down. Measured exact, so the bound is tight on purpose.
        "k1.window.cosine": 1.0 - 1e-6,
        "k1.window.rel": 0.001,
        # K1 on a Quark FP8 in-projection (k1_fp8), against the same check. SGLang
        # quantizes the projection's input to FP8 a token and K1 multiplies it
        # in bf16, so every output moves by that quantization, about 1%; the
        # bank row is still exact. Worst of 10385 layer steps of three requests
        # at TP8 with ReplaySSM and a bf16 state, with room:
        #
        #   core    cos 0.99788   rel 0.065    window  cos 0.99994  rel 0.0115
        #   ring_k  cos 0.99990   rel 0.0139   ring_v  cos 0.99970  rel 0.0245
        #   ring_g  cos 0.99988   rel 0.0167   ring_b  cos 0.99995  rel 0.0100
        #
        # ring_* are the ReplaySSM ring rows the commit replays (raw k / v, the
        # log-decay gate, beta), in place of the ssm state.
        "k1_fp8.core.cosine": 0.997,
        "k1_fp8.core.rel": 0.08,
        "k1_fp8.bank.cosine": 1.0 - 1e-6,
        "k1_fp8.bank.rel": 0.0,
        "k1_fp8.window.cosine": 0.9999,
        "k1_fp8.window.rel": 0.015,
        # The cosine is what test_kimi_k3_wide_fp8.py measures on wide snapshots
        # (ring=False, bf16/fp32 state); the rel metric stays unbounded until measured.
        "k1_fp8.ssm.cosine": 0.9995,
        "k1_fp8.ring_k.cosine": 0.9998,
        "k1_fp8.ring_k.rel": 0.02,
        "k1_fp8.ring_v.cosine": 0.9995,
        "k1_fp8.ring_v.rel": 0.03,
        "k1_fp8.ring_g.cosine": 0.9997,
        "k1_fp8.ring_g.rel": 0.025,
        "k1_fp8.ring_b.cosine": 0.9999,
        "k1_fp8.ring_b.rel": 0.015,
    },
)


def _k3_widths() -> tuple[int, ...]:
    base = tuple(range(1, MAX_ROWS + 1))
    if not WIDE:
        return base
    from sgl_amd_mono.kimi_k31 import wide

    return base + wide.wide_widths()


SPEC = MonoSpec(
    name="kimi_k3_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=_k3_widths(),
            tp_sizes=(TP_SIZE,),
            resident_blocks=BLOCKS,
        )
    },
    # Narrow launches cover decode and DSpark verify steps <= 8 rows; wide launches
    # cover multi-request DSpark verify steps (including DCP8 and HiCache KV offload).
    features=frozenset(
        {Feature.SPECULATIVE_DECODING, Feature.CONTEXT_PARALLEL, Feature.KV_OFFLOAD}
    ),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            # the launch holds the experts TP sharded, not expert-parallel
            "moe_a2a_backend": frozenset({None, "none"}),
            # DSpark's target steps are uniform verify steps; other drafts differ
            "speculative_algorithm": frozenset({None, "DSPARK"}),
        },
        # The shapes are on the text config, which this cannot reach; they are
        # checked in _config_disable_reasons.
        hf_config={"model_type": "kimi_k3"},
        env={"SGLANG_USE_AITER": True},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)


class _Layer(msgspec.Struct):
    """One layer's launches: their build keys and the weights they read in
    place. ``k1`` is None on the full-attention layers, which keep SGLang's
    attention and enter at ``forward_pre_o_proj``."""

    i: int
    key: Any
    module: Any  # the KimiK3DecoderLayer, for the tensors read per step
    k1: Any = None  # the KDA front launch (gfx950/k1.py), where it applies
    mla: Any = None  # (MlaPreBuild, weights): an MLA layer's projection front
    # o_proj's storage and its FP8 scale (None for bf16): _o_proj_operands
    w_o: Any = None
    w_os: Any = None


def _k2_blocks(layer) -> int:
    """Bank rows readable at this layer's MLP seam. A block-write layer
    snapshots its own pre-attention prefix on the attention side, so by the
    seam it has one row more than it inherited."""
    return layer.prev_valid_blocks + int(layer.is_block_write_layer)


def _weights(layer, i: int, rows: int, runner) -> _Layer:
    """This layer's build key at one launch width. The weights themselves are
    read in place at launch, so nothing is packed or copied here."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.layer import K2Build

    moe = layer.mlp
    ofmt, w_o, w_os = _o_proj_operands(layer.self_attn.o_proj)
    return _Layer(
        k1=k1_mod.build(layer, i, rows) if K1 else None,
        mla=_mla_build(layer, rows) if MLA_PRE else None,
        i=i,
        key=K2Build(
            tokens=rows,
            nblocks=_k2_blocks(layer),
            eps=layer.mlp_res_norm.variance_epsilon,
            out_eps=layer.post_attention_layernorm.variance_epsilon,
            ln_eps=moe.routed_expert_norm.variance_epsilon,
            reset=layer.is_block_write_layer,
            opt=k2.K2_OPT if K2_OPT is None else int(K2_OPT),
            guint=_gu_interleaved(moe),
            ofmt=ofmt,
            fault_site=runner.fault_site,
            fault_after=runner.fault_after,
        ),
        module=layer,
        w_o=w_o,
        w_os=w_os,
    )


def _valid_bf16_tensor(t, shape: tuple[int, ...]) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.bfloat16
        and tuple(t.shape) == shape
        and t.is_contiguous()
    )


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
      ``_QREP_PREDICATE``'s conjunction (any order), whose ``forward_batch`` is a named
      (not ``*`` / ``**``) parameter of the function, never rebound.
    - ``pre = self.__dict__.get("_mono_pre")`` is its one assignment; every other use of
      ``pre`` is ``pre is (not) None``, a read of ``pre["<key>"]`` or ``pre.get("<key>")``
      of another key than "q"; ``"_mono_pre"`` appears only there.
    - ``pre["q"]`` is read once, as ``name = pre["q"].view(-1 or pre["q_lora"].shape[0],
      heads, self.qk_head_dim)``, a statement directly in ``if self.q_lora_rank is not
      None and pre is not None`` (or ``if pre is not None``), itself a statement of the
      function body.
    - Before that ``if``: only imports, a docstring, the two assignments above and names
      set to constants. In its branch before the view: only single-name assignments of
      head-count expressions.
      After the view: only name assignments (in ifs too), never using ``name``, whose
      calls have SGLang's callee forms (``self.indexer``, ``self.should_run_indexer``,
      ``maybe_capture_indexer_topk``, ``pre[...].unsqueeze``; a name filter on those
      forms, the callee objects themselves are not checked).
    - The statement right after the ``if`` chain assigns or returns
      ``self._split_q_nope_pe(name, <a name or constant>)``, no keywords; any later use
      of ``name`` is the first argument of a ``self._split_q_nope_pe`` call (its other
      arguments are not checked there).
      What the split's outputs go through afterwards is not checked.
    - No other way to bind a name (``:=``, ``match``, lambdas, nested functions or
      classes, comprehensions, ``global`` / ``nonlocal``, ``except ... as``, ``import
      ... as``, an import of the names above), no rebinding of ``self``, no parameter,
      import or assignment named ``maybe_capture_indexer_topk`` (a name-form check), no
      call by a reflective name (``setattr``,
      ``exec``, ``vars``, ...; aliases are not resolved), and no ``assert``, ``raise``
      or ``try``."""
    import ast

    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    # no construct that binds or shadows a name where a Name store would not show it
    hidden = (ast.NamedExpr, ast.Match, ast.Lambda, ast.ClassDef, ast.Global)
    hidden += (ast.Nonlocal, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    # nor a statement that stops the step on one path (SGLang's handoff has none)
    hidden += (ast.Assert, ast.Raise, ast.Try)
    # nor a call by a reflective name (setattr, exec, ...), wherever it sits
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
    named = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    # every parameter name, the variadic ones included (for the shadowing checks)
    params = named | {p.arg for p in (a.vararg, a.kwarg) if p is not None}
    # the predicate reads the step's own batch: a parameter never rebound
    if {"q_replicate_active", "pre"} & params or "forward_batch" not in named:
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

    def sglang_call(c) -> bool:
        # the indexer's top-k and the k_nope reshape: self.indexer(...),
        # self.should_run_indexer(...), maybe_capture_indexer_topk(...), pre[...].unsqueeze(...)
        f = c.func
        if isinstance(f, ast.Name):
            return f.id == "maybe_capture_indexer_topk"
        return isinstance(f, ast.Attribute) and (
            (
                f.attr in ("indexer", "should_run_indexer")
                and isinstance(f.value, ast.Name)
                and f.value.id == "self"
            )
            or (
                f.attr == "unsqueeze"
                and isinstance(f.value, ast.Subscript)
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "pre"
            )
        )

    if "maybe_capture_indexer_topk" in params | imported or any(
        isinstance(n, ast.Name)
        and n.id == "maybe_capture_indexer_topk"
        and not isinstance(n.ctx, ast.Load)
        for n in ast.walk(fn)
    ):
        return None  # SGLang's module-level helper, not a parameter, import or local
    rest = guard.body[at + 1 :]
    if not all(plain(s) for s in rest) or any(
        (isinstance(n, ast.Name) and n.id == name)
        or isinstance(n, exits)
        or (isinstance(n, ast.Call) and not sglang_call(n))
        for s in rest
        for n in ast.walk(s)
    ):
        return None  # the viewed q is used again, or the branch does more than SGLang's
    # the next statement after the if chain is `... = self._split_q_nope_pe(name, ...)` or
    # `return self._split_q_nope_pe(name, ...)`: the split runs whenever the view did
    nxt = fn.body.index(guard) + 1
    after = fn.body[nxt] if nxt < len(fn.body) else None
    head = after.value if isinstance(after, (ast.Assign, ast.Return)) else None
    if not (
        isinstance(head, ast.Call)
        and ast.dump(head.func) == dump("self._split_q_nope_pe")
        and len(head.args) == 2
        and not head.keywords
        and isinstance(head.args[0], ast.Name)
        and head.args[0].id == name
        and isinstance(head.args[1], (ast.Name, ast.Constant))
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


def _sglang_mono_pre_supports_qrep(attn) -> bool:
    """Whether SGLang's ``forward_absorb_rocm_prepare`` views ``_mono_pre['q']`` with
    ``num_local_heads * dcp_size`` heads under DCP q-replicate and the local heads without
    it (sgl-project/sglang#53130), read from the view's head expression."""
    cached = getattr(attn, "_k3_mono_pre_qrep_ok", None)
    if cached is not None:
        return cached
    fn = getattr(attn, "forward_absorb_rocm_prepare", None)
    ok = False
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
                    ok = _head_count(head, assigned, qrep=False) == (1, 1, 0) and (
                        _head_count(head, assigned, qrep=True) == (1, 1, 1)
                    )
        except (OSError, TypeError, SyntaxError):
            ok = False
    attn._k3_mono_pre_qrep_ok = ok
    return ok


def _mla_out_view(out: dict, nhq: int) -> dict:
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

    rows = out["q"].shape[0]
    qb = nhq * mp.QHD
    if out["q"].shape[1] == qb and out["qn"].shape[1] == nhq:
        return out
    return {
        **out,
        "q": out["q"].view(-1)[: rows * qb].view(rows, qb),
        "qn": out["qn"].view(-1)[: rows * nhq * mp.KV].view(rows, nhq, mp.KV),
    }


# Resolve an MLA linear projection as contiguous bf16 or Quark per-channel FP8;
# raises ValueError naming the misfit.
def _mla_proj_operands(linear, name: str, shape: tuple[int, int]):
    from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt

    if linear is None:
        raise ValueError(f"{name} is missing")
    # GPTQ / AWQ-style linears keep qweight and have no weight.
    weight = getattr(linear, "weight", None)
    if weight is None:
        raise ValueError(f"{name} has no weight")
    if getattr(linear, "bias", None) is not None:
        raise ValueError(f"{name} has a bias, which the launch does not add")
    if weight.dtype == torch.bfloat16:
        if tuple(weight.shape) != shape or not weight.is_contiguous():
            raise ValueError(
                f"{name} is {weight.dtype} {tuple(weight.shape)}, not contiguous bf16 {shape}"
            )
        return wfmt.BF16, weight, None
    shuffled, storage, scale = _quark_fp8_storage(linear=linear, name=name)
    if tuple(storage.shape) != shape:
        raise ValueError(f"the FP8 {name} is {tuple(storage.shape)}, not {shape}")
    return (wfmt.FP8_SHUFFLED if shuffled else wfmt.FP8), storage, scale


def _mla_build(layer, rows: int):
    """An MLA layer's projection-front key and the weights it reads in place, or
    None (a KDA layer, or weights the launch does not take: SGLang's path)."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
    from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    attn = layer.self_attn
    fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    if hasattr(attn, "qkv_conv1d") or fused is None:
        return None
    layer_id = getattr(layer, "layer_id", -1)
    idx = getattr(attn, "indexer", None)  # an MLA layer without one has no attribute
    has_indexer = idx is not None
    par = get_parallel()
    if par.attn_dcp_size > 1:
        # Under DCP8 q-replicate, run the full 96-head q_b_proj_qrep_weight and
        # in-place w_kc_qrep once SGLang's _mono_pre handoff supports full heads.
        dcp_size = getattr(attn, "dcp_size", 1)
        # the 96-head front runs only on bf16 layers with an indexer (Kimi-K3.1)
        if not (
            has_indexer
            and getattr(par, "dcp_enabled", False)
            and getattr(par, "dcp_replicate_q_proj", False)
            and dcp_size > 1
            and not getattr(attn, "use_deep_gemm_bmm", False)
            and getattr(attn, "num_local_heads", 0) * dcp_size == mp.NHQ_QREP
            and getattr(attn, "qk_head_dim", None) == mp.QHD
            and _sglang_mono_pre_supports_qrep(attn)
        ):
            logger.info(
                "monokernel: no DCP MLA front for layer %d: indexer=%s dcp=%s qrep=%s heads=%s qrep_handoff=%s",
                layer_id,
                has_indexer,
                dcp_size,
                getattr(par, "dcp_replicate_q_proj", False),
                getattr(attn, "num_local_heads", 0) * dcp_size,
                _sglang_mono_pre_supports_qrep(attn),
            )
            return None
        nhq, qn_rm = mp.NHQ_QREP, True
    else:
        nhq, qn_rm = mp.NHQ, False
    try:
        qkv_fmt, w_qkv, w_qkv_s = _mla_proj_operands(
            linear=fused, name="fused_qkv_a_proj_with_mqa", shape=(mp.LATW, HIDDEN)
        )
        g_fmt, w_g, w_g_s = _mla_proj_operands(
            linear=getattr(attn, "g_proj", None),
            name="g_proj",
            shape=(mp.GATE, HIDDEN),
        )
        qb_fmt, w_qb, w_qb_s = _mla_proj_operands(
            linear=getattr(attn, "q_b_proj", None),
            name="q_b_proj",
            shape=(mp.QB, mp.QL),
        )
    except ValueError as err:
        logger.info("monokernel: no MLA front for layer %d: %s", layer_id, err)
        return None
    if qkv_fmt != g_fmt:
        logger.info(
            "monokernel: no MLA front for layer %d: w_qkv format %d != g_proj format %d",
            layer_id,
            qkv_fmt,
            g_fmt,
        )
        return None
    if has_indexer and (qkv_fmt != wfmt.BF16 or qb_fmt != wfmt.BF16):
        logger.info(
            "monokernel: no MLA front for layer %d: indexer layers require bf16 projections",
            layer_id,
        )
        return None
    w = {
        "ares_nw": layer.self_attention_res_norm.weight,
        "ares_qk": layer.self_attention_res_proj.weight.view(HIDDEN),
        "in_nw": layer.input_layernorm.weight,
        "w_qkv": w_qkv,
        "w_qkv_s": w_qkv_s,
        "w_g": w_g,
        "w_g_s": w_g_s,
        "w_qb": w_qb,
        "w_qb_s": w_qb_s,
        "q_nw": attn.q_a_layernorm.weight,
        "kv_nw": attn.kv_a_layernorm.weight,
    }
    if has_indexer:
        mods = {"w_k": "wk", "w_p": "weights_proj", "w_iq": "wq_b"}
        biased = [
            m
            for m in mods.values()
            if getattr(getattr(idx, m, None), "bias", None) is not None
        ]
        if biased:  # stages 1 and 2 read the weights only
            logger.info(
                "monokernel: no MLA front for layer %d: biased %s", layer_id, biased
            )
            return None
        for n, m in mods.items():
            w[n] = getattr(getattr(idx, m, None), "weight", None)
        want_idx = {
            "w_k": (mp.IK, HIDDEN),
            "w_p": (mp.IW, HIDDEN),
            "w_iq": (mp.IQ, mp.QL),
        }
        for n, shp in want_idx.items():
            t = w[n]
            if not _valid_bf16_tensor(t, shp):
                logger.info("monokernel: no MLA front for layer %d: %s is %s", layer_id, n, None if t is None else (t.dtype, tuple(t.shape)))  # fmt: skip
                return None
    # the absorb's weight as SGLang's bf16 bmm multiplies it: w_kc itself, unscaled
    wkc, wsc = getattr(attn, "w_kc", None), getattr(attn, "w_scale", None)
    qn = (
        isinstance(wkc, torch.Tensor)
        and wkc.dtype == torch.bfloat16
        and tuple(wkc.shape) == (mp.NHQ, mp.QN, mp.KV)
        and isinstance(wsc, (int, float))
        and wsc == 1.0
    )
    if qn and not qn_rm:
        w["w_kc"] = wkc.transpose(1, 2).contiguous().view(mp.NHQ * mp.KV, mp.QN)
    key = mp.MlaPreBuild(
        tokens=rows,
        qn=qn,
        nhq=nhq,
        qn_rm=qn and qn_rm,
        has_indexer=has_indexer,
        qkv_fmt=qkv_fmt,
        qb_fmt=qb_fmt,
        nblocks=layer.prev_valid_blocks,
        eps=layer.self_attention_res_norm.variance_epsilon,
        out_eps=layer.input_layernorm.variance_epsilon,
        q_eps=attn.q_a_layernorm.variance_epsilon,
        kv_eps=attn.kv_a_layernorm.variance_epsilon,
        write_idx=(layer.prev_valid_blocks if layer.is_block_write_layer else -1),
    )
    return (key, w)


def _mla_outputs(rows: int, device, nhq: int | None = None) -> dict:
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

    nhq = mp.NHQ if nhq is None else nhq

    def z(n):
        return torch.zeros(rows, n, dtype=torch.bfloat16, device=device)

    return {
        "q": z(nhq * mp.QHD), "latent": z(mp.KV + mp.ROPE), "k_nope": z(mp.KV),
        "q_lora": z(mp.QL), "gate": z(mp.GATE), "idx_q": z(mp.IQ),
        "idx_k": z(mp.IK), "idx_w": z(mp.IW),
        "qn": torch.zeros(rows, nhq, mp.KV, dtype=torch.bfloat16, device=device),
    }  # fmt: skip


def _gu_interleaved(moe) -> bool:
    """Whether the experts carry the gate / up interleaved MXFP4 layout.

    ``is_shuffled`` alone cannot answer this: Mxfp4MoEMethod and Quark's MXFP4
    MoE set it on both of their branches, and only the SiTU branch AITER runs
    as A8W4 or A16W4 interleaves; ``AITER_SITUV2_A4W4`` alone keeps gate and up
    separated. Reading those bytes with the wrong layout does not fail, it
    returns wrong numbers, so ask the same question both methods asked.
    """
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )

    if getattr(moe.experts.moe_runner_config, "activation", None) != "situ":
        return False
    return _aiter_situ_uses_gu_interleaved_weights()


def _up_shard(moe, rank: int):
    """This rank's rows of the replicated latent up projection. Slicing the
    first dimension of a contiguous weight keeps it contiguous: a view."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import UP_N

    return moe.routed_expert_up_proj.weight.narrow(0, rank * UP_N, UP_N)


def _k2_launch(
    runner, packed: _Layer, core, prefix, blocks, out, prefix_in=None
) -> None:
    """The launch itself: this layer's weights read in place, the activations
    the caller owns."""
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    module = packed.module
    moe, experts = module.mlp, module.mlp.experts
    k2.k2_launch(
        packed.key,
        core=core,
        w_o=packed.w_o,
        w_os=packed.w_os,
        prefix=prefix,
        blocks=blocks,
        ares_nw=module.mlp_res_norm.weight,
        ares_qk=module.mlp_res_proj.weight.view(HIDDEN),
        in_nw=module.post_attention_layernorm.weight,
        w_gate=moe.gate.weight,
        bias=moe.gate.e_score_correction_bias,
        w_ld=moe.routed_expert_down_proj.weight,
        w_sgu=moe.shared_experts.gate_up_proj.weight,
        w_sd=moe.shared_experts.down_proj.weight,
        w13=experts.w13_weight,
        w13s=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2s=experts.w2_weight_scale,
        ln_w=moe.routed_expert_norm.weight,
        w_up=_up_shard(moe, runner.rank),
        out=out,
        prefix_in=prefix_in,
        **runner.launch_args(),
        # One tag a layer: a layer takes at most one launch in a step.
        layer=128 + packed.i,
    )


def _launch(runner, packed: _Layer, attn_res, core, forward_batch):
    """One layer, from the attention's TP-local core to the layer's output."""
    from sglang.srt.layers.layer_boundary.residual.batch import written_residual

    module = packed.module
    rows = core.shape[0]
    core = core.contiguous()
    # A block-write layer restarts the prefix, so it takes none.
    prefix_sum = (
        None if module.is_block_write_layer else written_residual(forward_batch)
    )
    _check(packed, core, prefix_sum, attn_res, rows)
    prefix = runner.prefix[:rows]
    out = torch.empty_like(prefix)
    # The kernel reads the stream from prefix_sum (lent out read-only), updates its
    # own prefix and writes out + prefix itself.
    _k2_launch(runner, packed, core, prefix, attn_res.block_residual, out, prefix_sum)
    return out


def _check(packed: _Layer, core, prefix_sum, attn_res, rows: int) -> None:
    """What the routing decision could not know, fail-loud.

    Every tensor reaches the kernel as a raw address, so a wrong dtype or
    stride is read as if it were the expected one. The step was routed for
    every rank at once, so falling back here is not an option.
    """
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    module = packed.module
    assert core.shape == (rows, CORE) and core.dtype == torch.bfloat16, (
        tuple(core.shape),
        core.dtype,
    )
    assert attn_res is not None, "the attention-residual stream is not open"
    # The kernel's two prefix modes are chosen at build time, so the caller's
    # mode has to be the one this layer was built for.
    assert (prefix_sum is None) == module.is_block_write_layer
    if prefix_sum is not None:
        assert (
            prefix_sum.shape == (rows, HIDDEN)
            and prefix_sum.dtype == torch.bfloat16
            and prefix_sum.is_contiguous()
        ), (tuple(prefix_sum.shape), prefix_sum.dtype)
    assert attn_res.num_valid_blocks == packed.key.nblocks, (
        attn_res.num_valid_blocks,
        packed.key.nblocks,
    )
    bank = attn_res.block_residual
    assert (
        bank.dtype == torch.bfloat16
        and bank.size(0) >= rows
        and bank.size(2) == HIDDEN
        and bank.stride(2) == 1
    ), (bank.dtype, tuple(bank.shape), bank.stride())


def _mla_covered(packed, hidden, bank, forward_batch=None) -> bool:
    """The MLA front takes a layer whose stream is a written tensor (not an owed
    contribution), with its bank open at the row count it was built for."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    key, _ = packed.mla
    if not (
        bank is not None
        and isinstance(hidden, torch.Tensor)
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and hidden.dim() == 2
        and hidden.shape[0] == key.tokens
        and hidden.shape[1] == HIDDEN
        and bank.num_valid_blocks == key.nblocks
        and bank.block_residual.dtype == torch.bfloat16
    ):
        return False
    if forward_batch is None:
        return key.nhq == mp.NHQ
    attn = packed.module.self_attn
    q_rep_active = bool(
        getattr(attn, "dcp_size", 1) > 1
        and attn.is_dcp_q_replicate_phase(forward_batch)
        and not attn.use_deep_gemm_bmm
        and attn.w_kc_qrep is not None
        and attn.q_b_proj_qrep_weight is not None
    )
    if key.nhq > mp.NHQ:
        if not (
            q_rep_active
            and _valid_bf16_tensor(attn.q_b_proj_qrep_weight, (key.nhq * mp.QHD, mp.QL))
        ):
            return False
        if key.qn and not _valid_bf16_tensor(attn.w_kc_qrep, (key.nhq, mp.QN, mp.KV)):
            return False
        return True
    return not q_rep_active


class _LanguageModelShim:
    """``general_mm_embed_routine``'s language model for a routed step.

    The routine reads the real model (``get_input_embeddings``, ``pp_group``)
    and then calls it; this hands the reads through and takes the call.
    """

    def __init__(self, built: _Built) -> None:
        self._built, self._lm = built, built.lm

    def __getattr__(self, name):
        return getattr(self._lm, name)

    def __call__(self, input_ids=None, forward_batch=None, input_embeds=None, **kw):
        return self._built.language_forward(
            input_ids, forward_batch, input_embeds, **kw
        )


class _Built:
    """One rank's monokernels: the runner, each layer's weights, and the forward."""

    def __init__(self, model, runner, by_width: dict, k1_by_width: dict) -> None:
        self.model = model
        self.lm = getattr(model, "language_model", model)
        self.backbone = self.lm.model
        # {launch width: {layer id: its packed launch}}; a layer's build
        # key carries the width, so each width has its own compiled kernel
        self.runner, self._by_width = runner, by_width
        # {launch width: that width's K1 layers as one batch}, None where K1
        # took no layer
        self._k1_by_width = k1_by_width
        # This step's shared K1 state and the layers that took it, set by
        # _backbone before the first layer and read by _core and finish().
        self._k1_state, self._k1_engaged = None, []
        # Read once: _core asks this for every layer of every step, and a
        # served step must not pay for a check that is off.
        self._check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        # The conv windows waiting for finish(), under that flag only.
        self._check_handoff = []
        self._shim = _LanguageModelShim(self)
        # The MLA front's outputs, one set a width (the layers run in sequence)
        dev = runner.device
        self._mla_out = {
            rows: _mla_outputs(
                rows,
                dev,
                nhq=max(p.mla[0].nhq for p in packed.values() if p.mla is not None),
            )
            for rows, packed in by_width.items()
            if any(p.mla is not None for p in packed.values())
        }

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        """``KimiK3ForConditionalGeneration.forward`` for a routed step: SGLang's
        multimodal routine around the backbone, which it calls through the shim."""
        from sglang.srt.managers.mm_utils import general_mm_embed_routine
        from sglang.srt.managers.schedule_batch import Modality

        self._plan, self._positions = plan, positions
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self._shim,
            data_embedding_funcs={Modality.IMAGE: self.model.get_image_feature},
            positions=positions,
            pp_proxy_tensors=kwargs.get("pp_proxy_tensors"),
        )

    def language_forward(self, input_ids, forward_batch, input_embeds, **kw):
        """``KimiK3LinearForCausalLM.forward``: the backbone, then the head."""
        lm = self.lm
        hidden = self._backbone(input_ids, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            hidden, aux = hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, input_ids, forward_batch, input_embeds):
        """``KimiK3LinearModel.forward``: the embedding, every layer, the final
        norm. Pipeline and DP attention are refused, so this rank is the first
        and the last, and the bank is always open."""
        from sglang.srt.layers.layer_boundary.residual.batch import final_norm, start
        from sglang.srt.models.kimi_k3 import AuxHiddenStateList, _cdiv
        from sglang.srt.utils.common import BumpAllocator

        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        m = self.backbone
        self.runner.step_begin()
        # Once a step, before the first layer: every K1 layer's conv window
        # staged in one write. _k1_engaged collects the layers that take it,
        # which is what finish() hands back for.
        k1_batch = self._k1_by_width[self._plan.key.rows]
        self._k1_state = k1_mod.prepare(k1_batch, forward_batch)
        self._k1_engaged = []
        self._check_handoff = []
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        start(forward_batch)
        zero_allocator = BumpAllocator(
            buffer_size=2 * (m.end_layer - m.start_layer),
            dtype=torch.float32,
            device=hidden.device,
        )
        m.attn_bank.open(hidden, _cdiv(m.end_layer, m.config.attn_res_block_size), None)
        # The bank's stream is captured after a layer, not at the next layer's
        # input (``captures_at_input`` is False whenever the bank is open).
        aux = AuxHiddenStateList()
        captures = m.dspark_layers_to_capture or ()
        for i in range(m.start_layer, m.end_layer):
            hidden = self._layer(i, hidden, forward_batch, zero_allocator)
            if i in captures:
                aux.append(m._dspark_capture_stream(i, hidden, forward_batch))
        # Once a step, after the last layer: the conv inputs handed back to
        # SGLang's speculative window, which only reads them at accept.
        k1_mod.finish(k1_batch, self._k1_state, self._k1_engaged)
        if self._check:
            from sgl_amd_mono.kimi_k3 import check as check_mod

            # The conv window is handed back once a step, so it is scored here
            # and not in the layer that produced it.
            check_mod.compare_handoff(self)
        hidden = final_norm(
            hidden, forward_batch, m._final_read, capture=None, skip_empty=True
        )
        m.attn_bank.close()
        if m.dspark_layers_to_capture is not None:
            return hidden, aux
        return hidden

    def _layer(self, i, hidden, forward_batch, zero_allocator):
        """One layer: the launch where it is covered, else the model's own."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        layer = self.backbone.layers[i]
        packed = self._by_width[self._plan.key.rows].get(i)
        if packed is None:  # the dense prefix keeps the model's forward
            return layer(
                positions=self._positions,
                hidden_states=hidden,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                capture_gathered=None,
            )
        bank = self.backbone.attn_bank.current
        core = self._core(layer, packed, hidden, bank, forward_batch, zero_allocator)
        out = _launch(self.runner, packed, bank, core, forward_batch)
        # The launch folded the residual in, so the stream is written rather
        # than finished through ffn_boundary.
        return set_written(out, forward_batch)

    def _core(self, layer, packed, hidden, bank, forward_batch, zero_allocator):
        """This layer's TP-local attention output, which the layer launch takes.

        A KDA layer takes the K1 launch where it is covered: K1 runs the
        attention-side seam, ``input_layernorm`` and the whole KDA attention in
        one launch, so it replaces both the boundary's prepare and the model's
        attention. K1 reads the residual stream and does not add to it -- the
        attention's contribution is added at the MLP seam, which is K2's -- so
        the stream is handed on unchanged, which is what the boundary's prepare
        would have left for K2's ``written_residual``.

        Everything else -- the full-attention layers, and a KDA layer K1
        declines -- keeps SGLang's path. K1 writes nothing before it declines,
        so that fallback is clean.
        """
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        state = self._k1_state
        if packed.mla is not None and _mla_covered(packed, hidden, bank, forward_batch):
            return self._mla_core(
                layer, packed, hidden, bank, forward_batch, zero_allocator
            )
        if packed.k1 is not None and k1_mod.covered(packed.k1, hidden, bank, state):
            ref = None
            if self._check:
                from sgl_amd_mono.kimi_k3 import check as check_mod

                # Before K1, on the same input and the caches as they were.
                ref = check_mod.reference(
                    self, layer, packed, hidden, bank, forward_batch, zero_allocator
                )
            set_written(hidden, forward_batch)
            self._k1_engaged.append(packed.i)
            core = k1_mod.launch(self.runner, packed.k1, hidden, bank, state)
            if self._check:
                from sgl_amd_mono.kimi_k3 import check as check_mod

                check_mod.compare(self, packed.i, core, ref, bank)
            return core
        x = layer.attn_boundary.prepare(hidden, forward_batch, capture_gathered=None)
        return self._pre_o_proj(layer, x, forward_batch, zero_allocator)

    def _mla_core(self, layer, packed, hidden, bank, forward_batch, zero_allocator):
        """An MLA layer: the projection front as one launch, then SGLang's MLA
        forward (RoPE, the KV write, the indexer's norms / quantization / logits /
        top-k, the sparse attention) on its outputs, up to o_proj's input."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
        from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

        key, w = packed.mla
        rows = hidden.shape[0]
        attn = layer.self_attn
        out = _mla_out_view(self._mla_out[rows], key.nhq)
        w_run = w
        if key.nhq > mp.NHQ:
            w_run = {**w, "w_qb": attn.q_b_proj_qrep_weight}
            if key.qn:
                w_run["w_kc"] = attn.w_kc_qrep
        # The launch reads the stream and does not add to it, as K1 does.
        set_written(hidden, forward_batch)
        mp.mla_pre_launch(
            key, prefix=hidden, blocks=bank.block_residual, w=w_run, out=out,
            scratch=self.runner.scratch, layer=packed.i, epoch=self.runner.epoch,
        )  # fmt: skip
        if key.write_idx >= 0:
            bank._count_write(None)  # the launch wrote the bank row
        off, nbytes = mp.scratch_layout(key)["x"]
        x = (
            self.runner.scratch[off : off + nbytes]
            .view(torch.bfloat16)
            .view(rows, HIDDEN)
        )
        attn.__dict__["_mono_pre"] = {
            "q": out["q"], "latent": out["latent"], "k_nope": out["k_nope"],
            "q_lora": out["q_lora"], "gate": out["gate"],
            "q_nope_out": out["qn"] if key.qn else None,
        }  # fmt: skip
        idx = getattr(attn, "indexer", None)
        if idx is not None:
            idx.__dict__["_mono_pre"] = {
                "q": out["idx_q"], "k": out["idx_k"], "w": out["idx_w"],
            }  # fmt: skip
        try:
            return self._pre_o_proj(layer, x, forward_batch, zero_allocator)
        finally:
            attn.__dict__.pop("_mono_pre", None)
            if idx is not None:
                idx.__dict__.pop("_mono_pre", None)

    def _pre_o_proj(self, layer, x, forward_batch, zero_allocator):
        """``KimiK3DecoderLayer._run_self_attn`` stopped before o_proj: the
        attention's TP-local output, which is where the launch starts."""
        from sglang.srt.layers.layer_boundary import (
            AttentionInputs,
            get_attn_tp_context,
        )

        attn = layer.self_attn
        latent = getattr(attn, "prepare_qkv_latent", None)
        if latent is not None:
            # MLA layers with q_lora_rank read their latent inputs from here.
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(x, forward_batch, latent)
            )
        try:
            return attn.forward_pre_o_proj(
                hidden_states=x,
                positions=self._positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
        finally:
            if latent is not None:
                get_attn_tp_context().clear_attn_inputs()


class _WideBuilt(_Built):
    """Narrow launches for <= MAX_ROWS rows, and the wide KDA layer launch above (None
    when this draft length serves none of the routed wide widths)."""

    def __init__(self, narrow, wide_runner) -> None:
        self.__dict__.update(narrow.__dict__)
        self._wide = wide_runner
        self._shim = _LanguageModelShim(self)

    def _backbone(self, input_ids, forward_batch, input_embeds):
        rows = self._plan.key.rows
        if rows <= MAX_ROWS:
            return super()._backbone(input_ids, forward_batch, input_embeds)
        if self._wide is None or not self._wide.serves(rows):
            # a multiple of this L in the spec's widths only for another draft length:
            # SGLang's own backbone, as for a step past the widths
            return self.backbone(
                input_ids, self._positions, forward_batch, input_embeds
            )
        try:
            return self._wide.backbone(self, input_ids, forward_batch, input_embeds)
        except Exception as e:
            import traceback

            from sgl_amd_mono.kimi_k31 import wide

            wide._say(
                f"wide backbone raised {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()[-2000:]}"
            )
            raise
