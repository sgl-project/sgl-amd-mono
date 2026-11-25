# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (SGLang's Qwen4-Exp) on gfx950 at TP2: the decode path of
``qwen38_flash_next.spec.SPEC`` (RFC sgl-project/sglang#43949), which routes TP2
servers here. Plain decode (no speculative decoding) of the MXFP4 checkpoint with
BF16 attention and GDN projections, 1 to 192 rows a step.

A routed decode step runs ``_Built.forward`` in place of
``Qwen4ExpForConditionalGeneration.forward``, in the order SGLang's forward runs:

    embedding    model.model.embed_tokens (decode rows are text tokens)
    streams      the embedding widened to the 4 HC streams
    PLE          _prepare_ple_batch, the next PLE layer's start_prefetch one layer
                 ahead, state + layer.ple(...) before that layer's launch,
                 _commit_ple_batch after the last layer
    GDN layer    the ``gdn`` launch (``launcher``): the whole layer, its GDN states
                 updated in place; then SGLang's decode track-save
    QSA layer    the ``qsa_front`` launch (attention mix, qkv_proj and
                 index_qk_proj), SGLang's QSA attention fed those projections
                 (``qsa_core``), the ``qsa_back`` launch (o_proj and the MLP half)
    final mix    model.model.hyper_connection_mixer.mix
    logits       model.logits_processor; output.hidden_states = the streams

The first launch of a step and the one after a PLE layer compute the state's
norms first (``stats``). The GDN states are SGLang's, updated at the step's mamba
cache indices; a pad row (index <= 0) writes none. ``SGLANG_AMD_MONO_CHECK=1``
(eager) also runs SGLang's layer next to each layer's launches (``check``).
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import contextmanager

import torch

from sgl_amd_mono.qwen38_flash_next.decode import decode_widths
from sgl_amd_mono.qwen38_flash_next.decode.weights import is_gdn, moe_rip, scale_cols
from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L

logger = logging.getLogger(__name__)

TP_SIZES = (L.TP,)
# SGLang's decode CUDA graph capture sizes without speculative decoding up to
# MAX_TOKENS (``generate_decode_cuda_graph_batch_sizes``), which the binding makes
# the capture sizes up to the server's largest (``covered_capture_sizes``)
WIDTHS = (1, 2, 4, 8, 12, *range(16, L.MAX_TOKENS + 1, 8))
# The server arguments this path accepts on top of ``SPEC.requires``, checked as
# SGLang checks those
SERVER_ARGS = {
    # a request's verify rows would update one GDN state in sequence
    "speculative_algorithm": frozenset({None}),
    # the kernels keep the GDN state in f32
    "mamba_ssm_dtype": frozenset({None, "float32"}),
    # both lay a slot's states out over every layer's (an envelope), whose slot
    # pitch and alignment the conv state reads do not take
    "enable_unified_memory": frozenset({False}),
    "enable_page_major_kv_layout": frozenset({False}),
}
# the text config's fields the kernels are built for, and their values
HF_CONFIG = {**dict(zip(L.SHAPE_NAMES, L.SHAPES)), "hidden_act": "silu"}
GDN_LAYERS, QSA_LAYERS = 36, 12

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "rows",
        "forward_mode",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/arg_groups/fields/*.py": (
        "speculative_algorithm",
        "mamba_ssm_dtype",
        "enable_unified_memory",
        "enable_page_major_kv_layout",
        "linear_attn_backend",
        "linear_attn_decode_backend",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype", "conv", "temporal"),
    "srt/configs/qwen4_exp.py": ("text_config", "hc_count", "hc_lowrank"),
    "srt/configs/qwen3_next.py": (
        "hidden_size",
        "hidden_act",
        "norm_topk_prob",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_size"',
        '"tp_rank"',
        '"tp_group"',
    ),
    "srt/distributed/parallel_state.py": ("cpu_group",),
    "srt/models/qwen3_vl.py": ("logits_processor", "lm_head", "pooler"),
    "srt/models/qwen3_5.py": (
        "model",
        "layers",
        "embed_tokens",
        "linear_attn",
        "in_proj_qkvz",
        "in_proj_ba",
        "conv1d",
        "out_proj",
        "A_log",
        "dt_bias",
        "norm",
        "activation",
        "o_proj",
        "qkv_proj",
        "mlp",
        "attn",
        "_prepare_qkv_gate",
        "_prepare_qkv_gate(positions)",
        "_prepare_qkv_gate(hidden_states)",
        "_prepare_qkv_gate(forward_batch)",
    ),
    "srt/models/qwen4_exp.py": (
        "_prepare_ple_batch",
        "_prepare_ple_batch(input_ids)",
        "_prepare_ple_batch(forward_batch)",
        "_prepare_ple_batch(ngram_size)",
        "_prepare_ple_batch(ngram_eos_token_id)",
        "_commit_ple_batch",
        "_commit_ple_batch(batch)",
        "_commit_ple_batch(forward_batch)",
        "start_prefetch",
        "start_prefetch(batch)",
        "start_prefetch(forward_batch)",
        "ple",
        "has_ple",
        "ple_ngram_size",
        "ple_ngram_eos_token_id",
        "hyper_connection_mixer",
        "mix",
        "last_hc_hidden_states",
        "attn_hyper_connection",
        "mlp_hyper_connection",
        "attn_output_gate",
        "is_qsa",
        "indexer",
        "_compute_qsa_topk_indices",
        "is_mrope_enabled",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": ("index_qk_proj",),
    "srt/layers/hyperconnection.py": (
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
        "hc_norm",
        "variance_epsilon",
    ),
    "srt/layers/linear.py": ("output_sizes",),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "activation",
        "norm_before_gate",
        "group_size",
    ),
    "srt/models/qwen2_moe.py": ("gate", "shared_expert_gate", "experts"),
    "srt/layers/moe/fused_moe_triton/layer.py": ("num_fused_shared_experts",),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "is_shuffled",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "MambaAttnBackendBase",
        "linear_attn_backend",
        "forward_metadata",
        "state_layer_cache",
        "_track_mamba_state_decode",
        "_track_mamba_state_decode(forward_batch)",
        "_track_mamba_state_decode(conv_states)",
        "_track_mamba_state_decode(ssm_states)",
        "_track_mamba_state_decode(cache_indices)",
        "_track_mamba_state_decode(layer_id)",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    "srt/mem_cache/memory_pool.py": ("conv", "temporal"),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
}

# Each layer's launches against SGLang's own layer on the same input streams
# (``check``). The launches decode the GDN projections' MXFP4 copies where SGLang
# runs the bf16 weights, so a GDN layer's own contribution (delta, update) is held
# to a quantization-sized bound; a wrong layout or stride lands near 0. The fed
# projections are the same bf16 GEMMs summed in another order.
BOUNDS = {
    "qsa.mix.cosine": 0.999,
    "qsa.qkv.cosine": 0.9999,
    "qsa.ix.cosine": 0.9999,
    "gdn.streams.cosine": 0.999,
    "gdn.delta.cosine": 0.9,
    "gdn.conv.cosine": 0.98,
    "gdn.state.cosine": 0.999,
    "gdn.update.cosine": 0.9,
    "qsa.streams.cosine": 0.999,
    "qsa.delta.cosine": 0.9,
    # SGLang's radix track-save copies the states
    "gdn.track.max_abs": 0.0,
}

# ``SGLANG_AMD_MONO_CHECK_REF=mxfp4``: the reference's GDN projections run the
# kernels' operands (the MXFP4 copies decoded, bf16 activations), so the GDN
# metrics without the MoE are held to rounding-order bounds. Streams and delta
# carry the two MoE implementations' difference. In 13 check runs (MI355X and
# MI350X, 1 to 192 rows, contexts of about 55, 6,000 and 108,000 tokens) their
# lowest were 0.9879 (streams) and 0.9407 (delta). With a layer broken on purpose
# (one rank's attention or expert half lost, every attention output zeroed, the
# router rows shifted by one, the shared expert's gate zeroed) each width's lowest
# was at most 0.948 and 0.717. Each bound sits midway between the two in
# log(1 - cosine).
CHECK_MXFP4_BOUNDS = {
    "qsa.mix.cosine": 0.999,
    "qsa.qkv.cosine": 0.9999,
    "qsa.ix.cosine": 0.9999,
    "gdn.streams.cosine": 0.975,
    "gdn.delta.cosine": 0.87,
    "gdn.conv.cosine": 0.9995,
    "gdn.state.cosine": 0.9999,
    "gdn.update.cosine": 0.999,
    "qsa.streams.cosine": 0.975,
    "qsa.delta.cosine": 0.87,
    "gdn.track.max_abs": 0.0,
}


def check_bounds(ref: str) -> dict[str, float]:
    """The check's bounds for its reference (``check.REFS``)."""
    return dict(CHECK_MXFP4_BOUNDS if ref == "mxfp4" else BOUNDS)


def _text(config):
    return getattr(config, "text_config", config)


def config_disable_reasons(cfg, hf_config) -> list[str]:
    """What the TP2 decode path refuses before load, past ``SERVER_ARGS``."""
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    text = _text(hf_config)
    reasons = [
        f"needs Qwen3.8-Flash-Next's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    if not getattr(text, "norm_topk_prob", True):
        reasons.append("norm_topk_prob false: the router renormalizes the top-10")
    quant = (getattr(hf_config, "quantization_config", None) or {}).get("quant_method")
    if quant != "quark":
        reasons.append(f"needs the Quark MXFP4 checkpoint, got {quant!r}")
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal != torch.float32 or dtypes.conv != torch.bfloat16:
        reasons.append(
            f"GDN state {dtypes.temporal} / conv {dtypes.conv}: the kernels update "
            "f32 / bf16 in place"
        )
    # the step's GDN slots come from the triton backend's decode metadata
    decode = cfg.linear_attn_decode_backend or cfg.linear_attn_backend
    if decode != "triton":
        reasons.append(
            f"GDN decode backend {decode}: the launches read triton's metadata"
        )
    if os.environ.get("SGLANG_AMD_MONO_ROCM_LLVM"):
        reasons.append(
            "SGLANG_AMD_MONO_ROCM_LLVM: the decode kernels are validated with "
            "FlyDSL's bundled LLVM"
        )
    return reasons


def _get(obj, path: str):
    for name in path.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _brief(t) -> str:
    return f"{tuple(t.shape)} {t.dtype}" if isinstance(t, torch.Tensor) else "missing"


def _expect(reasons, where, t, shape, dtype=torch.bfloat16) -> None:
    """``t``: ``shape`` ``dtype``, dense (the kernels read it in place)."""
    if not isinstance(t, torch.Tensor):
        reasons.append(f"{where}: missing")
    elif tuple(t.shape) != tuple(shape) or t.dtype != dtype:
        reasons.append(f"{where}: {_brief(t)}, expected {tuple(shape)} {dtype}")
    elif not t.is_contiguous():
        reasons.append(f"{where}: not contiguous")


def _linear(reasons, where, linear, shape) -> None:
    _expect(reasons, where, _get(linear, "weight"), shape)
    if _get(linear, "bias") is not None:
        reasons.append(f"{where} has a bias")


def _gdn_reasons(reasons: list[str], where: str, attn) -> None:
    for name, shape in (
        ("in_proj_qkvz", (L.QKVZ, L.HIDDEN)),
        ("in_proj_ba", (L.BA, L.HIDDEN)),
        ("out_proj", (L.HIDDEN, L.VALUE_DIM)),
    ):
        _linear(reasons, f"{where} {name}", _get(attn, name), shape)
    _expect(reasons, f"{where} A_log", _get(attn, "A_log"), (L.NV,), torch.float32)
    dt_bias = _get(attn, "dt_bias")  # copied to bf16 if held otherwise
    if not (
        isinstance(dt_bias, torch.Tensor)
        and dt_bias.is_floating_point()
        and tuple(dt_bias.shape) == (L.NV,)
    ):
        reasons.append(f"{where} dt_bias: {_brief(dt_bias)}, expected ({L.NV},) float")
    _expect(reasons, f"{where} norm", _get(attn, "norm.weight"), (L.GDN_DV,))
    conv = _get(attn, "conv1d.weight")
    if not (
        isinstance(conv, torch.Tensor)
        and conv.dtype == torch.bfloat16
        and conv.numel() == L.CONV_DIM * L.CONV_TAPS
        and conv.is_contiguous()
    ):
        reasons.append(
            f"{where} conv1d: {_brief(conv)}, expected {L.CONV_DIM} x {L.CONV_TAPS} "
            "bf16"
        )
    sizes = _get(attn, "in_proj_qkvz.output_sizes")
    split = [L.KEY_DIM, L.KEY_DIM, L.VALUE_DIM, L.VALUE_DIM]
    if sizes is not None and [x // L.TP for x in sizes] != split:
        reasons.append(
            f"{where} in_proj_qkvz output_sizes {sizes}, expected [q, k, v, z]"
        )
    # the launch's GDN: SiLU after a bias-free conv, norm(o) * sigmoid(z) per head
    activation = _get(attn, "activation")
    if activation not in ("silu", "swish"):
        reasons.append(
            f"{where} conv activation {activation!r}, the launch applies SiLU"
        )
    if _get(attn, "conv1d.bias") is not None:
        reasons.append(f"{where} conv1d has a bias")
    norm = _get(attn, "norm")
    if getattr(norm, "activation", None) != "sigmoid":
        reasons.append(
            f"{where} gated norm activation {getattr(norm, 'activation', None)!r}, "
            "the launch gates with sigmoid"
        )
    if (
        getattr(norm, "norm_before_gate", None) is not True
        or getattr(norm, "group_size", None) is not None
    ):
        reasons.append(
            f"{where} gated norm variant: norm(o) * gate(z) over the head expected"
        )


def _qsa_reasons(reasons: list[str], where: str, layer) -> None:
    # the stage imports FlyDSL, which the config refusals check for: imported
    # here, after them
    from sgl_amd_mono.qwen38_flash_next.decode.wide.stages import qsa as qsa_

    _linear(reasons, f"{where} o_proj", _get(layer, "o_proj"), (L.HIDDEN, L.QSA_CORE))
    # qsa_front projects x through both, row-major [rows, HIDDEN]
    _linear(reasons, f"{where} qkv_proj", _get(layer, "qkv_proj"), (qsa_.QKV, L.HIDDEN))
    _linear(
        reasons,
        f"{where} index_qk_proj",
        _get(layer, "indexer.index_qk_proj"),
        (qsa_.IX, L.HIDDEN),
    )
    if getattr(layer, "attn_output_gate", True) is not True:
        reasons.append(f"{where}: the qsa_back launch applies the output gate")
    if getattr(layer, "is_qsa", False) is not True:
        reasons.append(f"{where}: the launches feed the QSA indexer")


def _hc_eps(reasons: list[str], where: str, layer) -> list[float]:
    eps = []
    for branch in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = getattr(layer, branch, None)
        for name, path, shape in (
            ("down", "input_mix_weight_down.weight", (L.LOWRANK, L.HC_HIDDEN)),
            ("up", "input_mix_weight_up.weight", (L.HC_HIDDEN, L.LOWRANK)),
            ("inject", "block_inject_weight.weight", (L.HC, L.HC_HIDDEN)),
        ):
            _expect(reasons, f"{where} {branch} {name}", _get(hc, path), shape)
        norm = _get(hc, "hc_norm.weight")  # copied to bf16 if held otherwise
        if not isinstance(norm, torch.Tensor) or tuple(norm.shape) != (L.HC_HIDDEN,):
            reasons.append(
                f"{where} {branch} norm: {_brief(norm)}, expected ({L.HC_HIDDEN},)"
            )
        eps.append(_get(hc, "hc_norm.variance_epsilon"))
    return eps


def _moe_reasons(reasons: list[str], where: str, mlp):
    """The experts' padded intermediate, or None (a reason recorded)."""
    _expect(reasons, f"{where} router", _get(mlp, "gate.weight"), (L.EXPERTS, L.HIDDEN))
    _expect(
        reasons,
        f"{where} shared_expert_gate",
        _get(mlp, "shared_expert_gate.weight"),
        (1, L.HIDDEN),
    )
    if _get(mlp, "experts.num_fused_shared_experts") != 1:
        reasons.append(
            f"{where}: the kernels read the shared expert as expert {L.EXPERTS}"
        )
    w13, w2 = _get(mlp, "experts.w13_weight"), _get(mlp, "experts.w2_weight")
    if not (isinstance(w13, torch.Tensor) and isinstance(w2, torch.Tensor)):
        reasons.append(f"{where} experts: missing w13_weight / w2_weight")
        return None
    try:
        rip = moe_rip(w13, w2)
    except ValueError as e:
        reasons.append(f"{where} experts: {e}")
        return None
    if w13.shape[0] != L.E_ALL:
        reasons.append(f"{where} experts: {w13.shape[0]} slots, expected {L.E_ALL}")
        return None
    # AITER's shuffled layout: one e8m0 a 32-block, K blocks padded to 8
    for name, rows, k in (("w13", 2 * rip, L.HIDDEN), ("w2", L.HIDDEN, rip)):
        scale = _get(mlp, f"experts.{name}_weight_scale")
        want = L.E_ALL * rows * scale_cols(k)
        if (
            not isinstance(scale, torch.Tensor)
            or scale.element_size() != 1
            or scale.numel() != want
            or not scale.is_contiguous()
        ):
            reasons.append(
                f"{where} {name} scales: {_brief(scale)}, expected {want} dense e8m0"
            )
    for name, w in (("w13", w13), ("w2", w2)):
        if getattr(w, "is_shuffled", None) is False:
            reasons.append(f"{where} {name}: not in AITER's shuffled layout")
        if not w.is_contiguous():
            reasons.append(f"{where} {name}: not contiguous")
    return rip


def layout_reasons(model) -> list[str]:
    """What the kernels refuse in the loaded model (one TP2 rank's modules)."""
    reasons = []
    layers = list(model.model.layers)
    kinds = ["gdn" if is_gdn(x) else "qsa" for x in layers]
    if (kinds.count("gdn"), kinds.count("qsa")) != (GDN_LAYERS, QSA_LAYERS):
        reasons.append(
            f"{kinds.count('gdn')} GDN and {kinds.count('qsa')} QSA layers, expected "
            f"{GDN_LAYERS} and {QSA_LAYERS}"
        )
    rips, eps = set(), set()
    for i, layer in enumerate(layers):
        where = f"layer {i}"
        if is_gdn(layer):
            _gdn_reasons(reasons, where, layer.linear_attn)
        else:
            _qsa_reasons(reasons, where, layer)
        eps.update(_hc_eps(reasons, where, layer))
        rip = _moe_reasons(reasons, where, layer.mlp)
        if rip is not None:
            rips.add(rip)
    if len(rips) > 1:
        reasons.append(f"the layers' experts are padded differently: {sorted(rips)}")
    # a launch's combine computes the next mix's norms: one epsilon for all
    if len(eps) > 1:
        reasons.append(f"the HC norms' epsilons differ: {sorted(eps, key=str)}")
    return reasons


# What ``_Built.forward`` passes to SGLang's decode track-save, after ``self``
TRACK_SAVE_ARGS = (
    "forward_batch",
    "conv_states",
    "ssm_states",
    "cache_indices",
    "layer_id",
)


def backend_reasons() -> list[str]:
    """The forward reads each GDN layer's states through the linear-attention
    backend (``state_layer_cache``, sgl-project/sglang#46980) and calls its decode
    track-save after each GDN launch (a copy with the radix cache on, a no-op with
    it off): refused when the running SGLang's backend lacks either. Checked after
    load: the backend module imports the model runner, which resolving server args
    must not."""
    where = (
        "sglang.srt.layers.attention.hybrid_linear_attn_backend.MambaAttnBackendBase"
    )
    try:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )
    except ImportError as e:
        return [f"linear-attention backend: cannot import {where}: {e}"]
    reasons = []
    if not callable(getattr(MambaAttnBackendBase, "state_layer_cache", None)):
        reasons.append(
            f"GDN state read: {where} has no state_layer_cache "
            "(sgl-project/sglang#46980)"
        )
    fn = getattr(MambaAttnBackendBase, "_track_mamba_state_decode", None)
    params = tuple(inspect.signature(fn).parameters)[1:] if callable(fn) else None
    if params is None or params[: len(TRACK_SAVE_ARGS)] != TRACK_SAVE_ARGS:
        reasons.append(
            f"radix track-save: {where}._track_mamba_state_decode{params} does not "
            f"take {TRACK_SAVE_ARGS}"
        )
    return reasons


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    return layout_reasons(model) + backend_reasons()


def gdn_state_reasons(conv: torch.Tensor, ssm: torch.Tensor) -> list[str]:
    """The GDN states the kernels update in place: the conv state [slots,
    CONV_DIM, 3] bf16, a slot's rows dense (``conv_dim_first``: a row's taps read
    8 bytes at a time, so a slot's pitch is a multiple of 4 elements and the base
    8-byte aligned), and the recurrent state [slots, NV, DV, DK] f32, a slot
    dense."""
    reasons = []
    taps = L.CONV_TAPS - 1
    if (
        conv.dtype != torch.bfloat16
        or conv.dim() != 3
        or tuple(conv.shape[1:]) != (L.CONV_DIM, taps)
    ):
        reasons.append(
            f"conv state {tuple(conv.shape)} {conv.dtype}, expected "
            f"[slots, {L.CONV_DIM}, {taps}] bfloat16"
        )
    elif (
        tuple(conv.stride()[1:]) != (taps, 1)
        or conv.stride(0) % 4
        or conv.data_ptr() % 8
    ):
        reasons.append(
            f"conv state strides {conv.stride()} at {conv.data_ptr():#x}: a slot's "
            f"[{L.CONV_DIM}, {taps}] dense, its pitch a multiple of 4 elements and "
            "the base 8-byte aligned expected"
        )
    if (
        ssm.dtype != torch.float32
        or ssm.dim() != 4
        or tuple(ssm.shape[1:]) != (L.NV, L.GDN_DV, L.GDN_DK)
    ):
        reasons.append(
            f"recurrent state {tuple(ssm.shape)} {ssm.dtype}, expected "
            f"[slots, {L.NV}, {L.GDN_DV}, {L.GDN_DK}] float32"
        )
    elif not ssm[0].is_contiguous():
        reasons.append(f"recurrent state slot not dense: strides {ssm.stride()}")
    return reasons


class _Output(torch.nn.Module):
    """A linear's stand-in: ``out`` (qsa_front's projection of the step's x) for
    its input's rows, as SGLang's linears return (output, bias)."""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.out = out

    def forward(self, x):
        return self.out[: x.shape[0]], None


@contextmanager
def fed_projections(layer, qkv: torch.Tensor, ixqk: torch.Tensor):
    """For the block, the layer's ``qkv_proj`` and its indexer's ``index_qk_proj``
    return the launch's projections of x (``qkv_out``, ``ix_out``), so SGLang's
    attention runs on them. ponytail: a module swap around SGLang's calls; the
    upgrade is an attention entry in SGLang's Qwen4-Exp layer that takes them."""
    swaps = ((layer, "qkv_proj", qkv), (layer.indexer, "index_qk_proj", ixqk))
    saved = [getattr(module, name) for module, name, _ in swaps]
    for module, name, out in swaps:
        setattr(module, name, _Output(out))
    try:
        yield
    finally:
        for (module, name, _), proj in zip(swaps, saved):
            setattr(module, name, proj)


def qsa_core(layer, x, qkv, ixqk, positions, forward_batch):
    """A QSA layer's ``self_attention`` up to o_proj's input, on the launch's
    projections of ``x`` -> (attention output [S, QSA_CORE], output gate [S,
    QSA_CORE]); the qsa_back launch applies the gate."""
    with fed_projections(layer, qkv, ixqk):
        q, k, v, gate = layer._prepare_qkv_gate(
            positions=positions, hidden_states=x, forward_batch=forward_batch
        )
        topk = layer._compute_qsa_topk_indices(x, positions, forward_batch)
    out = layer.attn(q, k, v, forward_batch, topk_indices=topk)
    s = x.shape[0]
    return out.reshape(s, -1).contiguous(), gate.reshape(s, -1).contiguous()


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` (a CPU group; None for one rank) passed
    ``ok`` true. Every rank calls it at the same point."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _on_every_rank(fn, group, what: str):
    """``fn()``, raised on every rank if it raised on any: a rank that raised alone
    would leave the others in the next collective."""
    failure, out = None, None
    try:
        out = fn()
    except Exception as e:
        logger.exception("Qwen3.8-Flash-Next monokernel: %s failed", what)
        failure = e
    if not tp_agree(failure is None, group):
        raise RuntimeError(
            f"Qwen3.8-Flash-Next monokernel: {what} failed on "
            f"{'this' if failure else 'another'} rank"
        ) from failure
    return out


class _Built:
    """One rank's launches: the launcher (operands, buffers, builds) and the
    forward."""

    def __init__(self, model, launcher):
        self.model, self.la = model, launcher
        self.first_seen: set = set()
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        if self.check:
            from sgl_amd_mono.qwen38_flash_next.decode.check import reference_mode

            self.check_ref = reference_mode()

    def _gdn_state(self, i: int, lab) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer i's GDN states in SGLang's pool, read as SGLang's GDN decode
        reads them, bound into the layer's launch arguments when they move."""
        cache = lab.state_layer_cache(i)
        conv, ssm = cache.conv[0], cache.temporal
        if not self.la.bound(i, conv, ssm):
            why = gdn_state_reasons(conv, ssm)
            if why:
                raise RuntimeError(
                    f"Qwen3.8-Flash-Next monokernel: layer {i}: " + "; ".join(why)
                )
            self.la.bind(i, conv, ssm)
        return conv, ssm

    def forward(
        self,
        plan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
        *_,
        **__,
    ):
        """``Qwen4ExpForConditionalGeneration.forward`` for a routed decode step."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.models.qwen4_exp import _commit_ple_batch, _prepare_ple_batch

        model, la = self.model, self.la
        lm, rows = model.model, plan.key.rows
        lab = get_attn_backend().linear_attn_backend
        md = lab.forward_metadata
        if plan.key not in self.first_seen:  # shapes only: no host sync
            self.first_seen.add(plan.key)
            logger.info(
                "Qwen3.8-Flash-Next monokernel: first %s step of %d rows",
                plan.key.forward_mode,
                rows,
            )
        check = None
        if self.check and not torch.cuda.is_current_stream_capturing():
            from sgl_amd_mono.qwen38_flash_next.decode import check as check_mod

            check = check_mod.StepCheck(
                positions, forward_batch, rows, self.check_ref, la
            )
        pos = forward_batch.mrope_positions if model.is_mrope_enabled else positions
        la.begin_step()
        lm.last_hc_hidden_states = None
        state = lm.embed_tokens(input_ids).repeat(1, L.HC)
        ple_batch = None
        if lm.has_ple:
            ple_batch = _prepare_ple_batch(
                input_ids=input_ids,
                forward_batch=forward_batch,
                ngram_size=lm.ple_ngram_size,
                ngram_eos_token_id=lm.ple_ngram_eos_token_id,
            )
        slots = la.st_idx[rows]
        slots.copy_(md.mamba_cache_indices[:rows])
        x, qkv, ixqk = la.x[:rows], la.qkv[:rows], la.ixqk[:rows]
        modules, seq, stats = lm.layers, 0, True
        for i, layer in enumerate(modules):
            if i + 1 < len(modules) and modules[i + 1].ple is not None:
                modules[i + 1].ple.start_prefetch(
                    batch=ple_batch, forward_batch=forward_batch
                )
            if layer.ple is not None:
                state = state + layer.ple(state, forward_batch, ple_batch)
                stats = True
            keys = la.keys(i, rows, stats)
            dyn = {"state": state, "st_idx": slots}
            gdn = self._gdn_state(i, lab) if la.layers[i].kind == "gdn" else None
            if check is not None:
                check.before(layer, la.layers[i], pos, state, slots, gdn)
            seq = la.launch(keys[0], i, seq, dyn)
            if gdn is not None:
                # SGLang's own call (its GDN backend's decode), with its slots
                lab._track_mamba_state_decode(
                    forward_batch, gdn[0], gdn[1], md.mamba_cache_indices, i
                )
            else:
                if check is not None:
                    check.front(i, layer, state, x, qkv, ixqk)
                attn, gate = qsa_core(layer, x, qkv, ixqk, pos, forward_batch)
                seq = la.launch(keys[1], i, seq, dyn | {"attn": attn, "gate": gate})
            if check is not None:
                check.after(la.layers[i], state)
            stats = False
        if ple_batch is not None:
            _commit_ple_batch(batch=ple_batch, forward_batch=forward_batch)
        if check is not None:
            check.track(forward_batch, md)
            check.finish()
        hidden = lm.hyper_connection_mixer.mix(state)[0]
        lm.last_hc_hidden_states = state
        if get_embedding:
            return model.pooler(hidden, forward_batch)
        out = model.logits_processor(
            input_ids, hidden, model.lm_head, forward_batch, None
        )
        if isinstance(out, LogitsProcessorOutput):
            out.hidden_states = state
        return out


def build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen38_flash_next.decode import launcher, weights

    p = get_parallel()
    tp, rank, group = p.tp_size, p.tp_rank, p.tp_group.cpu_group
    widths = decode_widths(ctx.widths)
    layers = _on_every_rank(
        lambda: weights.build_layers(model, ctx.copies),
        group,
        "building the weight copies",
    )
    experts = model.model.layers[0].mlp.experts
    la = launcher.Launcher(
        layers,
        widths,
        tp,
        rank,
        group,
        experts.w13_weight.device,
        moe_rip(experts.w13_weight, experts.w2_weight),
        ctx.deadline_s,
    )
    # the launch state is zero (the peer buffer from its allocation) before any
    # rank's first launch: the compile agreement below orders every rank's
    # launches after every rank's synchronize
    torch.cuda.synchronize()
    keys = la.all_keys()
    # FlyDSL's disk cache hands a rank the kernels another rank built (a lock per
    # kernel), so each rank starts at its own share
    start = len(keys) * rank // tp
    _on_every_rank(
        lambda: launcher.compile_all(keys[start:] + keys[:start]),
        group,
        "compiling the kernels",
    )
    built = _Built(model, la)
    logger.info(
        "Qwen3.8-Flash-Next monokernel: %d builds for widths %s, %d launches a "
        "step, %s TP all-reduce lines, wait deadline %.0f s, experts padded to %d; "
        "check mode %s",
        len(keys),
        list(widths),
        la.launches,
        la.codec,
        ctx.deadline_s,
        la.rip,
        built.check,
    )
    return built
