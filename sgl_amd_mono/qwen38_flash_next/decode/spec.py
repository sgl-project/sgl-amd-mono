# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (SGLang's Qwen4-Exp) on gfx950 at TP2: the decode path of
``qwen38_flash_next.spec.SPEC`` (RFC sgl-project/sglang#43949), which routes TP2
servers here. Plain decode (no speculative decoding) of the MXFP4 checkpoint with
BF16 attention and GDN projections.

A routed decode step (1 to 16 rows) runs ``_Built.forward`` in place of
``Qwen4ExpForConditionalGeneration.forward``, in the order SGLang's forward runs:

    embedding     model.model.embed_tokens (decode rows are text tokens)
    streams       the embedding widened to the 4 HC streams
    PLE           _prepare_ple_batch, the next PLE layer's start_prefetch one layer
                  ahead, state + layer.ple(...) before that layer's attention mix,
                  _commit_ple_batch after the last layer
    per layer     the HC mixes, the GDN core, out_proj / o_proj and the MoE as
                  launches (``gfx950/runner.py``); a QSA layer's attention up to
                  o_proj's input is SGLang's, at the M-RoPE positions
    radix track   linear_attn_backend._track_mamba_state_decode after each GDN
                  launch, as SGLang's GDN backend calls it
    final mix     model.model.hyper_connection_mixer.mix
    logits        model.logits_processor; output.hidden_states = the streams

The GDN states are SGLang's, updated in place at the step's mamba cache indices; a
pad row (index <= 0) writes none.
"""

from __future__ import annotations

import inspect
import logging

import torch

from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import layout as KL
from sgl_amd_mono.qwen38_flash_next.decode.gfx950.weights import is_gdn, moe_rip

logger = logging.getLogger(__name__)

TP_SIZES = KL.TPS
# every TP size's widths; ``_Built`` runs the TP size's own
WIDTHS = tuple(
    range(
        min(lo for lo, _ in KL.WIDTH_RANGE.values()),
        max(hi for _, hi in KL.WIDTH_RANGE.values()) + 1,
    )
)
# The server arguments this path accepts on top of ``SPEC.requires``, checked as
# SGLang checks those
SERVER_ARGS = {
    # a request's verify rows would update one GDN state in sequence
    "speculative_algorithm": frozenset({None}),
    # the kernels keep the GDN state in f32
    "mamba_ssm_dtype": frozenset({None, "float32"}),
}
# the text config's fields the kernels are built for, and their values
HF_CONFIG = {
    **dict(zip(KL.SHAPE_NAMES, KL.SHAPES)),
    "num_attention_heads": KL.QSA_HEADS,
    "head_dim": KL.QSA_HD,
}
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
        "_compute_qsa_topk_indices",
        "is_mrope_enabled",
    ),
    "srt/layers/hyperconnection.py": (
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
        "hc_norm",
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
        "_track_mamba_state_decode",
        "_track_mamba_state_decode(forward_batch)",
        "_track_mamba_state_decode(conv_states)",
        "_track_mamba_state_decode(ssm_states)",
        "_track_mamba_state_decode(cache_indices)",
        "_track_mamba_state_decode(layer_id)",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput", "hidden_states"),
    "srt/mem_cache/memory_pool.py": ("mamba2_layer_cache", "conv", "temporal"),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions",),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_req_to_token_pool",
    ),
}


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
    if not isinstance(t, torch.Tensor):
        reasons.append(f"{where}: missing")
    elif tuple(t.shape) != tuple(shape) or t.dtype != dtype:
        reasons.append(f"{where}: {_brief(t)}, expected {tuple(shape)} {dtype}")


def _gdn_reasons(reasons: list[str], where: str, layer, lay: KL.Layout) -> None:
    attn = layer.linear_attn
    for name, shape in (
        ("in_proj_qkvz", (lay.QKVZ, KL.HIDDEN)),
        ("in_proj_ba", (lay.BA, KL.HIDDEN)),
        ("out_proj", (KL.HIDDEN, lay.CORE)),
    ):
        _expect(reasons, f"{where} {name}", _get(attn, f"{name}.weight"), shape)
    _expect(reasons, f"{where} A_log", _get(attn, "A_log"), (lay.NV,), torch.float32)
    dt_bias = _get(attn, "dt_bias")  # copied to bf16 if held otherwise
    if not (
        isinstance(dt_bias, torch.Tensor)
        and dt_bias.is_floating_point()
        and tuple(dt_bias.shape) == (lay.NV,)
    ):
        reasons.append(
            f"{where} dt_bias: {_brief(dt_bias)}, expected ({lay.NV},) float"
        )
    _expect(reasons, f"{where} norm", _get(attn, "norm.weight"), (KL.HD,))
    conv = _get(attn, "conv1d.weight")
    if not (
        isinstance(conv, torch.Tensor)
        and conv.dtype == torch.bfloat16
        and conv.numel() == lay.CONV * KL.CONV_W
        and conv.is_contiguous()
    ):
        reasons.append(
            f"{where} conv1d: {_brief(conv)}, expected {lay.CONV} x {KL.CONV_W} bf16"
        )
    sizes = _get(attn, "in_proj_qkvz.output_sizes")
    split = [lay.KEY, lay.KEY, lay.VAL, lay.VAL]
    if sizes is not None and [x // lay.tp for x in sizes] != split:
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


def _scale_cols(k: int) -> int:
    return (k // 32 + 7) // 8 * 8  # mx_layout.scale_cols, without importing FlyDSL


def _moe_reasons(reasons: list[str], where: str, layer, lay: KL.Layout):
    """The experts' padded intermediate, or None (a reason recorded)."""
    mlp = layer.mlp
    _expect(reasons, f"{where} router", _get(mlp, "gate.weight"), (KL.E, KL.HIDDEN))
    _expect(
        reasons,
        f"{where} shared_expert_gate",
        _get(mlp, "shared_expert_gate.weight"),
        (1, KL.HIDDEN),
    )
    if _get(mlp, "experts.num_fused_shared_experts") != 1:
        reasons.append(f"{where}: the kernels read the shared expert as expert {KL.E}")
    w13, w2 = _get(mlp, "experts.w13_weight"), _get(mlp, "experts.w2_weight")
    if not (isinstance(w13, torch.Tensor) and isinstance(w2, torch.Tensor)):
        reasons.append(f"{where} experts: missing w13_weight / w2_weight")
        return None
    try:
        rip = moe_rip(w13, w2, lay.RI)
    except ValueError as e:
        reasons.append(f"{where} experts: {e}")
        return None
    ne = KL.E + 1
    if w13.shape[0] != ne:
        reasons.append(f"{where} experts: {w13.shape[0]} slots, expected {ne}")
        return None
    # AITER's shuffled layout: the scales one e8m0 a 32-block, K blocks padded to 8
    for name, rows, k in (("w13", 2 * rip, KL.HIDDEN), ("w2", KL.HIDDEN, rip)):
        scale = _get(mlp, f"experts.{name}_weight_scale")
        want = ne * rows * _scale_cols(k)
        if (
            not isinstance(scale, torch.Tensor)
            or scale.element_size() != 1
            or scale.numel() != want
        ):
            reasons.append(
                f"{where} {name} scales: {_brief(scale)}, expected {want} e8m0"
            )
    for name, w in (("w13", w13), ("w2", w2)):
        if getattr(w, "is_shuffled", None) is False:
            reasons.append(f"{where} {name}: not in AITER's shuffled layout")
    if rip < lay.RIP:
        reasons.append(
            f"{where} experts: intermediate padded to {rip}, the launch reads {lay.RIP}"
        )
    return rip


def layout_reasons(model, tp: int) -> list[str]:
    """What the kernels refuse in the loaded model at TP ``tp``."""
    lay = KL.Layout(tp)
    reasons = []
    layers = list(model.model.layers)
    kinds = ["gdn" if is_gdn(x) else "qsa" for x in layers]
    if (kinds.count("gdn"), kinds.count("qsa")) != (GDN_LAYERS, QSA_LAYERS):
        reasons.append(
            f"{kinds.count('gdn')} GDN and {kinds.count('qsa')} QSA layers, expected "
            f"{GDN_LAYERS} and {QSA_LAYERS}"
        )
    rips = set()
    for i, layer in enumerate(layers):
        where = f"layer {i}"
        if is_gdn(layer):
            _gdn_reasons(reasons, where, layer, lay)
        else:
            o_proj = _get(layer, "o_proj.weight")
            _expect(reasons, f"{where} o_proj", o_proj, (KL.HIDDEN, lay.CORE))
            if getattr(layer, "attn_output_gate", True) is not True:
                reasons.append(f"{where}: the o_proj launch applies the output gate")
        for branch in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, branch, None)
            for name, path, shape in (
                ("down", "input_mix_weight_down.weight", (KL.LORA, KL.HH)),
                ("up", "input_mix_weight_up.weight", (KL.HH, KL.LORA)),
                ("inject", "block_inject_weight.weight", (KL.HC, KL.HH)),
            ):
                _expect(reasons, f"{where} {branch} {name}", _get(hc, path), shape)
            norm = _get(hc, "hc_norm.weight")
            if not isinstance(norm, torch.Tensor) or tuple(norm.shape) != (KL.HH,):
                reasons.append(
                    f"{where} {branch} norm: {_brief(norm)}, expected ({KL.HH},)"
                )
        rip = _moe_reasons(reasons, where, layer, lay)
        if rip is not None:
            rips.add(rip)
    if len(rips) > 1:
        reasons.append(f"the layers' experts are padded differently: {sorted(rips)}")
    return reasons


# What ``_Built.forward`` passes to SGLang's decode track-save, after ``self``
TRACK_SAVE_ARGS = (
    "forward_batch",
    "conv_states",
    "ssm_states",
    "cache_indices",
    "layer_id",
)


def track_save_reasons() -> list[str]:
    """The forward calls the linear-attention backend's decode track-save after
    each GDN launch (a copy with the radix cache on, a no-op with it off): refused
    when the running SGLang does not take that call. Checked after load: the
    backend module imports the model runner, which resolving server args must
    not."""
    where = (
        "sglang.srt.layers.attention.hybrid_linear_attn_backend.MambaAttnBackendBase"
    )
    try:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )
    except ImportError as e:
        return [f"radix track-save: cannot import {where}: {e}"]
    fn = getattr(MambaAttnBackendBase, "_track_mamba_state_decode", None)
    params = tuple(inspect.signature(fn).parameters)[1:] if callable(fn) else None
    if params is None or params[: len(TRACK_SAVE_ARGS)] != TRACK_SAVE_ARGS:
        return [
            f"radix track-save: {where}._track_mamba_state_decode{params} does not "
            f"take {TRACK_SAVE_ARGS}"
        ]
    return []


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    return layout_reasons(model, get_parallel().tp_size) + track_save_reasons()


def gdn_state_reasons(conv: torch.Tensor, ssm: torch.Tensor, tp: int) -> list[str]:
    """The GDN caches the kernels update in place: conv [slots, CONV, CONV_W - 1]
    bf16 (window last, any strides) and the recurrent state [slots, NV, HD, HD]
    f32, a slot dense."""
    lay = KL.Layout(tp)
    reasons = []
    if conv.dtype != torch.bfloat16 or tuple(conv.shape[-2:]) != (
        lay.CONV,
        KL.CONV_W - 1,
    ):
        reasons.append(
            f"conv state {tuple(conv.shape)} {conv.dtype}, expected "
            f"[slots, {lay.CONV}, {KL.CONV_W - 1}] bfloat16"
        )
    if ssm.dtype != torch.float32 or tuple(ssm.shape[-3:]) != (lay.NV, KL.HD, KL.HD):
        reasons.append(
            f"recurrent state {tuple(ssm.shape)} {ssm.dtype}, expected "
            f"[slots, {lay.NV}, {KL.HD}, {KL.HD}] float32"
        )
    elif not ssm[0].is_contiguous():
        reasons.append(f"recurrent state slot not dense: strides {ssm.stride()}")
    return reasons


def gdn_state_view(cache) -> tuple[torch.Tensor, torch.Tensor]:
    """A GDN layer's (conv state as [slots, CONV, 3], recurrent state) from SGLang's
    mamba pool entry, which keeps the conv window first."""
    conv, ssm = cache.conv[0], cache.temporal
    if conv.dim() == 3 and conv.shape[1] == KL.CONV_W - 1:
        conv = conv.transpose(1, 2)
    return conv, ssm


def qsa_core(layer, x: torch.Tensor, positions: torch.Tensor, forward_batch):
    """A QSA layer's ``self_attention`` up to o_proj's input -> (attention output
    [S, CORE], output gate [S, CORE]); the o_proj launch applies the gate."""
    q, k, v, gate = layer._prepare_qkv_gate(
        positions=positions, hidden_states=x, forward_batch=forward_batch
    )
    kwargs = {}
    if layer.is_qsa:
        kwargs["topk_indices"] = layer._compute_qsa_topk_indices(
            x, positions, forward_batch
        )
    out = layer.attn(q, k, v, forward_batch, **kwargs)
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
    """One rank's kernels: the layers' operands, the launcher and the forward."""

    def __init__(self, model, layers, launcher, widths, tp: int):
        self.model, self.layers, self.la, self.tp = model, layers, launcher, tp
        device = launcher.epoch.device
        # the step's GDN slots as int32, per width (a graph records the copy)
        self.st_idx = {
            w: torch.zeros(w, dtype=torch.int32, device=device) for w in widths
        }
        self._states: dict[int, tuple] = {}
        self.first_seen: set = set()

    def _gdn_state(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer i's GDN states in SGLang's pool, checked once a binding."""
        from sglang.srt.model_executor.forward_context import get_req_to_token_pool

        conv, ssm = gdn_state_view(get_req_to_token_pool().mamba2_layer_cache(i))
        bound = self._states.get(i)
        if (
            bound is None
            or bound[0].data_ptr() != conv.data_ptr()
            or bound[1].data_ptr() != ssm.data_ptr()
        ):
            why = gdn_state_reasons(conv, ssm, self.tp)
            if why:
                raise RuntimeError(
                    f"Qwen3.8-Flash-Next monokernel: layer {i}: " + "; ".join(why)
                )
            self._states[i] = (conv, ssm)
        return self._states[i]

    def forward(
        self,
        plan,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        """``Qwen4ExpForConditionalGeneration.forward`` for a routed decode step."""
        model = self.model
        rows = plan.key.rows
        if rows not in self.st_idx:
            # a width the spec declares for the TP1 path (20, 24, 28 or 32 rows) that
            # these launches do not run: the model's own forward (the class's: the
            # runtime switched the instance's)
            return type(model).forward(
                model,
                input_ids,
                positions,
                forward_batch,
                get_embedding,
                pp_proxy_tensors,
            )
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.models.qwen4_exp import _commit_ple_batch, _prepare_ple_batch

        lm, la = model.model, self.la
        lab = get_attn_backend().linear_attn_backend
        md = lab.forward_metadata
        if plan.key not in self.first_seen:  # shapes only: no host sync
            self.first_seen.add(plan.key)
            logger.info(
                "Qwen3.8-Flash-Next monokernel: first %s step of %d rows",
                plan.key.forward_mode,
                rows,
            )
        pos = forward_batch.mrope_positions if model.is_mrope_enabled else positions
        la.begin_step()
        lm.last_hc_hidden_states = None
        state = lm.embed_tokens(input_ids).repeat(1, KL.HC)
        ple_batch = None
        if lm.has_ple:
            ple_batch = _prepare_ple_batch(
                input_ids=input_ids,
                forward_batch=forward_batch,
                ngram_size=lm.ple_ngram_size,
                ngram_eos_token_id=lm.ple_ngram_eos_token_id,
            )
        slots = self.st_idx[rows]
        slots.copy_(md.mamba_cache_indices[:rows])
        msq: torch.Tensor | None = None
        modules = lm.layers
        for w in self.layers:
            layer = modules[w.idx]
            if w.idx + 1 < len(modules) and modules[w.idx + 1].ple is not None:
                modules[w.idx + 1].ple.start_prefetch(
                    batch=ple_batch, forward_batch=forward_batch
                )
            if layer.ple is not None:
                state = state + layer.ple(state, forward_batch, ple_batch)
                msq = None
            gdn = None if w.gdn is None else self._gdn_state(w.idx)
            x, inj = la.hc_front(w.idx, w.attn_hc, state, msq, la.attn_scratch)
            if gdn is not None:
                core, gate = la.gdn(w.idx, w.gdn, x, gdn[0], gdn[1], slots), None
                # SGLang's own call (its GDN backend's decode), with its int64 slots
                lab._track_mamba_state_decode(
                    forward_batch, gdn[0], gdn[1], md.mamba_cache_indices, w.idx
                )
            else:
                core, gate = qsa_core(layer, x, pos, forward_batch)
            mid, msq = la.oproj(w, core, gate, state, inj)
            x, inj = la.hc_front(w.idx, w.mlp_hc, mid, msq, la.mlp_scratch)
            state, msq = la.moe(w.idx, w.moe, x, mid, inj)
        if ple_batch is not None:
            _commit_ple_batch(batch=ple_batch, forward_batch=forward_batch)
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

    from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import runner, weights

    p = get_parallel()
    tp, rank, group = p.tp_size, p.tp_rank, p.tp_group.cpu_group
    lo, hi = KL.WIDTH_RANGE[tp]
    widths = tuple(w for w in ctx.widths if lo <= w <= hi)
    if not widths:
        raise ValueError(
            f"Qwen3.8-Flash-Next monokernel: TP{tp} runs {lo} to {hi} rows on the "
            f"kernels, and this server's decode graphs route {sorted(ctx.widths)}"
        )
    experts = model.model.layers[0].mlp.experts
    device = experts.w13_weight.device
    layers = _on_every_rank(
        lambda: weights.build_layers(model, ctx.copies),
        group,
        "building the weight copies",
    )
    rip = moe_rip(experts.w13_weight, experts.w2_weight, KL.Layout(tp).RI)
    la = runner.Launcher(widths, rank, tp, group, device, rip, ctx.deadline_s)
    hc_eps = sorted({h.eps for w in layers for h in (w.attn_hc, w.mlp_hc)})
    gdn_eps = sorted({w.gdn.norm_eps for w in layers if w.gdn is not None})
    keys = runner.build_keys(widths, tp, rip, hc_eps, gdn_eps, la.fault)
    # FlyDSL's disk cache hands a rank the kernels another rank built (a lock per
    # kernel), so each rank starts at its own share
    start = len(keys) * rank // tp
    stream = torch.cuda.current_stream()

    def compile_all():
        for build, key in keys[start:] + keys[:start]:
            runner.compile_build(build, key, stream)

    _on_every_rank(compile_all, group, "compiling the kernels")
    built = _Built(model, layers, la, widths, tp)
    logger.info(
        "Qwen3.8-Flash-Next monokernel: %d launches compiled for widths %s; experts "
        "padded to %d, w2 read to %d; HC eps %s, GDN norm eps %s",
        len(keys),
        list(widths),
        rip,
        KL.Layout(tp).RIP,
        hc_eps,
        gdn_eps,
    )
    return built
