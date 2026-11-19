# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash's DSpark draft on gfx950: its spec and forward (RFC #43949).

A routed draft step (a block of 7 rows a request, 1 or 2 requests, run as a
TARGET_VERIFY batch) runs ``_Draft.forward``, which takes and returns what
``DSparkMiniMaxDraftModel.forward`` does: the final-normed hidden states of the block,
the block's K / V rows written to the draft's caches. The 5 layers and the final norm
are one launch (``drafter/``, after vLLM's MiniMax-M3.1 drafter monokernel): MXFP8
GEMVs on the checkpoint's MXFP8 shards kept at load, the window attention over SGLang's
bf16 caches in e4m3 tiles, the TP sums in-kernel. The draft's numerics move only its
acceptance. ``SGLANG_AMD_MONO_M31_DRAFT=0`` runs the model's own forward.
"""

from __future__ import annotations

import logging
import os
import re

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.minimax_m31 import spec as target
from sgl_amd_mono.minimax_m31.drafter import layout
from sgl_amd_mono.minimax_m31.drafter import runner as R

logger = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_M31_DRAFT"
# the verify's context K / V rows through SGLang's fused write (one GEMM, one launch)
KV_ENV = "SGLANG_AMD_MONO_M31_DRAFT_KV"
TP = 4
_CKPT = re.compile(
    r"(?:language_model\.)?model\.dspark\.layers\.(?P<i>\d+)\.decoder_layer\."
    r"(?:self_attn|mlp)\.(?P<p>[a-z]+_proj)\.(?P<kind>weight|weight_scale_inv)"
)

# by file under sglang/ (``f(p)``: a parameter, ``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Requirements",
    ),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "next_token_logits",
        "hidden_states",
    ),
    # DSparkMiniMaxDraftModel.forward, which the routed forward stands in for
    "srt/models/dspark_minimax.py": (
        "DSparkMiniMaxDraftModel",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "decoder_layer",
        "block_size",
        "_build_fused_kv_write_bundle",
    ),
    "srt/models/dspark.py": (
        "forward_embed",
        "_fused_kv_write_layout",
        "_fused_kv_write_layout(project)",
        "_fused_kv_write_cache",
    ),
    "srt/models/minimax_m3.py": (
        "rotary_emb",
        "q_norm",
        "k_norm",
        "input_layernorm",
        "post_attention_layernorm",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/runtime_context.py": ("get_parallel", "get_schedule", '"tp_group"'),
    "srt/speculative/dflash_utils.py": ("get_dflash_attention_sliding_window_size",),
}


def _tokens_per_request(model) -> int:
    """A draft block's rows a request: the block size (the anchor and the drafts)."""
    return int(model.block_size)


def _load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each draft layer's MXFP8 projections (weight and
    e8m0 scales, in the checkpoint's [N, K] order), on the host until the build."""
    from sglang.srt.runtime_context import get_parallel

    m = _CKPT.fullmatch(name)
    if m is None or m["p"] not in R.SHARD_DIM:
        return
    p = get_parallel()
    part = R.shard(tensor, m["p"], m["kind"], p.tp_rank, p.tp_size)
    copies["m31 draft", int(m["i"]), m["p"], m["kind"]] = part.cpu().contiguous()


def _layer_weights(copies: dict) -> list[dict]:
    return [
        {
            (p, kind): copies["m31 draft", i, p, kind]
            for p in R.SHARD_DIM
            for kind in ("weight", "weight_scale_inv")
        }
        for i in range(layout.LAYERS)
    ]


def _cfg(config, name):
    v = getattr(config, name, None)
    return getattr(getattr(config, "text_config", None), name, None) if v is None else v


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel, get_schedule
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
    )

    reasons = []
    if get_parallel().tp_size != TP:
        reasons.append(f"draft TP {get_parallel().tp_size}: the launch takes TP {TP}")
    if int(getattr(model, "block_size", 0)) != layout.QPR:
        reasons.append(
            f"draft block {getattr(model, 'block_size', None)}: the launch takes {layout.QPR}"
        )
    layers = getattr(model, "layers", [])
    if len(layers) != layout.LAYERS:
        reasons.append(f"{len(layers)} draft layers: the launch takes {layout.LAYERS}")
    page = int(get_schedule().page_size or 0)
    if page != layout.BLOCK:
        reasons.append(
            f"page size {page}: a partition block is one {layout.BLOCK}-slot page"
        )
    config = getattr(model, "config", hf_config)
    window = get_dflash_attention_sliding_window_size(config)
    if window != layout.WINDOW - 1:
        reasons.append(
            f"draft window left {window}: the launch takes {layout.WINDOW - 1}"
        )
    if not _cfg(config, "use_gemma_norm"):
        reasons.append("the draft's norms are not Gemma RMSNorms")
    if (_cfg(config, "swiglu_limit"), _cfg(config, "swiglu_alpha")) != (7.0, 1.702):
        reasons.append("the draft's SwiGLU-OAI is not limit 7, alpha 1.702")
    for i, L in enumerate(layers):
        a = L.self_attn
        if (a.num_heads, a.num_kv_heads, a.head_dim) != (layout.HEADS, 1, layout.HD):
            reasons.append(
                f"draft layer {i}: heads {a.num_heads} / {a.num_kv_heads} of {a.head_dim} a rank"
            )
        if (
            not getattr(a.rotary_emb, "is_neox_style", False)
            or a.rotary_emb.cos_sin_cache.shape[-1] != layout.ROT
        ):
            reasons.append(f"draft layer {i}: needs NeoX RoPE over {layout.ROT} dims")
    missing = [
        (i, p, kind)
        for i in range(layout.LAYERS)
        for p in R.SHARD_DIM
        for kind in ("weight", "weight_scale_inv")
        if ("m31 draft", i, p, kind) not in copies
    ]
    if missing:
        reasons.append(
            f"the draft's MXFP8 checkpoint shards were not all seen at load ({missing[0]} ..)"
        )
    if (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        != layout.BLOCKS
    ):
        reasons.append(f"the launch needs {layout.BLOCKS} CUs")
    return reasons


class _Draft:
    """One rank's draft: the launch for a routed step; without a runner (``ENV=0``),
    the model's own forward."""

    def __init__(self, model, original, runner=None):
        self.model, self.original, self.runner = model, original, runner

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        *args,
        **kwargs,
    ):
        """``DSparkMiniMaxDraftModel.forward`` for a routed draft step."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_context import get_attn_backend

        if self.runner is None:
            return self.original(
                input_ids, positions, forward_batch, input_embeds, *args, **kwargs
            )
        x = (
            self.model.forward_embed(input_ids)
            if input_embeds is None
            else input_embeds
        )
        h = self.runner.forward(
            x.contiguous(), positions, forward_batch, get_attn_backend()
        )
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=h)


def _kv_project(runner):
    """``project`` of SGLang's fused context-KV write: the context hidden -> every layer's
    [K | V] columns, layer-major, against the draft's MXFP8 K / V rows dequantized once
    (bf16 holds e4m3 x 2^e exactly)."""
    w, s = runner.packed["qkv"]
    rows = slice(layout.QS, layout.QKV_N)
    q = w[:, rows].reshape(-1, layout.HIDDEN).view(torch.float8_e4m3fn).float()
    e = s[:, rows].reshape(-1, layout.HIDDEN // 32).float() - 127
    weight = (q.view(q.shape[0], -1, 32) * torch.exp2(e)[..., None]).view(q.shape)
    weight = weight.to(torch.bfloat16).contiguous()
    return lambda x: torch.nn.functional.linear(x.to(weight.dtype), weight)


def _build(model, hf_config, ctx) -> _Draft:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.minimax_m31 import draft_sampler

    original = type(model).forward.__get__(model, type(model))
    draft_sampler.install(model, ctx.deadline_s)
    if os.environ.get(ENV, "1") == "0":
        logger.info(
            "monokernel: MiniMax-M3.1 DSpark draft attached, %s=0: the model's own forward",
            ENV,
        )
        return _Draft(model, original)
    p = get_parallel()
    weights = _layer_weights(ctx.copies)
    runner = R.DraftRunner(
        model,
        weights,
        p.tp_rank,
        p.tp_group.cpu_group,
        model.norm.weight.device,
        ctx.deadline_s,
    )
    for key in [k for k in ctx.copies if k[0] == "m31 draft"]:
        del ctx.copies[key]
    if os.environ.get(KV_ENV, "1") == "1":
        # SGLang's fused write takes any projection; the MiniMax draft offers none on ROCm
        project = _kv_project(runner)
        model._build_fused_kv_write_bundle = lambda pool: model._fused_kv_write_layout(
            pool, project=project
        )
        model._fused_kv_write_cache = None
    logger.info(
        "monokernel: MiniMax-M3.1 DSpark draft attached (5 layers and the final norm a launch)"
    )
    return _Draft(model, original, runner)


SPEC = MonoSpec(
    name="minimax_m31_dspark_draft_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(r * layout.QPR for r in range(1, layout.MAX_REQS + 1)),
            tp_sizes=(TP,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; the draft's attention backend is its own
    requires=Requirements(attention_backends=frozenset({"triton"})),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    load_tensor=_load_tensor,
    tokens_per_request=_tokens_per_request,
)
