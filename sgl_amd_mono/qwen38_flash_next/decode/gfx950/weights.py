# SPDX-License-Identifier: Apache-2.0
"""Each decoder layer's launch operands, from the loaded SGLang modules.

SGLang's tensors are read in place, except these private copies (``copies``,
which SGLang logs at startup):

- ``mxfp4``: a GDN layer's in_proj_qkvz and out_proj, which the checkpoint keeps
  in bf16, quantized for the kernels' FP4 GEMVs (``mxfp4_copy``). SGLang's modules
  keep their bf16 weights, so every step off the kernels computes as without them.
- ``hc_down``: each hyper-connection's ``input_mix_weight_down`` and
  ``block_inject_weight`` as the one [336, 4 H] matrix the HC front reads.
- ``bf16``: an HC norm weight or a GDN ``dt_bias`` SGLang holds in another dtype.

``shuffle_16x16`` and ``swizzle_scales`` are AITER's ``shuffle_weight(layout=(16,
16))`` and the ASM FP4 GEMM's scale swizzle as plain views and permutes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import layout as KL

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "model",
        "layers",
        "linear_attn",
        "in_proj_qkvz",
        "in_proj_ba",
        "conv1d",
        "out_proj",
        "A_log",
        "dt_bias",
        "norm",
        "o_proj",
        "mlp",
    ),
    "srt/models/qwen4_exp.py": ("attn_hyper_connection", "mlp_hyper_connection"),
    "srt/layers/hyperconnection.py": (
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
        "hc_norm",
        "variance_epsilon",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": ("eps",),
    "srt/models/qwen2_moe.py": ("gate", "shared_expert_gate", "experts"),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    ),
}


class Fp4(NamedTuple):
    weight: torch.Tensor  # [N, K / 2] uint8, shuffled 16 x 16
    scale: torch.Tensor  # [N, K / 32] uint8 e8m0, swizzled


class Hc(NamedTuple):
    norm: torch.Tensor  # [4 H] bf16 (GroupedGemmaRMSNorm's weight, 1 + w applied)
    down: torch.Tensor  # [336, 4 H] bf16: down | inject | zero rows
    up: torch.Tensor  # [4 H, 320] bf16
    eps: float


class Gdn(NamedTuple):
    qkvz: Fp4
    ba: torch.Tensor  # [2 NV, H] bf16
    conv: torch.Tensor  # [CONV, 1, 4] bf16
    a_log: torch.Tensor  # [NV] f32
    dt_bias: torch.Tensor  # [NV] bf16
    norm: torch.Tensor  # [128] bf16
    norm_eps: float
    out: Fp4


class Moe(NamedTuple):
    gate: torch.Tensor  # [E, H] bf16
    sgate: torch.Tensor  # [1, H] bf16
    w13: torch.Tensor  # [E + 1, 2 RIP, H / 2] fp4x2, shuffled
    w13s: torch.Tensor
    w2: torch.Tensor  # [E + 1, H, RIP / 2]
    w2s: torch.Tensor


class Layer(NamedTuple):
    idx: int
    attn_hc: Hc
    mlp_hc: Hc
    gdn: Gdn | None  # None: a QSA layer
    o_proj: torch.Tensor | None  # a QSA layer's o_proj [H, CORE] bf16
    moe: Moe

    @property
    def kind(self) -> str:
        return "qsa" if self.gdn is None else "gdn"


def is_gdn(layer) -> bool:
    return getattr(layer, "linear_attn", None) is not None


def moe_rip(w13: torch.Tensor, w2: torch.Tensor, ri: int) -> int:
    """The padded intermediate RIP the MXFP4 expert weights hold, from their shapes.

    w13 [E', 2 RIP, H/2] holds gate rows [0, RI) and up rows [RIP, RIP + RI); w2 [E',
    H, RIP/2]; one byte per two FP4 values. SGLang pads the packed bytes to 128
    (``get_moe_weight_sizes``: RIP 512 at RI 320), the kernels' own layout the
    values (``Layout.RIP``: 384). The launch takes RIP as the weights' stride and
    reads w2's K only to ``Layout.RIP``.
    """
    if w13.element_size() != 1 or w2.element_size() != 1:
        raise ValueError(f"expert weights are not packed FP4: {w13.dtype}, {w2.dtype}")
    if w13.dim() != 3 or w2.dim() != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError(f"expert weight shapes {tuple(w13.shape)}, {tuple(w2.shape)}")
    if w13.shape[1] % 2:
        raise ValueError(f"w13 rows {w13.shape[1]} are not gate | up")
    rip = w13.shape[1] // 2
    if w2.shape[2] * 2 != rip:
        raise ValueError(f"w13 pads the intermediate to {rip}, w2 to {w2.shape[2] * 2}")
    if w13.shape[2] * 2 != KL.HIDDEN or w2.shape[1] != KL.HIDDEN:
        raise ValueError(f"expert weights are not {KL.HIDDEN} wide: {tuple(w13.shape)}")
    if rip < ri or rip % 32:
        raise ValueError(f"padded intermediate {rip} for {ri} (MXFP4 groups of 32)")
    return rip


def fuse_hc_down_inject(down: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
    """``input_mix_weight_down`` [320, 4 H] and ``block_inject_weight`` [4, 4 H],
    then zero rows to 336: the HC front's one down matrix. A new tensor."""
    width = KL.HH
    if down.shape != (KL.LORA, width) or inject.shape != (KL.HC, width):
        raise ValueError(
            f"HC down {tuple(down.shape)} / inject {tuple(inject.shape)}, "
            f"expected ({KL.LORA}, {width}) / ({KL.HC}, {width})"
        )
    if down.dtype != inject.dtype:
        raise ValueError(f"HC down {down.dtype} and inject {inject.dtype} differ")
    out = down.new_zeros(KL.DOWN_ROWS, width)
    out[: KL.LORA].copy_(down)
    out[KL.LORA : KL.LORA + KL.HC].copy_(inject)
    return out


def shuffle_16x16(q: torch.Tensor) -> torch.Tensor:
    """A packed FP4 weight [N, K / 2] with each 16-row group's 16-byte K chunks as
    [chunk][row % 16][16 B] (``mx_layout.fp4_tile_dword``). A new tensor."""
    n, kb = q.shape
    if q.element_size() != 1 or n % 16 or kb % 32:
        raise ValueError(f"packed FP4 weight {tuple(q.shape)} {q.dtype}")
    return (
        q.view(n // 16, 16, kb // 16, 16).permute(0, 2, 1, 3).contiguous().view(n, kb)
    )


def swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """e8m0 scales [N, K / 32] in the ASM FP4 GEMM's layout
    (``mx_layout.scale_index``); N a multiple of 32 and K / 32 of 8, as both GDN
    linears are. A new tensor."""
    sm, sn = s.shape
    if s.element_size() != 1 or sm % 32 or sn % 8:
        raise ValueError(f"e8m0 scales {tuple(s.shape)} {s.dtype}")
    v = s.view(sm // 32, 2, 16, sn // 8, 2, 4)
    return v.permute(0, 3, 5, 2, 4, 1).contiguous().view(sm, sn)


def unshuffle_16x16(q: torch.Tensor) -> torch.Tensor:
    n, kb = q.shape
    return (
        q.view(n // 16, kb // 16, 16, 16).permute(0, 2, 1, 3).contiguous().view(n, kb)
    )


def unswizzle_scales(s: torch.Tensor) -> torch.Tensor:
    sm, sn = s.shape
    v = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    return v.permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_EVEN = (True, False, True, False, True, False, True, False)  # even mantissa


def mxfp4_dequant(
    q: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16
) -> torch.Tensor:
    """An ``mxfp4_copy`` (shuffled fp4, swizzled e8m0) -> the values [N, K] it
    stands for (exact in bf16)."""
    q, s = unshuffle_16x16(q.view(torch.uint8)), unswizzle_scales(s.view(torch.uint8))
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).view(q.shape[0], -1).long()
    table = torch.tensor(E2M1, dtype=torch.float32, device=q.device)
    mag = table[codes & 7]
    val = torch.where(codes >= 8, -mag, mag)
    scale = torch.exp2(s.float() - 127.0)
    return (
        (val.view(q.shape[0], -1, 32) * scale[..., None]).view(q.shape[0], -1).to(dtype)
    )


def mxfp4_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """``x`` [..., K] through the kernels' activation MXFP4 (as AITER's
    ``per_1x32_f4_quant_hip``): 1 x 32 blocks along K, e8m0 code ceil(log2(amax /
    6)) clamped to 1..253, elements to the nearest e2m1 (ties to the even
    mantissa, saturating at 6) -> the values, in ``x``'s dtype."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], xf.shape[-1] // 32, 32)
    d = blocks.abs().amax(-1) / 6.0
    bits = d.view(torch.int32)
    code = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = torch.exp2(code.clamp(1, 253).float() - 127.0)[..., None]
    y = (blocks / scale).clamp(-6.0, 6.0)
    table = torch.tensor(E2M1, dtype=torch.float32, device=x.device)
    dist = (y.abs()[..., None] - table).abs()
    best = dist == dist.amin(-1, keepdim=True)
    even = best & torch.tensor(_E2M1_EVEN, device=x.device)
    idx = torch.where(even.any(-1), even.int().argmax(-1), best.int().argmax(-1))
    out = torch.copysign(table[idx], y) * scale
    return out.reshape(xf.shape).to(x.dtype)


Quantize = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def aiter_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AITER's dynamic MXFP4 quantization: [N, K] bf16 -> packed [N, K / 2] and
    e8m0 [N, K / 32]."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    return dynamic_mxfp4_quant(w.contiguous())


def mxfp4_copy(
    w: torch.Tensor, quantize: Quantize = aiter_mxfp4
) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 linear weight [N, K] as the kernels read an MXFP4 one with
    ``shuffled``: (fp4 [N, K / 2] uint8, e8m0 [N, K / 32] uint8), new tensors."""
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise ValueError(f"linear weight {tuple(w.shape)} {w.dtype}, expected 2-D bf16")
    q, s = quantize(w)
    n, k = w.shape
    # AITER may hand the scales back column-major
    q, s = q.view(torch.uint8).contiguous(), s.view(torch.uint8).contiguous()
    if q.shape != (n, k // 2) or s.shape != (n, k // 32):
        raise ValueError(
            f"MXFP4 quantization of {(n, k)} gave {tuple(q.shape)}, {tuple(s.shape)}"
        )
    return shuffle_16x16(q), swizzle_scales(s)


class _Reads:
    """The SGLang tensors read, to check that building the copies left them as
    they were (same storage, no in-place write)."""

    def __init__(self):
        self.seen: dict[int, tuple[str, torch.Tensor, int, int]] = {}

    def __call__(self, name: str, t: torch.Tensor) -> torch.Tensor:
        self.seen.setdefault(id(t), (name, t, t.data_ptr(), t._version))
        return t

    def changed(self) -> list[str]:
        return [
            name
            for name, t, ptr, version in self.seen.values()
            if t.data_ptr() != ptr or t._version != version
        ]


def _bf16(t: torch.Tensor, key: tuple, copies: dict) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t
    out = t.to(torch.bfloat16)
    copies[("bf16", *key)] = out
    return out


def _hc(hc, idx: int, branch: str, copies: dict, read: _Reads) -> Hc:
    where = f"layer {idx} {branch}"
    down = read(f"{where} down", hc.input_mix_weight_down.weight)
    inject = read(f"{where} inject", hc.block_inject_weight.weight)
    fused = fuse_hc_down_inject(down, inject)
    copies[("hc_down", idx, branch)] = fused
    norm = read(f"{where} norm", hc.hc_norm.weight)
    return Hc(
        norm=_bf16(norm, (idx, branch, "hc_norm"), copies),
        down=fused,
        up=read(f"{where} up", hc.input_mix_weight_up.weight),
        eps=float(hc.hc_norm.variance_epsilon),
    )


def _fp4(linear, idx: int, name: str, copies: dict, read: _Reads, quantize) -> Fp4:
    w = read(f"layer {idx} {name}", linear.weight)
    q, s = mxfp4_copy(w, quantize)
    copies[("mxfp4", idx, name)] = q
    copies[("mxfp4", idx, name + ".scale")] = s
    return Fp4(q, s)


def _gdn(attn, idx: int, copies: dict, read: _Reads, quantize) -> Gdn:
    where = f"layer {idx}"
    return Gdn(
        qkvz=_fp4(attn.in_proj_qkvz, idx, "in_proj_qkvz", copies, read, quantize),
        ba=read(f"{where} in_proj_ba", attn.in_proj_ba.weight),
        conv=read(f"{where} conv1d", attn.conv1d.weight),
        a_log=read(f"{where} A_log", attn.A_log),
        dt_bias=_bf16(read(f"{where} dt_bias", attn.dt_bias), (idx, "dt_bias"), copies),
        norm=read(f"{where} norm", attn.norm.weight),
        norm_eps=float(attn.norm.eps),
        out=_fp4(attn.out_proj, idx, "out_proj", copies, read, quantize),
    )


def _moe(mlp, idx: int, read: _Reads) -> Moe:
    ex = mlp.experts
    where = f"layer {idx}"
    return Moe(
        gate=read(f"{where} router", mlp.gate.weight),
        sgate=read(f"{where} shared_expert_gate", mlp.shared_expert_gate.weight),
        w13=read(f"{where} w13", ex.w13_weight),
        w13s=read(f"{where} w13 scale", ex.w13_weight_scale),
        w2=read(f"{where} w2", ex.w2_weight),
        w2s=read(f"{where} w2 scale", ex.w2_weight_scale),
    )


def build_layers(model, copies: dict, quantize: Quantize = aiter_mxfp4) -> list[Layer]:
    """Every decoder layer's operands, the copies added to ``copies``. Raises if an
    SGLang tensor read changed."""
    read = _Reads()
    out = []
    with torch.no_grad():
        for idx, layer in enumerate(model.model.layers):
            gdn = is_gdn(layer)
            out.append(
                Layer(
                    idx=idx,
                    attn_hc=_hc(layer.attn_hyper_connection, idx, "attn", copies, read),
                    mlp_hc=_hc(layer.mlp_hyper_connection, idx, "mlp", copies, read),
                    gdn=(
                        _gdn(layer.linear_attn, idx, copies, read, quantize)
                        if gdn
                        else None
                    ),
                    o_proj=(
                        None
                        if gdn
                        else read(f"layer {idx} o_proj", layer.o_proj.weight)
                    ),
                    moe=_moe(layer.mlp, idx, read),
                )
            )
    changed = read.changed()
    if changed:
        raise RuntimeError(
            f"building the copies changed SGLang's tensors: {changed[:5]}"
        )
    return out
