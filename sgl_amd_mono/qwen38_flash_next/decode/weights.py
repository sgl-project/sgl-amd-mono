# SPDX-License-Identifier: Apache-2.0
"""Each decoder layer's launch operands from the loaded SGLang modules: the
kernels' arguments every step of the layer shares (``Layer.args``, names of
``wide.layer.ABI``).

SGLang's tensors are read in place, except these private copies (``copies``,
which SGLang logs at startup):

- ``mxfp4``: a GDN layer's in_proj_qkvz and out_proj, which the checkpoint keeps
  in bf16, quantized to MXFP4 in AITER's ASM FP4 GEMM layout (``mxfp4_copy``);
  the kernels decode them to bf16 operands (W4A16). SGLang's modules keep their
  bf16 weights, so every step off the kernels computes as without them.
- ``hc_down``: each hyper-connection's ``input_mix_weight_down`` and
  ``block_inject_weight`` as the one [336, 4 H] matrix the mixes read.
- ``router``: the router with the shared expert's gate as its last row, [513, H],
  which the kernels route over (the shared expert is expert 512).
- ``bf16``: an HC norm weight or a GDN ``dt_bias`` SGLang holds in another dtype.

``shuffle_16x16`` and ``swizzle_scales`` are AITER's ``shuffle_weight(layout=(16,
16))`` and ``e8m0_shuffle`` as plain views and permutes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L

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
        "qkv_proj",
        "mlp",
    ),
    "srt/models/qwen4_exp.py": (
        "attn_hyper_connection",
        "mlp_hyper_connection",
        "indexer",
    ),
    "srt/layers/attention/qsa/qsa_indexer.py": ("index_qk_proj",),
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


class Layer(NamedTuple):
    idx: int
    kind: str  # "gdn" or "qsa"
    ple: bool  # SGLang adds the layer's PLE to its input streams
    args: dict  # ABI name -> tensor: the operands every step of the layer reads
    hc_eps: float  # the attention hyper-connection's norm epsilon
    mlp_hc_eps: float  # the MLP hyper-connection's
    gdn_eps: float | None  # the GDN gated norm's epsilon; None: a QSA layer


def is_gdn(layer) -> bool:
    return getattr(layer, "linear_attn", None) is not None


def moe_rip(w13: torch.Tensor, w2: torch.Tensor) -> int:
    """The padded intermediate RIP the MXFP4 expert weights hold, from their shapes.

    w13 [E', 2 RIP, H / 2] holds gate rows [0, RI) and up rows [RIP, RIP + RI); w2
    [E', H, RIP / 2]; one byte per two FP4 values. SGLang pads the packed bytes to
    128 (``get_moe_weight_sizes``: RIP 512 at RI 320). The kernels take RIP as the
    weights' stride, in 128-row steps, and read w2's K to RI rounded up to 128.
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
    if w13.shape[2] * 2 != L.HIDDEN or w2.shape[1] != L.HIDDEN:
        raise ValueError(f"expert weights are not {L.HIDDEN} wide: {tuple(w13.shape)}")
    if rip < L.RI or rip % 128:
        raise ValueError(f"intermediate padded to {rip}: {L.RI} in steps of 128")
    return rip


def scale_cols(k: int) -> int:
    """e8m0 columns of a K-wide MXFP4 row in AITER's shuffled layout."""
    return (k // 32 + 7) // 8 * 8


def fuse_hc_down_inject(down: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
    """``input_mix_weight_down`` [320, 4 H] and ``block_inject_weight`` [4, 4 H],
    then zero rows to 336: the mixes' one down matrix. A new tensor."""
    width = L.HC_HIDDEN
    if down.shape != (L.LOWRANK, width) or inject.shape != (L.HC, width):
        raise ValueError(
            f"HC down {tuple(down.shape)} / inject {tuple(inject.shape)}, "
            f"expected ({L.LOWRANK}, {width}) / ({L.HC}, {width})"
        )
    if down.dtype != inject.dtype:
        raise ValueError(f"HC down {down.dtype} and inject {inject.dtype} differ")
    out = down.new_zeros(L.DOWN_ROWS, width)
    out[: L.LOWRANK].copy_(down)
    out[L.INJECT : L.INJECT + L.HC].copy_(inject)
    return out


def router_gate(gate: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """The router [E, H] and the shared expert's gate [1, H] as the [E + 1, H]
    matrix the kernels route over. A new tensor."""
    if gate.shape != (L.EXPERTS, L.HIDDEN) or shared.shape != (1, L.HIDDEN):
        raise ValueError(
            f"router {tuple(gate.shape)} / shared expert gate {tuple(shared.shape)}"
        )
    return torch.cat([gate, shared], 0).contiguous()


def shuffle_16x16(q: torch.Tensor) -> torch.Tensor:
    """A packed FP4 weight [N, K / 2] with each 16-row group's 16-byte K pieces as
    [piece][row % 16][16 B] (``tiled.Fp4Weights`` shuffled). A new tensor."""
    n, kb = q.shape
    if q.element_size() != 1 or n % 16 or kb % 32:
        raise ValueError(f"packed FP4 weight {tuple(q.shape)} {q.dtype}")
    return (
        q.view(n // 16, 16, kb // 16, 16).permute(0, 2, 1, 3).contiguous().view(n, kb)
    )


def swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """e8m0 scales [N, K / 32] in ``e8m0_shuffle``'s layout (``tiled.Fp4Weights``
    shuffled); N a multiple of 32 and K / 32 of 8, as both GDN linears are. A new
    tensor."""
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


def mxfp4_dequant(
    q: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16
) -> torch.Tensor:
    """An ``mxfp4_copy`` (shuffled fp4, swizzled e8m0) -> the values [N, K] it
    stands for (exact in bf16): what the kernels' operands hold."""
    q, s = unshuffle_16x16(q.view(torch.uint8)), unswizzle_scales(s.view(torch.uint8))
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).view(q.shape[0], -1).long()
    table = torch.tensor(E2M1, dtype=torch.float32, device=q.device)
    mag = table[codes & 7]
    val = torch.where(codes >= 8, -mag, mag)
    scale = torch.exp2(s.float() - 127.0)
    return (
        (val.view(q.shape[0], -1, 32) * scale[..., None]).view(q.shape[0], -1).to(dtype)
    )


Quantize = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def aiter_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AITER's dynamic MXFP4 quantization: [N, K] bf16 -> packed [N, K / 2] and
    e8m0 [N, K / 32]."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    return dynamic_mxfp4_quant(w.contiguous())


def mxfp4_copy(
    w: torch.Tensor, quantize: Quantize = aiter_mxfp4
) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 linear weight [N, K] as the kernels read a shuffled MXFP4 one:
    (fp4 [N, K / 2] uint8, e8m0 [N, K / 32] uint8), new tensors."""
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


class _Builder:
    """One pass over the layers: the reads checked, the copies recorded."""

    def __init__(self, copies: dict, quantize: Quantize):
        self.copies, self.quantize, self.read = copies, quantize, _Reads()

    def bf16(self, t: torch.Tensor, key: tuple) -> torch.Tensor:
        if t.dtype == torch.bfloat16:
            return t
        out = t.to(torch.bfloat16)
        self.copies[("bf16", *key)] = out
        return out

    def hc(self, hc, idx: int, branch: str, prefix: str) -> tuple[dict, float]:
        where = f"layer {idx} {branch}"
        read = self.read
        fused = fuse_hc_down_inject(
            read(f"{where} down", hc.input_mix_weight_down.weight),
            read(f"{where} inject", hc.block_inject_weight.weight),
        )
        self.copies[("hc_down", idx, branch)] = fused
        norm = read(f"{where} norm", hc.hc_norm.weight)
        return {
            f"{prefix}_norm": self.bf16(norm, (idx, branch, "hc_norm")),
            f"{prefix}_di": fused,
            f"{prefix}_up": read(f"{where} up", hc.input_mix_weight_up.weight),
        }, float(hc.hc_norm.variance_epsilon)

    def fp4(self, linear, idx: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        q, s = mxfp4_copy(
            self.read(f"layer {idx} {name}", linear.weight), self.quantize
        )
        self.copies[("mxfp4", idx, name)] = q
        self.copies[("mxfp4", idx, name + ".scale")] = s
        return q, s

    def gdn(self, attn, idx: int) -> dict:
        where, read = f"layer {idx}", self.read
        w_qkvz, w_qkvz_s = self.fp4(attn.in_proj_qkvz, idx, "in_proj_qkvz")
        w_out, w_out_s = self.fp4(attn.out_proj, idx, "out_proj")
        return {
            "w_qkvz": w_qkvz,
            "w_qkvz_s": w_qkvz_s,
            "w_ba": read(f"{where} in_proj_ba", attn.in_proj_ba.weight),
            "w_conv": read(f"{where} conv1d", attn.conv1d.weight),
            "a_log": read(f"{where} A_log", attn.A_log),
            "dt_bias": self.bf16(
                read(f"{where} dt_bias", attn.dt_bias), (idx, "dt_bias")
            ),
            "gdn_norm": read(f"{where} norm", attn.norm.weight),
            "w_out": w_out,
            "w_out_s": w_out_s,
        }

    def qsa(self, layer, idx: int) -> dict:
        where, read = f"layer {idx}", self.read
        return {
            "w_out": read(f"{where} o_proj", layer.o_proj.weight),
            "w_qkv": read(f"{where} qkv_proj", layer.qkv_proj.weight),
            "w_ix": read(f"{where} index_qk_proj", layer.indexer.index_qk_proj.weight),
        }

    def moe(self, mlp, idx: int) -> dict:
        where, read, ex = f"layer {idx}", self.read, mlp.experts
        gate = router_gate(
            read(f"{where} router", mlp.gate.weight),
            read(f"{where} shared_expert_gate", mlp.shared_expert_gate.weight),
        )
        self.copies[("router", idx)] = gate
        return {
            "w_gate": gate,
            "w13": read(f"{where} w13", ex.w13_weight),
            "w13s": read(f"{where} w13 scale", ex.w13_weight_scale),
            "w2": read(f"{where} w2", ex.w2_weight),
            "w2s": read(f"{where} w2 scale", ex.w2_weight_scale),
        }

    def layer(self, layer, idx: int) -> Layer:
        attn_hc, hc_eps = self.hc(layer.attn_hyper_connection, idx, "attn", "a")
        mlp_hc, mlp_hc_eps = self.hc(layer.mlp_hyper_connection, idx, "mlp", "m")
        gdn = is_gdn(layer)
        mixer = self.gdn(layer.linear_attn, idx) if gdn else self.qsa(layer, idx)
        return Layer(
            idx=idx,
            kind="gdn" if gdn else "qsa",
            ple=getattr(layer, "ple", None) is not None,
            args={**attn_hc, **mlp_hc, **mixer, **self.moe(layer.mlp, idx)},
            hc_eps=hc_eps,
            mlp_hc_eps=mlp_hc_eps,
            gdn_eps=float(layer.linear_attn.norm.eps) if gdn else None,
        )


def build_layers(model, copies: dict, quantize: Quantize = aiter_mxfp4) -> list[Layer]:
    """Every decoder layer's operands, the copies added to ``copies``. Raises if an
    SGLang tensor read changed."""
    b = _Builder(copies, quantize)
    with torch.no_grad():
        out = [b.layer(layer, i) for i, layer in enumerate(model.model.layers)]
    changed = b.read.changed()
    if changed:
        raise RuntimeError(
            f"building the copies changed SGLang's tensors: {changed[:5]}"
        )
    return out
