# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 GEMV for one gfx950 rank.

Weights are float8_e4m3fn with a fp32 scale per 128x128 block. Dynamic
activations use the same 1x128 group scale (amax / 448). On gfx950, SGLang
stores the weight in aiter's (16, 16) preshuffle; each expert of a 3D MoE
weight is shuffled on its own last two axes. The preshuffled path calls the
same block-scale GEMM those weights were shuffled for. A plain (row-major)
weight is dequantized per block and multiplied in fp32.
"""

from __future__ import annotations

import os

import torch

from sgl_amd_mono.qwen35_a17b_fp8.gfx950.layout import BLOCK, FP8_MAX

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("aiter_w8a8_block_fp8_linear",),
}


def _shuffled(weight: torch.Tensor) -> bool:
    if bool(getattr(weight, "is_shuffled", False)):
        return True
    return bool(getattr(weight, "aiter_bpreshuffled", False))


def block_fp8_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``x @ W.T`` for a block-FP8 weight ``W`` of shape ``[N, K]``.

    ``x`` is bf16 ``[M, K]``. ``scale`` is fp32 ``[N/128, K/128]``.
    """
    if x.numel() == 0:
        return x.new_empty(x.shape[0], weight.shape[0])
    if _shuffled(weight):
        return _aiter_gemv(x, weight, scale)
    return _plain_gemv(x, weight, scale)


def _allow_block_fp8_linear() -> None:
    """The gfx950 small-M helper imports symbols its MoE module no longer defines.

    That import runs before the helper can decline the shape, so every block-FP8
    linear dies with it. Publishing the names lets the import finish; disabling
    the helper then takes the same block-scale GEMM the larger shapes use.
    """
    import importlib

    moe = importlib.import_module("sglang.kernels.ops.moe.smallm_moe_gfx950")
    os.environ.setdefault("SGLANG_ROCM_SMALLM_FP8_BS", "0")
    if hasattr(moe, "_check"):
        return

    class _Kernel:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("small-M block-FP8 helper is unavailable")

    moe._Kernel = _Kernel
    moe._check = lambda *_args, **_kwargs: None
    moe._hip_lib = lambda: None
    moe._hipcc = lambda: "hipcc"


def _aiter_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    _allow_block_fp8_linear()
    return aiter_w8a8_block_fp8_linear(x, weight, [BLOCK, BLOCK], scale)


def _quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """Dynamic 1x128 e4m3, the same group the block-scale GEMM quantizes."""
    m, k = x.shape
    groups = x.float().reshape(m, k // BLOCK, BLOCK)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    q = (groups / scale).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn).float() * scale
    return q.reshape(m, k)


def _plain_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Row-major block-FP8. gfx950 serves the preshuffled weight through aiter."""
    w = weight.view(torch.float8_e4m3fn).float()
    ws = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (_quantize_rows(x) @ (w * ws).transpose(0, 1)).to(dtype=x.dtype)
