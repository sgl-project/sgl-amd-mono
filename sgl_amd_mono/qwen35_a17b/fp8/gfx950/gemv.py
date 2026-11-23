# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 GEMV for one gfx950 rank.

Weights are float8_e4m3fn with a fp32 scale per 128x128 block. Dynamic
activations use the same 1x128 group scale (amax / 448). On gfx950, SGLang
stores the weight in aiter's (16, 16) preshuffle; each expert of a 3D MoE
weight is shuffled on its own last two axes. The preshuffled path calls the
same block-scale GEMM those weights were shuffled for. A plain (row-major)
weight is dequantized per block and multiplied in fp32.

At decode widths that GEMM walks all of K in one workgroup, so a long K is
bound by latency, not bandwidth. ``block_fp8_gemv_wg`` reads the same
preshuffled weight but splits K into slices across the warps of a workgroup,
one batched fp8 dot per 128-wide group scaled by its activation and weight
scales as the GEMM does, and adds the slices in fp32 at the end.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import BLOCK, FP8_MAX

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("aiter_w8a8_block_fp8_linear",),
}

# block_fp8_gemv_wg: one 16-row block of the preshuffle per workgroup, K in 8
# slices over 4 warps; tl.dot takes at least 16 rows. Verify steps (4 rows a
# request, up to 8 requests) take the 32-row tile.
WG_BN = 16
WG_SPLIT = 8
WG_ROWS = 16
WG_MAX_ROWS = 32
WG_WARPS = 4
# num_stages of the K loop in ``wg_dot``. No one value is right for every Triton
# build: 2 does not compile on Triton 3.8 (LLVM translation of the pipelined
# preshuffled-weight load), and 1 gives wrong results for M > 16 on Triton 3.7.
# ``select_wg_stages`` runs the kernel once per process and keeps the first
# candidate that compiles and matches a torch product.
WG_STAGE_CANDIDATES = (2, 1)
_wg_stages: int | None = None

# The miscompile depends on N (N=4096 K=4096 passed where N=5120 K=4096 failed),
# so the probe uses the TP4 qkvz and o_proj shapes plus a wider one. Rows 8 fills
# the 16-row tile, rows 17..32 the 32-row tile with masked rows.
_PROBE_SHAPES = ((5120, 4096), (4096, 2048), (12288, 4096))
_PROBE_ROWS = (8, 17, 24, 32)
_PROBE_REPEATS = 3  # a Triton 3.7 miscompile changes from run to run
_PROBE_TOL = 0.02  # of the largest reference value; bf16 output rounds to 0.4%


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


def block_fp8_gemv_q(
    x: torch.Tensor, xq, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``block_fp8_gemv`` from ``x``'s FP8 pair ``xq``: e4m3 rows and their
    column-major 1x128 scale (``norm.quant_buffers``). A plain weight reads ``x``."""
    if x.numel() == 0:
        return x.new_empty(x.shape[0], weight.shape[0])
    if not _shuffled(weight):
        return _plain_gemv(x, weight, scale)
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    _allow_block_fp8_linear()
    q, q_scale = xq
    return aiter_w8a8_block_fp8_linear(
        q, weight, [BLOCK, BLOCK], scale, input_scale=q_scale
    )


@triton.jit
def wg_dot(
    x_ptr,
    stride_x,
    xs_ptr,
    stride_xs_m,
    stride_xs_k,
    w_ptr,
    ws_ptr,
    stride_ws_n,
    stride_ws_k,
    pid,
    M,
    K: tl.constexpr,
    S: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    STAGES: tl.constexpr,
):
    """fp32 ``[ROWS, BN]``: rows ``[0, M)`` of the product for weight rows
    ``pid * BN + [0, BN)``, K in ``S`` slices of one batched dot each."""
    s = tl.arange(0, S)
    m = tl.arange(0, ROWS)
    kk = tl.arange(0, GROUP)
    nn = tl.arange(0, BN)
    # (n, k) of the (16, 16) preshuffle sits at
    # [n // 16, k // 32, k // 16 % 2, n % 16, k % 16] of [N/16, K/32, 2, 16, 16]
    w_row = (pid * (BN // 16) + nn // 16) * (K * 16) + (nn % 16) * 16
    acc = tl.zeros([S, ROWS, BN], dtype=tl.float32)
    for step in tl.range(0, K // S // GROUP, num_stages=STAGES):
        kb = s * (K // S // GROUP) + step
        k = kb[:, None] * GROUP + kk[None, :]
        x = tl.load(
            x_ptr + m[None, :, None] * stride_x + k[:, None, :],
            mask=m[None, :, None] < M,
            other=0.0,
        )
        xs = tl.load(
            xs_ptr + m[None, :] * stride_xs_m + kb[:, None] * stride_xs_k,
            mask=m[None, :] < M,
            other=0.0,
        )
        w_col = (k // 32) * 512 + (k // 16 % 2) * 256 + k % 16
        w = tl.load(w_ptr + w_col[:, :, None] + w_row[None, None, :])
        ws = tl.load(ws_ptr + (pid * BN // GROUP) * stride_ws_n + kb * stride_ws_k)
        acc += tl.dot(x, w) * (xs[:, :, None] * ws[:, None, None])
    return tl.sum(acc, axis=0)


@triton.jit
def _wg_kernel(
    x_ptr,
    stride_x,
    xs_ptr,
    stride_xs_m,
    stride_xs_k,
    w_ptr,
    ws_ptr,
    stride_ws_n,
    stride_ws_k,
    out_ptr,
    stride_out,
    M,
    K: tl.constexpr,
    S: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = wg_dot(
        x_ptr,
        stride_x,
        xs_ptr,
        stride_xs_m,
        stride_xs_k,
        w_ptr,
        ws_ptr,
        stride_ws_n,
        stride_ws_k,
        pid,
        M,
        K=K,
        S=S,
        GROUP=GROUP,
        BN=BN,
        ROWS=ROWS,
        STAGES=STAGES,
    )
    m = tl.arange(0, ROWS)
    n = pid * BN + tl.arange(0, BN)
    out = out_ptr + m[:, None] * stride_out + n[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=m[:, None] < M)


def wg_rows(rows: int) -> int:
    """The row tile of ``block_fp8_gemv_wg`` for ``rows`` rows."""
    return WG_ROWS if rows <= WG_ROWS else WG_MAX_ROWS


def wg_fits(rows: int, k: int, weight: torch.Tensor) -> bool:
    """Whether ``block_fp8_gemv_wg`` serves ``rows`` of K ``k`` on ``weight``."""
    return (
        0 < rows <= WG_MAX_ROWS
        and _shuffled(weight)
        and k % (WG_SPLIT * BLOCK) == 0
        and weight.shape[0] % WG_BN == 0
    )


def block_fp8_gemv_wg(
    x: torch.Tensor, xq, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``block_fp8_gemv_q`` with K split across the warps of each workgroup."""
    m, k = x.shape
    if not wg_fits(m, k, weight):
        return block_fp8_gemv_q(x, xq, weight, scale)
    q, q_scale = xq
    return _launch_wg(q, q_scale, weight, scale, m, x.dtype, wg_stages())


def _launch_wg(q, q_scale, weight, scale, m, dtype, stages):
    out = torch.empty(m, weight.shape[0], dtype=dtype, device=q.device)
    _wg_kernel[(weight.shape[0] // WG_BN,)](
        q,
        q.stride(0),
        q_scale,
        q_scale.stride(0),
        q_scale.stride(1),
        weight,
        scale,
        scale.stride(0),
        scale.stride(1),
        out,
        out.stride(0),
        m,
        K=q.shape[1],
        S=WG_SPLIT,
        GROUP=BLOCK,
        BN=WG_BN,
        ROWS=wg_rows(m),
        STAGES=stages,
        num_warps=WG_WARPS,
    )
    return out


def wg_stages() -> int:
    """The K-loop stage count ``select_wg_stages`` chose for this process."""
    return _wg_stages if _wg_stages is not None else select_wg_stages()


def select_wg_stages(device=None) -> int:
    """Choose ``wg_dot``'s ``num_stages`` once per process, before graph capture.

    Tries each of ``WG_STAGE_CANDIDATES`` in order and keeps the first that
    compiles and matches a torch product on every probe shape and row count.
    Raises when none does, so a Triton build with no correct loop stops the
    server at startup instead of serving wrong projections.
    """
    global _wg_stages
    if _wg_stages is not None:
        return _wg_stages
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "wg_dot's num_stages is chosen by running the kernel, which cannot "
            "happen during CUDA graph capture; call select_wg_stages() first"
        )
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    rejected = []
    with torch.cuda.device(device):
        for stages in WG_STAGE_CANDIDATES:
            reason = _probe(stages, device)
            if reason is None:
                if rejected:
                    logger.warning(
                        "wg_dot uses num_stages=%d on Triton %s (%s)",
                        stages,
                        triton.__version__,
                        "; ".join(rejected),
                    )
                else:
                    logger.info(
                        "wg_dot uses num_stages=%d on Triton %s",
                        stages,
                        triton.__version__,
                    )
                _wg_stages = stages
                return stages
            rejected.append(f"num_stages={stages}: {reason}")
    raise RuntimeError(
        f"wg_dot has no usable num_stages on Triton {triton.__version__}: "
        + "; ".join(rejected)
    )


def _probe(stages: int, device) -> str | None:
    """None when ``_wg_kernel`` compiles with ``stages`` and matches a torch
    product on every probe shape and row count, else why not."""
    fp8 = torch.float8_e4m3fn
    gen = torch.Generator(device=device).manual_seed(0)

    def pow2(*shape):
        exp = torch.randint(-1, 2, shape, generator=gen, device=device)
        return 2.0 ** exp.float()

    try:
        for n, k in _PROBE_SHAPES:
            plain = (torch.randn(n, k, generator=gen, device=device) * 0.1).to(fp8)
            # (n, k) of the (16, 16) preshuffle: [n // 16, k // 32, k // 16 % 2, n % 16, k % 16]
            shuffled = (
                plain.view(torch.uint8)
                .view(n // 16, 16, k // 32, 2, 16)
                .permute(0, 2, 3, 1, 4)
                .reshape(n, k)
                .view(fp8)
            )
            scale = pow2(n // BLOCK, k // BLOCK)
            w = plain.float() * scale.repeat_interleave(BLOCK, 0).repeat_interleave(
                BLOCK, 1
            )
            for m in _PROBE_ROWS:
                q = torch.randn(m, k, generator=gen, device=device).to(fp8)
                q_scale = pow2(k // BLOCK, m).t()
                want = (q.float() * q_scale.repeat_interleave(BLOCK, 1)) @ w.t()
                limit = _PROBE_TOL * want.abs().max().item()
                for _ in range(_PROBE_REPEATS):
                    got = _launch_wg(
                        q, q_scale, shuffled, scale, m, torch.bfloat16, stages
                    )
                    err = (got.float() - want).abs().max().item()
                    if not err <= limit:  # also true for NaN
                        return (
                            f"wrong result at N={n} K={k} M={m}: max error "
                            f"{err:.3g} over {limit:.3g}"
                        )
    except Exception as e:
        first = str(e).strip().splitlines()
        return f"{type(e).__name__}: {first[0] if first else ''}"
    return None


def _allow_block_fp8_linear() -> None:
    """The gfx950 small-M helper imports symbols its MoE module no longer defines.

    That import runs before the helper can decline the shape, so every block-FP8
    linear dies with it. Publishing the names lets the import finish; disabling
    the helper then takes the same block-scale GEMM the larger shapes use. A
    module that still defines them is left alone, helper included.
    """
    import importlib

    moe = importlib.import_module("sglang.kernels.ops.moe.smallm_moe_gfx950")
    if hasattr(moe, "_check"):
        return
    os.environ.setdefault("SGLANG_ROCM_SMALLM_FP8_BS", "0")

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
