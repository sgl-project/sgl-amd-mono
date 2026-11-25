"""The NVFP4 row quantizer of the M3.1 launches against SGLang's own (``_store_nvfp4_quarter``
of sglang/kernels/ops/attention/minimax_m3_qk_norm_rope.py), byte for byte, on one gfx950
GPU: payload and E4M3 scales of 128-column rows across scales, ties and zeros."""

import pytest
import torch

pytest.importorskip("flydsl")
triton = pytest.importorskip("triton")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("gfx950 only", allow_module_level=True)

import triton.language as tl  # noqa: E402

from sgl_amd_mono.minimax_m31 import nvfp4  # noqa: E402

qk = pytest.importorskip("sglang.kernels.ops.attention.minimax_m3_qk_norm_rope")
_quarter_cols, _store_nvfp4_quarter = qk._quarter_cols, qk._store_nvfp4_quarter


@triton.jit
def _sglang_rows(x_ptr, pay_ptr, sc_ptr, n, BLOCK_T: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    r5 = rows[:, None, None, None, None]
    mask = r5 < n
    for g in tl.static_range(4):
        x = tl.load(x_ptr + r5 * 128 + _quarter_cols(g), mask=mask, other=0.0)
        _store_nvfp4_quarter(
            pay_ptr + r5 * 64, sc_ptr + r5 * 8, mask, x.to(tl.float32), g, BLOCK_T, 1
        )


def _reference(x):
    n = x.shape[0]
    pay = torch.zeros(n, 64, dtype=torch.uint8, device=x.device)
    sc = torch.zeros(n, 8, dtype=torch.uint8, device=x.device)
    _sglang_rows[(triton.cdiv(n, 16),)](x, pay, sc, n, BLOCK_T=16)
    return pay, sc


def _ours(x):
    n = x.shape[0]
    pay = torch.zeros(n, 64, dtype=torch.uint8, device=x.device)
    sc = torch.zeros(n, 8, dtype=torch.uint8, device=x.device)
    nvfp4.build_rows(n)(
        x.data_ptr(), pay.data_ptr(), sc.data_ptr(), stream=torch.cuda.current_stream()
    )
    torch.cuda.synchronize()
    return pay, sc


def _tie_row():
    """Groups of amax 6 (scale exactly 1): every e2m1 edge and its bf16 neighbours."""
    import math

    vals = []
    for e in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        ulp = 2.0 ** (math.floor(math.log2(e)) - 7)
        vals += [e - ulp, e, e + ulp]
    groups = [[6.0] + vals[:15], [6.0] + vals[15:] + [-v for v in vals[:9]]]
    row = []
    for g in range(8):
        sign = -1.0 if g % 4 >= 2 else 1.0
        row += [sign * v for v in groups[g % 2]]
    return row


def _rows():
    g = torch.Generator(device="cuda").manual_seed(0)
    parts = [
        torch.randn(256, 128, device="cuda", generator=g) * s
        for s in (1e-4, 1e-2, 1.0, 30.0, 400.0)
    ]
    ties = torch.tensor([_tie_row()] * 4, device="cuda")
    zeros = torch.zeros(4, 128, device="cuda")
    zeros[1, ::2] = -0.0
    zeros[2, 5] = 1e-30  # a group whose scale underflows to 0 in E4M3: inf / NaN codes
    return torch.cat(parts + [ties, zeros]).bfloat16()


def test_rows_match_sglang_byte_for_byte():
    x = _rows()
    want_pay, want_sc = _reference(x)
    got_pay, got_sc = _ours(x)
    bad = (got_sc != want_sc).any(1) | (got_pay != want_pay).any(1)
    assert not bad.any(), (
        f"{int(bad.sum())} of {x.shape[0]} rows differ, first {bad.nonzero()[:4].flatten().tolist()}"
    )
