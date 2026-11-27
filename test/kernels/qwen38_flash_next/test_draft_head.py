# SPDX-License-Identifier: Apache-2.0
"""The FP8 draft lm_head (W8A16 GEMV) against its dequantized fp32 reference."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [1, 3, 8, 32, 40])
def test_gemv_matches_dequantized_reference(rows):
    from sgl_amd_mono.qwen38_flash_next.gfx950 import draft_head

    torch.manual_seed(rows)
    n, k = 248320, 2560
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q, s = draft_head.quantize_rows(w)
    x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
    ref = x.float() @ (q.float() * s[:, None]).t()
    got = draft_head.gemv(x, q, s, torch.float32)
    torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)
    if rows <= 32:
        ids = draft_head.gemv_argmax(x, q, s)
        assert (ids == got.argmax(1)).float().mean() >= 0.99
