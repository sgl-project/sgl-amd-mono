# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# tests/models/qwen4_exp/amd_specialized_decode/test_stages.py
"""The Qwen3.8-Flash-Next wide decode's build choices on CPU: the widths that get
a build."""

import pytest

from sgl_amd_mono.qwen38_flash_next.decode import decode_widths


def test_decode_widths_take_every_capture_size():
    """Every CUDA graph capture size in 1-192 gets a build at TP2, in 1-64 at TP1
    (SGLang pads a step to the smallest captured width that holds it), and a build
    of any width in that range is valid; a TP the kernels are not built for has
    no build."""
    from sgl_amd_mono.qwen38_flash_next.decode.wide.layer import LayerBuild

    widths = (1, 2, 3, 4, 8, 12, 16, 24, 32, 64, 128, 129, 136, 160, 184, 192)
    assert decode_widths([*widths, 193, 200, 256], 2) == widths
    assert decode_widths([*widths, 193, 200, 256], 1) == widths[: widths.index(64) + 1]
    assert decode_widths([72, 128], 1) == ()
    with pytest.raises(ValueError, match="TP 4"):
        decode_widths(widths, 4)
    for tp, cap in ((2, 192), (1, 64)):
        for rows in range(1, cap + 1):
            LayerBuild(kind="gdn", tokens=rows, tp=tp)
        for rows in (0, cap + 1):
            with pytest.raises(ValueError):
                LayerBuild(kind="gdn", tokens=rows, tp=tp)
    with pytest.raises(ValueError, match="TP 3"):
        LayerBuild(kind="gdn", tokens=1, tp=3)
