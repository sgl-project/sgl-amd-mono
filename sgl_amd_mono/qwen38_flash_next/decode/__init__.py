# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next decode steps of 1 to 192 rows at TP2 on the wide kernels of
vllm-project/vllm#65188 (``wide``): one persistent launch a GDN layer, and two a QSA
layer around SGLang's QSA attention. The TP2 path of
``qwen38_flash_next.spec.SPEC``: its refusals, build and forward in ``spec``, the
launch state in ``launcher``, the layers' operands in ``weights`` and check mode
in ``check`` (README.md)."""

from sgl_amd_mono.qwen38_flash_next.decode.wide import layout as L


def decode_widths(capture_sizes) -> tuple[int, ...]:
    """The widths that get a build: the decode CUDA graph capture sizes from
    ``MIN_TOKENS`` to ``MAX_TOKENS`` (SGLang pads a step to the smallest captured
    width that holds it)."""
    return tuple(
        sorted(s for s in set(capture_sizes) if L.MIN_TOKENS <= s <= L.MAX_TOKENS)
    )
