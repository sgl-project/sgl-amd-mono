# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next's TP2 decode path of ``qwen38_flash_next.spec.SPEC`` (RFC
#43949): its refusals, build and forward in ``spec``, its check mode in ``check``,
and the gfx950 kernels, their launch runner and their weight operands in ``gfx950``.

The kernels come from vLLM's Qwen3.8-Flash-Next specialized decode
(vllm-project/vllm#63032 at 910ca119ca, Apache-2.0).
"""
