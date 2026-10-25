# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next's TP2 decode launches (RFC #43949): the gfx950 kernels, their
launch runner and their weight operands in ``gfx950``.

The kernels come from vLLM's Qwen3.8-Flash-Next specialized decode
(vllm-project/vllm#63032 at 910ca119ca, Apache-2.0).
"""
