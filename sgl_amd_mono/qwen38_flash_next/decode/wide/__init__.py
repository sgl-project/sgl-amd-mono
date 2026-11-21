# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# vllm/models/qwen4_exp/amd/specialized_decode/wide/__init__.py
"""The wide decode kernels: one persistent launch per decoder layer, each a
sequence of grid-wide phases over every row of the step."""
