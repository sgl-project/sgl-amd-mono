# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#67421 at 28f094565b (Apache-2.0):
# vllm/models/qwen3_5/amd/specialized_decode/wide/__init__.py
"""The wide decode kernels: one persistent launch per decoder layer, each a
sequence of grid-wide phases over every row of the step."""
