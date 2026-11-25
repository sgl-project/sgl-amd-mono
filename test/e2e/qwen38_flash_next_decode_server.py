"""The Qwen3.8-Flash-Next servers that the decode serving tests launch: the MXFP4
checkpoint (BF16 attention and GDN projections), decode steps (no speculative
decoding) of up to 192 rows on two MI355X at TP2 (``ARGS``), of up to 64 rows on one
at TP1 (``TP1_ARGS``)."""

import os

MODEL = os.environ.get(
    "QWEN38_FLASH_NEXT_MXFP4_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4"
)
ENV = {"SGLANG_USE_AITER": "1", "AITER_ONLINE_TUNE": "0"}
# TP2, as the checkpoint card serves it; EP 1 (the kernels shard every expert by TP)
# and the GDN decode on the triton backend (the launches read its metadata). Thinking
# off: the answers fit the tests' token budgets. Decode graphs up to 192 rows, the
# launches' largest width (SGLang's default capture sizes up to it). The radix cache
# stays on: the forward runs SGLang's decode-time GDN state save. With it each request
# holds 5 mamba state slots, and SGLang caps the running requests at the cache's slots
# / 5 (178 at the default 890 slots), so the cache gets 192 x 5. Without
# --monokernel-backend, so a test can launch the normal path too.
ARGS = [
    "--quantization",
    "quark",
    "--tp-size",
    "2",
    "--ep-size",
    "1",
    "--linear-attn-backend",
    "triton",
    "--moe-runner-backend",
    "aiter",
    "--mem-fraction-static",
    "0.85",
    "--max-running-requests",
    "192",
    "--max-mamba-cache-size",
    "960",
    "--cuda-graph-max-bs-decode",
    "192",
    "--reasoning-parser",
    "qwen3",
    "--default-chat-template-kwargs",
    '{"enable_thinking": false}',
]


def _with(args: list[str], values: dict[str, str]) -> list[str]:
    """``args`` with the value after each flag of ``values`` replaced."""
    out = list(args)
    for flag, value in values.items():
        out[out.index(flag) + 1] = value
    return out


# TP1 on one MI355X: decode steps of up to 64 rows, vllm-project/vllm#67706's TP1
# cap. At most 64 running requests, each with its 5 mamba state slots, and decode
# graphs up to 64 rows (with the spec's widths, 14 graphs). SGLang takes 0.85 of the
# static fraction with the aiter attention backend past 8,192 tokens of context: at
# 0.85 (0.72 of the GPU) the 179 GB of weights and the 34.5 GB of 320 mamba slots
# leave no KV cache, at 0.95 (0.81) 17.4 GB (760,000 tokens) on one MI350X
TP1_ARGS = _with(
    ARGS,
    {
        "--tp-size": "1",
        "--mem-fraction-static": "0.95",
        "--max-running-requests": "64",
        "--max-mamba-cache-size": "320",
        "--cuda-graph-max-bs-decode": "64",
    },
)
