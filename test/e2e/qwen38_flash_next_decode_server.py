"""The Qwen3.8-Flash-Next servers that the TP2 decode serving test launches: two
MI355X, TP2, the MXFP4 checkpoint (BF16 attention and GDN projections), decode steps
(no speculative decoding)."""

import os

MODEL = os.environ.get(
    "QWEN38_FLASH_NEXT_MXFP4_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4"
)
ENV = {"SGLANG_USE_AITER": "1", "AITER_ONLINE_TUNE": "0"}
# The checkpoint card's TP2 cell (thinking off), except EP 1 (the kernels shard every
# expert by TP) and the GDN decode on the triton backend (the launches read its
# metadata). The radix cache stays on: the forward runs SGLang's decode-time GDN state
# save. Without --monokernel-backend, so a test can launch the normal path too.
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
    "64",
    "--cuda-graph-max-bs-decode",
    "64",
    "--reasoning-parser",
    "qwen3",
    "--default-chat-template-kwargs",
    '{"enable_thinking": false}',
]
