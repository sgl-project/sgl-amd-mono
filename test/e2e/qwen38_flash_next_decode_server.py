"""The Qwen3.8-Flash-Next servers that the TP2 decode serving tests launch: two
MI355X, TP2, the MXFP4 checkpoint (BF16 attention and GDN projections), decode steps
(no speculative decoding) of up to 192 rows."""

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
