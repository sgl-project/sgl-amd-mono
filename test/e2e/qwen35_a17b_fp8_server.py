"""Qwen3.5-397B-A17B block-FP8 server for the TP4 monokernel test.

Four MI355X GPUs, aiter attention, decode graphs at batch sizes 1 through 8.
The model path is the block-FP8 checkpoint, not the MXFP4-AttnFP8 one.
"""

import os

MODEL = os.environ.get("QWEN35_FP8_MODEL_PATH", "/data/Qwen3.5-397B-A17B-FP8")
ENV = {
    "SGLANG_USE_AITER": "1",
    "AITER_ONLINE_TUNE": "0",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SAFETENSORS_FAST_GPU": "1",
}
ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--attention-backend",
    "aiter",
    "--moe-runner-backend",
    "aiter",
    "--mem-fraction-static",
    "0.85",
    "--max-running-requests",
    "16",
    "--disable-radix-cache",
    "--chunked-prefill-size",
    "8192",
    "--reasoning-parser",
    "qwen3",
    "--cuda-graph-config",
    '{"decode":{"backend":"full","max_bs":8,"bs":[1,2,3,4,5,6,7,8]}}',
]
