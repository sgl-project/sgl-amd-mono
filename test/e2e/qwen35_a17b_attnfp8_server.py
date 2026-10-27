"""Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2 server for the TP4 verify monokernel.

Four MI355X GPUs. EAGLE verify rows are 4 per request, so batch sizes 1 and 2
are the widths 4 and 8 the kernel serves. The arms differ only by the
monokernel flag that bench_server.py adds.
"""

import os

MODEL = os.environ.get(
    "QWEN35_ATTNFP8_MODEL_PATH", "/data/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2"
)
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    "AITER_FLYDSL_FORCE": "1",
    "AITER_ONLINE_TUNE": "0",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT8",
    "SAFETENSORS_FAST_GPU": "1",
}
ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--attention-backend",
    "aiter",
    "--moe-runner-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "16",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "2",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]
