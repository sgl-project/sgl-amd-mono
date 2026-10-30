"""The Qwen3.8-27B servers the serving tests launch: MI35x, Quark MXFP4, TP1 or
TP2, the OOB MI355X recipe (the cookbook has no AMD recipe yet)."""

import os

MODEL = os.environ.get("QWEN3_8_27B_MODEL_PATH", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4")
ENV = {
    "SGLANG_USE_AITER": "1",
    "SAFETENSORS_FAST_GPU": "1",
}
# the spec's TP_MAX_ROWS: decode steps up to this many rows run the monokernels
ROW_CAP = {1: 4, 2: 8}


def args(tp: int) -> list:
    """The OOB config's server args at ``tp``, without --monokernel-backend, so a
    test can launch the normal path too."""
    return [
        "--trust-remote-code",
        "--quantization",
        "quark",
        "--tp-size",
        str(tp),
        "--attention-backend",
        "aiter",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--chunked-prefill-size",
        "16384",
        "--context-length",
        "139264",
        "--disable-radix-cache",
        "--max-running-requests",
        "256",
        "--mem-fraction-static",
        "0.9",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
    ]
