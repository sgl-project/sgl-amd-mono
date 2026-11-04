"""The Kimi-K3 server the serving tests launch: MI35x, TP8, with DSpark.

The measured win is on verify steps of 8 rows. That is DSpark block 7 at one
request in flight. It is also the only width the runtime serves here, so the
recipe pins it rather than leaving it to the defaults.
"""

import os

MODEL = os.environ.get("KIMI_K3_MODEL_PATH", "moonshotai/Kimi-K3")
DRAFT = os.environ.get(
    "KIMI_K3_DSPARK_MODEL_PATH", "moonshotai/RadixArk-Kimi-K3-DSpark"
)
# K1 has a build for either SSM state dtype; this picks the one served.
SSM_DTYPE = os.environ.get("KIMI_K3_MAMBA_SSM_DTYPE", "float32")
# aiter's SiTU MoE mode: a8w4 or a16w4 lay the experts out gate / up
# interleaved, a4w4 separated. The launch reads either layout.
SITU = os.environ.get("KIMI_K3_SITU", "a8w4")
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_AITER_K3_OPT": "1",
    "AITER_FLYDSL_FORCE": "1",
    "AITER_SITUV2_A8W4": "1" if SITU == "a8w4" else "0",
    "AITER_SITUV2_A4W4": "1" if SITU == "a4w4" else "0",
    # the aiter backend derates mem-fraction past a 8192 context; opt out so
    # --mem-fraction-static means what it says
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "1",
}
# Without --monokernel-backend, so a test can launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--attention-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--mem-fraction-static",
    "0.9",
    # One request a step, so every decode step is 1 x 8 rows and the launches
    # cover all of them. At 2 a step is 16 rows, which is wider than MAX_ROWS:
    # the launch declines those steps and they take the standard path, which
    # dilutes anything the launches get wrong. A GSM8K regression that is 9
    # points at this setting reads as half a point at 2.
    "--max-running-requests",
    "1",
    "--chunked-prefill-size",
    "4096",
    "--page-size",
    "128",
    "--mamba-full-memory-ratio",
    "0.9",
    "--mamba-ssm-dtype",
    SSM_DTYPE,
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DRAFT,
    "--speculative-dspark-block-size",
    "7",
]
