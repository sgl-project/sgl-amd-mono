"""The GLM-5.3-Flash servers that the monokernel serving test launches: MI35x, TP4,
decode steps, with ``MTP`` the NextN draft's verify steps, with ``BF16_STATE``
a bf16 KDA state, with ``WIDE_MTP`` the wide verify launch, and with ``HICACHE``
HiCache under it."""

import os

MODEL = os.environ.get("GLM_53_FLASH_MODEL_PATH", "amd/GLM-5.3-Flash-Quark-MXFP4")
ENV = {"SGLANG_USE_AITER": "1", "AITER_ONLINE_TUNE": "0"}
# The cookbook's MI355X high-throughput MXFP4 cell, except: a checkpoint whose shared
# expert is MXFP4 (SGLang fuses it into the routed experts), EP 1 (the kernels shard
# every expert by TP) and a bf16 KV cache. Without --enable-amd-monokernel, so a test
# can launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--quantization",
    "quark",
    "--tp-size",
    "4",
    "--ep-size",
    "1",
    "--mem-fraction-static",
    "0.82",
    "--max-running-requests",
    "64",
    "--cuda-graph-max-bs-decode",
    "64",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--dsa-prefill-backend",
    "tilelang",
    "--dsa-decode-backend",
    "tilelang",
    "--kv-cache-dtype",
    "bfloat16",
    "--moe-runner-backend",
    "aiter",
    "--reasoning-parser",
    "glm45",
    "--mm-feature-transport",
    "cpu",
]
# MTP with one draft chain of 3 tokens: 4 verify rows a request, so verify steps of 1
# to 4 requests fit the 16-row width
MTP = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
# The KDA state pool in bf16 (f32 without it). SGLang's HIP prefill takes a bf16
# state with sgl-project/sglang#49036; without it, its Triton prefill runs
BF16_STATE = ["--mamba-ssm-dtype", "bfloat16"]
# The wide launch instead of the narrow ones: MTP with one draft chain of 6 tokens on
# SGLang's ReplaySSM ring, so verify steps of 3 to 32 requests (18 to 192 rows) run
# the wide launch and narrower ones SGLang's path
WIDE_MTP = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "6",
    "--enable-linear-replayssm-spec",
]
# HiCache as the AgentX c64 recipe runs it (pages written to the host when evicted),
# over a device KV pool of 32768 tokens that a test can fill to evict prefixes
HICACHE = [
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--enable-hierarchical-cache",
    "--hicache-write-policy",
    "write_back",
    "--hicache-io-backend",
    "kernel",
    "--hicache-mem-layout",
    "page_first",
    "--hicache-ratio",
    "4",
    "--max-total-tokens",
    "32768",
    "--enable-metrics",
]
