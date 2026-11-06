"""The models the AMD PR test runs, one entry a served model.

A PR runs the entries whose ``packages`` it touches; a change to ``common/``,
``sgl_amd_mono/__init__.py`` or ``pyproject.toml`` runs all of them. Each entry
names the server its accuracy and speed tests launch (a recipe module under
``test/e2e`` and its argument list, or ``env`` / ``args`` given here), the
checkpoints on the CI nodes, its kernel and serving tests, its kernel-only
timing scripts and the client concurrencies of the speed test.

An entry without a checkpoint on the CI nodes, or on a GPU the nodes do not
have, still runs its unit and kernel tests; the report names what it skipped.
"""

import os

# The checkpoint root on the cluster's shared storage (the runner's environment
# sets it). A job mirrors its model's checkpoints to the node's local disk first
# (slurm_job.sh) and serves them from there, at the same paths.
HF = os.environ.get("CI_MODEL_ROOT", "/models")

# The CI nodes' GPU.
GPU = "mi355x"

# Tests every PR runs, on any change.
COMMON_KERNEL_TESTS = [
    "test/kernels/test_guard_hip.py",
    "test/kernels/test_guard_multi_rank.py",
    "test/kernels/test_compiled_dispatch.py",
]

# Paths whose change runs every model.
SHARED_PATHS = ("sgl_amd_mono/common/", "sgl_amd_mono/__init__.py", "pyproject.toml")

# The fixed-length speed test (InferenceX's): random prompts of 8k +/- 20% tokens
# in, 1k +/- 20% out, 10 prompts a concurrency slot.
ISL, OSL, RANGE_RATIO, PROMPTS_PER_SLOT = 8192, 1024, 0.8, 10

# Accuracy: (sgl-eval benchmark, its arguments). A PR passes a benchmark when it
# scores at least the base branch's score less the tolerance (absolute).
EVALS = {
    "gsm8k": {"args": ["--max-tokens", "8192"], "tolerance": 0.015},
    "gpqa": {"args": ["--max-tokens", "32768"], "tolerance": 0.04},
    "aime26": {
        "args": ["--max-tokens", "32768", "--n-repeats", "4"],
        "tolerance": 0.07,
    },
    # vision models only
    "mmmu_pro": {"args": ["--max-tokens", "8192"], "tolerance": 0.03},
}
TEXT_EVALS = ["gsm8k", "gpqa", "aime26"]

# The speed gate. [Perf] and [Feat] PRs need a median-TPOT gain of at least
# PERF_GAIN at one of the model's gate concurrencies; every PR fails on a loss of
# more than PERF_LOSS at any measured concurrency.
PERF_GAIN, PERF_LOSS = 0.05, 0.02
GAIN_TYPES = ("Perf", "Feat")

MODELS = {
    "deepseek_v41": {
        "name": "DeepSeek-V4.1-Flash",
        "packages": ["deepseek_v41"],
        "tp": 4,
        "paths": {
            "DEEPSEEK_V41_FLASH_MODEL_PATH": f"{HF}/deepseek-ai/DeepSeek-V4.1-Flash"
        },
        "recipe": ("deepseek_v41_server", "ARGS"),
        "kernel_tests": [
            "test/kernels/test_deepseek_v41_kernels.py",
            "test/kernels/test_deepseek_v41_k2i.py",
        ],
        "e2e_tests": ["test/e2e/test_deepseek_v41_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 4 test/kernels/deepseek_v41/time_layer.py --widths 6,12"
        ],
        "evals": TEXT_EVALS,
        # DSpark verify steps of 6 rows a request: 8 requests fill the 48-row width
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "deepseek_v4": {
        "name": "DeepSeek-V4-Pro",
        "packages": ["deepseek_v4"],
        "tp": 8,
        "paths": {"DEEPSEEK_V4_PRO_MODEL_PATH": f"{HF}/deepseek-ai/DeepSeek-V4-Pro"},
        # the README's server: SGLang's two-pool fp8 unified KV
        "model_env": "DEEPSEEK_V4_PRO_MODEL_PATH",
        "env": {
            "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
            "SGLANG_DSV4_UNIFIED_KV_FP8": "1",
            "AITER_BF16_FP8_MOE_BOUND": "0",
        },
        "args": [
            "--tp",
            "8",
            "--attention-backend",
            "dsv4",
            "--enable-deepseek-v4-fp4-indexer",
            "--page-size",
            "256",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs-decode",
            "8",
            "--max-running-requests",
            "16",
        ],
        "kernel_tests": ["test/kernels/test_deepseek_v4_compile.py"],
        "e2e_tests": [],
        "microbench": [],
        "evals": TEXT_EVALS,
        # decode steps of up to 8 rows
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "glm53": {
        "name": "GLM-5.3",
        "packages": ["glm53"],
        "tp": 8,
        "paths": {"GLM_53_MODEL_PATH": f"{HF}/zai-org/GLM-5.3"},
        "recipe": ("glm53_server", "ARGS"),
        "kernel_tests": ["test/kernels/test_glm53_kernels.py"],
        "e2e_tests": ["test/e2e/test_glm53_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/glm53/time_layers.py --kind model --rows 2,4,8"
        ],
        "evals": TEXT_EVALS,
        # decode steps of up to 8 rows
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "glm53_flash": {
        "name": "GLM-5.3-Flash",
        "packages": ["glm53_flash"],
        "tp": 4,
        # amd/GLM-5.3-Flash-Quark-MXFP4; not on the CI nodes yet
        "paths": {"GLM_53_FLASH_MODEL_PATH": f"{HF}/amd/GLM-5.3-Flash-Quark-MXFP4"},
        "recipe": ("glm53_flash_server", "ARGS"),
        "kernel_tests": ["test/kernels/test_glm53_flash_kernels.py"],
        "e2e_tests": ["test/e2e/test_glm53_flash_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 4 test/kernels/glm53_flash/time_layers.py --kind model --rows 2,4,8"
        ],
        "evals": TEXT_EVALS,
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "kimi_k3": {
        "name": "Kimi-K3",
        "packages": ["kimi_k3"],
        "tp": 8,
        "paths": {
            "KIMI_K3_MODEL_PATH": f"{HF}/moonshotai/Kimi-K3",
            "KIMI_K3_DSPARK_MODEL_PATH": f"{HF}/RadixArk/Kimi-K3-DSpark",
        },
        "recipe": ("kimi_k3_server", "ARGS"),
        "kernel_tests": [
            "test/kernels/test_kimi_k3_kernels.py",
            "test/kernels/test_kimi_k3_k1.py",
            "test/kernels/test_kimi_k3_k1_conv.py",
            "test/kernels/test_kimi_k3_mla_pre.py",
        ],
        "e2e_tests": ["test/e2e/test_kimi_k3_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/kimi_k3/time_k2.py --rows 8 --flush 1024"
        ],
        "evals": TEXT_EVALS + ["mmmu_pro"],
        # the recipe runs one request a step (DSpark verify steps of 8 rows)
        "concurrency": [1],
        "gate_concurrency": [1],
    },
    "kimi_k31": {
        "name": "Kimi-K3.1",
        "packages": ["kimi_k31", "kimi_k3"],
        "tp": 8,
        # moonshotai/Kimi-K3.1; not on the CI nodes yet, and no serving recipe
        "paths": {},
        "kernel_tests": ["test/kernels/test_kimi_k31_k1_ring.py"],
        "e2e_tests": [],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/kimi_k31/layer_bench.py --rows 16,128"
        ],
        "evals": [],
        "concurrency": [],
        "gate_concurrency": [],
    },
    "minimax_m3": {
        "name": "MiniMax-M3 (MXFP4)",
        "packages": ["minimax_m3", "minimax"],
        "tp": 4,
        "paths": {
            "MINIMAX_M3_MODEL_PATH": f"{HF}/amd/MiniMax-M3-MXFP4",
            "MINIMAX_M3_EAGLE3_PATH": f"{HF}/Inferact/MiniMax-M3-EAGLE3-GQA",
        },
        "recipe": ("minimax_m3_server", "ARGS"),
        "kernel_tests": [
            "test/kernels/test_minimax_m3_compile.py",
            "test/kernels/test_minimax_m3_rows.py",
        ],
        "e2e_tests": ["test/e2e/test_minimax_m3_server.py"],
        "microbench": [],
        "evals": TEXT_EVALS,
        # EAGLE3 verify steps of 4 rows a request: 4 requests fill the 16-row width
        "concurrency": [1, 2, 4, 8],
        "gate_concurrency": [1, 4],
    },
    "minimax_m31": {
        "name": "MiniMax-M3.1-Flash",
        "packages": ["minimax_m31", "minimax"],
        "tp": 4,
        # the launches cover MI350X only; the CI nodes are MI355X
        "gpus": ["mi350x"],
        "paths": {},
        "kernel_tests": ["test/kernels/test_minimax_m31_ffn.py"],
        "e2e_tests": ["test/e2e/test_minimax_m31_server.py"],
        "microbench": [],
        "evals": [],
        "concurrency": [],
        "gate_concurrency": [],
    },
    "qwen38_flash_next": {
        "name": "Qwen3.8-Flash-Next (MXFP4)",
        "packages": ["qwen38_flash_next"],
        "tp": 1,
        # amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8; not on the CI nodes yet
        "paths": {
            "QWEN38_FLASH_NEXT_MODEL_PATH": f"{HF}/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
        },
        "recipe": ("qwen38_flash_next_server", "ARGS"),
        "kernel_tests": ["test/kernels/test_qwen38_flash_next_launches.py"],
        "e2e_tests": ["test/e2e/test_qwen38_flash_next_server.py"],
        "microbench": [],
        "evals": TEXT_EVALS,
        # NEXTN verify steps of 4 rows a request, up to 8 requests
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "muse_spark": {
        "name": "Muse-Spark",
        "packages": ["muse_spark"],
        "tp": 8,
        # no serving recipe under test/e2e yet
        "paths": {},
        "kernel_tests": [
            "test/kernels/test_muse_spark_k1.py",
            "test/kernels/test_muse_spark_k2.py",
            "test/kernels/test_muse_spark_k3.py",
            "test/kernels/test_muse_spark_k3_tp8.py",
            "test/kernels/test_muse_spark_kattn.py",
            "test/kernels/test_muse_spark_layer.py",
            "test/kernels/test_muse_spark_dattn.py",
            "test/kernels/test_muse_spark_dhalf.py",
            "test/kernels/test_muse_spark_dmlp.py",
        ],
        "e2e_tests": [],
        "microbench": [],
        "evals": [],
        "concurrency": [],
        "gate_concurrency": [],
    },
}


if __name__ == "__main__":
    import sys

    # `python3 models.py paths MODEL`: the checkpoints a job mirrors, one a line
    if len(sys.argv) == 3 and sys.argv[1] == "paths":
        print("\n".join(MODELS.get(sys.argv[2], {}).get("paths", {}).values()))
    else:
        sys.exit("usage: models.py paths MODEL")
