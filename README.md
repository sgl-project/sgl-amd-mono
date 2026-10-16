# SGLang-AMD-Mono

## Introduction

**SGLang-AMD-Mono** is the monokernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for AMD Instinct GPUs. A monokernel is a persistent kernel that runs most of a decoder layer, including its attention, its mixture-of-experts block and both tensor-parallel all-reduces, in one or two launches. This removes the launch gaps and collective latency that dominate small-batch decoding.

The library implements SGLang's monokernel interface ([RFC #43949](https://github.com/sgl-project/sglang/issues/43949)) and loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point), which registers each model's spec under the monokernel backend `amd`. With `--monokernel-backend amd`, every decode and speculative-verify step that a spec serves runs on its kernels, and every other step runs SGLang's standard path.

> **Experimental:** `--monokernel-backend` is an experimental SGLang feature. The kernels, the models and setups they cover, and the plugin interface may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | MI350X, MI355X (gfx950) | 2, 4 | All 40 layers in decode and DSpark verify steps of up to 48 rows, and the DSpark draft stages |
| GLM-5.3-Flash | MI355X (gfx950) | 2, 4, 8 | The 41 MoE layers but the last (one launch each) in decode steps of up to 8 rows (4 at TP2) |

### Design

- **One or two launches per layer.** DeepSeek-V4.1-Flash runs two. K1 runs the attention's input mixing and projections and writes the KV cache. K2 runs the sparse attention, the output projections, the FFN mixing, the MoE and both all-reduces. On the indexer layers, K2i runs between them: it scores the FP4 index keys in one launch, and SGLang's top-k kernels select each row's top-k from its logits. GLM-5.3-Flash runs each covered layer as one launch.
- **Graph-safe routing.** Each step is routed by its CUDA graph key alone, so every rank makes the same decision and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** Every wait on another workgroup or GPU has a deadline. A failed wait stops the launch on every rank, so no token from a failed step is returned.
- **In-place weights.** The kernels read the weights SGLang loaded. The few weights that need another layout are copied once at startup, and each copy is logged.

## Quick Start

### Requirements

- SGLang with the monokernel interface (`sglang.srt.monokernels`, in review as [sgl-project/sglang#44353](https://github.com/sgl-project/sglang/pull/44353))
- ROCm with FlyDSL and AITER, as included in SGLang's ROCm images
- AMD Instinct MI350X or MI355X

### Installation

```bash
git clone https://github.com/sgl-project/sgl-amd-mono.git
cd sgl-amd-mono
pip install -e .
```

### Usage

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --fp8-gemm-backend aiter \
  --monokernel-backend amd
```

If the model, GPU or configuration is not supported, the server refuses to start and lists every reason.

## Repository Structure

```text
sgl-amd-mono/
├── sgl_amd_mono/
│   ├── common/          # HIP device runtime: guarded waits, epochs, peer memory
│   ├── deepseek_v41/
│   │   ├── spec.py      # target model: spec and forward
│   │   ├── draft.py     # DSpark draft model: spec and forward
│   │   └── gfx950/      # kernels and launch runner
│   └── glm53_flash/
│       ├── spec.py      # spec and forward
│       ├── check.py     # SGLANG_AMD_MONO_CHECK: each layer against SGLang's own
│       └── gfx950/      # kernels and launch runner
└── test/
    ├── unit/            # CPU: the specs and the SGLang interface contract
    ├── kernels/         # GPU: each launch against SGLang's reference ops
    └── e2e/             # GPU: serving accuracy and fault injection
```

## Testing

```bash
pytest test/unit               # CPU only
pytest test/kernels test/e2e   # 4 x MI350X or MI355X, with DEEPSEEK_V41_FLASH_MODEL_PATH set
                               # (GLM-5.3-Flash: 8 GPUs, GLM_53_FLASH_MODEL_PATH)
```

## Contributing

Format code with [pre-commit](https://pre-commit.com/) (`pre-commit install`). Every module that reads SGLang internals declares them in `SGLANG_READS`. `test/unit/test_contract.py` checks each declared name against the installed SGLang, so an incompatible SGLang change fails on CPU.

## License

[Apache License 2.0](LICENSE)
