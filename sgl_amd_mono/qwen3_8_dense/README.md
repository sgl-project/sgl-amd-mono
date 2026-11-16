# Qwen3.8-27B dense monokernels on gfx950

This document describes `sgl_amd_mono/qwen3_8_dense/`. It covers the two launches K1 and K2, the files, the tests and the
SGLang integration. SGLang uses the folder through the monokernel interface of RFC sgl-project/sglang#43949. The kernels
come from sgl-project/sglang#43558.

## 1. Scope

The width of a step is its number of rows. A decode step has one row for each request. GDN is Gated DeltaNet, the linear
attention of Qwen3.8. Of the model's 64 decoder layers, 48 are GDN layers and 16 are full-attention layers. Every layer
has a dense MLP.

| | the model in `spec.py` |
| --- | --- |
| model class | `Qwen3_5ForConditionalGeneration` |
| checkpoint | `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, Quark MXFP4 with e8m0 scales |
| GPU and TP | gfx950 (MI350X, MI355X), TP 1 or 2 |
| SGLang steps | decode |
| widths | 1 to 4 at TP1 and 1 to 8 at TP2 (`TP_MAX_ROWS`; wider steps were slower than SGLang's path on MI355X) |
| launches per step | TP1: 112 (48 K1 and 64 K2). TP2: 112 on each rank |
| other steps | SGLang's forward runs prefill, extend, wider decode steps and steps with input embeddings |

| file | contents |
| --- | --- |
| `__init__.py` | `HF_CONFIG`, the Qwen3.8-27B config values that the kernels are compiled for. The spec refuses a model with other values. |
| `spec.py` | the `MonoSpec`, its refusal reasons, the build and the forward `_Built.forward` (`_tp1`, `_tp2`) |
| `check.py` | check mode, which `SGLANG_AMD_MONO_CHECK=1` turns on |
| `gfx950/dense_gdn.py` | K1 as the FlyDSL kernel `dense_gdn_pre`, its build key `K1Build` and its scratch layout |
| `gfx950/dense_ffn.py` | K2 as the FlyDSL kernel `dense_ffn`, its build key `FfnBuild`, and the MXFP4 shuffle `shuffle_w` |
| `gfx950/weights.py` | which layers the kernels take, and their MXFP4 operands in the order the kernels read |
| `gfx950/runner.py` | the step epoch, one scratch buffer for each kernel, the precompile and one call for each launch |
| `gfx950/ops.py` | the device helpers that `sgl_amd_mono.common` does not have |

| test file | contents |
| --- | --- |
| `test/unit/test_qwen3_8_dense_spec.py` | CPU tests of the registry entry, the widths, the routing, the refusal reasons and the ASM FP4 layout |
| `test/kernels/test_qwen3_8_dense_kernels.py` | K1 and K2 against SGLang's own ops, bit for bit, at TP1 and on both TP2 ranks, widths 1 to 8 (`qwen3_8_dense/check_k1.py`, `check_k2.py`) |
| `test/e2e/qwen3_8_dense_server.py` | the server flags (the MI355X OOB config) and the environment of the serving tests |
| `test/e2e/test_qwen3_8_dense_server.py` | the serving tests at TP1 and TP2, monokernels on against off, and check mode |

## 2. The launches

Each launch is one persistent grid of 256 CTAs with 512 threads each.

**TP1.** For each layer, `_tp1` runs:

1. On a GDN layer, K1: the residual add and the input norm, `in_proj_qkvz` and `in_proj_ba`, the causal conv1d update,
   the gates, the gated delta rule on the fp32 state, and the gated RMSNorm. Its output is the input of `out_proj`. On an
   attention layer, SGLang runs the input norm and the attention up to the input of `o_proj`.
2. K2: `out_proj` or `o_proj`, the residual add, the post-attention norm, `gate_up_proj`, SiLU and mul, and `down_proj`.

**TP2.** `_tp2` keeps SGLang's layer loop, its norms and both all-reduces of each layer. K1 runs a GDN layer from the
normed input to its partial `out_proj` output, on the rank's half of the heads. K2 runs the rank's half of the MLP from
the normed input to its partial `down_proj` output. No value crosses ranks inside a launch.

SGLang runs the embedding, the final norm and the logits processor.

## 3. Numerics and weights

- Each GEMM in K1 and K2 takes the K splits of SGLang's Triton `gemm_afp4wfp4`, in the same order. The conv, the delta
  rule and the gated norm use the rounding, reduction order and FMA contractions of SGLang's Triton kernels. Thus both
  launches match SGLang's Triton FP4 path bit for bit (`NUMERICS` is `BITWISE`, every bound 0).
- SGLang's ASM FP4 GEMM (`--fp4-gemm-backend aiter`, or `auto` on gfx95 for AITER-tuned shapes) quantizes activations
  with its own kernel. On it the launches are close to SGLang's path but not equal. Under
  `--enable-deterministic-inference` or `--rl-on-policy-target` the spec refuses a layer on that GEMM.
- The kernels read each MXFP4 weight in the (16, 16)-shuffled order with plain e8m0 scales. A layer on the Triton GEMM is
  copied once into that order. A layer on the ASM GEMM is in that order already and is read in place. Only its scales
  are copied out of the ASM tile order, and `in_proj_qkvz` is copied as it is, since SGLang may repack it after the
  build. All copies go through `ctx.copies`, so SGLang counts them before it sizes the KV pool.
- The GDN state is SGLang's mamba pool. K1 binds to it at the first launch and updates it in place.

## 4. SGLang integration

`spec.py` accepts gfx950, TP 1 or 2, bf16, Quark MXFP4, an fp32 SSM state with a bf16 conv state,
`SGLANG_USE_AITER=1` and FlyDSL. It refuses pipeline parallelism, aux hidden states (speculative decoding) and
`--enable-linear-replayssm`. If there is a reason, the server stops at startup with an error that lists it.

On each rank, `_build` compiles one K1 and one K2 launch for each CUDA graph width up to the cap before the first step.
SGLang chooses between the launches and its forward from the CUDA graph key of a step, so each rank makes the same
choice and a graph can replay it.

The folder needs no SGLang changes. Start the server with `--enable-amd-monokernel` (`--monokernel-backend amd` before
sgl-project/sglang#45389).

## 5. Correctness checks

- The unit tests run on CPU. Section 1 gives what they check.
- The kernel tests run on 2 GPUs with random weights in SGLang's layouts. They compare K1 and K2 with SGLang's stock ops,
  bit for bit, at widths 1 to 8.
- Check mode (`SGLANG_AMD_MONO_CHECK=1`, eager only) runs SGLang's own layer code next to each launch on the same input
  and checks every output and state row against `NUMERICS`. Use `--fp4-gemm-backend triton`; on the ASM GEMM it warns
  at startup and the bit-exact bounds miss.
- The e2e test compares a monokernel server with the same server without it: the startup routing line, GSM8K past the
  cap, the KL divergence of decode against prefill, a batch that crosses the cap and back, an image request and the
  decode speed inside and past the cap.

## 6. Limitations

- the checkpoint `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (or one with the same config and quantization)
- TP 1 or 2 on gfx950
- decode steps of up to 4 rows at TP1 and 8 at TP2; no speculative verify steps
- bit-exact results only on `--fp4-gemm-backend triton`
