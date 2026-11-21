# MiniMax-M3.1-Flash monokernels on gfx950

This document describes `sgl_amd_mono/minimax_m31/`. It covers the launches, the files, the tests and the switches.
SGLang uses the folder through the monokernel interface of RFC sgl-project/sglang#43949. `sgl_amd_mono/minimax/spec.py`
selects this folder for MiniMax-M3.1-Flash and `sgl_amd_mono/minimax_m3/` for MiniMax-M3.

## 1. Scope

A row is one query token of a step. A DSpark verify step has 8 rows for each request: the 7 draft tokens and the
token before them.

| | target model in `spec.py` |
| --- | --- |
| model class | `MiniMaxM3SparseForConditionalGeneration` (`minimax_m3_vl`) |
| checkpoint | `MiniMaxAI/MiniMax-M3.1-Flash`: NVFP4 routed experts, which SGLang converts to MXFP4 at load, and MXFP8 (block 32) for the other weights |
| GPU and TP | gfx950 (MI350X and MI355X), TP 4 |
| SGLang steps | decode, and DSpark `target_verify` |
| widths | the FFN launch and K1: 1 to 16 rows. K2a and K2b: verify steps of 1 or 2 requests (8 or 16 rows). |
| KV cache | SGLang's NVFP4 pool (`MiniMaxM3Nvfp4KVPool`), page size 128 |
| other steps | SGLang's forward runs prefill, extend and the steps wider than 16 rows |

The model has 60 layers, and each layer has sparse attention. Layers 0 to 2 have a dense FFN. Layers 3 to 59 have the
MoE: this document calls them the 57 sparse layers.

| file | contents |
| --- | --- |
| `__init__.py` | the package description |
| `spec.py` | the `MonoSpec`, its refusal reasons, the build and the forward `_Built.forward` |
| `runner.py` | the host side of the FFN launch and of K1: compiled kernels, scratch and the epoch word for each width |
| `ffn.py` | the FFN launch: from the attention output to the reduced MoE output |
| `a16w4.py` | the MFMA operands of the routed experts: bf16 activations, MXFP4 weights in AITER's post-load layout |
| `a16w8.py` | the MFMA operands of the shared expert: bf16 activations, MXFP8 weights |
| `front.py` | K1, the attention front: from a sparse layer's input to the inputs of its attention |
| `index.py` | K2a, the verify indexer: from the index Q of a verify step to the top-16 context blocks of each row |
| `sparse.py` | K2b, the verify sparse attention: from Q and the top-16 blocks to the attention output |
| `nvfp4.py` | the row quantizer of SGLang's NVFP4 pool, which K1 uses to write K, V and the index K |
| `sync.py` | the tagged hand-offs and the bounded waits of one launch |
| `residency.py` | the HIP occupancy check: a persistent grid needs every CTA resident at once |

| test file | contents |
| --- | --- |
| `test/unit/test_minimax_m31_spec.py` | CPU tests of the spec: the registry entry, the widths and the refusals |
| `test/kernels/test_minimax_m31_ffn.py` | the FFN launch against SGLang's operations, and the o_proj partial bit for bit |
| `test/kernels/test_minimax_m31_nvfp4.py` | the NVFP4 rows against SGLang's quantizer, byte for byte |
| `test/kernels/test_minimax_m31_attention.py` | K1, K2a and K2b against SGLang's kernels |
| `test/kernels/minimax_m31/time_ffn.py` | the time of the FFN launch at TP 4, with cold weights and a fixed number of routed experts |
| `test/e2e/test_minimax_m31_server.py` | the serving tests: verify and decode GSM8K, check mode, and a stalled rank |

## 2. A step

In a routed step, `_Built.forward` runs SGLang's embedding and the 3 layers with a dense FFN. Then it runs each
sparse layer:

1. Layer 3: SGLang runs the input norm and the attention. Layers 4 to 59: K1 runs the input norm and the attention
   front (section 4).
2. The attention core:
   - in a verify step of 1 or 2 requests, K2a picks the blocks of each row and K2b attends over them;
   - in any other routed step, SGLang's sparse attention runs on the Q and the index Q that K1 wrote.
3. The FFN launch runs o_proj and the FFN, and writes the layer output and the residual.

After the last layer, SGLang runs the final norm and the logits processor.

## 3. The FFN launch

The FFN launch is one grid of 256 CTAs with 512 threads each, one launch for each sparse layer and rank. All CTAs stay
resident until the launch ends. The table gives its stages in order.

| stage | CTAs | work |
| --- | --- | --- |
| `push` | 128 x 48 columns | o_proj as SGLang runs it: per-token FP8 rows times the rowwise-FP8 weight. It sends the bf16 partial sum to each rank. |
| `reduce` | 192 | adds the partial sums of the ranks in rank order, then the residual. It writes the residual and the sum of squares. |
| `norm` | one CTA for each row | the Gemma RMSNorm of the sum, into X (bf16) |
| `router` | 8 x 16 experts x 16 K parts | the FP32 router GEMV with f32 MFMA |
| `route` | one CTA for each row | sums the parts in order, then sigmoid, bias, top 4 and the renormalized weights times 2.0 |
| `table` | all | the union of the picked experts in LDS. Slot 0 is the shared expert for all rows. |
| `up` | 48 x slots | 16 gate rows and 16 up rows of a slot's expert, then SwiGLU-OAI into MID (bf16) |
| `down` | 384 x 16 rows | the down GEMV of each slot, times the route weight, plus the shared expert. It sends the partial sum to each rank. |
| `final` | 192 | adds the partial sums of the ranks in rank order and writes `out` |

The launch reads SGLang's MXFP4 routed experts and its rowwise-FP8 o_proj in place. The shared expert keeps the MXFP8
values of the checkpoint: the load hook keeps the shard of this rank, and the build lays it out for `a16w8.py`.

## 4. K1, K2a and K2b

K1 is one grid of 240 CTAs: 20 heads times 12 chunks of 512 hidden columns. CTA (h, c) computes the 128 projection
columns of head h over chunk c.

| stage | work |
| --- | --- |
| `x` | the residual plus the MoE output in f32, the new residual (bf16), the sum of squares, and the FP8 X of the chunk |
| `gemm` | the FP8 MFMA of the chunk on the rowwise-FP8 fused weight, into a partial sum (f32) |
| `head` | the last CTA of the head adds the partial sums in chunk order and applies the norm scale. Then it norms and rotates Q and the index Q, and stores K, V and the index K into SGLang's pool. |

K1 quantizes X once for each 512-column chunk, not once for each row. Thus its bits differ from SGLang's, but its error
is not larger (`test_minimax_m31_attention.py`).

K2a is one grid of 256 CTAs. A wave scores one 128-token block for the 8 rows of a request: it dequantizes the NVFP4
index K to fp16 as SGLang's kernel does, and runs fp16 MFMA. The last CTA of the request picks the 16 blocks of each
row in SGLang's order: the higher score first, then the lower block id. The last block of a row is always picked.

K2b is one grid of 16 CTAs for each row, one CTA for each picked block. A CTA of 4 waves attends to its block, and the
last CTA of the row merges the softmax states of the 16 blocks into the bf16 output.

## 5. Hand-offs and guarded waits

- Stages and ranks pass values through the tagged pairs of `sync.py`. Each value has a tag from the step epoch and the
  layer. A reader waits for the current tag, so no step has to clear the scratch.
- Each wait has a deadline. A wait that passes it stops the launch on every rank, so no token of that step goes to a
  client. The fault test of `test_minimax_m31_server.py` stalls one wait to check this.

## 6. Switches

| variable | effect |
| --- | --- |
| `SGLANG_AMD_MONO_M31_FRONT=0` | SGLang runs the attention front, K2a and K2b |
| `SGLANG_AMD_MONO_M31_INDEX=0` | SGLang runs the index scores, the top-k and the sparse attention |
| `SGLANG_AMD_MONO_M31_SPARSE=0` | SGLang runs the sparse attention |
| `SGLANG_AMD_MONO_M31_K2_DEBUG=1` | K2a and K2b log out-of-range slots and lengths |
| `SGLANG_AMD_MONO_CHECK=1` | check mode: eager steps; each sparse layer's FFN launch against SGLang's modules on a copy of its input |
