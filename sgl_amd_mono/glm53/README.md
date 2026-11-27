# GLM-5.3 monokernels on gfx950

`sgl_amd_mono/glm53/` runs GLM-5.3's decode and MTP verify steps of up to 16 rows (12 at TP4) on MI355X (gfx950), at
TP 4 or 8, through SGLang's monokernel interface (RFC sgl-project/sglang#43949). Each of the 78 decoder layers is one
launch, and the final norm, LM head and logits all-gather are one more (the tail). The NextN draft's steps run on the
same launches. SGLang's forward runs prefill, extend and the wider steps.

## Widths

A step's rows are its requests times their tokens (1 in decode, the MTP draft tokens in verify). A step runs at the
next width the server built; its rows past the step's are pad rows (a one-row launch never completes on gfx950).

| TP | widths (`spec.tp_widths`) | a wider step |
| --- | --- | --- |
| 8 | 2, 4, 8, 12, 16 | SGLang's forward |
| 4 | 2, 4, 8, 12 | SGLang's forward |

At TP4 a rank has 16 attention heads, and a 16-row launch's LDS arena outgrows the 160 KiB a workgroup has, so
`spec.TP_WIDTHS` caps TP4 at 12 rows. Up to 8 rows a launch's arena stays within 128 KiB. GLM-5.2's MXFP4 experts on
MXFP8 rows (the FP4 MFMA, `native_fp4`) launch up to 8 rows at either TP size (`spec.NATIVE_FP4_ROWS`). The NextN
draft runs its extends of whole requests on the launches at their own rows: any rows up to 4, even rows up to 8, and
past 8 the TP size's widths (an extend of 5 or 7 rows, or a 9-row extend of 3 requests of 3 tokens, runs SGLang's
path: above 4 rows a launch has even rows). It runs its decode steps of up to 4 rows on the launches; a draft decode
step of 5 or more rows runs SGLang's path.

## Files

| file | contents |
| --- | --- |
| `__init__.py` | `HF_CONFIG`, the GLM-5.3 config values the kernel builds in |
| `spec.py` | the spec, its refusal reasons, the build, the forward and the check bounds `NUMERICS` |
| `draft.py` | the spec and forward of the NextN (MTP) draft |
| `check.py` | check mode (`SGLANG_AMD_MONO_CHECK=1`): each launch against SGLang's layer on the same input |
| `tables.py` | a step's row tables and the draft's top-k seed |
| `gfx950/kernel.py` | the FlyDSL layer launch and the tail |
| `gfx950/runner.py` | the build key `Glm53Build`, the runtimes and the per-layer ops |
| `gfx950/weights.py` | the tensors a launch reads, from SGLang's loaded modules |
| `gfx950/layout.py`, `config.py`, `tiles.py`, `packing.py` | shapes, scratch layout, CTA schedule and weight packing |

Tests: `test/unit/test_glm53_*.py` (CPU), `test/kernels/test_glm53_kernels.py` with the scripts in
`test/kernels/glm53/` (8 GPUs, random weights), and `test/e2e/test_glm53_server.py` (the server at TP 8 on the
checkpoint `GLM_53_MODEL_PATH` names).

## Checkpoints

| module | `zai-org/GLM-5.3` | `amd/GLM-5.3-Quark-MXFP4-AttnFP8` | `amd/GLM-5.2-MXFP4` |
| --- | --- | --- | --- |
| attention and indexer linears | FP8 (128 x 128 blocks) | FP8 (128 x 128 blocks) | bf16, copied once at startup |
| dense MLPs (layers 0 to 2) | FP8 | MXFP4 | bf16, copied once at startup |
| routed and shared experts | FP8 | MXFP4 | MXFP4 |

`Glm53Build.mxfp4` selects the MLP format. FP8 MLPs quantize their activations to FP8 per 128 values. MXFP4 MLPs
quantize them to MXFP4 (one E8M0 scale per 32 values) and run FP4 x FP4 MFMAs; the quantization equals SGLang's
bitwise (AITER `fused_moe` per_1x32 for the experts, Triton `dynamic_mxfp4_quant` for the dense MLPs).

`amd/GLM-5.2-MXFP4` has GLM-5.3's architecture, so the same launches serve it, but its Quark checkpoint keeps only the
routed and shared experts in MXFP4 and every other matrix in bf16. `Glm53Build.native_fp4` runs those experts against
MXFP8 rows on the FP4 MFMA (AITER's A4W4 weights in ATOM's addressing) instead of the MXFP4 rows above, which is why
the two forms cannot share a launch: the kernel asserts `act_fp4` and `atom_experts` apart. The bf16 matrices have no
launch path, so `weights.py` copies each one to 128 x 128 block FP8 at build, about 2.7 GiB a rank, and records it in
`copies` for the startup log. In check mode this makes the launch and SGLang's forward read different weights on the
dense layers and on the draft's entry, so `check.dense` takes SGLang's per-128 FP8 MLP on that copy as its stock side
and `NUMERICS` carries the wider `draft.eh.rel`.
