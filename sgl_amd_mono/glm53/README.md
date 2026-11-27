# GLM-5.3 monokernels on gfx950

`sgl_amd_mono/glm53/` runs GLM-5.3's decode and MTP verify steps of up to 8 rows on MI355X (gfx950), at TP 4 or 8
(and of up to 4 rows at TP 2, see [Widths by TP size](#widths-by-tp-size)), through SGLang's monokernel interface
(RFC sgl-project/sglang#43949). Each of the 78 decoder layers is one launch, and the final norm, LM head and logits
all-gather are one more (the tail). The NextN draft's steps run on the same launches. SGLang's forward runs prefill,
extend and the steps wider than the TP size's widths.

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
`test/kernels/glm53/` (8 GPUs, random weights; TP 2, 4 and 8), and `test/e2e/test_glm53_server.py` (the server at TP 8
on the checkpoint `GLM_53_MODEL_PATH` names). `test/e2e/glm53_server.py`'s `TP2_ARGS` is the CI's TP 2 server
(`scripts/ci/models.py`, `glm53_mxfp4_tp2`).

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

## Widths by TP size

A step of S rows runs at the smallest built width of at least S rows (`spec.TP_WIDTHS`, `tp_widths`); a step wider
than the largest runs SGLang's forward. SGLang routes steps of 1 to 8 rows at every TP size (the spec's
`DeviceLimits` widths are not per TP size), so its startup lines list all of them; the build logs the widths it built
and the routed steps that fall back.

| TP | heads a rank | widths | decode, no MTP | MTP (3 steps, 4 draft tokens) |
| --- | --- | --- | --- | --- |
| 8 | 8 | 2, 4, 8 | concurrency 1 to 8 | 1 or 2 requests (4 or 8 verify rows) |
| 4 | 16 | 2, 4, 8 | concurrency 1 to 8 | 1 or 2 requests |
| 2 | 32 | 4 | concurrency 1 to 4 | 1 request (4 verify rows) |

TP 2 serves only the MXFP4 checkpoint (`amd/GLM-5.3-Quark-MXFP4-AttnFP8`), the only one that fits on two GPUs, and
builds width 4 only:

- **No width 8.** With 1024 expert rows a rank, an 8-row MoE's mid and down products outgrow the launch's 128 KiB of
  LDS.
- **No width 2.** The indexer launch faults below 4 rows at TP 2, so steps of 1 to 3 rows run at width 4 (padded).

With MTP, a verify step has (draft tokens) x (requests) rows, so at TP 2 the kernels take 1 request with 4 draft
tokens, or up to 2 with 2. The NextN draft builds its extend launch only at width 4 too: a draft extend of another
row count (1 request with 2 draft tokens; any with 3) runs SGLang's forward, and with 3 draft tokens the draft builds
no extend launch at all. Its decode steps run at width 4.

Long prompts at TP 2 also need two prefill fixes outside this package: one 131072-token chunk (the e2e recipe's
`--chunked-prefill-size`) passes 2^31 elements in two Triton kernels' int32 offsets above 116,508 tokens at 32 heads a
rank. sgl-project/sglang#50870 (merged) fixes the sparse MLA kernels; ROCm/aiter#7952 fixes AITER's
`fused_qk_rope_cat_and_cache_mla`, which faults without it.
