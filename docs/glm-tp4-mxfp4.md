# TP4 A8W8 and A4W4 expert paths

The GLM plugin supports the existing block-FP8 A8W8 MoE and a native
MXFP4 A4W4 MoE adapted from [FlyDSL PR #1256](https://github.com/ROCm/FlyDSL/pull/1256)
at `cd335a8d908094e2296d68d052260624bf6a8465`. A8W4 is rejected before
runtime allocation. This change is on top of the CP, long-context and
multi-token port, whose accepted kernel baseline is `bdad3b2`.

## Selecting the path

The loaded expert weights determine the activation format. The plugin builds
one quantization path per model, with the existing CP-off/CP-on indexer
variants and shared-top-k runtime. It does not requantize weights in forward.

| Loaded experts | Default activation | Explicit forward argument |
| --- | --- | --- |
| Block-FP8 weights and FP32 block scales | FP8 (A8W8) | `moe_activation_dtype="fp8"` |
| Quark MXFP4 weights and E8M0 scales | FP4 (A4W4) | `moe_activation_dtype="fp4"` |

`built.forward(..., moe_activation_dtype=None, indexer_cp=False)` follows
the loaded format. An explicit activation value must match the weights;
otherwise it raises before any device state is changed, including on a
fallback step. CP is still a Python forward argument, defaults to `False`,
and needs separate captured graphs for its two choices.

Direct layer tests construct `Glm53Build(..., expert_mxfp4=True,
moe_activation_dtype="fp4")` and `Glm53Weights(..., expert_mxfp4=True)`.
The default build remains A8W8. A4W4 is restricted to TP4 and physical widths
2, 4, 6, 8, 12 and 16. The plugin pads logical rows to these widths, including
13–15 to 16; more than 16 rows use the model fallback.

## Loaded-weight contract

The A4W4 loader consumes SGLang's existing Quark/AITER tensors in place:

- `experts.scheme` is `QuarkW4A4MXFp4MoE`, with SiLU and separated gate/up rows.
- There are 256 routed experts and one fused shared expert, with TP4 local
  intermediate width 512.
- `w13_weight` / `w2_weight` are contiguous `uint8` packed E2M1 tensors with
  shapes `[257, 1024, 3072]` / `[257, 6144, 256]`, shuffled by AITER `(16,16)`.
- `w13_weight_scale` / `w2_weight_scale` are contiguous `uint8` E8M0 tensors,
  shapes `[257, 1024, 192]` / `[257, 6144, 16]`, in AITER's shuffled scale layout.

The loader rejects ambiguous INT4 bytes, the W4A8 scheme, SiTU/GU-interleaved
storage, missing scales and unshuffled weights. It does not repack these
expert tensors. Attention and indexer projection weights still require the
existing shuffled block-FP8 format; MLA/index caches retain the existing
formats. A Quark configuration is admitted for loaded-tensor inspection,
not treated as proof that every layer is compatible. In particular, an
online configuration that converts attention to FP4 does not meet this
contract. Existing layer shape, rotary, router and indexer guards still apply.

No full checkpoint or SGLang serving run was performed for this change.
The loading evidence consists of the pinned SGLang source contract and
host tests using the required tensor layouts, plus GPU tests using AITER's
actual weight and E8M0 shuffle functions. This does not establish compatibility
with an arbitrary GLM-5.2/Quark checkpoint or serving-level quality.

## Arithmetic

Both expert inputs and the SiLU/up intermediate round to BF16 before A4
quantization. Each group of 32 values uses an E8M0 scale: clamp amax to at
least `1e-10`, divide by 6 with the PR's FP32 multiplication, and round the
scale upward to a power of two. Hardware conversion packs E2M1 values;
scaled native MFMA uses format 4 for both operands. The independent PyTorch
reference uses the E2M1 value table, ties-to-even thresholds and `frexp`
scale calculation, with no kernel quantization helper or native MFMA call.

Activation tiles retain the existing scratch/LDS stride: each 128-value
FP4 block occupies 16 data words plus 16 zero words. This port connects the
native arithmetic and SGL weight interface; it does not yet port every
native-FP4 grouped down-projection/cohort optimization from the PR. Alternative
attention/cache formats and widths above 16 also remain outside this change.

## Validation and reproduction

Run on four available gfx950 GPUs with the project's FlyDSL/AITER environment:

```bash
pytest -q test/unit/test_glm53*.py test/unit/test_contract.py -k glm53
pytest -q test/kernels/test_glm53_kernels.py -k a4w4

# Independent MoE reference within a complete layer launch:
torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_mxfp4.py --rows 16 --profile-launches
```

The reference starts at the kernel's post-attention boundary. It checks input
FP4 codes and E8M0 scales exactly, independently reconstructs up/gate and
SiLU, and checks both isolated down/reduction and the composed MoE output.
The relative-L2 gate remains `0.001`. This is independent evidence for the
changed MoE subgraph; it is not an independent attention golden.

The harness restores caches for ordinary and graph replay, checks output,
ordered top-k and cache hashes, and verifies untouched cache regions. Its
`--save-control` / `--check-control` options compare CP variants exactly;
controls must stay on the GPU server and be deleted after the comparison.
`--profile-launches` verifies one mono-kernel launch per layer outside timing.
The pytest long-context test removes its controls automatically.

A8W8 regression uses `check_multitoken.py --check-routing` with identical
fixtures under the accepted and candidate source paths. This compares outputs,
attention/MoE intermediates, ordered routes and top-k, and cache hashes.
The optional `--bench` flag measures GPU latency after the correctness checks;
[the A4W4 timing report](glm-tp4-a4w4-performance.md) defines its scope.
Serving throughput is not measured by this harness.


## Results (2026-10-20)

Validation used node 46, MI355X GPUs 4–7 in `hzm_work`, Torch
`2.9.1+rocm7.2.0.git7e1940d4`, FlyDSL 0.4.0 and AITER. SGLang host
contracts use source snapshot `9e231077722f10671075a17c9bf7ee64b5ff0715`.
No TP8 or node 50 runs were used.

The final host selection passed **179 tests**. The GPU matrix completed
**11 A4W4 cases and three A8W8 baseline/candidate pairs**. The
[small metrics artifact](benchmarks/glm-tp4-mxfp4-20261020.json) records each
case, rank-max errors, source hashes and explicit validation gaps.

| Path and workload | Result |
| --- | --- |
| A4W4, widths 2/4/6/8/12/16, two requests, context 2K | Exact input FP4 codes/scales; composed MoE relative L2 at most `3.63e-5` |
| A4W4, external top-k, widths 12/16 | Passed; composed MoE relative L2 at most `4.43e-5` |
| A4W4, zeros/signs/E2M1 midpoint ties/E8M0 scale boundaries | Exact input codes/scales and final output |
| A4W4, width 16, context 128K, three padding rows, tied routes, CP off/on | Both pass the independent MoE reference; CP outputs, intermediate values, indices and cache hashes match exactly |
| A8W8, widths 4/16 at 2K and width 16 at 128K with CP | Exact output, attention/MoE intermediates, route order, top-k and cache hashes against the accepted pre-change source |

Every A4W4 case checks ordinary and graph replay with restored caches.
Profiling confirms **one mono-kernel launch per layer per rank** for the
16-row A4W4 case, and the A8W8 16-row regression also retains one launch.
The worst A4W4 composed-output relative L2 is `4.425663792062551e-5`, with
max absolute BF16 output difference `0.03125`; the gate remains `0.001`.
These measurements establish kernel arithmetic and state behavior on the
synthetic fixtures, not A4W4-versus-A8W8 model quality or performance.
