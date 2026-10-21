# GLM TP4 synchronization with FlyDSL #1256

This port synchronizes the current SGL weight/cache contract with
[FlyDSL PR #1256](https://github.com/ROCm/FlyDSL/pull/1256) at
`032072c9406475cde47cf2017506f7c2604d5cce`. Its control is SGL commit
`ff5c64a`, which already includes A8W8, A4W4, 16-token support and optional
indexer CP. All measurements use TP4 on MI355X; they are synthetic complete
layer tests through the plugin runner.

## Changes

| Area | Adapted behavior |
| --- | --- |
| Native up/gate | Grouped and global expert unions reuse routed weights across rows/cohorts, with E8M0 staging and indexed/external task widths. |
| Native down | Grouped/union projection, ordered contribution replay, aligned vector storage, bounded mid staging, prefetch and tail scheduling. The 16-token external-index path shares down weights across both cohorts. |
| Sparse attention | One gathered KV tile supplies both eight-head groups of a 16-head TP4 shard. |
| Other projections/reductions | A4 global UV and padded O staging; paired router and owner O/FFN reductions on the twelve-row external-index path; 4K-capacity CTA placement. |
| Cache/scratch lifetime | External cohort cache writes are deferred until readers retire. New KV rows are patched using SGL request IDs, causal positions and physical slots; the final valid writer commits each slot. QKV completion can retire normalized-input readers before UV reuses the arena. |
| Physical row widths | Native builders accept 1–16. The plugin uses exact 1–12 and the 16-row cohort bucket for 13–16 logical tokens. Direct 13/14/15 builds use 13 sequential one-row tiles, two seven-row cohorts, or three sequential five-row tiles. |
| FlyDSL APIs | Coop router sum, full-vector construction and explicit system polling scope. General FP32 wave sums retain SGL's descending XOR tree. |
| Peer storage | Collective IPC unmap completion before freeing storage, propagated errors, collective retry and idempotent close. |

Input normalization and native mid quantization now occupy separate mailbox
ranges. Both use the invocation's epoch tag, so sharing a range allowed a
down consumer to accept an older BF16 payload as FP4. The regression that
exposed this first differed at down's staged codes while completed mid
values/scales were correct. The accepted fix separates the payload lifetimes.
The single-row UG loop also covers both TP4 intermediate-task rounds, and
seven-row cohort scratch reserves the pre-transformed index query.

The general coop ADD replacement was rejected by the strict baseline gate:
its ascending XOR tree changed softmax denominators and one BF16 output in
the single-request, sixteen-token external-index case. Q/K, split maxima,
attention numerators and cache bytes remained exact. General wave sums keep
their original order; the router's eight-lane coop sum matches its existing
order. This preserves the numerical contract without loosening a tolerance.

## SGL integration scope

The entry point retains block-FP8 attention, BF16 MLA cache, shuffled paged
FP8 index cache, BF16 RoPE and the existing Quark/AITER MXFP4 expert layout.
`forward(..., indexer_cp=True)` remains an optional argument, defaulting to
`False`. Expert weights select A8W8 or A4W4; an explicit activation argument
must match them. No A8W4 or TP8 validation belongs to this port.

The source is adapted to SGL's physical-slot selector, Hadamard/FP8 index
arithmetic and 64-bit cache addresses. SGL's live-bound score loop already
keeps its query in LDS across all score tiles assigned to one CTA, covering
the upstream query-reuse mechanism. Upstream paired split-key tasks are
TP8-specific. Alternate attention/cache formats, external-index DCP, alternate
checkpoint packers and widths above 16 have no new SGL entry point here.

## Validation method

The independent MoE reference starts at the observed post-attention boundary.
It checks activation FP4 codes/E8M0 scales exactly and applies relative
L2 <= `0.001` to up/gate, down/reduction and composed MoE output. It does not
constitute an independent attention reference.

Causal per-token controls compare the complete layer with batched execution,
including attention/MoE intermediates, ordered routes/top-k, cache hashes,
untouched cache bytes, graph replay and future-query/request isolation.
Numerical tensors retain the `0.001` gate. Baseline A4 controls require bitwise
equality; A8 controls record exactness in addition to their unchanged error
gate. No tolerance is relaxed by this synchronization.

Performance pairs use identical inputs and source-isolated baseline/candidate
processes. Seeds 821 and 823 reverse execution order. HIP graphs contain
32 layer calls and each timed trial replays ten graphs; the reported latency
uses the maximum event duration across the four ranks. Compilation, weight
packing and setup are excluded. This is a repeated single-layer working set,
not serving throughput, request latency or a rotating full-model weight set.

## Regression results (2026-10-21)

The frozen candidate completed **156/156 TP4 GPU cases**: one original-failure
reproducer, ten serial/batched controls, 38 edge cases, 71 native-width / owner /
indexed timing cases, and 36 external-index / long-context timing cases.
Every expected case name was matched against the executed manifest. Four
additional router-probe cases also passed; their source was not adopted.

Coverage includes independent MoE checks for all physical widths 1–16, exact
ties and disjoint/overlapping routes, quantization boundaries, padding, 4K
placement, cross-cohort causal visibility, cache preservation/replay, distinct
TP O-projection partials, 128K indexer CP and a single mono-kernel launch.

| Numerical check | Worst relative L2 across ranks/cases | Result |
| --- | ---: | --- |
| Independent up/gate | 1.39e-7 | Pass, gate 0.001 |
| Independent down/reduction and composed MoE | 5.65e-5 | Pass, gate 0.001 |
| Serial vs batched output | 1.87e-5 | Pass, gate 0.001 |
| Baseline vs candidate controls | 0 | All 60 controls bitwise exact |

Activation FP4 codes and E8M0 scales match the independent reference exactly.
Serial/batched output differences reach 0.015625 absolute after BF16 rounding;
their checked attention, routing-probability and mid tensors are exact. This
serial comparison has a different contract from bitwise baseline equivalence.

CPU tests: **297 passed, 2 failed**. Both failures are DeepSeek-only SGLang
source contracts (`test_contract.py`, `deepseek_v41/draft.py` and
`deepseek_v41/spec.py`); the same failures reproduce on `ff5c64a` with the
installed SGLang source. The separately committed peer-lifetime fix passed
six host tests and the TP4 multi-buffer lifecycle/error-retry check.

The following tables compare `ff5c64a` with the frozen candidate. Values are
microseconds per complete layer, averaged from the two seeds’ run medians.
Positive latency reduction means faster; small changes are retained as measured.
All 2K tables use **conc=2, seq=2/4/6/8**, or 4/8/12/16 total query tokens.
The baseline already supports all four shapes.

### Indexed layer, CP off

| Seq | A8W8 baseline | A8W8 candidate | Reduction | A4W4 baseline | A4W4 candidate | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 149.12 | 148.75 | +0.25% | 116.82 | 119.61 | -2.39% |
| 4 | 254.61 | 254.13 | +0.19% | 214.75 | 203.90 | +5.06% |
| 6 | 353.04 | 353.74 | -0.20% | 323.16 | 254.45 | +21.26% |
| 8 | 404.05 | 404.42 | -0.09% | 344.83 | 283.77 | +17.71% |

### Indexed layer, CP on

| Seq | A8W8 baseline | A8W8 candidate | Reduction | A4W4 baseline | A4W4 candidate | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 146.02 | 145.32 | +0.48% | 114.28 | 115.07 | -0.69% |
| 4 | 256.97 | 256.46 | +0.20% | 216.02 | 209.92 | +2.82% |
| 6 | 344.61 | 344.18 | +0.13% | 318.96 | 252.39 | +20.87% |
| 8 | 399.64 | 399.00 | +0.16% | 336.03 | 279.30 | +16.88% |

### External-index layer (selection supplied)

| Seq | A8W8 baseline | A8W8 candidate | Reduction | A4W4 baseline | A4W4 candidate | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 107.12 | 106.66 | +0.43% | 80.08 | 81.32 | -1.54% |
| 4 | 219.72 | 219.60 | +0.05% | 178.17 | 171.57 | +3.70% |
| 6 | 314.92 | 315.12 | -0.06% | 281.92 | 227.51 | +19.30% |
| 8 | 355.37 | 355.47 | -0.03% | 294.14 | 211.07 | +28.24% |

### Long-context confirmation

The initial A4W4 eight-row 128K CP pair was 1.59% slower. Two additional
seeds reversed that result (+1.16% and +0.46% latency reduction). Across all
three seeds it is effectively unchanged:

| Format / rows | Seeds | Baseline (µs) | Candidate (µs) | Reduction |
| --- | --- | ---: | ---: | ---: |
| A4W4 / 8 | 821, 823, 827 | 332.58 | 332.52 | +0.02% |
| A4W4 / 16 | 821 | 477.46 | 419.69 | +12.10% |
| A8W8 / 16 | 821 | 543.44 | 545.29 | -0.34% |

The sixteen-row long-context results are single-seed observations. The small
A4W4 four-row regression at 2K is repeatable: -2.94% / -1.83% indexed CP off,
-0.85% / -0.53% CP on, and -1.41% / -1.67% with external indices. It is not
removed from the tables. Reverting only the eight-lane router coop sum to its
original DPP implementation gave -0.32% and +0.43% latency reduction versus
the candidate. The direction changed with the seed/order, so this probe was
not adopted. The small-batch regression's cause remains unresolved.

Detailed case arguments, source hashes, per-seed trials, rank-max errors and
environment versions are in the [metrics JSON](benchmarks/glm-tp4-flydsl1256-sync-20261021.json).
The environment used Torch 2.9.1 / ROCm 7.2 and the actually imported FlyDSL
0.4.0 source. Its installed package metadata still says 0.3.1; that metadata
does not identify the loaded implementation.

No full-checkpoint, multi-layer quality, MTP acceptance or SGLang serving
measurement is claimed by these layer tests.

## Reproduction

Use the project ROCm/FlyDSL/AITER environment and four available gfx950 GPUs.
Keep control tensors on the GPU server.

```bash
# Independent MoE arithmetic inside the complete layer:
torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_mxfp4.py --rows 16 --external --profile-launches

# Complete layer, causal serial/batched equivalence:
torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_multitoken.py --activation fp4 \
  --requests 2 --query-tokens 8 --capacity 2048 --check-routing \
  --schedule serial --save-control /remote/controls/serial

torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_multitoken.py --activation fp4 \
  --requests 2 --query-tokens 8 --capacity 2048 --check-routing \
  --check-control /remote/controls/serial --check-causality
```

For version pairs, select the implementation using `PYTHONPATH` while running
the candidate harness, save a control on the first source and check it on the
second. Add `--bench` to both and use `--capacity 2048` for the 2K tables.
For A4, set `check_mxfp4.py --rows` to twice the per-request `seq`,
with `--cp` or `--external` as appropriate. For A8, use `check_multitoken.py`
with `--requests 2 --query-tokens seq --check-routing` and `--indexer-cp`;
external A8 layers reuse an indexed control via `--external-control`.
`check_owner_reduction.py --save-control /remote/controls/owner` and its
`--check-control` counterpart exercise twelve-row owner reductions with
different O-projection partials on all four ranks, in addition to the
fixture's already-distinct expert shards.
The checked-in pytest wrappers cover exact native widths and tiled/serial
controls. Delete saved controls when the comparison is complete.
