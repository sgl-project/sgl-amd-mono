# GLM TP4 multi-token port

This follow-up adapts the multi-token mechanisms already implemented in
[ROCm/FlyDSL#1256](https://github.com/ROCm/FlyDSL/pull/1256), pinned to
`cd335a8d908094e2296d68d052260624bf6a8465`, to SGLang's shuffled block-128 FP8
GLM-5.3 layer and paged cache. It extends TP4 to **16 total query tokens**:
two requests with 4, 6 or 8 query tokens run at widths 8, 12 or 16.

The first CP commit is `1e795ce`; `0524d9a` adds the cache-address and
long-context I/O changes. This port builds on both. CP remains the Python
`forward(..., indexer_cp=True)` argument, defaulting to `False`, with the
same choice required on all TP ranks and fixed by graph capture.

## Adapted mechanisms

| Mechanism from the reference PR | FP8 integration |
| --- | --- |
| Bounded row tiles and resident CTA cohorts | Width 16 uses two eight-row tiles and 128 CTAs per tile, in one layer launch. |
| Front-stage completion and cache visibility | All 256 CTAs join after cache/query publication, before wide queries read another row's cache. |
| Wider input projection tiles | TP4 reuses projection weights over eight or six query columns. |
| Shared input normalization and K-sliced staging | Twelve- and sixteen-row block-FP8 launches normalize each input once and publish tagged BF16 values; projection CTAs stage bounded K slices. |
| Global input projection | Reuse-only sixteen-row layers project all sixteen inputs with the same weight loads. |
| Global Q_B projection | Sixteen-row launches reuse weights across both tiles, including non-sharded index-Q work. |
| UV latent reuse | Merge the split-attention latent once per head for all four output tiles; reuse-only sixteen-row layers also share UV weights across cohorts and pad the O-projection LDS rows. |
| Index-query reuse | Score CTAs retain transformed queries across their key tiles. Sixteen-row cohorts pre-transform each query once when index-Q is unsharded, preserving SGL's BF16, RoPE and Hadamard order. |
| Expert union and token masks | Reuse shuffled FP8 expert weights across matching rows, retaining every row's route order and shared expert. Width 16 gathers the union across both tiles. |
| Paired up/gate reductions | Preserve the original wave accumulation order while sharing synchronization between two output tiles. |
| Wide down-projection prefetch | Use the reference FP8 prefetch depth of four. |

These are adaptations of the reference schedule to the existing FP8 loads
and arithmetic. The PR's native FP4 instructions, E8M0 scale packing,
MXFP4 checkpoint loader, native-FP4 grouped down-projection, alternative
attention/cache formats and widths above 16 are outside this FP8 contract.
The TP8-only paired split-attention path is also outside this TP4 change.
The port does not remove their
format guards or claim full GLM-5.2 checkpoint compatibility.

## Request and runtime contract

The plugin selects widths 2, 4, 6, 8, 12 and 16 for TP4. Logical rows 13–15
pad to 16; wider steps fall back to the model's forward. Direct runtimes
require physical widths 1–12 or 16, with 16 restricted to TP4. Wide TP4
launches require the full 256-CU MI355X because all persistent CTAs must
be resident. TP8 keeps its existing widths 2, 4 and 8.

For linear speculative chains (`speculative_eagle_topk=1`), each query's
causal extent is its absolute position plus one. Request-level and expanded
per-query DSA page tables retain their request ownership. Slot zero and
width padding write no cache and attend no keys. Tree speculation is refused.
The forward also copies SGLang's requested auxiliary layer inputs before
reusing output storage and passes the packed captures to the logits processor.

These host contracts have focused CPU tests. Synthetic layer tests do not
establish checkpoint accuracy, acceptance rate or serving throughput; those
still require the existing SGLang reference and end-to-end checkpoint tests.

For the direct upstream-versus-adapted comparison with one layer call on
both sides at context capacity 2K, see the [2K comparison](glm-tp4-ctx2k.md).

## Baselines and reproduction

Use the unmodified repository at `0bce188` for the original implementation.
Its plugin supports decode through four TP4 rows. Directly forcing multiple
tokens of one request into its wider kernels exposed first-run versus replay
differences in the ordered indices and attention output. Those failing runs
are diagnostics and are excluded from valid performance comparisons.

The causal baseline runs the original two-request decode once per query
position: 4, 6 or 8 layer calls per query block. The new path runs one layer
call for the same 8, 12 or 16 tokens. Report these call counts with latency;
the resulting speedup includes batching and eliminated launches.

The harness selects an unchanged implementation through `PYTHONPATH`; the
test script and fixture come from this port. Keep control tensors on the
GPU server and remove them after recording results. Complete caches are
hashed in bounded host chunks, rather than saved to disk.

```bash
# Four available MI355X GPUs; run sources serially on the same devices.
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="$DEPS:$AITER:$ORIGIN" \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$CANDIDATE/test/kernels/glm53/check_multitoken.py" \
  --query-tokens 8 --capacity 131072 --schedule serial \
  --save-control "$REMOTE_RESULTS/origin-c2-q8" --check-causality --bench

HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="$DEPS:$AITER:$CANDIDATE" \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$CANDIDATE/test/kernels/glm53/check_multitoken.py" \
  --query-tokens 8 --capacity 131072 --indexer-cp \
  --check-control "$REMOTE_RESULTS/origin-c2-q8" --check-causality --bench
```

Repeat with query counts 4 and 6, with CP disabled, and at capacity 4096.
The external-index path can use `--external-control` to consume the indexed
control's physical top-k slots. `--check-routing` also compares ordered expert
slots, probabilities and FP32 up/gate intermediates; `--route-ties` exercises
identical expert routes. `--profile-launches` checks actual device launch counts
outside the timing loop. The CPU and GPU regression entry points are
`test/unit/test_glm53*.py` and `test/kernels/test_glm53_kernels.py`.

Cross-source gates require exact input, ordered indices, full cache hashes,
rank agreement and replay stability. Output and intermediate relative L2
must be at most 0.001; bitwise equality is recorded separately. Causality
checks change the final token of request 0 and require every earlier token
and request 1 to remain unchanged. Timing uses GPU events, 32 steps per
graph, ten replays and the maximum rank duration in each trial. Timing includes
the existing epoch-update kernel. Compilation, fixture creation, snapshots and
correctness checks are outside timing.

## Validation results

Measured on node 46, MI355X devices 4–7, TP4, Torch 2.9.1/ROCm 7.2 and
FlyDSL 0.4.0. The SGLang host contract was checked against
`9e231077722f10671075a17c9bf7ee64b5ff0715`. No TP8 GPU tests were run.
The final confirmation ran CP-on and CP-off candidates before the original
serial control on the same devices, reversing the earlier source order.

| Context | Query tokens/request | Total rows | Origin serial (µs) | CP off (µs) | CP on (µs) | CP-on speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 4 | 8 | 399.85 | 259.81 | 255.00 | 1.57× |
| 4K | 6 | 12 | 598.56 | 360.33 | 345.18 | 1.73× |
| 4K | 8 | 16 | 800.56 | 411.19 | 401.81 | 1.99× |
| 128K | 4 | 8 | 908.73 | 471.23 | 395.25 | 2.30× |
| 128K | 6 | 12 | 1366.61 | 673.10 | 543.36 | 2.52× |
| 128K | 8 | 16 | 1824.66 | 725.50 | 543.56 | 3.36× |

Each value is the median of four trials after taking the maximum GPU-event
duration across the four ranks. These are whole query-block latencies for one
synthetic layer. Origin uses 4/6/8 layer calls; the candidate uses one.
The speedup includes batching and eliminated launches, and is not a serving
throughput measurement or an ablation of any single optimization.

The reuse-only sixteen-row layer, supplied with the indexed control's physical
top-k slots, measured **354.61 µs** versus **585.81 µs** for eight original
serial calls (**1.65×**). It also passed explicit routing and intermediate checks.

Final validation:

- All 27 confirmation processes passed: 18 indexed source comparisons, one
  launch-count check, four CP protocol cases, two routing cases and two
  external-index source comparisons.
- All compared outputs and intermediates were bitwise equal on every rank
  (relative L2 = 0); ordered indices, ordered expert slots and full cache hashes
  matched. Graph replay and future-token/request isolation passed.
- CP protocol cases covered twelve rows with two padding rows, sixteen rows
  with three padding rows, tied index scores and full 128K contexts. Poisoned
  workspaces, changed cache generations and delayed peers passed. The largest
  index-score oracle relative error was `1.52e-7`.
- Profiler observed one `glm5_monokernel` launch per rank for sixteen rows,
  plus the existing epoch-update kernel. Profiling was outside timed trials.
- 142 focused host/unit tests and 12 GLM/GLMFlash SGLang contract checks passed.
  The imported AITER/SGLang modules require a visible target GPU even for the
  host checks. These counts do not cover the complete repository test suite.

[Machine-readable measurements](benchmarks/glm-tp4-multitoken-20261020.json)
include source fingerprints, all maximum-rank timing trials, and individual
validation outcomes. Full validation tensors were kept remotely and removed
after recording the results. Checkpoint accuracy, speculative acceptance rate
and end-to-end serving tests remain unmeasured.

## Probe observations

The original forced multi-query paths failed replay as described above, so
only the original causal serial schedule appears in the valid timing table.
A naive shared-normalization port using full-row mailbox reads regressed the
sixteen-row path; the retained implementation uses the PR's K-sliced staging.
The initial K-sliced port had a misplaced Python helper definition; moving it
outside the old dispatch branch resolved the tracing failure before validation.

An early UV-reuse candidate (`kernel.py` SHA256
`55240f15171f667aa6de827277b3e92a4da1e43239f365b84d1c8ca069940ae4`)
reported one HIP illegal-address error during a 4K/eight-row/CP-on benchmark,
after its initial/replay checks passed. The identical retry and six diagnostic
runs using seeds 719/720/721 passed. Its cause remains unconfirmed; the original
failure log was overwritten by the retry, while the traceback and replay
observation were retained in the task record. Final-source confirmation passed
all listed cases; it does not establish the cause of that earlier failure.
