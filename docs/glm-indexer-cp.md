# GLM indexer context parallelism

This opt-in path adapts indexer context partitioning, distributed radix selection
and query projection sharding from
[ROCm/FlyDSL#1256](https://github.com/ROCm/FlyDSL/pull/1256), pinned to
`cd335a8d908094e2296d68d052260624bf6a8465`, to this plugin's GLM-5.3 FP8
layer. It preserves SGLang's paged FP8 index cache and shuffled FP8 weights.

Pass `indexer_cp=True` to the mono-kernel forward on **every TP rank**.
The keyword defaults to `False` and accepts only a Python bool. CP remains
part of the kernel's compilation key; the plugin builds and precompiles both
indexer variants before any forward or graph capture. The earlier
`SGLANG_AMD_MONO_GLM_INDEXER_CP` environment variable is no longer read.
The [multi-token follow-up](glm-tp4-multitoken.md) extends TP4 to 16 total
query tokens. TP8 retains its eight-row range and existing width buckets;
one-row requests are padded to width 2.

```python
# built is the object returned by the GLM-5.3 spec's build().
built.forward(plan, input_ids, positions, forward_batch, indexer_cp=True)
```

The direct layer API accepts the same keyword. Prepare matching runtimes
collectively on every rank, then use the same layer object for either mode:

```python
op = Glm53LayerOp(base_runtime, weights, layer, cp_runtime=cp_runtime)
out = op.forward(h, caches, cos, sin, step, indexer_cp=True)
```

The direct caller advances the selected runtime once after all layer launches
in a step. The plugin's full forward handles this, including the shared
reuse-only runtime. Unselected indexer runtimes keep their epoch. Weight
tensors and caches are shared; workspaces, peer buffers and epochs are separate.
Preparing both variants adds startup compilation and runtime memory, even
when calls use only the default mode.

This is a **capture-time host choice**. A graph replays the variant captured
inside it; changing a Python argument cannot change an already captured graph.
Callers supporting both policies must retain distinct graph instances (include
CP in their own graph-cache key). SGLang's current generic graph key contains
no CP field and its ordinary server call sites omit the new keyword, so those
calls retain the default. This plugin change does not add a serving CLI switch
or automatic per-request graph selection.

## Computation and ownership

Each rank owns a contiguous shard of the live context's 64-key tiles.
Ownership uses the current request length, independently for each row;
it does not shard the allocated maximum context. Empty shards do no score
work. Every owner computes the original complete 32-head score.

The compiled geometry and current live length select the following paths:

| Condition | Path |
| --- | --- |
| Compiled context at most 16,384 | Also shard the 256 index-Q projection tiles across TP ranks and exchange the packed BF16 query before the existing RoPE/Hadamard scoring path |
| TP8, width 2, compiled context in (16,384, 131,072] | Two independent CP4 groups, each processing the same requests; attention and MoE still reduce over TP8 |
| Live row at most 114,688 keys | Exchange full scores within the CP group and retain the existing local selector |
| Live row above 114,688 keys | Keep scores in local shards; use 4K-key histogram/count/emit tasks and exchange summaries and the final top-k |

Width two includes SGLang's padded single-row bucket and two active requests.
Group membership and Q sharding are compile-time choices. The long/short
selector choice reads device metadata and can change during replay of the
same graph, independently for each row. Padding publishes the existing
readiness marker without reading score payloads.

Long selection runs four 256-bin radix rounds. The coordinator selects the
threshold, ranks compute counts and offsets, and disjoint owners publish the
selected logical IDs. Each rank converts them through its page table to the
compact physical-slot output. Additional counts retain the SGL selector's
12/24-bit candidate-bin output order: high-bin winners, remaining strict
winners, then the earliest threshold ties. The ordering is not a score sort.

The upstream BF16 selector changes paths at 16K because it stages its local
scores in LDS. SGLang's existing paged selector already streams the score
vector. On the synthetic FP8 fixture, full-score exchange won at 64K and 96K,
while distributed selection won at 128K and 1M. The adapted threshold is 112K;
it is a tuning choice for this implementation, not a cache-format limit.

The following contracts are retained:

- AITER-shuffled block-128 FP8 projection and expert weights.
- Replicated BF16 MLA KV and paged FP8 index caches, including power-of-two
  scales, 16-by-16 shuffling, RoPE and Hadamard transforms.
- Top-k physical slot IDs at the compact indptr offsets, with existing
  ordering and tie behavior.
- One kernel launch per covered layer and existing guarded TP reductions.

Inputs, indexer weights, request metadata and cache contents must be identical
across TP ranks, as in the existing plugin path. This is score parallelism
over replicated caches; it does not distribute cache storage.

Query, score, histogram, threshold, count, offset and final-index peer
arenas each have two launch slots. Each value and its launch epoch use the
existing system-scope tagged-pair mailbox protocol. All new waits use the
existing deadline guard. Guard flags follow every payload arena.

Before a rank can reuse slot N for launch N+2, it must finish launch N+1's TP
reductions. Those require every peer to have completed launch N's selection,
so no selector can still be reading slot N. This relies on ordered launches
and replicated TP forward; concurrent use of one runtime is not supported.
Local histogram and score scratch has no cross-rank readers.

For long compiled contexts, local score scratch is
`rows * ceil(ceil(max_seq / 64) / cp_ranks) * 64 * 8` bytes. Full-score peer
slots are capped at `rows * 114688 * 8` bytes each; histogram and compaction
arenas add bounded per-partition storage. This reduces the previous CP path's
full-vector traffic and peer allocation for long contexts. The replicated
index and MLA caches are unchanged. CP remains off by default.

## Tests

The synthetic harness calls the real plugin runner with nonzero weights,
shuffled physical pages, a strided page table, unequal live lengths and a
padded row. It compares CP off/on projected queries, score values, ordered physical top-k,
full-layer outputs and cache bytes exactly. It checks all-rank output
agreement, untouched cache regions, poisoned workspaces, changing lengths
and page writes under graph replay, a delayed rank, and three launches per step to exercise
slot parity across step boundaries. It calls `forward(indexer_cp=...)` and
also replays the CP graph twice followed by the non-CP graph once, checking
that only the selected runtime consumes an epoch.

The independent Torch score oracle starts from the unchanged upstream stage's
queries and head weights. It checks score arithmetic, including query RoPE
and Hadamard, with relative L2 error below 0.01. This is not an independent
golden for the entire layer or a checkpoint quality test. A separate stable
Torch sort checks exact selected-set membership and earliest ties. Long cases
reconstruct scores from each CP group and replay lengths on both sides of the
selector boundary. Full-context cases exercise simultaneous long requests.

```bash
pytest -q test/unit/test_glm53_index_plan.py test/unit/test_source_digest.py
pytest -q test/unit/test_glm53_forward.py test/unit/test_glm53_forward_spec.py
pytest -q test/kernels/test_glm53_kernels.py

# A single paired synthetic run; all selected GPUs must be available.
torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_indexer_cp.py --bench-iters 100
```

Timing uses alternating CP off/on trials, device events, three layer launches
and the runtime's epoch advance per graph, and the maximum elapsed time across
ranks. The reported per-layer time divides graph time by three. All three launches share
one synthetic weight allocation and cache per variant. This measures a
layer/protocol fixture, not serving throughput or a model's weight working set.

Full serving validation additionally needs a compatible SGLang installation,
the GLM-5.3 FP8 checkpoint, and a caller that passes the policy when capturing
each graph. The existing `TestMonokernelCheck` and `TestMonokernelDecode` in
`test/e2e/test_glm53_server.py` provide the reference and quality checks; their
unmodified server call sites currently exercise `indexer_cp=False`.
`SGLANG_AMD_MONO_CHECK` is for correctness and must be off during timing.

## Remaining port scope and rejected probe

The [multi-token follow-up](glm-tp4-multitoken.md) adapts the row/cohort and
grouped-expert scheduling to SGLang's FP8 path through 16 total query tokens.
The [A4W4 follow-up](glm-tp4-mxfp4.md) adds native FP4 arithmetic and an
in-place Quark MXFP4 expert loader under a separate weight contract. Full
checkpoint and serving compatibility remain unverified.

The upstream TP8/width-8 sparse-attention tile32 change was tested separately
against tile64. CP off/on remained exact within the tile32 candidate, but the
three-launch output differed from tile64 by relative L2 `0.005527`, exceeding
the probe's preselected `0.001` gate, and the final ordered indices changed.
Tile64 is retained. This does not establish a serving-quality failure: the
change needs an independent attention/layer reference to assess its different
reduction and rounding behavior. No tile32 performance result is promoted.
The harness's `--save-control` / `--check-control` options preserve small
output/intermediate controls on the test server for such source comparisons;
they do not relax the bitwise CP equivalence checks.

Full checkpoint equivalence, SGLang serving quality and serving performance
remain separate validation gates. `test/unit/test_glm53_spec.py` cannot collect
in the isolated test environment because its SGLang MoE imports require
`transformers`. The SGLang source-contract checks do run there.

## Validation environment (2026-10-20)

The results below describe the initial CP port, commit `1e795ce`. The
[TP4 long-context follow-up](glm-tp4-long-context.md) covers subsequent
cache-addressing fixes, tuning and validation.

Tests use node 47, MI355X, Torch `2.9.1+rocm7.2.0.git7e1940d4`, ROCm
`7.2.26015-fc0010cf6a`, FlyDSL `0.4.0` and AITER. FlyDSL `0.3.2` lacks the
`generic_store` API already required by the unchanged baseline guard.
Dependencies are isolated from existing shared installations. SGLang contract
checks use source at `9e231077722f10671075a17c9bf7ee64b5ff0715`.

The initial CP source passed **108** partition/layout, forward-dispatch,
forward-spec and source-key tests, plus **6** GLM-specific SGLang source-contract
checks. The expanded GPU suite passed **14 cases** in 517.79 seconds, covering
TP4 and TP8, empty shards, exact ties at short and long context, mixed rows,
all-active long requests, 1M at width two, delayed rank zero, poisoned
workspaces and replay across 114,688/114,689 keys. CP off/on projected queries,
complete scores, ordered physical indices, full output and cache bytes were
bitwise equal. The reported independent score relative L2 errors were below
`1e-7`; all ranks passed the unchanged `0.01` bound.

Initial CP paired synthetic timings in microseconds (median of four
max-rank trials; each trial alternates the two modes):

| TP | Width (active) | Allocated context | CP off | CP on | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 4 (3) | 4,096 | 142.14 | 137.51 | 1.034x |
| 8 | 8 (7) | 4,096 | 174.81 | 162.70 | 1.074x |
| 4 | 2 (1) | 114,752 | 189.27 | 173.92 | 1.088x |
| 4 | 2 (1) | 131,072 | 202.61 | 172.39 | 1.175x |
| 8 | 2 (1) | 131,072 | 182.68 | 159.13 | 1.148x |
| 8 | 2 (2), both long | 131,072 | 200.47 | 173.32 | 1.157x |
| 4 | 2 (1) | 1,048,576 | 1,025.29 | 274.66 | 3.733x |

TP4 and TP8/4K confirmation runs use 200 graph replays per trial; TP8/128K
runs use 100. The one-active-row cases start at capacity minus three keys.
The two-long-row case starts at 131,069 and 131,068; 4K cases have unequal
request lengths recorded by the harness. Compare modes within a row; these
are not interchangeable batches. They share one synthetic weight allocation
across three layer calls, and do not measure serving throughput.

Two additional controls support the adapted policy. With the same final
fixture at 114,749 live keys, the previous full-score CP path took 178.69 us,
versus 173.92 us for distributed selection. In separate matched TP8/128K
runs, forcing CP8 took 183.64 us for one active request and 219.49 us for two
long requests; the CP4 groups took 159.13 and 173.32 us respectively. The CP8
control changes only the group-planning helper. These are synthetic tuning
evidence, not a universal optimum for all serving batches.

Reproduce the long-context confirmation or simultaneous long requests with:

```bash
torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_indexer_cp.py \
  --rows 2 --capacity 1048576 --bench-iters 200

torchrun --standalone --nproc-per-node=8 \
  test/kernels/glm53/check_indexer_cp.py \
  --rows 2 --capacity 131072 --pad-rows 0 --full-context --bench-iters 100
```
