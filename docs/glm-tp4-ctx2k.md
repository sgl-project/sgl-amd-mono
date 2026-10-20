# TP4 multi-sequence comparison at 2K

This comparison uses the upstream SGL mono-kernel at
`0bce1886519813a8086488e599dfa155a69e74a1` (the GitHub HEAD checked on
2026-10-20) and the adapted kernel at
`bdad3b2fb1743e79bc66591a7d12798e0df99f1e`.

Every timed query block makes **one indexed layer call per rank**. TP is 4,
concurrency is 2 requests, and each request contributes 2/4/6/8 query tokens.
The harness calls `Glm53Runtime` directly. Unsupported or failing upstream
shapes have blank latency cells; there is no substitution of serial timings.

## Results

Node 46, MI355X devices 4–7, Torch 2.9.1/ROCm 7.2, FlyDSL 0.4.0, seed 719.
Latency is in microseconds per layer call for the whole query block.

| Tokens/request | Total rows | Upstream (µs) | Adapted CP off (µs) | Adapted CP on (µs) |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 147.25 | 151.86 | 146.46 |
| 4 | 8 | 269.66 | 255.86 | 258.66 |
| 6 | 12 |  | 353.70 | 345.26 |
| 8 | 16 |  | 413.51 | 410.57 |

- At two query tokens per request, CP off is 3.1% slower than upstream;
  CP on is effectively unchanged (0.5% lower measured latency).
- At four query tokens per request, latency falls by 5.1% with CP off and
  4.1% with CP on. At this short context, CP provides no consistent extra win.
- Upstream twelve-row execution compiles but fails the first-call versus
  replay check. On the reported rank, 1,875 ordered index entries and four
  output elements changed; maximum output change was `0.00048828125`.
  Its timing gate was not reached, so the six-token cell is blank.
- Upstream's builder restricts physical rows to `1 <= S <= 12`.
  Two requests of eight tokens need sixteen rows, so that cell is blank.

All filled timing cells passed replay, causal/request isolation, cache checks
and comparison with the causal serial numerical reference. The adapted output
and checked intermediates were bitwise equal to that reference. Serial runs
were used only for numerical validation and were not timed. The sixteen-row
CP-on profiler check observed one layer kernel plus the existing epoch update.

The upstream **forward** spec remains decode-only with a TP4 cap of four rows.
The raw eight-row kernel timing above does not imply that upstream's forward
selects it for multi-token steps. This is a synthetic layer comparison, not a
checkpoint or serving benchmark.

## Context and timing definition

`ctx=2K` means a compiled maximum context capacity of 2048, consistent with
the preceding 4K/128K tests. Query causal lengths stay below that capacity:
for request `r` (0 or 1), token `i`, and request width `q`, the fixture uses
`2048 - q - 2 - r + i` live keys. The complete per-query lengths are recorded
in the result file. This is not 2048 historical tokens followed by extra queries.

Each graph contains 32 repeated steps. A trial measures ten graph replays
using GPU events and takes the maximum duration across ranks. Compilation,
weight/fixture preparation and correctness checks are outside timing; the
existing epoch update is included. Each run contains four timed trials.
The two- and four-token cases were repeated with the source order reversed;
the table reports the median of all eight trials for those cases, and four
trials for the six- and eight-token adapted cases.

[Machine-readable results](benchmarks/glm-tp4-ctx2k-20261020.json) contain all
maximum-rank trials, source fingerprints, validation outcomes, unsupported
shapes and the retained replay-failure observation.

## Reproduction

Use the same checked-in harness/fixture for both unchanged source checkouts.
Controls and logs belong on the GPU server. For each `Q` in `2 4 6 8`:

```bash
# Numerical oracle only; deliberately no --bench.
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="$DEPS:$AITER:$UPSTREAM" \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$CURRENT/test/kernels/glm53/check_multitoken.py" \
  --requests 2 --query-tokens "$Q" --capacity 2048 --schedule serial \
  --save-control "$REMOTE_RESULTS/oracle-q$Q" --check-causality

# One layer call. Use $UPSTREAM for Q=2/4/6, or $CURRENT for all four widths.
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="$DEPS:$AITER:$IMPLEMENTATION" \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$CURRENT/test/kernels/glm53/check_multitoken.py" \
  --requests 2 --query-tokens "$Q" --capacity 2048 --schedule batched \
  --check-control "$REMOTE_RESULTS/oracle-q$Q" --check-causality --bench
```

For the adapted CP-on column, add `--indexer-cp`. Do not change the upstream
width guard or replace an unsupported call with several smaller calls.
Remove the remote oracle tensors after recording the results.
