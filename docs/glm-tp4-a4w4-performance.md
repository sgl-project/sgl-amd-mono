# TP4 A4W4 layer performance at 2K

This measures the A4W4 path at kernel commit `a645c98`, using two requests
and 2/4/6/8 query tokens per request. Each query block makes one complete
indexed layer call per TP rank. Attention remains block-FP8; MoE weights and
activations use MXFP4/E2M1. This is a synthetic layer benchmark, not serving
throughput or a full-checkpoint measurement.

## Measurement

Node 46, MI355X devices 4–7, Podman `hzm_work`, TP4, Torch
`2.9.1+rocm7.2.0.git7e1940d4`, FlyDSL 0.4.0 and AITER. No TP8 or node 50
runs are used. GPU jobs are serialized.

`ctx=2K` is capacity 2048, with live causal lengths
`2048 - query_tokens - 2 - request_id + token_id`. Each rank holds one
synthetic layer's weights. Expert shards differ across ranks. The same
allocation is reused in the timing graph; this does not reproduce rotation
through an entire model's weights.

The normal A4W4 test runs first: independent MoE reference, exact activation
codes/scales, untouched cache regions, rank agreement and ordinary/graph
replay. The unchanged relative-L2 gate is `0.001`. Timing then captures 32
steps and measures ten graph replays with GPU events, including the existing
epoch update. Every trial takes the maximum duration across the four ranks,
divided by 320 layer calls. Compilation, weight packing and correctness
checks are outside the interval. Output, top-k and cache hashes are checked
again after timing.

Each run has eight trials. Two seeds, 821 and 823, reverse CP and shape order.
The summary uses the median of all 16 trials for each shape/CP setting;
individual trials and per-seed medians are retained in the result artifact.
CP-on and CP-off runs use identical fixtures for each seed.

## Results

Latency in microseconds per layer for the whole query block:

| Query tokens/request | Total rows | CP off (µs) | CP on (µs) |
| ---: | ---: | ---: | ---: |
| 2 | 4 | 116.62 | 114.03 |
| 4 | 8 | 213.90 | 216.23 |
| 6 | 12 | 324.07 | 319.94 |
| 8 | 16 | 344.55 | 336.09 |

All 16 runs passed the independent MoE gate and state/replay checks before
timing. Outputs, indices and cache hashes remained unchanged after timing. At 2K, CP changes latency by a few percent and
does not consistently improve every shape. In particular, the two seeds
give CP-on medians of 116.02 and 112.05 µs at two tokens/request; their
combined median should not be read as a seed-independent latency.

[Machine-readable results](benchmarks/glm-tp4-a4w4-performance-20261020.json)
contain all maximum-rank trials, per-seed medians, live lengths, error
metrics and kernel source hashes.

## Reproduction

Run each `ROWS` in `4 8 12 16` with seeds 821 and 823, reversing the CP order
on the second seed:

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 \
  torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_mxfp4.py \
  --rows "$ROWS" --capacity 2048 --seed 821 --bench

# Same fixture, CP enabled:
HIP_VISIBLE_DEVICES=4,5,6,7 \
  torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_mxfp4.py \
  --rows "$ROWS" --capacity 2048 --seed 821 --bench --cp
```

## Comparison scope

These numbers are not divided by the older A8W8 table: that table uses a
different synthetic weight/routing fixture. An exploratory paired-format
harness encoded the same logical MXFP4 weights exactly in FP8, but stopped
before timing because its new A8W8 mathematical reference disagreed at the
input FP8 quantization stage. Changing its operation order reduced, but did
not eliminate, the mismatch (input relative L2 `0.0016458`; up/gate
`0.0023763` against the `0.001` gate). No kernel was changed and no tolerance
was relaxed. A same-fixture A8W8 speedup is therefore not reported here.
The validated A8W8 before/after regression from the implementation change is
recorded separately in the [A4W4 integration report](glm-tp4-mxfp4.md).
