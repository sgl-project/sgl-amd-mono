# TP4 long-context follow-up

This follows the opt-in indexer CP port in `1e795ce`. The follow-up evaluates
TP4 with two or four active requests. Enable CP with `indexer_cp=True`;
the default remains `False`. The serving caller and graph-selection requirements in
[the CP guide](glm-indexer-cp.md) still apply.

## Paged KV addressing

Four requests with 1,048,576-token capacity allocate more than 4 GiB for the
replicated BF16 MLA cache. The initial implementation formed buffer offsets
from the physical slot, which can wrap the hardware's 32-bit byte offset.
Its CP on/off outputs agreed, but the untouched-cache check failed: stores
modified different physical rows. CP equivalence alone did not catch this.

The adapted cache writer and sparse gather form each wave-uniform row's full
64-bit address in the buffer descriptor, then use small offsets within the
row. The gather issues two independent row loads before consuming them, as
in FlyDSL PR #1256 at `cd335a8d908094e2296d68d052260624bf6a8465`.
The fix applies to paged KV access with CP either off or on: physical pool
size is independent of a request's live length and compiled capacity.

The regression translates identical logical data by 4,194,304 physical slots
and compares it to the low-address run. This puts the data above the old byte
offset limit even with short requests. Outputs, intermediates, logical top-k,
queries, scores and logical cache contents must match exactly; the entire
unused prefix must remain unchanged. No full cache tensor is written to disk.

## TP4 indexer scheduling

Two additional adaptations preserve the existing score arithmetic and layout:

- Batch independent query mailbox polls before the original RoPE/Hadamard
  calculation, while retaining the query tile across score tasks.
- For full-score exchange, use one wave per peer to publish all 64 scores.
  The tile is staged in the existing reduction workspace, with a barrier
  before publication and another before workspace reuse.

These scheduling choices are restricted to TP4. The distributed selector
threshold remains 114,688 live keys per row. The native FP4, wider-row, MTP
and serving integration gaps described in the CP guide remain open.

## Reproduction and measurement

Use the real runner's synthetic fixture on four available gfx950 GPUs:

```bash
torchrun --standalone --nproc-per-node=4 \
  test/kernels/glm53/check_indexer_cp.py \
  --rows 4 --capacity 131072 --pad-rows 0 --full-context \
  --bench-steps 32 --bench-iters 10

pytest -q test/kernels/test_glm53_kernels.py::test_indexer_cp_tp4_large_cache
pytest -q test/kernels/test_glm53_kernels.py::test_indexer_cp_tp4_full_context
```

`--seed` varies weights, inputs, caches and replay inputs. For source comparisons,
use `--save-control PREFIX --exact-control` on the control and
`--check-control PREFIX --exact-control` on the candidate. These require
bitwise output/intermediate agreement and SHA256 agreement for cache contents,
projected queries and live scores. With `--slot-offset`, physical indices and
the hashed cache region are normalized for the translation; the untouched
check still covers the complete physical allocation.

Store controls on the GPU test server and remove them after recording results.
Hashes are computed outside timing, copying bounded chunks to host memory.
Each step retains three calls with shared synthetic weights and an epoch
advance. `--bench-steps 32` captures 32 such steps in a timing graph to reduce
host submission gaps. Each trial queues an untimed graph before its start
event; times use the maximum rank duration over four alternating CP off/on
trials, divided by the number of layer calls. The default of one step per
graph reproduces the initial CP guide's timing organization. These are layer
measurements, not checkpoint accuracy or serving throughput.

## Validation and results (2026-10-20)

Final confirmation used node 46, Podman `hzm_work`, MI355X and
`HIP_VISIBLE_DEVICES=4,5,6,7`, with Torch `2.9.1+rocm7.2.0.git7e1940d4`,
ROCm `7.2.26015-fc0010cf6a`, isolated FlyDSL `0.4.0` and AITER. Host process
and device visibility checks confirmed that another user's single-GPU job
used a different card. Earlier node 47 timings overlapping a shared-GPU job
were excluded. No TP8 tests were run for this follow-up.

The final source passed **108 CPU tests** in 3.82 seconds and **14 TP4 GPU
tests** in 278.30 seconds. The CPU selection covers index planning, forward
dispatch, forward specs and source digests; the dependency limitation for
`test_glm53_spec.py` in the CP guide remains. The GPU selection includes the
nine existing TP4 cases, four simultaneous long-context cases and the
low/high-address cache regression. It covers exact ties, empty shards,
padding, delayed peers and changing-length graph replay. CP off/on queries,
scores, ordered indices, outputs and cache bytes agree exactly. Independent
score relative L2 errors stay below `1e-7`, against the unchanged `0.01` gate.

All source comparisons passed exact output/intermediate and state-hash checks.
The original `1e795ce` source is the control where its cache allocation fits;
the 4-request/1M control includes the cache-addressing fix because the original
fails its untouched-cache check. The translated short-request regression also
reproduced the original failure before the fix.

Final candidate layer times in microseconds, using seed 719 and the 32-step
timing organization above. Every row is active and starts near the listed
capacity; 64K means 65,536 tokens:

| Active requests | Context capacity per request | CP off | CP on | CP speedup |
| --- | --- | --- | --- | --- |
| 2 | 4,096 | 94.45 | 92.65 | 1.019x |
| 4 | 4,096 | 142.62 | 138.75 | 1.028x |
| 4 | 65,536 | 239.86 | 213.09 | 1.126x |
| 2 | 131,072 | 224.75 | 184.89 | 1.216x |
| 4 | 131,072 | 320.46 | 252.25 | 1.270x |
| 4 | 262,144 | 487.89 | 280.28 | 1.741x |
| 4 | 1,048,576 | 1,750.60 | 453.00 | 3.864x |

These ratios measure total CP benefit. To isolate this follow-up's query-poll
and peer-exchange scheduling, a separate control retains the same 64-bit
cache addressing and paired gather but omits both scheduling changes:

| Active requests / context | Cache fix only, CP on | Final candidate, CP on | Latency reduction |
| --- | --- | --- | --- |
| 4 / 64K | 217.64 us | 213.13 us | 2.07% |
| 4 / 128K | 255.30 us | 252.58 us | 1.07% |

Each of these four measurements has less than 0.1% spread across its four
trials. Compared with the original CP commit, the complete follow-up has
mixed smaller-context results because it also changes cache access: at
4/128K, seeds 719, 733 and 751 give CP-on times of 250.55 to 252.25 us,
256.39 to 251.58 us and 251.10 to 251.00 us, respectively. At 4/256K the
change is effectively flat (280.36 to 280.28 us). The demonstrated incremental
scheduling gains are the isolated 64K/128K comparisons, not a uniform gain
over all workloads or the total 3.864x CP speedup at 1M.
