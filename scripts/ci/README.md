# AMD PR test

`.github/workflows/pr-test-amd.yml` runs the checks of the PR review guideline on
two dedicated MI355X nodes. A run takes hours, so a maintainer starts it.

## Running it

- Add the `run-ci` label to the PR. The run removes the label when it starts, so a
  later push needs a maintainer again.
- Or run the workflow (Actions, "PR Test (AMD)", "Run workflow") with the PR number.
  `models` runs the named models instead of the PR's scope (`common` for the unit
  and common kernel tests); `image` pins the SGLang image.

The run sets the `AMD PR test` status on the PR's head commit and keeps one PR
comment up to date with the report. Each model's logs (servers, benchmarks,
tests) are in the run's `mono-ci-<model>` artifact.

## What runs

`scope.py` maps the changed files to models (`models.py`):

- a model whose own package (`sgl_amd_mono/<pkg>/`, or its tests) changed runs
  the **full** tier;
- a change to `sgl_amd_mono/common/`, `sgl_amd_mono/__init__.py` or
  `pyproject.toml` runs every other model's **smoke** tier, and `common`;
- a PR that changes no model runs `common`: the unit tests and the common kernel
  tests.

For each model, `run_model.py` installs the base branch, then the PR, over the
image's sgl-amd-mono, and runs both with the same server (the model's recipe under
`test/e2e`, with `--enable-amd-monokernel`):

| Check | Full tier | Smoke tier |
| --- | --- | --- |
| Tests | unit, the model's kernel tests and serving tests (`test/e2e`) | same |
| Kernel microbenchmark | the model's timing scripts, base and PR | same |
| Accuracy | sgl-eval GSM8K, GPQA Diamond, AIME26 (and MMMU-Pro for vision models), base and PR | the serving tests' GSM8K and KL bounds |
| E2E performance | InferenceX fixed length: `sglang.benchmark.serving`, random 8k in / 1k out, range ratio 0.8, 10 prompts a slot, at the model's concurrencies; base and PR on the same node, back to back | the gate concurrencies only |
| Scope | the server's startup lines that say which steps the launches serve | same |

The base branch's accuracy is cached per model, image and base commit, so it
runs once until either moves. A model new in the PR is compared against SGLang's
own path (monokernels off).

## The gate

The status fails when:

- a test fails, or a serving test, benchmark or server does not finish;
- the PR's server logs no monokernel routing (every step ran SGLang's own path);
- an accuracy benchmark scores below the base branch by more than its tolerance
  (`EVALS` in `models.py`);
- median TPOT is more than 2% worse than the base branch at any concurrency;
- for a `[Perf]` or `[Feat]` PR, no gate concurrency gains 5% or more in median
  TPOT;
- the PR changes existing lines of `.github/CODEOWNERS` (a new model's added
  owner line is a warning for the reviewer).

Warnings do not fail it: a model without a checkpoint on the CI nodes, a GPU the
nodes do not have, a benchmark the CI cannot run yet, or a model package without
a `README.md` (Doc). Until the CI covers a check, the author runs it and gives the
results in the PR, and the reviewer verifies them. LiveCodeBench, SWE-bench
Verified and OCRBench are not in the CI yet.

## Infrastructure

- Nodes: two MI355X nodes kept for this CI, `crsuse2-m2m-v3-060` and
  `crsuse2-m2m-v3-123` of the Crusoe cluster's `amd-aifw-sglang-ci` partition
  (`NODES` in `runner.sh`). sglang's own CI uses the partition's other nodes: its
  runners must set `SLURM_EXCLUDE` to these two, or a job here waits behind theirs.
- Runners: two, on the cluster's login node (`runner.sh`), label
  `sgl-amd-mono-mi355x`, r1 pinned to v3-060 and r2 to v3-123. A runner takes one
  job at a time, so the two nodes run two models at once and never two jobs on one
  node. To set them up, as the CI account on the login node (with
  `loginctl enable-linger`), and with a registration token from a repo admin:

  ```bash
  bash scripts/ci/runner.sh install 1 && bash scripts/ci/runner.sh install 2
  bash scripts/ci/runner.sh register 1   # paste the token; then the same for 2
  bash scripts/ci/runner.sh status
  ```
- `slurm_launch.sh` (on the runner) submits `slurm_job.sh` as a whole-node job,
  streams its log and collects its results; `slurm_job.sh` (on the node) fetches
  the PR and base commits onto node-local disk, starts the SGLang ROCm container
  and runs `run_model.py` in it. The CI scripts come from the base branch; only
  the package and its tests come from the PR.
- Image: the newest `rocm/sgl-dev:v*-rocm10-mi35x-YYYYMMDD`, unless the run pins one.
- Checkpoints: under `$CI_MODEL_ROOT` on shared storage (`paths` in `models.py`).
  Shared storage reads too slowly to load servers from, so a job first mirrors its
  model's checkpoints to the node's local disk (once; again when the source's
  config or index changes) and the container sees the mirror at the same paths.
  `CI_STAGE_ONLY=1` makes a job only fill the mirror.
- GPQA is a gated dataset: the job fills its cache with the token in
  `~/.sgl_amd_mono_ci/hf_token` on the runner account, before any PR code runs.

## Adding a model

Add an entry to `MODELS` in `models.py`: its packages, the server (a recipe module
under `test/e2e` and its argument list), the checkpoint paths on the CI nodes, its
kernel and serving tests, timing scripts, accuracy benchmarks and concurrencies
(`gate_concurrency`: the ones its launches cover).

## Trial runs

`MONO_CI_NUM_EXAMPLES` (examples a benchmark) and `MONO_CI_CONCURRENCY`
(comma-separated) shorten a run of the CI itself; such a run does not touch the
accuracy cache.
