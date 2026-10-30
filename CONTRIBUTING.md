# Contributing to SGLang-AMD-Mono

These rules apply to every pull request. They come from this repository's
merged PRs and reviews. New code follows them; existing code that differs
changes only in a PR about that change.

## Workflow

- **One change per PR.** PRs are squash-merged, so the title and body become
  the commit on `main`. An unrelated fix goes in its own `[Fix]` PR.
- **Stacked PRs** start with `Stacked on #N; merge #N first.` and are rebased
  once the base merges.
- **Keep the PR a draft** until it has the results described below.
- **Run `pre-commit run --all-files`.** The `lint` check runs the same hooks.
- **CI runs no tests yet.** Run the tests your change touches and report them
  in the PR; reviewers rerun what they can.
- **Review.** A code owner from `.github/CODEOWNERS` approves. A change to
  `common/` also needs the owners of the models it affects. Reviewers say what
  they verified on which commit, and label each item Required, Suggestion or
  Nit.
- **Merge** with "Squash and merge" once `lint` and `Check PR title` pass and
  the description matches the commit that merges.

## PR Title

`Check PR title` enforces `[Type] scope: Title` or
`[Type] [Hardware] scope: Title`.

| Type | Use for |
| --- | --- |
| `Feat` | A new model, coverage (layers, widths, TP sizes, speculative modes) or option |
| `Fix` | Wrong outputs, a crash, a hang, or a configuration that should be refused |
| `Perf` | Faster, with the same outputs |
| `Refactor` | The same behavior and output bits, with different code |
| `Test` / `Doc` / `CI` | Tests, documentation, or workflows only |
| `Chore` | Everything else: comments, code owners, placeholder packages |

- **Scope** is the package directory (`deepseek_v41`, `common`), `sgl_amd_mono`
  for the package root, `github` for `.github/`, or `docs` for
  repository-root documentation. Don't abbreviate (`dsv41`);
  join two packages with a slash (`glm53/glm53_flash`).
- **Hardware** is optional, for a change specific to one GPU (`[MI355X]`).
  Don't add `[AMD]`; every PR here is for AMD GPUs.
- **Title** starts with a capitalized imperative verb and has no trailing
  period: `[Fix] deepseek_v41: Use SGLang's boundary arithmetic in the FFN seam`.

## PR Description

Fill in every template section, in plain English, with numbers in tables. The
body becomes the commit message, so it must describe the commit that merges.
After review changes the code, update the text, and rerun or label any number
measured on an earlier commit.

**Every result names its setup:** the GPUs and how many, the TP size, the image
tag and digest, the SGLang commit and any unmerged SGLang PRs applied, AITER and
FlyDSL versions if they matter, the model revision, the server flags (or the
`test/e2e/*_server.py` file), and this branch's commit.

- **Motivation:** the problem and why it matters. If the change depends on or
  responds to an SGLang change, link it as `sgl-project/sglang#N`; a bare `#N`
  links to this repository. Define terms a reader outside your model may not
  know.
- **Modifications:** the changes by area, new refusals, each new weight copy
  with its size per rank, and what you deliberately left unchanged.
- **Accuracy Tests**, for kernel or forward changes:
  - Monokernels on and off in the same setup, on all 1,319 GSM8K questions or
    the model's `test/e2e` file. A parity claim gives a paired comparison: the
    questions right in only one arm, and McNemar's p.
  - Kernel checks, bitwise or against the spec's `NUMERICS`, naming the widths
    and TP sizes covered. Say what you did not run.
- **Speed Tests and Profiling**, when speed can change:
  - Per-user decode tok/s or TPOT, on and off (or `main` and this PR), from 1
    request to past the largest width. Use a fresh server per arm, A B B A
    order, and at least two rounds, and state the spread. With speculative
    decoding, give the accept length.
  - For kernel changes, the VGPR, spill, LDS and scratch changes, from a
    compile-only build of every variant touched. A spill can hide at a width
    the speed test did not run.
  - Recommended: an InferenceX AgentX run, on and off, which also catches
    prefill costs on long cached contexts. Name the harness version, corpus
    snapshot, window length and simulated accept length, and give
    interactivity (P50, P90) and throughput per GPU.
  - Report slowdowns as plainly as speedups.
- **A section that does not apply** says `Not applicable:` and why. A
  comment-only change says how you showed the code is unchanged (for example,
  the same AST once docstrings are dropped).
- **Checklist:** tick only what is true, give each unticked item a reason
  (`(N/A: no kernel change)`), and replace any "results to follow" before merge.

## Code

### Monokernel Contract

Breaking one of these is a blocker.

- **Refuse; never fall back silently.** A configuration the launches cannot
  serve as SGLang would fails at startup with its reason. Declare what you can
  in `Requirements`; `config_disable_reasons` and `model_disable_reasons`
  return every other reason, not only the first.
- **Route by the graph key** (the batch size the graph was captured for). A
  captured graph replays without Python, so each per-step decision comes from
  the graph key or device data, never from a host value read during capture.
- **Bound every wait.** Waits on another workgroup or GPU go through the guard
  in `common/` with a deadline, so a failed wait stops the launch on every
  rank.
- **Serve only what the spec declares.** Widths and TP sizes come from
  `devices` (`DeviceLimits`); prefill and wider steps run SGLang's path.
- **Read SGLang's weights in place.** A weight that needs another layout or
  precision is copied once at startup and logged, and the PR gives its size
  per rank.
- **Keep host syncs off the per-step path:** `.item()`, `.cpu()`, `.tolist()`,
  `torch.cuda.synchronize()`. Build-time and check-mode code may use them.
- **Declare what you read from SGLang** in the module's `SGLANG_READS`;
  `test/unit/test_contract.py` checks it against the installed SGLang.
- **Read SGLang attributes directly.** Use `getattr(obj, name, default)` only
  for an attribute SGLang sets on some paths, and make the default refuse.
- **Patch SGLang only as a last resort:** declared in `SGLANG_READS`, restored
  in `finally`, and with the PR naming the SGLang hook that would remove it.

### Errors

- `assert` for invariants this package controls; `ValueError` for a refused
  configuration; `RuntimeError` for a broken environment.
- `try` only for SGLang's moved modules
  (`except ModuleNotFoundError:  # SGLang before sgl-project/sglang#N`), for
  turning a builder's `ValueError` into a refusal reason, or for `try` /
  `finally` cleanup. Never catch `Exception` to continue.
- Don't guard against states that cannot happen: for every check, ask what
  breaks if you delete it.

### Structure

- A model lives in `sgl_amd_mono/<model>/`: `spec.py` (the spec and its
  forward) and `<isa>/` such as `gfx950/` (kernels and launch runner),
  optionally `weights.py` and `check.py` (the `SGLANG_AMD_MONO_CHECK` mode).
- Device code that more than one model uses lives in `common/`. Reuse it and
  the FlyDSL and AITER helpers instead of copying; a fix to one copy misses the
  others.
- Split code by kernel stage or launch, not by line count. Don't extract a
  private helper that is called once, or add a base class before a second
  implementation exists.
- Keep new modules under about 2,000 lines. Split the existing larger kernel
  files in their own `[Refactor]` PRs.
- Name magic numbers; a measured value's comment says where and on what. Define
  a constant in the one module that uses it, and pass parameters down instead
  of repeating a caller's default.

### Style

- **Imports:** isort's black profile (pre-commit runs it); absolute imports
  such as `from sgl_amd_mono.common.plan import BLOCKS`; no `import *` or
  `sys.path` edits. Import at the module top, except inside the function that
  uses it for (1) modules that touch the GPU at import, such as AITER, so
  `pytest test/unit` collects without a GPU, and (2) SGLang runtime internals a
  spec reads when it builds or runs. The monokernel interface (`MonoSpec`,
  `Requirements`) stays at the top.
- **Names:** `_` prefix only for names used within their own module. A name
  says what the value is; model, kernel and SGLang terms are fine (K1, seam,
  width, verify row).
- **Types:** `X | None` and built-in generics (`list[int]`); no new `Optional`
  or `Union`, and no `Any` when the type is known.
- **Environment variables:** `SGLANG_AMD_MONO_<NAME>`, only for check modes,
  fault injection, debugging or turning a path off, and documented in the
  README. Tuning lives in the spec, not in knobs.
- **Logging:** `logger = logging.getLogger(__name__)` with `%`-style arguments
  (`logger.info("built %d layers", n)`). At startup a spec logs what it built
  and each weight it copied. No `print` in `sgl_amd_mono/`.

### Comments and Attribution

- **Adapted code** keeps the original's license and copyright lines in its
  header, names the source repository and commit, and adds a new source to
  `NOTICE`. MIT and Apache-2.0 both require the notices.
- **Docstrings:** a summary line and at most one paragraph.
- **Comments** state what the code cannot show: why an order, rounding or
  layout matches SGLang or AITER bit for bit; cross-rank and mailbox
  invariants; LDS, register and load-order constraints; upstream anchors
  (`sgl-project/sglang#N`); and in tests, the measured values behind a bound.
- **Don't write** history, comments that restate the code, timings in code,
  author tags, process markers, or TODOs without an issue link.
- Write code names in double backticks (``` ``name`` ```), in English.

## Tests

| Directory | Runs on | Covers |
| --- | --- | --- |
| `test/unit` | CPU | Specs, refusals, routing and the SGLang contract. Must collect without a GPU; mark GPU cases with `pytest.mark.skipif`. |
| `test/kernels` | GPU | Each launch against SGLang's reference ops or a simple torch reference, at every width (`SPEC.devices[...].widths`) and TP size the spec serves, compiling every variant it can select. |
| `test/e2e` | GPU server | Accuracy past the largest width, the launches running on every rank, check mode within `NUMERICS` (if any), and a fault injection that stops the server. |

- **Test what the kernel computes**, not how the code is built; a test that
  restates the implementation is not a reference.
- **Justify every bound**, derived or measured, with the numbers in a comment
  beside it. Never widen a bound to make a test pass.
- **A `[Fix]` adds a test that fails before the fix**, and the PR says so.
- **Exercise the changed path:** if the change is to a slow wait, make the
  wait slow.
- **Run a new GPU test several times in a row** before submitting it.
- **A change to `common/`** runs the kernel tests of every model that uses the
  changed code, or says which it could not run.

## Adding a Model

1. Create `sgl_amd_mono/<model>/`. A `[Chore]` PR that only claims the package
   and its code owner is fine.
2. Register the spec in `SPECS` in `sgl_amd_mono/__init__.py`.
3. Add the owner's line to `.github/CODEOWNERS`.
4. Add the model to the README: Supported Models, a Usage command matching the
   e2e server flags, Repository Structure and Testing.
5. Add its unit, kernel and e2e tests.
