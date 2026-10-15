<!-- PR title: [Type] scope: title, or [Type] [Hardware] scope: title.
Type: Feat, Fix, Refactor, Perf, Doc, Test, CI, or Chore.
Scope: any module or model name, e.g. deepseek_v41, common, or ci.
Hardware: optional single tag without spaces, in uppercase (AMD, MI355X).
Reverts: use GitHub's default title, Revert "original title".
Examples: [Fix] deepseek_v41: Fix the K2 pad-row write; [Perf] [MI355X] deepseek_v41: Tune the FFN launch. -->

## Motivation

<!-- Describe the purpose and goals of this pull request. -->

## Modifications

<!-- Detail the changes made in this pull request. -->

## Accuracy Tests

<!-- If this pull request changes kernel or forward code, give accuracy results with the monokernels on and off, on the same server. -->

## Speed Tests and Profiling

<!-- If this pull request affects speed, give per-user decode results with the monokernels on and off, including past the largest width. -->

## Checklist

- [ ] `pre-commit run --all-files` passes.
- [ ] Tests under `test/` cover the change.
- [ ] The SGLang commit or PR this was tested against is named above.
- [ ] Accuracy and speed results are attached when kernel or forward code changes.

## Review and Merge Process

1. The `lint` and `Check PR title` checks must pass.
2. Ask a code owner from `.github/CODEOWNERS` to review.
3. Merge with "Squash and merge": the PR title becomes the commit title on `main`.
