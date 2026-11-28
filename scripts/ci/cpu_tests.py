"""The AMD PR test's first stage: every test under ``test/unit`` on a CPU runner.

    python scripts/ci/cpu_tests.py --pr pr --base base --out cpu-tests

Runs ``pytest test/unit`` in the PR's checkout and, with ``--base``, in its base
branch's, against the same SGLang source (``PYTHONPATH``), and fails on the PR's
new failures only: a test that fails on the base branch too, with the same message
(``report._same_failure``), is the base branch's. Without ``--base`` (a push to
main) every failure fails. Prints a markdown summary for the job summary.
"""

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import _same_failure  # noqa: E402
from run_model import failure_signature  # noqa: E402

TIMEOUT = 1800
# a failure that only says this runner lacks AITER or a GPU
NEEDS_GPU = re.compile(
    r"No module named '(aiter|flydsl\._mlir)|No CUDA GPUs are available"
    r"|Torch not compiled with CUDA enabled|Found no NVIDIA driver|no HIP GPUs"
)


def run_unit(src: Path, xml: Path) -> dict:
    """pytest ``test/unit`` of the checkout ``src``: counts, and how each failure
    failed ({test id: signature}); None for the signatures when pytest wrote no
    junit XML (it did not run, or timed out before writing it)."""
    xml.unlink(missing_ok=True)
    try:
        rc = subprocess.run(
            # a file that does not import is one failure; the other files still run
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rA",
                "--continue-on-collection-errors",
                f"--junitxml={xml}",
                "test/unit",
            ],
            cwd=src,
            timeout=TIMEOUT,
        ).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    res = {
        "rc": rc,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "failed_msgs": None,
        "gpu_only": set(),
    }
    if not xml.exists():
        return res
    state, msgs = {}, {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        cls, name = case.get("classname"), case.get("name")
        tid = f"{cls}::{name}" if cls else name  # a file that did not import: its path
        bad = [c for c in case if c.tag in ("failure", "error")]
        if bad:
            state[tid] = "failed"
            msgs.setdefault(tid, failure_signature(bad[0], src.resolve()))
            # a file that does not import says why in its text, not its message
            if NEEDS_GPU.search(f"{bad[0].get('message') or ''} {bad[0].text or ''}"):
                res["gpu_only"].add(tid)
        elif any(c.tag == "skipped" for c in case):
            state.setdefault(tid, "skipped")
        else:
            state.setdefault(tid, "passed")
    for st in state.values():
        res[st] += 1
    res["failed_msgs"] = msgs
    return res


def _names(ids, n=5):
    short = [f"`{i.removeprefix('test.unit.')}`" for i in ids[:n]]
    return ", ".join(short) + (f" and {len(ids) - n} more" if len(ids) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", required=True, help="the PR's checkout")
    ap.add_argument("--base", default="", help="its base branch's checkout, if any")
    ap.add_argument("--out", required=True, help="where the junit XML files go")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pr = run_unit(Path(args.pr), out / "unit-pr.xml")
    base = run_unit(Path(args.base), out / "unit-base.xml") if args.base else None
    sglang = os.environ.get("SGLANG_SHA", "?")

    lines = [
        "## static-checks (unit + sglang contract)",
        "",
        f"`test/unit` on CPU against SGLang `{sglang}`: "
        f"{pr['passed']} passed, {pr['failed']} failed, {pr['skipped']} skipped "
        f"(pytest exit {pr['rc']}).",
        "",
    ]
    if pr["failed_msgs"] is None:
        lines.append(
            "❌ pytest wrote no results"
            + (" (timed out)" if pr["rc"] == 124 else "")
            + "; see the step log."
        )
        print("\n".join(lines))
        return 1
    # what this runner cannot run (no AITER, no GPU) is `common`'s to judge, on a
    # GPU node: a new test that needs them does not fail here
    gpu_only = sorted(pr["gpu_only"])
    failed = sorted(i for i in pr["failed_msgs"] if i not in gpu_only)
    if base is None:
        new, known, why = failed, [], "no base branch to compare with"
    elif base["failed_msgs"] is None:
        new, known, why = failed, [], "the base branch's run wrote no results"
    else:
        new = [
            i
            for i in failed
            if not _same_failure(pr["failed_msgs"][i], base["failed_msgs"].get(i))
        ]
        known = [i for i in failed if i not in new]
        why = ""
    if pr["rc"] not in (0, 1):
        # pytest stopped partway (interrupted, an internal error): what it did not
        # run cannot be compared with the base branch
        lines.append(
            f"❌ pytest exited {pr['rc']}: not every test ran; see the step log."
        )
        print("\n".join(lines))
        return 1
    if gpu_only:
        lines.append(
            f"➖ {len(gpu_only)} need AITER or a GPU; `common` runs them on a GPU node: {_names(gpu_only)}"
        )
    if known:
        lines.append(
            f"⚠️ {len(known)} also fail on the base branch, the same way (not this PR's): {_names(known)}"
        )
    if new:
        lines.append(
            f"❌ **{len(new)} new failure(s)**: {_names(new)}"
            + (f" ({why})" if why else "")
        )
        lines += ["", "<details><summary>Messages</summary>", ""]
        lines += [f"- `{i}`: {pr['failed_msgs'][i][:300]}" for i in new]
        lines += ["", "</details>"]
    elif not known and not gpu_only:
        lines.append("✅ every test passed")
    print("\n".join(lines))
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
