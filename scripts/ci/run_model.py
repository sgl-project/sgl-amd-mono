"""One model's AMD PR test, inside the SGLang ROCm container on a whole CI node.

    python3 scripts/ci/run_model.py --model deepseek_v41 \\
        --pr /ci/pr --base /ci/base --out /ci/out --cache /cache

The base branch first, then the PR, each installed in turn over the image's
sgl-amd-mono (editable):

* base: the kernel-only timing scripts, then one server (the model's recipe with
  ``--enable-amd-monokernel``) for the speed test and, unless a result for the same
  image and base commit is cached, the accuracy benchmarks;
* PR: the unit tests, the model's kernel tests, the timing scripts, one server for
  the speed test and the accuracy benchmarks, then the model's serving tests.

The smoke tier (a model that runs only because shared code changed) skips the
accuracy benchmarks, whose bound its serving tests hold, and times only the gate
concurrencies.

Both servers take the PR's recipe, so the two sides differ only in the package. A
model new in the PR has no base package: its base side is SGLang's own path
(monokernels off). Every step runs whatever an earlier one did; ``result.json``
(rewritten after each step) and ``logs/`` hold the outcome. ``report.py`` judges it.
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    COMMON_KERNEL_TESTS,
    EVALS,
    GPU,
    ISL,
    MODELS,
    OSL,
    PROMPTS_PER_SLOT,
    RANGE_RATIO,
)

PORT = 31000
LAUNCH_TIMEOUT = 3600
SGL_EVAL = "sgl-eval==0.1.3"
# Trial runs of the CI itself: fewer examples a benchmark, other concurrencies.
NUM_EXAMPLES = os.environ.get("MONO_CI_NUM_EXAMPLES")
CONCURRENCY = os.environ.get("MONO_CI_CONCURRENCY")


class Run:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.key = args.model
        self.m = MODELS.get(args.model, {})
        self.result = {
            "model": args.model,
            "tier": args.tier,
            "name": self.m.get("name", "common"),
            "image": os.environ.get("CI_IMAGE", ""),
            "node": os.uname().nodename,
            "pr_sha": os.environ.get("CI_PR_SHA", ""),
            "base_sha": os.environ.get("CI_BASE_SHA", ""),
            "skipped": {},
            "tests": {},
            "microbench": {},
            "perf": {},
            "accuracy": {},
            "routing": {},
        }
        for k, v in self.m.get("paths", {}).items():
            os.environ[k] = v

    # -- plumbing -------------------------------------------------------------

    def save(self):
        (self.out / "result.json").write_text(json.dumps(self.result, indent=2))

    def log(self, msg):
        print(f"[mono-ci {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def sh(self, cmd, name, cwd=None, env=None, timeout=None, echo=True):
        """Run ``cmd`` (a string, through bash) with its output in logs/<name>.log,
        and echoed when ``echo``. Returns the exit code; 124 on timeout."""
        path = self.logs / f"{name}.log"
        self.log(f"$ {cmd}  (cwd={cwd or os.getcwd()}, log={path.name})")
        argv = ["bash", "-c", cmd]
        if timeout:
            argv = ["timeout", "--kill-after=60", str(timeout), *argv]
        with open(path, "a") as f:
            p = subprocess.Popen(
                argv,
                cwd=cwd,
                env={**os.environ, **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            for line in p.stdout:
                f.write(line)
                if echo:
                    sys.stdout.write(line)
            rc = p.wait()
        try:  # what the step left running (torchrun ranks, servers)
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.log(f"exit {rc}{' (timed out)' if rc == 124 else ''}: {name}")
        return rc

    def install(self, src):
        """Make ``src`` the installed sgl-amd-mono, with its own kernel cache."""
        side = Path(src).name
        os.environ["SGLANG_AMD_MONO_CACHE_DIR"] = f"/tmp/sgl-amd-mono-cache-{side}"
        rc = self.sh(
            "pip uninstall -y -q sgl-amd-mono; "
            f"pip install -q --no-deps --no-build-isolation -e {shlex.quote(str(src))}",
            f"install-{side}",
        )
        if rc:
            raise RuntimeError(f"installing {src} failed")

    # -- tests ----------------------------------------------------------------

    def pytest(self, name, files, timeout):
        files = [f for f in files if (Path(self.args.pr) / f).exists()]
        if not files:
            return
        xml = self.logs / f"{name}.xml"
        rc = self.sh(
            f"python3 -m pytest -v -rA --junitxml={xml} " + " ".join(files),
            name,
            cwd=self.args.pr,
            timeout=timeout,
        )
        counts = {"passed": 0, "failed": 0, "skipped": 0, "rc": rc}
        if xml.exists():
            for case in ET.parse(xml).getroot().iter("testcase"):
                tags = {c.tag for c in case}
                if tags & {"failure", "error"}:
                    counts["failed"] += 1
                elif "skipped" in tags:
                    counts["skipped"] += 1
                else:
                    counts["passed"] += 1
        elif rc:
            counts["failed"] = 1  # collection or the run itself failed
        if rc == 124:
            counts["failed"] = max(counts["failed"], 1)
            counts["timeout"] = True
        self.result["tests"][name] = counts
        self.save()

    # -- servers --------------------------------------------------------------

    def recipe(self):
        """(model path, env, args) of the model's server, from the PR's tree."""
        m = self.m
        if "recipe" in m:
            mod, attr = m["recipe"]
            path = Path(self.args.pr) / "test" / "e2e" / f"{mod}.py"
            spec = importlib.util.spec_from_file_location(mod, path)
            r = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(r)
            return r.MODEL, dict(r.ENV), list(getattr(r, attr))
        return os.environ[m["model_env"]], dict(m["env"]), list(m["args"])

    def model_ready(self):
        """Why this model's server cannot run here, or None."""
        if GPU not in self.m.get("gpus", [GPU]):
            return f"the launches cover {', '.join(self.m['gpus'])} only; CI nodes are {GPU}"
        if not self.m.get("paths") or not ("recipe" in self.m or "args" in self.m):
            return "no checkpoint or serving recipe on the CI nodes"
        missing = [p for p in self.m["paths"].values() if not Path(p).exists()]
        if missing:
            return f"checkpoint not on the CI nodes: {', '.join(missing)}"
        return None

    def serve(self, side, mono=True):
        model, env, args = self.recipe()
        args = [*args, "--host", "127.0.0.1", "--port", str(PORT)]
        if mono:
            args.append("--enable-amd-monokernel")
        log = open(self.logs / f"server-{side}.log", "w")
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            *args,
        ]
        self.log(f"server {side}: {shlex.join(cmd)}  env={env}")
        p = subprocess.Popen(
            cmd,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + LAUNCH_TIMEOUT
        while time.time() < deadline:
            if p.poll() is not None:
                self.log(f"server {side} exited {p.returncode} during startup")
                return None, model, args
            try:
                if requests.get(
                    f"http://127.0.0.1:{PORT}/health_generate", timeout=60
                ).ok:
                    self.log(f"server {side} ready")
                    return p, model, args
            except requests.RequestException:
                pass
            time.sleep(10)
        self.log(f"server {side} not ready after {LAUNCH_TIMEOUT}s")
        self.stop(p)
        return None, model, args

    def stop(self, p):
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=120)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run("pkill -9 -f sglang.launch_server; sleep 15", shell=True)

    def routing(self, side):
        """The startup lines that say which steps the launches serve."""
        text = (self.logs / f"server-{side}.log").read_text(errors="replace")
        lines = sorted(set(re.findall(r"monokernel \S+ on \S+ runs [^\n]*", text)))
        self.result["routing"][side] = lines
        self.save()

    def perf(self, side, model):
        res = self.result["perf"].setdefault(side, {})
        smoke = self.args.tier == "smoke"
        levels = self.m["gate_concurrency" if smoke else "concurrency"]
        if CONCURRENCY:
            levels = [int(c) for c in CONCURRENCY.split(",")]
        for c in levels:
            out = self.logs / f"bench-{side}-c{c}.jsonl"
            out.unlink(missing_ok=True)
            n = max(c * PROMPTS_PER_SLOT, PROMPTS_PER_SLOT)
            rc = self.sh(
                f"python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 "
                f"--port {PORT} --model {shlex.quote(model)} --dataset-name random "
                f"--random-input-len {ISL} --random-output-len {OSL} "
                f"--random-range-ratio {RANGE_RATIO} --max-concurrency {c} "
                f"--num-prompts {n} --warmup-requests {c} --output-file {out}",
                f"bench-{side}-c{c}",
                timeout=3 * 3600,
            )
            rows = out.read_text().splitlines() if out.exists() else []
            if rc or not rows:
                res[str(c)] = {"error": f"exit {rc}"}
            else:
                d = json.loads(rows[-1])
                res[str(c)] = {
                    k: d.get(k)
                    for k in (
                        "median_tpot_ms",
                        "median_ttft_ms",
                        "median_itl_ms",
                        "output_throughput",
                        "completed",
                    )
                }
            self.save()

    def evals(self, side, threads):
        res = self.result["accuracy"].setdefault(side, {})
        for name in self.m["evals"]:
            spec = EVALS[name]
            out = self.logs / f"eval-{side}-{name}"
            cmd = (
                f"sgl-eval run {name} --base-url http://127.0.0.1:{PORT}/v1 "
                f"--num-threads {threads} --out-dir {out} " + " ".join(spec["args"])
            )
            if NUM_EXAMPLES:
                cmd += f" --num-examples {NUM_EXAMPLES}"
            log_path = self.logs / f"eval-{side}-{name}.log"
            rc = self.sh(cmd, f"eval-{side}-{name}", timeout=8 * 3600, echo=False)
            log = log_path.read_text(errors="replace")
            if "gated dataset" in log:
                # a gated dataset the job cached with the CI's token: read the cache
                log_path.unlink()
                rc = self.sh(
                    cmd,
                    f"eval-{side}-{name}",
                    env={"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                    timeout=8 * 3600,
                    echo=False,
                )
                log = log_path.read_text(errors="replace")
            metrics = sorted(out.glob("*/metrics.json"))
            if "gated dataset" in log or (rc and "offline mode" in log.lower()):
                res[name] = {
                    "skipped": "gated dataset: the CI needs an HF token for it"
                }
            elif rc or not metrics:
                res[name] = {"error": f"exit {rc}"}
            else:
                d = json.loads(metrics[-1].read_text())
                agg = d.get("aggregate", {})
                # sgl-eval's headline: pass@1 (the mean over repeats) with repeats
                score = agg.get("pass@1", agg.get("score"))
                if d.get("partial") or not d.get("total_completion_tokens"):
                    res[name] = {"error": "no completions (partial or empty run)"}
                else:
                    res[name] = {
                        "score": score,
                        "n_repeats": d.get("n_repeats"),
                        "num_examples": d.get("num_examples"),
                        "no_answer": agg.get("no_answer"),
                    }
            self.log(f"{side} {name}: {res[name]}")
            self.save()

    def microbench(self, side, src):
        for i, cmd in enumerate(self.m.get("microbench", [])):
            name = f"microbench-{side}-{i}"
            rc = self.sh(cmd, name, cwd=src, timeout=3600)
            tail = (self.logs / f"{name}.log").read_text(errors="replace")[-4000:]
            self.result["microbench"].setdefault(side, []).append(
                {"cmd": cmd, "rc": rc, "tail": tail}
            )
            self.save()

    # -- the run --------------------------------------------------------------

    def baseline_cache(self):
        image = re.sub(r"[^A-Za-z0-9._-]", "_", self.result["image"] or "unknown")
        base = self.result["base_sha"] or "unknown"
        return Path(self.args.cache) / "baselines" / self.key / image / f"{base}.json"

    def side(self, side, mono, run_evals):
        p, model, args = self.serve(side, mono)
        if p is None:
            self.result["perf"][side] = {"error": "server did not start"}
            if run_evals:
                self.result["accuracy"][side] = {"error": "server did not start"}
            self.save()
            return
        try:
            self.routing(side)
            self.perf(side, model)
            if run_evals:
                # no more requests in flight than the launches cover: wider steps
                # take SGLang's own path, and the score would not be the kernels'
                self.evals(side, max(self.m["gate_concurrency"], default=1))
        finally:
            self.stop(p)

    def run(self):
        a = self.args
        # the checkouts belong to the job's user, not the container's root
        self.sh("git config --global --add safe.directory '*'", "setup")
        self.sh(f"pip install -q {SGL_EVAL}", "setup")
        self.result["sglang"] = subprocess.run(
            "git -C /sgl-workspace/sglang log -1 --format='%h %cs %s'",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if self.key == "common":
            self.install(a.pr)
            self.pytest("unit", ["test/unit"], 1800)
            self.pytest("kernels", COMMON_KERNEL_TESTS, 3600)
            return
        why = self.model_ready()
        if why:
            self.result["skipped"]["serving"] = why
        base_has_pkg = all(
            (Path(a.base) / "sgl_amd_mono" / p).is_dir() for p in self.m["packages"]
        )
        self.result["baseline"] = (
            "base branch, monokernels on"
            if base_has_pkg
            else "monokernels off (new model)"
        )
        cache = self.baseline_cache()
        full = a.tier == "full"

        # base
        if base_has_pkg:
            self.install(a.base)
            self.microbench("base", a.base)
        else:
            self.install(a.pr)
        if not why:
            # a trial run (fewer examples) neither reads nor writes the cache
            if NUM_EXAMPLES:
                cache = Path(os.devnull)
            cached = json.loads(cache.read_text()) if cache.is_file() else None
            if cached:
                self.result["accuracy"]["base"] = cached
                self.result["accuracy"]["base_cached"] = str(cache)
            self.side("base", base_has_pkg, run_evals=full and not cached)
            got = self.result["accuracy"].get("base", {})
            if (
                not (cached or NUM_EXAMPLES)
                and got
                and all("score" in v for v in got.values())
            ):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(got))

        # PR
        self.install(a.pr)
        self.pytest("unit", ["test/unit"], 1800)
        self.pytest("kernels", self.m.get("kernel_tests", []), 2 * 3600)
        self.microbench("pr", a.pr)
        if not why:
            self.side("pr", True, run_evals=full)
            self.pytest("e2e", self.m.get("e2e_tests", []), 4 * 3600)
        elif self.m.get("e2e_tests"):
            self.result["skipped"]["e2e"] = why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pr", required=True, help="the PR's checkout")
    ap.add_argument("--base", required=True, help="the base branch's checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cache", required=True, help="persistent: pip, datasets, baselines"
    )
    ap.add_argument("--tier", choices=("full", "smoke"), default="full")
    args = ap.parse_args()
    if args.model != "common" and args.model not in MODELS:
        sys.exit(f"unknown model {args.model}")
    run = Run(args)
    try:
        run.run()
    except Exception as e:  # report what ran; the report marks the error
        run.result["error"] = f"{type(e).__name__}: {e}"
        run.log(run.result["error"])
    finally:
        run.result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run.save()


if __name__ == "__main__":
    main()
