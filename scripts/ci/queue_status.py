"""Keep each waiting PR test's ``AMD PR test`` status saying where it is in the queue.

    python scripts/ci/queue_status.py [--run RUN_ID] [--models "a, b"] [--dry-run]

The PR test's GPU jobs share the two runners labelled ``sgl-amd-mono-mi355x`` with
the nightly, and a run's status otherwise reads "Running: <models>" from the moment
it is planned to the moment its last job ends, which can be most of a day: it looks
hung while it waits. This rewrites the pending description of every PR test still
running, from the jobs GitHub reports, as one of:

    Waiting for a GPU runner: ~N jobs ahead. Will run: <models>
    3/14 done; running qwen38_flash_next; 9 waiting, ~N jobs ahead

"Ahead" counts the GPU jobs running now and those queued by an older PR test or
nightly run. GitHub does not promise to hand queued jobs out oldest run first, so it
is an estimate. Only a status that is still pending and points at the run is
rewritten, and not once the run's GPU jobs have all ended: the report job then
sets the final status.

``--run`` handles that one run only (the plan job calls it before its GPU jobs
exist, with ``--models`` for the description); without it, every PR test that is
queued or in progress (the queue-status workflow, on a schedule). Reads
``GITHUB_TOKEN`` (actions: read, statuses: write) and ``GITHUB_REPOSITORY``.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
RUNNER_LABEL = "sgl-amd-mono-mi355x"
CONTEXT = "AMD PR test"
PR_TEST = "pr-test-amd.yml"
GPU_WORKFLOWS = (PR_TEST, "nightly-amd.yml")


def api(path, method="GET", body=None):
    req = urllib.request.Request(
        f"{API}/{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def active_runs(repo, workflow):
    """The workflow's queued and in-progress runs."""
    runs = []
    for status in ("queued", "in_progress"):
        q = urllib.parse.urlencode({"status": status, "per_page": 100})
        runs += api(f"repos/{repo}/actions/workflows/{workflow}/runs?{q}")[
            "workflow_runs"
        ]
    return runs


def gpu_jobs(repo, run_id):
    """A run's jobs on the mono runners (by label: a job's name says nothing)."""
    jobs, page = [], 1
    while True:
        q = urllib.parse.urlencode({"per_page": 100, "page": page})
        batch = api(f"repos/{repo}/actions/runs/{run_id}/jobs?{q}")["jobs"]
        jobs += [j for j in batch if RUNNER_LABEL in j.get("labels", [])]
        if len(batch) < 100:
            return jobs
        page += 1


def model_of(job):
    """``test (qwen38_flash_next, full)`` -> ``qwen38_flash_next``; the name otherwise."""
    name = job["name"]
    return name[name.find("(") + 1 :].split(",")[0].strip() if "(" in name else name


def describe(jobs, ahead, models=""):
    """The status description for a run's GPU jobs, ``ahead`` jobs before it."""
    done = [j for j in jobs if j["status"] == "completed"]
    running = sorted(model_of(j) for j in jobs if j["status"] == "in_progress")
    waiting = len(jobs) - len(done) - len(running)
    if not running and not done:
        names = sorted(model_of(j) for j in jobs) or [
            m.strip() for m in models.split(",") if m.strip()
        ]
        will = ", ".join(names)
        n = len(names)
        return f"Waiting for a GPU runner: ~{ahead} jobs ahead. Will run {n}: {will}"
    desc = f"{len(done)}/{len(jobs)} done"
    if running:
        desc += f"; running {', '.join(running)}"
    if waiting:
        desc += f"; {waiting} waiting, ~{ahead} jobs ahead"
    return desc


def current_status(repo, sha):
    """The commit's latest ``AMD PR test`` status, or None."""
    for s in api(f"repos/{repo}/commits/{sha}/statuses?per_page=100"):
        if s["context"] == CONTEXT:
            return s  # newest first
    return None


def pr_sha(repo, run):
    """The PR head the run tests: the event's for a label or push run; a dispatched
    run's is the PR's head now (what its plan job resolved, short of a push since)."""
    if run["event"] == "pull_request_target":
        return run["head_sha"]
    pr = run["display_title"].rsplit("#", 1)[-1].strip()
    if not pr.isdigit():
        return None
    return api(f"repos/{repo}/pulls/{pr}")["head"]["sha"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, help="this PR test run only")
    ap.add_argument("--models", default="", help="what the run will test, for --run")
    ap.add_argument("--dry-run", action="store_true", help="print, set nothing")
    args = ap.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]

    runs = {w: active_runs(repo, w) for w in GPU_WORKFLOWS}
    jobs = {r["id"]: gpu_jobs(repo, r["id"]) for w in GPU_WORKFLOWS for r in runs[w]}
    running = sum(j["status"] == "in_progress" for js in jobs.values() for j in js)
    queued = [
        (r["created_at"], r["id"], sum(j["status"] == "queued" for j in jobs[r["id"]]))
        for w in GPU_WORKFLOWS
        for r in runs[w]
    ]

    targets = [r for r in runs[PR_TEST] if args.run in (None, r["id"])]
    if args.run and not targets:
        # The plan job runs before its own run shows up as in progress in every list.
        targets = [api(f"repos/{repo}/actions/runs/{args.run}")]
        jobs.setdefault(args.run, [])
    for run in targets:
        mine = jobs[run["id"]]
        if mine and all(j["status"] == "completed" for j in mine):
            continue  # the report job sets the final status
        # Jobs of older runs still queued, and every job running now but this run's.
        ahead = sum(
            n for t, rid, n in queued if (t, rid) < (run["created_at"], run["id"])
        )
        ahead += running - sum(j["status"] == "in_progress" for j in mine)
        desc = describe(mine, ahead, args.models)[:140]
        sha = pr_sha(repo, run)
        url = run["html_url"]
        cur = current_status(repo, sha) if sha else None
        if args.run is None and (
            not cur or cur["state"] != "pending" or cur["target_url"] != url
        ):
            continue  # not this run's pending status (finished, or a newer run's)
        if cur and cur["description"] == desc:
            continue
        print(f"run {run['id']} ({run['display_title']}): {desc}")
        if not args.dry_run and sha:
            api(
                f"repos/{repo}/statuses/{sha}",
                "POST",
                {
                    "state": "pending",
                    "context": CONTEXT,
                    "target_url": url,
                    "description": desc,
                },
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
