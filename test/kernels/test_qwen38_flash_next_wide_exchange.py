# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.8-Flash-Next wide decode's TP exchange between two processes on two
gfx950 GPUs, through ``common.peer.PeerBuffer`` (``qwen38_flash_next/check_wide_exchange.py``):
both codecs, every decode width SGLang captures up to 192 rows and 129, 1,000
launches a width, in vLLM's build setting and SGLang's (stop mode, ``grid``). Every
launch's new state equals the host reference bit for bit on both ranks. Three
controls break one launch on rank 1 in vLLM's setting and check that the matching
check fires: a lost publish (rank 0's wait gives up at ``abi.WAIT_TICKS`` and the
launch ends), a corrupt input (the reference check) and different injection scales
(the cross-rank digest). In SGLang's setting the fault injection stalls one rank's
wait: its deadline faults it at that wait's page, and its flag faults the other rank
at the flag's page before that rank's own waits give up.

The harness spawns and waits for each rank itself, so a rank that never stops is
seen and killed at the hang limit instead of hanging the test."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).with_name("qwen38_flash_next") / "check_wide_exchange.py"
# SGLang's decode CUDA graph capture sizes up to 192 rows without speculative
# decoding (generate_decode_cuda_graph_batch_sizes), and 129: a partial line
WIDTHS = sorted({1, 2, 4, 8, 12, *range(16, 193, 8), 129})
ITERS = 1000
CONTROL_ROWS, CONTROL_ITER = 129, 10
WAIT_S = 10  # abi.WAIT_SECONDS
SITE_GRID, SITE_PEER, SITE_FLAG = 1, 2, 4  # abi's; in stop mode the fault's page
GRACE = 1 / 8  # grid.GRACES: a grid barrier gives up a grace after the deadline
# the fault test: the stalled rank's deadline (SGLang's least: half of the 10 s
# `--watchdog-timeout` it accepts) and the other rank's (SGLang's default), and
# the worker kernel's waits, stalled once the epoch reaches 2 (launch LAYERS,
# after a whole step): its first, the attention lines in `hc.combine`, which only
# the CTAs with a row take (the rest wait at the grid barrier after it, which gives
# up a grace later), or its second, that grid barrier, which every CTA takes
FAULT_WAIT_S, OTHER_WAIT_S = 5.0, 60.0
FAULT_ROWS, FAULT_AFTER, LAYERS = 129, 2, 48
LINES, BARRIER = (1, SITE_PEER), (2, SITE_GRID)  # (worker wait, its page's site)
# a GPU memory fault as the runtime reports it: ROCr's "Memory access fault by GPU
# node-%u (Agent handle: %p) on address %p%s. Reason: %s." (glibc's %p of 0 is
# "(nil)"), or CLR's "[host: ..., GPU index: %u, faulting addr: 0x%x, kernel: ...]"
# after its log prefix ":<level>:<file>:<line>: <us> us:", whose stamp is
# CLOCK_MONOTONIC (``amd::Os::timeNanos``), the clock of the progress files
ROCR_FAULT = re.compile(
    r"Memory access fault by GPU node-\d+ .*? on address (\(nil\)|0x[0-9a-fA-F]+)"
)
CLR_FAULT = re.compile(
    r"^:\d+:[^:\n]*:\s*\d+\s*:\s*(\d+) us:[^\n]*?"
    r"\[(?:host: [^,\]\n]*, )?GPU index: \d+, faulting addr: (0x[0-9a-fA-F]+)",
    re.M,
)


def faults(log: str) -> list[tuple[int, float | None]]:
    """Every GPU memory fault `log` reports: (address, monotonic s when CLR
    reported it; None for ROCr's report, which has no time)."""
    out = [(0 if a == "(nil)" else int(a, 16), None) for a in ROCR_FAULT.findall(log)]
    return out + [(int(a, 16), int(us) / 1e6) for us, a in CLR_FAULT.findall(log)]


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


pytestmark = pytest.mark.skipif(_gfx950_gpus() < 2, reason="needs 2 gfx950 GPUs")


def _start(tmp_path, args, rank_args=((), ()), env=None):
    """Start both ranks (`rank_args`: each rank's own arguments)."""
    procs = []
    for r in range(2):
        renv = {**os.environ, **(env or {})}
        renv["FLYDSL_RUNTIME_CACHE_DIR"] = str(tmp_path / f"flydsl{r}")
        with open(tmp_path / f"rank{r}.log", "w") as log:
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(WORKER), "--rank", str(r),
                     "--init-file", str(tmp_path / "init"),
                     "--out", str(tmp_path / f"rank{r}.json"), *args, *rank_args[r]],
                    env=renv, stdout=log, stderr=subprocess.STDOUT,
                )
            )  # fmt: skip
    return procs


def _wait(procs, hang_s):
    """Each rank's exit time (monotonic s, to 50 ms); a rank still running at
    `hang_s` is killed and fails the test."""
    t0, ends = time.monotonic(), [None] * len(procs)
    while None in ends and time.monotonic() - t0 < hang_s:
        for r, p in enumerate(procs):
            if ends[r] is None and p.poll() is not None:
                ends[r] = time.monotonic()
        time.sleep(0.05)
    hung = [r for r, e in enumerate(ends) if e is None]
    for r in hung:
        procs[r].kill()
        procs[r].wait()
    assert not hung, f"rank(s) {hung} still running after {hang_s} s"
    return ends


def _spawn(tmp_path, args, hang_s):
    """Run both ranks; return each rank's results."""
    procs = _start(tmp_path, args)
    _wait(procs, hang_s)
    for r, p in enumerate(procs):
        log = (tmp_path / f"rank{r}.log").read_text()[-3000:]
        assert p.returncode == 0, f"rank {r} exited {p.returncode}:\n{log}"
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(2)]


def _control(tmp_path, control, codec="bf16"):
    out = _spawn(
        tmp_path,
        ["--codecs", codec, "--widths", str(CONTROL_ROWS),
         "--iters", str(CONTROL_ITER + 1), "--control", control,
         "--control-iter", str(CONTROL_ITER)],
        hang_s=1200,
    )  # fmt: skip
    return [o["records"][0] for o in out]


@pytest.mark.parametrize("setting", ["vllm", "sglang"])
def test_exchange_every_width_bit_equal(tmp_path, setting):
    out = _spawn(
        tmp_path,
        ["--widths", ",".join(map(str, WIDTHS)), "--iters", str(ITERS),
         "--setting", setting],
        hang_s=3600,
    )  # fmt: skip
    for o in out:
        assert [(r["codec"], r["rows"]) for r in o["records"]] == [
            (c, w) for c in ("bf16", "int4") for w in WIDTHS
        ]
        bad = [r for r in o["records"] if not r["pass"] or r["iters"] != ITERS]
        assert not bad, f"rank {o['rank']}: {json.dumps(bad, indent=1)}"


def test_lost_publish_gives_up_at_the_deadline(tmp_path):
    """Rank 1 publishes its lines to its own mailbox only: rank 0's waits on them
    give up after WAIT_SECONDS (its grid barrier's waits may too), record the peer
    site, raise its alarm and end the launch, whose state then differs from the
    reference."""
    r0, _ = _control(tmp_path, "lost_publish")
    assert r0["failed"] == 1
    assert str(SITE_PEER) in r0["failed_record"]["sites"]
    assert r0["alarm_words"][0] == 1
    assert r0["launch_ms_control"] >= 0.95 * WAIT_S * 1000
    assert r0["mismatch_at_control"] > 0


@pytest.mark.parametrize("codec", ["bf16", "int4"])
def test_corrupt_input_fails_the_reference(tmp_path, codec):
    """Rank 1's attention partial is not the reference's: both ranks combine the
    same lines, so their states agree with each other and not with the reference."""
    for r in _control(tmp_path, "corrupt_input", codec):
        assert r["failed"] == 0
        assert r["mismatch_iters"] == 1 and r["mismatch_at_control"] > 0
        assert r["digest_differ_iters"] == 0


def test_rank_mismatch_fails_the_digest(tmp_path):
    """Rank 1's injection scales differ from rank 0's: the ranks' states differ, which
    the cross-rank digest sees; rank 0 still matches the reference."""
    r0, r1 = _control(tmp_path, "rank_mismatch")
    for r in (r0, r1):
        assert r["failed"] == 0
        assert r["digest_differ_iters"] == 1 and r["digest_differs_at_control"]
    assert r0["mismatch_iters"] == 0
    assert r1["mismatch_iters"] == 1 and r1["mismatch_at_control"] > 0


@pytest.mark.parametrize(
    "stalled, wait, other_wait_s",
    [(1, BARRIER, OTHER_WAIT_S), (0, LINES, OTHER_WAIT_S), (1, BARRIER, FAULT_WAIT_S)],
    ids=["barrier", "lines", "barrier-same-deadline"],
)
def test_a_stalled_rank_faults_both_ranks_by_its_deadline(
    tmp_path, stalled, wait, other_wait_s
):
    """SGLang's setting, a wait of rank `stalled`'s launch LAYERS stalled
    (SGLANG_AMD_MONO_FAULT_INJECT): after the first step on both ranks, its
    deadline (FAULT_WAIT_S; a grid barrier's a grace later) faults it at that
    wait's page, not at a wait behind it, and the flag it raised in the other
    rank's peer buffer faults the other rank (waiting on its lines) at the flag's
    page: long before that rank's own OTHER_WAIT_S deadline, or with the same
    deadline before its waits on the stalled rank's lines give up two graces
    later. A stalled rank that raised its own flag too could fault at the flag's
    page first: a barrier case catches that about 2 runs in 3 (4 of 6 runs, before
    the staggered deadlines; 0 of 6 at the lines), so the two miss it about 11 % of
    the time."""
    other = 1 - stalled
    waits = [other_wait_s, other_wait_s]
    waits[stalled] = FAULT_WAIT_S
    site, kind = wait
    pages = {stalled: kind << 12, other: SITE_FLAG << 12}
    deadline_s = FAULT_WAIT_S * (1 + GRACE if kind == SITE_GRID else 1)
    procs = _start(
        tmp_path,
        ["--setting", "sglang", "--codecs", "bf16", "--widths", str(FAULT_ROWS),
         "--iters", str(2 * LAYERS)],
        rank_args=[("--wait-s", str(w), "--progress", str(tmp_path / f"progress{r}"))
                   for r, w in enumerate(waits)],
        env={"SGLANG_AMD_MONO_FAULT_INJECT": f"{stalled}:{site}:{FAULT_AFTER}"},
    )  # fmt: skip
    ends = _wait(procs, hang_s=600)
    last, stamps = [], []
    for r, p in enumerate(procs):
        log = (tmp_path / f"rank{r}.log").read_text()
        found = faults(log)
        # only stop_launch's faults, at the page of the rank's wait kind
        assert p.returncode != 0 and found, (
            f"rank {r} exited {p.returncode}, no fault:\n{log[-3000:]}"
        )
        assert all(a == pages[r] for a, _ in found), (r, hex(pages[r]), found)
        # every launch of the first step returned, and none after it
        rows = (tmp_path / f"progress{r}").read_text().split()
        assert [int(i) for i in rows[::2]] == list(range(LAYERS)), rows[::2]
        last.append(float(rows[-1]))
        stamps.append(min((t for _, t in found if t is not None), default=None))
    # the stalled rank: its deadline, from its stalled launch's start, then the
    # runtime's teardown
    took = ends[stalled] - last[stalled]
    assert 0.95 * deadline_s <= took <= deadline_s + 15, took
    # the other rank: the stalled rank's flag, not its own deadline
    if other_wait_s == OTHER_WAIT_S:
        assert ends[other] - last[other] < OTHER_WAIT_S / 2, ends[other] - last[other]
    assert abs(ends[other] - ends[stalled]) < 15, ends
    if None not in stamps:
        # CLR's fault times: the stalled rank at its deadline, the other with it
        took = stamps[stalled] - last[stalled]
        assert 0.95 * deadline_s <= took <= deadline_s + 1, took
        assert abs(stamps[other] - stamps[stalled]) < 1, stamps
