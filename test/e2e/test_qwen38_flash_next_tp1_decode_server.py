"""Qwen3.8-Flash-Next (the MXFP4 checkpoint) with --monokernel-backend amd on one
MI355X, TP1 decode of 1 to 64 rows (``TP1_ARGS``): the server runs at most 64
requests, so every decode step is one the wide launches take. The startup line shows
every captured decode graph runs them. Against the normal path on the same server
arguments: GSM8K, with requests joining and leaving batches of up to 64 rows, and per
prompt the KL of decode against prefill (the tokens of covered steps scored again by
the prefill path) at 1 to 64 rows. The second turn of a conversation decoding from
the GDN states the radix cache saved during the first; each layer's launches against
SGLang's own layer under SGLANG_AMD_MONO_CHECK, at 1 to 64 rows and at contexts past
QSA's 2,048-token selection budget (about 6,000 and 108,000 tokens); and the rank
stalled by SGLANG_AMD_MONO_FAULT_INJECT stopping the server without tokens at its own
deadline. The TP2 tests are test_qwen38_flash_next_decode_server.py."""

import json
import os
import re
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from qwen38_flash_next_decode_server import ENV, MODEL, TP1_ARGS
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper as assert_decode_hits_match,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from test_qwen38_flash_next_decode_server import _kl_per_prompt

from sgl_amd_mono import BACKEND
from sgl_amd_mono.qwen38_flash_next.spec import SPEC

# a cold FlyDSL cache compiles every width's launches at startup
LAUNCH_TIMEOUT = 3600
# The decode graphs: the spec's widths up to 64, the largest capture size; SGLang's
# default sizes up to 64 are among them, and 20 and 28 are the spec's
WIDTHS = sorted(w for w in SPEC.devices["gfx950"].widths if w <= 64)
# SGLang's startup line when every captured decode graph runs the launches: routing
# is fixed at startup, so without it a test could pass on the normal path alone
ROUTING = (
    "monokernel qwen38_flash_next_hip on gfx950 runs decode steps of batch "
    f"size {WIDTHS} ({len(WIDTHS)} of {len(WIDTHS)} captured decode graphs)"
)
# A GSM8K score drop against the normal path that fails the run: a bound for gross
# failures
GSM8K_DROP = 0.03
# KL of decode against prefill per prompt (the TP2 file's _kl_per_prompt, by the steps
# of SGLang's kl_test_utils), one batch of the first `rows` LongBench-v2 prompts at a
# time, 512 output tokens each, on each path's server: one graph per batch, so every
# decode step covered; 3, 13, 17 and 33 rows are one past a graph's, so their steps
# carry pad rows. 209 prompt runs in all, over the first 64 prompts.
KL_ROWS = (1, 2, 3, 12, 13, 16, 17, 33, 48, 64)
# The mean over them of (monokernels - normal path), the same prompts on both: 0.0085
# to 0.0231 (mean 0.0157, sd 0.0037) over 36 pairs of rounds on two MI355X GPUs, 3
# rounds a path in each of 4 client processes (2 a GPU), each with its own prompt
# truncations; pairs within a process only, two rounds of one path in a process at
# most 0.0103 apart. Per process the mean was 0.0144 to 0.0173, and the pair of its
# first rounds (the test runs one round a path, after GSM8K) 0.0130 to 0.0229. The
# limit is the largest multiple of 0.005 up to the mean + 0.02 (0.0357): added to
# every pair, +0.02 reaches it in 18 of 36 and +0.025 in 35. The mean weighs a batch
# by its rows: the 1- to 3-row batches hold 6 of the 209 prompt runs, so a change
# confined to them reaches the limit at about +0.67 a prompt.
KL_MARGIN = 0.035
# The second turn's mean KL against prefill (8 prompts, on the monokernel server):
# 0.024 on one MI350X and 0.022 on one MI355X (per prompt 0.001 to 0.104). The bound
# was set over the largest monokernel batch mean of the KL batches on one MI350X
# (0.113, 3 rounds); in the MI355X calibration one monokernel batch of 120 (13 rows)
# reached 0.145, from one prompt at 0.877
KL_BOUND = 0.14
# 72 requests at a time, more than the server runs: its batches fill to the largest
# width, 64 rows
THREADS = 72
# Check mode: batches of (rows, paragraphs of 12 tokens): one a width from 1 row to
# the largest at about 55 tokens, then contexts QSA's indexer selects from (past its
# budget of 2,048 tokens): about 6,000 tokens at 1, 16 and 64 rows, and about 108,000
# at 1 and 4. SGLang prefills a batch's requests before it decodes any, so each
# batch decodes at its rows
CHECK_BATCHES = (
    *((rows, 4) for rows in (1, 4, 16, 32, 64)),
    *((rows, 500) for rows in (1, 16, 64)),
    *((rows, 9000) for rows in (1, 4)),
)
PARAGRAPH = "The river valley archive lists the flood height of every year. "
CHECK_METRICS = (
    "gdn.streams.cosine",
    "gdn.delta.cosine",
    "gdn.conv.cosine",
    "gdn.state.cosine",
    "gdn.update.cosine",
    "qsa.streams.cosine",
    "qsa.delta.cosine",
    "qsa.mix.cosine",
    "qsa.qkv.cosine",
    "qsa.ix.cosine",
)
# The fault test: the rank's launches stall at their second wait once the step epoch
# reaches 1000 (startup's graph warmups run a few steps of each width). The first is
# layer 0's, a stats build, whose waits at the request's 1 row run (trace order) the
# stats phase's grid barrier, then the attention mix's split-K partials, a tagged
# hand-off (`hc._down_unit`'s `tagged.poll`). The rank faults at the hand-off's page
# (its grid barriers give up a grace later); there is no peer to flag
FAULT_INJECT = "0:2:1000"
FAULT_PAGES = {0x3000}


def _launch(mono=True, extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    mono_args = ["--monokernel-backend", BACKEND] if mono else []
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=TP1_ARGS + mono_args + list(extra_args),
        env=env,
        return_stdout_stderr=log,
    )


def _logs(files) -> str:
    return "".join(Path(f.name).read_text() for f in files)


def _gsm8k() -> float:
    return run_sgl_eval(
        SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=MODEL,
            eval_name="gsm8k",
            num_examples=400,
            num_threads=THREADS,
            temperature=0.0,
            seed=42,
        )
    )["score"]


class TestMonokernelAccuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.gsm8k, cls.kl, cls.routed = {}, {}, {}
        for mono in (False, True):
            with (
                tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
                tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
            ):
                process = _launch(mono=mono, log=(out, err))
                try:
                    cls.gsm8k[mono] = _gsm8k()
                    cls.kl[mono] = _kl_per_prompt(KL_ROWS)
                finally:
                    terminate_and_kill_process_tree(process)
                cls.routed[mono] = ROUTING in _logs((out, err))
        print(f"GSM8K: normal path {cls.gsm8k[False]}, monokernels {cls.gsm8k[True]}")
        print(
            "KL per prompt:",
            json.dumps({"normal": cls.kl[False], "mono": cls.kl[True]}),
        )

    def test_gsm8k_against_the_normal_path(self):
        drop = self.gsm8k[False] - self.gsm8k[True]
        self.assertLess(drop, GSM8K_DROP, self.gsm8k)

    def test_kl_against_the_normal_path(self):
        # the monokernel server's decode graphs ran the launches, the other's not
        self.assertEqual(self.routed, {False: False, True: True})
        # every prompt scored on both paths, paired
        self.assertEqual([len(self.kl[m]) for m in (False, True)], [sum(KL_ROWS)] * 2)
        diff = statistics.mean(
            m - n for m, n in zip(self.kl[True], self.kl[False], strict=True)
        )
        print(f"KL per prompt, mean of monokernels - normal path: {diff:.4f}")
        self.assertLess(diff, KL_MARGIN)


class TestMonokernelDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    def test_a_every_decode_graph_runs_the_monokernels(self):
        # TP1 builds every captured width
        log = _logs(self.logs)
        self.assertIn(ROUTING, log)
        self.assertIn(f"monokernel: TP1, {4 * len(WIDTHS)} builds for widths", log)
        # SGLang's line counts the widths routed to the plugin, and a routed width TP1
        # does not build runs SGLang's forward: each width logs its first step on the
        # launches, and the build line leaves no width to SGLang's forward
        for w in WIDTHS:
            self.assertIn(f"monokernel: first decode step of {w} rows", log)
        self.assertNotIn("run SGLang's forward", log)

    def test_b_a_second_turn_decodes_from_the_saved_gdn_states(self):
        # the second turn hits the first turn's decoded tokens in the radix cache:
        # its GDN states are the ones the launches' steps saved through SGLang's
        # track-save; a state saved before its layer's launch, or none, fails the KL
        assert_decode_hits_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=8,
            max_new_tokens=512,
        )


class TestMonokernelCheck(CustomTestCase):
    def test_each_covered_layer_within_its_numerics(self):
        # SGLANG_AMD_MONO_CHECK (eager) runs SGLang's layer next to each layer's
        # launches; with the MXFP4 reference the GDN projections are the launches'
        # arithmetic, so the bounds hold the kernels' own difference. One batch at a
        # time (ignore_eos: a request that stops at its first token never decodes);
        # the radix cache on, so the GDN state saves are compared where a step
        # tracks one (gdn.track)
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={
                    "SGLANG_AMD_MONO_CHECK": "1",
                    "SGLANG_AMD_MONO_CHECK_REF": "mxfp4",
                },
                extra_args=["--disable-cuda-graph"],
                log=(out, err),
            )
            try:
                for rows, paragraphs in CHECK_BATCHES:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [
                                f"Record {i}. " + PARAGRAPH * paragraphs
                                for i in range(rows)
                            ],
                            "sampling_params": {
                                "max_new_tokens": 16,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=1800,
                    )
                    r.raise_for_status()
            finally:
                terminate_and_kill_process_tree(process)
            log = _logs((out, err))
        # the build's lines: its widths and the weight copies (the GDN projections'
        # MXFP4 copies among them)
        print(
            *re.findall(
                r".*monokernel(?: qwen38_flash_next_hip on gfx950:|: TP).*", log
            ),
            sep="\n",
        )
        steps = [
            (int(rows), int(first), int(last), fields)
            for rows, first, last, fields in re.findall(
                r"monokernel check step \((\d+) rows, positions (\d+)\.\.(\d+)\): (.*)",
                log,
            )
        ]
        values = {}
        for _, first, _, fields in steps:
            context = "long" if first >= 2048 else "short"
            for name, v in re.findall(r"(\S+)=(\S+)@\d+", fields):
                values.setdefault(name, []).append(float(v))
                values.setdefault(f"{name} ({context})", []).append(float(v))
        for name, vs in sorted(values.items()):
            print(f"{name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} steps)")
        # each batch's steps ran at its rows and its context: a prompt's decode
        # positions start past its paragraphs, by the record number's few tokens
        for rows, paragraphs in CHECK_BATCHES:
            low, high = 12 * paragraphs, 12 * paragraphs + 64
            self.assertTrue(
                any(r == rows and low <= first <= last <= high for r, first, last, _ in steps),
                (rows, paragraphs, sorted({(r, f) for r, f, _, _ in steps})),
            )  # fmt: skip
        for metric in CHECK_METRICS:
            self.assertIn(metric, values)
        violations = re.findall(r"monokernel check violation: .*", log)
        self.assertEqual(violations, [], "\n".join(violations[:20]))


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # the rank stalls at a tagged hand-off (FAULT_INJECT); at the deadline
        # (watchdog / 2) it faults its launch at FAULT_PAGES and the server stops
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={
                    "SGLANG_AMD_MONO_FAULT_INJECT": FAULT_INJECT,
                    # the stop faults the GPU on purpose: no crash dumps, which
                    # would hold the rank up for minutes
                    "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
                    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
                },
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 2048,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=600,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                log = _logs((out, err))
                # a launch stops with a store to its wait kind's page (CLR's
                # "faulting addr: 0x...", ROCr's "on address 0x...")
                faults = {
                    0 if a == "(nil)" else int(a, 16)
                    for a in re.findall(
                        r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))",
                        log,
                    )
                }
                self.assertEqual(faults, FAULT_PAGES, log[-2000:])
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
