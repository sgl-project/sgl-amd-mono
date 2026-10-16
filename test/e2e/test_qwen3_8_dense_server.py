"""Qwen3.8-27B with --monokernel-backend amd on MI35x at TP1 and TP2 (the OOB
recipe), against the same server without it. Decode steps up to the row cap
(TP1 4 rows, TP2 8) run the monokernels, wider ones the normal path. Each
server's startup line shows the capped widths run the monokernels; GSM8K past the
cap scores as the normal path does; covered decode steps match the normal prefill
(KL consistency) as closely as the normal path's own; a request decoding while
the batch steps past the cap and back still matches; an image request decodes;
and decode is no slower than the normal path, inside the cap and past it.

The off server runs first in each class and its numbers are the reference; the
bounds are arbitrary until the first MI355X run measures both paths."""

import base64
import functools
import io
import os
import re
import statistics
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests
from PIL import Image
from qwen3_8_dense_server import ENV, MODEL, ROW_CAP, args
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND

URL = DEFAULT_URL_FOR_TEST
LAUNCH_TIMEOUT = 3600
SPEC = "qwen3_8_dense_hip"
# Arbitrary until measured: gross failures, and how far on may trail off
GSM8K_BOUND = 0.85
GSM8K_DROP = 0.03
# Decode against prefill, mean KL over the prompts. The prompts are a fresh
# random draw per cache, so the bound covers the normal path's spread: it
# measured 0.009 to 0.034 over three draws at TP1 and TP2 on MI355X. The
# monokernels are held to the normal path by KL_MARGIN.
KL_BOUND = 0.06
KL_MARGIN = 0.005
# Arbitrary until measured: on's per-request decode rate over off's, run-to-run noise
SPEED_FLOOR = 0.95
DECODE_TOKENS = 256


def _launch(tp, *, mono, log=None, extra_env=None, extra_args=()):
    backend = ["--monokernel-backend", BACKEND] if mono else []
    return popen_launch_server(
        MODEL,
        URL,
        timeout=LAUNCH_TIMEOUT,
        other_args=args(tp) + backend + ["--watchdog-timeout", "600", *extra_args],
        env={**os.environ, **ENV, **(extra_env or {})},
        return_stdout_stderr=log,
    )


@functools.cache
def _prompts():
    """LongBench-v2 prompts (the KL test's, about 3000 tokens each)."""
    return get_input_ids(tokenizer_path=MODEL, num_samples=16, trust_remote_code=True)


def _generate(*, input_ids, max_new_tokens, logprob_start_len):
    """Greedy /generate of every prompt in one batch, with logprobs; the server
    disables the radix cache, so each prompt prefills from scratch."""
    r = requests.post(
        URL + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": logprob_start_len,
        },
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()


def _rescored_kl(prompts, decoded):
    """Mean over the prompts of the decoded tokens' KL against a prefill of
    prompt and output (``kl_test_utils.compare_kl_divergence``'s estimate)."""
    outputs = [[x[0] for x in r["meta_info"]["output_token_logprobs"]] for r in decoded]
    full = [p + r["output_ids"] for p, r in zip(prompts, decoded)]
    prefill = _generate(input_ids=full, max_new_tokens=0, logprob_start_len=0)
    kls = []
    for r, o in zip(prefill, outputs):
        i = [x[0] for x in r["meta_info"]["input_token_logprobs"]][-len(o) :]
        logr = np.array(i) - np.array(o)
        kls.append(float(np.mean(np.exp(logr) - 1 - logr)))
    print(f"{kls=}")
    return sum(kls) / len(kls)


def _decode_kl(rows):
    """``rows`` prompts decoded together: every decode step is ``rows`` wide."""
    prompts = _prompts()[:rows]
    decoded = _generate(
        input_ids=prompts, max_new_tokens=DECODE_TOKENS, logprob_start_len=-1
    )
    return _rescored_kl(prompts, decoded)


def _gsm8k(parallel):
    metrics = run_eval_few_shot_gsm8k(
        SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            parallel=parallel,
            max_new_tokens=512,
            host="127.0.0.1",
            port=int(URL.split(":")[-1]),
        )
    )
    print(f"{metrics=}")
    return metrics["accuracy"]


def _decode_speed(batch):
    """Per-request decode tokens/s at ``batch`` requests decoding together:
    median of 3 runs after a warmup (the short prompts' prefill is in the time)."""
    body = {
        "text": [f"Story {i}: the lighthouse keeper woke and" for i in range(batch)],
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": DECODE_TOKENS,
            "ignore_eos": True,
        },
    }
    rates = []
    for run in range(4):
        start = time.perf_counter()
        requests.post(URL + "/generate", json=body, timeout=600).raise_for_status()
        if run:
            rates.append(DECODE_TOKENS / (time.perf_counter() - start))
    return statistics.median(rates)


def _speed_batches(cap):
    return (1, cap, 2 * cap)


class _Serving:
    """A TP's off server measured first, then its on server kept for the tests."""

    TP: int

    @classmethod
    def setUpClass(cls):
        cls.cap = ROW_CAP[cls.TP]
        cls.off = cls._measure(mono=False)
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(cls.TP, mono=True, log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    @classmethod
    def _measure(cls, *, mono):
        process = _launch(cls.TP, mono=mono)
        try:
            return {
                "gsm8k": _gsm8k(parallel=2 * cls.cap),
                "kl": {rows: _decode_kl(rows) for rows in (1, cls.cap)},
                "speed": {b: _decode_speed(b) for b in _speed_batches(cls.cap)},
            }
        finally:
            terminate_and_kill_process_tree(process)

    def test_the_capped_batches_run_the_monokernels(self):
        # routing is fixed at startup; without this, the other tests could pass on
        # the normal path alone. The framework routes the captured graphs up to
        # the spec's widths (8, per ISA); the spec runs those up to the cap.
        log = "".join(Path(f.name).read_text() for f in self.logs)
        m = re.search(
            rf"monokernel {SPEC} on gfx950 runs decode steps of batch size \[([\d, ]+)\]",
            log,
        )
        self.assertIsNotNone(m, "no monokernel startup line")
        widths = [int(w) for w in m.group(1).split(",")]
        self.assertEqual((widths[0], widths[-1]), (1, 8), widths)
        self.assertIn(
            f"{SPEC}: TP{self.TP} runs decode steps of up to {self.cap} rows", log
        )

    def test_gsm8k_past_the_cap(self):
        # twice the cap at a time: the batch steps past the cap and back
        acc = _gsm8k(parallel=2 * self.cap)
        print(f"GSM8K TP{self.TP}: on {acc:.3f}, off {self.off['gsm8k']:.3f}")
        self.assertGreater(acc, GSM8K_BOUND)
        self.assertGreaterEqual(acc, self.off["gsm8k"] - GSM8K_DROP)

    def test_covered_steps_match_prefill(self):
        for rows in (1, self.cap):
            kl, off = _decode_kl(rows), self.off["kl"][rows]
            print(f"decode vs prefill KL TP{self.TP} {rows} rows: on {kl}, off {off}")
            with self.subTest(rows=rows):
                self.assertLess(kl, KL_BOUND)
                self.assertLessEqual(kl, off + KL_MARGIN)

    def test_batch_crosses_the_cap_and_back(self):
        # a long request decodes alone (monokernels), then with a burst of
        # cap + 2 short ones (normal path), then alone again
        prompt = _prompts()[0]
        burst = {
            "text": [f"Note {i}: the harbour at dawn was" for i in range(self.cap + 2)],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 128},
        }
        with ThreadPoolExecutor(max_workers=1) as pool:
            long = pool.submit(
                _generate,
                input_ids=[prompt],
                max_new_tokens=1024,
                logprob_start_len=-1,
            )
            time.sleep(3)
            r = requests.post(URL + "/generate", json=burst, timeout=600)
            decoded = long.result()
        r.raise_for_status()
        self.assertTrue(all(x["text"] for x in r.json()))
        kl = _rescored_kl([prompt], decoded)
        print(f"crossing request KL TP{self.TP}: {kl}")
        self.assertLess(kl, KL_BOUND)

    def test_image_request(self):
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (220, 20, 20)).save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        r = requests.post(
            URL + "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {
                                "type": "text",
                                "text": "What color is this image? Answer in one word.",
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
        print(f"image answer: {answer!r}")
        self.assertIn("red", answer.lower())

    def test_decode_speed_inside_and_past_the_cap(self):
        # past the cap the normal path runs: on must not cost there; inside it the
        # gain is printed, not bounded, until measured
        for batch in _speed_batches(self.cap):
            on, off = _decode_speed(batch), self.off["speed"][batch]
            print(f"decode tok/s a request TP{self.TP} batch {batch}: on {on:.1f}, "
                  f"off {off:.1f}, ratio {on / off:.3f}")  # fmt: skip
            with self.subTest(batch=batch):
                self.assertGreaterEqual(on, SPEED_FLOOR * off)


class TestMonokernelTP1(_Serving, CustomTestCase):
    TP = 1


class TestMonokernelTP2(_Serving, CustomTestCase):
    TP = 2


class TestMonokernelCheck(CustomTestCase):
    def test_each_covered_layer_within_its_numerics(self):
        # SGLANG_AMD_MONO_CHECK (eager) runs the normal layer next to each
        # covered layer, at 1 row and at the cap
        for tp, cap in ROW_CAP.items():
            with (
                self.subTest(tp=tp),
                tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
                tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
            ):
                process = _launch(
                    tp,
                    mono=True,
                    log=(out, err),
                    extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                    extra_args=["--disable-cuda-graph"],
                )
                try:
                    for n in (1, cap):
                        r = requests.post(
                            URL + "/generate",
                            json={
                                "text": [f"Record {i}: the tide" for i in range(n)],
                                "sampling_params": {
                                    "max_new_tokens": 4,
                                    "temperature": 0,
                                },
                            },
                            timeout=600,
                        )
                        r.raise_for_status()
                finally:
                    terminate_and_kill_process_tree(process)
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                self.assertTrue(re.search(r"monokernel check layer \d+", log))
                violations = re.findall(r"monokernel check violation: .*", log)
                self.assertEqual(violations, [], "\n".join(violations[:20]))


if __name__ == "__main__":
    unittest.main()
