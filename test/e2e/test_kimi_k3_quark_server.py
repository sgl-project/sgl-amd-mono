"""Kimi-K3's serving checks on the Quark checkpoint (amd/Kimi-K3-Quark-MXFP4-AttnFP8),
with the bf16 KDA state and the ReplaySSM ring: kimi_k3_quark_server.

moonshotai/Kimi-K3 reaches none of these paths: its attention projections are
bf16, and kimi_k3_server keeps the fp32 state and per-token snapshots. Here the
launches read the Quark FP8 in-projection and o_proj, and K1 writes the ring
rows SGLang's commit replays. K1 reads the FP8 in-projection only as SGLang's
merged fused_qkvg_proj (sgl-project/sglang#44362); without it K1 is not built,
and test_k1_runs_the_fp8_in_projection says so. The fault test is not repeated;
the guard does not depend on the checkpoint.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
import test_kimi_k3_server as k3
from kimi_k3_quark_server import ARGS, ENV, MODEL
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_prefill_cache_hit_helper as assert_prefill_hit_match,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

# K1's startup and first-step lines (sgl_amd_mono.kimi_k3.gfx950.k1._say).
K1_ENGAGED = "Kimi-K3 K1: engaged on a decode step (fp8 in_proj)"
K1_NOT_BUILT = "Kimi-K3 K1: not built for the KDA layers"
# sgl-eval GSM8K (200 questions) scored 0.98 with the monokernels on; the bound
# is for gross failures, as in test_kimi_k3_server.
GSM8K_BOUND = 0.95
# Mean KL of 48 prompts (the helper's default) x 512 tokens decoded after a
# radix-cache prefix hit, against prefill: 0.0117 measured at TP8; about 2.5x that.
KL_CACHE_HIT_BOUND = 0.03


class TestMonokernelKimiK3Quark(k3.TestMonokernelKimiK3):
    recipe = (MODEL, ENV, ARGS)

    def _log(self) -> str:
        return "".join(Path(f.name).read_text() for f in self.logs)

    def test_k1_runs_the_fp8_in_projection(self):
        # K2's routing line alone passes while K1 declines every KDA layer.
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={"text": "1 + 1 =", "sampling_params": {"max_new_tokens": 16}},
            timeout=600,
        ).raise_for_status()
        log = self._log()
        self.assertNotIn(K1_NOT_BUILT, log)
        self.assertIn(K1_ENGAGED, log)

    def test_gsm8k_across_the_switch(self):
        # sgl-eval with the model's sampling (temperature 1.0, top_p 0.95,
        # thinking), capped at 16k tokens an answer to fit a CI job.
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                model=MODEL,
                eval_name="gsm8k",
                num_examples=200,
                num_threads=2,
                max_tokens=16384,
                temperature=1.0,
                top_p=0.95,
                seed=42,
                sgl_eval_thinking=True,
            )
        )
        self.assertGreater(metrics["score"], GSM8K_BOUND, metrics)

    def test_steps_after_a_cached_prefix_match_prefill(self):
        # The bf16 state is restored from the radix cache here, which the parent's
        # cache-flushing KL check never reaches. The decode-cache-hit helper is
        # not used: at --mamba-track-interval 1024 under half its turns hit.
        info = requests.get(DEFAULT_URL_FOR_TEST + "/server_info", timeout=60).json()
        # the helper returns without scoring when the radix cache is off
        self.assertFalse(info["disable_radix_cache"])
        assert_prefill_hit_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_CACHE_HIT_BOUND}},
            MODEL,
            max_new_tokens=512,
            trust_remote_code=True,
        )


if __name__ == "__main__":
    unittest.main()
