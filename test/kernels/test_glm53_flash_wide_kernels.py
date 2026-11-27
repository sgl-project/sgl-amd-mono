# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's wide verify launch (``glm53_flash.gfx950.wide``) against torch and
the standalone stages, at the request counts of the smallest, a middle and the
largest MoE order: the layer launch at TP4 on its own and after a run's first layer
(``--t1``: the previous MoE's all-reduce and mHC write-back), a chain of launches
eager and replayed in a HIP graph, and each stage alone on one GPU (the mHC pre
slices, the KDA verify task, milestone 1a, the direct-to-LDS tile GEMM). Each script
prints PASS or FAIL (``glm53_flash_wide/*.py`` say what each compares)."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase

from sgl_amd_mono.glm53_flash import wide

DIR = Path(__file__).with_name("glm53_flash_wide")
REQS = ",".join(str(r) for r in (min(wide.REQ_BUCKETS), 16, max(wide.REQ_BUCKETS)))


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


@unittest.skipIf(_gfx950_gpus() < wide.TP, f"needs {wide.TP} gfx950 GPUs")
class TestMonoGlm53FlashWideKernels(CustomTestCase):
    def _run(self, script: str, *args: str, tp: int = 1) -> None:
        cmd = [sys.executable]
        if tp > 1:
            cmd += [
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={tp}",
                "--master-port=29640",
            ]
        cmd += [str(DIR / script), *args]
        result = subprocess.run(cmd, timeout=1800, capture_output=True, text=True)
        out = result.stdout + result.stderr
        what = f"{script} {' '.join(args)}"
        self.assertEqual(result.returncode, 0, f"{what}:\n{out[-4000:]}")
        self.assertIn("PASS", result.stdout, f"{what}:\n{out[-4000:]}")
        self.assertNotIn("FAIL", result.stdout, f"{what}:\n{out[-4000:]}")

    def test_layer_launch(self):
        self._run("layer_bench.py", "--reqs", REQS, tp=wide.TP)

    def test_layer_launch_after_a_runs_first_layer(self):
        self._run("layer_bench.py", "--reqs", REQS, "--t1", tp=wide.TP)

    def test_chain_eager_and_replayed(self):
        self._run("chain_bench.py", "--reqs", REQS, tp=wide.TP)

    def test_mhc_pre_stages(self):
        rows = ",".join(
            str(r * wide.SUPPORTED_L[0]) for r in (min(wide.REQ_BUCKETS), 16)
        )
        self._run("mhc_bench.py", "--rows", rows)

    def test_kda_verify_task(self):
        self._run("kda_bench.py", "--reqs", REQS)

    def test_milestone_1a(self):
        self._run("m1a_bench.py", "--reqs", REQS)

    def test_tile_gemm_stage(self):
        self._run("dma_bench.py")


if __name__ == "__main__":
    unittest.main()
