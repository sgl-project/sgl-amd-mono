"""The MiniMax-M3.1-Flash DSpark draft's gfx950 launches against SGLang's arithmetic.

``minimax_m31/check_draft.py`` (the five draft layers in one launch, against an f32
reference) and ``minimax_m31/check_markov.py`` (the sampled Markov block with bf16 step
logits) under torchrun at TP4: outputs within the bounds, bitwise graph replays with every
rank agreeing."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


CHECKS = Path(__file__).with_name("minimax_m31")


@unittest.skipIf(_gfx950_gpus() < 4, "needs 4 gfx950 GPUs")
class TestMonoMinimaxM31Draft(CustomTestCase):
    def _check(self, module: str, *args: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=4",
            "--master-port=29631",
            str(CHECKS / f"{module}.py"),
            *args,
        ]
        result = subprocess.run(cmd, timeout=1800)
        self.assertEqual(result.returncode, 0, f"{module} at TP4")

    def test_draft_launch(self):
        self._check("check_draft")

    def test_sampled_markov_block(self):
        self._check("check_markov")


if __name__ == "__main__":
    unittest.main()
