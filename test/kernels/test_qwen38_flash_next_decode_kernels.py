"""The Qwen3.8-Flash-Next TP2 decode launches on random weights in SGLang's post-load
layouts, under torchrun at TP2 (``qwen38_flash_next_decode/check_blocks.py``): the HC
front against SGLang's ``GatedResidual.mix``; the GDN block against AITER's GDN decode ops
and SGLang's gated norm, pad rows writing no state; out_proj, o_proj and the MoE
(the experts padded as SGLang pads them, and at the kernels' own padding) with the
all-reduce and the HC combine; a rank whose peer never launches failing its wait at
the guard's deadline and flagging its peers; and two layers built from SGLang modules by ``weights.build_layers``,
run by ``runner.Launcher`` at every width 1 to 16, rerun and replayed from a graph
bitwise."""

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


CHECK = Path(__file__).with_name("qwen38_flash_next_decode") / "check_blocks.py"


@unittest.skipIf(_gfx950_gpus() < 2, "needs 2 gfx950 GPUs")
class TestMonoQwen38FlashNextKernels(CustomTestCase):
    def _run(self, kind: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            "--master-port=29652",
            str(CHECK),
            "--kind",
            kind,
        ]
        result = subprocess.run(cmd, timeout=3600)
        self.assertEqual(result.returncode, 0, kind)

    def test_hc_front_matches_sglang_mix(self):
        self._run("front")

    def test_gdn_block_matches_the_gdn_decode(self):
        self._run("gdn")

    def test_oproj_and_moe_blocks_match_torch(self):
        self._run("blocks")

    def test_a_stalled_wait_fails_at_its_deadline_and_flags_the_peers(self):
        self._run("stall")

    def test_every_width_reruns_and_replays_bitwise(self):
        self._run("steps")


if __name__ == "__main__":
    unittest.main()
