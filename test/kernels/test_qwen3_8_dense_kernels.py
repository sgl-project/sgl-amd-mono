"""The Qwen3.8-27B dense gfx950 monokernels bit for bit against SGLang's stock
decode ops, each check on one GPU (``qwen3_8_dense/check_*.py``): K1's GDN
front (``dense_gdn_pre``) and K2's FFN (``dense_ffn``) at TP1 and at a TP2
rank's shard shapes, widths 1..max_tokens(tp); and the stock GEMM properties K2
relies on."""

import importlib.util
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


CHECKS = Path(__file__).with_name("qwen3_8_dense")


@unittest.skipIf(
    _gfx950_gpus() < 1 or importlib.util.find_spec("flydsl") is None,
    "needs a gfx950 GPU and FlyDSL",
)
class TestMonoQwen38DenseKernels(CustomTestCase):
    def _check(self, module: str, *args: str) -> None:
        cmd = [sys.executable, str(CHECKS / f"{module}.py"), *args]
        result = subprocess.run(cmd, timeout=3600)
        self.assertEqual(result.returncode, 0, f"{module} {' '.join(args)}")

    def test_stock_gemms(self):
        self._check("check_k2", "--stock")

    def test_k2_tp1(self):
        self._check("check_k2", "--tp", "1")

    def test_k2_tp2(self):
        self._check("check_k2", "--tp", "2")

    def test_k1_tp1(self):
        self._check("check_k1", "--tp", "1")

    def test_k1_tp2(self):
        self._check("check_k1", "--tp", "2")


if __name__ == "__main__":
    unittest.main()
