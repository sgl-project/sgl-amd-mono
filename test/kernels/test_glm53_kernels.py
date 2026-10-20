"""Synthetic GLM-5.3 layer equivalence through the plugin's actual runner.

No checkpoint is needed. Serving/reference accuracy lives in test/e2e.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

CHECK = Path(__file__).with_name("glm53") / "check_indexer_cp.py"


def _gfx950_gpus():
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


@pytest.mark.parametrize(
    "tp,rows,capacity,ties",
    [
        (4, 2, 64, False),
        (4, 4, 4096, False),
        (4, 4, 4096, True),
        (4, 2, 16448, False),
        (4, 2, 114752, False),
        (4, 4, 131072, False),
        (4, 2, 131072, False),
        (4, 2, 131072, True),
        (4, 2, 1048576, False),
        (8, 8, 4096, False),
        (8, 8, 131072, False),
        (8, 2, 131072, False),
    ],
)
def test_indexer_cp(tp, rows, capacity, ties):
    _check(tp, rows, capacity, ties)


@pytest.mark.parametrize("rows", [2, 8])
def test_indexer_cp_full_context(rows):
    _check(8, rows, 131072, False, pad_rows=0, full_context=True)


def _check(tp, rows, capacity, ties, pad_rows=1, full_context=False):
    if _gfx950_gpus() < tp:
        pytest.skip(f"needs {tp} gfx950 GPUs")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={tp}",
        str(CHECK),
        "--rows",
        str(rows),
        "--capacity",
        str(capacity),
        "--pad-rows",
        str(pad_rows),
    ]
    if full_context:
        cmd.append("--full-context")
    if ties:
        cmd.append("--ties")
    subprocess.run(cmd, check=True, timeout=1800)
