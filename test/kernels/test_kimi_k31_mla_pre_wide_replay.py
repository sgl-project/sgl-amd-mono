# SPDX-License-Identifier: Apache-2.0
"""Single-GPU numerical and CUDA-graph replay checks for ``mla_pre_wide``."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_BLOCKS = 256
_QB_LOCAL = 2304
_QB_QREP = 18432


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs a GPU"
    if importlib.util.find_spec("flydsl") is None:
        return "needs FlyDSL"
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    if "gfx950" not in arch:
        return f"needs gfx950 (got {arch or 'unknown'})"
    if props.multi_processor_count < _BLOCKS:
        return f"needs >= {_BLOCKS} CUs (got {props.multi_processor_count})"
    return None


@pytest.mark.parametrize(
    ("m", "qb_rows"),
    [
        (16, _QB_LOCAL),
        (32, _QB_LOCAL),
        (33, _QB_LOCAL),
        (48, _QB_LOCAL),
        (16, _QB_QREP),
        (32, _QB_QREP),
        (33, _QB_QREP),
        (36, _QB_QREP),
        (42, _QB_QREP),
        (48, _QB_QREP),
        (64, _QB_QREP),
    ],
)
def test_mla_pre_wide_replay(m: int, qb_rows: int) -> None:
    """Eager + CUDA-graph fresh-input replay matches fp32 reference on all slots."""
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    bench_dir = str(Path(__file__).resolve().parent / "kimi_k31")
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import mla_pre_wide_bench as bench

    assert bench.run_one(m, qb_rows, nl=4, flush_mb=0, show_tl=False, check_only=True)
