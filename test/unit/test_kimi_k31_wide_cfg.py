# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 wide launch tile configs keep every stage of a launch in one BLOCKS round."""

import math

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k3.gfx950.attention.kda import NPROJ
from sgl_amd_mono.kimi_k31 import wide


def _tasks(cfg: str, rows: int, cols: int, split_k: bool = True) -> int:
    # gfx950/layer.py: row tiles x column tiles (x split-K slices, except shared down)
    bm, bn, _, _, ks = (int(x) for x in cfg.split(":"))
    return math.ceil(rows / bm) * math.ceil(cols / bn) * (ks if split_k else 1)


def test_every_row_count_fits_one_round() -> None:
    """A front or in_proj tile choice with more tasks than BLOCKS costs a whole second round;
    the row ranges keep both in one round at every row count, not only today's buckets."""
    for b in range(wide.NARROW + 1, max(wide.CFGS) + 1):
        _, inp, _, front, _ = wide._row_cfgs(b)
        assert _tasks(cfg=front, rows=b, cols=wide.NF) <= BLOCKS, (b, front)
        assert _tasks(cfg=inp, rows=b, cols=NPROJ) <= BLOCKS, (b, inp)


@pytest.mark.parametrize("L", wide.SUPPORTED_L)
def test_every_bucket_fits_one_round(L: int) -> None:
    for r in wide.REQ_BUCKETS:
        b = L * r
        up, inp, oproj, front, sdown = wide.cfg_rows(b=b, L=L)
        tasks = {
            "up": _tasks(cfg=up, rows=b, cols=wide.SL),
            "in_proj": _tasks(cfg=inp, rows=b, cols=NPROJ),
            "o_proj": _tasks(cfg=oproj, rows=b, cols=wide.H),
            "front": _tasks(cfg=front, rows=b, cols=wide.NF),
            "shared down": _tasks(cfg=sdown, rows=b, cols=wide.H, split_k=False),
        }
        assert max(tasks.values()) <= BLOCKS, (L, b, tasks)


def test_row_ranges_are_sorted_and_disjoint() -> None:
    for table in (
        wide.FRONT_ROW_RANGES,
        wide.INP_ROW_RANGES,
        wide.UP_ROW_RANGES,
        wide.SDOWN_ROW_RANGES,
    ):
        prev_hi = -1
        for lo, hi, _ in table:
            assert lo <= hi
            assert lo > prev_hi, f"overlapping or unsorted range {(lo, hi)}"
            prev_hi = hi
