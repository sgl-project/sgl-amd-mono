# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 wide launch tile configs keep the MoE front and in_proj in one BLOCKS round."""

import math

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k3.gfx950.attention.kda import NPROJ
from sgl_amd_mono.kimi_k31 import wide


def _tasks(cfg: str, rows: int, cols: int) -> int:
    # gfx950/layer.py: row tiles x column tiles x split-K slices
    bm, bn, _, _, ks = (int(x) for x in cfg.split(":"))
    return math.ceil(rows / bm) * math.ceil(cols / bn) * ks


def test_every_row_count_fits_one_round() -> None:
    """A front or in_proj tile choice with more tasks than BLOCKS costs a whole second round;
    every row count up to the largest CFGS key fits, not only today's buckets."""
    for b in range(wide.NARROW + 1, max(wide.CFGS) + 1):
        _, inp, _, front, _ = wide._row_cfgs(b)
        assert _tasks(front, rows=b, cols=wide.NF) <= BLOCKS, (b, front)
        assert _tasks(inp, rows=b, cols=NPROJ) <= BLOCKS, (b, inp)


@pytest.mark.parametrize("L", wide.SUPPORTED_L)
def test_every_bucket_is_within_the_checked_rows(L: int) -> None:
    # a bucket past the largest CFGS key falls back to that key's tiles, which the test above does not check
    for r in wide.REQ_BUCKETS:
        b = L * r
        assert wide.NARROW < b <= max(wide.CFGS), (L, b)
        assert wide.cfg_rows(b, L) == wide._row_cfgs(b)


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
