"""The token-parallel row gathers' pair layout (``token_par.half_pair``), on CPU."""

import pytest

from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import Q_RANK
from sgl_amd_mono.deepseek_v41.gfx950.stages.seam import HIDDEN
from sgl_amd_mono.deepseek_v41.gfx950.token_par import (
    LINE,
    chunk_pairs,
    half_pair,
    peer_layout,
)
from sgl_amd_mono.deepseek_v41.spec import SPEC

MAX_TOKENS = max(SPEC.devices["gfx950"].widths)

PAIR = 8  # bytes of a (word, tag) pair
ROW_PAIRS = {"xg": HIDDEN // 2, "nbg": HIDDEN // 2, "qrg": Q_RANK // 2}


@pytest.mark.parametrize("n", sorted(set(ROW_PAIRS.values())))
def test_halves_cover_the_row_once(n):
    stored = {}
    for chunk in range(n // 4):
        for h in range(2):
            at = half_pair(n, chunk, h)
            for j in range(2):
                assert at + j not in stored
                stored[at + j] = 4 * chunk + 2 * h + j
    assert sorted(stored) == list(range(n))
    assert sorted(stored.values()) == list(range(n))
    for q in range(n // 2):
        assert chunk_pairs(n, q) == stored[2 * q]
        assert stored[2 * q + 1] == chunk_pairs(n, q) + 1


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("s", range(1, MAX_TOKENS + 1))
def test_a_wave_store_fills_whole_lines(s, tp):
    layout = peer_layout(s, tp)
    assert all(off % LINE == 0 for off, _ in layout.values()), layout
    # the other parity's regions start a half (``layer.peer_half_bytes``) later
    widest = peer_layout(MAX_TOKENS, tp)
    assert max(off + size for off, size in widest.values()) % LINE == 0
    for region, n in ROW_PAIRS.items():
        off, size = layout[region]
        assert size == s * n * PAIR
        for t in range(s):
            for h in range(2):
                for wave0 in range(0, n // 4, 64):
                    lanes = range(wave0, min(wave0 + 64, n // 4))
                    at = [off + (t * n + half_pair(n, c, h)) * PAIR for c in lanes]
                    assert at == list(range(at[0], at[0] + 16 * len(at), 16))
                    assert at[0] % LINE == 0 and (at[-1] + 16) % LINE == 0
