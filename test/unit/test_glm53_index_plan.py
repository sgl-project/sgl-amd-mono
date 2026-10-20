"""CP must cover every live score exactly once, including empty shards."""

import pytest

from sgl_amd_mono.glm53.gfx950.layout import (
    INDEX_KEYS_PER_TASK,
    INDEX_LOCAL_CAPACITY,
    index_cp_partitions,
    index_cp_world,
    index_q_sharded,
    index_score_capacity,
    index_shard_tiles,
    layout,
    stage_tasks,
)


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize(
    "live",
    [0, 1, 63, 64, 65, 2047, 2048, 2049, 4096, 16383, 16384, 16385, 131072, 1048576],
)
def test_live_tiles_have_one_owner(tp, live):
    shards = [index_shard_tiles(live, rank, tp) for rank in range(tp)]
    assert all(0 <= begin <= end for begin, end in shards)
    assert all(a[1] == b[0] for a, b in zip(shards, shards[1:]))
    tiles = [tile for begin, end in shards for tile in range(begin, end)]
    assert tiles == list(range((live + 63) // 64))
    # Every valid logical key is scored once. Only the last tile can have a tail.
    assert sum(min(INDEX_KEYS_PER_TASK, live - tile * 64) for tile in tiles) == live


@pytest.mark.parametrize("tp", [4, 8])
def test_short_live_requests_fit_a_larger_compiled_capacity(tp):
    capacity = 131072
    task_capacity = index_score_capacity(capacity, tp) // INDEX_KEYS_PER_TASK
    for live in (0, 1, 64, 4097, 16385, capacity):
        for rank in range(tp):
            begin, end = index_shard_tiles(live, rank, tp)
            assert end - begin <= task_capacity
    tasks = dict(
        stage_tasks(4, 64 // tp, 2048, True, capacity, indexer_cp=True, npes=tp)
    )
    assert tasks["index_score"] == 4 * task_capacity


@pytest.mark.parametrize("tp,rows", [(4, 2), (4, 4), (8, 2), (8, 4), (8, 8)])
@pytest.mark.parametrize("capacity", [64, 16384, 16448, 114688, 114752, 1048576])
def test_peer_slots_do_not_overlap_other_payloads_or_guard(tp, rows, capacity):
    # layout() currently imports the guard's constants from its device module.
    pytest.importorskip("flydsl")
    args = (rows, 64 // tp, tp, 2048, True, capacity)
    base_sc, base_sy = layout(*args, inter=2048 // tp)
    cp_sc, cp_sy = layout(*args, inter=2048 // tp, indexer_cp=True)
    assert "index_scores" in base_sc
    assert ("index_scores" in cp_sc) == (capacity > INDEX_LOCAL_CAPACITY)
    assert cp_sc["_bytes"] < base_sc["_bytes"]
    stride = cp_sy["_index_scores_stride"]
    assert (
        stride >= rows * min(capacity, INDEX_LOCAL_CAPACITY) * 8 and stride % 256 == 0
    )
    begin = cp_sy["index_scores"]
    assert begin == base_sy["flags"]
    end = begin
    for name in (
        "index_scores",
        "index_cp_q",
        "index_cp_hist",
        "index_cp_threshold",
        "index_cp_counts",
        "index_cp_offsets",
        "index_cp_indices",
    ):
        if name in cp_sy:
            assert cp_sy[name] == end
            end += 2 * cp_sy[f"_{name}_stride"]
    assert end == cp_sy["flags"]
    assert cp_sy["_bytes"] - cp_sy["flags"] == base_sy["_bytes"] - base_sy["flags"]
    for name in ("attn", "ffn", "dcp", "_part_stride", "_dcp_part_stride"):
        assert cp_sy[name] == base_sy[name]


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("capacity", [16384, 16448, 114688, 114752, 131072, 1048576])
def test_long_stages_cover_each_shard_partition(tp, capacity):
    rows = 4
    tasks = dict(
        stage_tasks(rows, 64 // tp, 2048, True, capacity, indexer_cp=True, npes=tp)
    )
    if capacity <= INDEX_LOCAL_CAPACITY:
        assert "index_hist_0" not in tasks
        return
    parts = index_cp_partitions(capacity, tp)
    assert parts * 4096 >= index_score_capacity(capacity, tp)
    assert (parts - 1) * 4096 < index_score_capacity(capacity, tp)
    for digit in range(4):
        assert tasks[f"index_hist_{digit}"] == rows * parts
        assert tasks[f"index_prefix_{digit}"] == rows
    assert tasks["index_count"] == tasks["index_emit"] == rows * parts
    assert tasks["index_offsets"] == tasks["index_select"] == rows


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("capacity", [64, 4096, 16384, 16448, 1048576])
def test_query_tiles_cover_all_heads(tp, capacity):
    for cp in (False, True):
        tasks = dict(
            stage_tasks(4, 64 // tp, 2048, True, capacity, indexer_cp=cp, npes=tp)
        )
        sharded = index_q_sharded(capacity, cp)
        tiles = tasks["index_q"]
        assert tiles * (tp if sharded else 1) == 256
        if sharded:
            assert [
                rank * tiles + tile for rank in range(tp) for tile in range(tiles)
            ] == list(range(256))


@pytest.mark.parametrize(
    "rows,capacity,group",
    [
        (2, 16384, 8),
        (2, 16448, 4),
        (2, 131072, 4),
        (2, 131136, 8),
        (4, 131072, 8),
        (8, 131072, 8),
    ],
)
def test_tp8_cp_subgroups_are_disjoint_replicas(rows, capacity, group):
    assert index_cp_world(rows, 8, capacity) == group
    groups = [list(range(start, start + group)) for start in range(0, 8, group)]
    assert sum(groups, []) == list(range(8))
    for ranks in groups:
        assert [rank % group for rank in ranks] == list(range(group))
        tiles = [
            tile
            for rank in ranks
            for tile in range(*index_shard_tiles(capacity, rank % group, group))
        ]
        assert tiles == list(range(capacity // 64))
