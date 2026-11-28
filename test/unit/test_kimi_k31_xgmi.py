# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Kimi-K3.1 XGMI heap layout, slicing, and prefill hooks."""

from __future__ import annotations

import threading
import types

import pytest
import torch

from sgl_amd_mono.kimi_k31.xgmi import (
    DECODE_BLOCKS,
    HEAP_BYTES,
    PREFILL_AG_BLOCKS,
    PREFILL_MAX_ROWS,
    RS_BLOCKS,
    TP,
    XgmiHookSummary,
    install_xgmi_hooks,
    slice_replicated_weight,
    xgmi_enabled,
)
from sgl_amd_mono.kimi_k31.xgmi.heap import (
    AG_FLAGS_BYTES,
    AG_FLAGS_OFF,
    AG_REGION_BYTES,
    AG_REGION_OFF,
    DECODE_REGION_BYTES,
    DECODE_REGION_OFF,
    EPOCH_TOTAL_WORDS,
    PREFILL_AG_HALF_BYTES,
    RS_FLAGS_BYTES,
    RS_FLAGS_OFF,
    RS_REGION_BYTES,
    RS_REGION_OFF,
    RS_SLOT_BYTES,
)


def test_heap_budget_under_400mb():
    assert HEAP_BYTES <= int(0.4 * (1 << 30))
    assert HEAP_BYTES % (2 << 20) == 0
    assert DECODE_REGION_OFF + DECODE_REGION_BYTES <= RS_REGION_OFF
    assert RS_REGION_OFF + RS_REGION_BYTES <= AG_REGION_OFF
    assert AG_REGION_OFF + AG_REGION_BYTES <= RS_FLAGS_OFF
    assert RS_FLAGS_OFF + RS_FLAGS_BYTES <= AG_FLAGS_OFF
    assert AG_FLAGS_OFF + AG_FLAGS_BYTES <= HEAP_BYTES
    assert PREFILL_MAX_ROWS * 2032 <= PREFILL_AG_HALF_BYTES
    assert PREFILL_MAX_ROWS * 3824 <= PREFILL_AG_HALF_BYTES
    assert PREFILL_MAX_ROWS * (2112 * 2) <= PREFILL_AG_HALF_BYTES
    assert PREFILL_MAX_ROWS * 1792 * 2 <= RS_SLOT_BYTES
    assert EPOCH_TOTAL_WORDS == DECODE_BLOCKS + RS_BLOCKS + PREFILL_AG_BLOCKS


# xgmi_ops.hip constexpr name -> heap.py name; both sides must describe one heap layout.
_HIP_TO_PY_LAYOUT = {
    "TP": "TP",
    "DECODE_BLOCKS": "DECODE_BLOCKS",
    "RS_BLOCKS": "RS_BLOCKS",
    "RS_RING_SLOTS": "RS_RING_SLOTS",
    "AG_BLOCKS": "PREFILL_AG_BLOCKS",
    "DECODE_RS_BYTES": "DECODE_RS_BYTES",
    "DECODE_AG_BYTES": "DECODE_AG_BYTES",
    "DECODE_HALF_BYTES": "DECODE_HALF_BYTES",
    "DECODE_REGION_OFF": "DECODE_REGION_OFF",
    "DECODE_REGION_BYTES": "DECODE_REGION_BYTES",
    "RS_SLOT_BYTES": "RS_SLOT_BYTES",
    "RS_REGION_OFF": "RS_REGION_OFF",
    "RS_REGION_BYTES": "RS_REGION_BYTES",
    "AG_HALF_BYTES": "PREFILL_AG_HALF_BYTES",
    "AG_REGION_OFF": "AG_REGION_OFF",
    "AG_REGION_BYTES": "AG_REGION_BYTES",
    "RS_FLAGS_OFF": "RS_FLAGS_OFF",
    "RS_FLAGS_BYTES": "RS_FLAGS_BYTES",
    "AG_FLAGS_OFF": "AG_FLAGS_OFF",
    "AG_FLAGS_BYTES": "AG_FLAGS_BYTES",
    "EPOCH_DECODE_OFF": "EPOCH_DECODE_OFF",
    "EPOCH_RS_OFF": "EPOCH_RS_OFF",
    "EPOCH_AG_OFF": "EPOCH_AG_OFF",
    "EPOCH_TOTAL_WORDS": "EPOCH_TOTAL_WORDS",
    "MIN_DEADLINE_S": "MIN_DEADLINE_S",
    "MAX_DEADLINE_S": "MAX_DEADLINE_S",
}
# HIP-only constants: launch geometry, and the pre-granule heap size checked separately.
_HIP_ONLY = {
    "DECODE_THREADS",
    "RS_THREADS",
    "AG_THREADS",
    "HEAP_MIN_BYTES",
    "HEAP_GRANULE",
}


def _hip_layout_constants() -> dict:
    import re
    from pathlib import Path

    import sgl_amd_mono.kimi_k31.xgmi as xgmi_pkg

    src = (Path(xgmi_pkg.__file__).parent / "xgmi_ops.hip").read_text()
    decl = re.compile(r"^constexpr (?:int|int64_t|double) (\w+) = ([^;{]+);$", re.M)
    env = {"align_up_4k": lambda n: (n + 4095) & ~4095}
    values = {}
    for name, expr in decl.findall(src):
        values[name] = env[name] = eval(re.sub(r"(\d)ll\b", r"\1", expr), {}, env)
    return values


def test_hip_heap_layout_matches_python():
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod

    hip = _hip_layout_constants()
    assert set(hip) == set(_HIP_TO_PY_LAYOUT) | _HIP_ONLY
    mismatched = {
        name: (value, getattr(heap_mod, _HIP_TO_PY_LAYOUT[name]))
        for name, value in hip.items()
        if name in _HIP_TO_PY_LAYOUT
        and value != getattr(heap_mod, _HIP_TO_PY_LAYOUT[name])
    }
    assert not mismatched, mismatched
    granule = hip["HEAP_GRANULE"]
    assert (hip["HEAP_MIN_BYTES"] + granule - 1) // granule * granule == HEAP_BYTES


def test_zero_copy_weight_slice():
    weight = torch.randn(7168, 3584, dtype=torch.bfloat16)
    sliced = slice_replicated_weight(weight, rank=3, tp=TP)
    assert sliced.shape == (896, 3584)
    assert sliced.is_contiguous()
    assert sliced.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()


def test_xgmi_opt_in_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_AMD_MONO_ENABLE_K31_XGMI", raising=False)
    assert not xgmi_enabled()
    monkeypatch.setenv("SGLANG_AMD_MONO_ENABLE_K31_XGMI", "1")
    assert xgmi_enabled()


def _dummy_heap(*, cpu_group=None, world_size: int = TP):
    return types.SimpleNamespace(
        cpu_group=cpu_group,
        rank=0,
        world_size=world_size,
        heap_bytes=HEAP_BYTES,
        heap_mib=HEAP_BYTES / (1 << 20),
    )


def _make_moe_layer(
    *,
    shard_front,
    row_sharded_tail,
    takes,
    tail_eligible,
    row_front_on: bool = True,
    min_tok: int = 512,
    norm_dtype: torch.dtype = torch.bfloat16,
):
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    mlp = object.__new__(KimiK3MoE)
    mlp.use_latent_moe = True
    mlp.routed_expert_up_proj = types.SimpleNamespace(
        weight=torch.empty(7168, 3584, dtype=torch.bfloat16)
    )
    mlp._front_w = torch.empty(768 + 896 + 3584, 7168, dtype=torch.bfloat16)
    mlp._front_sizes = (768, 896, 3584)
    mlp._row_front_on = row_front_on
    mlp._row_front_shared = None
    mlp._row_stream = False
    mlp._a8w4_moe_min_tokens = min_tok
    mlp._a8w4_moe_max_tokens = 65536
    mlp._row_sharded_tail_eligible = tail_eligible
    mlp.shard_front = shard_front
    mlp._row_sharded_tail = row_sharded_tail
    mlp._row_tail = lambda *args: None
    mlp._latent_norm = lambda tensor: tensor

    bank_state = AttnBankState(
        bank=None,
        attn_score_proj=None,
        attn_score_norm=None,
        ffn_score_proj=None,
        ffn_score_norm=None,
        ffn_row_shard=RowShardRead(takes=takes, shard_front=shard_front),
    )
    readout = _FfnReadout(bank_state)
    edge = types.SimpleNamespace(
        incoming=types.SimpleNamespace(need=types.SimpleNamespace(read=readout))
    )
    boundary = types.SimpleNamespace(
        plan=types.SimpleNamespace(edges={"default": edge}),
        declaration=types.SimpleNamespace(read=readout),
    )
    layer = types.SimpleNamespace(
        mlp=mlp,
        ffn_boundary=boundary,
        post_attention_layernorm=types.SimpleNamespace(
            weight=torch.empty(7168, dtype=norm_dtype)
        ),
    )
    return layer, bank_state


def _tail_eligible_at_4096(num_tokens, has_prefix):
    return num_tokens >= 4096 and has_prefix


def _takes_at_4096(num_tokens):
    return num_tokens >= 4096


def test_install_xgmi_hooks_unified_eligibility_and_fallback():
    fallback_calls = []

    def orig_front(normed, num_tokens):
        fallback_calls.append(num_tokens)
        return True

    tail_fallback_calls = []

    def orig_tail(latent_rows, shared_rows, num_tokens, prefix_sum):
        tail_fallback_calls.append(num_tokens)
        return latent_rows

    def make_layer(
        row_front_on: bool, min_tok: int, *, norm_dtype: torch.dtype = torch.bfloat16
    ):
        return _make_moe_layer(
            shard_front=orig_front,
            row_sharded_tail=orig_tail,
            takes=_takes_at_4096,
            tail_eligible=_tail_eligible_at_4096,
            row_front_on=row_front_on,
            min_tok=min_tok,
            norm_dtype=norm_dtype,
        )

    active_layer, active_bank = make_layer(row_front_on=True, min_tok=768)
    disabled_layer, disabled_bank = make_layer(row_front_on=False, min_tok=256)
    fp16_norm_layer, fp16_norm_bank = make_layer(
        row_front_on=True, min_tok=512, norm_dtype=torch.float16
    )
    orig_fp16_read = fp16_norm_bank.ffn_row_shard
    orig_fp16_tail = fp16_norm_layer.mlp._row_sharded_tail

    model = types.SimpleNamespace(
        layers=[active_layer, disabled_layer, fp16_norm_layer]
    )
    summary = install_xgmi_hooks(model, heap=_dummy_heap())
    assert isinstance(summary, XgmiHookSummary)
    assert summary.hooked_layers == 3
    assert summary.rebound_banks == 2

    # Non-BF16 post_attention_layernorm layer is skipped with zero mutation
    assert fp16_norm_bank.ffn_row_shard is orig_fp16_read
    assert fp16_norm_layer.mlp._row_sharded_tail is orig_fp16_tail

    # Unified contract honors max(prefill_min_rows=512, min_tok=768) == 768 on active_layer
    assert not active_bank.ffn_row_shard.takes(512)
    assert not active_layer.mlp._row_sharded_tail_eligible(512, True)
    assert active_bank.ffn_row_shard.takes(768)
    assert active_layer.mlp._row_sharded_tail_eligible(768, True)

    # Active layer enforces BF16 boundary invariant instead of splitting front/tail dispatch
    with pytest.raises(TypeError, match="bfloat16"):
        active_bank.ffn_row_shard.shard_front(
            torch.empty(96, 7168, dtype=torch.float16), 768
        )
    with pytest.raises(TypeError, match="bfloat16"):
        active_layer.mlp._row_sharded_tail(
            torch.empty(768, 3584, dtype=torch.float16),
            torch.empty(768, 7168, dtype=torch.bfloat16),
            768,
            torch.empty(96, 7168, dtype=torch.bfloat16),
        )

    # Active layer above prefill_max_rows (32768) delegates front + tail cleanly to upstream
    assert active_bank.ffn_row_shard.takes(40000)
    assert active_layer.mlp._row_sharded_tail_eligible(40000, True)
    assert active_bank.ffn_row_shard.shard_front(
        torch.empty(5000, 7168, dtype=torch.bfloat16), 40000
    )
    active_layer.mlp._row_sharded_tail(
        torch.empty(8, 3584, dtype=torch.bfloat16),
        torch.empty(8, 7168, dtype=torch.bfloat16),
        40000,
        torch.empty(1, 7168, dtype=torch.bfloat16),
    )
    assert fallback_calls == [40000]
    assert tail_fallback_calls == [40000]

    # Disabled row_front keeps original >= 4096 threshold and delegates shard_front fallback
    assert not disabled_bank.ffn_row_shard.takes(1024)
    assert disabled_bank.ffn_row_shard.takes(4096)
    assert disabled_bank.ffn_row_shard.shard_front(
        torch.empty(512, 7168, dtype=torch.bfloat16), 4096
    )
    assert fallback_calls == [40000, 4096]


def test_install_xgmi_hooks_graph_capture_falls_back(monkeypatch):
    calls = []
    layer, bank_state = _make_moe_layer(
        shard_front=lambda normed, num_tokens: (
            calls.append(("front", num_tokens)) or True
        ),
        row_sharded_tail=lambda lat, sh, num_tokens, pref: (
            calls.append(("tail", num_tokens)) or lat
        ),
        takes=_takes_at_4096,
        tail_eligible=_tail_eligible_at_4096,
    )
    install_xgmi_hooks(types.SimpleNamespace(layers=[layer]), heap=_dummy_heap())

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not bank_state.ffn_row_shard.takes(1024)
    assert not layer.mlp._row_sharded_tail_eligible(1024, True)
    assert bank_state.ffn_row_shard.takes(4096)
    assert bank_state.ffn_row_shard.shard_front(
        torch.empty(512, 7168, dtype=torch.bfloat16), 4096
    )
    layer.mlp._row_sharded_tail(
        torch.empty(8, 3584, dtype=torch.bfloat16),
        torch.empty(8, 7168, dtype=torch.bfloat16),
        4096,
        torch.empty(1, 7168, dtype=torch.bfloat16),
    )
    assert calls == [("front", 4096), ("tail", 4096)]


def _two_moe_layers():
    def build_layer():
        layer, bank = _make_moe_layer(
            shard_front=lambda normed, num_tokens: False,
            row_sharded_tail=lambda *args: None,
            takes=lambda num_tokens: False,
            tail_eligible=lambda num_tokens, has_prefix: False,
        )
        return layer, layer.mlp, bank, bank.ffn_row_shard

    return build_layer(), build_layer()


def _install_two(layer0, layer1):
    install_xgmi_hooks(
        types.SimpleNamespace(layers=[layer0, layer1]),
        heap=_dummy_heap(cpu_group=object(), world_size=2),
    )


def _gather_same(out_list, local_obj, group=None):
    for idx in range(len(out_list)):
        out_list[idx] = local_obj


def _gather_same_then(mutate):
    def gather(out_list, local_obj, group=None):
        _gather_same(out_list, local_obj)
        mutate()

    return gather


def _patch_install_collectives(monkeypatch, *, all_gather, all_reduce=None):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "all_reduce",
        all_reduce or (lambda tensor, op=None, group=None: None),
    )
    monkeypatch.setattr(dist, "all_gather_object", all_gather)


def _record_votes(votes: list):
    def record_vote(tensor, op=None, group=None):
        votes.append(int(tensor.item()))

    return record_vote


def test_install_xgmi_hooks_divergent_row_front_shared_fails_untouched(monkeypatch):
    (layer0, _, bank0, orig_read0), (layer1, _, bank1, orig_read1) = _two_moe_layers()

    def fake_all_gather_mismatch(out_list, local_obj, group=None):
        pmin, pmax, fp4, entries = local_obj
        rebound, on, need_all, min_t, max_t = entries[0]
        out_list[0] = local_obj
        out_list[1] = (
            pmin,
            pmax,
            fp4,
            ((rebound, on, not need_all, min_t, max_t), entries[1]),
        )

    _patch_install_collectives(monkeypatch, all_gather=fake_all_gather_mismatch)
    with pytest.raises(RuntimeError, match="TP ranks disagree"):
        _install_two(layer0, layer1)
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1


def test_install_xgmi_hooks_readout_removed_mid_install_leaves_layer0(monkeypatch):
    (layer0, _, bank0, orig_read0), (layer1, _, bank1, _) = _two_moe_layers()

    def clear_readout():
        bank1.ffn_row_shard = None

    _patch_install_collectives(monkeypatch, all_gather=_gather_same_then(clear_readout))
    with pytest.raises(RuntimeError, match="mutated during"):
        _install_two(layer0, layer1)
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is None


def test_install_xgmi_hooks_front_w_cleared_mid_install_votes_zero(monkeypatch):
    (layer0, _, bank0, orig_read0), (layer1, mlp1, bank1, orig_read1) = (
        _two_moe_layers()
    )
    votes = []

    def clear_front_w():
        mlp1._front_w = None

    _patch_install_collectives(
        monkeypatch,
        all_gather=_gather_same_then(clear_front_w),
        all_reduce=_record_votes(votes),
    )
    with pytest.raises(RuntimeError, match="mutated during"):
        _install_two(layer0, layer1)
    assert votes == [0]
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1


def test_install_xgmi_hooks_prepare_failure_votes_zero(monkeypatch):
    (layer0, _, bank0, orig_read0), (layer1, mlp1, bank1, orig_read1) = (
        _two_moe_layers()
    )
    votes = []

    def break_sizes():
        mlp1._front_sizes = ("bad", 896, 3584)

    _patch_install_collectives(
        monkeypatch,
        all_gather=_gather_same_then(break_sizes),
        all_reduce=_record_votes(votes),
    )
    with pytest.raises(RuntimeError, match="mutated during"):
        _install_two(layer0, layer1)
    assert votes == [0]
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1


def test_install_xgmi_hooks_config_failure_still_joins_collectives(monkeypatch):
    (layer0, _, bank0, orig_read0), (layer1, _, bank1, orig_read1) = _two_moe_layers()
    gathered_sentinels = []
    votes = []

    def fake_gather_record(out_list, local_obj, group=None):
        gathered_sentinels.append(local_obj)
        _gather_same(out_list, local_obj)

    monkeypatch.setenv("SGLANG_AMD_MONO_K31_XGMI_PREFILL_MIN", "not_an_int")
    _patch_install_collectives(
        monkeypatch, all_gather=fake_gather_record, all_reduce=_record_votes(votes)
    )
    with pytest.raises(RuntimeError, match="TP ranks disagree"):
        _install_two(layer0, layer1)
    assert gathered_sentinels == [None] and votes == [0]
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1


def test_install_xgmi_hooks_keeps_row_shard_read_fields():
    import dataclasses

    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead

    fields = {f.name for f in dataclasses.fields(RowShardRead)}
    if "attn_rows" not in fields:
        pytest.skip("this SGLang's RowShardRead has no attn_rows")
    layer, bank = _make_moe_layer(
        shard_front=lambda normed, num_tokens: False,
        row_sharded_tail=lambda *args: None,
        takes=_takes_at_4096,
        tail_eligible=_tail_eligible_at_4096,
    )

    def attn_rows(num_tokens):
        return None

    bank.ffn_row_shard = dataclasses.replace(bank.ffn_row_shard, attn_rows=attn_rows)
    install_xgmi_hooks(types.SimpleNamespace(layers=[layer]), heap=_dummy_heap())
    # The row-sharded o_proj leaves the attention sum to attn_rows; dropping it would
    # reduce-scatter o_proj's placeholder output instead.
    assert bank.ffn_row_shard.attn_rows is attn_rows
    assert bank.ffn_row_shard.shard_front is layer.mlp.shard_front


def test_install_xgmi_hooks_remote_rank_failure_leaves_layers_untouched(monkeypatch):
    (layer0, _, bank0, orig_read0), (layer1, _, bank1, orig_read1) = _two_moe_layers()
    _patch_install_collectives(
        monkeypatch,
        all_gather=_gather_same,
        all_reduce=lambda tensor, op=None, group=None: tensor.zero_(),
    )
    with pytest.raises(RuntimeError, match="mutated during"):
        _install_two(layer0, layer1)
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1


def test_install_xgmi_hooks_runtime_row_front_shared_mutation_raises(monkeypatch):
    (layer0, mlp0, bank0, _), (layer1, _, _, _) = _two_moe_layers()
    _patch_install_collectives(monkeypatch, all_gather=_gather_same)
    _install_two(layer0, layer1)
    mlp0._row_front_shared = object()
    with pytest.raises(RuntimeError, match="_row_front_shared mutated"):
        bank0.ffn_row_shard.shard_front(
            torch.empty(64, 7168, dtype=torch.bfloat16), 512
        )


def test_install_xgmi_hooks_runtime_row_front_on_mutation_raises(monkeypatch):
    (layer0, mlp0, bank0, _), (layer1, _, _, _) = _two_moe_layers()
    _patch_install_collectives(monkeypatch, all_gather=_gather_same)
    _install_two(layer0, layer1)
    mlp0._row_front_on = False
    with pytest.raises(RuntimeError, match="_row_front_on mutated"):
        bank0.ffn_row_shard.takes(512)


class _FakeStream:
    def __init__(self, name: str):
        self.name = name
        self.waited: list[str] = []

    def wait_stream(self, other):
        self.waited.append(other.name)


class _FakeEvent:
    def __init__(self):
        self.records = 0

    def record(self, _stream):
        self.records += 1


def _owned_heap(monkeypatch):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    import sgl_amd_mono.kimi_k31.xgmi.kernels as kernels_mod

    ctx = types.SimpleNamespace(
        events=[],
        s_main=_FakeStream("main"),
        s_comm=_FakeStream("comm"),
        capturing=False,
    )
    ctx.cur_stream = ctx.s_main

    class FakeOps:
        def unregister_heap(self, heap_id):
            ctx.events.append(("unregister", heap_id))

    monkeypatch.setattr(kernels_mod, "ops", lambda: FakeOps())
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: ctx.capturing
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev=None: ctx.cur_stream)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda _dev=None: ctx.events.append(("sync",))
    )

    heap = object.__new__(heap_mod.XgmiHeap)
    heap.rank = 3
    heap.world_size = TP
    heap.device = torch.device("cpu")
    heap.block_epoch = torch.zeros(EPOCH_TOTAL_WORDS, dtype=torch.int32)
    heap.streams = types.SimpleNamespace(
        comm=ctx.s_comm,
        close=lambda: ctx.events.append(("streams.close",)),
    )
    heap.chunk_events = [_FakeEvent() for _ in range(8)]
    heap._state_lock = threading.Lock()
    heap._owner_thread = None
    heap._active_scope = False
    heap._active_launch = False
    heap._closed = False
    heap._last_stream = None
    heap._heap_id = 42
    return heap, ctx


def test_xgmi_heap_guards_reject_graph_capture(monkeypatch):
    heap, ctx = _owned_heap(monkeypatch)
    ctx.capturing = True
    with pytest.raises(RuntimeError, match="graph capture"):
        with heap.launch_guard():
            pass
    with pytest.raises(RuntimeError, match="graph capture"):
        with heap.overlap_guard():
            pass


def test_xgmi_heap_launch_guard_rejects_reentry_and_active_close(monkeypatch):
    heap, _ = _owned_heap(monkeypatch)
    with heap.launch_guard() as heap_id:
        assert heap_id == 42
        with pytest.raises(RuntimeError, match="Concurrent or reentrant"):
            with heap.launch_guard():
                pass
        with pytest.raises(RuntimeError, match="active"):
            heap.close()


def test_xgmi_heap_overlap_guard_rejects_nesting_and_active_close(monkeypatch):
    heap, _ = _owned_heap(monkeypatch)
    with heap.overlap_guard():
        with pytest.raises(RuntimeError, match="nested"):
            with heap.overlap_guard():
                pass
        with pytest.raises(RuntimeError, match="active"):
            heap.close()
        with heap.launch_guard() as heap_id:
            assert heap_id == 42


def test_xgmi_heap_launch_on_new_stream_waits_for_previous(monkeypatch):
    heap, ctx = _owned_heap(monkeypatch)
    with heap.launch_guard():
        pass
    ctx.cur_stream = ctx.s_comm
    with heap.launch_guard() as heap_id:
        assert heap_id == 42
    assert ctx.s_comm.waited == ["main"]


def test_xgmi_heap_rejects_non_owner_thread_before_events(monkeypatch):
    from sgl_amd_mono.kimi_k31.xgmi.prefill import overlapped_o_proj_rs

    heap, _ = _owned_heap(monkeypatch)
    with heap.launch_guard():
        pass
    other_err: list[Exception] = []

    def worker():
        try:
            overlapped_o_proj_rs(
                heap=heap,
                hidden=torch.zeros(16, 16, dtype=torch.bfloat16),
                o_proj_weight=torch.zeros(32, 16, dtype=torch.bfloat16),
                n_chunks=4,
            )
        except Exception as err:
            other_err.append(err)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert len(other_err) == 1 and isinstance(other_err[0], RuntimeError)
    assert "owning TP worker thread" in str(other_err[0])
    assert all(ev.records == 0 for ev in heap.chunk_events)


def test_xgmi_heap_close_syncs_unregisters_and_is_idempotent(monkeypatch):
    heap, ctx = _owned_heap(monkeypatch)
    heap.close()
    assert [e[0] for e in ctx.events] == [
        "sync",
        "unregister",
        "streams.close",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        with heap.launch_guard():
            pass
    heap.close()


def _fake_tp_world(monkeypatch) -> list:
    import torch.distributed as dist

    trace = []
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: TP)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    def fake_gather(out_list, obj, group=None):
        trace.append(("gather", obj))
        for idx in range(len(out_list)):
            out_list[idx] = obj if idx == 0 else (HEAP_BYTES, [1] * 64)

    def fake_reduce(tensor, op=None, group=None):
        if tensor.numel() == 1:
            trace.append(("reduce", int(tensor.item())))
        else:
            trace.append(("preflight", tuple(int(v) for v in tensor.tolist())))

    monkeypatch.setattr(dist, "all_gather_object", fake_gather)
    monkeypatch.setattr(dist, "all_reduce", fake_reduce)
    return trace


def test_xgmi_heap_init_local_failure_participates_in_handshake(monkeypatch):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    import sgl_amd_mono.kimi_k31.xgmi.kernels as kernels_mod

    trace = _fake_tp_world(monkeypatch)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _dev: types.SimpleNamespace(gcnArchName="gfx950"),
    )
    monkeypatch.setattr(
        kernels_mod,
        "ops",
        lambda: (_ for _ in ()).throw(RuntimeError("rank-local JIT failure")),
    )

    with pytest.raises(RuntimeError, match="handshake failed"):
        heap_mod.XgmiHeap(cpu_group=object(), rank=0, device=torch.device("cpu"))
    assert trace == [("gather", (0, None)), ("reduce", 0)]


def test_xgmi_heap_init_sync_failure_votes_zero_without_barrier(monkeypatch):
    import torch.distributed as dist

    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    import sgl_amd_mono.kimi_k31.xgmi.kernels as kernels_mod

    trace = []

    class FakeOps:
        def create_local_heap(self, block_epoch, heap_bytes):
            return 99, torch.ones(64, dtype=torch.uint8)

        def open_peer_heaps(self, heap_id, peer_handles_cpu, rank):
            trace.append(("open", heap_id, rank))

        def unregister_heap(self, heap_id):
            trace.append(("unregister", heap_id))

    class FakeStreams:
        def __init__(self, device):
            pass

        def close(self):
            trace.append(("streams.close",))

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: TP)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _dev: types.SimpleNamespace(gcnArchName="gfx950"),
    )
    monkeypatch.setattr(kernels_mod, "ops", lambda: FakeOps())
    monkeypatch.setattr(heap_mod, "XgmiStreams", FakeStreams)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _dev=None: (_ for _ in ()).throw(RuntimeError("gpu sync failure")),
    )
    monkeypatch.setattr(
        dist,
        "barrier",
        lambda group=None: trace.append(("barrier",)),
    )

    def fake_gather(out_list, obj, group=None):
        for idx in range(len(out_list)):
            out_list[idx] = obj

    def fake_reduce(tensor, op=None, group=None):
        if tensor.numel() == 1:
            trace.append(("reduce", int(tensor.item())))
        else:
            trace.append(("preflight", tuple(int(v) for v in tensor.tolist())))

    monkeypatch.setattr(dist, "all_gather_object", fake_gather)
    monkeypatch.setattr(dist, "all_reduce", fake_reduce)

    with pytest.raises(RuntimeError, match="handshake failed"):
        heap_mod.XgmiHeap(cpu_group=object(), rank=0, device=torch.device("cpu"))
    assert trace == [
        ("open", 99, 0),
        ("reduce", 0),
        ("unregister", 99),
        ("streams.close",),
    ]

    # If unregister_heap's own sync also raises, preserve original error and retain streams
    trace.clear()

    class FailingUnregisterOps(FakeOps):
        def unregister_heap(self, heap_id):
            trace.append(("unregister_fail", heap_id))
            raise RuntimeError("unregister sync failure")

    monkeypatch.setattr(kernels_mod, "ops", lambda: FailingUnregisterOps())
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        heap_mod.XgmiHeap(cpu_group=object(), rank=0, device=torch.device("cpu"))
    assert "gpu sync failure" in str(exc_info.value.__cause__)
    assert trace == [
        ("open", 99, 0),
        ("reduce", 0),
        ("unregister_fail", 99),
    ]


def test_xgmi_heap_rank_mismatch_votes_zero(monkeypatch):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod

    trace = _fake_tp_world(monkeypatch)
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        heap_mod.XgmiHeap(cpu_group=object(), rank=1, device=torch.device("cpu"))
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert trace == [("gather", (0, None)), ("reduce", 0)]


# Non-finite, non-positive, sub-tick, and overflowing deadlines.
@pytest.mark.parametrize(
    "bad_deadline", [float("nan"), float("inf"), -1.0, 0.0, 5e-9, 1e9]
)
def test_xgmi_heap_bad_deadline_arg_votes_zero(monkeypatch, bad_deadline):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod

    trace = _fake_tp_world(monkeypatch)
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        heap_mod.XgmiHeap(
            cpu_group=object(),
            rank=0,
            device=torch.device("cpu"),
            deadline_s=bad_deadline,
        )
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert trace == [("gather", (0, None)), ("reduce", 0)]


@pytest.mark.parametrize("bad_env", ["nan", "0", "not_a_float"])
def test_xgmi_heap_bad_deadline_env_votes_zero(monkeypatch, bad_env):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod

    trace = _fake_tp_world(monkeypatch)
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_XGMI_DEADLINE_S", bad_env)
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        heap_mod.XgmiHeap(cpu_group=object(), rank=0, device=torch.device("cpu"))
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert trace == [("gather", (0, None)), ("reduce", 0)]


def _patch_parallel(monkeypatch, cpu_group) -> None:
    import sglang.srt.runtime_context as rc_mod

    monkeypatch.setattr(
        rc_mod,
        "get_parallel",
        lambda: types.SimpleNamespace(
            tp_group=types.SimpleNamespace(cpu_group=cpu_group),
            tp_rank=0,
        ),
    )


def test_maybe_install_current_device_failure_votes_zero(monkeypatch):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    from sgl_amd_mono.kimi_k31.xgmi import maybe_install_xgmi_hooks

    trace = _fake_tp_world(monkeypatch)
    monkeypatch.setenv("SGLANG_AMD_MONO_ENABLE_K31_XGMI", "1")
    monkeypatch.setattr(heap_mod, "_GLOBAL_HEAP", None)
    _patch_parallel(monkeypatch, object())
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("cuda device init failed")),
    )
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        maybe_install_xgmi_hooks(types.SimpleNamespace(layers=[]))
    assert "cuda device init failed" in str(exc_info.value.__cause__)
    assert trace == [("preflight", (0, 0)), ("gather", (0, None)), ("reduce", 0)]


def test_xgmi_opt_in_disagreement_raises_in_spec_and_hook(monkeypatch):
    import torch.distributed as dist

    import sgl_amd_mono.kimi_k31.spec as k31_spec
    from sgl_amd_mono.kimi_k31.xgmi import maybe_install_xgmi_hooks

    _fake_tp_world(monkeypatch)
    monkeypatch.setenv("SGLANG_AMD_MONO_ENABLE_K31_XGMI", "1")
    _patch_parallel(monkeypatch, object())
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def fake_reduce_optin_mismatch(tensor, op=None, group=None):
        tensor[0] = 0
        tensor[1] = -1

    monkeypatch.setattr(dist, "all_reduce", fake_reduce_optin_mismatch)
    with pytest.raises(RuntimeError, match="disagrees across TP ranks"):
        k31_spec._agree_xgmi_enabled()
    with pytest.raises(RuntimeError, match="disagrees across TP ranks"):
        maybe_install_xgmi_hooks(types.SimpleNamespace(layers=[]))


def test_get_global_heap_cached_mismatch_votes_zero(monkeypatch):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod

    trace = _fake_tp_world(monkeypatch)
    fake_group = object()
    cached = object.__new__(heap_mod.XgmiHeap)
    cached.cpu_group = fake_group
    cached.rank = 0
    cached.device = torch.device("cpu")
    cached._closed = False
    monkeypatch.setattr(heap_mod, "_GLOBAL_HEAP", cached)
    with pytest.raises(ValueError, match="cached-heap validation failed"):
        heap_mod.get_global_heap(
            cpu_group=fake_group, rank=1, device=torch.device("cpu")
        )
    assert trace == [("preflight", (1, -1)), ("reduce", 0)]


def test_get_global_heap_mixed_cache_state_fails_preflight(monkeypatch):
    import torch.distributed as dist

    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod

    trace = _fake_tp_world(monkeypatch)
    monkeypatch.setattr(heap_mod, "_GLOBAL_HEAP", None)

    def fake_reduce_mixed(tensor, op=None, group=None):
        trace.append(("preflight", tuple(int(v) for v in tensor.tolist())))
        tensor[0] = 0
        tensor[1] = -1

    monkeypatch.setattr(dist, "all_reduce", fake_reduce_mixed)
    with pytest.raises(RuntimeError, match="mixed XgmiHeap cache state"):
        heap_mod.get_global_heap(cpu_group=object(), rank=0, device=torch.device("cpu"))
    assert trace == [("preflight", (0, 0))]


@pytest.mark.parametrize("bad_chunks", [0, -2, 16, True])
def test_overlapped_o_proj_rs_rejects_bad_n_chunks(bad_chunks):
    from sgl_amd_mono.kimi_k31.xgmi.prefill import overlapped_o_proj_rs

    with pytest.raises(ValueError, match="n_chunks"):
        overlapped_o_proj_rs(
            heap=types.SimpleNamespace(chunk_events=[object()] * 8),
            hidden=torch.zeros(16, 16, dtype=torch.bfloat16),
            o_proj_weight=torch.zeros(64, 16, dtype=torch.bfloat16),
            n_chunks=bad_chunks,
        )


def test_overlapped_prefill_avoids_comm_stream_allocations(monkeypatch):
    """Overlapped o_proj RS and normed allgather must allocate only on main_stream
    and synchronize main_stream on exception without calling record_stream."""
    import contextlib

    from sgl_amd_mono.kimi_k31.xgmi import hook as xgmi_hook
    from sgl_amd_mono.kimi_k31.xgmi import prefill as xgmi_prefill

    in_comm = {"flag": False}
    events_log = []

    class FakeStream:
        def __init__(self, name: str):
            self.name = name

        def wait_stream(self, other):
            events_log.append((self.name, "wait_stream", other.name))

        def wait_event(self, ev):
            events_log.append((self.name, "wait_event", ev.idx))

    class FakeEvent:
        def __init__(self, idx: int):
            self.idx = idx

        def record(self, stream):
            events_log.append((stream.name, "record", self.idx))

    main_s = FakeStream("main")
    comm_s = FakeStream("comm")

    @contextlib.contextmanager
    def fake_stream_ctx(s):
        prev = in_comm["flag"]
        in_comm["flag"] = s is comm_s
        try:
            yield
        finally:
            in_comm["flag"] = prev

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: main_s)
    monkeypatch.setattr(torch.cuda, "stream", fake_stream_ctx)

    def fake_rs(heap, partial, *, out_shard=None, out_col_off=0):
        assert in_comm["flag"]
        assert out_shard is not None and partial.is_contiguous()
        return out_shard

    monkeypatch.setattr(xgmi_prefill, "reduce_scatter_rows", fake_rs)

    fake_heap = types.SimpleNamespace(
        streams=types.SimpleNamespace(comm=comm_s),
        chunk_events=[FakeEvent(i) for i in range(4)],
        overlap_guard=contextlib.nullcontext,
    )
    x = torch.empty(16, 32, dtype=torch.bfloat16)
    w = torch.empty(64, 32, dtype=torch.bfloat16)
    out = xgmi_prefill.overlapped_o_proj_rs(
        heap=fake_heap, hidden=x, o_proj_weight=w, n_chunks=4
    )
    assert out.shape == (2, 64)
    assert ("main", "wait_stream", "comm") in events_log

    seen_ag = {}

    def fake_ag(heap, x_shard, *, out_all=None, tmp=None):
        assert in_comm["flag"]
        assert out_all is not None and x_shard.is_contiguous()
        seen_ag["out_all"] = out_all
        seen_ag["tmp"] = tmp
        return out_all

    monkeypatch.setattr(xgmi_hook, "row_allgather", fake_ag)
    normed = torch.empty(1024, 7168, dtype=torch.bfloat16)
    out_all, keep = xgmi_hook._gather_normed_all(
        heap=fake_heap, normed=normed, need_normed_all=True, overlap_normed=True
    )
    assert out_all is seen_ag["out_all"]
    assert out_all.shape == (1024 * TP, 7168)
    assert seen_ag["tmp"] is not None
    assert len(keep) == 2

    events_log.clear()

    def failing_ag(heap, x_shard, *, out_all=None, tmp=None):
        raise RuntimeError("injected chunk failure")

    monkeypatch.setattr(xgmi_hook, "row_allgather", failing_ag)
    with pytest.raises(RuntimeError, match="injected chunk failure"):
        xgmi_hook._gather_normed_all(
            heap=fake_heap, normed=normed, need_normed_all=True, overlap_normed=True
        )
    assert ("main", "wait_stream", "comm") in events_log
