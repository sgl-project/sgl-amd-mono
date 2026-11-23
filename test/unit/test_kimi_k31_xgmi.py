# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Kimi-K3.1 XGMI heap layout, slicing, and prefill hooks."""

from __future__ import annotations

import threading
import types

import pytest
import torch

from sgl_amd_mono.kimi_k31.xgmi import (
    DECODE_BLOCKS,
    DEFAULT_DEADLINE_S,
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
    assert RS_FLAGS_BYTES == 32768 and AG_FLAGS_BYTES == 16384
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
    assert DEFAULT_DEADLINE_S >= 900.0


def test_zero_copy_weight_slice():
    weight = torch.randn(7168, 3584, dtype=torch.bfloat16)
    sliced = slice_replicated_weight(weight, rank=3, tp=TP)
    assert sliced.shape == (896, 3584)
    assert sliced.is_contiguous()
    assert sliced.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()


def test_xgmi_opt_in_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_XGMI", raising=False)
    assert not xgmi_enabled()
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_XGMI", "1")
    assert xgmi_enabled()


def test_install_xgmi_hooks_unified_eligibility_and_fallback():
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    class DummyHeap:
        cpu_group = None
        rank = 0
        world_size = TP
        heap_bytes = HEAP_BYTES
        heap_mib = HEAP_BYTES / (1 << 20)

    fallback_calls = []

    def orig_front(normed, num_tokens):
        fallback_calls.append(num_tokens)
        return True

    tail_fallback_calls = []

    def orig_tail(latent_rows, shared_rows, num_tokens, prefix_sum):
        tail_fallback_calls.append(num_tokens)
        return latent_rows

    def make_layer(
        row_front_on: bool,
        min_tok: int,
        *,
        norm_dtype: torch.dtype = torch.bfloat16,
        front_dtype: torch.dtype = torch.bfloat16,
        up_dtype: torch.dtype = torch.bfloat16,
    ):
        mlp = object.__new__(KimiK3MoE)
        mlp.use_latent_moe = True
        mlp.routed_expert_up_proj = types.SimpleNamespace(
            weight=torch.empty(7168, 3584, dtype=up_dtype)
        )
        mlp._front_w = torch.empty(768 + 896 + 3584, 7168, dtype=front_dtype)
        mlp._front_sizes = (768, 896, 3584)
        mlp._row_front_on = row_front_on
        mlp._row_front_shared = None
        mlp._row_stream = False
        mlp._a8w4_moe_min_tokens = min_tok
        mlp._a8w4_moe_max_tokens = 65536
        mlp._row_sharded_tail_eligible = lambda num_tokens, has_prefix: (
            num_tokens >= 4096 and has_prefix
        )
        mlp.shard_front = orig_front
        mlp._row_sharded_tail = orig_tail
        mlp._row_tail = lambda *args: None
        mlp._latent_norm = lambda tensor: tensor

        bank_state = AttnBankState(
            bank=None,
            attn_score_proj=None,
            attn_score_norm=None,
            ffn_score_proj=None,
            ffn_score_norm=None,
            ffn_row_shard=RowShardRead(
                takes=lambda num_tokens: num_tokens >= 4096,
                shard_front=orig_front,
            ),
        )
        readout = _FfnReadout(bank_state)
        edge = types.SimpleNamespace(
            incoming=types.SimpleNamespace(need=types.SimpleNamespace(read=readout))
        )
        boundary = types.SimpleNamespace(
            plan=types.SimpleNamespace(edges={"default": edge}),
            declaration=types.SimpleNamespace(read=readout),
        )
        post_norm = types.SimpleNamespace(weight=torch.empty(7168, dtype=norm_dtype))
        return (
            types.SimpleNamespace(
                mlp=mlp,
                ffn_boundary=boundary,
                post_attention_layernorm=post_norm,
            ),
            bank_state,
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
    summary = install_xgmi_hooks(model, heap=DummyHeap())
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
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    class DummyHeap:
        cpu_group = None
        rank = 0
        world_size = TP
        heap_bytes = HEAP_BYTES
        heap_mib = HEAP_BYTES / (1 << 20)

    calls = []
    mlp = object.__new__(KimiK3MoE)
    mlp.use_latent_moe = True
    mlp.routed_expert_up_proj = types.SimpleNamespace(
        weight=torch.empty(7168, 3584, dtype=torch.bfloat16)
    )
    mlp._front_w = torch.empty(768 + 896 + 3584, 7168, dtype=torch.bfloat16)
    mlp._front_sizes = (768, 896, 3584)
    mlp._row_front_on = True
    mlp._row_front_shared = None
    mlp._row_stream = False
    mlp._a8w4_moe_min_tokens = 512
    mlp._a8w4_moe_max_tokens = 65536
    mlp._row_sharded_tail_eligible = lambda num_tokens, has_prefix: (
        num_tokens >= 4096 and has_prefix
    )
    mlp.shard_front = lambda normed, num_tokens: (
        calls.append(("front", num_tokens)) or True
    )
    mlp._row_sharded_tail = lambda lat, sh, num_tokens, pref: (
        calls.append(("tail", num_tokens)) or lat
    )
    mlp._row_tail = lambda *args: None
    mlp._latent_norm = lambda tensor: tensor

    bank_state = AttnBankState(
        bank=None,
        attn_score_proj=None,
        attn_score_norm=None,
        ffn_score_proj=None,
        ffn_score_norm=None,
        ffn_row_shard=RowShardRead(
            takes=lambda num_tokens: num_tokens >= 4096,
            shard_front=mlp.shard_front,
        ),
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
            weight=torch.empty(7168, dtype=torch.bfloat16)
        ),
    )
    install_xgmi_hooks(types.SimpleNamespace(layers=[layer]), heap=DummyHeap())

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


def test_install_xgmi_hooks_tp_plan_agreement_and_immutability(monkeypatch):
    import torch.distributed as dist
    from sglang.srt.layers.layer_boundary.residual.attn_bank import (
        AttnBankState,
        _FfnReadout,
    )
    from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import RowShardRead
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    class DummyHeap:
        cpu_group = object()
        rank = 0
        world_size = 2
        heap_bytes = HEAP_BYTES
        heap_mib = HEAP_BYTES / (1 << 20)

    def build_layer():
        m = object.__new__(KimiK3MoE)
        m.use_latent_moe = True
        m.routed_expert_up_proj = types.SimpleNamespace(
            weight=torch.empty(7168, 3584, dtype=torch.bfloat16)
        )
        m._front_w = torch.empty(768 + 896 + 3584, 7168, dtype=torch.bfloat16)
        m._front_sizes = (768, 896, 3584)
        m._row_front_on = True
        m._row_front_shared = None
        m._row_stream = False
        m._a8w4_moe_min_tokens = 512
        m._a8w4_moe_max_tokens = 65536
        m._row_sharded_tail_eligible = lambda num_tokens, has_prefix: False
        m.shard_front = lambda normed, num_tokens: False
        m._row_sharded_tail = lambda *args: None
        m._row_tail = lambda *args: None
        m._latent_norm = lambda tensor: tensor

        read = RowShardRead(takes=lambda num_tokens: False, shard_front=m.shard_front)
        bs = AttnBankState(
            bank=None,
            attn_score_proj=None,
            attn_score_norm=None,
            ffn_score_proj=None,
            ffn_score_norm=None,
            ffn_row_shard=read,
        )
        ro = _FfnReadout(bs)
        ed = types.SimpleNamespace(
            incoming=types.SimpleNamespace(need=types.SimpleNamespace(read=ro))
        )
        bd = types.SimpleNamespace(
            plan=types.SimpleNamespace(edges={"default": ed}),
            declaration=types.SimpleNamespace(read=ro),
        )
        lyr = types.SimpleNamespace(
            mlp=m,
            ffn_boundary=bd,
            post_attention_layernorm=types.SimpleNamespace(
                weight=torch.empty(7168, dtype=torch.bfloat16)
            ),
        )
        return lyr, m, bs, read

    layer0, mlp0, bank0, orig_read0 = build_layer()
    layer1, mlp1, bank1, orig_read1 = build_layer()

    # Divergent _row_front_shared across ranks fails before mutating any layer
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None: None)

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

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_mismatch)
    with pytest.raises(RuntimeError, match="TP ranks disagree"):
        install_xgmi_hooks(
            types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap()
        )
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1

    # Matching TP plans install cleanly, then runtime mutation of _row_front_shared/_row_front_on raises
    def fake_all_gather_match(out_list, local_obj, group=None):
        for idx in range(len(out_list)):
            out_list[idx] = local_obj

    # Removing ffn_row_shard on layer 1 during all_gather_object leaves layer 0 untouched
    def fake_all_gather_clears_readout(out_list, local_obj, group=None):
        for idx in range(len(out_list)):
            out_list[idx] = local_obj
        bank1.ffn_row_shard = None

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_clears_readout)
    with pytest.raises(RuntimeError, match="mutated during"):
        install_xgmi_hooks(
            types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap()
        )
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is None
    bank1.ffn_row_shard = orig_read1

    # Clearing _front_w during all_gather_object votes 0 in all_reduce with zero hooks changed
    votes = []

    def record_vote(tensor, op=None, group=None):
        votes.append(int(tensor.item()))

    saved_front_w = mlp1._front_w

    def fake_all_gather_clears_front_w(out_list, local_obj, group=None):
        for idx in range(len(out_list)):
            out_list[idx] = local_obj
        mlp1._front_w = None

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_clears_front_w)
    monkeypatch.setattr(dist, "all_reduce", record_vote)
    with pytest.raises(RuntimeError, match="mutated during"):
        install_xgmi_hooks(
            types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap()
        )
    assert votes == [0]
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1
    mlp1._front_w = saved_front_w

    # Failure while preparing layer 1 turns into a failed TP vote with layer 0 untouched
    votes.clear()
    saved_sizes = mlp1._front_sizes

    def fake_all_gather_breaks_sizes(out_list, local_obj, group=None):
        for idx in range(len(out_list)):
            out_list[idx] = local_obj
        mlp1._front_sizes = ("bad", 896, 3584)

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_breaks_sizes)
    with pytest.raises(RuntimeError, match="mutated during"):
        install_xgmi_hooks(
            types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap()
        )
    assert votes == [0]
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1
    mlp1._front_sizes = saved_sizes

    # Pre-gather config failure still enters all_gather_object(None) + all_reduce(0)
    gathered_sentinels = []
    votes.clear()

    def fake_gather_record(out_list, local_obj, group=None):
        gathered_sentinels.append(local_obj)
        for idx in range(len(out_list)):
            out_list[idx] = local_obj

    monkeypatch.setenv("SGLANG_AMD_MONO_K31_XGMI_PREFILL_MIN", "not_an_int")
    monkeypatch.setattr(dist, "all_gather_object", fake_gather_record)
    with pytest.raises(RuntimeError, match="TP ranks disagree"):
        install_xgmi_hooks(
            types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap()
        )
    assert gathered_sentinels == [None] and votes == [0]
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_XGMI_PREFILL_MIN", raising=False)

    # Remote rank failing validation (all_reduce MIN -> 0) leaves all local layers untouched
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_match)
    monkeypatch.setattr(
        dist, "all_reduce", lambda tensor, op=None, group=None: tensor.zero_()
    )
    with pytest.raises(RuntimeError, match="mutated during"):
        install_xgmi_hooks(
            types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap()
        )
    assert bank0.ffn_row_shard is orig_read0 and bank1.ffn_row_shard is orig_read1

    # Matching TP agreement installs cleanly; runtime mutation of agreed fields raises
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None: None)
    install_xgmi_hooks(types.SimpleNamespace(layers=[layer0, layer1]), heap=DummyHeap())
    mlp0._row_front_shared = object()
    with pytest.raises(RuntimeError, match="_row_front_shared mutated"):
        bank0.ffn_row_shard.shard_front(
            torch.empty(64, 7168, dtype=torch.bfloat16), 512
        )
    mlp0._row_front_shared = None
    mlp0._row_front_on = False
    with pytest.raises(RuntimeError, match="_row_front_on mutated"):
        bank0.ffn_row_shard.takes(512)


def test_xgmi_heap_guard_lifecycle_and_overlap(monkeypatch):
    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    import sgl_amd_mono.kimi_k31.xgmi.kernels as kernels_mod
    from sgl_amd_mono.kimi_k31.xgmi.prefill import overlapped_o_proj_rs

    class FakeStream:
        def __init__(self, name: str):
            self.name = name
            self.waited: list[str] = []

        def wait_stream(self, other):
            self.waited.append(other.name)

    class FakeEvent:
        def __init__(self):
            self.records = 0

        def record(self, _stream):
            self.records += 1

    events: list[tuple] = []
    s_main = FakeStream("main")
    s_comm = FakeStream("comm")
    cur_stream = [s_main]
    capturing = [False]
    fake_streams = types.SimpleNamespace(
        comm=s_comm,
        close=lambda: events.append(("streams.close",)),
    )

    class FakeOps:
        def unregister_heap(self, heap_id):
            events.append(("unregister", heap_id))

    monkeypatch.setattr(kernels_mod, "ops", lambda: FakeOps())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev=None: cur_stream[0])
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda _dev=None: events.append(("sync",))
    )

    heap = object.__new__(heap_mod.XgmiHeap)
    heap.rank = 3
    heap.world_size = TP
    heap.device = torch.device("cpu")
    heap.block_epoch = torch.zeros(EPOCH_TOTAL_WORDS, dtype=torch.int32)
    heap.streams = fake_streams
    heap.chunk_events = [FakeEvent() for _ in range(8)]
    heap._state_lock = threading.Lock()
    heap._owner_thread = None
    heap._active_scope = False
    heap._active_launch = False
    heap._closed = False
    heap._last_stream = None
    heap._heap_id = 42

    # Capture is rejected by both overlap_guard and launch_guard
    capturing[0] = True
    with pytest.raises(RuntimeError, match="graph capture"):
        with heap.launch_guard():
            pass
    with pytest.raises(RuntimeError, match="graph capture"):
        with heap.overlap_guard():
            pass
    capturing[0] = False

    # First eager launch binds owner thread; reentrant launch and close while active are rejected
    with heap.launch_guard() as heap_id:
        assert heap_id == 42
        with pytest.raises(RuntimeError, match="Concurrent or reentrant"):
            with heap.launch_guard():
                pass
        with pytest.raises(RuntimeError, match="active"):
            heap.close()

    # overlap_guard allows owning thread launches, rejects nested overlap and active close
    with heap.overlap_guard():
        with pytest.raises(RuntimeError, match="nested"):
            with heap.overlap_guard():
                pass
        with pytest.raises(RuntimeError, match="active"):
            heap.close()
        with heap.launch_guard() as heap_id:
            assert heap_id == 42

    # Cross-stream ordering on the owning thread inserts wait_stream
    cur_stream[0] = s_comm
    with heap.launch_guard() as heap_id:
        assert heap_id == 42
    assert s_comm.waited == ["main"]

    # Another thread calling overlapped_o_proj_rs is rejected BEFORE touching chunk_events
    other_err: list[Exception] = []

    def worker():
        try:
            overlapped_o_proj_rs(
                heap,
                torch.zeros(16, 16, dtype=torch.bfloat16),
                torch.zeros(32, 16, dtype=torch.bfloat16),
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

    # close() synchronizes, unregisters owned native heap, and rejects future launches
    heap.close()
    assert [e[0] for e in events] == [
        "sync",
        "unregister",
        "streams.close",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        with heap.launch_guard():
            pass
    # close() is idempotent
    heap.close()


def test_xgmi_heap_init_local_failure_participates_in_handshake(monkeypatch):
    import torch.distributed as dist

    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    import sgl_amd_mono.kimi_k31.xgmi.kernels as kernels_mod

    trace = []

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: TP)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
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

    with pytest.raises(RuntimeError, match="handshake failed"):
        heap_mod.XgmiHeap(object(), 0, torch.device("cpu"))
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
        def __init__(self, device, comm_cus=32):
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
    monkeypatch.setattr(heap_mod, "CuMaskStreams", FakeStreams)
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
        heap_mod.XgmiHeap(object(), 0, torch.device("cpu"))
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
        heap_mod.XgmiHeap(object(), 0, torch.device("cpu"))
    assert "gpu sync failure" in str(exc_info.value.__cause__)
    assert trace == [
        ("open", 99, 0),
        ("reduce", 0),
        ("unregister_fail", 99),
    ]


def test_xgmi_heap_rank_mismatch_and_invalid_deadline_and_chunks(monkeypatch):
    import torch.distributed as dist

    import sgl_amd_mono.kimi_k31.xgmi.heap as heap_mod
    from sgl_amd_mono.kimi_k31.xgmi.prefill import overlapped_o_proj_rs

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

    # Rank mismatch routes through gather((0, None)) + vote 0 instead of stranding peers
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        heap_mod.XgmiHeap(object(), 1, torch.device("cpu"))
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert trace == [("gather", (0, None)), ("reduce", 0)]

    # Non-finite, non-positive, sub-tick, and overflowing deadlines are rejected before gather
    for bad_deadline in (float("nan"), float("inf"), -1.0, 0.0, 5e-9, 1e9):
        trace.clear()
        with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
            heap_mod.XgmiHeap(object(), 0, torch.device("cpu"), deadline_s=bad_deadline)
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert trace == [("gather", (0, None)), ("reduce", 0)]

    # current_device() failure via maybe_install_xgmi_hooks routes through gather + vote 0
    import sglang.srt.runtime_context as rc_mod

    from sgl_amd_mono.kimi_k31.xgmi import maybe_install_xgmi_hooks

    trace.clear()
    fake_group = object()
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_XGMI", "1")
    monkeypatch.setattr(heap_mod, "_GLOBAL_HEAP", None)
    monkeypatch.setattr(
        rc_mod,
        "get_parallel",
        lambda: types.SimpleNamespace(
            tp_group=types.SimpleNamespace(cpu_group=fake_group),
            tp_rank=0,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("cuda device init failed")),
    )
    with pytest.raises(RuntimeError, match="handshake failed") as exc_info:
        maybe_install_xgmi_hooks(types.SimpleNamespace(layers=[]))
    assert "cuda device init failed" in str(exc_info.value.__cause__)
    assert trace == [("preflight", (0, 0)), ("gather", (0, None)), ("reduce", 0)]

    # Mixed SGLANG_AMD_MONO_K31_XGMI across TP ranks fails vote in both spec.py and maybe_install_xgmi_hooks
    import sgl_amd_mono.kimi_k31.spec as k31_spec

    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def fake_reduce_optin_mismatch(tensor, op=None, group=None):
        tensor[0] = 0
        tensor[1] = -1

    monkeypatch.setattr(dist, "all_reduce", fake_reduce_optin_mismatch)
    with pytest.raises(RuntimeError, match="disagrees across TP ranks"):
        k31_spec._agree_xgmi_enabled()
    with pytest.raises(RuntimeError, match="disagrees across TP ranks"):
        maybe_install_xgmi_hooks(types.SimpleNamespace(layers=[]))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "all_reduce", fake_reduce)

    # Cached-heap mismatch participates in TP all_reduce vote so peers do not strand
    trace.clear()
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

    # Mixed cache state (this rank uncached, peer still cached) fails preflight on all ranks
    trace.clear()
    monkeypatch.setattr(heap_mod, "_GLOBAL_HEAP", None)

    def fake_reduce_mixed(tensor, op=None, group=None):
        trace.append(("preflight", tuple(int(v) for v in tensor.tolist())))
        tensor[0] = 0
        tensor[1] = -1

    monkeypatch.setattr(dist, "all_reduce", fake_reduce_mixed)
    with pytest.raises(RuntimeError, match="mixed XgmiHeap cache state"):
        heap_mod.get_global_heap(
            cpu_group=fake_group, rank=0, device=torch.device("cpu")
        )
    assert trace == [("preflight", (0, 0))]
    monkeypatch.setattr(dist, "all_reduce", fake_reduce)
    monkeypatch.setattr(heap_mod, "_GLOBAL_HEAP", None)

    # overlapped_o_proj_rs rejects unsafe n_chunks before any launch or event record
    dummy_heap = types.SimpleNamespace(chunk_events=[object()] * 8)
    x = torch.zeros(16, 16, dtype=torch.bfloat16)
    w = torch.zeros(64, 16, dtype=torch.bfloat16)
    for bad_chunks in (0, -2, 16, True):
        with pytest.raises(ValueError, match="n_chunks"):
            overlapped_o_proj_rs(dummy_heap, x, w, n_chunks=bad_chunks)
