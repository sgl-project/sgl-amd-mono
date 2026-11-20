"""The Qwen3.8-Flash-Next TP2 decode on CPU: the launch state's builds and
arguments (``launcher.Launcher``), a routed step's launch sequence on a recording
``wide.layer.run``, SGLang's PLE and QSA calls in SGLang's order with qsa_front's
projections fed to SGLang's attention, the radix track-save after the GDN
launches, the GDN state binding, the check mode's per-layer calls, and the build's
compile order and settings."""

from types import SimpleNamespace as NS

import pytest
import torch
from qwen38_flash_next_decode_fakes import meta

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

from sgl_amd_mono.qwen38_flash_next.decode import check, launcher
from sgl_amd_mono.qwen38_flash_next.decode import spec as D
from sgl_amd_mono.qwen38_flash_next.decode.weights import Layer
from sgl_amd_mono.qwen38_flash_next.decode.wide import layer as K
from sgl_amd_mono.qwen38_flash_next.decode.wide.abi import PEER_HEAD
from sgl_amd_mono.qwen38_flash_next.decode.wide.common.grid import SLOT_WORDS

H = 8  # the embedding width the stand-in model uses
ROWS = 2
KINDS = ["gdn", "gdn", "gdn", "qsa", "gdn"]  # 6 launches a step; PLE on layer 1
BASES = (1 << 40, 2 << 40)  # the peer buffers' addresses


class _Peers:
    """``common.peer.PeerBuffer`` without peer memory."""

    def __init__(self, nbytes, group, rank, tp, device):
        self.nbytes = nbytes
        self.addresses = torch.tensor(BASES[:tp], dtype=torch.int64)


def _layers(kinds=KINDS, eps=1e-6):
    return [
        Layer(
            idx=i,
            kind=k,
            ple=i == 1,
            args={"w_out": torch.zeros(1)},
            hc_eps=eps,
            mlp_hc_eps=eps,
            gdn_eps=eps if k == "gdn" else None,
        )
        for i, k in enumerate(kinds)
    ]


def _launcher(widths=(1, 2), kinds=KINDS, rank=0, deadline_s=30.0):
    return launcher.Launcher(
        _layers(kinds),
        widths,
        tp=2,
        rank=rank,
        group=None,
        device=torch.device("cpu"),
        rip=512,
        deadline_s=deadline_s,
        make_peers=_Peers,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (
        "SGLANG_AMD_MONO_CHECK",
        "SGLANG_AMD_MONO_FAULT_INJECT",
        "ROCM_QUICK_REDUCE_QUANTIZATION",
    ):
        monkeypatch.delenv(name, raising=False)


def test_the_launcher_builds_in_sglang_s_setting(monkeypatch):
    la = _launcher(widths=(2, 1, 129), deadline_s=42.5)
    assert la.widths == (1, 2, 129) and la.launches == 6
    gdn0, gdn1 = la.keys(0, 129, True)[0], la.keys(1, 129, False)[0]
    front, back = la.keys(3, 129, False)
    assert gdn0 == K.LayerBuild(
        kind="gdn",
        tokens=129,
        tp=2,
        stats=True,
        qkvz_shuffled=True,
        out_shuffled=True,
        rip=512,
        hc_eps=1e-6,
        gdn_eps=1e-6,
        allreduce="bf16",
        conv_dim_first=True,
        wait_s=42.5,
        stop=True,
    )
    assert gdn1 == K.LayerBuild(**{**gdn0.__dict__, "stats": False})
    assert front == K.LayerBuild(
        kind="qsa_front", tokens=129, tp=2, rip=512, allreduce="bf16", wait_s=42.5, stop=True
    )  # fmt: skip
    assert back == K.LayerBuild(**{**front.__dict__, "kind": "qsa_back"})
    # a step: stats on the first layer and the PLE layer
    assert [k.stats for k in la.step_keys(1)] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    # four builds a width
    keys = la.all_keys()
    assert len(keys) == 4 * 3 == len(set(keys))
    assert keys == sorted(keys, key=repr)


def test_the_exchange_stays_bf16_and_the_fault_injection_reaches_the_builds(
    monkeypatch,
):
    # SGLang's decode all-reduce is exact at these sizes whatever QuickReduce's
    # setting, so the launches' exchange is too
    monkeypatch.setenv("ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    monkeypatch.setenv("SGLANG_AMD_MONO_FAULT_INJECT", "1:2:1000")
    for rank, fault in ((0, (0, 0)), (1, (2, 1000))):
        la = _launcher(rank=rank)
        assert la.codec == "bf16"
        for k in la.all_keys():
            assert k.allreduce == "bf16" and (k.fault_site, k.fault_after) == fault


def test_an_odd_launch_count_is_refused():
    with pytest.raises(ValueError, match="5 launches"):
        _launcher(kinds=["gdn", "gdn", "gdn", "qsa"])


def test_the_static_arguments_are_the_abi():
    la = _launcher(widths=(1, 16))
    assert la.scratch.numel() == K.scratch_bytes(16) and not la.scratch.any()
    assert la.x.shape == (16, 2560) and la.qkv.shape == (16, 6656)
    assert la.ixqk.shape == (16, 640)
    assert {w: t.dtype for w, t in la.st_idx.items()} == {
        1: torch.int32,
        16: torch.int32,
    }
    # every rank's mailbox after its flag words
    assert la.peer_table.tolist() == [BASES[0] + PEER_HEAD, BASES[1] + PEER_HEAD, 0, 0]
    gdn, qsa = la.args[0], la.args[3]
    for a in (gdn, qsa):
        K.ABI.pack(a)  # every name, nothing else
        assert a["peers"] is la.peer_table and a["scratch"] is la.scratch
        assert a["x"] is la.x and a["epoch"] is la.epoch and a["health"] is la.health
        assert a["folds"] == la.sync.data_ptr() + 4 * 2 * SLOT_WORDS
        # no alarm word: stop mode's flags take its place; no trace stamps
        assert a["alarm"] == 0 and a["trace"] == 0
    assert (gdn["layer"], qsa["layer"]) == (0, 3)
    assert qsa["qkv_out"] is la.qkv and qsa["ix_out"] is la.ixqk
    assert gdn["qkv_out"] == 0 and gdn["conv_state"] == 0


def test_bind_sets_the_gdn_states_and_their_pitches():
    la = _launcher()
    conv = torch.zeros(4, 15364, dtype=torch.bfloat16)[:, :15360].unflatten(
        1, (5120, 3)
    )
    ssm = torch.zeros(4, 24, 128, 128)
    assert not la.bound(2, conv, ssm)
    la.bind(2, conv, ssm)
    assert la.bound(2, conv, ssm) and not la.bound(0, conv, ssm)
    a = la.args[2]
    assert a["conv_state"] is conv and a["conv_stride"] == 15364
    assert a["ssm"] is ssm and a["ssm_stride"] == 24 * 128 * 128


def test_launches_alternate_the_barrier_slots(monkeypatch):
    runs = []
    monkeypatch.setattr(K, "run", lambda key, a: runs.append((key, a)))
    la = _launcher()
    base = la.sync.data_ptr()
    slots = (base, base + 4 * SLOT_WORDS)
    key = la.keys(0, 1, True)[0]
    seq = 0
    for _ in range(3):
        seq = la.launch(key, 0, seq, {"state": "s"})
    assert seq == 3
    assert [(a["sync"], a["sync_next"]) for _, a in runs] == [
        (slots[0], slots[1]),
        (slots[1], slots[0]),
        (slots[0], slots[1]),
    ]
    # the step's arguments do not stay in the layer's
    assert runs[0][1]["state"] == "s" and la.args[0]["state"] == 0


class _Ple:
    def __init__(self, log):
        self.log = log

    def start_prefetch(self, batch, forward_batch):
        self.log.append(("prefetch", batch))

    def __call__(self, state, forward_batch, batch):
        self.log.append(("ple", batch))
        return torch.full_like(state, 100)


def _qsa_module(log, la):
    """SGLang's QSA layer as the forward calls it: its prepare and indexer call
    the layer's projections, which the forward feeds."""
    layer = NS(ple=None, is_qsa=True)
    layer.qkv_proj = layer_qkv = NS(name="sglang qkv_proj")
    layer.indexer = NS(index_qk_proj=NS(name="sglang index_qk_proj"))

    def prepare(positions, hidden_states, forward_batch):
        qkv, _ = layer.qkv_proj(hidden_states)
        log.append(("prepare", positions, qkv is not None and qkv.data_ptr()))
        return "q", "k", "v", torch.full((hidden_states.size(0), 2, 3), 7.0)

    def topk(x, positions, fb):
        ix, _ = layer.indexer.index_qk_proj(x)
        log.append(("topk", ix.data_ptr()))
        return "topk"

    def attn(q, k, v, forward_batch, topk_indices=None):
        log.append(("attn", topk_indices, layer.qkv_proj is layer_qkv))
        return torch.ones(ROWS, 2, 3)

    layer._prepare_qkv_gate = prepare
    layer._compute_qsa_topk_indices = topk
    layer.attn = attn
    return layer


@pytest.fixture
def sglang(monkeypatch):
    """SGLang's PLE batch calls, GDN pool and linear-attention backend, recorded."""
    import sglang.srt.model_executor.forward_context as forward_context
    import sglang.srt.models.qwen4_exp as qwen4_exp

    log = []
    pool = NS(
        conv=[meta(9, 5120, 3)], temporal=meta(9, 24, 128, 128, dtype=torch.float32)
    )
    md = NS(mamba_cache_indices=torch.tensor([3, 7, -1, -1], dtype=torch.int64))
    lab = NS(
        forward_metadata=md,
        state_layer_cache=lambda i: log.append(("state", i)) or pool,
        _track_mamba_state_decode=lambda *args: None,
    )
    monkeypatch.setattr(
        qwen4_exp,
        "_prepare_ple_batch",
        lambda **kw: log.append(("prepare ple", sorted(kw))) or "ple-batch",
    )
    monkeypatch.setattr(
        qwen4_exp,
        "_commit_ple_batch",
        lambda batch, forward_batch: log.append(("commit", batch)),
    )
    monkeypatch.setattr(
        forward_context, "get_attn_backend", lambda: NS(linear_attn_backend=lab)
    )
    runs, epochs = [], []

    def run(key, a):
        runs.append((key, a))
        epochs.append(int(a["epoch"][0]))  # the step epoch the launch reads
        a["state"].add_(10)  # a launch updates the streams in place

    monkeypatch.setattr(K, "run", run)
    return NS(log=log, lab=lab, md=md, pool=pool, runs=runs, epochs=epochs)


def _model(log, la):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    modules = [NS(ple=None) for _ in KINDS]
    modules[1].ple = _Ple(log)
    modules[3] = _qsa_module(log, la)
    lm = NS(
        layers=modules,
        embed_tokens=lambda ids: torch.ones(ids.numel(), H),
        has_ple=True,
        ple_ngram_size=2,
        ple_ngram_eos_token_id=0,
        hyper_connection_mixer=NS(mix=lambda st: (st[:, :H],)),
        last_hc_hidden_states="stale",
    )
    model = NS(model=lm, is_mrope_enabled=True, lm_head="lm-head")
    model.logits_processor = lambda ids, hidden, head, fb, aux: LogitsProcessorOutput(
        next_token_logits=hidden
    )
    return model


def _plan(rows=ROWS):
    key = registry.ShapeKey(
        forward_mode="decode", batch_size=rows, tokens_per_request=1
    )
    return registry.Plan(key=key)


def _step(built, fb):
    return built.forward(_plan(), torch.tensor([5, 6]), torch.tensor([9, 9]), fb)


def test_a_routed_step_runs_every_launch_in_sglang_s_order(sglang):
    la = _launcher(widths=(1, 2))
    model = _model(sglang.log, la)
    built = D._Built(model, la)
    fb = NS(mrope_positions=torch.arange(3 * ROWS).reshape(3, ROWS))
    out = _step(built, fb)

    runs = sglang.runs
    assert [(k.kind, k.tokens, k.stats, a["layer"]) for k, a in runs] == [
        ("gdn", ROWS, True, 0),
        # layer 1 adds its PLE first: the state's norms are recomputed
        ("gdn", ROWS, True, 1),
        ("gdn", ROWS, False, 2),
        ("qsa_front", ROWS, False, 3),
        ("qsa_back", ROWS, False, 3),
        ("gdn", ROWS, False, 4),
    ]
    # the builds the build compiles for the width (stats where step_keys has it)
    assert [k for k, _ in runs] == la.step_keys(ROWS)
    # the epoch bumped once, before the step's first launch
    assert sglang.epochs == [1] * 6
    # the step's slots as int32, pads included; the launches alternate the slots
    assert (
        runs[0][1]["st_idx"].tolist() == [3, 7]
        and runs[0][1]["st_idx"].dtype == torch.int32
    )
    assert [a["sync"] for _, a in runs] == [la.slots[i % 2] for i in range(6)]
    # SGLang's GDN states bound into every GDN launch
    for _, a in (r for r in runs if r[0].kind == "gdn"):
        assert (
            a["conv_state"] is sglang.pool.conv[0] and a["ssm"] is sglang.pool.temporal
        )
        assert a["conv_stride"] == 15360 and a["ssm_stride"] == 24 * 128 * 128
    # qsa_back: SGLang's attention output and its gate, flattened
    back = runs[4][1]
    assert back["attn"].shape == (ROWS, 6) and (back["gate"] == 7).all()
    log = sglang.log
    assert [e[0] for e in log if e[0] != "state"] == [
        "prepare ple",
        "prefetch",
        "ple",
        "prepare",
        "topk",
        "attn",
        "commit",
    ]
    # the QSA layer at the M-RoPE positions, on qsa_front's projections (SGLang's
    # own projections back for the attention)
    prepare = next(e for e in log if e[0] == "prepare")
    assert prepare[1] is fb.mrope_positions and prepare[2] == la.qkv.data_ptr()
    assert next(e for e in log if e[0] == "topk")[1] == la.ixqk.data_ptr()
    assert next(e for e in log if e[0] == "attn")[1:] == ("topk", True)
    assert model.model.layers[3].indexer.index_qk_proj.name == "sglang index_qk_proj"
    # embedding 1, + PLE 100 at layer 1, + 10 a launch; the logits from the mixed rows
    assert torch.equal(out.hidden_states, torch.full((ROWS, 4 * H), 1.0 + 100 + 6 * 10))
    assert torch.equal(out.next_token_logits, out.hidden_states[:, :H])
    assert model.model.last_hc_hidden_states is out.hidden_states
    # each GDN layer's states checked once, then bound
    _step(built, fb)
    assert [e[1] for e in sglang.log if e[0] == "state"] == [0, 1, 2, 4] * 2
    assert sglang.epochs == [1] * 6 + [2] * 6


def test_the_projections_are_sglang_s_again_after_a_failure(sglang):
    la = _launcher()
    model = _model(sglang.log, la)
    layer = model.model.layers[3]

    def boom(*a, **k):
        raise RuntimeError("boom")

    layer._compute_qsa_topk_indices = boom
    with pytest.raises(RuntimeError, match="boom"):
        _step(D._Built(model, la), NS(mrope_positions=torch.zeros(3, ROWS)))
    assert layer.qkv_proj.name == "sglang qkv_proj"
    assert layer.indexer.index_qk_proj.name == "sglang index_qk_proj"


def _sglang_track_save(calls, md, gdn_layers):
    """SGLang's decode track-save with a layer id: nothing without a track mask, else
    one all-layers copy at the layer whose pool index is the last."""
    mamba_map = {layer: i for i, layer in enumerate(gdn_layers)}

    def track(forward_batch, conv_states, ssm_states, cache_indices, layer_id=None):
        last = mamba_map.get(layer_id) == len(mamba_map) - 1
        if forward_batch.mamba_track_mask is None or not last:
            return
        calls.append(("copy all layers", cache_indices is md.mamba_cache_indices))

    return track


def test_the_radix_track_save_copies_once_after_the_last_gdn_launch(
    sglang, monkeypatch
):
    la = _launcher()
    calls = []
    monkeypatch.setattr(K, "run", lambda key, a: calls.append((key.kind, a["layer"])))
    sglang.lab._track_mamba_state_decode = _sglang_track_save(
        calls, sglang.md, [0, 1, 2, 4]
    )
    built = D._Built(_model(sglang.log, la), la)
    fb = NS(
        mrope_positions=torch.arange(3 * ROWS).reshape(3, ROWS),
        mamba_track_mask=torch.tensor([True, False]),
    )
    _step(built, fb)
    # SGLang's int64 slots of the step, after the last GDN launch
    assert calls[-2:] == [("gdn", 4), ("copy all layers", True)]
    assert [c[0] for c in calls].count("copy all layers") == 1
    fb.mamba_track_mask = None  # radix cache off
    calls.clear()
    _step(built, fb)
    assert "copy all layers" not in [c[0] for c in calls]


def test_a_gdn_state_the_kernels_cannot_update_raises(sglang):
    la = _launcher()
    built = D._Built(_model(sglang.log, la), la)
    bf16 = NS(conv=[meta(9, 5120, 3)], temporal=meta(9, 24, 128, 128))
    with pytest.raises(RuntimeError, match="layer 0: recurrent state"):
        built._gdn_state(0, NS(state_layer_cache=lambda i: bf16))
    assert not la.bound(0, bf16.conv[0], bf16.temporal)


def test_check_mode_checks_each_layer_against_sglang_s(sglang, monkeypatch):
    calls = []

    class Step:
        def __init__(self, positions, forward_batch, rows, ref, launcher_):
            calls.append(("step", rows, ref, launcher_ is la))

        def before(self, layer, w, positions, state, slots, gdn):
            calls.append(("before", w.idx, gdn is not None))

        def front(self, i, layer, state, x, qkv, ixqk):
            calls.append(("front", i, qkv.data_ptr() == la.qkv.data_ptr()))

        def after(self, w, state):
            calls.append(("after", w.idx))

        def track(self, forward_batch, md):
            calls.append(("track", forward_batch is fb))

        def finish(self):
            calls.append(("finish",))

    monkeypatch.setattr(check, "StepCheck", Step)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("SGLANG_AMD_MONO_CHECK", "1")
    monkeypatch.setenv(check.REF_ENV, "mxfp4")
    la = _launcher()
    model = _model(sglang.log, la)
    built = D._Built(model, la)
    fb = NS(mrope_positions=torch.arange(3 * ROWS).reshape(3, ROWS))
    _step(built, fb)

    def layer(i):
        if KINDS[i] == "qsa":
            return [("before", i, False), ("front", i, True), ("after", i)]
        return [("before", i, True), ("after", i)]

    assert calls == [
        ("step", ROWS, "mxfp4", True),
        *[c for i in range(len(KINDS)) for c in layer(i)],
        ("track", True),
        ("finish",),
    ]
    # no check while a graph captures
    calls.clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _step(built, fb)
    assert calls == []
    monkeypatch.setenv(check.REF_ENV, "fp8")
    with pytest.raises(ValueError, match="SGLANG_AMD_MONO_CHECK_REF"):
        D._Built(model, la)


def test_the_build_compiles_every_width_s_builds_from_the_rank_s_share(monkeypatch):
    """The served widths (SGLang's routed ones) at SGLang's deadline; rank 1 of 2
    starts compiling half way, and the launch state is synchronized first."""
    import sglang.srt.runtime_context as runtime_context

    from sgl_amd_mono.common import peer
    from sgl_amd_mono.qwen38_flash_next.decode import weights

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: NS(tp_size=2, tp_rank=1, tp_group=NS(cpu_group=None)),
    )
    monkeypatch.setattr(peer, "PeerBuffer", _Peers)
    monkeypatch.setattr(
        weights,
        "build_layers",
        lambda model, copies: _layers(["gdn", "gdn", "gdn", "qsa"] * 12),
    )
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: events.append("sync"))
    monkeypatch.setattr(launcher, "compile_all", lambda keys: events.append(list(keys)))
    byte = torch.zeros(1, dtype=torch.uint8)  # the launch state's device: CPU
    experts = NS(
        w13_weight=byte.expand(513, 1024, 1280), w2_weight=byte.expand(513, 2560, 256)
    )
    model = NS(model=NS(layers=[NS(mlp=NS(experts=experts))]))
    widths = (1, 2, 4, 20, 192)
    ctx = NS(widths=widths, deadline_s=37.5, copies={})
    built = D.build(model, None, ctx)
    la = built.la
    keys = la.all_keys()
    assert len(keys) == 4 * len(widths) and la.widths == widths and la.rip == 512
    assert events[0] == "sync" and events[1] == keys[10:] + keys[:10]
    assert {k.wait_s for k in keys} == {37.5} and all(k.stop for k in keys)
