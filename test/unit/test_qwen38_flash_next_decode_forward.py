"""The Qwen3.8-Flash-Next TP2 decode forward on CPU: a routed step's launch sequence
on a recording launcher, SGLang's PLE and QSA calls in SGLang's order, the radix
track-save after the GDN launches, the check mode's per-layer calls, and the model's
own forward at a width the spec routes for the TP1 path."""

from types import SimpleNamespace as NS

import pytest
import torch
from qwen38_flash_next_decode_fakes import meta, needs_gpu_on_rocm

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

from sgl_amd_mono.qwen38_flash_next.decode import check
from sgl_amd_mono.qwen38_flash_next.decode import spec as D
from sgl_amd_mono.qwen38_flash_next.decode.gfx950.weights import Layer

pytestmark = needs_gpu_on_rocm

H = 8  # the embedding width the stand-in model uses
ROWS = 2


class _Launcher:
    def __init__(self):
        self.epoch = torch.zeros(1, dtype=torch.int32)
        self.attn_scratch, self.mlp_scratch = "attn-scratch", "mlp-scratch"
        self.calls = []

    def begin_step(self):
        self.calls.append(("begin",))

    def hc_front(self, idx, hc, state, msq, scratch):
        self.calls.append(("front", idx, hc, msq, scratch))
        return state[:, :H].clone(), torch.zeros(state.size(0), 4)

    def gdn(self, idx, g, x, conv, ssm, st_idx):
        self.calls.append(
            ("gdn", idx, tuple(conv.shape), st_idx.dtype, st_idx.tolist())
        )
        return x

    def oproj(self, w, core, gate, state, inj):
        self.calls.append(("oproj", w.idx, None if gate is None else tuple(gate.shape)))
        return state + 1, f"msq-oproj-{w.idx}"

    def moe(self, idx, m, x, state, inj):
        self.calls.append(("moe", idx))
        return state + 10, f"msq-moe-{idx}"


class _Ple:
    def __init__(self, log):
        self.log = log

    def start_prefetch(self, batch, forward_batch):
        self.log.append(("prefetch", batch))

    def __call__(self, state, forward_batch, batch):
        self.log.append(("ple", batch))
        return torch.full_like(state, 100)


def _qsa_module(log):
    def prepare(positions, hidden_states, forward_batch):
        log.append(("qsa positions", positions))
        return "q", "k", "v", torch.full((hidden_states.size(0), 2, 3), 7.0)

    def attn(q, k, v, forward_batch, topk_indices=None):
        log.append(("attn", topk_indices))
        return torch.ones(ROWS, 2, 3)

    return NS(
        ple=None,
        is_qsa=True,
        _prepare_qkv_gate=prepare,
        _compute_qsa_topk_indices=lambda x, positions, fb: "topk",
        attn=attn,
    )


@pytest.fixture
def sglang(monkeypatch):
    """SGLang's PLE batch calls, GDN pool and linear-attention backend, recorded."""
    import sglang.srt.model_executor.forward_context as forward_context
    import sglang.srt.models.qwen4_exp as qwen4_exp

    log = []
    pool = NS(
        conv=[meta(9, 3, 5120)], temporal=meta(9, 24, 128, 128, dtype=torch.float32)
    )
    md = NS(mamba_cache_indices=torch.tensor([3, 7, 0, 0], dtype=torch.int64))
    pools = dict.fromkeys((0, 1, 2), pool)  # the GDN layers of ``_model``
    lab = NS(
        forward_metadata=md,
        state_layer_cache=pools.__getitem__,
        _track_mamba_state_decode=lambda *args: None,
    )
    monkeypatch.setattr(
        qwen4_exp,
        "_prepare_ple_batch",
        lambda **kw: log.append(("prepare", sorted(kw))) or "ple-batch",
    )
    monkeypatch.setattr(
        qwen4_exp,
        "_commit_ple_batch",
        lambda batch, forward_batch: log.append(("commit", batch)),
    )
    monkeypatch.setattr(
        forward_context, "get_attn_backend", lambda: NS(linear_attn_backend=lab)
    )
    monkeypatch.delenv("SGLANG_AMD_MONO_CHECK", raising=False)
    return NS(log=log, lab=lab, md=md)


def _model(log):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    modules = [NS(ple=None), NS(ple=_Ple(log)), NS(ple=None), _qsa_module(log)]
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
    layers = [
        Layer(
            i,
            f"attn-hc-{i}",
            f"mlp-hc-{i}",
            None if i == 3 else f"gdn-{i}",
            None,
            f"moe-{i}",
        )
        for i in range(4)
    ]
    return model, layers


def _plan(rows=ROWS):
    key = registry.ShapeKey(
        forward_mode="decode", batch_size=rows, tokens_per_request=1
    )
    return registry.Plan(key=key)


def _step(built, fb):
    return built.forward(_plan(), torch.tensor([5, 6]), torch.tensor([9, 9]), fb)


def test_a_routed_step_runs_every_launch_in_sglang_s_order(sglang):
    model, layers = _model(sglang.log)
    la = _Launcher()
    built = D._Built(model, layers, la, widths=(1, 2), tp=2)
    fb = NS(mrope_positions=torch.arange(3 * ROWS).reshape(3, ROWS))
    out = _step(built, fb)

    # the conv state's window-first pool, transposed; SGLang's slots as int32
    gdn = ((9, 5120, 3), torch.int32, [3, 7])
    assert la.calls == [
        ("begin",),
        ("front", 0, "attn-hc-0", None, "attn-scratch"),
        ("gdn", 0, *gdn),
        ("oproj", 0, None),
        ("front", 0, "mlp-hc-0", "msq-oproj-0", "mlp-scratch"),
        ("moe", 0),
        # layer 1 adds its PLE first: the state's sums of squares are recomputed
        ("front", 1, "attn-hc-1", None, "attn-scratch"),
        ("gdn", 1, *gdn),
        ("oproj", 1, None),
        ("front", 1, "mlp-hc-1", "msq-oproj-1", "mlp-scratch"),
        ("moe", 1),
        ("front", 2, "attn-hc-2", "msq-moe-1", "attn-scratch"),
        ("gdn", 2, *gdn),
        ("oproj", 2, None),
        ("front", 2, "mlp-hc-2", "msq-oproj-2", "mlp-scratch"),
        ("moe", 2),
        ("front", 3, "attn-hc-3", "msq-moe-2", "attn-scratch"),
        ("oproj", 3, (ROWS, 6)),  # the QSA output gate, flattened
        ("front", 3, "mlp-hc-3", "msq-oproj-3", "mlp-scratch"),
        ("moe", 3),
    ]
    log = sglang.log
    assert [e[0] for e in log] == [
        "prepare",
        "prefetch",
        "ple",
        "qsa positions",
        "attn",
        "commit",
    ]
    assert log[1] == ("prefetch", "ple-batch") and log[-1] == ("commit", "ple-batch")
    # the QSA layer at the M-RoPE positions, with its top-k
    assert log[3][1] is fb.mrope_positions and log[4][1] == "topk"
    # embedding 1, + PLE 100 at layer 1, + 11 a layer; the logits from the mixed rows
    assert torch.equal(out.hidden_states, torch.full((ROWS, 4 * H), 1.0 + 100 + 4 * 11))
    assert torch.equal(out.next_token_logits, out.hidden_states[:, :H])
    assert model.model.last_hc_hidden_states is out.hidden_states


def test_a_tp1_width_runs_the_model_s_forward(sglang):
    calls = []

    class Model:
        def forward(self, *args):
            calls.append(args)
            return "the model's forward"

    model = Model()
    model.forward = lambda *a: pytest.fail("the runtime's switched forward re-entered")
    la = _Launcher()
    built = D._Built(model, [], la, widths=tuple(range(1, 17)), tp=2)
    # 20 rows: one of the TP1 path's widths, which SGLang routes at TP2 too
    out = built.forward(_plan(20), "ids", "positions", "batch")
    assert out == "the model's forward" and la.calls == []
    assert calls == [("ids", "positions", "batch", False, None)]


def _sglang_track_save(calls, md, gdn_layers):
    """SGLang's decode track-save with a layer id: nothing without a track mask, else
    one all-layers copy at the layer whose pool index is the last."""
    mamba_map = {layer: i for i, layer in enumerate(gdn_layers)}

    def track(forward_batch, conv_states, ssm_states, cache_indices, layer_id=None):
        last = mamba_map.get(layer_id) == len(mamba_map) - 1
        if forward_batch.mamba_track_mask is None or not last:
            return
        calls.append(
            ("copy all layers", forward_batch, cache_indices is md.mamba_cache_indices)
        )

    return track


def test_the_radix_track_save_copies_once_after_the_last_gdn_launch(sglang):
    model, layers = _model(sglang.log)
    la = _Launcher()
    sglang.lab._track_mamba_state_decode = _sglang_track_save(
        la.calls, sglang.md, [0, 1, 2]
    )
    built = D._Built(model, layers, la, widths=(2,), tp=2)
    fb = NS(
        mrope_positions=torch.arange(3 * ROWS).reshape(3, ROWS),
        mamba_track_mask=torch.tensor([True, False]),
    )
    _step(built, fb)
    kinds = [c[0] for c in la.calls]
    assert kinds.count("copy all layers") == 1
    i = kinds.index("copy all layers")
    # SGLang's int64 slots of the step, between the last GDN launch and its out_proj
    assert la.calls[i] == ("copy all layers", fb, True)
    assert la.calls[i - 1][:2] == ("gdn", 2) and la.calls[i + 1] == ("oproj", 2, None)
    fb.mamba_track_mask = None  # radix cache off
    la.calls.clear()
    _step(built, fb)
    assert "copy all layers" not in [c[0] for c in la.calls]
    assert ("gdn", 2) in [c[:2] for c in la.calls]


def test_a_gdn_pool_the_kernels_cannot_update_raises(sglang):
    model, layers = _model(sglang.log)
    built = D._Built(model, layers, _Launcher(), widths=(2,), tp=2)
    bf16 = NS(conv=[meta(9, 3, 5120)], temporal=meta(9, 24, 128, 128))
    with pytest.raises(RuntimeError, match="layer 0: recurrent state"):
        built._gdn_state(0, NS(state_layer_cache=lambda i: bf16))


def test_check_mode_checks_each_layer_against_sglang_s(sglang, monkeypatch):
    calls = []

    class Step:
        def __init__(self, positions, forward_batch, rows, ref):
            calls.append(("step", rows, ref))

        def before(self, layer, w, positions, state, slots, gdn):
            calls.append(("before", w.idx, gdn is not None))

        def front(self, w, hc, state, x):
            calls.append(("front", w.idx, hc))

        def core(self, core):
            calls.append(("core", tuple(core.shape)))

        def after(self, w, state):
            calls.append(("after", w.idx))

        def track(self, states, forward_batch, md):
            calls.append(("track", sorted(states), forward_batch is fb))

        def finish(self):
            calls.append(("finish",))

    monkeypatch.setattr(check, "StepCheck", Step)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("SGLANG_AMD_MONO_CHECK", "1")
    monkeypatch.setenv(check.REF_ENV, "mxfp4")
    model, layers = _model(sglang.log)
    for i, m in enumerate(model.model.layers):
        m.attn_hyper_connection, m.mlp_hyper_connection = f"attn-{i}", f"mlp-{i}"
    built = D._Built(model, layers, _Launcher(), widths=(2,), tp=2)
    fb = NS(mrope_positions=torch.arange(3 * ROWS).reshape(3, ROWS))
    _step(built, fb)

    def layer(i, gdn):
        core = [("core", (ROWS, H))] if gdn else []
        return [
            ("before", i, gdn),
            ("front", i, f"attn-{i}"),
            *core,
            ("front", i, f"mlp-{i}"),
            ("after", i),
        ]

    assert calls == [
        ("step", ROWS, "mxfp4"),
        *[c for i in range(4) for c in layer(i, i != 3)],
        ("track", [0, 1, 2], True),
        ("finish",),
    ]
    # no check while a graph captures
    calls.clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _step(built, fb)
    assert calls == []
    monkeypatch.setenv(check.REF_ENV, "fp8")
    with pytest.raises(ValueError, match="SGLANG_AMD_MONO_CHECK_REF"):
        D._Built(model, layers, _Launcher(), widths=(2,), tp=2)
