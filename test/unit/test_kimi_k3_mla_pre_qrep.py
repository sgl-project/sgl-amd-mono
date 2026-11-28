# SPDX-License-Identifier: Apache-2.0
"""Unit tests for narrow DCP8 replicated-Q ``mla_pre`` wiring in ``kimi_k3/spec.py``."""

import sys
from types import SimpleNamespace

import torch

from sgl_amd_mono.kimi_k3 import spec
from sgl_amd_mono.kimi_k3.gfx950 import layer as k2_layer
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN


def _local_only_prepare(self, forward_batch):
    q_replicate_active = (
        self.is_dcp_q_replicate_phase(forward_batch)
        and not self.use_deep_gemm_bmm
        and self.w_kc_qrep is not None
        and self.q_b_proj_qrep_weight is not None
    )
    pre = self.__dict__.get("_mono_pre")
    if self.q_lora_rank is not None and pre is not None:
        return pre["q"].view(-1, self.num_local_heads, self.qk_head_dim)
    elif q_replicate_active:
        return self.q_b_proj_qrep_weight.view(
            -1, self.num_local_heads * self.dcp_size, self.qk_head_dim
        )
    return None


def _qrep_prepare(self, forward_batch):
    q_replicate_active = (
        self.is_dcp_q_replicate_phase(forward_batch)
        and not self.use_deep_gemm_bmm
        and self.w_kc_qrep is not None
        and self.q_b_proj_qrep_weight is not None
    )
    pre = self.__dict__.get("_mono_pre")
    if self.q_lora_rank is not None and pre is not None:
        num_q_heads = (
            self.num_local_heads * self.dcp_size
            if q_replicate_active
            else self.num_local_heads
        )
        return pre["q"].view(pre["q_lora"].shape[0], num_q_heads, self.qk_head_dim)
    return None


def _fake_layer(
    *,
    dcp_size=8,
    num_local_heads=12,
    qrep_active=True,
    has_qrep_weights=True,
    has_wkc=True,
    prepare_fn=_qrep_prepare,
    use_deep_gemm_bmm=False,
):
    def z(*s, dt=torch.bfloat16):
        return torch.zeros(*s, dtype=dt)

    idx = SimpleNamespace(
        wk=SimpleNamespace(weight=z(mp.IK, HIDDEN)),
        weights_proj=SimpleNamespace(weight=z(mp.IW, HIDDEN)),
        wq_b=SimpleNamespace(weight=z(mp.IQ, mp.QL)),
    )
    nhq_full = num_local_heads * dcp_size
    attn = SimpleNamespace(
        indexer=idx,
        fused_qkv_a_proj_with_mqa=SimpleNamespace(weight=z(mp.LATW, HIDDEN)),
        g_proj=SimpleNamespace(weight=z(mp.GATE, HIDDEN)),
        q_b_proj=SimpleNamespace(weight=z(mp.QB, mp.QL)),
        q_a_layernorm=SimpleNamespace(weight=z(mp.QL), variance_epsilon=1e-6),
        kv_a_layernorm=SimpleNamespace(weight=z(mp.KV), variance_epsilon=1e-6),
        # Match SGLang Kimi loader strides: w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        w_kc=z(mp.NHQ, mp.KV, mp.QN).transpose(1, 2) if has_wkc else None,
        w_scale=1.0,
        dcp_size=dcp_size,
        num_local_heads=num_local_heads,
        qk_head_dim=mp.QHD,
        use_deep_gemm_bmm=use_deep_gemm_bmm,
        q_b_proj_qrep_weight=z(nhq_full * mp.QHD, mp.QL) if has_qrep_weights else None,
        w_kc_qrep=z(nhq_full, mp.QN, mp.KV) if (has_qrep_weights and has_wkc) else None,
        is_dcp_q_replicate_phase=lambda fb: qrep_active,
        forward_absorb_rocm_prepare=prepare_fn,
    )
    return SimpleNamespace(
        layer_id=3,
        self_attn=attn,
        self_attention_res_norm=SimpleNamespace(
            weight=z(HIDDEN), variance_epsilon=1e-5
        ),
        self_attention_res_proj=SimpleNamespace(weight=z(1, HIDDEN)),
        input_layernorm=SimpleNamespace(weight=z(HIDDEN), variance_epsilon=1e-5),
        prev_valid_blocks=3,
        is_block_write_layer=False,
    )


def test_qrep_scratch_fits_runner_allocation() -> None:
    key = mp.MlaPreBuild(tokens=8, nblocks=7, qn=True, nhq=mp.NHQ_QREP, qn_rm=True)
    assert mp.scratch_bytes(key) <= k2_layer.scratch_alloc_bytes()


def test_tag_offset_clears_the_qrep_scratch_when_it_is_the_widest(monkeypatch) -> None:
    """The tagged x rows start past every narrow launch's scratch: with the other launches'
    layouts shrunk to nothing, the 96-head replicated-Q mla_pre still ends below them."""
    import itertools

    from sgl_amd_mono.kimi_k3 import MAX_BLOCKS, MAX_ROWS
    from sgl_amd_mono.kimi_k3.gfx950 import k1

    monkeypatch.setattr(k2_layer.k2b, "scratch_bytes", lambda key: 0)
    monkeypatch.setattr(k2_layer, "scratch_bytes", lambda key: 0)
    monkeypatch.setattr(k1, "scratch_bytes", lambda rows: 0)
    widest = max(
        mp.scratch_bytes(
            mp.MlaPreBuild(tokens=MAX_ROWS, nblocks=nb, qn=qn, nhq=nhq, qn_rm=rm)
        )
        for nb, qn, rm, nhq in itertools.product(
            range(MAX_BLOCKS + 1), (False, True), (False, True), (mp.NHQ, mp.NHQ_QREP)
        )
    )
    assert k2_layer.tag_offset() >= widest


def test_mla_build_dcp8_gates(monkeypatch) -> None:
    def _set_par(**kw):
        par = SimpleNamespace(
            attn_dcp_size=kw.get("attn_dcp_size", 8),
            dcp_enabled=kw.get("dcp_enabled", True),
            dcp_replicate_q_proj=kw.get("dcp_replicate_q_proj", True),
        )
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.runtime_context",
            SimpleNamespace(get_parallel=lambda: par),
        )

    # 1. CI-pinned / local-only SGLang without #53130 -> returns None under DCP8
    _set_par()
    assert spec._mla_build(_fake_layer(prepare_fn=_local_only_prepare), rows=8) is None

    # 2. Unsupported DCP size (dcp_size=4 -> 48 heads != 96) -> returns None
    assert spec._mla_build(_fake_layer(dcp_size=4), rows=8) is None

    # 3. dcp_replicate_q_proj=False or use_deep_gemm_bmm=True -> returns None
    _set_par(dcp_replicate_q_proj=False)
    assert spec._mla_build(_fake_layer(), rows=8) is None
    _set_par(dcp_replicate_q_proj=True)
    assert spec._mla_build(_fake_layer(use_deep_gemm_bmm=True), rows=8) is None

    # 4. DCP8 with #53130 (even before _prepare_replicated_q_proj populates qrep weights) -> builds nhq=96, qn_rm=True
    res = spec._mla_build(_fake_layer(has_qrep_weights=False), rows=8)
    assert res is not None
    key, w = res
    assert key.nhq == mp.NHQ_QREP and key.qn is True and key.qn_rm is True
    assert "w_kc" not in w  # in-place w_kc_qrep read at runtime, zero build-time copy

    # 5. DCP1 keeps local nhq=12, transposed w_kc
    _set_par(attn_dcp_size=1, dcp_enabled=False, dcp_replicate_q_proj=False)
    res1 = spec._mla_build(_fake_layer(dcp_size=1, has_qrep_weights=False), rows=8)
    assert res1 is not None
    key1, w1 = res1
    assert key1.nhq == mp.NHQ and key1.qn is True and key1.qn_rm is False
    assert tuple(w1["w_kc"].shape) == (mp.NHQ * mp.KV, mp.QN)


def test_mla_covered_runtime_checks() -> None:
    rows = 8
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    bank = SimpleNamespace(
        num_valid_blocks=3,
        block_residual=torch.zeros(rows, 8, HIDDEN, dtype=torch.bfloat16),
    )
    key = mp.MlaPreBuild(tokens=rows, nblocks=3, qn=True, nhq=mp.NHQ_QREP, qn_rm=True)

    # Valid qrep weights + active phase -> covered
    layer_ok = _fake_layer(has_qrep_weights=True, qrep_active=True)
    packed_ok = SimpleNamespace(module=layer_ok, mla=(key, {}))
    assert spec._mla_covered(packed_ok, hidden, bank, forward_batch=object()) is True

    # Missing forward_batch or inactive phase -> falls back
    assert spec._mla_covered(packed_ok, hidden, bank, forward_batch=None) is False
    layer_ok.self_attn.is_dcp_q_replicate_phase = lambda fb: False
    assert spec._mla_covered(packed_ok, hidden, bank, forward_batch=object()) is False

    # Missing qrep weights (e.g. _prepare_replicated_q_proj skipped layer) -> falls back
    layer_no_w = _fake_layer(has_qrep_weights=False, qrep_active=True)
    packed_no_w = SimpleNamespace(module=layer_no_w, mla=(key, {}))
    assert spec._mla_covered(packed_no_w, hidden, bank, forward_batch=object()) is False

    # Wrong row count -> falls back
    assert (
        spec._mla_covered(
            packed_ok,
            torch.zeros(rows - 1, HIDDEN, dtype=torch.bfloat16),
            bank,
            forward_batch=object(),
        )
        is False
    )


def test_mla_covered_rejects_local_key_when_qrep_active() -> None:
    rows = 8
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    bank = SimpleNamespace(
        num_valid_blocks=3,
        block_residual=torch.zeros(rows, 8, HIDDEN, dtype=torch.bfloat16),
    )
    key_local = mp.MlaPreBuild(tokens=rows, nblocks=3, qn=True, nhq=mp.NHQ, qn_rm=False)
    layer_qrep = _fake_layer(dcp_size=8, has_qrep_weights=True, qrep_active=True)
    packed = SimpleNamespace(module=layer_qrep, mla=(key_local, {}))
    assert spec._mla_covered(packed, hidden, bank, forward_batch=object()) is False


def test_mla_out_view_stays_contiguous_when_narrowing() -> None:
    out = spec._mla_outputs(rows=8, device=torch.device("cpu"), nhq=mp.NHQ_QREP)
    v = spec._mla_out_view(out, mp.NHQ)
    assert tuple(v["q"].shape) == (8, mp.QB) and v["q"].is_contiguous()
    assert tuple(v["qn"].shape) == (8, mp.NHQ, mp.KV) and v["qn"].is_contiguous()


def test_mla_pre_launch_rejects_undersized_q_output() -> None:
    import pytest

    key = mp.MlaPreBuild(tokens=8, nblocks=3, qn=False, nhq=mp.NHQ_QREP)
    blocks = torch.zeros(8, 8, HIDDEN, dtype=torch.bfloat16)
    bad_out = {"q": torch.zeros(8, mp.QB, dtype=torch.bfloat16)}
    with pytest.raises(AssertionError):
        mp.mla_pre_launch(
            key,
            prefix=None,
            blocks=blocks,
            w=None,
            out=bad_out,
            scratch=None,
            layer=3,
            epoch=None,
        )


def test_qrep_ast_rejects_local_view_with_unrelated_dcp_refs() -> None:
    """A branch that still views pre['q'] with local heads must be rejected even if
    q_replicate_active and self.dcp_size appear elsewhere in the same if-body."""

    def _false_positive_prepare(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if self.q_lora_rank is not None and pre is not None:
            q = pre["q"].view(-1, self.num_local_heads, self.qk_head_dim)
            if q_replicate_active and self.dcp_size > 1:
                _ = pre["latent"]
            return q
        return None

    layer = _fake_layer(prepare_fn=_false_positive_prepare)
    assert spec._sglang_mono_pre_supports_qrep(layer.self_attn) is False


def test_qrep_ast_reads_the_head_count_under_both_q_replicate_states() -> None:
    """The view's head count must be the local heads without q-replicate and
    num_local_heads * dcp_size with it, from assignments made before the view."""

    def _inverted(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            return pre["q"].view(
                -1,
                self.num_local_heads * (1 if q_replicate_active else self.dcp_size),
                self.qk_head_dim,
            )
        return None

    def _reassigned_after_view(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            heads = self.num_local_heads
            q = pre["q"].view(-1, heads, self.qk_head_dim)
            heads = (
                self.num_local_heads * self.dcp_size if q_replicate_active else heads
            )
            return q, heads
        return None

    def _inline(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            return pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return None

    def _nested_override(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            heads = (
                self.num_local_heads * self.dcp_size
                if q_replicate_active
                else self.num_local_heads
            )
            if q_replicate_active:
                heads = self.num_local_heads
            return pre["q"].view(-1, heads, self.qk_head_dim)
        return None

    def _literal(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            return pre["q"].view(
                -1,
                self.num_local_heads * (5 if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return None

    def _local_view_on_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            if q_replicate_active:
                return pre["q"].view(-1, self.num_local_heads, self.qk_head_dim)
            return pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return None

    def _rebound_predicate(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        q_replicate_active = q_replicate_active and self.dcp_size > 8
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            return pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return None

    def _other_predicate(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch) and self.dcp_size > 8
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            return pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return None

    def _tuple_view_on_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            if q_replicate_active:
                return pre["q"].view((-1, self.num_local_heads, self.qk_head_dim))
            return pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return None

    for prepare, ok in (
        (_inverted, False),
        (_reassigned_after_view, False),
        (_nested_override, False),
        (_literal, False),
        (_local_view_on_the_qrep_path, False),
        (_rebound_predicate, False),
        (_other_predicate, False),
        (_tuple_view_on_the_qrep_path, False),
        (_inline, True),
    ):
        layer = _fake_layer(prepare_fn=prepare)
        assert spec._sglang_mono_pre_supports_qrep(layer.self_attn) is ok, (
            prepare.__name__
        )
