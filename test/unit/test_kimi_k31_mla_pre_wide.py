# SPDX-License-Identifier: Apache-2.0
"""Derived-property tests for wide MLA projection-front layout and tile budgets."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.common.plan import BLOCKS, WAVES
from sgl_amd_mono.kimi_k31 import wide
from sgl_amd_mono.kimi_k31.gfx950 import mla_pre_wide as mpw

_STREAM = object()


def _dispatch_directly(monkeypatch) -> list:
    """Stand AITER's compiled dispatch in for the fake launchers; return the launchers it ran.
    Each launch must pass the current stream as its last positional argument."""
    from aiter.ops.flydsl.kernels import tensor_shim

    dispatched = []

    def run(exe, *args):
        assert args and args[-1] is _STREAM, args[-1:]
        dispatched.append(exe)
        exe(*args)

    monkeypatch.setattr(tensor_shim, "_run_compiled", run)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _STREAM)
    return dispatched


def _no_mono_prepare(self, forward_batch):
    q_replicate_active = (
        self.is_dcp_q_replicate_phase(forward_batch)
        and not self.use_deep_gemm_bmm
        and self.w_kc_qrep is not None
        and self.q_b_proj_qrep_weight is not None
    )
    return self.q_b_proj.weight.view(-1, self.num_local_heads, self.qk_head_dim)


def _local_only_prepare(self, forward_batch):
    q_replicate_active = (
        self.is_dcp_q_replicate_phase(forward_batch)
        and not self.use_deep_gemm_bmm
        and self.w_kc_qrep is not None
        and self.q_b_proj_qrep_weight is not None
    )
    pre = self.__dict__.get("_mono_pre")
    if self.q_lora_rank is not None and pre is not None:
        q = pre["q"].view(-1, self.num_local_heads, self.qk_head_dim)
    elif q_replicate_active:
        return self.q_b_proj_qrep_weight.view(
            -1, self.num_local_heads * self.dcp_size, self.qk_head_dim
        )
    return self._split_q_nope_pe(q, None)


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
        q = pre["q"].view(pre["q_lora"].shape[0], num_q_heads, self.qk_head_dim)
    q_nope, q_pe, k_pe = self._split_q_nope_pe(q, None)
    return q_nope


@pytest.mark.parametrize("m", [1, 8, 12, 16, 24, 30, 32, 48, 64])
@pytest.mark.parametrize("qb_rows", [mpw.QB_LOCAL, mpw.QB_QREP])
def test_tile_and_stage_budgets_fit_one_wave(m: int, qb_rows: int) -> None:
    """Stage 1 and Stage 2 CTA counts fit in one BLOCKS wave and satisfy MFMA alignment."""
    bm, bn, wm, wn = mpw.tile_shape(m)
    assert bm >= m and wm * wn == WAVES
    assert (bm // wm) % 16 == 0 and (bn // wn) % 16 == 0
    assert mpw.S1_TASKS <= BLOCKS
    n2, ks2, kt2 = mpw.stage2_spec(qb_rows)
    assert n2 * ks2 <= BLOCKS
    assert kt2 * ks2 == mpw.QL and kt2 % 64 == 0
    lay = mpw.scratch_layout(m, qb_rows, timeline=True)
    assert lay["nbytes"] > 0 and "tl" in lay["offs"]
    assert ("fbuf2" in lay["offs"]) == (ks2 > 1)
    expected_cnt = (mpw.S1_TILES + (n2 if ks2 > 1 else 0)) * mpw.CNT_STRIDE
    assert mpw.counters(qb_rows) == expected_cnt


def test_invalid_rows_and_qb_rows_rejected() -> None:
    with pytest.raises(AssertionError):
        mpw.tile_shape(0)
    with pytest.raises(AssertionError):
        mpw.tile_shape(mpw.MAX_ROWS + 1)
    with pytest.raises(AssertionError):
        mpw.stage2_spec(100)
    with pytest.raises(AssertionError):
        mpw.mla_pre_wide(16, launches=0)
    with pytest.raises(AssertionError):
        mpw.mla_pre_wide(16, launches=mpw.TAG_SLOTS + 1)


def _fake_mla_layer(
    *,
    use_dsa=True,
    output_gate=True,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_head_dim=192,
    qkv_dtype=torch.bfloat16,
    qb_dtype=torch.bfloat16,
    qb_shape=(mpw.QB_LOCAL, mpw.QL),
    q_nw_contig=True,
    has_bias=False,
    dcp_size=1,
    qrep_active=False,
    qrep_dtype=torch.bfloat16,
    prepare_fn=_qrep_prepare,
):
    def z(*s, dt=torch.bfloat16):
        return torch.zeros(*s, dtype=dt)

    q_nw = z(mpw.QL) if q_nw_contig else z(mpw.QL * 2)[::2]
    idx = SimpleNamespace(
        wk=SimpleNamespace(weight=z(mpw.IK, wide.H), bias=None),
        weights_proj=SimpleNamespace(weight=z(mpw.IW, wide.H), bias=None),
        wq_b=SimpleNamespace(weight=z(mpw.IQ, mpw.QL), bias=None),
    )
    attn = SimpleNamespace(
        use_dsa=use_dsa,
        use_output_gate=output_gate,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_head_dim,
        num_local_heads=12,
        dcp_size=dcp_size,
        use_deep_gemm_bmm=False,
        w_kc_qrep=z(1) if qrep_active else None,
        q_b_proj_qrep_weight=z(mpw.QB_QREP, mpw.QL, dt=qrep_dtype)
        if qrep_active
        else None,
        is_dcp_q_replicate_phase=lambda fb: qrep_active,
        forward_absorb_rocm_prepare=prepare_fn,
        indexer=idx,
        fused_qkv_a_proj_with_mqa=SimpleNamespace(
            weight=z(mpw.LATW, wide.H, dt=qkv_dtype),
            bias=z(mpw.LATW) if has_bias else None,
        ),
        g_proj=SimpleNamespace(weight=z(mpw.GATE, wide.H), bias=None),
        q_b_proj=SimpleNamespace(weight=z(*qb_shape, dt=qb_dtype), bias=None),
        q_a_layernorm=SimpleNamespace(weight=q_nw, variance_epsilon=1e-6),
        kv_a_layernorm=SimpleNamespace(weight=z(mpw.KV), variance_epsilon=1e-6),
    )
    return SimpleNamespace(self_attn=attn)


def test_mla_pre_fusable_gates(monkeypatch) -> None:
    """Every safety gate in _mla_pre_fusable rejects incompatible layers."""
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE", raising=False)
    assert wide._mla_pre_fusable(_fake_mla_layer()) is True
    assert (
        wide._mla_pre_fusable(_fake_mla_layer(prepare_fn=_local_only_prepare)) is True
    )

    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE", "0")
    assert wide._mla_pre_fusable(_fake_mla_layer()) is False
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE", "1")

    for bad in (
        {"use_dsa": False},
        {"output_gate": False},
        {"q_lora_rank": 512},
        {"kv_lora_rank": 256},
        {"qk_rope_head_dim": 32},
        {"qk_head_dim": 128},
        {"qkv_dtype": torch.float8_e4m3fn},
        {"qb_dtype": torch.float8_e4m3fn},
        {"qb_shape": (1024, mpw.QL)},
        {"q_nw_contig": False},
        {"has_bias": True},
        {"prepare_fn": _no_mono_prepare},
        {"prepare_fn": None},
    ):
        assert wide._mla_pre_fusable(_fake_mla_layer(**bad)) is False, bad


def test_mono_pre_caps_and_verify_qb_consistency(monkeypatch) -> None:
    """Allocation, warmup, and dispatch share the exact per-layer verify eligibility."""
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", raising=False)
    # 1. Pre-_mono_pre SGLang (e.g. 703e763d5f): disabled for both local and qrep
    old_local = _fake_mla_layer(dcp_size=1, prepare_fn=_no_mono_prepare)
    assert wide._sglang_mono_pre_caps(old_local.self_attn) == (False, False)
    par_dcp1 = SimpleNamespace(dcp_enabled=False, dcp_replicate_q_proj=False)
    assert wide._mla_verify_qb(old_local, par_dcp1) is None

    # 2. Local-only _mono_pre SGLang (e.g. CI-pinned ba76f34b85) + DCP8 qrep:
    #    must skip MLA setup across ALL buckets (including b <= 48) since verify falls back
    par_dcp8 = SimpleNamespace(dcp_enabled=True, dcp_replicate_q_proj=True)
    ci_dcp8 = _fake_mla_layer(
        dcp_size=8, qrep_active=True, prepare_fn=_local_only_prepare
    )
    assert wide._sglang_mono_pre_caps(ci_dcp8.self_attn) == (True, False)
    assert wide._mla_verify_qb(ci_dcp8, par_dcp8) is None
    w = object.__new__(wide.Wide)
    w.L = 8
    w.buckets = wide.buckets(8)
    w.dsa_layers = {3: SimpleNamespace(i=3, layer=ci_dcp8, mla_pre_qb=None)}
    for b in wide.buckets(8):
        assert w._mla_bucket_specs(b) == {}

    # 3. Unsupported DCP width (dcp_size=4 -> 9216 != QB_QREP): skip allocation & warmup
    dcp4 = _fake_mla_layer(dcp_size=4, qrep_active=True, prepare_fn=_qrep_prepare)
    assert wide._mla_verify_qb(dcp4, par_dcp8) is None

    # 4. Local-only SGLang + DCP1: L=8 enables buckets 16/32 (64 skipped); L=6 bucket 30 runs m_mla=30
    ci_dcp1 = _fake_mla_layer(dcp_size=1, prepare_fn=_local_only_prepare)
    qb_local = wide._mla_verify_qb(ci_dcp1, par_dcp1)
    assert qb_local == mpw.QB_LOCAL
    w.dsa_layers = {3: SimpleNamespace(i=3, layer=ci_dcp1, mla_pre_qb=qb_local)}
    assert w._mla_bucket_specs(16) == {mpw.QB_LOCAL: 16}
    assert w._mla_bucket_specs(32) == {mpw.QB_LOCAL: 32}
    assert w._mla_bucket_specs(64) == {}
    w.L = 6
    w.buckets = wide.buckets(6)
    assert w._mla_bucket_specs(24) == {mpw.QB_LOCAL: 24}
    # dense ladder: rows 25..30 have bucket 30, so bucket 48 (rows 43..48) is past the cap
    assert w._mla_bucket_specs(30) == {mpw.QB_LOCAL: 30}
    assert w._mla_bucket_specs(48) == {}
    assert w._mla_bucket_specs(96) == {}

    # 5. PR #53130 SGLang + DCP8 qrep: L=8 enables buckets 16/32 (64 skipped); L=6 bucket 30 runs m_mla=30
    new_dcp8 = _fake_mla_layer(dcp_size=8, qrep_active=True, prepare_fn=_qrep_prepare)
    assert wide._sglang_mono_pre_caps(new_dcp8.self_attn) == (True, True)
    qb_qrep = wide._mla_verify_qb(new_dcp8, par_dcp8)
    assert qb_qrep == mpw.QB_QREP
    w.L = 8
    w.buckets = wide.buckets(8)
    w.dsa_layers = {3: SimpleNamespace(i=3, layer=new_dcp8, mla_pre_qb=qb_qrep)}
    assert w._mla_bucket_specs(16) == {mpw.QB_QREP: 16}
    assert w._mla_bucket_specs(32) == {mpw.QB_QREP: 32}
    assert w._mla_bucket_specs(64) == {}
    w.L = 6
    w.buckets = wide.buckets(6)
    assert w._mla_bucket_specs(24) == {mpw.QB_QREP: 24}
    # dense ladder: rows 25..30 have bucket 30, so bucket 48 (rows 43..48) is past the cap
    assert w._mla_bucket_specs(30) == {mpw.QB_QREP: 30}
    assert w._mla_bucket_specs(48) == {}
    assert w._mla_bucket_specs(96) == {}


def test_mono_pre_caps_read_the_q_view_head_count() -> None:
    """A branch that views pre["q"] with local heads is local-only even when
    q_replicate_active and self.dcp_size appear elsewhere in it: wide would hand a 96-head
    q to a 12-head view."""

    def _local_view_dcp_elsewhere(self, forward_batch):
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
        return self._split_q_nope_pe(q, None)

    def _qrep_view_inline(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_inverted(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (1 if q_replicate_active else self.dcp_size),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_reassigned_after_view(self, forward_batch):
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
        return self._split_q_nope_pe(q, heads)

    def _qrep_nested_override(self, forward_batch):
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
            q = pre["q"].view(-1, heads, self.qk_head_dim)
        return self._split_q_nope_pe(q, None)

    def _qrep_literal(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (5 if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_local_view_on_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            if q_replicate_active:
                q = pre["q"].view(-1, self.num_local_heads, self.qk_head_dim)
                return q
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_rebound_predicate(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        q_replicate_active = q_replicate_active and self.dcp_size > 8
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_other_predicate(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch) and self.dcp_size > 8
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_tuple_view_on_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            return (
                pre["q"].view((-1, self.num_local_heads, self.qk_head_dim))
                if q_replicate_active
                else pre["q"].view(
                    -1,
                    self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                    self.qk_head_dim,
                )
            )
        return None

    def _qrep_alias_reshapes_on_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if (handoff := pre) is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            if q_replicate_active:
                q = handoff["q"].view(-1, self.num_local_heads, self.qk_head_dim)
        return self._split_q_nope_pe(q, None)

    def _qrep_half_head_dim(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim // 2,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_reshaped_after_view(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            if q_replicate_active:
                q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        return self._split_q_nope_pe(q, None)

    def _qrep_rebound_batch(self, forward_batch):
        forward_batch = forward_batch.attn_sub_batch
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_unrelated_walrus(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            if (n := self.num_local_heads) > 0:
                _ = n
        return self._split_q_nope_pe(q, None)

    def _qrep_other_handoff(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_other")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            handoff = self.__dict__.get("_mono_pre")
        return self._split_q_nope_pe(q, None)

    def _qrep_reshaped_on_return(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return q.reshape(-1, self.num_local_heads, self.qk_head_dim)

    def _qrep_match_rebinds_batch(self, forward_batch):
        match forward_batch:
            case SimpleNamespace(attn_sub_batch=forward_batch):
                pass
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_shifted_dcp_factor(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size - 3 if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_guard_skips_the_qrep_state(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None and not q_replicate_active:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_split_bypasses_q(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(
            self.__dict__["_mono_pre"]["q"].view(
                -1, self.num_local_heads, self.qk_head_dim
            ),
            None,
        )

    def _qrep_imported_batch(self, forward_batch):
        from sglang.srt.model_executor import forward_batch

        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_second_handoff_read(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            stale = self.__dict__["_mono_pre"]
        return self._split_q_nope_pe(q, None)

    def _qrep_split_never_gets_q(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(pre["q_lora"], None)

    def _qrep_q_in_the_second_slot(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(pre["q_lora"], q)

    def _qrep_early_exit_before_split(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        if q_replicate_active:
            return self._prepare_fallback(forward_batch)
        return self._split_q_nope_pe(q, None)

    def _qrep_early_exit_in_branch(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            if q_replicate_active:
                return self._prepare_fallback(forward_batch)
        return self._split_q_nope_pe(q, None)

    def _qrep_handoff_field_write(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if self.q_lora_rank is not None and pre is not None:
            q = pre["q"].view(
                pre["q_lora"].shape[0],
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            pre["q_lora"] = pre["q_lora"][:1]
        return self._split_q_nope_pe(q, None)

    def _qrep_split_only_off_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return (
            self._prepare_fallback(forward_batch)
            if q_replicate_active
            else self._split_q_nope_pe(q, None)
        )

    def _qrep_asserts_off_the_qrep_path(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
            assert not q_replicate_active
        return self._split_q_nope_pe(q, None)

    def _qrep_rebinds_self(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            self = self.other
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_exits_before_the_guard(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if q_replicate_active:
            return self._prepare_fallback(forward_batch)
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_writes_dcp_size(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        self.dcp_size = 1
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_setattr_before_the_guard(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        ignored = setattr(self, "dcp_size", 1)
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    def _qrep_setattr_before_the_view(self, forward_batch):
        q_replicate_active = (
            self.is_dcp_q_replicate_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        pre = self.__dict__.get("_mono_pre")
        if pre is not None:
            ignored = setattr(self, "dcp_size", 1)
            q = pre["q"].view(
                -1,
                self.num_local_heads * (self.dcp_size if q_replicate_active else 1),
                self.qk_head_dim,
            )
        return self._split_q_nope_pe(q, None)

    cases = (
        (_local_view_dcp_elsewhere, (True, False)),
        (_qrep_view_inline, (True, True)),
        (_qrep_inverted, (False, False)),
        (_qrep_reassigned_after_view, (True, False)),
        (_qrep_nested_override, (False, False)),
        (_qrep_literal, (True, False)),
        (_qrep_local_view_on_the_qrep_path, (False, False)),
        (_qrep_rebound_predicate, (False, False)),
        (_qrep_other_predicate, (False, False)),
        (_qrep_tuple_view_on_the_qrep_path, (False, False)),
        (_qrep_alias_reshapes_on_the_qrep_path, (False, False)),
        (_qrep_half_head_dim, (False, False)),
        (_qrep_reshaped_after_view, (False, False)),
        (_qrep_rebound_batch, (False, False)),
        (_qrep_unrelated_walrus, (False, False)),
        (_qrep_other_handoff, (False, False)),
        (_qrep_reshaped_on_return, (False, False)),
        (_qrep_match_rebinds_batch, (False, False)),
        (_qrep_shifted_dcp_factor, (True, False)),
        (_qrep_guard_skips_the_qrep_state, (False, False)),
        (_qrep_split_bypasses_q, (False, False)),
        (_qrep_imported_batch, (False, False)),
        (_qrep_second_handoff_read, (False, False)),
        (_qrep_split_never_gets_q, (False, False)),
        (_qrep_q_in_the_second_slot, (False, False)),
        (_qrep_early_exit_before_split, (False, False)),
        (_qrep_early_exit_in_branch, (False, False)),
        (_qrep_handoff_field_write, (False, False)),
        (_qrep_split_only_off_the_qrep_path, (False, False)),
        (_qrep_asserts_off_the_qrep_path, (False, False)),
        (_qrep_rebinds_self, (False, False)),
        (_qrep_exits_before_the_guard, (False, False)),
        (_qrep_writes_dcp_size, (False, False)),
        (_qrep_setattr_before_the_guard, (False, False)),
        (_qrep_setattr_before_the_view, (False, False)),
    )
    wrong = {
        prepare.__name__: got
        for prepare, caps in cases
        if (
            got := wide._sglang_mono_pre_caps(
                SimpleNamespace(forward_absorb_rocm_prepare=prepare)
            )
        )
        != caps
    }
    assert not wrong, wrong


def test_mla_row_cap_respects_reachable_verify_rows(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", raising=False)
    runner = object.__new__(wide.Wide)
    runner.L = 6
    runner.buckets = (12, 18, 24, 30, 36, 42, 48)
    runner.dsa_layers = {3: SimpleNamespace(mla_pre_qb=mpw.QB_QREP)}
    assert runner._mla_bucket_specs(30) == {mpw.QB_QREP: 30}
    assert runner._mla_bucket_specs(36) == {}
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", "48")
    for bucket in (36, 42, 48):
        assert runner._mla_bucket_specs(bucket) == {mpw.QB_QREP: bucket}


def test_mla_first_bucket_starts_at_the_served_lower_cut(monkeypatch) -> None:
    """The first bucket's smallest step is the lowest served width, not L rows: a launch
    capped below it is never reached, so it must not be allocated or warmed."""
    monkeypatch.delenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", raising=False)
    runner = object.__new__(wide.Wide)
    runner.dsa_layers = {3: SimpleNamespace(mla_pre_qb=mpw.QB_QREP)}
    runner.L = 6
    runner.buckets = wide.buckets(6)
    # 6-row steps are narrow: bucket 12's smallest wide step is 2 requests, 12 rows
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", "6")
    assert runner._mla_bucket_specs(12) == {}
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", "12")
    assert runner._mla_bucket_specs(12) == {mpw.QB_QREP: 12}
    # an override's first bucket serves only past the ladder bucket below it
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_BUCKETS", "128,256,384")
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", "48")
    lo = max(r for r in wide.REQ_BUCKETS if r < 16)
    runner.L = 8
    runner.buckets = wide.buckets(8)
    assert 8 * (lo + 1) > 48
    assert runner._mla_bucket_specs(128) == {}


@pytest.mark.parametrize("max_rows", ["0", str(mpw.MAX_ROWS + 1), "invalid"])
def test_mla_row_cap_rejects_unsupported_launch_bounds(monkeypatch, max_rows) -> None:
    monkeypatch.setenv("SGLANG_AMD_MONO_K31_WIDE_MLA_PRE_MAX_ROWS", max_rows)
    with pytest.raises(ValueError):
        wide._mla_max_rows(mpw.QB_QREP)


@pytest.mark.parametrize(
    "dcp_size,qrep,want_qb", [(1, False, mpw.QB_LOCAL), (8, True, mpw.QB_QREP)]
)
def test_run_mla_pre_wiring_and_partial_padding(
    monkeypatch, dcp_size: int, qrep: bool, want_qb: int
) -> None:
    """_run_mla_pre selects DCP1/DCP8 weights, passes zero-copy vs padded input, and returns matching _mono_pre views."""
    recorded = []

    def fake_builder(m, qb_rows, launches=1, q_eps=1e-6, kv_eps=1e-6, timeline=False):
        def _launch(*args, stream=None):
            recorded.append((m, qb_rows, launches, args))

        return _launch, None

    monkeypatch.setattr(mpw, "mla_pre_wide", fake_builder)
    dispatched = _dispatch_directly(monkeypatch)

    b, rows = 32, 24
    layer = _fake_mla_layer(dcp_size=dcp_size, qrep_active=qrep)
    pk = SimpleNamespace(
        i=3, layer=layer, mla_pre=True, mla_pre_qb=want_qb, q_eps=1e-6, kv_eps=1e-6
    )
    w = object.__new__(wide.Wide)
    w.dsa_layers = {3: pk}
    w.slots_dsa = {3: 0}
    w.epoch = torch.zeros(4, dtype=torch.int32)

    def z(*s, dt=torch.bfloat16):
        return torch.zeros(*s, dtype=dt)

    def make_bufs(m_mla: int):
        return dict(
            mla_specs={want_qb: m_mla},
            mla_state={want_qb: (None, z(512, dt=torch.uint8), z(128, dt=torch.int32))},
            mla_x=torch.ones(m_mla, wide.H, dtype=torch.bfloat16),
            mla_out=dict(
                q=z(m_mla * want_qb),
                latent=z(m_mla, mpw.KV + mpw.ROPE),
                k_nope=z(m_mla, mpw.KV),
                q_lora=z(m_mla, mpw.QL),
                gate=z(m_mla, mpw.GATE),
                idx_q=z(m_mla, mpw.IQ),
                idx_k=z(m_mla, mpw.IK),
                idx_w=z(m_mla, mpw.IW),
            ),
        )

    bufs = make_bufs(b)

    # 1. Full bucket contiguous: zero-copy input, shares w.epoch
    h_full = torch.randn(b, wide.H, dtype=torch.bfloat16)
    attn_pre, idx_pre = w._run_mla_pre(pk, bufs, h_full, fb=None)
    assert len(recorded) == 1
    assert recorded[0][0] == b and recorded[0][1] == want_qb
    assert recorded[0][3][0] == h_full.data_ptr()
    assert recorded[0][3][19] == w.epoch.data_ptr()
    assert tuple(attn_pre["q"].shape) == (b, want_qb)
    assert tuple(attn_pre["latent"].shape) == (b, mpw.KV + mpw.ROPE)
    assert tuple(idx_pre["q"].shape) == (b, mpw.IQ)

    # 2. Partial bucket: copies + zeroes tail rows, slices outputs to [:rows]
    h_part = torch.randn(rows, wide.H, dtype=torch.bfloat16)
    attn_pre2, idx_pre2 = w._run_mla_pre(pk, bufs, h_part, fb=None)
    assert recorded[1][3][0] == bufs["mla_x"].data_ptr()
    assert torch.equal(bufs["mla_x"][:rows], h_part)
    assert torch.count_nonzero(bufs["mla_x"][rows:]) == 0
    assert tuple(attn_pre2["q"].shape) == (rows, want_qb)
    assert tuple(idx_pre2["w"].shape) == (rows, mpw.IW)

    # 3. A synthetic buffer capped at m_mla=32: runs rows=30/32 (zero-copy at 32), falls back at 36+
    m_cap = mpw.MAX_ROWS_QREP if qrep else mpw.MAX_ROWS_LOCAL
    bufs_cap = make_bufs(m_cap)
    h30 = torch.randn(30, wide.H, dtype=torch.bfloat16)
    res30 = w._run_mla_pre(pk, bufs_cap, h30, fb=None)
    assert res30 is not None and tuple(res30[0]["q"].shape) == (30, want_qb)
    assert recorded[-1][0] == m_cap
    h32 = torch.randn(32, wide.H, dtype=torch.bfloat16)
    res32 = w._run_mla_pre(pk, bufs_cap, h32, fb=None)
    assert res32 is not None and tuple(res32[0]["q"].shape) == (32, want_qb)
    assert recorded[-1][3][0] == h32.data_ptr()
    assert (
        w._run_mla_pre(
            pk, bufs_cap, torch.randn(36, wide.H, dtype=torch.bfloat16), fb=None
        )
        is None
    )
    assert w._run_mla_pre(pk, {}, h_full, fb=None) is None

    # 4. Shape guards and runtime/warmup mismatch safely fall back without launching
    assert (
        w._run_mla_pre(pk, bufs, torch.empty(0, wide.H, dtype=torch.bfloat16), fb=None)
        is None
    )
    assert (
        w._run_mla_pre(
            pk, bufs, torch.randn(b + 1, wide.H, dtype=torch.bfloat16), fb=None
        )
        is None
    )
    assert (
        w._run_mla_pre(
            pk, bufs, torch.randn(b, wide.H - 1, dtype=torch.bfloat16), fb=None
        )
        is None
    )
    assert (
        w._run_mla_pre(
            pk, bufs, torch.randn(1, b, wide.H, dtype=torch.bfloat16), fb=None
        )
        is None
    )

    # Runtime phase flip opposite to warmed pk.mla_pre_qb must fall back, not JIT-compile unwarmed variant
    layer.self_attn.is_dcp_q_replicate_phase = lambda fb: not qrep
    if not qrep:
        layer.self_attn.dcp_size = 8
        layer.self_attn.w_kc_qrep = z(1)
        layer.self_attn.q_b_proj_qrep_weight = z(mpw.QB_QREP, mpw.QL)
    assert w._run_mla_pre(pk, bufs, h_full, fb=None) is None
    # the four launches above went through the compiled dispatch; no fallback launched
    assert len(recorded) == 4 and len(dispatched) == 4


def test_dsa_step_sets_and_pops_mono_pre_on_normal_and_exception(monkeypatch) -> None:
    """_dsa_step injects _mono_pre into attn and indexer and always pops both in finally."""
    layer = _fake_mla_layer()
    rows, b = 16, 16
    h = torch.randn(rows, wide.H, dtype=torch.bfloat16)
    prefix = torch.randn(rows, wide.H, dtype=torch.bfloat16)
    layer.attn_boundary = SimpleNamespace(
        prepare=lambda hidden, fb, capture_gathered=None: h
    )
    pk = SimpleNamespace(i=3, layer=layer, moe=SimpleNamespace())

    fake_batch_mod = SimpleNamespace(written_residual=lambda fb: prefix)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.layer_boundary.residual.batch",
        fake_batch_mod,
    )

    seen_pre = []
    attn_pre = {"q": "attn_q"}
    idx_pre = {"q": "idx_q"}

    w = object.__new__(wide.Wide)
    w.slots_dsa = {3: 0}
    w._run_mla_pre = lambda pk_, bufs_, h_, fb_: (attn_pre, idx_pre)
    w._build_dsa_launch = lambda pk_, b_, cfg_: lambda *args, stream=None: None
    w._dsa_launch_args = lambda *args: ()
    dispatched = _dispatch_directly(monkeypatch)
    monkeypatch.setattr(wide, "_routed", lambda *args, **kwargs: None)

    def ok_pre_o_proj(layer_, x_, positions_, fb_, zero_allocator_):
        seen_pre.append(
            (
                layer_.self_attn.__dict__.get("_mono_pre"),
                layer_.self_attn.indexer.__dict__.get("_mono_pre"),
            )
        )
        return torch.zeros(rows, wide.KO, dtype=torch.bfloat16)

    monkeypatch.setattr(wide, "_pre_o_proj", ok_pre_o_proj)
    bufs = dict(
        cfg=None,
        lat=torch.zeros(b, 1),
        logits=torch.zeros(b, 1),
        ids=torch.zeros(b, 1),
        w=torch.zeros(b, 1),
        lat_p=[torch.zeros(b, 1), torch.zeros(b, 1)],
        sortv=None,
    )

    assert w._dsa_step(pk, h, None, None, None, b, bufs, rows, ping=0) == 1
    assert seen_pre == [(attn_pre, idx_pre)]
    assert len(dispatched) == 1  # the dsa_post launch, through the compiled dispatch
    assert "_mono_pre" not in layer.self_attn.__dict__
    assert "_mono_pre" not in layer.self_attn.indexer.__dict__

    def failing_pre_o_proj(*args, **kwargs):
        assert layer.self_attn.__dict__.get("_mono_pre") is attn_pre
        assert layer.self_attn.indexer.__dict__.get("_mono_pre") is idx_pre
        raise RuntimeError("boom")

    monkeypatch.setattr(wide, "_pre_o_proj", failing_pre_o_proj)
    with pytest.raises(RuntimeError, match="boom"):
        w._dsa_step(pk, h, None, None, None, b, bufs, rows, ping=0)
    assert "_mono_pre" not in layer.self_attn.__dict__
    assert "_mono_pre" not in layer.self_attn.indexer.__dict__


def test_mla_warmup_preserves_tp_epoch_and_clears_scratch(monkeypatch) -> None:
    """Rank-local MLA warmup must not mutate self.epoch and must clear warmup scratch tags."""
    recorded = []

    def fake_builder(m, qb_rows, launches=1, q_eps=1e-6, kv_eps=1e-6, timeline=False):
        def _launch(*args, stream=None):
            recorded.append((m, qb_rows, args[19]))

        return _launch, None

    monkeypatch.setattr(mpw, "mla_pre_wide", fake_builder)
    dispatched = _dispatch_directly(monkeypatch)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    b = 32
    layer = _fake_mla_layer(dcp_size=8, qrep_active=True)
    pk = SimpleNamespace(
        i=3,
        layer=layer,
        mla_pre=True,
        mla_pre_qb=mpw.QB_QREP,
        nvb=1,
        front_rows=None,
        eps=1e-6,
        q_eps=1e-6,
        kv_eps=1e-6,
        o_fmt=None,  # the warmup only keys its DSA builds on it
    )
    w = object.__new__(wide.Wide)
    w.L = 8
    w.state_bf16, w.ring = False, True  # an fp32 KDA state on the ReplaySSM ring
    w.device = torch.device("cpu")
    w.group = None
    w.layers = {}
    w.dsa_layers = {3: pk}
    w.slots_dsa = {3: 0}
    w.buckets = (b,)
    w.epoch = torch.tensor([7, 0, 123, 456], dtype=torch.int32)
    w._build_dsa_launch = lambda pk_, b_, cfg_: lambda *args, stream=None: None
    w._dsa_launch_args = lambda *args: ()

    def z(*s, dt=torch.bfloat16):
        return torch.zeros(*s, dtype=dt)

    scr_local = torch.full((512,), 0x3C, dtype=torch.uint8)
    scr_qrep = torch.full((512,), 0x5A, dtype=torch.uint8)
    bufs = dict(
        cfg=None,
        mla_specs={mpw.QB_LOCAL: b, mpw.QB_QREP: b},
        mla_state={
            mpw.QB_LOCAL: (None, scr_local, z(128, dt=torch.int32)),
            mpw.QB_QREP: (None, scr_qrep, z(128, dt=torch.int32)),
        },
        core=z(b, wide.KO),
        pout=[z(b, wide.H), z(b, wide.H)],
        mla_x=z(b, wide.H),
        mla_out=dict(
            q=z(b * mpw.QB_QREP),
            latent=z(b, mpw.KV + mpw.ROPE),
            k_nope=z(b, mpw.KV),
            q_lora=z(b, mpw.QL),
            gate=z(b, mpw.GATE),
            idx_q=z(b, mpw.IQ),
            idx_k=z(b, mpw.IK),
            idx_w=z(b, mpw.IW),
        ),
    )
    w._ensure = lambda *, b, L: bufs

    w.warmup(buckets=(b,))

    # Collective dsa_post bumped self.epoch[0] once (7 -> 8); MLA warmup left it untouched.
    assert int(w.epoch[0].item()) == 8
    assert len(recorded) == 1
    assert len(dispatched) == 2  # one dsa_post and one mla_pre_wide warmup launch
    assert recorded[0][0] == b and recorded[0][1] == mpw.QB_QREP
    assert recorded[0][2] != w.epoch.data_ptr()
    assert torch.count_nonzero(scr_qrep) == 0
    # Unwarmed layout's isolated scratch is untouched
    assert torch.count_nonzero(scr_local) == scr_local.numel()
