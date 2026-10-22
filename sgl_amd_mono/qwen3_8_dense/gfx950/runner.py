# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang PR #43558 (Apache-2.0):
# python/sglang/srt/layers/qwen3_8_dense_mono.py
"""The device state the dense launches share (the epoch and one scratch buffer
per kernel), their precompile, and one call per launch with its outputs."""

from __future__ import annotations

import inspect
import os
from typing import Optional

import msgspec
import torch

from sgl_amd_mono.common import guard

from . import dense_ffn as D
from . import dense_gdn as K


class DenseFfnWeights(msgspec.Struct, frozen=True):
    """A layer's K2 operands: MXFP4 weights shuffled, scales as loaded."""

    w13: torch.Tensor
    w13s: torch.Tensor
    w2: torch.Tensor
    w2s: torch.Tensor
    ln_w: torch.Tensor  # post_attention_layernorm
    w_o: Optional[torch.Tensor] = None  # TP1: o_proj / out_proj, folded in
    w_os: Optional[torch.Tensor] = None


class DenseGdnWeights(msgspec.Struct, frozen=True):
    """A GDN layer's K1 operands: MXFP4 weights shuffled, the rest as loaded."""

    w_qkvz: torch.Tensor
    w_qkvzs: torch.Tensor
    w_ba: torch.Tensor
    w_bas: torch.Tensor
    conv_w: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_w: torch.Tensor
    ln_w: torch.Tensor  # input_layernorm
    w_o: Optional[torch.Tensor] = None  # TP2: out_proj, folded in
    w_os: Optional[torch.Tensor] = None


class DenseGdnState(msgspec.Struct, frozen=True):
    """A GDN layer's mamba pool cache, which K1 updates in place."""

    conv_state: torch.Tensor  # bf16 [slots, conv, CONV_W - 1]
    rstate: torch.Tensor  # f32 [slots, nv, HD, HD]


def compile_only(launch) -> None:
    """Compile a FlyDSL launch into its caches without running it."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    env = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(env)
    os.environ[env] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = old


def ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL's disk cache at ~/.flydsl/cache when the configured one is
    missing or read-only."""
    path = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if path:
        try:
            os.makedirs(path, exist_ok=True)
            if os.access(path, os.W_OK):
                return
        except OSError:
            pass
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = os.path.expanduser("~/.flydsl/cache")


class DenseRuntime:
    """The epoch (word 0 the step, ``guard.GUARD_DEADLINE`` the deadline) and
    the K1 / K2 scratch, sized for ``max_rows`` rows at TP ``tp``."""

    def __init__(
        self,
        *,
        tp: int,
        max_rows: int,
        eps: float,
        norm_eps: float,
        fault_site: int,
        fault_after: int,
        device: torch.device,
    ):
        assert tp in K.TP_SIZES and max_rows <= D.max_tokens(tp)
        self.tp, self.max_rows, self.device = tp, max_rows, device
        self.eps, self.norm_eps = eps, norm_eps
        self.fault = dict(fault_site=fault_site, fault_after=fault_after)
        self.geo = K.Geo(tp)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        k1 = max(K.scratch_bytes(k) for k in self._k1_keys(range(1, max_rows + 1)))
        k2 = max(D.scratch_bytes(k) for k in self._k2_keys(range(1, max_rows + 1)))
        # zeroed once: the kernels' flags are tagged with the epoch step
        self.k1_scratch = torch.zeros(k1, dtype=torch.uint8, device=device)
        self.k2_scratch = torch.zeros(k2, dtype=torch.uint8, device=device)

    def k1_key(self, s: int, *, first: bool = False) -> K.K1Build:
        if self.tp == 1:
            return K.K1Build(
                tokens=s,
                first=first,
                eps=self.eps,
                norm_eps=self.norm_eps,
                **self.fault,
            )
        return K.K1Build(
            tokens=s,
            norm_eps=self.norm_eps,
            tp=self.tp,
            normed=True,
            oproj=True,
            **self.fault,
        )

    def k2_key(self, s: int) -> D.FfnBuild:
        if self.tp == 1:
            return D.FfnBuild(
                tokens=s, eps=self.eps, shuffled=True, oproj=True, **self.fault
            )
        return D.FfnBuild(
            tokens=s, shuffled=True, tp=self.tp, normed=True, **self.fault
        )

    def _k1_keys(self, widths) -> list:
        firsts = (False, True) if self.tp == 1 else (False,)
        return [self.k1_key(s, first=f) for s in widths for f in firsts]

    def _k2_keys(self, widths) -> list:
        return [self.k2_key(s) for s in widths]

    def precompile(self, widths) -> None:
        for key in self._k1_keys(widths):
            compile_only(K.build(key))
        for key in self._k2_keys(widths):
            compile_only(D.build(key))

    def set_deadline(self, s: float) -> None:
        ticks = guard.deadline_ticks(s)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def advance_step(self) -> None:
        self.epoch[:1].add_(1)

    def gdn(
        self,
        w: DenseGdnWeights,
        state: DenseGdnState,
        idx: torch.Tensor,
        h: torch.Tensor,
        residual: Optional[torch.Tensor],
        *,
        layer: int,
    ):
        """K1 on ``s`` rows: TP1 takes the un-normed stream and returns
        ``(core, res_out, None)``; TP2 takes the normed input and returns
        ``(core, None, out)``, ``out`` this rank's out_proj share."""
        s = h.shape[0]
        core = h.new_empty(s, self.geo.val)
        res_out = out = None
        extra = {}
        if self.tp == 1:
            res_out = torch.empty_like(h)
            extra = dict(residual=residual, res_out=res_out, ln_w=w.ln_w)
            key = self.k1_key(s, first=residual is None)
        else:
            out = h.new_empty(s, K.HIDDEN)
            extra = dict(w_o=w.w_o, w_os=w.w_os, out=out)
            key = self.k1_key(s)
        K.dense_gdn_pre(
            key,
            hidden=h,
            w_qkvz=w.w_qkvz,
            w_qkvzs=w.w_qkvzs,
            w_ba=w.w_ba,
            w_bas=w.w_bas,
            conv_w=w.conv_w,
            conv_state=state.conv_state,
            a_log=w.a_log,
            dt_bias=w.dt_bias,
            norm_w=w.norm_w,
            rstate=state.rstate,
            st_idx=idx,
            core=core,
            scratch=self.k1_scratch,
            epoch=self.epoch,
            layer=layer,
            **extra,
        )
        return core, res_out, out

    def ffn(
        self,
        w: DenseFfnWeights,
        core: torch.Tensor,
        residual: torch.Tensor,
        *,
        layer: int,
    ):
        """TP1 K2: o_proj on ``core``, the residual add, post-norm and MLP;
        returns ``(out, res_out)``."""
        s = core.shape[0]
        out = residual.new_empty(s, D.HIDDEN)
        res_out = torch.empty_like(residual)
        D.dense_ffn(
            self.k2_key(s),
            hidden=core,
            residual=residual,
            ln_w=w.ln_w,
            w13=w.w13,
            w13s=w.w13s,
            w2=w.w2,
            w2s=w.w2s,
            out=out,
            res_out=res_out,
            scratch=self.k2_scratch,
            epoch=self.epoch,
            layer=layer,
            w_o=w.w_o,
            w_os=w.w_os,
        )
        return out, res_out

    def ffn_partial(self, w: DenseFfnWeights, x: torch.Tensor, *, layer: int):
        """TP2 K2: this rank's MLP share on the post-normed ``x``, before the
        all-reduce."""
        out = x.new_empty(x.shape[0], D.HIDDEN)
        D.dense_ffn(
            self.k2_key(x.shape[0]),
            hidden=x,
            residual=None,
            ln_w=None,
            w13=w.w13,
            w13s=w.w13s,
            w2=w.w2,
            w2s=w.w2s,
            out=out,
            res_out=None,
            scratch=self.k2_scratch,
            epoch=self.epoch,
            layer=layer,
        )
        return out
