# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the FFN launch (one compiled kernel, scratch region and the shared
epoch word per row count of one TP rank) and of the attention front."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from sgl_amd_mono.common import guard

from .ffn import (
    ATTN_K,
    BLOCKS,
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    THREADS,
    Experts,
    FfnBuild,
    build_ffn,
    kernel_name,
    scratch_bytes,
    scratch_layout,
)
from .residency import active_workgroups_per_cu

# every wait's deadline until the spec sets SGLang's (BuildContext.deadline_s)
DEFAULT_DEADLINE_S = 60.0


def _writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled read-only cache."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


class FfnRunner:
    """``sym``: this rank's peer buffer; ``peers``: every rank's, int64 on
    device. The kernels move the epoch on themselves, so every rank that runs
    the same launches agrees on it. Runners on one peer buffer must share
    ``epoch``: a tag repeated over the buffer's lifetime could match a stale
    pair."""

    def __init__(
        self,
        tp: int,
        rank: int,
        sym: int,
        peers: torch.Tensor,
        device: torch.device,
        widths: tuple[int, ...] = tuple(range(1, MAX_TOKENS + 1)),
        experts: Experts = Experts.ALL,
        timeline: bool = False,
        epoch: torch.Tensor | None = None,
        **build,
    ) -> None:
        _writable_flydsl_cache()
        self.tp, self.rank, self.sym, self.peers = tp, rank, sym, peers
        self.widths = widths
        fault = guard.fault_injection(rank)
        if fault is not None:
            build = {"fault_site": fault[0], "fault_after": fault[1], **build}
        self.keys = {
            s: FfnBuild(tokens=s, tp=tp, experts=experts, timeline=timeline, **build)
            for s in widths
        }
        if epoch is None:
            epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=device)
        self.epoch = epoch
        self.set_deadline(DEFAULT_DEADLINE_S)
        self.scratch = {
            s: torch.zeros(scratch_bytes(s), dtype=torch.uint8, device=device)
            for s in widths
        }
        self._launch = {s: build_ffn(key) for s, key in self.keys.items()}
        self._compiled: dict[int, Any] = {}
        # every width's driver-reported resident CTAs per CU
        self.residency: dict[int, int] = {}

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, as the guard block's clock ticks."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def _args(self, s, attn, res_in, res_out, out, slots, w, mode):
        return (
            attn.data_ptr(),
            res_in.data_ptr(),
            res_out.data_ptr(),
            out.data_ptr(),
            slots.data_ptr(),
            w.post_norm.data_ptr(),
            w.router.data_ptr(),
            w.router_bias.data_ptr(),
            w.w13.data_ptr(),
            w.s13.data_ptr(),
            w.w2.data_ptr(),
            w.s2.data_ptr(),
            w.w13s.data_ptr(),
            w.s13s.data_ptr(),
            w.w2s.data_ptr(),
            w.s2s.data_ptr(),
            w.wo.data_ptr(),
            w.wos.data_ptr(),
            self.scratch[s].data_ptr(),
            self.sym,
            self.peers.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            mode,
            torch.cuda.current_stream(),
        )

    def warm(self, weights, compute_units: int | None = None) -> None:
        """Compile and load every width (a launch that returns at once); with
        ``compute_units``, refuse a width whose grid cannot be resident."""
        import flydsl.compiler as flyc

        dummy = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=self.epoch.device)
        for s in self.widths:
            if s not in self._compiled:
                args = self._args(s, dummy, dummy, dummy, dummy, dummy, weights, 0)
                self._compiled[s] = flyc.compile(self._launch[s], *args)
            if compute_units is not None and s not in self.residency:
                name = kernel_name(self.keys[s])
                per_cu = active_workgroups_per_cu(self._compiled[s], name, THREADS)
                self.residency[s] = per_cu
                if per_cu * compute_units < BLOCKS:
                    raise RuntimeError(
                        f"{name}: {per_cu} resident CTA(s) per CU x "
                        f"{compute_units} CUs < its {BLOCKS}-CTA grid"
                    )

    def __call__(
        self,
        attn: torch.Tensor,
        residual: torch.Tensor,
        weights,
        slots: torch.Tensor,
    ):
        """``attn``: this rank's attention output, o_proj's input; ``slots``: the
        step's int64 slot mapping. A row with a negative slot (padding) still runs
        the seam (its residual output is unspecified) but joins no expert and its
        MoE output is zero. Returns the reduced MoE output and the residual after
        attention."""
        s = attn.shape[0]
        assert attn.dtype == torch.bfloat16 and tuple(attn.shape) == (s, ATTN_K)
        assert attn.is_contiguous() and residual.is_contiguous()
        assert (
            slots.dtype == torch.int64 and slots.is_contiguous() and slots.numel() >= s
        )
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        args = self._args(s, attn, residual, res_out, out, slots, weights, 1)
        self._compiled[s](*args)
        return out, res_out

    def region(self, s: int, name: str) -> torch.Tensor:
        off, nbytes = scratch_layout(s)[name]
        return self.scratch[s][off : off + nbytes]


class FrontRunner:
    """Host side of the attention front (``front.py``): one compiled kernel per row
    count for the NVFP4 pool's layout, one scratch region (a launch's partials and
    every head's arrival counter, which the head's last CTA resets). Every width
    compiles on the first call: the pool exists only inside a forward, and SGLang
    runs a step eagerly before it captures one."""

    def __init__(self, device: torch.device, widths: tuple[int, ...]) -> None:
        from .front import SCRATCH_BYTES

        _writable_flydsl_cache()
        self.widths = widths
        self.scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self._compiled: dict[int, Any] = {}
        self._layout = None

    @staticmethod
    def layout(caches, cos_sin: torch.Tensor) -> dict:
        """The build parameters of a layer's pool views (``Nvfp4Cache`` K, V, index K)
        and the RoPE cache."""
        k, v, ik = caches
        assert k.payload.dim() == 4 and k.payload.shape[1] == 1, k.payload.shape
        assert v.payload.stride() == k.payload.stride()
        assert v.scales.stride() == k.scales.stride()
        assert ik.payload.dim() == 3 and ik.payload.shape[-2] == k.payload.shape[-2]
        assert cos_sin.dim() == 2 and cos_sin.shape[1] == 64 and cos_sin.is_contiguous()
        assert cos_sin.dtype in (torch.float32, torch.bfloat16)
        return dict(
            page=k.payload.shape[-2],
            kv_pay_page=k.payload.stride(0),
            kv_pay_row=k.payload.stride(2),
            kv_sc_page=k.scales.stride(0),
            kv_sc_row=k.scales.stride(2),
            ik_pay_page=ik.payload.stride(0),
            ik_pay_row=ik.payload.stride(1),
            ik_sc_page=ik.scales.stride(0),
            ik_sc_row=ik.scales.stride(1),
            cos_bf16=cos_sin.dtype == torch.bfloat16,
        )

    def __call__(self, x, residual, w, positions, slots, caches, cos_sin):
        """``x``: the previous FFN launch's output; ``residual``: its residual out;
        ``w``: the layer's front weights; ``caches``: its pool views. Stores the
        step's K, V and index K rows and returns (the residual entering the layer,
        Q, index Q)."""
        import flydsl.compiler as flyc

        from .front import HEAD, Q_HEADS, FrontBuild, build_front

        s = x.shape[0]
        assert x.dtype == torch.bfloat16 and tuple(x.shape) == (s, HIDDEN)
        assert x.is_contiguous() and residual.is_contiguous()
        assert positions.dtype == torch.int64 and slots.dtype == torch.int64
        assert positions.numel() >= s and slots.numel() >= s
        layout = self.layout(caches, cos_sin)
        if self._layout is None:
            self._layout = layout
        assert layout == self._layout, (layout, self._layout)
        res_new = torch.empty_like(residual)
        q = torch.empty(s, Q_HEADS * HEAD, dtype=torch.bfloat16, device=x.device)
        iq = torch.empty(s, HEAD, dtype=torch.bfloat16, device=x.device)
        k, v, ik = caches
        args = (
            x.data_ptr(),
            residual.data_ptr(),
            res_new.data_ptr(),
            w.input_norm.data_ptr(),
            w.wqkv.data_ptr(),
            w.wqkv_scale.data_ptr(),
            w.q_norm.data_ptr(),
            w.k_norm.data_ptr(),
            w.iq_norm.data_ptr(),
            w.ik_norm.data_ptr(),
            positions.data_ptr(),
            cos_sin.data_ptr(),
            slots.data_ptr(),
            k.payload.data_ptr(),
            k.scales.data_ptr(),
            v.payload.data_ptr(),
            v.scales.data_ptr(),
            ik.payload.data_ptr(),
            ik.scales.data_ptr(),
            q.data_ptr(),
            iq.data_ptr(),
            self.scratch.data_ptr(),
            torch.cuda.current_stream(),
        )
        if not self._compiled:
            for width in self.widths:
                launch = build_front(FrontBuild(tokens=width, **layout))
                self._compiled[width] = flyc.compile(launch, *args)
        self._compiled[s](*args)
        return res_new, q, iq


def watch_bad_inputs(name: str, scratch: torch.Tensor, at: int) -> None:
    """``SGLANG_AMD_MONO_M31_K2_DEBUG=1``: once the server serves, log a launch's
    out-of-range inputs (the kernel's DBG words: an OR of 1 slot, 2 length, 4 block,
    and the largest value) every 5 s, then clear them."""
    if os.environ.get("SGLANG_AMD_MONO_M31_K2_DEBUG", "0") != "1":
        return
    import logging
    import threading
    import time
    import urllib.request

    log = logging.getLogger(__name__)
    side = torch.cuda.Stream(device=scratch.device)

    def run():
        while True:
            time.sleep(5)
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:31000/v1/models", timeout=5
                ) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
        words = scratch.view(torch.int32)[at : at + 2]
        while True:
            time.sleep(5)
            with torch.cuda.stream(side):
                bits, value = words.cpu().tolist()
                if bits:
                    words.zero_()
            if bits:
                log.warning(
                    "%s saw out-of-range inputs: bits %d, largest %d", name, bits, value
                )

    threading.Thread(target=run, daemon=True).start()


class IndexRunner:
    """Host side of the verify indexer (``index.py``): a compiled kernel per row
    count and index-K pool layout, one scratch region (the rows' block scores and a
    counter a request, which the request's last CTA resets)."""

    def __init__(self, device: torch.device) -> None:
        from .index import DBG, SCRATCH_BYTES

        _writable_flydsl_cache()
        self.scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self._compiled: dict[Any, Any] = {}
        watch_bad_inputs("verify indexer", self.scratch, DBG)

    def __call__(self, iq, cache, req_to_token, seq_lens, slot_ids):
        """``iq``: the rows' index Q ([rows, 128] bf16, rows = requests x 8 drafts);
        ``cache``: the layer's index-K ``Nvfp4Cache``; ``seq_lens`` / ``slot_ids``: the
        verify rows' KV lengths and request slots. Returns SGLang's top-k block ids
        ([1, rows, 16] int32, ascending, -1 padded)."""
        import flydsl.compiler as flyc

        from .index import BLOCK, TOPK, IndexBuild, build_index

        rows = iq.shape[0]
        assert iq.dtype == torch.bfloat16 and iq.is_contiguous()
        assert cache.payload.dim() == 3 and cache.payload.shape[1] == BLOCK
        assert slot_ids.dtype == torch.int64 and seq_lens.dtype in (
            torch.int32,
            torch.int64,
        )
        assert req_to_token.dtype == torch.int32 and req_to_token.stride(1) == 1
        key = IndexBuild(
            tokens=rows,
            pay_page=cache.payload.stride(0),
            sc_page=cache.scales.stride(0),
            seq_i64=seq_lens.dtype == torch.int64,
        )
        out = torch.empty(1, rows, TOPK, dtype=torch.int32, device=iq.device)
        args = (
            iq.data_ptr(),
            cache.payload.data_ptr(),
            cache.scales.data_ptr(),
            req_to_token.data_ptr(),
            seq_lens.data_ptr(),
            slot_ids.data_ptr(),
            out.data_ptr(),
            self.scratch.data_ptr(),
            req_to_token.stride(0),
            cache.payload.shape[0],
            req_to_token.shape[0],
            req_to_token.shape[1],
            torch.cuda.current_stream(),
        )
        if key not in self._compiled:
            self._compiled[key] = flyc.compile(build_index(key), *args)
        self._compiled[key](*args)
        return out


class SparseRunner:
    """Host side of the verify sparse attention (``sparse.py``): a compiled kernel per
    row count and K / V pool layout, one scratch region (every row's blocks' softmax
    states and a counter a row, which the row's last CTA resets)."""

    def __init__(self, device: torch.device) -> None:
        from .sparse import DBG, SCRATCH_BYTES

        _writable_flydsl_cache()
        self.scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self._compiled: dict[Any, Any] = {}
        watch_bad_inputs("verify sparse attention", self.scratch, DBG)

    def __call__(self, q, k, v, req_to_token, seq_lens, slot_ids, topk):
        """``q``: the rows' Q ([rows, 16 x 128] bf16); ``k`` / ``v``: the layer's main
        K / V ``Nvfp4Cache``; ``topk``: [1, rows, 16] block ids. Returns the attention
        output, o_proj's input ([rows, 2048] bf16)."""
        import flydsl.compiler as flyc

        from .sparse import HEAD, HEADS, SparseBuild, build_sparse

        rows = q.shape[0]
        assert (
            q.dtype == torch.bfloat16
            and q.is_contiguous()
            and q.numel() == rows * HEADS * HEAD
        )
        assert k.payload.dim() == 4 and k.payload.shape[1] == 1
        assert (
            v.payload.stride() == k.payload.stride()
            and v.scales.stride() == k.scales.stride()
        )
        assert (
            topk.dtype == torch.int32
            and topk.is_contiguous()
            and topk.shape == (1, rows, 16)
        )
        key = SparseBuild(
            tokens=rows,
            pay_page=k.payload.stride(0),
            pay_row=k.payload.stride(2),
            sc_page=k.scales.stride(0),
            sc_row=k.scales.stride(2),
            seq_i64=seq_lens.dtype == torch.int64,
        )
        out = torch.empty(rows, HEADS * HEAD, dtype=torch.bfloat16, device=q.device)
        args = (
            q.data_ptr(),
            k.payload.data_ptr(),
            k.scales.data_ptr(),
            v.payload.data_ptr(),
            v.scales.data_ptr(),
            req_to_token.data_ptr(),
            seq_lens.data_ptr(),
            slot_ids.data_ptr(),
            topk.data_ptr(),
            out.data_ptr(),
            self.scratch.data_ptr(),
            req_to_token.stride(0),
            k.payload.shape[0],
            req_to_token.shape[0],
            req_to_token.shape[1],
            torch.cuda.current_stream(),
        )
        if key not in self._compiled:
            self._compiled[key] = flyc.compile(build_sparse(key), *args)
        self._compiled[key](*args)
        return out
