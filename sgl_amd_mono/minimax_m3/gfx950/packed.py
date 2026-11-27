# SPDX-License-Identifier: Apache-2.0
"""One rank's state the launches read, and the per-width kernel that reads it.

:class:`Packed` holds what every width shares -- per-layer weight views, the
bound caches, scratch, the peer buffer and the ``(ar, h_mid)`` buffer pairs --
and :class:`M3Kernel` is one built width. The split matters because FlyDSL
compiles per width and a build takes minutes, so everything that does not depend
on the token count is done once.
"""

from __future__ import annotations

import inspect
import logging

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled

from sgl_amd_mono.common import guard
from sgl_amd_mono.minimax_m3.gfx950.config import (
    HEAD_DIM,
)
from sgl_amd_mono.minimax_m3.gfx950.config import HIDDEN as HIDDEN_SIZE
from sgl_amd_mono.minimax_m3.gfx950.config import (
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    SPARSE_BLOCK,
    Dims,
)

logger = logging.getLogger(__name__)

FP8_CACHE = (torch.float8_e4m3fn, torch.uint8)
SWIGLU_ALPHA = 1.702  # aiter's swiglu_mul_batch bakes this in
# The fused shared expert's routing weight. SGLang's TopK gives it
# sum(routed) / routed_scaling_factor and the MoE output is scaled back by
# routed_scaling_factor, so with renormalized routed weights it contributes 1.0.
SHARED_WEIGHT = 1.0

SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/mem_cache/memory_pool.py": ("MiniMaxSparseKVPool", "use_aiter_sparse_pa"),
    "srt/layers/attention/minimax_sparse_backend.py": (
        "MiniMaxHybridAttnBackend",
        "index_topk_freq",
    ),
}


def compile_only(launch) -> None:
    """Build the launcher without dispatching its kernels."""
    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    _preload_compiled(launch, *([0] * len(names)), torch.cuda.current_stream())


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(what)


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


class Packed:
    """Per-rank state shared by every built width."""

    def __init__(self, model, hf_config, dims: Dims, ctx) -> None:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        self.dims = dims
        self.rank = par.tp_rank
        self.cpu_group = par.tp_group.cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.deadline_s = ctx.deadline_s
        self.fault_site, self.fault_after = guard.fault_injection(self.rank) or (0, 0)

        layers = model.model.layers
        self.sparse_ids = [i for i, l in enumerate(layers) if l.is_layer_sparse]
        self.first_layer = self.sparse_ids[0]
        config = getattr(hf_config, "text_config", None) or hf_config
        self._pack_layers(layers, config)
        self._alloc_buffers()
        self._bound = False
        # The private copies, reported rather than left implicit: unlike the
        # other model's relayout-only copies, the qkv / index and o_proj ones
        # are also QUANTIZED (bf16 -> per-channel FP8), which the standard path
        # does not do. RFC #43949 asks that kernels never quantize more than
        # the normal path; this one does, inherited from sglang#44319.
        copied = sum(
            t.numel() * t.element_size()
            for lw in self.sparse
            for t in (lw.w_qkv, lw.s_qkv, lw.w_o, lw.s_o, lw.gate)
        )
        logger.info(
            "[mono][rank %d] minimax_m3 packed: %d sparse layers (%d..%d), "
            "%.2f GiB of private weight copies (qkv / index and o_proj "
            "quantized bf16 -> FP8, router gate bf16)",
            self.rank, len(self.sparse), self.first_layer, self.sparse_ids[-1],
            copied / 2**30,
        )  # fmt: skip

    def bind(self) -> None:
        """Bind the KV caches and the request table, once, on the first routed
        step rather than at build.

        The caches do not exist at build: the framework builds the spec from
        ``install_monokernel_forward``, outside any forward, and SGLang's
        ``get_attn_backend`` reads a context ``ModelRunner`` publishes per
        forward call -- reading it at build asserts. The KV pools are sized
        after the model loads anyway, so there is nothing to bind to yet. The
        other model on this runtime binds its caches at its layer's first
        launch for the same reason.

        The cost of that is where the refusals land: a cache the kernels cannot
        read is reported on the first routed step, not at startup with the
        other ranks' reasons.

        This allocates, so it must not run inside a CUDA graph capture, or the
        tables would be cut from that graph's private memory pool while every
        later graph holds their addresses. It does not: the full-graph backend
        runs two eager warmup forwards before each capture, exactly so one-time
        setup is paid outside the capture region (the ``range(2)`` warmup loop
        ahead of ``torch.cuda.CUDAGraph()`` in
        ``model_executor/runner_backend/full_cuda_graph_backend.py``), and the
        first routed width's first warmup is where this lands.
        """
        if self._bound:
            return
        self._bind_caches()
        self._alloc_tables()
        self._bound = True
        logger.info(
            "[mono][rank %d] minimax_m3 caches bound: %d block-table columns, "
            "%d KV slots",
            self.rank, self.bt_width, self.num_slots,
        )  # fmt: skip

    # ------------------------------------------------------------- weights
    def _pack_layers(self, layers, config) -> None:
        """Each sparse layer's weight views and the build constants every width
        shares."""
        from sgl_amd_mono.minimax_m3.gfx950.weights import (
            SparseMoeLayer,
            sparse_selection,
        )

        heads = self.dims.index_heads
        cos_sin_tables: dict = {}
        self.sparse = [
            # dims is not passed: IndexHeads already carries the Dims it was
            # validated against, so taking it from heads.dims keeps one source
            # of truth rather than two that can disagree
            SparseMoeLayer.from_layer(layers[i], i, heads, cos_sin_tables)
            for i in self.sparse_ids
        ]
        _, init_blocks, local_blocks = sparse_selection(config)
        first = layers[self.first_layer]
        sm_scale = float(first.self_attn.scaling)
        _need(
            abs(sm_scale - HEAD_DIM**-0.5) < 1e-12,
            f"softmax scale {sm_scale}, the kernels bake in {HEAD_DIM**-0.5}",
        )
        self.k4_build = dict(
            npes=self.dims.tp,
            sm_scale=sm_scale,
            eps=float(config.rms_norm_eps),
            route_scale=float(first.mlp.routed_scaling_factor),
            shared_weight=SHARED_WEIGHT,
            swiglu_limit=float(config.swiglu_limit),
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            heads=heads,
            # Traced into the guard, so a build per setting; the stall itself is
            # a test hook and ``fault_injection`` returns None without its env var.
            fault_site=self.fault_site or None,
            fault_after=self.fault_after,
        )

    # -------------------------------------------------------------- caches
    def _bind_caches(self) -> None:
        """Bind FP8 K / V / index-K, ``local_kv_heads`` heads a rank, page-128
        allocation.

        Logical views stay NHD. Sparse K / V bytes use page-16 SHUFFLE storage;
        index-K and HiCache transfer rows remain NHD.
        """
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxHybridAttnBackend,
        )
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if isinstance(backend, MiniMaxHybridAttnBackend):
            backend = backend.sparse  # the sparse backend owns the MiniMax pool
        _need(
            backend.index_topk_freq == 1,
            "mono selects blocks per layer; set SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1",
        )
        pool = backend.kv_pool
        _need(
            isinstance(pool, MiniMaxSparseKVPool),
            f"{type(pool).__name__}: not the MiniMax sparse KV pool",
        )
        # The byte order, not the allocation: the buffer is the right shape and
        # dtype either way, but only the AITER sparse PA path writes it in
        # page-16 SHUFFLE order (sglang#41998). Reading NHD bytes as SHUFFLE
        # would not fault, it would produce plausible garbage.
        _need(pool.use_aiter_sparse_pa, "the kernels need page-16 SHUFFLE KV storage")
        main = pool.main_pool
        _need(
            pool.page_size == SPARSE_BLOCK,
            f"page size {pool.page_size}, kernels index {SPARSE_BLOCK}-token pages",
        )
        _need(
            main.kv_cache_layout == "nhd",
            f"KV cache logical layout {main.kv_cache_layout!r}, expected NHD views",
        )
        _need(
            main.head_num == self.dims.local_kv_heads,
            f"{main.head_num} kv heads a rank, kernels write "
            f"{self.dims.local_kv_heads}",
        )
        ids = self.sparse_ids
        k0, v0 = pool.get_key_buffer(ids[0]), pool.get_value_buffer(ids[0])
        i0 = pool.get_index_k_buffer(ids[0])
        _need(
            k0.dtype in FP8_CACHE and v0.dtype in FP8_CACHE,
            f"KV cache {k0.dtype} / {v0.dtype}, kernels write e4m3",
        )
        _need(i0.dtype in FP8_CACHE, f"index cache {i0.dtype}, kernels write e4m3")
        _need(
            k0.shape[0] == i0.shape[0]
            and k0.shape[-1] == HEAD_DIM
            and i0.shape[-1] == HEAD_DIM,
            f"KV cache {tuple(k0.shape)} / index cache {tuple(i0.shape)}",
        )
        self.num_slots = k0.shape[0]
        self.k_ptrs, self.v_ptrs, self.index_ptrs = [], [], []
        self.layer_wait = {}
        for lid in ids:
            k = pool.get_key_buffer(lid)
            v = pool.get_value_buffer(lid)
            ik = pool.get_index_k_buffer(lid)
            _need(
                k.is_contiguous() and v.is_contiguous() and ik.is_contiguous()
                and k.shape == k0.shape and v.shape == v0.shape and ik.shape == i0.shape,
                f"layer {lid}: cache views differ from layer {ids[0]}'s",
            )  # fmt: skip
            self.k_ptrs.append(_ptr(k))
            self.v_ptrs.append(_ptr(v))
            self.index_ptrs.append(_ptr(ik))
            # Cached pointers must honour HiCache's per-layer H2D loads, as the
            # pool's own get_key_buffer does on the module path: the bytes may
            # still be in flight when the launch would read them.
            self.layer_wait[lid] = pool._wait_for_layer

        req_to_token = backend.req_to_token
        _need(
            req_to_token.dtype == torch.int32 and req_to_token.dim() == 2,
            f"req_to_token {req_to_token.dtype} {tuple(req_to_token.shape)}",
        )
        self.req_to_token = req_to_token
        # EAGLE reserves an extra allocator page for draft lookahead. It is not
        # an increase to the configured model context or score capacity.
        self.max_context_len = int(backend.max_context_len)
        self.bt_width = (self.max_context_len + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        _need(
            0 < self.bt_width <= MAX_INDEX_BLOCKS,
            f"{self.bt_width} blocks a request, scores hold {MAX_INDEX_BLOCKS}",
        )
        self.cols = torch.arange(
            0, self.bt_width * SPARSE_BLOCK, SPARSE_BLOCK, dtype=torch.int64,
            device=req_to_token.device,
        )  # fmt: skip

    # ------------------------------------------------------------- buffers
    def _alloc_buffers(self) -> None:
        """Scratch, the peer buffer, the guard block, the step counter and the
        ``(ar, h_mid)`` pairs: everything whose size the rank's shapes fix.

        Built at build, not at bind, because the peer buffer is a collective --
        every rank exchanges its base address here, and build is the point
        where the ranks are in lockstep by construction."""
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.minimax_m3.gfx950.layout import Layout, sym_layout
        from sgl_amd_mono.minimax_m3.gfx950.pre_attn import K1Layout

        dims, dev = self.dims, self.device
        self.layout = Layout(dims=dims)
        # K1's scratch follows the rank's head tasks, so it is sized from the
        # same K1Layout the kernel builds rather than a module constant.
        self.scratch1 = torch.zeros(
            K1Layout(dims=dims).scratch_bytes, dtype=torch.uint8, device=dev
        )
        self.scratch4 = torch.zeros(
            self.layout.scratch_bytes, dtype=torch.uint8, device=dev
        )
        # The peer buffer carries the symmetric regions and, as its tail, the
        # guard's per-rank error flags -- the same shape the other model in this
        # package uses, so that flags_offset is simply the symmetric extent.
        self.sym_bytes = sym_layout(dims.tp)["_bytes"]
        self.peers = PeerBuffer(
            self.sym_bytes + guard.FLAG_BYTES, self.cpu_group, self.rank, dims.tp, dev
        )
        self.flags_offset = self.sym_bytes
        # The guard's block: the deadline the host writes once, and the sticky
        # error words a failed wait sets. Word 0 is the launch counter the
        # fault injection arms off; the mailbox tags come from ``step``, not
        # from here, so nothing else reads it.
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        self.set_deadline(self.deadline_s)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        bf16 = torch.bfloat16
        # row k = token k of the step; sparse layer i reads ars[i % 2] (the
        # previous layer's output) and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [
            torch.empty(MAX_TOKENS, HIDDEN_SIZE, dtype=bf16, device=dev)
            for _ in range(2)
        ]
        self.h_mids = [
            torch.empty(MAX_TOKENS, HIDDEN_SIZE, dtype=bf16, device=dev)
            for _ in range(2)
        ]
        self.h = torch.empty(MAX_TOKENS, HIDDEN_SIZE, dtype=bf16, device=dev)
        self.q = torch.empty(
            MAX_TOKENS, dims.local_q_heads * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)

    # -------------------------------------------------------------- tables
    def _alloc_tables(self) -> None:
        """What the caches and the request table size: the per-layer K1 argument
        tables and the per-width row tables. Built by ``bind``, not at build."""
        from sgl_amd_mono.minimax_m3.gfx950.pre_attn import K1_ARGS

        dev = self.device
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in),
                "w_qkv": _ptr(lw.w_qkv), "s_qkv": _ptr(lw.s_qkv),
                "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik),
                "cos_sin": _ptr(lw.cos_sin), "iq_out": _ptr(self.iq),
                "index_cache": self.index_ptrs[i],
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )
        # per token count: the step's block table and sequence lengths
        self.tables = {
            n: (
                torch.zeros(n, self.bt_width, dtype=torch.int32, device=dev),
                torch.ones(n, dtype=torch.int32, device=dev),
            )
            for n in range(1, MAX_TOKENS + 1)
        }

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, in ticks of the 100 MHz reference clock. The
        kernels read it from the guard block, so a change takes effect on the
        next launch without a rebuild -- and a graph replay reads the buffer
        too."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        # torch int32 is signed, and a deadline past ~21 s overflows the low word
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    # --------------------------------------------------------------- build
    def build_kernel(self, rows: int) -> M3Kernel:
        return M3Kernel(self, rows)

    def prepare_step(self, forward_batch, rows: int, query_len: int, is_verify: bool):
        """This step's causal rows from the replay-fresh decode / verify buffers."""
        from sgl_amd_mono.minimax_m3.gfx950.metadata import prepare_token_rows

        bt, seq_lens = self.tables[rows]
        return prepare_token_rows(
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.cols,
            bt,
            seq_lens,
            forward_batch.out_cache_loc[:rows],
            query_len,
            is_verify,
            self.max_context_len,
        )


class M3Kernel:
    """The ``rows``-token launch every sparse layer shares."""

    def __init__(self, packed: Packed, rows: int) -> None:
        from sgl_amd_mono.minimax_m3.gfx950.post_attn import build_post_attn_kernel

        self.packed = packed
        self.rows = rows
        # FlyDSL compiles on the first launch: the eager warm-up forward that
        # reaches this width before its graph capture
        self.launcher = build_post_attn_kernel(
            tokens=rows,
            dims=packed.dims,
            flags_offset=packed.flags_offset,
            **packed.k4_build,
        )

    def launch(self, layer_id, positions, hidden_states, residual, forward_batch, memo):
        """Sparse layer ``layer_id`` in a decoder layer's own terms:
        ``(hidden_states, residual)`` in, ``(hidden_states, residual)`` out.

        ``ar`` has to be in ``ars[i % 2]``, which it already is when the
        preceding launch left it there; the copy is for the first sparse layer,
        which the modules hand it. ``memo``: the step's row tables, derived once
        by its first routed layer.
        """
        p = self.packed
        i = layer_id - p.first_layer
        n = self.rows
        wait = p.layer_wait.get(layer_id)
        if wait is not None:
            wait(layer_id)  # HiCache's per-layer H2D load may still be in flight
        rows = memo.get("rows")
        if rows is None:
            rows = memo["rows"] = p.prepare_step(
                forward_batch, n, memo["query_len"], memo["is_verify"]
            )
        block_table, seq_lens = rows
        ar = p.ars[i % 2]
        if hidden_states.data_ptr() != ar.data_ptr():
            ar[:n].copy_(hidden_states.view(n, HIDDEN_SIZE))
        lw = p.sparse[i]
        _run_compiled(
            self.launcher,
            _ptr(p.h), _ptr(p.q), _ptr(block_table), _ptr(seq_lens),
            p.k_ptrs[i], p.v_ptrs[i], _ptr(lw.k_scale), _ptr(lw.v_scale),
            _ptr(lw.w_o), _ptr(lw.s_o), _ptr(lw.g_post), _ptr(lw.gate),
            _ptr(lw.bias), _ptr(lw.w13), _ptr(lw.s13), _ptr(lw.w2), _ptr(lw.s2),
            _ptr(p.h_mids[(i + 1) % 2]), _ptr(p.ars[(i + 1) % 2]),
            _ptr(p.scratch4), p.peers.local, _ptr(p.peers.addresses),
            _ptr(p.step), _ptr(p.epoch), p.rank, lw.layer_id,
            block_table.shape[1], memo["query_len"],
            _ptr(p.k1_args[i]), _ptr(positions),
            _ptr(forward_batch.out_cache_loc), _ptr(residual.view(n, HIDDEN_SIZE)),
            torch.cuda.current_stream(),
        )  # fmt: skip
        if i == len(p.sparse) - 1:
            # Opens the next step's epoch, which is what keeps a layer's mailbox
            # tags distinct from the same layer's tags one step ago. Enqueued
            # after the last launch of this step, so no kernel can see it early.
            p.step.add_(1)
        return p.ars[(i + 1) % 2][:n], p.h_mids[(i + 1) % 2][:n]

    def capture_input(self) -> torch.Tensor:
        """K1's BF16 residual after the input add, before the norm: the hidden
        state an EAGLE3 draft captures at this layer's attention boundary (a
        reused buffer: the collector copies it)."""
        return self.packed.h[: self.rows]

    # No ``health()``: a failed wait does not report back to the host. It sets
    # its sticky error word, flags every peer and stops the launch with a store
    # to the null page, so the process dies at the fault rather than running on
    # with addresses computed from data that never arrived (``common/guard.py``).
