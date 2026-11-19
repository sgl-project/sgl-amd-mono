# SPDX-License-Identifier: Apache-2.0
"""Let a spec-v2 decode step keep its seq_lens on the GPU (``SGLANG_AMD_MONO_M31_GPU_LENS=1``).

With any backend asking for the host mirror, SGLang's scheduler resolves a step's seq_lens on
the host before it launches the step (``FutureMap.resolve_seq_lens_cpu``, an event
synchronize on ROCm), so the GPU idles while the host builds every step. The MiniMax
backends ask by default; their only reader on a decode or verify step is the out-of-graph
``max_seqlen`` bound, which a replayed graph does not use: it keeps the capture's bound.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_M31_GPU_LENS"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/minimax_sparse_backend.py": (
        "MiniMaxSparseAttnBackend",
        "MiniMaxHybridAttnBackend",
        "init_forward_metadata_out_graph",
        "init_forward_metadata_out_graph(in_capture)",
        "_linear_verify",
        "max_context_len",
    ),
    "srt/managers/overlap_utils.py": (
        "decide_needs_cpu_seq_lens",
        '"needs_cpu_seq_lens"',
    ),
}


def install() -> None:
    """Once a process, before the attention backends are built."""
    if os.environ.get(ENV, "0") != "1":
        return
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxHybridAttnBackend,
        MiniMaxSparseAttnBackend,
    )

    sparse, hybrid = MiniMaxSparseAttnBackend, MiniMaxHybridAttnBackend
    if getattr(sparse, "_m31_gpu_lens", False):
        return
    original = sparse.init_forward_metadata_out_graph

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        mode = forward_batch.forward_mode
        if (
            forward_batch.seq_lens_cpu is None
            and self._linear_verify
            and (mode.is_decode_or_idle() or mode.is_target_verify())
        ):
            # the capture's bound (max_context_len): a replay runs the captured kernels
            in_capture = True
        return original(self, forward_batch, in_capture)

    sparse.init_forward_metadata_out_graph = init_forward_metadata_out_graph
    sparse.needs_cpu_seq_lens = False
    hybrid.needs_cpu_seq_lens = False
    sparse._m31_gpu_lens = True
    logger.info("M3.1: decode steps keep seq_lens on the GPU (%s=1)", ENV)
