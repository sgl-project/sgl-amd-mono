# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#43984 at 134919476c (kernels/ops/mono/gfx950/
# deepseek_v41/indexer/k2i.py), whose scorer comes from ROCm/ATOM at 68506823
# (MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.)
"""K2i: ATOM K2a's FP4 scorer (``score_fp4``) as a launch of its own between K1
and K2 of an index source. It writes each row's logits into ``ilog``, where
SGLang's top-k kernels read them.

BLOCKS CTAs of THREADS threads; each scores the task the step's walk plan gives
it (FULL) or its index does (REINDEX), and none waits on another.

Arguments (S rows, int32 unless noted):
  iq [S, 32, 64] E2M1, iqs [S, 4, 16, 4] E8M0, iw [S, 32] bf16 (SGLang's head weights)
  plane / pscale: the layer's FP4 index plane (page 64, ``score_fp4._keys``)
  iplan [BLOCKS, 4]: the FULL walk plan (``walk_plan``)
  itab [rows, itab_len] (row stride itab_stride): a FULL layer's page table,
      each entry 2 ** ishift (8) virtual pages; a REINDEX layer's candidate
      table, an entry a virtual page (ishift 0)
  ibatch [S]: the row whose itab row a FULL piece reads (its request's first)
  ibound [S]: columns a row's scorer reads
  ilog [S, lstride] f32: the logits, valid below each row's bound
  reindex: nonzero on a REINDEX layer
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources

from .score_fp4 import ROWS_LDS_WORDS, run_scores4

_STREAM = fx.Stream(None)
MAX_ROWS = 48  # a step's rows at most (8 DSpark requests of 6)


def build_k2i(tokens: int):
    """K2i for a step of ``tokens`` rows."""
    assert 0 < tokens <= MAX_ROWS, tokens
    name = kernel_symbol("dsv41_mono_k2i", s=tokens)
    keyed = (("tokens", tokens), ("sources", kernel_sources("deepseek_v41")))

    @fx.struct
    class RowsLds:
        rows: fx.Array[fx.Int32, ROWS_LDS_WORDS, 16]

    @fx.union
    class K2iLds:
        scorer: RowsLds

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2i(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, reindex: Int32,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(K2iLds).scorer.peek()
        args = {
            "iq": iq, "iqs": iqs, "iw": iw, "plane": plane, "pscale": pscale, "iplan": iplan,
            "itab": itab, "itab_stride": itab_stride, "itab_len": itab_len, "ibatch": ibatch,
            "ishift": ishift, "ibound": ibound, "ilog": ilog, "lstride": lstride, "reindex": reindex,
        }  # fmt: skip
        c = {
            "tid": tid, "lane": tid % 64, "wave": tid // 64, "args": args,
            "rows_lds": lds.rows.ptr,
        }  # fmt: skip
        run_scores4(c, tokens, fx.block_idx.x)

    @flyc.jit
    def launch(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, reindex: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k2i(
            iq, iqs, iw, plane, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog,
            lstride, reindex,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
