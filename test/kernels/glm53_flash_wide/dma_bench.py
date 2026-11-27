# SPDX-License-Identifier: Apache-2.0
"""Direct-to-LDS tile GEMM stage (``glm53_flash.gfx950.wide.tile_dma``) on 1 gfx950
GPU: rel L2 against fp32 and, with ``--time``, graph-replayed us against tgemm at
GLM-5.3-Flash's TP4 KDA shapes (in_proj 6416 x 4096, o_proj 4096 x 2048).
ks > 1 times the stage alone; ``--sum`` adds the partials' torch sum, or the
atomics' zeroing, each one launch.

    python dma_bench.py [--time] [--sum] [--m 18,36,96] [--shapes N:K,...] [--cfg bm:bn:wm:wn:ks:bk:stages,...] [--mode linbf]
"""

import argparse
import os
import sys

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kimi_k31"
    ),
)

from tile_bench import bench  # noqa: E402

from sgl_amd_mono.glm53_flash.gfx950.wide import tile_dma  # noqa: E402

SHAPES = "6416:4096,4096:2048"
# aiter#8006's picks at 18-96 rows (t BMxBNxBKxstages, w MxN, ks)
CFG = (
    "32:128:2:4:4:128:4,32:64:2:4:4:64:6,64:128:2:4:4:64:6,"
    "128:128:2:4:4:64:4,128:64:4:2:4:64:4,64:64:2:4:4:64:6"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--sum", action="store_true")
    ap.add_argument("--m", default="18,36,96")
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument(
        "--mode", default="linbf", help="part,atom,atombf,lin,linbf (ks > 1)"
    )
    a = ap.parse_args()
    if a.time:
        from aiter.tuned_gemm import tgemm
    worst = 0.0
    for m in map(int, a.m.split(",")):
        for nk in a.shapes.split(","):
            n, k = map(int, nk.split(":"))
            ncopy = max(4, -(-640 * 2**20 // (n * k * 2))) if a.time else 1
            ws = [
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                for _ in range(ncopy)
            ]
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().T
            tg = ""
            if a.time:
                ut = bench(lambda w: tgemm.mm(x, w, None, otype=torch.bfloat16), ws)
                tg = f"  tgemm {ut:6.1f} us"
            for cfg, mode in (
                (c, md) for c in a.cfg.split(",") for md in a.mode.split(",")
            ):
                bm, bn, wm, wn, ks, bk, stages = map(int, cfg.split(":"))
                if ks == 1 and mode != a.mode.split(",")[0]:
                    continue
                if k % ks or (k // ks) % bk or (bm // wm) % 16 or (bn // wn) % 16:
                    continue
                part = ks > 1 and mode == "part"
                shape = (ks, m, n) if part else (m, n)
                dt = (
                    torch.bfloat16
                    if ks == 1 or mode in ("atombf", "linbf")
                    else torch.float32
                )
                out = torch.zeros(shape, device="cuda", dtype=dt)
                try:
                    launch = tile_dma.standalone(
                        m, n, k, bm, bn, wm, wn, ks, bk, stages, mode
                    )

                    def fn(w, launch=launch, out=out, part=part, ks=ks):
                        if ks > 1 and not part:
                            if a.sum:
                                out.zero_()
                        launch(
                            x.data_ptr(),
                            w.data_ptr(),
                            out.data_ptr(),
                            stream=torch.cuda.current_stream(),
                        )
                        if a.sum and part:
                            out.sum(0)

                    out.zero_()
                    fn(ws[0])
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    print(
                        f"m={m} n={n} k={k} {cfg} {mode}: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    continue
                got = out.sum(0) if part else out.float()
                err = ((got - ref).norm() / ref.norm()).item()
                worst = max(worst, err)
                line = (
                    f"m={m:3d} n={n:5d} k={k:5d} t{bm}x{bn}x{bk}x{stages} w{wm}x{wn} ks{ks} "
                    f"{mode if ks > 1 else 'bf16':6s} relL2 {err:.1e}"
                )
                if a.time:
                    line += f"  dma {bench(fn, ws):6.1f} us"
                print(line + tg, flush=True)
            del ws
            torch.cuda.empty_cache()
    print("PASS" if worst < 1e-2 else "FAIL", f"worst relL2 {worst:.1e}")


if __name__ == "__main__":
    main()
