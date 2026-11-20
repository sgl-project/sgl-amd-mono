"""Check the MXFP4 activation quant of the kernels on one gfx950 GPU, bit for bit
against AITER:

- rot: ``hadamard32_quad`` gives the bf16 bits of the sgl-project/sglang#46059
  rotation, ROCm/aiter#6734's ``hadamard32_rotate`` (f32 butterfly stages 1 .. 16,
  times 1 / sqrt(32), rounded to bf16), on 2^20 values a seed and kind. With an
  exponent a block most f32 sums are exact; with an exponent an element they
  round, so another add order changes about 60 of the 2^20 values;
- quant: the rows that ``quant_x`` of K1 and K2 (``layer.py``) and of
  ``layer_block`` (``mono/stages/moe.py``) leaves in LDS are the dequantized
  ``per_1x32_f4_quant_hip(x, shuffle=False)``, plain and with
  ``hadamard_rotate=True``, at the widths and K the kernels take. Each input has
  an outlier in every third block and a block of zeros (AITER starts the amax at
  1e-10, the kernels at 0).
"""

import inspect
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter import per_1x32_f4_quant_hip
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64
from inputs import BF, CORE, HIDDEN, MAXT, gen

from sgl_amd_mono.common.plan import THREADS, kernel_sources
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.ops import (
    bf_hi,
    bf_lo,
    hadamard32_quad,
    ld,
    pack_bf16x8,
    st,
)
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950.mono.stages import moe

DEV = torch.device("cuda", 0)
SOURCES = (kernel_sources(layer.__name__), kernel_sources(moe.__name__))
HAS_H32 = "hadamard_rotate" in inspect.signature(per_1x32_f4_quant_hip).parameters
ROT_BLOCKS = 1 << 15
CASES = [("k12", k, s) for k in (HIDDEN, CORE) for s in (1, 3, MAXT)] + [
    ("layer_block", CORE, s) for s in (1, 2 * MAXT)
]
SEEDS = range(4)


def _rot_kernel(n):
    keyed = (n, SOURCES)

    @flyc.kernel(name=f"qwen38_check_h32_n{n}", known_block_size=[THREADS, 1, 1])
    def kr(x: Int64, y: Int64):
        _ = keyed
        tid = fx.thread_idx.x
        at = (fx.block_idx.x * THREADS + tid) * 4  # a lane's 8 bf16: 4 words
        wv = ld(x, at, n=4)
        vs = [half(wv[d]) for d in range(4) for half in (bf_lo, bf_hi)]
        st(y, at, pack_bf16x8(hadamard32_quad(vs, tid)))

    @flyc.jit
    def launch(x: Int64, y: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        kr(x, y).launch(grid=(n * 4 // THREADS,), block=(THREADS,), stream=stream)

    return launch


def _quant_kernel(path, k, s, h32):
    mod = layer if path == "k12" else moe
    xrow = k + mod.gemv.LDS_PAD
    groups = s * k // 8  # 8 bf16 a lane

    @fx.struct
    class Lds:
        xl: fx.Array[fx.Int32, s * xrow // 2, 16]

    keyed = (path, k, s, h32, SOURCES)

    @flyc.kernel(
        name=f"qwen38_check_quant_{path}_k{k}_s{s}_h{int(h32)}",
        known_block_size=[THREADS, 1, 1],
    )
    def kq(x: Int64, y: Int64):
        _ = keyed
        tid = fx.thread_idx.x
        xl = fx.SharedAllocator().allocate(Lds).peek().xl.ptr
        # lanes past the rows repeat the last group: same address, same words
        for i in range_constexpr((groups + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, groups - 1)
            t, kk = e // (k // 8), e % (k // 8) * 8
            fx.ptr_store(ld(x, (t * k + kk) // 2, n=4), xl + (t * xrow + kk) // 2)
        gpu.barrier()
        mod.quant_x({"S": s, "tid": tid, "xl": xl, "shuffled": True}, k, h32)
        for i in range_constexpr((groups + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, groups - 1)
            t, kk = e // (k // 8), e % (k // 8) * 8
            wv = fx.ptr_load(
                xl + (t * xrow + kk) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            st(y, (t * k + kk) // 2, fx.Vector(wv))

    @flyc.jit
    def launch(x: Int64, y: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        kq(x, y).launch(grid=(1,), block=(THREADS,), stream=stream)

    return launch


def _input(s, k, seed):
    g = gen(DEV, seed)
    x = torch.randn(s, k // 32, 32, generator=g, device=DEV)
    x = x * 2.0 ** torch.randint(-10, 11, (s, k // 32, 1), generator=g, device=DEV)
    x[:, ::3, 0] *= 300
    x[0, 0] = 0
    return x.reshape(s, k).to(BF)


def _input_wide(n, seed):
    g = gen(DEV, seed)
    x = torch.randn(n, 32, generator=g, device=DEV)
    return (x * 2.0 ** torch.randint(-12, 13, (n, 32), generator=g, device=DEV)).to(BF)


def h32_ref(x):
    """ROCm/aiter#6734's ``hadamard32_rotate`` on each 32 block of the last dim."""
    v = x.float().reshape(-1, 32).clone()
    for h in (1, 2, 4, 8, 16):
        lo = [j for j in range(32) if not j & h]
        hi = [j + h for j in lo]
        a, b = v[:, lo], v[:, hi]
        v[:, lo], v[:, hi] = a + b, a - b
    c = torch.tensor(0.17677669529663687, dtype=torch.float32, device=x.device)
    return (v * c).to(BF).reshape(x.shape)


def _dequant(xq, xs, s, k):
    scale = e8m0_to_f32(xs.view(torch.uint8)[:s, : k // 32])
    return (
        mxfp4_to_f32(xq.view(torch.uint8)[:s]) * scale.repeat_interleave(32, -1)
    ).to(BF)


def _run(launch, x):
    y = torch.empty_like(x)
    launch(x.data_ptr(), y.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return y


def check_rot():
    bad = []
    launch = _rot_kernel(ROT_BLOCKS)
    kinds = {
        "block": lambda seed: _input(1, ROT_BLOCKS * 32, seed).reshape(-1, 32),
        "element": lambda seed: _input_wide(ROT_BLOCKS, seed),
    }
    for kind, make in kinds.items():
        for seed in SEEDS:
            x = make(seed)
            y = _run(launch, x)
            diff = int((y.view(torch.int16) != h32_ref(x).view(torch.int16)).sum())
            name = f"rot {kind} seed {seed}"
            print(f"{name}: {diff} of {y.numel()} bf16 differ", flush=True)
            if diff:
                bad.append(name)
    return bad


def check_quant():
    bad = []
    for path, k, s in CASES:
        for h32 in (False, True):
            if h32 and not HAS_H32:
                print(f"quant {path} k{k} s{s}: skip h32, AITER has no hadamard_rotate")
                continue
            kw = {"hadamard_rotate": True} if h32 else {}
            launch = _quant_kernel(path, k, s, h32)
            for seed in SEEDS:
                x = _input(s, k, seed)
                y = _run(launch, x)
                ref = _dequant(*per_1x32_f4_quant_hip(x, shuffle=False, **kw), s, k)
                blocks = int((y != ref).reshape(s, k // 32, 32).any(-1).sum())
                if h32:
                    xr = h32_ref(x)
                    if not torch.equal(
                        _dequant(*per_1x32_f4_quant_hip(xr, shuffle=False), s, k), ref
                    ):
                        bad.append(f"h32_ref against aiter {path} k{k} s{s} {seed}")
                name = f"quant {path} k{k} s{s} h32={h32} seed {seed}"
                print(f"{name}: {blocks} of {s * k // 32} blocks differ", flush=True)
                if blocks:
                    bad.append(name)
    return bad


def main() -> int:
    torch.cuda.set_device(DEV)
    bad = check_rot() + check_quant()
    print(f"check_quant: {'OK' if not bad else 'FAIL'} {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
