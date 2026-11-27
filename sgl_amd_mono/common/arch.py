# SPDX-License-Identifier: Apache-2.0
"""The target the kernels build for, read once at import: its name, wave size and
the resident grid (one CTA a compute unit).

``SGLANG_AMD_MONO_ARCH`` / ``SGLANG_AMD_MONO_BLOCKS`` override what the device reports
(a compile-only build on a host without the GPU); otherwise the arch comes from
FlyDSL's compile target or the current device, and the grid from the device's
compute unit count. gfx950 keeps its fixed grid of 256.
"""

import functools
import os

GFX950_BLOCKS = 256


@functools.cache
def _device():
    try:
        import torch

        if torch.cuda.is_available() and torch.version.hip:
            return torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # no torch, no device: the overrides decide
        pass
    return None


def _arch() -> str:
    for var in ("SGLANG_AMD_MONO_ARCH", "FLYDSL_COMPILE_ARCH", "FLYDSL_GPU_ARCH"):
        v = os.environ.get(var)
        if v:
            return v.split(":")[0]
    d = _device()
    return d.gcnArchName.split(":")[0] if d is not None else "gfx950"


ARCH = _arch()
GFX1250 = ARCH.startswith("gfx125")
WAVE = 32 if GFX1250 else 64


def _no_retry() -> bool:
    """The GPU driver runs with retry (XNACK) off: a page fault is fatal, never replayed."""
    try:
        with open("/sys/module/amdgpu/parameters/noretry") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


# SGLANG_AMD_MONO_NO_XCNT=1 (opt-in): build gfx1250 code without XNACK replay safety, i.e.
# no ``s_wait_xcnt`` before each register reuse, when the driver runs with retry off and
# HSA_XNACK is not set. FlyDSL's LLVM reads LLVM_GFX1250_ANY_XNACK_CODEGEN when it builds
# the gfx1250 subtarget; the setting is process-wide (every FlyDSL kernel of the process).
NO_XCNT = (
    GFX1250
    and os.environ.get("SGLANG_AMD_MONO_NO_XCNT", "0") == "1"
    and os.environ.get("HSA_XNACK", "0") != "1"
    and _no_retry()
)
if NO_XCNT:
    os.environ.setdefault("LLVM_GFX1250_ANY_XNACK_CODEGEN", "0")


def _blocks() -> int:
    v = os.environ.get("SGLANG_AMD_MONO_BLOCKS")
    if v:
        return int(v)
    if not GFX1250:
        return GFX950_BLOCKS
    d = _device()
    if d is None:
        raise RuntimeError(
            f"{ARCH}: no device to size the resident grid; set SGLANG_AMD_MONO_BLOCKS"
        )
    return d.multi_processor_count


BLOCKS = _blocks()
