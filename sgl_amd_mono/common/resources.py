# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 974414f16, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/compile.py
"""What a compiled monokernel asks of the GPU, read from its code object.

A persistent grid needs every CTA resident at once, so a kernel the CUs cannot hold
``blocks`` of never finishes, and a kernel that spills VGPRs to scratch is slower.
``check``, called where a runner builds a launch, raises ``NotResident`` for the first
and holds the second to the caller's ``Budget`` as ``SGLANG_AMD_MONO_RESOURCES`` says:
``warn`` (default) logs, ``strict`` raises ``OverBudget``, ``report`` also logs every
kernel within it.

Two things here are FlyDSL's and not public: a launcher's ``_mem_cache`` and the text
of ``max_blocks_per_cu()``'s error. ``test_a_flydsl_build_is_read`` and
``test_flydsl_still_raises_the_text_read_as_no_blocks`` fail when FlyDSL changes them.

  python -m sgl_amd_mono.common.resources DUMP_DIR [--tsv OUT]

reads the kernels a ``FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=DUMP_DIR`` build left.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import logging
import os
import re
import struct
from typing import Optional

import msgspec

from sgl_amd_mono.common.plan import BLOCKS

logger = logging.getLogger(__name__)

MODES = ("report", "warn", "strict")
FIELDS = ("vgpr", "agpr", "sgpr", "vgpr_spill", "sgpr_spill", "scratch", "lds")
# a kernel's entry in the code object's metadata; older code objects lack the spill
# counts
_KEYS = {
    "vgpr": ".vgpr_count",
    "agpr": ".agpr_count",
    "sgpr": ".sgpr_count",
    "vgpr_spill": ".vgpr_spill_count",
    "sgpr_spill": ".sgpr_spill_count",
    "scratch": ".private_segment_fixed_size",
    "lds": ".group_segment_fixed_size",
}
_OPTIONAL = ("agpr", "vgpr_spill", "sgpr_spill")
_SHT_NOTE = 7
_NT_AMDGPU_METADATA = 32
_BIN = re.compile(r'bin\s*=\s*"')
# FlyDSL raises this for zero blocks and for a failed query alike (jit_executor.py,
# CompiledArtifact.max_blocks_per_cu)
_NO_BLOCKS = "occupancy query failed"

# by a code object's bytes: a launcher's cache keeps its earlier binaries, and FlyDSL
# keeps a kernel name unique only within its module
_RESIDENT: dict[bytes, int] = {}
_LOGGED: set[tuple[bytes, Budget]] = set()
_WARNED: set[str] = set()


class Unreadable(RuntimeError):
    """No code object to read: not compiled yet, or FlyDSL keeps it elsewhere now."""


class NotResident(RuntimeError):
    """The CUs cannot hold a kernel's grid at once."""


class OverBudget(RuntimeError):
    """A kernel takes more scratch than its budget (``strict``)."""


class NoOccupancy(RuntimeError):
    """FlyDSL cannot give a kernel's occupancy (``strict``)."""


_FAILURES = {
    e.__name__: e
    for e in (Unreadable, NotResident, OverBudget, NoOccupancy, ValueError, TypeError)
}


class Resources(msgspec.Struct, frozen=True):
    """One binary's metadata; a field its code object lacks is None."""

    name: str  # the last nested name (``kernel_symbol``'s ``<stem>_<params>``)
    symbol: str  # as the code object names it
    vgpr: int
    agpr: Optional[int]
    sgpr: int
    vgpr_spill: Optional[int]
    sgpr_spill: Optional[int]
    scratch: int  # bytes a lane
    lds: int  # bytes a workgroup

    def line(self, resident: Optional[int] = None) -> str:
        out = (
            f"{self.name}: VGPR {self.vgpr} AGPR {self.agpr} SGPR {self.sgpr}, "
            f"spilled VGPR {self.vgpr_spill} SGPR {self.sgpr_spill}, "
            f"scratch {self.scratch} B a lane, LDS {self.lds} B"
        )
        return out if resident is None else f"{out}, {resident} resident a CU"


class Budget(msgspec.Struct, frozen=True):
    """What a kernel may ask: ``scratch`` bytes a lane; ``blocks`` CTAs resident at once."""

    scratch: int = 0
    blocks: int = BLOCKS


def mode() -> str:
    m = os.environ.get("SGLANG_AMD_MONO_RESOURCES", "warn")
    if m not in MODES:
        raise ValueError(f"SGLANG_AMD_MONO_RESOURCES={m!r}: one of {MODES}")
    return m


def artifacts(compiled) -> list:
    """The FlyDSL artifacts of ``compiled``: an ``@flyc.jit`` launcher once built (an
    artifact a specialization), or one artifact. ``flyc.compile``'s result is neither:
    pass its artifact."""
    if hasattr(compiled, "_mem_cache"):
        arts = list(compiled._mem_cache.values())
    else:
        arts = [compiled]
    if not arts or not all(isinstance(getattr(a, "ir", None), str) for a in arts):
        raise Unreadable(f"{compiled!r}: no compiled module text to read")
    return arts


def code_object(module_text: str) -> bytes:
    """The ``gpu.binary``'s ELF: the ``bin = "..."`` string, MLIR-escaped (\\XX a
    byte, \\\\ and \\" themselves)."""
    m = _BIN.search(module_text)
    if m is None:
        raise Unreadable("no gpu.binary in the compiled module")
    out, i = bytearray(), m.end()
    try:
        while module_text[i] != '"':
            ch = module_text[i]
            if ch != "\\":
                out.append(ord(ch))
                i += 1
            elif module_text[i + 1] in '\\"':
                out.append(ord(module_text[i + 1]))
                i += 2
            else:
                out.append(int(module_text[i + 1 : i + 3], 16))
                i += 3
    except (IndexError, ValueError) as err:
        raise Unreadable(f"a truncated or misescaped gpu.binary: {err}") from err
    if out[:4] != b"\x7fELF":
        raise Unreadable("the gpu.binary is not an ELF code object")
    return bytes(out)


def read(elf: bytes) -> Resources:
    """The one kernel in ``elf``, from its AMDGPU metadata note."""
    kernels = _metadata(elf).get("amdhsa.kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        n = len(kernels) if isinstance(kernels, list) else "no"
        raise Unreadable(f"{n} kernels in one code object")
    (k,) = kernels
    try:
        symbol = k[".symbol"].removesuffix(".kd")
        values = {
            f: k.get(key) if f in _OPTIONAL else k[key] for f, key in _KEYS.items()
        }
    except (KeyError, TypeError, AttributeError) as err:
        raise Unreadable(f"the code object's kernel metadata: {err!r}") from err
    return Resources(name=_stem(symbol), symbol=symbol, **values)


def _metadata(elf: bytes) -> dict:
    """The msgpack map of ``elf``'s NT_AMDGPU_METADATA note (a 64-bit little-endian
    ELF, its notes in SHT_NOTE sections)."""
    if elf[:6] != b"\x7fELF\x02\x01":
        raise Unreadable("not a 64-bit little-endian ELF")
    try:
        (shoff,) = struct.unpack_from("<Q", elf, 0x28)
        shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
        for i in range(shnum):
            sh = shoff + i * shentsize
            if struct.unpack_from("<I", elf, sh + 4)[0] != _SHT_NOTE:
                continue
            at, size = struct.unpack_from("<QQ", elf, sh + 0x18)
            end = at + size
            while at + 12 <= end:
                namesz, descsz, kind = struct.unpack_from("<III", elf, at)
                name, desc = at + 12, at + 12 + (namesz + 3) // 4 * 4
                at = desc + (descsz + 3) // 4 * 4
                if (
                    kind == _NT_AMDGPU_METADATA
                    and elf[name : name + namesz] == b"AMDGPU\0"
                ):
                    meta = msgspec.msgpack.decode(elf[desc : desc + descsz])
                    if not isinstance(meta, dict):
                        raise Unreadable("the AMDGPU metadata is not a map")
                    return meta
    except (struct.error, msgspec.DecodeError) as err:
        raise Unreadable(f"the code object's metadata note: {err}") from err
    raise Unreadable("no AMDGPU metadata note in the code object")


def _stem(symbol: str) -> str:
    """The last name of a plain nested symbol (``kernel_symbol``'s), else the symbol."""
    if not (symbol.startswith("_ZN") and symbol.endswith("E")):
        return symbol
    rest, last = symbol[3:-1], symbol
    while rest:
        m = re.match(r"\d+", rest)
        if m is None or not 0 < int(m.group()) <= len(rest) - m.end():
            return symbol
        n, rest = int(m.group()), rest[m.end() :]
        last, rest = rest[:n], rest[n:]
    return last


def resident_per_cu(artifact) -> int:
    """The workgroups of ``artifact``'s kernel a CU holds at once, as FlyDSL asks HIP
    (its launch's block size and dynamic LDS): 0 for none or a failed query alike,
    which FlyDSL reports the same way; ``NoOccupancy`` when it cannot ask at all."""
    query = getattr(artifact, "max_blocks_per_cu", None)
    if query is None:
        raise NoOccupancy("this FlyDSL has no max_blocks_per_cu()")
    try:
        return query()
    except RuntimeError as err:
        if _NO_BLOCKS in str(err):
            return 0
        raise NoOccupancy(str(err)) from err
    except Exception as err:
        raise NoOccupancy(f"{type(err).__name__}: {err}") from err


def _compute_units() -> Optional[int]:
    """This device's CUs, or None without one (a compile-only build)."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _warn_once(log: bool, key: str, msg: str, *args) -> None:
    if log and key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *args)


def check(
    compiled, *, budget: Budget = Budget(), log: bool = True, group=None
) -> list[Resources]:
    """Read ``compiled``'s binaries (``artifacts``); raise ``NotResident`` for one whose
    grid of ``budget.blocks`` the CUs cannot hold, and hold each to ``budget.scratch``
    as ``mode()`` says. A code object seen before in the process is not queried again,
    nor logged again under the same budget. ``log``: False on the ranks whose builds
    repeat another's. ``group``: the ranks building these launches together (a CPU
    process group), each calling ``check`` as often as the others; what one raises,
    all raise, so none waits at its next collective on a rank that stopped."""
    if group is None or _world_size(group) == 1:
        return _check(compiled, budget, log)
    try:
        found, err = _check(compiled, budget, log), None
    except Exception as e:
        found, err = [], e
    _agree(group, err)
    return found


def _world_size(group) -> int:
    import torch.distributed as dist

    return dist.get_world_size(group)


def _gather(group, obj) -> list:
    import torch.distributed as dist

    every = [None] * dist.get_world_size(group)
    dist.all_gather_object(every, obj, group=group)
    return every


def _agree(group, err: Optional[Exception]) -> None:
    """Raise on every rank of ``group`` the lowest failing rank's error, if any."""
    mine = None if err is None else (type(err).__name__, str(err))
    failed = [(r, f) for r, f in enumerate(_gather(group, mine)) if f is not None]
    if not failed:
        return
    rank, (name, msg) = failed[0]
    more = f" (and {len(failed) - 1} more ranks)" if len(failed) > 1 else ""
    cls = _FAILURES.get(name)
    agreed = (cls or RuntimeError)(
        f"rank {rank}: {msg if cls else f'{name}: {msg}'}{more}"
    )
    raise agreed from err


def _check(compiled, budget: Budget, log: bool) -> list[Resources]:
    m = mode()
    try:
        arts = artifacts(compiled)
        elfs = [code_object(a.ir) for a in arts]
        found = [read(e) for e in elfs]
    except Unreadable as err:
        if m == "strict":
            raise
        _warn_once(log, "unreadable", "monokernel resources not checked: %s", err)
        return []
    cus = _compute_units()
    for art, elf, r in zip(arts, elfs, found):
        key = hashlib.blake2b(elf, digest_size=16).digest()
        resident = _RESIDENT.get(key)
        if resident is None and cus is not None:
            try:
                resident = _RESIDENT[key] = resident_per_cu(art)
            except NoOccupancy as err:
                if m == "strict":
                    raise
                _warn_once(
                    log, "occupancy", "monokernel residency not checked: %s", err
                )
        if resident is not None and cus is not None and resident * cus < budget.blocks:
            why = " (or HIP could not answer)" if resident == 0 else ""
            raise NotResident(
                f"{r.line(resident)}{why}: {cus} CUs hold {resident * cus} of the "
                f"grid's {budget.blocks} CTAs"
            )
        over = r.scratch > budget.scratch
        if over and m == "strict":
            raise OverBudget(f"{r.line(resident)}: past {budget.scratch} B a lane")
        if not log or (key, budget) in _LOGGED:
            continue
        if over:
            logger.warning(
                "monokernel spills: %s (budget %d B a lane)",
                r.line(resident),
                budget.scratch,
            )
        elif m == "report":
            logger.info("monokernel: %s", r.line(resident))
        else:
            continue
        _LOGGED.add((key, budget))
    return found


def read_dump(dump_dir: str) -> list[Resources]:
    """Every kernel a ``FLYDSL_DUMP_IR=1`` build dumped under ``dump_dir`` (its
    ``*gpu_module_to_binary.mlir``)."""
    out = []
    pattern = os.path.join(dump_dir, "*", "*gpu_module_to_binary.mlir")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            out.append(read(code_object(f.read())))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump_dir")
    ap.add_argument("--tsv", help="also write the table here")
    args = ap.parse_args(argv)
    found = read_dump(args.dump_dir)
    rows = ["\t".join(("kernel",) + FIELDS)]
    rows += [
        "\t".join([r.name] + [str(getattr(r, f)) for f in FIELDS])
        for r in sorted(found, key=lambda r: r.name)
    ]
    print("\n".join(rows))
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\n".join(rows) + "\n")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
