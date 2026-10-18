# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/plan/arith.py
# atom/mono/plan/shard.py
# atom/mono/plan/build_key.py
"""Plain-Python planning of the mono kernels: execution model, layouts, TP shards and build keys."""

import ast
import enum
import functools
import hashlib
import importlib.util
from pathlib import Path

import msgspec

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def first_task(bid, base):
    """Return CTA ``bid``'s first task of a stage placed from CTA ``base``."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# ---------------------------------------------------------------- mailbox layouts
PAIR_BYTES = 8  # a (value, tag) pair


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """Lay out ``items`` (name, pairs) in order: name -> (byte offset, bytes), ``align``-aligned."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


# ---------------------------------------------------------------- arithmetic
def sel(pred, a, b):
    """Return a if pred else b, as ``select`` on traced values."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def imin(a, b):
    return sel(a < b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


# ---------------------------------------------------------------- TP shards
class ShardError(ValueError):
    """A width the TP size does not divide, or a TP size the model refuses."""


class Shard(msgspec.Struct, frozen=True):
    tp: int

    def __post_init__(self):
        if self.tp < 1:
            raise ShardError(f"TP {self.tp}")

    def split(self, full: int, what: str) -> int:
        """``full`` / tp, refusing a remainder: a rank's share of ``what``."""
        if full % self.tp:
            raise ShardError(f"{what} {full} is not divisible by TP {self.tp}")
        return full // self.tp


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parents[1]  # the sgl_amd_mono package
# FlyDSL keys a kernel on its own directory's plain helpers only, so the plugin and
# aiter FlyDSL modules a kernel imports are hashed into its key
_TRACED = ("sgl_amd_mono", "aiter.ops.flydsl.kernels")
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _module_file(name: str):
    """The source file of module ``name``, or None when ``name`` is not a module."""
    if name.partition(".")[0] == "sgl_amd_mono":
        base = _ROOT.parent.joinpath(*name.split("."))
        return next(
            (f for f in (base / "__init__.py", base.with_suffix(".py")) if f.is_file()),
            None,
        )
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:  # an attribute, or no aiter (nothing compiles)
        return None
    if spec is None or not (spec.origin or "").endswith(".py"):
        return None
    return Path(spec.origin)


def _imported(name: str, tree: ast.Module, package: bool):
    """Every module name ``tree`` can import, at any depth; for ``from m import x``
    both ``m`` and ``m.x``, as ``x`` may be a submodule."""
    parent = name if package else name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = parent.rsplit(".", node.level - 1)[0]
                base = f"{anchor}.{base}" if base else anchor
            yield base
            yield from (f"{base}.{a.name}" for a in node.names)


def _code(tree: ast.Module) -> str:
    """``tree`` without its docstrings; comments and layout are already gone."""
    for node in ast.walk(tree):
        if isinstance(node, _DOCUMENTED) and ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


@functools.cache
def kernel_sources(module: str) -> str:
    """The digest a kernel built in ``module`` closes over: the code of ``module`` and
    of every plugin and aiter FlyDSL module it imports, transitively. Only a code
    edit to one of them rebuilds it; comments, docstrings and other modules do not."""
    codes, seen, todo = {}, set(), [module]
    while todo:
        name = todo.pop()
        traced = any(name == p or name.startswith(f"{p}.") for p in _TRACED)
        if name in seen or not traced:
            continue
        seen.add(name)
        f = _module_file(name)
        if f is None:
            continue
        tree = ast.parse(f.read_bytes())
        todo.extend(_imported(name, tree, f.name == "__init__.py"))
        codes[name] = _code(tree)
    if module not in codes:
        raise FileNotFoundError(f"no source for kernel module {module!r}")
    h = hashlib.sha256()
    for name in sorted(codes):
        h.update(f"{name}\0{codes[name]}\0".encode())
    return h.hexdigest()[:16]


def key_tuple(key, sources: str) -> tuple:
    """Return ``key``'s (field, value) pairs, then ("sources", ``sources``)."""
    out = []
    for f in msgspec.structs.fields(key):
        value = getattr(key, f.name)
        if not isinstance(value, _SCALARS):
            raise TypeError(
                f"{type(key).__name__}.{f.name} = {value!r}: a build parameter must be"
                " a scalar to reach the JIT cache key"
            )
        out.append((f.name, value))
    out.append(("sources", sources))
    return tuple(out)
