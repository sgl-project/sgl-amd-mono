"""The kernels' cache key: every FlyDSL launcher closes over the sources digest, and
the digest follows the aiter helpers the sources import."""

import ast
import importlib.machinery
import importlib.util
from pathlib import Path

import sgl_amd_mono
from sgl_amd_mono.common import plan

PACKAGE = Path(sgl_amd_mono.__file__).resolve().parent


def _flyc(fn: ast.FunctionDef, kind: str) -> bool:
    """Whether ``fn`` is decorated ``@flyc.<kind>`` (``jit``, or ``kernel(...)``)."""
    for d in fn.decorator_list:
        d = d.func if isinstance(d, ast.Call) else d
        if (
            isinstance(d, ast.Attribute)
            and d.attr == kind
            and isinstance(d.value, ast.Name)
            and d.value.id == "flyc"
        ):
            return True
    return False


def _reads(node: ast.AST) -> set:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _launchers():
    """Each ``@flyc.jit`` launcher in the package, with whether it closes over a
    ``keyed`` built from ``kernel_sources`` (itself or through a kernel it calls)."""
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for fn in ast.walk(tree):
            if not (isinstance(fn, ast.FunctionDef) and _flyc(fn, "jit")):
                continue
            builder = parent.get(fn)
            while builder is not None and not isinstance(builder, ast.FunctionDef):
                builder = parent.get(builder)
            name = f"{path.relative_to(PACKAGE)}:{fn.lineno} {fn.name}"
            if builder is None:
                yield name, False
                continue
            digested = any(
                isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "keyed" for t in n.targets)
                and "kernel_sources" in _reads(n.value)
                for n in ast.walk(builder)
            )
            keyed_kernels = {
                f.name
                for f in ast.walk(builder)
                if isinstance(f, ast.FunctionDef)
                and _flyc(f, "kernel")
                and "keyed" in _reads(f)
            }
            reads = _reads(fn)
            yield name, digested and ("keyed" in reads or bool(reads & keyed_kernels))


def test_every_launcher_closes_over_the_sources_digest():
    # FlyDSL keys a launcher on its closure and same-directory helpers only, so one
    # that never reads the digest keeps serving kernels built from old sources
    launchers = dict(_launchers())
    assert launchers, "found no @flyc.jit launcher"
    missing = sorted(name for name, ok in launchers.items() if not ok)
    assert not missing, f"close over kernel_sources(...) as `keyed`: {missing}"


def test_an_aiter_helper_edit_changes_the_key(tmp_path, monkeypatch):
    # FlyDSL's own key ignores modules outside the kernel's directory, so without
    # this a persistent cache would serve kernels traced against an old helper
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "k.py").write_text(
        "from aiter.ops.flydsl.kernels import buffer_ops as bo\n"
    )
    helper = tmp_path / "buffer_ops.py"
    unrelated = tmp_path / "unrelated.py"
    origins = {
        "aiter.ops.flydsl.kernels.buffer_ops": helper,
        "aiter.ops.flydsl.kernels.unrelated": unrelated,
    }

    def find_spec(name):
        origin = origins.get(name)
        return (
            None
            if origin is None
            else importlib.machinery.ModuleSpec(name, None, origin=str(origin))
        )

    monkeypatch.setattr(plan, "_ROOT", tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    def digest():
        plan.source_digest.cache_clear()
        return plan.source_digest("kernels")

    helper.write_text("A = 1\n")
    unrelated.write_text("B = 1\n")
    before = digest()
    unrelated.write_text("B = 2\n")
    assert digest() == before, "a helper the sources do not import changed the key"
    helper.write_text("A = 2\n")
    assert digest() != before, "an edit to an imported aiter helper kept the key"
