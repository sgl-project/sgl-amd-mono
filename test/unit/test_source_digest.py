"""The kernels' cache key: every FlyDSL launcher closes over its module's sources
digest, and the digest follows the code that module imports."""

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


def _digests_own_module(node: ast.AST) -> bool:
    """Whether ``node`` calls ``kernel_sources(__name__)``."""
    return any(
        isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "kernel_sources"
        and [getattr(a, "id", None) for a in n.args] == ["__name__"]
        for n in ast.walk(node)
    )


def _launchers():
    """Each ``@flyc.jit`` launcher in the package, with whether it closes over a
    ``keyed`` built from ``kernel_sources(__name__)`` (itself or through a kernel it
    calls)."""
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
                and _digests_own_module(n.value)
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
    assert not missing, f"close over kernel_sources(__name__) as `keyed`: {missing}"


def test_the_digest_follows_the_code_the_kernel_module_imports(tmp_path, monkeypatch):
    # FlyDSL's own key ignores modules outside the kernel's directory, so without
    # this a persistent cache would serve kernels traced against old helpers
    model = tmp_path / "sgl_amd_mono" / "model"
    model.mkdir(parents=True)
    (model / "__init__.py").write_text("")
    (model / "kernel.py").write_text(
        "from aiter.ops.flydsl.kernels import buffer_ops as bo\n"
        "from sgl_amd_mono.model import layout\n"
        "def build():\n"
        "    from .tiles import TILE\n"
    )
    helper = tmp_path / "buffer_ops.py"

    def find_spec(name):
        if name != "aiter.ops.flydsl.kernels.buffer_ops":
            return None
        return importlib.machinery.ModuleSpec(name, None, origin=str(helper))

    monkeypatch.setattr(plan, "_ROOT", tmp_path / "sgl_amd_mono")
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    def digest():
        plan.kernel_sources.cache_clear()
        return plan.kernel_sources("sgl_amd_mono.model.kernel")

    edits = {
        helper: "A = 1\n",
        model / "layout.py": "B = 1\n",
        model / "tiles.py": "TILE = 1\n",
        model / "runner.py": "C = 1\n",
    }
    for f, text in edits.items():
        f.write_text(text)
    before = digest()
    (model / "runner.py").write_text("C = 2\n")
    (model / "layout.py").write_text('"""Doc."""\nB = 1  # a comment\n')
    assert digest() == before, "an unimported module or a comment changed the key"
    for f, text in edits.items():
        if f.name != "runner.py":
            f.write_text(text.replace("1", "2"))
            assert digest() != before, f"an edit to imported {f.name} kept the key"
            f.write_text(text)
