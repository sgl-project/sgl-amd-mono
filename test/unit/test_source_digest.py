"""The kernels' cache key: every FlyDSL launcher closes over its module's import-time
sources digest and every build parameter, and the digest follows what the module
imports."""

import ast
import functools
import sys
import types
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


def _digests_at_import(tree: ast.Module) -> bool:
    """Whether the module assigns ``SOURCES = kernel_sources(__name__)`` at import."""
    return any(
        isinstance(n, ast.Assign)
        and [getattr(t, "id", None) for t in n.targets] == ["SOURCES"]
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "kernel_sources"
        and [getattr(a, "id", None) for a in n.value.args] == ["__name__"]
        for n in tree.body
    )


def _keys_the_build(builder: ast.FunctionDef) -> bool:
    """Whether ``builder`` assigns ``keyed`` from ``SOURCES`` and each of its
    parameters, or its whole ``locals()``."""
    params = {a.arg for a in builder.args.args + builder.args.kwonlyargs}
    for n in ast.walk(builder):
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", None) == "keyed" for t in n.targets
        ):
            reads = _reads(n.value)
            if "SOURCES" in reads and ("locals" in reads or params <= reads):
                return True
    return False


def _launchers():
    """Each ``@flyc.jit`` launcher in the package, with whether it closes over a
    ``keyed`` that holds its module's import-time digest and every build parameter
    (itself or through a kernel it calls)."""
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
            keyed_kernels = {
                f.name
                for f in ast.walk(builder)
                if isinstance(f, ast.FunctionDef)
                and _flyc(f, "kernel")
                and "keyed" in _reads(f)
            }
            reads = _reads(fn)
            yield (
                name,
                (
                    _digests_at_import(tree)
                    and _keys_the_build(builder)
                    and ("keyed" in reads or bool(reads & keyed_kernels))
                ),
            )


def test_every_launcher_closes_over_the_sources_digest():
    # FlyDSL keys a launcher on its closure's scalars and same-directory helpers only,
    # so one missing the digest or a build parameter keeps serving stale kernels
    launchers = dict(_launchers())
    assert launchers, "found no @flyc.jit launcher"
    missing = sorted(name for name, ok in launchers.items() if not ok)
    assert not missing, (
        "close over `keyed` built from the module's `SOURCES = kernel_sources(__name__)`"
        f" and every build parameter: {missing}"
    )


def test_the_digest_follows_the_code_the_kernel_module_imports(tmp_path, monkeypatch):
    # FlyDSL's own key ignores modules outside the kernel's directory, so without
    # this a persistent cache would serve kernels traced against old helpers
    plugin, aiter = tmp_path / "sgl_amd_mono", tmp_path / "kernels"
    model = plugin / "model"
    for d in (model, aiter):
        d.mkdir(parents=True)
    for package, root in (
        (plugin, "sgl_amd_mono"),
        (aiter, "aiter.ops.flydsl.kernels"),
    ):
        (package / "__init__.py").write_text("")
        module = types.ModuleType(root)
        module.__file__ = str(package / "__init__.py")
        monkeypatch.setitem(sys.modules, root, module)
    (model / "__init__.py").write_text("")
    (model / "kernel.py").write_text(
        "from aiter.ops.flydsl.kernels import buffer_ops as bo\n"
        "from aiter.ops.flydsl.kernels.dpp import shuffle\n"
        "import sgl_amd_mono.model.layout\n"
        "def build():\n"
        "    from ..model.tiles import TILE\n"
    )
    for cached in ("_module_code", "kernel_sources"):
        monkeypatch.setattr(
            plan, cached, functools.cache(getattr(plan, cached).__wrapped__)
        )

    def digest():
        plan._module_code.cache_clear()
        plan.kernel_sources.cache_clear()
        return plan.kernel_sources("sgl_amd_mono.model.kernel")

    edits = {
        aiter / "buffer_ops.py": "A = 1\n",
        aiter / "dpp.py": "def shuffle():\n    return 1\n",
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
