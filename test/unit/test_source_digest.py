"""The kernels' cache key follows the aiter FlyDSL helpers their sources import."""

import importlib.machinery
import importlib.util

from sgl_amd_mono.common import plan


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
