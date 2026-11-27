# SPDX-License-Identifier: Apache-2.0
"""Verify compiled_launch compiles once with compile_only=True on FlyDSL 0.5+ and passes through on 0.4."""

from types import SimpleNamespace

from sgl_amd_mono.kimi_k3.gfx950 import compiled_launch as mod


def test_compiled_launch_freezes_once_and_invokes_every_call(monkeypatch):
    compile_calls = []
    exec_calls = []

    def fake_compiled(*values):
        exec_calls.append(values)

    def fake_compile(jit_fn, *args, compile_only=False, **kwargs):
        compile_calls.append((jit_fn, args, compile_only, kwargs))
        return fake_compiled

    def body(a, b, stream=None):
        return (a, b, stream)

    jit_launch = SimpleNamespace(func=body)
    monkeypatch.setattr(mod, "_HAS_COMPILE_ONLY", True)
    monkeypatch.setattr(mod.flyc, "compile", fake_compile)

    wrapped = mod.compiled_launch(jit_launch)
    wrapped(1, 2)
    wrapped(3, 4, stream="s")

    assert len(compile_calls) == 1
    assert compile_calls[0][2] is True
    assert exec_calls == [(1, 2, None), (3, 4, "s")]


def test_compiled_launch_passes_through_without_compile_only(monkeypatch):
    def jit_launch(a, b):
        return a + b

    monkeypatch.setattr(mod, "_HAS_COMPILE_ONLY", False)
    wrapped = mod.compiled_launch(jit_launch)
    assert wrapped is jit_launch
    assert wrapped(2, 5) == 7
