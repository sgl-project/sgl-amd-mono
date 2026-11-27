# SPDX-License-Identifier: Apache-2.0
"""Reuse a frozen, void host launcher; only its runtime arguments may change."""

from functools import wraps
from inspect import signature
from threading import Lock

import flydsl.compiler as flyc

_HAS_COMPILE_ONLY = "compile_only" in signature(flyc.compile).parameters


def compiled_launch(jit_launch):
    if not _HAS_COMPILE_ONLY:
        return jit_launch
    parameters = signature(jit_launch.func)
    lock = Lock()
    compiled = None

    @wraps(jit_launch.func)
    def launch(*args, **kwargs):
        nonlocal compiled
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        values = tuple(bound.arguments.values())
        if compiled is not None:
            return compiled(*values)
        with lock:
            if compiled is None:
                compiled = flyc.compile(
                    jit_launch, *bound.args, compile_only=True, **bound.kwargs
                )
        return compiled(*values)

    return launch
