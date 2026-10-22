# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The few device helpers and the argument-list check the dense launches share
that ``sgl_amd_mono.common`` does not carry."""

import inspect
import math

import flydsl.expr as fx
import msgspec

from sgl_amd_mono.common.ops import hw_exp2, ld_i32, uniform

LOG2E = 1.0 / math.log(2.0)
TAG_SLOTS = 256


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (word 0 of the guard block,
    bumped once a step in the graph) and the launch's slot, so no pair of
    another step or launch passes. Never 0, the zeroed scratch's tag."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def pow2(code):
    """The fp32 power of two an e8m0 code stands for."""
    return (code << 23).bitcast(fx.Float32)


# A kernel's argument list, named once.
#
# A FlyDSL kernel and its launcher each spell out their positional parameters, and
# the host passes them by position: an argument out of order is a pointer read as
# another, silently. ``KernelAbi`` is the one list of names; the build checks the
# kernel's and the launcher's parameters against it, and the host packs by name.


def _python_params(fn) -> tuple[str, ...]:
    """Parameter names of a ``@flyc.kernel`` / ``@flyc.jit`` object's Python
    function (FlyDSL keeps it as ``_func`` / ``func``)."""
    py = getattr(fn, "func", None) or getattr(fn, "_func", None)
    if py is None:
        raise TypeError(f"no Python function behind {fn!r}")
    return tuple(inspect.signature(py).parameters)


class KernelAbi(msgspec.Struct, frozen=True):
    names: tuple[str, ...]

    def check(self, kernel, launcher) -> None:
        """The kernel takes exactly ``names``; the launcher too, then ``stream``."""
        got = _python_params(kernel)
        if got != self.names:
            raise TypeError(f"kernel parameters {got} != ABI {self.names}")
        got = _python_params(launcher)
        if got != (*self.names, "stream"):
            raise TypeError(f"launcher parameters {got} != ABI {self.names} + stream")

    def pack(self, values: dict) -> list:
        """``values`` (every name, nothing else) in ABI order."""
        if len(values) == len(self.names) and all(n in values for n in self.names):
            return [values[n] for n in self.names]
        extra = sorted(set(values) - set(self.names))
        missing = sorted(set(self.names) - set(values))
        raise TypeError(f"ABI mismatch: missing {missing}, unexpected {extra}")
