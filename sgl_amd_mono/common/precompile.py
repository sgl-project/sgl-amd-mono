# SPDX-License-Identifier: Apache-2.0
"""Compile a FlyDSL launch ahead of its first call, without running it."""

import inspect
import os

import torch


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    # FlyDSL's documented compile-without-launch switch: the compile_only=True keyword
    # of flyc.compile needs FlyDSL >= 0.5.1, newer than the one in SGLang's ROCm image.
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = old
