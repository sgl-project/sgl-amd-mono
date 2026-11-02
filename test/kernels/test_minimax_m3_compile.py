"""The MiniMax-M3 launch compiles for every width the spec serves at every TP size
(FlyDSL's compile-only mode, one gfx950 GPU), with a fault-injection site too."""

import pytest
import torch

from sgl_amd_mono.minimax_m3.gfx950.config import TP_SIZES, Dims

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

# MiniMax-M3's constants (its config.json)
BUILD = dict(
    sm_scale=128**-0.5,
    eps=1e-6,
    route_scale=2.0,
    shared_weight=1.0,
    swiglu_limit=7.0,
    init_blocks=0,
    local_blocks=1,
    # where the guard's peer flags sit in the peer buffer: a constant the kernel
    # bakes in, so any offset compiles the same code shape
    flags_offset=0,
)


def _build(tp: int, rows: int, **extra):
    from sgl_amd_mono.minimax_m3.gfx950.post_attn import build_post_attn_kernel

    dims = Dims(tp=tp)
    return build_post_attn_kernel(
        npes=tp, dims=dims, heads=dims.index_heads, tokens=rows, **BUILD, **extra
    )


@pytest.mark.parametrize(
    "tp, rows", [(tp, rows) for tp in TP_SIZES for rows in Dims(tp=tp).widths]
)
def test_every_width_compiles(tp, rows):
    from sgl_amd_mono.minimax_m3.gfx950.packed import compile_only
    from sgl_amd_mono.minimax_m3.spec import SPEC

    assert rows in SPEC.widths("gfx950")
    compile_only(_build(tp, rows))


@pytest.mark.parametrize("tp", TP_SIZES)
def test_a_fault_site_compiles(tp):
    from sgl_amd_mono.minimax_m3.gfx950.packed import compile_only

    rows = Dims(tp=tp).widths[0]
    compile_only(_build(tp, rows, fault_site=1, fault_after=0))
