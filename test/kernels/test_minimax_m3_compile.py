"""The MiniMax-M3 launch compiles for every width the spec serves (FlyDSL's
compile-only mode, one gfx950 GPU), with a fault-injection site too."""

import pytest
import torch

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
    npes=4,
    sm_scale=128**-0.5,
    eps=1e-6,
    route_scale=2.0,
    shared_weight=1.0,
    swiglu_limit=7.0,
    init_blocks=0,
    local_blocks=1,
)


@pytest.mark.parametrize("rows", range(1, 17))
def test_every_width_compiles(rows):
    from sgl_amd_mono.minimax_m3.gfx950.post_attn import build_post_attn_kernel
    from sgl_amd_mono.minimax_m3.gfx950.runner import compile_only
    from sgl_amd_mono.minimax_m3.spec import SPEC

    assert rows in SPEC.widths("gfx950")
    compile_only(build_post_attn_kernel(tokens=rows, **BUILD))


def test_a_fault_site_compiles():
    from sgl_amd_mono.minimax_m3.gfx950.post_attn import build_post_attn_kernel
    from sgl_amd_mono.minimax_m3.gfx950.runner import compile_only

    compile_only(build_post_attn_kernel(tokens=4, fault_site=1, fault_after=0, **BUILD))
