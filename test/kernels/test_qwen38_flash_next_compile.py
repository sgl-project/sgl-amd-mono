"""The Qwen3.8-Flash-Next launches compile for every batch size the spec serves
(FlyDSL's compile-only mode, one gfx950 GPU): the four launch kinds of a verify
(prelude, PLE, middle, final mixer), the bf16-SSM build, the MTP draft's launch,
and a fault-injection site."""

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

# representative cache strides (a 262144-token context); the strides are build
# parameters, so the served ones compile at the first forward
STRIDES = {"rtt_stride": 262208, "pos_stride": 0, "eos_token_id": 248044}
KINDS = {
    "prelude": {"plan": "gdn", "prelude": True, "first_src": "embed"},
    "ple": {"plan": "gdn,gdn,qsa", "ple_first": True},
    "middle": {"plan": "gdn,gdn,gdn,qsa"},
    "final": {"plan": "gdn,gdn,gdn,qsa", "final_mix": True},
}


def _compile(**kw):
    from sgl_amd_mono.qwen38_flash_next.gfx950.config import MonoBuild
    from sgl_amd_mono.qwen38_flash_next.gfx950.kernel import (
        build_qwen38_flash_next_kernel,
    )
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import compile_only

    compile_only(build_qwen38_flash_next_kernel(MonoBuild(**{**STRIDES, **kw})))


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("requests", range(1, 9))
def test_every_launch_compiles(requests, kind):
    from sgl_amd_mono.qwen38_flash_next.spec import SPEC, TOKENS

    assert requests * TOKENS in SPEC.widths("gfx950")
    _compile(nreq=requests, ntok=TOKENS, **KINDS[kind])


def test_the_bf16_ssm_launch_compiles():
    # SGLANG_MAMBA_SSM_DTYPE=bfloat16, int64 mamba slot map, [3, S] MRoPE positions
    _compile(
        nreq=8, ntok=4, ssm_bf16=True, map_i64=True, pos_stride=32, **KINDS["middle"]
    )


@pytest.mark.parametrize("tokens", [1, 4], ids=["draft decode", "draft extend"])
def test_the_mtp_draft_launch_compiles(tokens):
    _compile(nreq=8, ntok=tokens, plan="qsa", final_mix=True, first_src="mtp")


def test_a_fault_site_compiles():
    _compile(nreq=1, ntok=4, fault_site=1, fault_after=0, **KINDS["prelude"])
