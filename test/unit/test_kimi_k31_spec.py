# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3.1 spec routes steps of up to 8 rows to Kimi-K3's launches and the
verify steps above them to the wide launch."""

from types import SimpleNamespace

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import MAX_ROWS
from sgl_amd_mono.kimi_k31 import spec as k31
from sgl_amd_mono.kimi_k31 import wide


def test_registered_under_the_k31_architecture():
    assert (
        sgl_amd_mono.SPECS[("KimiK31ForConditionalGeneration", None)]
        == "sgl_amd_mono.kimi_k31.spec:SPEC"
    )


def test_the_narrow_and_wide_widths_meet_without_a_gap():
    # _Built routes rows up to NARROW to Kimi-K3's launches, which are built up to
    # MAX_ROWS; a width between the two would route to a launch never built
    assert wide.NARROW == MAX_ROWS
    widths = k31.SPEC.widths("gfx950")
    assert tuple(range(1, MAX_ROWS + 1)) == widths[:MAX_ROWS]
    assert all(w > wide.NARROW for w in widths[MAX_ROWS:])


def test_the_wide_launch_names_its_server_args():
    def cfg(**kw):
        return SimpleNamespace(
            **{
                "speculative_algorithm": "DFLASH",
                "speculative_num_draft_tokens": 6,
                "enable_linear_replayssm_spec": True,
                **kw,
            }
        )

    assert k31._wide_config_disable_reasons(cfg()) == []
    (why,) = k31._wide_config_disable_reasons(cfg(enable_linear_replayssm_spec=False))
    assert (
        "--enable-linear-replayssm-spec" in why and "SGLANG_AMD_MONO_K31_WIDE=0" in why
    )
    decode = cfg(speculative_algorithm=None, speculative_num_draft_tokens=None)
    assert len(k31._wide_config_disable_reasons(decode)) == 2
