# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65188 at 294644b81e (Apache-2.0):
# vllm/models/qwen4_exp/amd/specialized_decode/wide/abi.py
"""The words the kernels and the host share outside the kernels' arguments.

A rank's health words: word FAILED is set once any in-kernel wait of the rank
gave up (sticky: its later waits exit at once, so a failed step runs to its
end), then a record a CTA of the first wait it gave up on: flag, site, the
counter or flag seen, the target, CTA, clock.

The output rank's flags: one int32 word a TP rank in the first PEER_HEAD bytes
of rank 0's peer buffer. A rank whose wait gives up raises its word there and
drains the store before it sets FAILED. `runtime.guards.FailStop` reads
them after every step.
"""

RECORD_WORDS = 8
FAILED = 0
PEER_HEAD = 256  # bytes: the output rank's flags, a word a rank
WAIT_SECONDS = 10  # an in-kernel wait gives up after this long (vLLM's setting)
WAIT_TICKS = WAIT_SECONDS * 100_000_000  # s_memrealtime ticks at 100 MHz

# wait sites a record names, and in stop mode the page a stopped launch faults at
# (`grid.check`)
SITE_GRID = 1  # a grid barrier
SITE_PEER = 2  # a TP exchange line
SITE_FOLD = 3  # a tagged hand-off between the launch's CTAs (`stages.tagged`)
# stop mode's page for a peer's flag seen (no record names it)
SITE_FLAG = 4


def health_words(ctas: int) -> int:
    """A rank's health words for a grid of `ctas`: the header, a record a CTA."""
    return (ctas + 1) * RECORD_WORDS
