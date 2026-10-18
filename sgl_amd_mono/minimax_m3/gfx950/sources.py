# SPDX-License-Identifier: Apache-2.0
"""The kernel sources in every build's cache key: an edit to them rebuilds."""

from sgl_amd_mono.common.plan import source_digest

SOURCES = source_digest("common", "minimax_m3/gfx950")
