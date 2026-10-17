# SPDX-License-Identifier: Apache-2.0
"""K1's CTA layout, the MoE weight-table extents, and the kernels' mailbox use.

Like :mod:`test_minimax_m3_dims` and :mod:`test_minimax_m3_layout`, the values
below were module-level constants in the kernels' in-tree ancestor, bound by
value at import from a TP read out of the environment. Every one was compared
against that ancestor at TP 2, 4 and 8 before it was written down here; this
test keeps them from drifting afterwards.

The traced kernel bodies need a GPU and FlyDSL, so they cannot run here. What
this file pins instead is everything decided at BUILD time -- plain Python --
plus a static check of the one thing a GPU-free run cannot otherwise catch:
that every mailbox hand-off passes a named ``Addr`` and not a raw address.
``Mailbox`` dereferences ``a.value`` and ``a.space`` unconditionally, so a raw
address is an AttributeError at trace time, i.e. on a GPU.
"""

import ast
import importlib
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.minimax_m3.gfx950.config import TP_SIZES, Dims

GFX = Path(__file__).resolve().parents[2] / "sgl_amd_mono/minimax_m3/gfx950"


class _MockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answer imports under ``roots`` with MagicMock modules.

    The kernel modules import FlyDSL and aiter at module scope, which a CPU test
    environment does not have. Nothing they reach is evaluated at import -- the
    traced code is all inside functions -- so a mock module is enough to let the
    bodies of :mod:`pre_attn` and :mod:`post_attn` execute and expose their
    build-time structs.
    """

    def __init__(self, roots):
        self.roots = tuple(roots)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots or any(
            fullname == r or fullname.startswith(r + ".") for r in self.roots
        ):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = MagicMock(name=spec.name)
        m.__name__, m.__spec__, m.__path__, m.__all__ = spec.name, spec, [], []
        return m

    def exec_module(self, module):
        pass


# Device helpers the kernels import from ``sgl_amd_mono.common.ops``, which that
# module does not define yet; and the module path ``select.py`` imports, which
# does not exist in this tree. Both are the shared-runtime move, tracked
# separately -- stubbed only where still absent, so this test stops stubbing by
# itself once they land.
_ABSENT_OPS = (
    "fp8x8_bf16_pk", "readlane", "unlikely", "wave_max", "wave_sum",
    "block_max", "kernel_symbol",
)  # fmt: skip
_ABSENT_MODULE = "sgl_amd_mono.minimax_m3.gfx950.common"


def _kernel_modules():
    """(pre_attn, post_attn), imported with whatever is absent mocked out."""
    roots = [r for r in ("flydsl", "aiter") if importlib.util.find_spec(r) is None]
    if importlib.util.find_spec(_ABSENT_MODULE) is None:
        roots.append(_ABSENT_MODULE)
    if roots:
        sys.meta_path.insert(0, _MockFinder(roots))
    import sgl_amd_mono.common.ops as ops

    for name in _ABSENT_OPS:
        if not hasattr(ops, name):
            setattr(ops, name, MagicMock(name=name))
    pre = importlib.import_module("sgl_amd_mono.minimax_m3.gfx950.pre_attn")
    post = importlib.import_module("sgl_amd_mono.minimax_m3.gfx950.post_attn")
    return pre, post


pre_attn, post_attn = _kernel_modules()

# tp -> K1's CTA layout and scratch regions
K1_EXPECTED = {
    2: dict(
        q_off=0, k_off=4096, v_off=4352,
        n_head_tasks=38, k_task0=32, v_task0=34, iq_task=36, ik_task=37,
        kv_page_bytes=4096, tasks_per_cta=2, one_gemv=304, one_gemv_ctas=152,
        other_heads=3, other_gemv=24, ht0=152, nt0=190, xt0=206,
        scratch_x8=671744, scratch_x8s=868352, scratch_hdone=868480,
        scratch_rdone=873344, scratch_bytes=873472,
    ),
    4: dict(
        q_off=0, k_off=2048, v_off=2176,
        n_head_tasks=20, k_task0=16, v_task0=17, iq_task=18, ik_task=19,
        kv_page_bytes=2048, tasks_per_cta=1, one_gemv=160, one_gemv_ctas=160,
        other_heads=3, other_gemv=24, ht0=160, nt0=180, xt0=196,
        scratch_x8=376832, scratch_x8s=573440, scratch_hdone=573568,
        scratch_rdone=576128, scratch_bytes=576256,
    ),
    8: dict(
        q_off=0, k_off=1024, v_off=1152,
        n_head_tasks=12, k_task0=8, v_task0=9, iq_task=10, ik_task=11,
        kv_page_bytes=2048, tasks_per_cta=1, one_gemv=96, one_gemv_ctas=96,
        other_heads=3, other_gemv=24, ht0=96, nt0=108, xt0=124,
        scratch_x8=245760, scratch_x8s=442368, scratch_hdone=442496,
        scratch_rdone=444032, scratch_bytes=444160,
    ),
}

# tp -> the rank's share of the MXFP4 expert tables
MOE_EXPECTED = {
    2: dict(
        w13_rows=3072, w13_bytes=9437184, w2_bytes=4718592,
        w2_scale_cols=48, s2_bytes=38043648, s13_bytes=76087296,
    ),
    4: dict(
        w13_rows=1536, w13_bytes=4718592, w2_bytes=2359296,
        w2_scale_cols=24, s2_bytes=19021824, s13_bytes=38043648,
    ),
    8: dict(
        w13_rows=768, w13_bytes=2359296, w2_bytes=1179648,
        w2_scale_cols=16, s2_bytes=12681216, s13_bytes=19021824,
    ),
}


@pytest.mark.parametrize("tp", TP_SIZES)
def test_k1_cta_layout(tp):
    k1 = pre_attn.K1Layout(dims=Dims(tp=tp))
    got = {name: getattr(k1, name) for name in K1_EXPECTED[tp]}
    assert got == K1_EXPECTED[tp]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_moe_weight_tables(tp):
    w = post_attn.MoeWeights(dims=Dims(tp=tp))
    got = {name: getattr(w, name) for name in MOE_EXPECTED[tp]}
    assert got == MOE_EXPECTED[tp]
    # w13's scale is HIDDEN-wide and never pads, so it stays a module constant
    assert post_attn.W13_SCALE_COLS == 192


@pytest.mark.parametrize("tp", TP_SIZES)
def test_w2_scale_width_is_the_padded_one(tp):
    """``group_scale_byte`` strides an aiter-shuffled w2 scale by ``cols // 8``.
    At TP8 the natural 12 columns are stored as 16, and the unpadded stride
    would read the wrong scale byte for every k past 256."""
    dims = Dims(tp=tp)
    w = post_attn.MoeWeights(dims=dims)
    assert w.w2_scale_cols == dims.padded_scale_cols
    assert (w.w2_scale_cols != dims.scale_cols) == (tp == 8)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_k1_tasks_fit_the_co_scheduled_ctas(tp):
    """The GEMV, head, norm and other-index-head CTAs are disjoint and must fit
    the BLOCKS the kernel co-schedules. TP2's 304 GEMV tasks only fit because a
    CTA folds two of them: unfolded it would need 382."""
    k1 = pre_attn.K1Layout(dims=Dims(tp=tp))
    assert k1.xt0 + k1.other_gemv <= BLOCKS
    unfolded = k1.one_gemv + k1.n_head_tasks + 16 + k1.other_gemv
    assert (unfolded > BLOCKS) == (tp == 2)
    assert (k1.tasks_per_cta > 1) == (tp == 2)


@pytest.mark.parametrize("tp", TP_SIZES)
def test_k1_scratch_regions_are_ordered_and_disjoint(tp):
    k1 = pre_attn.K1Layout(dims=Dims(tp=tp))
    offsets = [
        pre_attn.SCRATCH_QKV,
        k1.scratch_x8,
        k1.scratch_x8s,
        k1.scratch_hdone,
        k1.scratch_rdone,
    ]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert k1.scratch_bytes > offsets[-1]
    # qkv holds a (value, tag) pair per row of the WIDEST projection any build
    # asks for, so the x8 rows start past every mode's rows
    assert k1.scratch_x8 >= Dims(tp=tp).max_qkv_rows * 8


def test_no_per_rank_shape_is_a_module_constant_again():
    """The point of the conversion is that a rank's share is not known until the
    server's TP is. These names were module constants of a one-TP build; a
    module constant with any of them back would be bound at import again."""
    gone = {
        "TP", "LOCAL_Q_HEADS", "LOCAL_KV_HEADS", "LOCAL_INDEX_HEADS", "INTER",
        "O_K", "MAX_QKV_ROWS", "PADDED_SCALE_COLS", "ONE_INDEX_HEAD",
        "N_HEAD_TASKS", "K_OFF", "V_OFF", "IQ_TASK", "IK_TASK", "KV_PAGE_BYTES",
        "ONE_GEMV", "ONE_GEMV_CTAS", "TASKS_PER_CTA", "HT0", "NT0", "XT0",
        "SCRATCH_X8", "SCRATCH_X8S", "SCRATCH_HDONE", "SCRATCH_RDONE",
        "SCRATCH_BYTES", "W13_ROWS", "W13_BYTES", "W2_BYTES", "W2_SCALE_COLS",
        "S2_BYTES", "S13_BYTES",
    }
    for mod in (pre_attn, post_attn):
        assert gone.isdisjoint(vars(mod)), sorted(gone & set(vars(mod)))


# ------------------------------------------------------- mailbox hand-off form
KERNEL_FILES = ("index_score.py", "pre_attn.py", "post_attn.py")
MAILBOX_CALLS = {"mb_put", "mb_put_bf", "mb_put_words", "mb_poll", "k1_poll"}
# the constructors and helpers that return a named ``Addr``
ADDR_SOURCES = {"sreg", "preg", "mba", "sy", "at", "scores_copy", "Addr"}
# the scratch regions a stage reads back as plain words through ``rsrc``: their
# flag is a mailbox pair of its own, so their base is deliberately NOT an Addr
PLAIN_WORD_REGIONS = {"attnq", "xn8p", "xscp", "mid8p", "midsp"}


def _addr_nodes(call):
    """The address expressions of a mailbox call: a putter's first argument, and
    element 0 of every (addr, index, n) spec tuple anywhere in its arguments."""
    out = []
    if call.func.id != "mb_poll" and call.func.id != "k1_poll" and call.args:
        out.append(call.args[0])
    for node in ast.walk(call):
        if isinstance(node, ast.Tuple) and len(node.elts) == 3:
            out.append(node.elts[0])
    return out


@pytest.mark.parametrize("name", KERNEL_FILES)
def test_every_mailbox_address_is_a_named_region(name):
    """``Mailbox`` dereferences ``a.value`` / ``a.space``, so a bare address or
    a sum of one is an AttributeError when the kernel is traced. Every address
    must therefore be a local or a call to one of ``ADDR_SOURCES`` -- never an
    arithmetic expression, and never the raw ``mb(...)``."""
    tree = ast.parse((GFX / name).read_text())
    bad = []
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in MAILBOX_CALLS
        ):
            continue
        for a in _addr_nodes(call):
            if isinstance(a, ast.Name):
                continue
            if isinstance(a, ast.Call) and isinstance(a.func, ast.Name):
                if a.func.id in ADDR_SOURCES:
                    continue
                if a.func.id == "mb":
                    bad.append((a.lineno, "raw mb(...)"))
                    continue
            bad.append((a.lineno, ast.dump(a)[:60]))
    assert not bad, bad


def test_raw_scratch_bases_are_only_the_plain_word_regions():
    """``mb`` returns a raw address and ``mba`` a mailbox one. Only the regions
    a stage reads back through ``rsrc`` may go through ``mb``; anything else
    taking that path would be a hand-off with no tag."""
    tree = ast.parse((GFX / "post_attn.py").read_text())
    raw = set()
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "mb"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            raw.add(call.args[0].value)
    assert raw == PLAIN_WORD_REGIONS


def test_the_k1_mailboxes_are_named_and_x8_is_the_one_exception():
    """K1's scratch hand-offs, as ``sreg`` names them. ``x8_mb`` is absent by
    design: the norm tasks' x8 rows are plain words that the GEMV CTAs read
    through ``rsrc``, flagged by ``x8s_mb``'s pair."""
    tree = ast.parse((GFX / "pre_attn.py").read_text())
    named = {
        c.args[2].value
        for c in ast.walk(tree)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "sreg"
        and isinstance(c.args[2], ast.Constant)
    }
    assert named == {"qkv", "x8s", "hdone", "rdone"}
