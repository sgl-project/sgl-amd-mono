"""``common.resources``: a code object's metadata read out of a FlyDSL launcher's
module text, and what ``check`` does with a kernel past its budget, one it cannot
read, one whose occupancy FlyDSL cannot give and one the CUs cannot hold, on one
rank and across ranks (two gloo processes on CPU).

Most code objects are built here: a 64-bit ELF with an AMDGPU metadata note, inside a
``gpu.binary``'s ``bin = "..."`` escaped as MLIR prints it. One test compiles a FlyDSL
kernel (compile-only, no GPU needed) and reads it.
"""

import inspect
import logging
import queue
import struct
import time
from types import SimpleNamespace

import msgspec
import pytest

from sgl_amd_mono.common import resources

SYMBOL = "_ZN5aiter8kernel_aE"
OTHER = "_ZN5aiter8kernel_bE"
FIELDS = {
    "vgpr": 256,
    "agpr": 0,
    "sgpr": 100,
    "vgpr_spill": 200,
    "sgpr_spill": 300,
    "scratch": 288,
    "lds": 90000,
}


def _kernel(symbol=SYMBOL, **fields) -> dict:
    meta = {**FIELDS, **fields}
    keys = resources._KEYS.items()
    return {".symbol": f"{symbol}.kd"} | {
        k: meta[f] for f, k in keys if meta.get(f) is not None
    }


def _elf(symbol=SYMBOL, kernels=None, note=32, **fields) -> bytes:
    """A 64-bit ELF: a ``note``-type note holding the metadata of ``kernels`` (one
    kernel of ``FIELDS`` by default), then a section of bytes MLIR escapes."""
    meta = {"amdhsa.kernels": kernels or [_kernel(symbol, **fields)]}
    desc = msgspec.msgpack.encode(meta)
    notes = struct.pack("<III", 7, len(desc), note) + b"AMDGPU\0\0" + desc
    sections = [(7, notes + bytes(-len(desc) % 4)), (1, b'\x00\x01"\\\x7f')]
    body, at = b"", 64
    headers = bytes(64)
    for kind, data in sections:
        headers += struct.pack("<IIQQQQIIQQ", 0, kind, 0, 0, at, len(data), 0, 0, 4, 0)
        body, at = body + data, at + len(data)
    ehdr = b"\x7fELF\x02\x01\x01" + bytes(9)
    ehdr += struct.pack("<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, at, 0, 64, 0, 0, 64, 3, 0)
    return ehdr + body + headers


def _escape(data: bytes) -> str:
    """As MLIR prints a string attribute (llvm::printEscapedString)."""
    out = []
    for b in data:
        if b == 0x5C:
            out.append("\\\\")
        elif 0x20 <= b < 0x7F and b != 0x22:
            out.append(chr(b))
        else:
            out.append(f"\\{b:02X}")
    return "".join(out)


def _module(elf: bytes) -> str:
    return (
        'gpu.binary @kernels [#gpu.object<#rocdl.target<chip = "gfx950">, '
        f'bin = "{_escape(elf)}">]'
    )


def _artifact(elf: bytes, blocks=lambda: 1) -> SimpleNamespace:
    return SimpleNamespace(ir=_module(elf), max_blocks_per_cu=blocks)


def _launcher(*elfs, blocks=lambda: 1) -> SimpleNamespace:
    """An ``@flyc.jit`` launcher once built: an artifact a specialization."""
    return SimpleNamespace(
        _mem_cache={i: _artifact(e, blocks) for i, e in enumerate(elfs)}
    )


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """No GPU unless a test gives one (so no occupancy query), the default mode, and
    nothing checked or warned about yet in this process."""
    monkeypatch.setattr(resources, "_compute_units", lambda: None)
    monkeypatch.setattr(resources, "_RESIDENT", {})
    monkeypatch.setattr(resources, "_LOGGED", set())
    monkeypatch.setattr(resources, "_WARNED", set())
    monkeypatch.delenv("SGLANG_AMD_MONO_RESOURCES", raising=False)


def _records(caplog) -> list:
    return [r for r in caplog.records if r.name == resources.__name__]


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in _records(caplog) if r.levelname == "WARNING"]


def test_read_a_launchers_code_object():
    (art,) = resources.artifacts(_launcher(_elf()))
    r = resources.read(resources.code_object(art.ir))
    assert (r.name, r.symbol) == ("kernel_a", SYMBOL)
    assert {f: getattr(r, f) for f in FIELDS} == FIELDS


@pytest.mark.parametrize("quote", ["\\22", '\\"'])
def test_the_escapes_come_back_byte_for_byte(quote):
    elf = _elf()
    assert resources.code_object(_module(elf).replace("\\22", quote)) == elf


def test_an_artifact_is_read_by_itself():
    assert len(resources.artifacts(_artifact(_elf()))) == 1


def test_counts_an_older_code_object_lacks_read_as_none():
    r = resources.read(_elf(vgpr_spill=None, sgpr_spill=None, agpr=None))
    assert (r.vgpr_spill, r.sgpr_spill, r.agpr) == (None, None, None)
    assert r.scratch == FIELDS["scratch"]


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(_mem_cache={0: SimpleNamespace(ir=text)})


UNREADABLE = {
    "not built yet": SimpleNamespace(_mem_cache={}),
    "no closing quote": _text(_module(_elf()).rsplit('"', 1)[0]),
    "a bad hex escape": _text(_module(_elf()).replace("\\00", "\\Z9", 1)),
    "a truncated code object": _text(_module(_elf()[:100])),
    "no AMDGPU metadata note": _text(_module(_elf(note=1))),
    "two kernels in one code object": _text(
        _module(_elf(kernels=[_kernel(), _kernel(OTHER)]))
    ),
    "a kernel without its .symbol": _text(_module(_elf(kernels=[{".vgpr_count": 1}]))),
    "no ELF": _text(_module(b"not an ELF")),
}


@pytest.mark.parametrize("bad", UNREADABLE.values(), ids=UNREADABLE.keys())
def test_an_unreadable_launcher_warns_once_and_is_skipped_unless_strict(
    monkeypatch, caplog, bad
):
    assert resources.check(bad) == [] and resources.check(bad) == []
    assert len(_warnings(caplog)) == 1
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "strict")
    with pytest.raises(resources.Unreadable):
        resources.check(bad)


def test_past_the_budget_warns_by_default(caplog):
    budget = resources.Budget(scratch=32)
    found = resources.check(_launcher(_elf()), budget=budget)
    assert [r.scratch for r in found] == [FIELDS["scratch"]]
    (msg,) = _warnings(caplog)
    assert "kernel_a" in msg and "budget 32 B" in msg


def test_past_the_budget_raises_when_strict(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "strict")
    with pytest.raises(resources.OverBudget, match="past 32 B a lane"):
        resources.check(_launcher(_elf()), budget=resources.Budget(scratch=32))


def test_within_the_budget_logs_only_when_reporting(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=resources.__name__)
    within = _launcher(_elf(scratch=0, vgpr_spill=0))
    resources.check(within)
    assert _records(caplog) == []
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "report")
    resources.check(within)
    assert [r.levelname for r in _records(caplog)] == ["INFO"]


def test_a_repeated_build_logs_nothing(caplog):
    caplog.set_level(logging.INFO, logger=resources.__name__)
    assert len(resources.check(_launcher(_elf()), log=False)) == 1
    assert _records(caplog) == []


def test_a_code_object_seen_before_is_returned_and_not_logged_again(caplog):
    launcher = _launcher(_elf())
    assert len(resources.check(launcher)) == len(resources.check(launcher)) == 1
    assert len(_warnings(caplog)) == 1


def test_the_same_name_with_other_bytes_is_checked_again(caplog):
    # FlyDSL keeps a kernel name unique only within its module: unnamed kernels of
    # two builds share one
    name = "kernel_0"
    resources.check(_launcher(_elf(symbol=name, scratch=0, vgpr_spill=0)))
    resources.check(_launcher(_elf(symbol=name, scratch=64)))
    (msg,) = _warnings(caplog)
    assert "scratch 64 B" in msg


def test_a_code_object_warned_about_still_raises_when_strict(monkeypatch):
    over = _launcher(_elf())
    resources.check(over, budget=resources.Budget(scratch=32))
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "strict")
    with pytest.raises(resources.OverBudget):
        resources.check(over, budget=resources.Budget(scratch=32))


def test_a_tighter_budget_warns_again(caplog):
    launcher = _launcher(_elf())
    resources.check(launcher, budget=resources.Budget(scratch=512))
    assert _warnings(caplog) == []
    resources.check(launcher, budget=resources.Budget(scratch=32))
    (msg,) = _warnings(caplog)
    assert "budget 32 B" in msg


def test_one_workgroup_a_cu_holds_the_grid_asked_once(monkeypatch, caplog):
    monkeypatch.setattr(resources, "_compute_units", lambda: 256)
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "report")
    caplog.set_level(logging.INFO, logger=resources.__name__)
    asked = []
    launcher = _launcher(_elf(scratch=0), blocks=lambda: asked.append(1) or 1)
    resources.check(launcher)
    resources.check(launcher)
    assert asked == [1]
    (rec,) = _records(caplog)
    assert rec.getMessage().endswith("1 resident a CU")


@pytest.mark.parametrize("mode", resources.MODES)
def test_a_grid_the_cus_cannot_hold_raises_in_every_mode_every_time(monkeypatch, mode):
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", mode)
    monkeypatch.setattr(resources, "_compute_units", lambda: 256)
    launcher = _launcher(_elf(scratch=0), blocks=lambda: 0)
    for _ in range(2):
        with pytest.raises(resources.NotResident, match="hold 0 of the grid's 256"):
            resources.check(launcher)


def test_flydsls_failed_query_counts_as_none_resident(monkeypatch):
    def failed():
        raise RuntimeError("max_blocks_per_cu(): HIP occupancy query failed")

    monkeypatch.setattr(resources, "_compute_units", lambda: 256)
    with pytest.raises(resources.NotResident, match="or HIP could not answer"):
        resources.check(_launcher(_elf(scratch=0), blocks=failed))


def _needs_constant_sizes():
    raise ValueError("max_blocks_per_cu() needs constant block and smem sizes")


def _no_runtime_library():
    raise OSError("libmlir_rocm_runtime.so: cannot open shared object file")


NO_OCCUPANCY = {
    "older FlyDSL": None,
    "a launch it cannot size": _needs_constant_sizes,
    "no runtime library": _no_runtime_library,
}


@pytest.mark.parametrize("blocks", NO_OCCUPANCY.values(), ids=NO_OCCUPANCY.keys())
def test_an_occupancy_flydsl_cannot_give_warns_once_unless_strict(
    monkeypatch, caplog, blocks
):
    monkeypatch.setattr(resources, "_compute_units", lambda: 256)
    elfs = [_elf(scratch=0), _elf(symbol=OTHER, scratch=0)]
    launcher = _launcher(*elfs, blocks=blocks)
    assert len(resources.check(launcher)) == 2
    (msg,) = _warnings(caplog)
    assert "residency not checked" in msg
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "strict")
    with pytest.raises(resources.NoOccupancy):
        resources.check(launcher)


def test_an_unknown_mode_is_refused(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "loud")
    with pytest.raises(ValueError, match="loud"):
        resources.check(_launcher(_elf()))


@pytest.mark.parametrize("tail", ["Ev", "E"])
def test_a_long_symbol_that_is_not_nested_reads_fast(tail):
    # 24 digit runs, not a nested name (no closing E, or a part with no length
    # prefix): a backtracking regex over it is exponential in its digit runs
    symbol = "_ZN5aiter8kernel_a" + "_x1" * 22 + tail
    t0 = time.perf_counter()
    r = resources.read(_elf(symbol=symbol))
    assert time.perf_counter() - t0 < 1.0
    assert r.name == symbol


def test_a_dump_directory_is_read_kernel_by_kernel(tmp_path):
    for sym in (SYMBOL, OTHER):
        d = tmp_path / sym
        d.mkdir()
        (d / "20_gpu_module_to_binary.mlir").write_text(_module(_elf(symbol=sym)))
    found = resources.read_dump(str(tmp_path))
    assert sorted(r.name for r in found) == ["kernel_a", "kernel_b"]


def _ranks(monkeypatch, *peers):
    """Two or more ranks: this one is rank 0, ``peers`` what the others report."""
    sent = []

    def gather(group, mine):
        sent.append(mine)
        return [mine, *peers]

    monkeypatch.setattr(resources, "_world_size", lambda group: 1 + len(peers))
    monkeypatch.setattr(resources, "_gather", gather)
    return sent


def test_ranks_that_all_pass_return_their_readings(monkeypatch):
    sent = _ranks(monkeypatch, None)
    assert len(resources.check(_launcher(_elf(scratch=0)), group="tp")) == 1
    assert sent == [None]


@pytest.mark.parametrize(
    "peers, raised, match",
    [
        ([("NotResident", "a")], resources.NotResident, r"^rank 1: a$"),
        (
            [("OverBudget", "a"), ("NotResident", "b")],
            resources.OverBudget,
            r"^rank 1: a \(and 1 more ranks\)$",
        ),
        ([("ValueError", "a")], ValueError, r"^rank 1: a$"),
        ([("KeyError", "'ir'")], RuntimeError, r"^rank 1: KeyError: 'ir'$"),
    ],
    ids=[
        "a peer fails",
        "the lowest of two, the other counted",
        "a ValueError keeps its type",
        "any other error by name",
    ],
)
def test_a_peers_failure_raises_on_this_rank(monkeypatch, peers, raised, match):
    _ranks(monkeypatch, *peers)
    with pytest.raises(raised, match=match):
        resources.check(_launcher(_elf(scratch=0)), group="tp")


def test_this_ranks_failure_is_sent_and_raised(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "strict")
    sent = _ranks(monkeypatch, None)
    with pytest.raises(resources.OverBudget, match="^rank 0: .*past 32 B") as err:
        resources.check(
            _launcher(_elf()), budget=resources.Budget(scratch=32), group="tp"
        )
    assert [s[0] for s in sent] == ["OverBudget"]
    assert isinstance(err.value.__cause__, resources.OverBudget)


def test_a_mistyped_mode_raises_a_value_error_with_a_group_too(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_RESOURCES", "loud")
    _ranks(monkeypatch, None)
    with pytest.raises(ValueError, match="^rank 0: .*loud"):
        resources.check(_launcher(_elf()), group="tp")


def _rank_main(rank: int, init: str, out) -> None:
    """One of two gloo ranks: rank 1's kernel cannot be resident, rank 0's can."""
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    resources._compute_units = lambda: 256
    blocks = (lambda: 0) if rank == 1 else (lambda: 1)
    try:
        resources.check(
            _launcher(_elf(scratch=0), blocks=blocks), group=dist.group.WORLD
        )
        out.put((rank, None))
    except Exception as e:
        out.put((rank, f"{type(e).__name__}: {e}"))
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_raise_together(tmp_path):
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_gloo_available():
        pytest.skip("no gloo")
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    init = f"file://{tmp_path / 'pg'}"
    procs = [ctx.Process(target=_rank_main, args=(r, init, out)) for r in (0, 1)]
    for p in procs:
        p.start()
    got = {}
    try:
        for _ in procs:
            rank, seen = out.get(timeout=120)
            got[rank] = seen
    except queue.Empty:
        pass
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
    assert got.keys() == {0, 1}, f"got {got}, exit codes {[p.exitcode for p in procs]}"
    for rank in (0, 1):
        assert got[rank].startswith("NotResident: rank 1: "), got
        assert "(or HIP could not answer)" in got[rank]


def test_flydsl_still_raises_the_text_read_as_no_blocks():
    jit = pytest.importorskip("flydsl.compiler.jit_executor")
    source = inspect.getsource(jit.CompiledArtifact.max_blocks_per_cu)
    assert resources._NO_BLOCKS in source


def test_a_flydsl_build_is_read(monkeypatch):
    """A kernel compiled here, as a runner preloads one: FlyDSL's own module text."""
    flyc = pytest.importorskip("flydsl.compiler")
    fx = pytest.importorskip("flydsl.expr")
    shim = pytest.importorskip("aiter.ops.flydsl.kernels.tensor_shim")
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int64

    from sgl_amd_mono.common.ops import rsrc
    from sgl_amd_mono.common.plan import BLOCKS, THREADS

    # compile-only needs no GPU when the target is named; no disk cache to read or fill
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel(name="sgl_amd_mono_resources_test", known_block_size=[THREADS, 1, 1])
    def kernel(out: Int64):
        tid = fx.thread_idx.x
        bo.buffer_store(fx.Int32(tid), rsrc(out), tid)

    @flyc.jit
    def launch(out: Int64, stream: fx.Stream = fx.Stream(None)):
        kernel(out).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    shim._preload_compiled(launch, 0)
    (r,) = resources.check(launch)
    assert r.name == "sgl_amd_mono_resources_test"
    assert r.vgpr > 0 and r.scratch == 0 and r.lds == 0
    assert r.vgpr_spill == 0 and r.sgpr_spill == 0
    (art,) = resources.artifacts(launch)
    import torch

    if torch.cuda.is_available():
        assert resources.resident_per_cu(art) >= 1
