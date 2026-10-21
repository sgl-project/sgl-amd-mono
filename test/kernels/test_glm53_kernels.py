"""Synthetic GLM-5.3 layer equivalence through the plugin's actual runner.

No checkpoint is needed. Serving/reference accuracy lives in test/e2e.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

CHECK = Path(__file__).with_name("glm53") / "check_indexer_cp.py"
MULTITOKEN_CHECK = CHECK.with_name("check_multitoken.py")


def _gfx950_gpus():
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


@pytest.mark.parametrize(
    "tp,rows,capacity,ties",
    [
        (4, 2, 64, False),
        (4, 4, 4096, False),
        (4, 4, 4096, True),
        (4, 2, 16448, False),
        (4, 2, 114752, False),
        (4, 4, 131072, False),
        (4, 2, 131072, False),
        (4, 2, 131072, True),
        (4, 2, 1048576, False),
        (8, 8, 4096, False),
        (8, 8, 131072, False),
        (8, 2, 131072, False),
    ],
)
def test_indexer_cp(tp, rows, capacity, ties):
    _check(tp, rows, capacity, ties)


@pytest.mark.parametrize("rows", [2, 8])
def test_indexer_cp_full_context(rows):
    _check(8, rows, 131072, False, pad_rows=0, full_context=True)


@pytest.mark.parametrize(
    "rows,capacity", [(2, 65536), (4, 131072), (4, 262144), (4, 1048576)]
)
def test_indexer_cp_tp4_full_context(rows, capacity):
    _check(4, rows, capacity, False, pad_rows=0, full_context=True)


def test_indexer_cp_tp4_large_cache(tmp_path):
    control = str(tmp_path / "cache_translation")
    _check(
        4,
        2,
        4096,
        False,
        pad_rows=0,
        full_context=True,
        extra_args=("--save-control", control, "--exact-control"),
    )
    _check(
        4,
        2,
        4096,
        False,
        pad_rows=0,
        full_context=True,
        extra_args=(
            "--check-control",
            control,
            "--exact-control",
            "--slot-offset",
            "4194304",
        ),
    )


@pytest.mark.parametrize("query", [4, 6, 8])
def test_multitoken_tp4(tmp_path, query):
    if _gfx950_gpus() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    control = str(tmp_path / "serial")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(MULTITOKEN_CHECK),
        "--query-tokens",
        str(query),
    ]
    if query == 8:
        command.append("--check-routing")
    try:
        subprocess.run(
            command + ["--schedule", "serial", "--save-control", control],
            check=True,
            timeout=1800,
        )
        for cp in ([], ["--indexer-cp"]):
            subprocess.run(
                command + ["--check-control", control, "--check-causality"] + cp,
                check=True,
                timeout=1800,
            )
    finally:
        for path in tmp_path.glob("serial.rank*.pt"):
            path.unlink()


@pytest.mark.parametrize("external,ties", [(True, False), (False, True)])
def test_multitoken_reuse_and_routes_tp4(tmp_path, external, ties):
    if _gfx950_gpus() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    indexed = str(tmp_path / "indexed")
    control = str(tmp_path / "serial")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(MULTITOKEN_CHECK),
        "--query-tokens",
        "8",
        "--check-routing",
        "--check-causality",
    ]
    if ties:
        command.append("--route-ties")

    def run(args):
        subprocess.run(command + args, check=True, timeout=1800)

    try:
        if external:
            run(["--schedule", "serial", "--save-control", indexed])
            command += ["--external-control", indexed]
        run(["--schedule", "serial", "--save-control", control])
        run(["--check-control", control] + ([] if external else ["--indexer-cp"]))
    finally:
        for path in tmp_path.glob("*.rank*.pt"):
            path.unlink()


@pytest.mark.parametrize(
    "capacity,pad_rows,ties", [(64, 3, False), (4096, 0, True), (131072, 0, False)]
)
def test_multitoken_cp_boundaries_tp4(capacity, pad_rows, ties):
    _check(
        4,
        16,
        capacity,
        ties,
        pad_rows=pad_rows,
        full_context=True,
        extra_args=("--request-width", "8", "--layers", "1", "--replays", "4"),
    )


def _check(tp, rows, capacity, ties, pad_rows=1, full_context=False, extra_args=()):
    if _gfx950_gpus() < tp:
        pytest.skip(f"needs {tp} gfx950 GPUs")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={tp}",
        str(CHECK),
        "--rows",
        str(rows),
        "--capacity",
        str(capacity),
        "--pad-rows",
        str(pad_rows),
    ]
    if full_context:
        cmd.append("--full-context")
    if ties:
        cmd.append("--ties")
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True, timeout=1800)


@pytest.mark.parametrize("rows", range(1, 17))
def test_a4w4_tp4(rows):
    _check_a4w4(["--rows", str(rows)])


@pytest.mark.parametrize("requests,query", [(1, 13), (2, 7), (3, 5), (2, 8)])
def test_a4w4_tiled_rows_match_serial_tp4(tmp_path, requests, query):
    if _gfx950_gpus() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    control = str(tmp_path / "serial")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(MULTITOKEN_CHECK),
        "--activation",
        "fp4",
        "--requests",
        str(requests),
        "--query-tokens",
        str(query),
        "--capacity",
        "2048",
        "--check-routing",
    ]
    try:
        subprocess.run(
            command + ["--schedule", "serial", "--save-control", control],
            check=True,
            timeout=1800,
        )
        subprocess.run(
            command + ["--check-control", control, "--check-causality"],
            check=True,
            timeout=1800,
        )
    finally:
        for path in tmp_path.glob("serial.rank*.pt"):
            path.unlink()


@pytest.mark.parametrize("rows", [12, 16])
def test_a4w4_external_topk_tp4(rows):
    _check_a4w4(["--rows", str(rows), "--external"])


def test_a4w4_quantization_edges_tp4():
    _check_a4w4(["--rows", "4", "--quant-edges"])


def test_a4w4_long_context_cp_tp4(tmp_path):
    control = str(tmp_path / "a4_cp")
    args = [
        "--rows",
        "16",
        "--capacity",
        "131072",
        "--pad-rows",
        "3",
        "--ties",
        "--seed",
        "823",
    ]
    try:
        _check_a4w4(args + ["--save-control", control])
        _check_a4w4(args + ["--cp", "--check-control", control, "--profile-launches"])
    finally:
        for path in tmp_path.glob("a4_cp.rank*.pt"):
            path.unlink()


def _check_a4w4(args):
    if _gfx950_gpus() < 4:
        pytest.skip("needs 4 gfx950 GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(CHECK.with_name("check_mxfp4.py")),
            *args,
        ],
        check=True,
        timeout=1800,
    )
