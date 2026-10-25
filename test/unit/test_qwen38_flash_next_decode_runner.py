"""The Qwen3.8-Flash-Next TP2 decode launches' build keys and the per-rank launch
state on CPU (FlyDSL and AITER's FlyDSL helpers importable; nothing compiled but
the o_proj launch ``test_the_guard_s_stop_and_fault_injection_are_compiled_in``
builds)."""

import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter.ops.flydsl.kernels")

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import blocks, front, runner


def test_build_keys_per_width():
    keys = runner.build_keys((1, 2), tp=2, rip=512)
    assert len(keys) == 12 and keys == runner.build_keys((2, 1), tp=2, rip=512)
    one = {k for _, k in keys if k.tokens == 1}
    assert one == {
        blocks.MoeBuild(tokens=1, tp=2, rip=512, kd=384),
        front.FrontBuild(tokens=1, tp=2, self_msq=False, eps=1e-6),
        front.FrontBuild(tokens=1, tp=2, self_msq=True, eps=1e-6),
        blocks.GdnBuild(tokens=1, tp=2, norm_eps=1e-6, shuffled=True),
        blocks.OprojBuild(tokens=1, tp=2, oq=True, og=False, shuffled=True),
        blocks.OprojBuild(tokens=1, tp=2, oq=False, og=True, shuffled=True),
    }
    builds = {type(k): b for b, k in keys}
    assert builds[blocks.MoeBuild] is blocks.build_moe
    assert builds[front.FrontBuild] is front.build
    # each distinct eps is its own front / GDN build
    eps = (1e-6, 1e-5)
    assert len(runner.build_keys((1,), 2, 512, hc_eps=eps, gdn_eps=eps)) == 9
    # the fault injection stalls a wait of the launches that wait on their peers
    faulted = {k for _, k in runner.build_keys((1,), 2, 512, fault=(3, 7))}
    peer_waits = (blocks.OprojBuild, blocks.MoeBuild)
    assert {
        (k.fault_site, k.fault_after) for k in faulted if isinstance(k, peer_waits)
    } == {(3, 7)}
    assert faulted - {k for k in faulted if isinstance(k, peer_waits)} == {
        k for k in one if not isinstance(k, peer_waits)
    }


@pytest.mark.parametrize("rip", [384, 512])
def test_the_moe_launch_reads_w2_to_384_at_either_padding(rip):
    d = runner.moe_key(4, 2, rip).dims()
    assert (d.ri, d.rip, d.kdown) == (320, rip, 384)


@pytest.mark.parametrize("rip, kd", [(512, 256), (512, 448), (384, 512), (336, 0)])
def test_moe_padding_dims_rejects(rip, kd):
    with pytest.raises(ValueError, match="MoE padding"):
        blocks.MoeBuild(tokens=4, tp=2, rip=rip, kd=kd).dims()


class _Peers:
    def __init__(self, nbytes, group, rank, npes, device):
        self.args = (nbytes, group, rank, npes)
        self.bytes = torch.ones(nbytes, dtype=torch.uint8)
        self.addresses = torch.tensor([1 << 32, 2 << 32], dtype=torch.int64)


def _deadline(epoch) -> int:
    lo, hi = epoch[guard.GUARD_DEADLINE : guard.GUARD_DEADLINE + 2].tolist()
    return (hi << 32) | (lo & 0xFFFFFFFF)


def test_launcher_state(monkeypatch):
    monkeypatch.delenv("SGLANG_AMD_MONO_FAULT_INJECT", raising=False)
    la = runner.Launcher((1, 16), 1, 2, "group", torch.device("cpu"), 512, 75.0, _Peers)
    assert la.peers.args == (runner.peer_bytes(2), "group", 1, 2)
    assert not la.peers.bytes.any()
    flags = guard.FLAG_BYTES
    assert la.peer_table.tolist() == [(1 << 32) + flags, (2 << 32) + flags]
    # the guard's flags are each peer buffer's head, past which the regions start
    assert (la.peer_table + blocks.FLAGS).tolist() == la.peers.addresses.tolist()
    assert runner.peer_bytes(2) == flags + blocks.peer_bytes(16, 2)
    assert la.fault == runner.NO_FAULT
    for s in (1, 16):
        attn, mlp = runner.scratch_bytes(s, 2, 512)
        assert la.attn_scratch[s].numel() >= attn and la.mlp_scratch[s].numel() >= mlp
        assert la.attn_scratch[s].numel() % 256 == 0
    assert la.mlp_scratch[16].numel() > la.mlp_scratch[1].numel()
    assert la.epoch.shape == (guard.GUARD_WORDS,)
    # 75 s of the 100 MHz clock: 7.5e9 ticks, a high word of 1 and a low word past int32
    assert _deadline(la.epoch) == guard.deadline_ticks(75.0) == 7_500_000_000
    la.begin_step()
    la.begin_step()
    assert la.epoch[0].item() == 2 and _deadline(la.epoch) == 7_500_000_000
    others = [i for i in range(guard.GUARD_WORDS) if i not in (0, 1, 2)]
    assert not la.epoch[others].any()


def test_launcher_fault_injection_is_this_rank_s(monkeypatch):
    monkeypatch.setenv("SGLANG_AMD_MONO_FAULT_INJECT", "1:3:7")
    cpu = torch.device("cpu")
    assert runner.Launcher((1,), 1, 2, "g", cpu, 512, 60.0, _Peers).fault == (3, 7)
    assert runner.Launcher((1,), 0, 2, "g", cpu, 512, 60.0, _Peers).fault == (0, 0)


PROBE = """
import dataclasses, pathlib, pickle, re, sys
from sgl_amd_mono.qwen38_flash_next.decode.gfx950 import blocks, runner
key = runner.oproj_key(1, 2, True)
if sys.argv[1] == "nostop":
    key = dataclasses.replace(key, stop=False)
elif sys.argv[1] == "fault":
    key = runner.oproj_key(1, 2, True, fault=(1, 0))
runner.compile_build(blocks.build_oproj, key, 0)
from flydsl.utils.env import runtime
for path in pathlib.Path(runtime.cache_dir).glob("*/*.pkl"):
    text = pickle.load(open(path, "rb"))._ir_text
    name = re.search(r'kernel_metadata<"(\\w+)"', text).group(1)
    binary = re.search(r'bin = "(.*?)"', text, re.S).group(1)
    print(name, len(binary), hash(binary))
"""


def _compile(tmp_path, variant: str) -> str:
    """The o_proj launch's binary: "default", "nostop" or "fault"."""
    env = dict(os.environ, ARCH="gfx950", HIP_VISIBLE_DEVICES="", PYTHONHASHSEED="0")
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(tmp_path / f"cache-{variant}")
    out = subprocess.run(
        [sys.executable, "-B", "-c", PROBE, variant],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    return out.stdout.strip()


def test_the_guard_s_stop_and_fault_injection_are_compiled_in(tmp_path):
    """A failed wait stops the launch (``OprojBuild.stop``), and the fault injection
    stalls a wait: each is in the binary."""
    default = _compile(tmp_path, "default")
    assert default.startswith("flashnext_split_oproj_s1_t2")
    for variant in ("nostop", "fault"):
        got = _compile(tmp_path, variant)
        assert got.split()[0] == default.split()[0] and got != default, variant
