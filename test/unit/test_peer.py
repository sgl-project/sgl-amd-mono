"""Peer teardown must retain mappings and allocations until collective success."""

import pytest

from sgl_amd_mono.common import peer


def buffer(monkeypatch, responses=()):
    calls = []
    obj = peer.PeerBuffer.__new__(peer.PeerBuffer)
    obj.rank, obj.npes, obj.group, obj.device = 0, 4, object(), "cuda:0"
    obj.local, obj.bytes, obj._opened = 100, object(), [200, 300, 400]
    obj._closed = obj._quiesced = False
    responses = iter(responses)
    monkeypatch.setattr(peer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        peer.torch.cuda, "synchronize", lambda device: calls.append("sync")
    )
    monkeypatch.setattr(peer.dist, "barrier", lambda **kw: calls.append("barrier"))

    def gather(out, error, group):
        calls.append(("agree", error))
        out[:] = [error, next(responses, None), None, None]

    monkeypatch.setattr(peer.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        peer, "_close_mapping", lambda base: calls.append(("unmap", base))
    )
    monkeypatch.setattr(
        peer, "_free_allocation", lambda base: calls.append(("free", base))
    )
    return obj, calls


def test_collective_close_is_idempotent(monkeypatch):
    obj, calls = buffer(monkeypatch)
    obj.close()
    assert calls == [
        "sync",
        "barrier",
        ("unmap", 200),
        ("unmap", 300),
        ("unmap", 400),
        ("agree", None),
        ("free", 100),
        ("agree", None),
    ]
    obj.close()
    assert sum(item == ("free", 100) for item in calls) == 1
    assert obj.local == 0 and obj.bytes is None and obj._closed


def test_failed_mapping_retained_for_retry(monkeypatch):
    obj, calls = buffer(monkeypatch)

    def unmap(base):
        calls.append(("unmap", base))
        if base == 300:
            raise RuntimeError("injected unmap failure")

    monkeypatch.setattr(peer, "_close_mapping", unmap)
    with pytest.raises(peer.PeerMemoryError, match="injected unmap failure"):
        obj.close()
    assert obj._opened == [300] and obj.local == 100 and obj.bytes is not None
    assert not any(item == ("free", 100) for item in calls)
    monkeypatch.setattr(
        peer, "_close_mapping", lambda base: calls.append(("retry", base))
    )
    obj.close()
    assert calls.count("barrier") == 1
    assert ("retry", 300) in calls and obj._closed


def test_remote_failure_keeps_successful_rank_in_retry(monkeypatch):
    obj, calls = buffer(monkeypatch, responses=["peer unmap failed", None, None])
    with pytest.raises(peer.PeerMemoryError, match="peer unmap failed"):
        obj.close()
    assert not obj._opened and obj.local == 100
    obj.close()
    assert calls.count(("free", 100)) == 1 and obj._closed


def test_remote_free_failure_retries_without_double_free(monkeypatch):
    obj, calls = buffer(monkeypatch, responses=[None, "peer free failed", None, None])
    with pytest.raises(peer.PeerMemoryError, match="peer free failed"):
        obj.close()
    assert obj.local == 0 and not obj._closed
    obj.close()
    assert calls.count(("free", 100)) == 1 and obj._closed


def test_local_free_failure_retains_owner(monkeypatch):
    obj, calls = buffer(monkeypatch)

    def fail(base):
        raise RuntimeError("free failed")

    monkeypatch.setattr(peer, "_free_allocation", fail)
    with pytest.raises(peer.PeerMemoryError, match="free failed"):
        obj.close()
    assert obj.local == 100 and obj.bytes is not None
    monkeypatch.setattr(
        peer, "_free_allocation", lambda base: calls.append(("free", base))
    )
    obj.close()
    assert obj._closed


def test_close_requires_live_group(monkeypatch):
    obj, calls = buffer(monkeypatch)
    monkeypatch.setattr(peer.dist, "is_initialized", lambda: False)
    with pytest.raises(peer.PeerMemoryError, match="process group"):
        obj.close()
    assert not calls and obj.local == 100
