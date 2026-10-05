"""Atomic same-key access and bounded Windows sharing retries."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace as NS

import pytest
from app.adapters import storage_local
from app.adapters.storage_local import LocalFileStore, _replace_atomically
from app.domain.errors import DomainError, NotFound


def test_same_key_calls_across_store_instances_are_serialized(tmp_path, monkeypatch):
    first, second = LocalFileStore(tmp_path), LocalFileStore(tmp_path)
    first.put("objects/shared", b"original")
    inside, attempted, release = Event(), Event(), Event()
    original_path = first._path

    def held_path(key):
        inside.set()
        assert release.wait(5)
        return original_path(key)

    monkeypatch.setattr(first, "_path", held_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(first.read, "objects/shared")
        assert inside.wait(5)

        def write():
            attempted.set()
            return second.put("objects/shared", b"replacement")

        writer = pool.submit(write)
        assert attempted.wait(5)
        # Main thread can observe ownership without timing a scheduler delay.
        lock = second._lock("objects/shared")
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        release.set()
        assert not acquired
        assert reader.result(timeout=5) == b"original"
        assert writer.result(timeout=5)
    assert second.read("objects/shared") == b"replacement"
    assert not list(tmp_path.rglob(".upload-*"))


def test_concurrent_replacements_and_reads_are_complete(tmp_path):
    stores = (LocalFileStore(tmp_path), LocalFileStore(tmp_path))
    values = (b"a" * 8192, b"b" * 16384)
    stores[0].put("documents/shared", values[0])
    barrier = Barrier(8)

    def access(index):
        barrier.wait(timeout=5)
        store = stores[index % 2]
        for _ in range(25):
            if index % 2:
                store.put("documents/shared", values[index % 2])
            else:
                assert store.read("documents/shared") in values

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(access, range(8)))
    assert not list(tmp_path.rglob(".upload-*"))


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_windows_sharing_errors_retry_atomic_replace_only(monkeypatch, winerror):
    calls, delays = [], []

    def replace(source, target):
        calls.append((source, target))
        if len(calls) < 3:
            error = PermissionError("Temporary sharing conflict")
            error.winerror = winerror
            raise error

    monkeypatch.setattr(storage_local, "os", NS(name="nt", replace=replace))
    monkeypatch.setattr(storage_local.time, "sleep", delays.append)
    _replace_atomically("temporary", Path("destination"))
    assert len(calls) == 3 and delays == [0.01, 0.02]


@pytest.mark.parametrize(
    "platform,winerror,calls_expected", [("nt", 5, 5), ("nt", 87, 1), ("posix", 5, 1)]
)
def test_retry_is_bounded_and_does_not_hide_other_failures(
    monkeypatch, platform, winerror, calls_expected
):
    calls, delays = [], []
    error = PermissionError("Cannot replace")
    error.winerror = winerror

    def replace(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(storage_local, "os", NS(name=platform, replace=replace))
    monkeypatch.setattr(storage_local.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as failure:
        _replace_atomically("temporary", Path("destination"))
    assert failure.value is error and len(calls) == calls_expected
    assert len(delays) == calls_expected - 1


def test_failed_replace_preserves_object_and_cleans_temporary(tmp_path, monkeypatch):
    store = LocalFileStore(tmp_path)
    store.put("object", b"existing")

    def fail(*args):
        raise PermissionError("Permanent access failure")

    monkeypatch.setattr(storage_local, "_replace_atomically", fail)
    with pytest.raises(PermissionError):
        store.put("object", b"new")
    assert store.read("object") == b"existing"
    assert not list(tmp_path.glob(".upload-*"))


@pytest.mark.parametrize("key", ["../outside", "/absolute", "folder//file", "C:/outside"])
def test_concurrency_guard_preserves_key_rejection(tmp_path, key):
    store = LocalFileStore(tmp_path)
    for operation in (
        lambda: store.put(key, b"x"),
        lambda: store.read(key),
        lambda: store.delete(key),
    ):
        with pytest.raises(DomainError):
            operation()


def test_read_rejects_absent_and_oversized_stored_objects(tmp_path):
    store = LocalFileStore(tmp_path, max_bytes=8)
    with pytest.raises(NotFound):
        store.read("absent")
    (tmp_path / "oversized").write_bytes(b"x" * 9)
    with pytest.raises(DomainError, match="exceeds"):
        store.read("oversized")


def test_symlink_guard_is_preserved_without_windows_symlink_privilege(tmp_path, monkeypatch):
    store = LocalFileStore(tmp_path)
    candidate = tmp_path / "link" / "object"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == candidate or original(path))
    with pytest.raises(DomainError, match="Symlink"):
        store.put("link/object", b"payload")
    assert not candidate.exists()
