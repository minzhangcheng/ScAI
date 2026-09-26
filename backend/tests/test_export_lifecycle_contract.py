"""Export owner, streaming, TTL, and worker-drain contracts."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

import pytest

from backend.services.export_lifecycle import (
    LifecycleFileResponse,
    OwnedArchiveFileResponse,
    claim_export_stream,
    cleanup_export_record,
    create_export_ownership_marker,
)
from backend.state import (
    prepared_exports,
    prepared_exports_lock,
)


def _identity(path: Path) -> tuple[int, int, int, int, int, int]:
    metadata = path.stat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _secure_zip_kwargs():
    return {
        "recursive": True,
        "include": lambda _relative: True,
        "max_files": 100,
        "max_source_bytes": 10 * 1024 * 1024,
        "reserve_bytes": 64 * 1024 * 1024,
    }


def _allocated_stage(directory: Path) -> tuple[int, Path, tuple[int, int]]:
    fd, raw = tempfile.mkstemp(prefix="owned_", suffix=".zip", dir=directory)
    metadata = os.fstat(fd)
    return fd, Path(raw), (int(metadata.st_dev), int(metadata.st_ino))


def _assert_secure_archive_changed(exc_info):
    assert exc_info.value.status_code == 409
    assert "changed" in exc_info.value.detail.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("range_header", "expected_chunks"),
    [
        (b"bytes=2-5", (b"cdef",)),
        (b"bytes=0-1,8-9", (b"ab", b"ij")),
    ],
)
async def test_owned_response_preserves_range_body_and_cleanup_order(
    tmp_path, range_header, expected_chunks,
):
    archive = tmp_path / "range-owned.bin"
    archive.write_bytes(b"abcdefghij")
    metadata = archive.stat()
    file_fd = os.open(archive, os.O_RDONLY)
    response = OwnedArchiveFileResponse(
        archive,
        file_fd=file_fd,
        owned_identity=(metadata.st_dev, metadata.st_ino),
        media_type="application/octet-stream",
    )
    messages = []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    await response(
        {
            "type": "http",
            "method": "GET",
            "headers": [(b"range", range_header)],
            "extensions": {},
        },
        receive,
        send,
    )

    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 206
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    assert all(chunk in body for chunk in expected_chunks)
    assert not archive.exists()
    with pytest.raises(OSError):
        os.fstat(file_fd)


def test_secure_archive_fails_closed_without_descriptor_primitives(
    monkeypatch, tmp_path,
):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("input", encoding="utf-8")
    fd, destination, _owned = _allocated_stage(tmp_path)
    monkeypatch.setattr(lifecycle.os, "supports_dir_fd", set())
    try:
        with pytest.raises(lifecycle.SecureArchiveError) as exc_info:
            lifecycle.build_secure_zip(
                source,
                destination,
                destination_fd=fd,
                **_secure_zip_kwargs(),
            )

        assert exc_info.value.status_code == 501
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


@pytest.mark.parametrize("swapped_entry", ["leaf", "parent"])
def test_secure_archive_rejects_selected_path_swap(monkeypatch, tmp_path, swapped_entry):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    parent = source if swapped_entry == "leaf" else source / "nested"
    parent.mkdir(parents=True)
    target = parent / "leaf.bin"
    target.write_bytes(b"old bytes")
    fd, destination, _owned = _allocated_stage(tmp_path)
    expected = _identity(target if swapped_entry == "leaf" else parent)
    matches_type = stat.S_ISREG if swapped_entry == "leaf" else stat.S_ISDIR
    original_fstat = lifecycle.os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if (
            not swapped
            and matches_type(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == expected[:2]
        ):
            if swapped_entry == "leaf":
                replacement = source / "replacement.bin"
                replacement.write_bytes(b"replacement bytes")
                replacement.replace(target)
            else:
                parent.rename(source / "nested-old")
                parent.mkdir()
                target.write_bytes(b"replacement bytes")
            swapped = True
        return metadata

    monkeypatch.setattr(lifecycle.os, "fstat", fstat)
    try:
        with pytest.raises(lifecycle.SecureArchiveError) as exc_info:
            lifecycle.build_secure_zip(
                source,
                destination,
                destination_fd=fd,
                **_secure_zip_kwargs(),
            )

        _assert_secure_archive_changed(exc_info)
        assert swapped
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


def test_secure_archive_rejects_midstream_identity_change(monkeypatch, tmp_path):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    target = source / "stream.bin"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    fd, destination, _owned = _allocated_stage(tmp_path)
    expected = _identity(target)
    original_fstat = lifecycle.os.fstat
    original_read = lifecycle.os.read
    target_fd = None
    mutated = False

    def fstat(fd):
        nonlocal target_fd
        metadata = original_fstat(fd)
        if stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == expected[:2]:
            target_fd = fd
        return metadata

    def read(fd, size):
        nonlocal mutated
        block = original_read(fd, size)
        if fd == target_fd and block and not mutated:
            with target.open("ab") as handle:
                handle.write(b"mutation")
            mutated = True
        return block

    monkeypatch.setattr(lifecycle.os, "fstat", fstat)
    monkeypatch.setattr(lifecycle.os, "read", read)
    try:
        with pytest.raises(lifecycle.SecureArchiveError) as exc_info:
            lifecycle.build_secure_zip(
                source,
                destination,
                destination_fd=fd,
                **_secure_zip_kwargs(),
            )

        _assert_secure_archive_changed(exc_info)
        assert mutated
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


@pytest.mark.parametrize("operation", ["zip", "copy"])
def test_secure_export_never_writes_or_unlinks_a_rebound_stage_inode(
    monkeypatch, tmp_path, operation,
):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    source_file = source / ("input.txt" if operation == "zip" else "archive.zip")
    source_file.write_bytes(b"source bytes" if operation == "zip" else b"source archive")
    fd, destination, owned = _allocated_stage(tmp_path)
    canary = tmp_path / "foreign-canary"
    canary.write_bytes(b"foreign bytes")
    rebound = False
    original_usage = shutil.disk_usage

    def disk_usage(path):
        nonlocal rebound
        if not rebound:
            destination.unlink()
            os.link(canary, destination)
            rebound = True
        return original_usage(path)

    monkeypatch.setattr(lifecycle.shutil, "disk_usage", disk_usage)
    try:
        with pytest.raises(lifecycle.SecureArchiveError) as exc_info:
            if operation == "zip":
                lifecycle.build_secure_zip(
                    source, destination, destination_fd=fd, **_secure_zip_kwargs(),
                )
            else:
                lifecycle.copy_secure_file(
                    source, Path("archive.zip"), destination, destination_fd=fd,
                    expected_identity=_identity(source_file), reserve_bytes=0,
                )
        _assert_secure_archive_changed(exc_info)
        assert rebound
        assert destination.read_bytes() == b"foreign bytes"
        assert canary.read_bytes() == b"foreign bytes"
        assert (destination.stat().st_dev, destination.stat().st_ino) != owned
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)
        canary.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_stream_disconnect_finally_unlinks_before_token_retirement(
    tmp_path,
):
    session_id = "export-stream-disconnect"
    token = "stream-disconnect-token"
    archive = tmp_path / "export_stream_disconnect.tar.gz"
    archive.write_bytes(b"payload")
    info = {
        "token": token,
        "path": str(archive),
        "path_owned": True,
        "filename": "session.tar.gz",
        "size": archive.stat().st_size,
        "session_id": session_id,
        "user_id": "owner",
        "phase": "ready",
        "task": None,
        "created_at": time.time(),
    }
    async with prepared_exports_lock:
        prepared_exports[token] = info

    body_started = asyncio.Event()
    hold_send = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            body_started.set()
            await hold_send.wait()

    async def receive():
        return {"type": "http.request"}

    async def stream_owner():
        claimed = await claim_export_stream(token, session_id)
        assert claimed is info
        response = LifecycleFileResponse(
            archive,
            media_type="application/gzip",
            filename="session.tar.gz",
            export_info=claimed,
        )
        await response(
            {
                "type": "http",
                "method": "GET",
                "headers": [],
                "extensions": {},
            },
            receive,
            send,
        )

    stream = asyncio.create_task(stream_owner())
    await asyncio.wait_for(body_started.wait(), timeout=1)
    async with prepared_exports_lock:
        assert prepared_exports[token]["phase"] == "streaming"
        assert prepared_exports[token]["task"] is stream

    stream.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stream, timeout=2)

    assert not archive.exists()
    async with prepared_exports_lock:
        assert token not in prepared_exports


@pytest.mark.asyncio
async def test_unowned_output_conflict_is_never_unlinked(monkeypatch, tmp_path):
    from backend.tasks import session_cleanup

    token = "unowned-output-conflict"
    existing = tmp_path / "v2-export-existing.tar.gz"
    existing.write_bytes(b"unrelated")
    monkeypatch.setattr(session_cleanup, "UPLOAD_TEMP_DIR", tmp_path)
    info = {
        "token": token,
        "path": str(existing),
        "path_owned": False,
        "session_id": "export-output-conflict",
        "phase": "building",
        "task": asyncio.current_task(),
        "created_at": time.time(),
    }
    async with prepared_exports_lock:
        prepared_exports[token] = info

    assert await cleanup_export_record(info) is True
    assert existing.read_bytes() == b"unrelated"
    async with prepared_exports_lock:
        assert token not in prepared_exports
    assert await session_cleanup.sweep_orphaned_export_archives() == 0
    assert existing.read_bytes() == b"unrelated"


@pytest.mark.asyncio
async def test_stale_cleanup_retains_failed_record_and_skips_live_builder(
    monkeypatch, tmp_path,
):
    from backend.config import PREPARED_EXPORT_TTL
    from backend.tasks.session_cleanup import _cleanup_stale_prepared_exports

    stale_path = tmp_path / "prepared_export_stale.tar.gz"
    live_path = tmp_path / "prepared_export_live.tar.gz"
    stale_path.write_bytes(b"stale")
    live_path.write_bytes(b"live")
    current = asyncio.current_task()
    exports = {
        "stale": {
            "token": "stale",
            "path": str(stale_path),
            "phase": "ready",
            "created_at": 0.0,
        },
        "live": {
            "token": "live",
            "path": str(live_path),
            "phase": "building",
            "task": current,
            "created_at": 0.0,
        },
    }
    lock = asyncio.Lock()
    original_unlink = type(stale_path).unlink
    fail_unlink = True

    def controlled_unlink(path, *args, **kwargs):
        if fail_unlink and path == stale_path:
            raise OSError("busy")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(stale_path), "unlink", controlled_unlink)
    now = PREPARED_EXPORT_TTL + 10.0

    await _cleanup_stale_prepared_exports(now, exports, lock)
    assert set(exports) == {"stale", "live"}
    assert exports["stale"]["phase"] == "cleanup_failed"
    assert stale_path.exists() and live_path.exists()

    fail_unlink = False
    await _cleanup_stale_prepared_exports(now + 1, exports, lock)
    assert set(exports) == {"live"}
    assert not stale_path.exists()
    assert live_path.exists()


def test_orphan_sweep_preserves_live_generation_and_near_prefix(
    monkeypatch, tmp_path,
):
    from backend.tasks import session_cleanup

    old_generation = tmp_path / "prepared_export_s.g1.old.tar.gz"
    live_generation = tmp_path / "prepared_export_s.g2.live.tar.gz"
    near_prefix = tmp_path / "xprepared_export_s.g1.keep.tar.gz"
    unowned_lookalike = tmp_path / "prepared_export_s.g3.unowned.tar.gz"
    for path in (
        old_generation, live_generation, near_prefix, unowned_lookalike,
    ):
        path.write_bytes(b"x")
    create_export_ownership_marker(old_generation)
    create_export_ownership_marker(live_generation)
    monkeypatch.setattr(session_cleanup, "UPLOAD_TEMP_DIR", tmp_path)

    removed = session_cleanup._unlink_orphaned_export_archives(
        None,
        frozenset({str(live_generation)}),
    )

    assert removed == 1
    assert not old_generation.exists()
    assert live_generation.exists()
    assert near_prefix.exists()
    assert unowned_lookalike.exists()
