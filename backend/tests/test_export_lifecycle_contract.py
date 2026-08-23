"""Export owner, streaming, TTL, and worker-drain contracts."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from backend.services.export_lifecycle import (
    LifecycleFileResponse,
    claim_export_stream,
    cleanup_export_record,
    create_export_ownership_marker,
    export_ownership_marker_path,
    unlink_export_archive_and_marker,
)
from backend.state import (
    get_session_lock,
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


def test_secure_archive_skips_symlink_present_at_selection_time(tmp_path):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (source / "linked.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available on this platform")
    fd, destination, _owned = _allocated_stage(tmp_path)
    try:
        lifecycle.build_secure_zip(
            source,
            destination,
            destination_fd=fd,
            **_secure_zip_kwargs(),
        )

        with zipfile.ZipFile(destination) as archive:
            assert archive.namelist() == ["safe.txt"]
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


def test_secure_archive_rejects_selected_leaf_swap(monkeypatch, tmp_path):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    target = source / "leaf.bin"
    target.write_bytes(b"old bytes")
    fd, destination, _owned = _allocated_stage(tmp_path)
    expected = _identity(target)
    original_fstat = lifecycle.os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if (
            not swapped
            and stat.S_ISREG(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == expected[:2]
        ):
            replacement = source / "replacement.bin"
            replacement.write_bytes(b"replacement bytes")
            replacement.replace(target)
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


def test_secure_archive_rejects_parent_directory_swap(monkeypatch, tmp_path):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    parent = source / "nested"
    parent.mkdir(parents=True)
    target = parent / "leaf.bin"
    target.write_bytes(b"old bytes")
    fd, destination, _owned = _allocated_stage(tmp_path)
    expected = _identity(parent)
    original_fstat = lifecycle.os.fstat
    swapped = False

    def fstat(fd):
        nonlocal swapped
        metadata = original_fstat(fd)
        if (
            not swapped
            and stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == expected[:2]
        ):
            old_parent = source / "nested-old"
            parent.rename(old_parent)
            parent.mkdir()
            (parent / "leaf.bin").write_bytes(b"replacement bytes")
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


def test_secure_archive_reserve_failure_removes_empty_stage(monkeypatch, tmp_path):
    from backend.services import export_lifecycle as lifecycle
    import shutil

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("input", encoding="utf-8")
    fd, destination, _owned = _allocated_stage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0, total=0, used=0),
    )

    try:
        with pytest.raises(lifecycle.SecureArchiveError) as exc_info:
            lifecycle.build_secure_zip(
                source,
                destination,
                destination_fd=fd,
                **_secure_zip_kwargs(),
            )

        assert exc_info.value.status_code == 507
        assert not destination.exists()
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


def test_secure_direct_copy_uses_expected_identity_and_removes_rejected_stage(
    tmp_path,
):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "archive.zip"
    source_file.write_bytes(b"archive bytes")
    fd, destination, _owned = _allocated_stage(tmp_path)
    try:
        lifecycle.copy_secure_file(
            source,
            Path("archive.zip"),
            destination,
            destination_fd=fd,
            expected_identity=_identity(source_file),
            reserve_bytes=0,
        )

        assert destination.read_bytes() == b"archive bytes"
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)


def test_secure_zip_never_writes_or_unlinks_a_rebound_stage_inode(
    monkeypatch, tmp_path,
):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_bytes(b"source bytes")
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
            lifecycle.build_secure_zip(
                source,
                destination,
                destination_fd=fd,
                **_secure_zip_kwargs(),
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


def test_secure_copy_never_writes_or_unlinks_a_rebound_stage_inode(
    monkeypatch, tmp_path,
):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "archive.zip"
    source_file.write_bytes(b"source archive")
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
            lifecycle.copy_secure_file(
                source,
                Path("archive.zip"),
                destination,
                destination_fd=fd,
                expected_identity=_identity(source_file),
                reserve_bytes=0,
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


def test_missing_descriptor_capability_removes_a_preallocated_owned_stage(
    monkeypatch, tmp_path,
):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_bytes(b"input")
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


def test_secure_zip_preserves_member_mtime_mode_and_zip64(monkeypatch, tmp_path):
    from backend.services import export_lifecycle as lifecycle

    source = tmp_path / "source"
    source.mkdir()
    member = source / "member.bin"
    member.write_bytes(b"zip64 metadata bytes")
    fixed_time = time.mktime((2024, 6, 1, 12, 34, 56, 0, 0, -1))
    os.utime(member, (fixed_time, fixed_time))
    os.chmod(member, 0o751)
    fd, destination, _owned = _allocated_stage(tmp_path)
    monkeypatch.setattr(lifecycle.zipfile, "ZIP64_LIMIT", 1)
    try:
        lifecycle.build_secure_zip(
            source,
            destination,
            destination_fd=fd,
            **_secure_zip_kwargs(),
        )
        with zipfile.ZipFile(destination) as archive:
            info = archive.getinfo("member.bin")
            assert info.date_time == time.localtime(fixed_time)[:6]
            assert info.external_attr >> 16 == stat.S_IFREG | 0o751
            assert archive.read("member.bin") == b"zip64 metadata bytes"
    finally:
        os.close(fd)
        destination.unlink(missing_ok=True)



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


def test_ownership_marker_collision_never_removes_existing_marker(tmp_path):
    archive = tmp_path / "v2-export-marker-conflict.tar.gz"
    archive.write_bytes(b"archive")
    marker = export_ownership_marker_path(archive)
    marker.write_bytes(b"unrelated-marker")
    marker_inode = marker.stat().st_ino

    with pytest.raises(FileExistsError):
        create_export_ownership_marker(archive)

    assert marker.read_bytes() == b"unrelated-marker"
    assert marker.stat().st_ino == marker_inode


def test_legacy_unlink_removes_owned_marker_when_archive_path_is_missing(tmp_path):
    archive = tmp_path / "prepared_export_missing.tar.gz"
    archive.write_bytes(b"owned")
    marker = create_export_ownership_marker(archive)
    archive.unlink()

    unlink_export_archive_and_marker(archive, remove_marker=True)

    assert not archive.exists()
    assert not marker.exists()


def test_identity_unlink_preserves_rebound_archive_and_removes_owned_marker(tmp_path):
    archive = tmp_path / "prepared_export_rebound.tar.gz"
    archive.write_bytes(b"owned")
    owned_metadata = archive.stat()
    owned_identity = (int(owned_metadata.st_dev), int(owned_metadata.st_ino))
    marker = create_export_ownership_marker(
        archive,
        expected_identity=owned_identity,
    )
    archive.unlink()
    archive.write_bytes(b"foreign")

    unlink_export_archive_and_marker(
        archive,
        remove_marker=True,
        expected_identity=owned_identity,
    )

    assert archive.read_bytes() == b"foreign"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_direct_cleanup_rejects_archive_inode_replacement(tmp_path):
    token = "direct-inode-replacement"
    archive = tmp_path / "prepared_export_inode.tar.gz"
    archive.write_bytes(b"owned")
    create_export_ownership_marker(archive)
    info = {
        "token": token,
        "path": str(archive),
        "path_owned": True,
        "ownership_marker_required": True,
        "ownership_marker_owned": True,
        "phase": "ready",
    }
    async with prepared_exports_lock:
        prepared_exports[token] = info
    archive.unlink()
    archive.write_bytes(b"replacement")

    assert await cleanup_export_record(info) is False
    assert archive.read_bytes() == b"replacement"
    async with prepared_exports_lock:
        assert prepared_exports[token] is info
        assert info["phase"] == "cleanup_failed"
        prepared_exports.pop(token, None)


@pytest.mark.asyncio
async def test_direct_cleanup_rejects_archive_non_regular_replacement(tmp_path):
    token = "direct-non-regular-replacement"
    archive = tmp_path / "prepared_export_non_regular.tar.gz"
    archive.write_bytes(b"owned")
    marker = create_export_ownership_marker(archive)
    info = {
        "token": token,
        "path": str(archive),
        "path_owned": True,
        "ownership_marker_required": True,
        "ownership_marker_owned": True,
        "phase": "ready",
    }
    async with prepared_exports_lock:
        prepared_exports[token] = info
    archive.unlink()
    archive.mkdir()

    assert await cleanup_export_record(info) is False
    assert archive.is_dir()
    assert marker.exists()
    async with prepared_exports_lock:
        assert prepared_exports[token] is info
        assert info["phase"] == "cleanup_failed"
        prepared_exports.pop(token, None)


@pytest.mark.asyncio
async def test_stale_cleanup_retires_unowned_record_without_unlinking(
    monkeypatch, tmp_path,
):
    from backend.config import PREPARED_EXPORT_TTL
    from backend.tasks import session_cleanup

    existing = tmp_path / "v2-export-existing.tar.gz"
    existing.write_bytes(b"unrelated")
    monkeypatch.setattr(session_cleanup, "UPLOAD_TEMP_DIR", tmp_path)
    exports = {
        "conflict": {
            "token": "conflict",
            "path": str(existing),
            "path_owned": False,
            "phase": "cleanup_failed",
            "created_at": 0.0,
        }
    }

    await session_cleanup._cleanup_stale_prepared_exports(
        PREPARED_EXPORT_TTL + 10.0,
        exports,
        asyncio.Lock(),
    )

    assert exports == {}
    assert existing.read_bytes() == b"unrelated"
    assert await session_cleanup.sweep_orphaned_export_archives() == 0
    assert existing.read_bytes() == b"unrelated"


@pytest.mark.asyncio
async def test_stale_cleanup_rejects_archive_inode_replacement(tmp_path):
    from backend.config import PREPARED_EXPORT_TTL
    from backend.tasks.session_cleanup import _cleanup_stale_prepared_exports

    archive = tmp_path / "prepared_export_ttl_inode.tar.gz"
    archive.write_bytes(b"owned")
    create_export_ownership_marker(archive)
    info = {
        "token": "ttl-inode-replacement",
        "path": str(archive),
        "path_owned": True,
        "ownership_marker_required": True,
        "ownership_marker_owned": True,
        "phase": "ready",
        "created_at": 0.0,
    }
    exports = {info["token"]: info}
    archive.unlink()
    archive.write_bytes(b"replacement")

    await _cleanup_stale_prepared_exports(
        PREPARED_EXPORT_TTL + 10.0,
        exports,
        asyncio.Lock(),
    )

    assert exports == {info["token"]: info}
    assert info["phase"] == "cleanup_failed"
    assert archive.read_bytes() == b"replacement"


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


def test_orphan_sweep_migrates_legacy_unmarked_archives_safely(
    monkeypatch, tmp_path,
):
    """Pre-marker archives are reclaimed without following unsafe paths."""
    from backend.tasks import session_cleanup

    old_legacy = tmp_path / "prepared_export_abcdefgh.tar.gz"
    old_interrupted = tmp_path / "export_stream_session-1.g2.ijklmnop.tar.gz"
    old_snapshot = (
        tmp_path / "v2-export-exp-0123456789abcdef-qrstuvwx.tar.gz"
    )
    young_legacy = tmp_path / "prepared_export_qrstuvwx.tar.gz"
    protected_legacy = tmp_path / "export_stream_yz012345.tar.gz"
    invalid_shape = tmp_path / "prepared_export_not-a-temp-name.tar.gz"
    outside = tmp_path / "outside.tar.gz"
    unsafe_link = tmp_path / "prepared_export_87654321.tar.gz"
    for path in (
        old_legacy,
        old_interrupted,
        old_snapshot,
        young_legacy,
        protected_legacy,
        invalid_shape,
        outside,
    ):
        path.write_bytes(b"archive")
    unsafe_link.symlink_to(outside)

    now = time.time()
    old_timestamp = now - 1000
    for path in (
        old_legacy,
        old_interrupted,
        old_snapshot,
        protected_legacy,
        invalid_shape,
    ):
        os.utime(path, (old_timestamp, old_timestamp))

    monkeypatch.setattr(session_cleanup, "UPLOAD_TEMP_DIR", tmp_path)
    removed = session_cleanup._unlink_orphaned_export_archives(
        100,
        frozenset({str(protected_legacy)}),
    )

    assert removed == 3
    assert not old_legacy.exists()
    assert not old_interrupted.exists()
    assert not old_snapshot.exists()
    assert young_legacy.exists()
    assert protected_legacy.exists()
    assert invalid_shape.exists()
    assert unsafe_link.is_symlink()
    assert outside.read_bytes() == b"archive"
    assert list(tmp_path.glob("*.scai-export-owner")) == []


@pytest.mark.asyncio
async def test_snapshot_export_worker_and_session_lock_drain_before_cancellation(
    monkeypatch, tmp_path,
):
    from backend.services.snapshot_v2.exporter import SnapshotExporter
    from backend.services.snapshot_v2.paths import SnapshotV2Paths
    from backend.services.snapshot_v2.service import SnapshotService

    session_id = "snapshot-export-worker-drain"
    output_path = tmp_path / "v2-export-worker.tar.gz"
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_export(self, _session_id, **_kwargs):
        entered.set()
        assert release.wait(timeout=3)
        output_path.write_bytes(b"archive")
        finished.set()
        return output_path

    monkeypatch.setattr(SnapshotExporter, "export", blocking_export)
    paths = SnapshotV2Paths(
        root=tmp_path / "v2",
        objects=tmp_path / "objects",
        staging=tmp_path / "staging",
        journals=tmp_path / "journals",
        integrity=tmp_path / "integrity",
        committed=tmp_path / "committed",
    )
    service = SnapshotService(session_id, paths)
    export = asyncio.create_task(
        service.export_snapshots(output_path=output_path)
    )
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    export.cancel()
    await asyncio.sleep(0.05)
    assert not export.done()
    assert get_session_lock(session_id).locked()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(export, timeout=2)

    assert finished.is_set()
    assert output_path.read_bytes() == b"archive"
    assert not get_session_lock(session_id).locked()
    output_path.unlink()
