"""Export owner, streaming, TTL, and worker-drain contracts."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

from backend.services.export_lifecycle import (
    LifecycleFileResponse,
    claim_export_stream,
    cleanup_export_record,
    create_export_ownership_marker,
    export_ownership_marker_path,
)
from backend.state import (
    get_session_lock,
    prepared_exports,
    prepared_exports_lock,
)


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
