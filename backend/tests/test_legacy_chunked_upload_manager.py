"""Security and cancellation contracts for the legacy chunked uploader."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from backend.utils.upload import ChunkedUploadManager


@pytest.mark.asyncio
async def test_completion_and_commit_phase_order(monkeypatch, tmp_path):
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await manager.init_upload(
        "session-a", "data.bin", 4, 1, "file",
    )
    assert (await manager.save_chunk(
        upload_id, 0, b"data", session_id="session-a",
    ))["success"]

    trace = []
    original_load = manager._load_upload_meta_locked
    original_assemble = manager._assemble_chunks_sync
    original_publish = manager._publish_staged_sync
    original_write = manager._write_upload_meta_locked
    original_cleanup = manager._cleanup_loaded_upload_locked

    async def traced_load(*args, **kwargs):
        phase = "commit" if (tmp_path / "result.bin").exists() else "complete"
        trace.append(f"{phase}:load-meta")
        return await original_load(*args, **kwargs)

    def traced_assemble(*args, **kwargs):
        trace.append("complete:assemble")
        return original_assemble(*args, **kwargs)

    def traced_publish(*args, **kwargs):
        trace.append("complete:publish")
        return original_publish(*args, **kwargs)

    async def traced_write(upload_id, meta):
        if meta.get("completed"):
            trace.append("commit:write-completed-meta")
        elif meta.get("assembled_path"):
            trace.append("complete:write-assembled-meta")
        else:
            trace.append("complete:write-staged-meta")
        return await original_write(upload_id, meta)

    async def traced_cleanup(*args, **kwargs):
        trace.append("commit:cleanup")
        return await original_cleanup(*args, **kwargs)

    monkeypatch.setattr(manager, "_load_upload_meta_locked", traced_load)
    monkeypatch.setattr(manager, "_assemble_chunks_sync", traced_assemble)
    monkeypatch.setattr(manager, "_publish_staged_sync", traced_publish)
    monkeypatch.setattr(manager, "_write_upload_meta_locked", traced_write)
    monkeypatch.setattr(manager, "_cleanup_loaded_upload_locked", traced_cleanup)

    result = await manager.complete_upload(
        upload_id,
        tmp_path / "result.bin",
        session_id="session-a",
        defer_cleanup=True,
    )
    assert result["success"] is True
    assert await manager.commit_completed_upload(
        upload_id, session_id="session-a",
    )
    assert trace == [
        "complete:load-meta",
        "complete:assemble",
        "complete:write-staged-meta",
        "complete:publish",
        "complete:write-assembled-meta",
        "commit:load-meta",
        "commit:write-completed-meta",
        "commit:cleanup",
    ]


@pytest.mark.asyncio
async def test_direct_completion_publishes_then_cleans_without_metadata_write(
    monkeypatch, tmp_path,
):
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await manager.init_upload(
        "session-a", "data.bin", 4, 1, "file",
    )
    assert (await manager.save_chunk(
        upload_id, 0, b"data", session_id="session-a",
    ))["success"]

    trace = []
    original_publish = manager._publish_staged_sync
    original_cleanup = manager._cleanup_loaded_upload_locked

    def traced_publish(*args, **kwargs):
        trace.append("publish")
        return original_publish(*args, **kwargs)

    async def traced_cleanup(*args, **kwargs):
        trace.append("cleanup")
        return await original_cleanup(*args, **kwargs)

    async def unexpected_write(*_args, **_kwargs):
        raise AssertionError("direct completion must not update metadata")

    monkeypatch.setattr(manager, "_publish_staged_sync", traced_publish)
    monkeypatch.setattr(manager, "_cleanup_loaded_upload_locked", traced_cleanup)
    monkeypatch.setattr(manager, "_write_upload_meta_locked", unexpected_write)

    result = await manager.complete_upload(
        upload_id, tmp_path / "result.bin", session_id="session-a",
    )
    assert result["success"] is True
    assert trace == ["publish", "cleanup"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_phase",
    [
        "assemble",
        "write-staged-meta",
        "publish",
        "write-assembled-meta",
        "write-completed-meta",
        "cleanup",
    ],
)
async def test_transaction_boundary_failures_preserve_current_filesystem_state(
    monkeypatch, tmp_path, failed_phase,
):
    manager = ChunkedUploadManager(tmp_path / failed_phase / "chunks", chunk_size=4)
    upload_id = await manager.init_upload(
        "session-a", "data.bin", 4, 1, "file",
    )
    assert (await manager.save_chunk(
        upload_id, 0, b"data", session_id="session-a",
    ))["success"]
    final_path = tmp_path / failed_phase / "result.bin"
    staging_path = manager.temp_dir / f"{upload_id}.assembled.tmp"
    meta_path = manager._get_upload_meta_path(upload_id)

    if failed_phase == "assemble":
        def fail_assemble(*_args, **_kwargs):
            staging_path.write_bytes(b"partial")
            raise RuntimeError("injected assembly failure")

        monkeypatch.setattr(manager, "_assemble_chunks_sync", fail_assemble)
        result = await manager.complete_upload(
            upload_id, final_path, session_id="session-a", defer_cleanup=True,
        )
        assert result["success"] is False
        assert not staging_path.exists()
        assert not final_path.exists()
        assert not meta_path.exists()
        assert manager._active_uploads == {}
        return

    original_write = manager._write_upload_meta_locked
    write_count = 0

    async def maybe_fail_write(current_upload_id, meta):
        nonlocal write_count
        write_count += 1
        should_fail = (
            (failed_phase == "write-staged-meta" and write_count == 1)
            or (failed_phase == "write-assembled-meta" and write_count == 2)
        )
        if should_fail:
            raise OSError(f"injected {failed_phase} failure")
        return await original_write(current_upload_id, meta)

    if failed_phase in {"write-staged-meta", "write-assembled-meta"}:
        monkeypatch.setattr(manager, "_write_upload_meta_locked", maybe_fail_write)
    if failed_phase == "publish":
        def fail_publish(*_args, **_kwargs):
            raise OSError("injected publish failure")

        monkeypatch.setattr(manager, "_publish_staged_sync", fail_publish)

    if failed_phase in {
        "write-staged-meta", "publish", "write-assembled-meta",
    }:
        with pytest.raises(OSError):
            await manager.complete_upload(
                upload_id, final_path, session_id="session-a", defer_cleanup=True,
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert staging_path.exists()
        assert not final_path.exists()
        assert bool(meta.get("staged_path")) is (failed_phase != "write-staged-meta")
        assert not meta.get("assembled_path")
        assert manager._active_uploads == {"session-a": 1}
        return

    result = await manager.complete_upload(
        upload_id, final_path, session_id="session-a", defer_cleanup=True,
    )
    assert result["success"] is True

    if failed_phase == "write-completed-meta":
        async def fail_completed_write(*_args, **_kwargs):
            raise OSError("injected completed metadata failure")

        monkeypatch.setattr(manager, "_write_upload_meta_locked", fail_completed_write)
    else:
        async def fail_cleanup(*_args, **_kwargs):
            raise OSError("injected cleanup failure")

        monkeypatch.setattr(manager, "_cleanup_loaded_upload_locked", fail_cleanup)

    assert not await manager.commit_completed_upload(
        upload_id, session_id="session-a",
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert final_path.read_bytes() == b"data"
    assert meta.get("completed", False) is (failed_phase == "cleanup")
    assert manager._active_uploads == {}


@pytest.mark.asyncio
async def test_init_cancellation_rolls_back_quota_and_staging_file(
    monkeypatch, tmp_path,
):
    manager = ChunkedUploadManager(tmp_path, chunk_size=4)

    async def cancelled_write(upload_id, _meta):
        tmp = manager._get_upload_meta_path(upload_id).with_suffix(".tmp")
        tmp.write_text("partial", encoding="utf-8")
        tmp.unlink()
        raise asyncio.CancelledError()

    monkeypatch.setattr(manager, "_write_upload_meta_locked", cancelled_write)

    for _ in range(manager.MAX_CONCURRENT_UPLOADS_PER_SESSION + 1):
        with pytest.raises(asyncio.CancelledError):
            await manager.init_upload(
                "session-a", "data.bin", 4, 1, "file",
            )

    assert manager._active_uploads == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_init_late_cancellation_reclaims_durable_unpublished_meta(
    monkeypatch, tmp_path,
):
    manager = ChunkedUploadManager(tmp_path, chunk_size=4)
    original_write = manager._write_upload_meta_locked
    written = asyncio.Event()
    release = asyncio.Event()

    async def delayed_return(upload_id, meta):
        await original_write(upload_id, meta)
        written.set()
        await release.wait()

    monkeypatch.setattr(manager, "_write_upload_meta_locked", delayed_return)
    init = asyncio.create_task(
        manager.init_upload("session-a", "data.bin", 4, 1, "file")
    )
    await asyncio.wait_for(written.wait(), timeout=1)
    assert list(tmp_path.glob("*.meta.json"))
    init.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(init, timeout=2)

    release.set()
    assert manager._active_uploads == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_assembly_removes_partial_file_and_remains_resumable(
    monkeypatch, tmp_path,
):
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await manager.init_upload(
        "session-a", "data.bin", 4, 1, "file",
    )
    assert (await manager.save_chunk(
        upload_id, 0, b"data", session_id="session-a",
    ))["success"]

    final_path = tmp_path / "result" / "data.bin"
    original_assemble = manager._assemble_chunks_sync

    def cancelled_assemble(_upload_id, _total_chunks, staging_path):
        Path(staging_path).write_bytes(b"partial")
        raise asyncio.CancelledError()

    monkeypatch.setattr(manager, "_assemble_chunks_sync", cancelled_assemble)
    with pytest.raises(asyncio.CancelledError):
        await manager.complete_upload(
            upload_id, final_path, session_id="session-a",
        )

    assert not final_path.exists()
    assert manager._get_upload_meta_path(upload_id).is_file()
    assert manager._get_chunk_path(upload_id, 0).is_file()
    assert manager._active_uploads == {"session-a": 1}

    monkeypatch.setattr(manager, "_assemble_chunks_sync", original_assemble)
    result = await manager.complete_upload(
        upload_id, final_path, session_id="session-a",
    )
    assert result["success"] is True
    assert final_path.read_bytes() == b"data"
    assert not manager._get_upload_meta_path(upload_id).exists()
    assert not manager._get_chunk_path(upload_id, 0).exists()
    assert not (manager.temp_dir / f"{upload_id}.assembled.tmp").exists()
    assert manager._active_uploads == {}


@pytest.mark.asyncio
async def test_upload_ids_are_unique_for_same_session_and_filename(tmp_path):
    manager = ChunkedUploadManager(tmp_path, chunk_size=4)
    first = await manager.init_upload(
        "session-a", "same.bin", 4, 1, "file",
    )
    second = await manager.init_upload(
        "session-a", "same.bin", 4, 1, "file",
    )

    assert first != second
    assert manager._get_upload_meta_path(first).is_file()
    assert manager._get_upload_meta_path(second).is_file()

    await manager.cancel_upload(first, session_id="session-a")
    await manager.cancel_upload(second, session_id="session-a")


@pytest.mark.asyncio
async def test_cross_session_operations_fail_closed_without_mutation(tmp_path):
    manager = ChunkedUploadManager(tmp_path, chunk_size=4)
    upload_id = await manager.init_upload(
        "session-b", "private.bin", 4, 1, "file",
    )
    meta_path = manager._get_upload_meta_path(upload_id)
    original_meta = meta_path.read_bytes()
    final_path = tmp_path / "escaped.bin"

    save = await manager.save_chunk(
        upload_id, 0, b"data", session_id="session-a",
    )
    status = await manager.get_upload_status(
        upload_id, session_id="session-a",
    )
    cancel = await manager.cancel_upload(
        upload_id, session_id="session-a",
    )
    complete = await manager.complete_upload(
        upload_id, final_path, session_id="session-a",
    )

    assert save.get("not_found") is True
    assert status is None
    assert cancel.get("not_found") is True
    assert complete.get("not_found") is True
    assert meta_path.read_bytes() == original_meta
    assert not manager._get_chunk_path(upload_id, 0).exists()
    assert not final_path.exists()

    assert (await manager.cancel_upload(
        upload_id, session_id="session-b",
    ))["success"] is True


@pytest.mark.asyncio
async def test_unknown_and_invalid_chunks_leave_no_artifacts_or_lock_slots(tmp_path):
    manager = ChunkedUploadManager(tmp_path, chunk_size=4)
    unknown = f"session-a_{'a' * 32}"

    for _ in range(3):
        result = await manager.save_chunk(
            unknown, 0, b"data", session_id="session-a",
        )
        assert result.get("not_found") is True
    assert manager._upload_locks == {}
    assert manager._upload_lock_users == {}
    assert list(tmp_path.iterdir()) == []

    upload_id = await manager.init_upload(
        "session-a", "data.bin", 4, 1, "file",
    )
    for index, payload in ((-1, b"data"), (1, b"data"), (0, b"large")):
        result = await manager.save_chunk(
            upload_id, index, payload, session_id="session-a",
        )
        assert result["success"] is False
    assert not list(tmp_path.glob(f"{upload_id}.chunk.*"))
    assert manager._upload_locks == {}
    assert manager._upload_lock_users == {}

    await manager.cancel_upload(upload_id, session_id="session-a")


@pytest.mark.asyncio
async def test_completion_rejects_chunks_with_wrong_per_index_lengths(tmp_path):
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await manager.init_upload(
        "session-a", "data.bin", 6, 2, "file",
    )

    assert (await manager.save_chunk(
        upload_id, 0, b"ab", session_id="session-a",
    ))["success"]
    assert (await manager.save_chunk(
        upload_id, 1, b"cdef", session_id="session-a",
    ))["success"]

    result = await manager.complete_upload(
        upload_id,
        tmp_path / "result" / "data.bin",
        session_id="session-a",
    )

    assert result["success"] is False
    assert "chunk size" in result["error"].lower()


@pytest.mark.asyncio
async def test_completion_verifies_init_bound_whole_file_sha256(tmp_path):
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    payload = b"abcdef"
    upload_id = await manager.init_upload(
        "session-a",
        "data.bin",
        len(payload),
        2,
        "file",
        expected_sha256=hashlib.sha256(b"different").hexdigest(),
    )
    assert (await manager.save_chunk(
        upload_id, 0, payload[:4], session_id="session-a",
    ))["success"]
    assert (await manager.save_chunk(
        upload_id, 1, payload[4:], session_id="session-a",
    ))["success"]

    result = await manager.complete_upload(
        upload_id,
        tmp_path / "result" / "data.bin",
        session_id="session-a",
    )

    assert result["success"] is False
    assert result["integrity_mismatch"] is True
    assert not (tmp_path / "result" / "data.bin").exists()


@pytest.mark.asyncio
async def test_session_cleanup_removes_corrupt_orphan_and_staging_artifacts_only(
    tmp_path,
):
    manager = ChunkedUploadManager(tmp_path, chunk_size=4)
    session_id = "sample"
    owned_id = f"{session_id}_{'a' * 32}"
    legacy_id = f"{session_id}_20260802030405_deadbeef"
    near_id = f"sample_extra_{'b' * 32}"

    owned_names = (
        f"{owned_id}.meta.json",
        f"{owned_id}.meta.json.tmp",
        f"{owned_id}.chunk.0",
        f"{owned_id}.chunk.1.tmp",
        f"{owned_id}.assembled.tmp",
        f"{legacy_id}.chunk.0",
    )
    near_names = (
        f"{near_id}.meta.json",
        f"{near_id}.chunk.0",
        f"{near_id}.assembled.tmp",
    )
    for name in owned_names + near_names:
        (tmp_path / name).write_text("corrupt", encoding="utf-8")

    manager._active_uploads[session_id] = 2
    manager._upload_locks[owned_id] = asyncio.Lock()
    manager._upload_lock_users[owned_id] = 0

    await manager.cleanup_session_artifacts(session_id)

    assert all(not (tmp_path / name).exists() for name in owned_names)
    assert all((tmp_path / name).exists() for name in near_names)
    assert session_id not in manager._active_uploads
    assert owned_id not in manager._upload_locks
    assert owned_id not in manager._upload_lock_users
