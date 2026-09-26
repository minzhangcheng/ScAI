"""Security and cancellation contracts for the legacy chunked uploader."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from backend.utils.upload import ChunkedUploadManager


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
