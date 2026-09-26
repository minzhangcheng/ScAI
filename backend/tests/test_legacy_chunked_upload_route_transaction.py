"""Route-level commit/rollback contracts for the legacy chunk uploader."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisState
from backend.routes import upload as upload_routes
from backend.utils.upload import ChunkedUploadManager


async def _ready_upload(
    manager: ChunkedUploadManager,
    session_id: str,
    *,
    file_type: str,
) -> str:
    upload_id = await manager.init_upload(
        session_id,
        f"{file_type}.bin",
        4,
        1,
        file_type,
    )
    result = await manager.save_chunk(
        upload_id,
        0,
        b"data",
        session_id=session_id,
    )
    assert result["success"] is True
    return upload_id


def _patch_route(
    monkeypatch,
    manager: ChunkedUploadManager,
    state: AnalysisState,
    uploads,
    save,
) -> SimpleNamespace:
    monkeypatch.setattr(
        "backend.services.upload.upload_manager",
        manager,
    )
    monkeypatch.setattr(
        upload_routes,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    monkeypatch.setattr(upload_routes, "assert_generation_writable", lambda *_: None)
    monkeypatch.setattr(upload_routes, "get_session_uploads", lambda _sid: uploads)
    monkeypatch.setattr(
        upload_routes,
        "_async_save_session_under_lifecycle",
        save,
    )
    return SimpleNamespace(state=SimpleNamespace(user_id=state.user_id))


@pytest.mark.asyncio
async def test_save_failure_rolls_back_file_and_remains_retryable(
    monkeypatch,
    tmp_path,
):
    session_id = "legacy-save-failure"
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await _ready_upload(
        manager, session_id, file_type="biochemistry",
    )
    state = AnalysisState(id=session_id, user_id="owner")
    failing_save = AsyncMock(side_effect=OSError("disk full"))
    request = _patch_route(
        monkeypatch,
        manager,
        state,
        tmp_path / "uploads",
        failing_save,
    )
    final_path = tmp_path / "uploads" / "biochemistry" / "biochemistry.bin"

    with pytest.raises(OSError, match="disk full"):
        await upload_routes.complete_chunked_upload(
            session_id,
            upload_id,
            request,
            group=None,
        )

    assert state.data_files == []
    assert not final_path.exists()
    assert manager._get_upload_meta_path(upload_id).is_file()
    assert manager._get_chunk_path(upload_id, 0).is_file()
    assert not (manager.temp_dir / f"{upload_id}.assembled.tmp").exists()
    assert manager._active_uploads[session_id] == 1

    successful_save = AsyncMock()
    monkeypatch.setattr(
        upload_routes,
        "_async_save_session_under_lifecycle",
        successful_save,
    )
    response = await upload_routes.complete_chunked_upload(
        session_id,
        upload_id,
        request,
        group=None,
    )

    assert response.success is True
    assert final_path.read_bytes() == b"data"
    assert len(state.data_files) == 1
    assert state.data_files[0].section == "biochemistry"
    successful_save.assert_awaited_once()
    assert not manager._get_upload_meta_path(upload_id).exists()
    assert not manager._get_chunk_path(upload_id, 0).exists()
    assert not (manager.temp_dir / f"{upload_id}.assembled.tmp").exists()
    assert session_id not in manager._active_uploads


@pytest.mark.asyncio
async def test_request_cancellation_drains_state_commit_before_cleanup(
    monkeypatch,
    tmp_path,
):
    session_id = "legacy-cancel-after-assembly"
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await _ready_upload(
        manager, session_id, file_type="biochemistry",
    )
    state = AnalysisState(id=session_id, user_id="owner")
    save_entered = asyncio.Event()
    release_save = asyncio.Event()

    async def delayed_save(*_args, **_kwargs):
        save_entered.set()
        await release_save.wait()

    request = _patch_route(
        monkeypatch,
        manager,
        state,
        tmp_path / "uploads",
        delayed_save,
    )
    final_path = tmp_path / "uploads" / "biochemistry" / "biochemistry.bin"
    completion = asyncio.create_task(
        upload_routes.complete_chunked_upload(
            session_id,
            upload_id,
            request,
            group=None,
        )
    )
    try:
        await asyncio.wait_for(save_entered.wait(), timeout=1)
        completion.cancel()
        await asyncio.sleep(0)
        assert not completion.done()
        release_save.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(completion, timeout=2)

        assert final_path.read_bytes() == b"data"
        assert len(state.data_files) == 1
        assert not manager._get_upload_meta_path(upload_id).exists()
        assert not manager._get_chunk_path(upload_id, 0).exists()
    finally:
        release_save.set()
        if not completion.done():
            completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_marker_failure_is_successful_and_retry_idempotent(
    monkeypatch,
    tmp_path,
):
    session_id = "legacy-marker-failure"
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await _ready_upload(
        manager, session_id, file_type="biochemistry",
    )
    state = AnalysisState(id=session_id, user_id="owner")
    save = AsyncMock()
    request = _patch_route(
        monkeypatch, manager, state, tmp_path / "uploads", save,
    )
    original_write = manager._write_upload_meta_locked
    fail_marker = True

    async def marker_write(upload_key, meta):
        if fail_marker and meta.get("completed"):
            raise OSError("marker disk failure")
        await original_write(upload_key, meta)

    monkeypatch.setattr(manager, "_write_upload_meta_locked", marker_write)

    first = await upload_routes.complete_chunked_upload(
        session_id, upload_id, request, group=None,
    )
    second = await upload_routes.complete_chunked_upload(
        session_id, upload_id, request, group=None,
    )

    assert first.success is True and first.data["cleanup_pending"] is True
    assert second.success is True and second.data["cleanup_pending"] is True
    assert second.data["reused_assembly"] is True
    assert len(state.data_files) == 1
    save.assert_awaited_once()
    assert manager._get_upload_meta_path(upload_id).is_file()
    assert manager._get_chunk_path(upload_id, 0).is_file()
    assert session_id not in manager._active_uploads

    fail_marker = False
    third = await upload_routes.complete_chunked_upload(
        session_id, upload_id, request, group=None,
    )
    assert third.success is True and third.data["cleanup_pending"] is False
    assert len(state.data_files) == 1
    assert not manager._get_upload_meta_path(upload_id).exists()
    assert not manager._get_chunk_path(upload_id, 0).exists()
    assert session_id not in manager._active_uploads


@pytest.mark.asyncio
async def test_unrelated_destination_conflict_never_consumes_upload(
    monkeypatch,
    tmp_path,
):
    session_id = "legacy-destination-conflict"
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await _ready_upload(
        manager, session_id, file_type="biochemistry",
    )
    state = AnalysisState(id=session_id, user_id="owner")
    request = _patch_route(
        monkeypatch, manager, state, tmp_path / "uploads", AsyncMock(),
    )
    final_path = tmp_path / "uploads" / "biochemistry" / "biochemistry.bin"
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"other")

    with pytest.raises(HTTPException) as exc_info:
        await upload_routes.complete_chunked_upload(
            session_id, upload_id, request, group=None,
        )

    assert exc_info.value.status_code == 400
    assert final_path.read_bytes() == b"other"
    assert state.data_files == []
    assert manager._get_upload_meta_path(upload_id).is_file()
    assert manager._get_chunk_path(upload_id, 0).is_file()


@pytest.mark.asyncio
async def test_clean_cancel_never_reclaims_later_unrelated_destination(
    monkeypatch,
    tmp_path,
):
    session_id = "legacy-clean-cancel-conflict"
    manager = ChunkedUploadManager(tmp_path / "chunks", chunk_size=4)
    upload_id = await _ready_upload(
        manager, session_id, file_type="biochemistry",
    )
    final_path = tmp_path / "uploads" / "biochemistry.bin"
    entered = threading.Event()
    release = threading.Event()
    original_assemble = manager._assemble_chunks_sync

    def delayed_assemble(*args, **kwargs):
        result = original_assemble(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=2)
        return result

    monkeypatch.setattr(manager, "_assemble_chunks_sync", delayed_assemble)
    completion = asyncio.create_task(
        manager.complete_upload(
            upload_id,
            final_path,
            session_id=session_id,
            defer_cleanup=True,
        )
    )
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    completion.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(completion, timeout=2)
    assert not final_path.exists()

    monkeypatch.setattr(manager, "_assemble_chunks_sync", original_assemble)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"EVIL")
    retried = await manager.complete_upload(
        upload_id,
        final_path,
        session_id=session_id,
        defer_cleanup=True,
    )

    assert retried["success"] is False
    assert retried["conflict"] is True
    assert final_path.read_bytes() == b"EVIL"
