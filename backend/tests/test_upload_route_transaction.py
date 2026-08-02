"""Transactional contracts for ordinary (non-chunked) upload routes."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.responses import Response

from backend.models.schemas import AnalysisState, TaskStatus
from backend.routes import upload as upload_routes


class ChunkUpload:
    def __init__(self, filename: str, chunks: list[bytes]):
        self.filename = filename
        self.size = sum(len(chunk) for chunk in chunks)
        self._chunks = list(chunks)

    async def read(self, _size=-1):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _configure_route(monkeypatch, tmp_path, state):
    uploads = tmp_path / "uploads"

    @asynccontextmanager
    async def barrier(_session_id, current):
        assert current is state
        yield

    monkeypatch.setattr(
        upload_routes,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    monkeypatch.setattr(upload_routes, "_upload_generation_barrier", barrier)
    monkeypatch.setattr(upload_routes, "get_session_uploads", lambda _sid: uploads)
    return uploads


async def _upload_data(state, file, *, section="pathology"):
    request = SimpleNamespace(state=SimpleNamespace(user_id=state.user_id))
    response = Response()
    result = await upload_routes.upload_data(
        state.id,
        request,
        response,
        file=file,
        section=section,
        data_type="image" if section != "sequencing" else "h5ad",
        group_id=None,
        sample_number=None,
        description=None,
    )
    return result, response


@pytest.mark.asyncio
async def test_data_upload_conflict_preserves_existing_file_and_state(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="upload-conflict", user_id="owner")
    uploads = _configure_route(monkeypatch, tmp_path, state)
    final = uploads / "pathology" / "_ungrouped" / "sample.png"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"existing")
    save = AsyncMock()
    monkeypatch.setattr(
        upload_routes, "_async_save_session_under_lifecycle", save,
    )

    with pytest.raises(HTTPException) as raised:
        await _upload_data(state, ChunkUpload("sample.png", [b"replacement"]))

    assert raised.value.status_code == 409
    assert final.read_bytes() == b"existing"
    assert state.data_files == []
    assert state.pending_uploaded_files == []
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_data_upload_save_failure_rolls_back_file_and_memory(
    monkeypatch, tmp_path,
):
    state = AnalysisState(
        id="upload-save-failure",
        user_id="owner",
        status=TaskStatus.WAITING_INPUT,
    )
    uploads = _configure_route(monkeypatch, tmp_path, state)

    async def fail_save(*_args, **_kwargs):
        raise OSError("disk failed")

    monkeypatch.setattr(
        upload_routes, "_async_save_session_under_lifecycle", fail_save,
    )

    with pytest.raises(OSError, match="disk failed"):
        await _upload_data(state, ChunkUpload("sample.png", [b"payload"]))

    final = uploads / "pathology" / "_ungrouped" / "sample.png"
    assert not final.exists()
    assert state.data_files == []
    assert state.pending_uploaded_files == []
    assert list(uploads.rglob("*.uploading")) == []


@pytest.mark.asyncio
async def test_sequencing_upload_cancellation_leaves_no_orphan_or_record(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="upload-cancel", user_id="owner")
    uploads = _configure_route(monkeypatch, tmp_path, state)
    save = AsyncMock()
    monkeypatch.setattr(
        upload_routes, "_async_save_session_under_lifecycle", save,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingUpload:
        filename = "matrix.h5ad"
        size = 7
        calls = 0

        async def read(self, _size=-1):
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            entered.set()
            await release.wait()
            return b""

    request = asyncio.create_task(
        _upload_data(state, BlockingUpload(), section="sequencing"),
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    request.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, timeout=2)

    final = uploads / "sequencing" / "_ungrouped" / "matrix.h5ad"
    assert not final.exists()
    assert list(uploads.rglob("*.uploading")) == []
    assert state.data_files == []
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_after_durable_save_keeps_committed_upload(
    monkeypatch, tmp_path,
):
    state = AnalysisState(
        id="upload-cancel-after-save",
        user_id="owner",
        status=TaskStatus.COMPLETED,
    )
    uploads = _configure_route(monkeypatch, tmp_path, state)
    save_started = asyncio.Event()
    release_save = asyncio.Event()

    async def save(*_args, **kwargs):
        assert kwargs["force"] is True
        save_started.set()
        await release_save.wait()

    monkeypatch.setattr(
        upload_routes, "_async_save_session_under_lifecycle", save,
    )
    request = asyncio.create_task(
        _upload_data(state, ChunkUpload("sample.png", [b"payload"])),
    )
    await asyncio.wait_for(save_started.wait(), timeout=1)
    request.cancel()
    release_save.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, timeout=2)

    final = uploads / "pathology" / "_ungrouped" / "sample.png"
    assert final.read_bytes() == b"payload"
    assert len(state.data_files) == 1
    assert len(state.pending_uploaded_files) == 1
    assert list(uploads.rglob("*.uploading")) == []


@pytest.mark.asyncio
async def test_successful_data_upload_commits_once_without_staging(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="upload-success", user_id="owner")
    uploads = _configure_route(monkeypatch, tmp_path, state)
    save = AsyncMock()
    monkeypatch.setattr(
        upload_routes, "_async_save_session_under_lifecycle", save,
    )

    result, response = await _upload_data(
        state, ChunkUpload("sample.png", [b"pay", b"load"]),
    )

    final = uploads / "pathology" / "_ungrouped" / "sample.png"
    assert result.success is True
    assert response.headers["X-Request-ID"]
    assert final.read_bytes() == b"payload"
    assert len(state.data_files) == 1
    assert list(uploads.rglob("*.uploading")) == []
    save.assert_awaited_once()
    assert save.await_args.kwargs["force"] is True


@pytest.mark.asyncio
async def test_supplementary_upload_conflict_and_oversize_leave_no_artifacts(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="generic-file-transaction", user_id="owner")
    uploads = _configure_route(monkeypatch, tmp_path, state)
    request = SimpleNamespace(state=SimpleNamespace(user_id="owner"))
    final = uploads / "files" / "notes.txt"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"existing")

    with pytest.raises(HTTPException) as conflict:
        await upload_routes.upload_generic_file(
            state.id,
            request,
            file=ChunkUpload("notes.txt", [b"replacement"]),
        )
    assert conflict.value.status_code == 409
    assert final.read_bytes() == b"existing"

    monkeypatch.setattr(upload_routes, "_GENERIC_UPLOAD_MAX_BYTES", 15)
    with pytest.raises(HTTPException) as oversized:
        await upload_routes.upload_generic_file(
            state.id,
            request,
            file=ChunkUpload(
                "large.bin",
                [b"a" * 10, b"b" * 10],
            ),
        )
    assert oversized.value.status_code == 413
    assert not (uploads / "files" / "large.bin").exists()
    assert list(uploads.rglob("*.uploading")) == []
