"""Tests for V1 API endpoints and upload service logic."""
import sqlite3

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace


class TestUploadDB:
    @pytest.mark.asyncio
    async def test_init_migrates_legacy_upload_table_without_inventing_size(
        self, tmp_path
    ):
        from backend.utils.db import AppDB

        path = tmp_path / "legacy.db"
        with sqlite3.connect(path) as db:
            db.execute(
                """CREATE TABLE upload_sessions (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                filename TEXT NOT NULL, expected_sha256 TEXT NOT NULL,
                total_chunks INTEGER NOT NULL, chunk_size INTEGER NOT NULL,
                received_chunks INTEGER NOT NULL DEFAULT 0,
                received_mask TEXT NOT NULL DEFAULT '',
                placeholder_file_id TEXT, created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0
                )"""
            )
            db.execute(
                """INSERT INTO upload_sessions VALUES
                ('legacy','s','f','abc',1,10,0,'','p','now','later',0)"""
            )

        app_db = AppDB(path)
        await app_db.init()

        upload = await app_db.get_upload_session("legacy")
        assert upload is not None
        assert "total_size" in upload
        assert upload["total_size"] is None

    @pytest.mark.asyncio
    async def test_create_upload_session(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="test.h5ad", expected_sha256="abc",
            total_size=2048, total_chunks=2, chunk_size=1024,
            expires_at_iso="2025-01-01T00:00:00",
            placeholder_file_id="pf1",
        )
        upload = await app_db.get_upload_session(uid)
        assert upload is not None
        assert upload["session_id"] == "sid1"
        assert upload["filename"] == "test.h5ad"
        assert upload["total_size"] == 2048
        assert upload["received_chunks"] == 0

    @pytest.mark.asyncio
    async def test_mark_chunk_received(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=3072, total_chunks=3, chunk_size=1024,
            expires_at_iso="2025-01-01T00:00:00",
            placeholder_file_id="pf1",
        )
        r1 = await app_db.mark_chunk_received(uid, 0)
        assert r1["is_new"] is True
        r2 = await app_db.mark_chunk_received(uid, 1)
        assert r2["is_new"] is True
        # Duplicate
        r3 = await app_db.mark_chunk_received(uid, 0)
        assert r3["is_new"] is False

        upload = await app_db.get_upload_session(uid)
        assert upload["received_chunks"] == 2

    @pytest.mark.asyncio
    async def test_complete_upload_session(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=1024, total_chunks=1, chunk_size=1024,
            expires_at_iso="2025-01-01T00:00:00",
            placeholder_file_id="pf1",
        )
        await app_db.complete_upload_session(uid)
        upload = await app_db.get_upload_session(uid)
        assert upload["completed"] == 1

    @pytest.mark.asyncio
    async def test_list_expired_uploads(self, app_db):
        # Create an expired upload
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=1024, total_chunks=1, chunk_size=1024,
            expires_at_iso="2020-01-01T00:00:00",
            placeholder_file_id="pf1",
        )
        expired = await app_db.list_expired_uploads("2025-01-01T00:00:00")
        assert len(expired) == 1
        assert expired[0]["id"] == uid

    @pytest.mark.asyncio
    async def test_delete_session_record(self, app_db):
        """delete_session_record cleans up uploads for a session."""
        await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=1024, total_chunks=1, chunk_size=1024,
            expires_at_iso="2025-01-01T00:00:00",
            placeholder_file_id="pf1",
        )
        uploads_before = await app_db.list_uploads_for_session("sid1")
        assert len(uploads_before) == 1

        await app_db.delete_session_record("sid1")
        uploads_after = await app_db.list_uploads_for_session("sid1")
        assert len(uploads_after) == 0

    @pytest.mark.asyncio
    async def test_nonexistent_upload_returns_none(self, app_db):
        result = await app_db.get_upload_session("nope")
        assert result is None


class TestUploadService:
    @pytest.mark.asyncio
    async def test_v1_chunk_route_bounds_multipart_read_before_service_validation(
        self, monkeypatch
    ):
        from fastapi import HTTPException
        from backend.routes import v1

        class OversizedUpload:
            def __init__(self):
                self.read_sizes = []

            async def read(self, size):
                self.read_sizes.append(size)
                return b"x" * size

        upload = OversizedUpload()
        service_upload = AsyncMock()
        monkeypatch.setattr(v1, "UPLOAD_CHUNK_SIZE", 8)
        monkeypatch.setattr(v1, "require_session_ownership", AsyncMock())
        monkeypatch.setattr(v1, "upload_chunk", service_upload)
        request = SimpleNamespace(state=SimpleNamespace(user_id="u1"))

        with pytest.raises(HTTPException) as caught:
            await v1.upload_chunk_endpoint(
                "sid1", "upload1", 0, request, upload,
            )

        assert caught.value.status_code == 413
        assert upload.read_sizes == [9]
        service_upload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_chunk_writes_file(self, tmp_path):
        """upload_chunk writes data atomically and marks received."""
        from backend.services.upload import upload_chunk
        from backend.models.schemas import AnalysisState
        from backend.state import sessions

        # Set up a mock DB with an upload session
        mock_db = AsyncMock()
        mock_upload = {
            "id": "u1",
            "session_id": "sid1",
            "filename": "test.h5ad",
            "expected_sha256": "abc",
            "total_size": 10,
            "total_chunks": 2,
            "chunk_size": 5,
            "received_chunks": 0,
            "completed": 0,
        }
        mock_db.get_upload_session = AsyncMock(side_effect=[mock_upload, {**mock_upload, "received_chunks": 1}])
        mock_db.mark_chunk_received = AsyncMock(return_value={"is_new": True, "received_chunks": 1, "total_chunks": 2})

        sessions["sid1"] = AnalysisState(id="sid1", user_id="u1")
        try:
            with patch("backend.services.upload.get_db", return_value=mock_db), \
                 patch("backend.services.upload.UPLOAD_TEMP_DIR", tmp_path):
                result = await upload_chunk("sid1", "u1", 0, b"hello")
        finally:
            sessions.pop("sid1", None)

        assert result["received_chunks"] == 1
        assert result["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_upload_chunk_rejects_unknown_session(self, tmp_path):
        from backend.services.upload import upload_chunk

        mock_db = AsyncMock()
        mock_db.get_upload_session = AsyncMock(return_value=None)

        with patch("backend.services.upload.get_db", return_value=mock_db):
            with pytest.raises(Exception) as exc_info:
                await upload_chunk("sid1", "nope", 0, b"data")
            assert "not found" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_upload_chunk_rejects_completed(self, tmp_path):
        from backend.services.upload import upload_chunk
        from backend.models.schemas import AnalysisState
        from backend.state import sessions

        mock_db = AsyncMock()
        mock_db.get_upload_session = AsyncMock(return_value={
            "id": "u1", "session_id": "sid1", "completed": 1,
            "chunk_size": 1024, "received_chunks": 2, "total_chunks": 2,
        })

        sessions["sid1"] = AnalysisState(id="sid1", user_id="u1")
        try:
            with patch("backend.services.upload.get_db", return_value=mock_db):
                with pytest.raises(Exception) as exc_info:
                    await upload_chunk("sid1", "u1", 0, b"data")
                assert "already completed" in str(exc_info.value.detail).lower()
        finally:
            sessions.pop("sid1", None)


class TestReportArtifacts:
    @pytest.mark.asyncio
    async def test_v1_artifacts_expose_only_current_branch_generation(
        self, tmp_path, monkeypatch,
    ):
        from backend.models.schemas import AnalysisState, TaskStatus
        from backend.routes import v1
        from backend.services.report.output_generation import ReportGenerationTransaction
        from backend.services.report.paths import resolve_report_output_dir

        state = AnalysisState(
            id="sid1",
            user_id="u1",
            status=TaskStatus.COMPLETED,
            restore_branch_id="branch/A",
        )
        monkeypatch.setattr(
            "backend.services.report.paths.get_session_reports",
            lambda _session_id: tmp_path,
        )
        branch_root = resolve_report_output_dir("sid1", "branch/A")
        transaction = ReportGenerationTransaction(branch_root)
        (transaction.staging_dir / "figures").mkdir()
        (transaction.staging_dir / "tables").mkdir()
        (transaction.staging_dir / "figures" / "panel.png").write_bytes(b"branch")
        (transaction.staging_dir / "report.html").write_text("branch", encoding="utf-8")
        transaction.commit_report({"id": "branch", "branch_id": "branch/A"})
        (tmp_path / "report.html").write_text("main", encoding="utf-8")
        monkeypatch.setattr(v1, "require_session_ownership", AsyncMock(return_value=state))
        request = SimpleNamespace(state=SimpleNamespace(user_id="u1"))

        payload = await v1.get_analysis("sid1", request)
        names = {item["name"] for item in payload["artifacts"]}
        assert names == {"report.json", "report.html", "figures/panel.png"}

        response = await v1.download_artifact("sid1", "report.html", request)
        assert Path(response.path).read_text(encoding="utf-8") == "branch"
        with pytest.raises(Exception) as exc_info:
            await v1.download_artifact("sid1", ".report-current.json", request)
        assert exc_info.value.status_code == 404
