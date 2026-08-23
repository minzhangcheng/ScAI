"""Tests for SHA256 rollback, mark_chunk idempotency, and complete_upload edge cases."""
import os

import hashlib
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from backend.services.upload import upload_chunk, complete_upload, cleanup_session_uploads
from backend.state import sessions
from backend.models.schemas import AnalysisState, DataFile, TaskStatus


def _make_state(user_id="u1", session_id="sid1"):
    """Create a minimal AnalysisState with one placeholder file."""
    state = AnalysisState(
        id=session_id,
        user_id=user_id,
        name="Test",
        data_files=[
            DataFile(id="pf1", filename="test.h5ad", file_path="", section="sequencing", data_type="h5ad"),
        ],
    )
    return state


class TestSHA256Rollback:
    @pytest.mark.asyncio
    async def test_integrity_fail_cleans_final_and_chunks(self, tmp_path, app_db):
        """SHA256 mismatch: final file + all chunks must be cleaned up."""
        content = b"hello world"

        uid = await app_db.create_upload_session(
            session_id="sid1", filename="test.h5ad", expected_sha256="wrong_sha256",
            total_size=len(content), total_chunks=1, chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )
        chunk_path = tmp_path / f"{uid}.chunk.0"
        chunk_path.write_bytes(content)
        await app_db.mark_chunk_received(uid, 0)

        state = _make_state()
        sessions["sid1"] = state

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        with patch("backend.services.upload.get_db", return_value=app_db), \
             patch("backend.services.upload.get_session_uploads", return_value=upload_dir), \
             patch("backend.services.upload.UPLOAD_TEMP_DIR", tmp_path), \
             patch("backend.services.upload._on_session_update", None), \
             patch("backend.services.upload._on_session_save", None):
            with pytest.raises(Exception) as exc_info:
                await complete_upload("sid1", uid)
            assert "integrity" in str(exc_info.value.detail).lower()

        final_path = upload_dir / "sequencing/_ungrouped/test.h5ad"
        assert not final_path.exists()
        assert not chunk_path.exists()

        sessions.pop("sid1", None)

    @pytest.mark.asyncio
    async def test_successful_upload_cleans_chunks(self, tmp_path, app_db):
        """Successful complete cleans up chunk files."""
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()

        uid = await app_db.create_upload_session(
            session_id="sid1", filename="test.h5ad", expected_sha256=expected,
            total_size=len(content), total_chunks=1, chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )
        chunk_path = tmp_path / f"{uid}.chunk.0"
        chunk_path.write_bytes(content)
        await app_db.mark_chunk_received(uid, 0)

        state = _make_state()
        sessions["sid1"] = state

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        with patch("backend.services.upload.get_db", return_value=app_db), \
             patch("backend.services.upload.get_session_uploads", return_value=upload_dir), \
             patch("backend.services.upload.UPLOAD_TEMP_DIR", tmp_path), \
             patch("backend.services.upload._on_session_update", AsyncMock()) as mock_update, \
             patch("backend.services.upload._on_session_save", AsyncMock()):
            result = await complete_upload("sid1", uid)

        assert result["success"] is True
        assert (upload_dir / "sequencing/_ungrouped/test.h5ad").exists()
        assert not chunk_path.exists()
        assert state.data_files[0].file_path == str(
            upload_dir / "sequencing/_ungrouped/test.h5ad"
        )

        sessions.pop("sid1", None)

    @pytest.mark.asyncio
    async def test_complete_rejects_chunk_tampered_after_acceptance(
        self, tmp_path, app_db
    ):
        content = b"hello"
        uid = await app_db.create_upload_session(
            session_id="sid1",
            filename="test.h5ad",
            expected_sha256=hashlib.sha256(content).hexdigest(),
            total_size=len(content),
            total_chunks=1,
            chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )
        chunk_path = tmp_path / f"{uid}.chunk.0"
        chunk_path.write_bytes(content + b"tampered")
        await app_db.mark_chunk_received(uid, 0)
        sessions["sid1"] = _make_state()
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        with patch("backend.services.upload.get_db", return_value=app_db), \
             patch("backend.services.upload.get_session_uploads", return_value=upload_dir), \
             patch("backend.services.upload.UPLOAD_TEMP_DIR", tmp_path):
            with pytest.raises(Exception) as exc_info:
                await complete_upload("sid1", uid)

        assert "expected 5" in str(exc_info.value.detail)
        assert not (upload_dir / "sequencing/_ungrouped/test.h5ad").exists()
        sessions.pop("sid1", None)

    @pytest.mark.asyncio
    async def test_idempotent_complete_returns_stored_path(self, tmp_path, app_db):
        """Calling complete twice returns the stored path from first call."""
        content = b"hello world"
        expected = hashlib.sha256(content).hexdigest()

        uid = await app_db.create_upload_session(
            session_id="sid1", filename="test.h5ad", expected_sha256=expected,
            total_size=len(content), total_chunks=1, chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )
        chunk_path = tmp_path / f"{uid}.chunk.0"
        chunk_path.write_bytes(content)
        await app_db.mark_chunk_received(uid, 0)
        await app_db.complete_upload_session(uid)

        state = _make_state()
        upload_dir = tmp_path / "uploads"
        stored_path = upload_dir / "sequencing/_ungrouped/test.h5ad"
        state.data_files[0].file_path = str(stored_path)
        sessions["sid1"] = state

        with patch("backend.services.upload.get_db", return_value=app_db), \
             patch("backend.services.upload.get_session_uploads", return_value=upload_dir), \
             patch("backend.services.upload._on_session_update", None), \
             patch("backend.services.upload._on_session_save", None):
            result = await complete_upload("sid1", uid)

        assert result["success"] is True
        assert result["path"] == str(stored_path)

        sessions.pop("sid1", None)


class TestMarkChunkIdempotency:
    @pytest.mark.asyncio
    async def test_duplicate_chunk_returns_false(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=3072, total_chunks=3, chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )

        r1 = await app_db.mark_chunk_received(uid, 0)
        assert r1["is_new"] is True
        r2 = await app_db.mark_chunk_received(uid, 0)
        assert r2["is_new"] is False
        r3 = await app_db.mark_chunk_received(uid, 1)
        assert r3["is_new"] is True

        upload = await app_db.get_upload_session(uid)
        assert upload["received_chunks"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_mark_chunk_no_double_count(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=3072, total_chunks=3, chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )

        import asyncio
        results = await asyncio.gather(
            app_db.mark_chunk_received(uid, 0),
            app_db.mark_chunk_received(uid, 0),
        )
        is_new_values = [r["is_new"] for r in results]
        assert is_new_values.count(True) == 1
        assert is_new_values.count(False) == 1

        upload = await app_db.get_upload_session(uid)
        assert upload["received_chunks"] == 1


class TestCleanupSessionUploads:
    @pytest.mark.asyncio
    async def test_cleanup_deletes_chunks_and_db(self, tmp_path, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_size=2048, total_chunks=2, chunk_size=1024,
            expires_at_iso="2025-12-01T00:00:00",
            placeholder_file_id="pf1",
        )

        (tmp_path / f"{uid}.chunk.0").write_bytes(b"a")
        (tmp_path / f"{uid}.chunk.1").write_bytes(b"b")

        with patch("backend.services.upload.get_db", return_value=app_db), \
             patch("backend.services.upload.UPLOAD_TEMP_DIR", tmp_path):
            await cleanup_session_uploads("sid1")

        assert not (tmp_path / f"{uid}.chunk.0").exists()
        assert not (tmp_path / f"{uid}.chunk.1").exists()

        uploads = await app_db.list_uploads_for_session("sid1")
        assert len(uploads) == 0
