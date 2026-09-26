"""Tests for SHA256 rollback, mark_chunk idempotency, and complete_upload edge cases."""

import hashlib
import pytest
from unittest.mock import patch

from backend.services.upload import complete_upload
from backend.state import sessions
from backend.models.schemas import AnalysisState, DataFile


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


class TestMarkChunkIdempotency:

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
