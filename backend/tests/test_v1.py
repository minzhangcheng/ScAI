"""Tests for V1 API endpoints and upload service logic."""
import os

import hashlib
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def config_dict():
    return {
        "schema_version": "1.0",
        "analysis": {"name": "API Test", "initial_instruction": "Test", "round_wait_seconds": 0},
        "api_config": {
            "default_llm": {"llm_base_url": "https://api.test.com", "api_key": "sk-test"},
            "literature": {"mode": "disabled"},
        },
        "data": {
            "groups": [{"key": "ctrl", "name": "Control", "sample_count": 1}],
            "files": [
                {"local_path": "test.h5ad", "section": "sequencing", "data_type": "h5ad", "group": "ctrl"},
            ],
        },
    }


class TestUploadDB:
    @pytest.mark.asyncio
    async def test_create_upload_session(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="test.h5ad", expected_sha256="abc",
            total_chunks=2, chunk_size=1024, expires_at_iso="2025-01-01T00:00:00",
            placeholder_file_id="pf1",
        )
        upload = await app_db.get_upload_session(uid)
        assert upload is not None
        assert upload["session_id"] == "sid1"
        assert upload["filename"] == "test.h5ad"
        assert upload["received_chunks"] == 0

    @pytest.mark.asyncio
    async def test_mark_chunk_received(self, app_db):
        uid = await app_db.create_upload_session(
            session_id="sid1", filename="f.h5ad", expected_sha256="abc",
            total_chunks=3, chunk_size=1024, expires_at_iso="2025-01-01T00:00:00",
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
            total_chunks=1, chunk_size=1024, expires_at_iso="2025-01-01T00:00:00",
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
            total_chunks=1, chunk_size=1024, expires_at_iso="2020-01-01T00:00:00",
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
            total_chunks=1, chunk_size=1024, expires_at_iso="2025-01-01T00:00:00",
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
    async def test_upload_chunk_writes_file(self, tmp_path):
        """upload_chunk writes data atomically and marks received."""
        from backend.services.upload import upload_chunk
        from backend.config import UPLOAD_TEMP_DIR

        # Set up a mock DB with an upload session
        mock_db = AsyncMock()
        mock_upload = {
            "id": "u1",
            "session_id": "sid1",
            "filename": "test.h5ad",
            "expected_sha256": "abc",
            "total_chunks": 2,
            "chunk_size": 1024,
            "received_chunks": 0,
            "completed": 0,
        }
        mock_db.get_upload_session = AsyncMock(side_effect=[mock_upload, {**mock_upload, "received_chunks": 1}])
        mock_db.mark_chunk_received = AsyncMock(return_value={"is_new": True, "received_chunks": 1, "total_chunks": 2})

        with patch("backend.services.upload.get_db", return_value=mock_db), \
             patch("backend.services.upload.UPLOAD_TEMP_DIR", tmp_path):
            result = await upload_chunk("sid1", "u1", 0, b"hello")

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

        mock_db = AsyncMock()
        mock_db.get_upload_session = AsyncMock(return_value={
            "id": "u1", "session_id": "sid1", "completed": 1,
            "chunk_size": 1024, "received_chunks": 2, "total_chunks": 2,
        })

        with patch("backend.services.upload.get_db", return_value=mock_db):
            with pytest.raises(Exception) as exc_info:
                await upload_chunk("sid1", "u1", 0, b"data")
            assert "already completed" in str(exc_info.value.detail).lower()


class TestValidation:
    def test_round_wait_minus1_valid_in_config(self, config_dict):
        """-1 should pass ConfigV1 validation (manual skip mode)."""
        from backend.models.config_schema import ConfigV1
        config_dict["analysis"]["round_wait_seconds"] = -1
        c = ConfigV1.model_validate(config_dict)
        assert c.analysis.round_wait_seconds == -1

    def test_agent_override_custom_prompt(self):
        from backend.models.config_schema import AgentOverride
        ov = AgentOverride(custom_prompt="Be precise")
        assert ov.custom_prompt == "Be precise"
        assert not hasattr(ov, "prompt")

    def test_agent_override_vision(self):
        from backend.models.config_schema import AgentOverride
        ov = AgentOverride(vision=True)
        assert ov.vision is True
        assert not hasattr(ov, "vision_enabled")