"""Tests for chunk_index boundary validation and total_size cap in upload service."""
import pytest
from pathlib import Path
from fastapi import HTTPException
from backend.services import upload as up


class TestChunkIndexBounds:
    @pytest.mark.asyncio
    async def test_chunk_index_out_of_range_rejected(self, monkeypatch, tmp_path):
        """chunk_index >= total_chunks must raise 400."""
        class _DB:
            async def get_upload_session(self, uid):
                return {"session_id": "s", "completed": False, "chunk_size": 10,
                        "total_chunks": 2, "total_size": 20}
            async def mark_chunk_received(self, uid, idx):
                return {"received_chunks": 1, "total_chunks": 2}
        async def _get_db():
            return _DB()
        monkeypatch.setattr(up, "get_db", _get_db)
        monkeypatch.setattr(up, "UPLOAD_TEMP_DIR", tmp_path)

        with pytest.raises(HTTPException) as ei:
            await up.upload_chunk("s", "uid", chunk_index=99, chunk_data=b"x")
        assert ei.value.status_code == 400


class TestInitUploadTotalSize:
    @pytest.mark.asyncio
    async def test_total_size_zero_rejected(self, monkeypatch, tmp_path):
        """total_size <= 0 must raise 413."""
        class _DB:
            async def create_upload_session(self, **kw):
                return "uid"
        async def _get_db():
            return _DB()

        class _Req:
            async def json(self):
                return {
                    "filename": "test.h5ad",
                    "expected_sha256": "abc" * 10,
                    "total_chunks": 2,
                    "chunk_size": 10,
                    "total_size": 0,
                }
            state = type("s", (), {"user_id": "u"})()

        class _State:
            user_id = "u"
            data_files = []
            def __init__(self):
                f = type("f", (), {"filename": "test.h5ad", "file_path": "", "id": "p1"})()
                self.data_files = [f]

        monkeypatch.setattr(up, "sessions", {"sid": _State()})
        monkeypatch.setattr(up, "UPLOAD_TEMP_DIR", tmp_path)
        monkeypatch.setattr(up, "get_db", _get_db)

        with pytest.raises(HTTPException) as ei:
            await up.init_upload("sid", _Req())
        assert ei.value.status_code == 413

    @pytest.mark.asyncio
    async def test_total_size_exceeds_max_rejected(self, monkeypatch, tmp_path):
        """total_size > MAX_UPLOAD_SIZE must raise 413."""
        from backend.config._config import MAX_UPLOAD_SIZE

        class _DB:
            async def create_upload_session(self, **kw):
                return "uid"
        async def _get_db():
            return _DB()

        class _Req:
            async def json(self):
                return {
                    "filename": "test.h5ad",
                    "expected_sha256": "abc" * 10,
                    "total_chunks": 2,
                    "chunk_size": 10,
                    "total_size": MAX_UPLOAD_SIZE + 1,
                }
            state = type("s", (), {"user_id": "u"})()

        class _State:
            user_id = "u"
            data_files = []
            def __init__(self):
                f = type("f", (), {"filename": "test.h5ad", "file_path": "", "id": "p1"})()
                self.data_files = [f]

        monkeypatch.setattr(up, "sessions", {"sid": _State()})
        monkeypatch.setattr(up, "UPLOAD_TEMP_DIR", tmp_path)
        monkeypatch.setattr(up, "get_db", _get_db)

        with pytest.raises(HTTPException) as ei:
            await up.init_upload("sid", _Req())
        assert ei.value.status_code == 413
