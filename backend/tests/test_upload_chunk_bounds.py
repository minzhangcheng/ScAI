"""Tests for chunk_index boundary validation and total_size cap in upload service."""
import pytest
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
        # Generation membership is a separate route/service contract. This
        # unit test starts from a current cached session so it can reach the
        # chunk-index validation it is specifically asserting.
        monkeypatch.setattr(up, "sessions", {"s": object()})
        monkeypatch.setattr(up, "assert_generation_writable", lambda *_: None)

        with pytest.raises(HTTPException) as ei:
            await up.upload_chunk("s", "uid", chunk_index=99, chunk_data=b"x")
        assert ei.value.status_code == 400

    @pytest.mark.asyncio
    async def test_chunk_length_is_bound_to_index_and_total_size(
        self, monkeypatch, tmp_path
    ):
        class _DB:
            async def get_upload_session(self, uid):
                return {
                    "session_id": "s",
                    "completed": False,
                    "chunk_size": 10,
                    "total_chunks": 2,
                    "total_size": 11,
                }

        async def _get_db():
            return _DB()

        monkeypatch.setattr(up, "get_db", _get_db)
        monkeypatch.setattr(up, "UPLOAD_TEMP_DIR", tmp_path)
        monkeypatch.setattr(up, "sessions", {"s": object()})
        monkeypatch.setattr(up, "assert_generation_writable", lambda *_: None)

        with pytest.raises(HTTPException) as first:
            await up.upload_chunk("s", "uid", chunk_index=0, chunk_data=b"x" * 9)
        assert first.value.status_code == 400
        assert "expected 10" in first.value.detail

        with pytest.raises(HTTPException) as last:
            await up.upload_chunk("s", "uid", chunk_index=1, chunk_data=b"xx")
        assert last.value.status_code == 400
        assert "expected 1" in last.value.detail


class TestInitUploadTotalSize:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "digest",
        ["abc", "g" * 64, "a" * 63, "a" * 65, "a" * 63 + " "],
    )
    async def test_init_rejects_malformed_sha256_before_db_write(
        self, monkeypatch, tmp_path, digest
    ):
        class _DB:
            async def create_upload_session(self, **_kwargs):
                raise AssertionError("invalid digest reached database creation")

        async def _get_db():
            return _DB()

        class _Req:
            state = type("s", (), {"user_id": "u"})()

            async def json(self):
                return {
                    "file_id": "p1",
                    "filename": "test.h5ad",
                    "expected_sha256": digest,
                    "total_chunks": 2,
                    "chunk_size": 10,
                    "total_size": 11,
                }

        class _State:
            user_id = "u"
            data_files = [
                type(
                    "f", (),
                    {
                        "filename": "test.h5ad",
                        "file_path": "",
                        "id": "p1",
                        "section": "sequencing",
                        "group_id": None,
                        "sample_number": None,
                    },
                )()
            ]

        monkeypatch.setattr(up, "sessions", {"sid": _State()})
        monkeypatch.setattr(up, "UPLOAD_TEMP_DIR", tmp_path)
        monkeypatch.setattr(up, "get_db", _get_db)
        monkeypatch.setattr(up, "assert_generation_writable", lambda *_: None)

        with pytest.raises(HTTPException) as exc:
            await up.init_upload("sid", _Req())

        assert exc.value.status_code == 400
        assert "64-character hex digest" in exc.value.detail
