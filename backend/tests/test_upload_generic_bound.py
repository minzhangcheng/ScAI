"""Tests for _stream_generic_upload bounding memory on non-sequencing uploads."""
import pytest
from backend.routes.upload import _stream_generic_upload


class _FakeUpload:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, n=-1):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_generic_upload_rejects_oversize(tmp_path):
    dest = tmp_path / "out.bin"
    up = _FakeUpload([b"a" * 10, b"b" * 10])
    with pytest.raises(Exception):
        await _stream_generic_upload(up, dest, max_bytes=15)


@pytest.mark.asyncio
async def test_generic_upload_ok_within_cap(tmp_path):
    dest = tmp_path / "out.bin"
    up = _FakeUpload([b"a" * 10])
    await _stream_generic_upload(up, dest, max_bytes=100)
    assert dest.read_bytes() == b"a" * 10
