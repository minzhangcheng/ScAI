"""Tests for _stream_generic_upload bounding memory on non-sequencing uploads."""
import asyncio

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
    assert not dest.exists()


@pytest.mark.asyncio
async def test_generic_upload_ok_within_cap(tmp_path):
    dest = tmp_path / "out.bin"
    up = _FakeUpload([b"a" * 10])
    await _stream_generic_upload(up, dest, max_bytes=100)
    assert dest.read_bytes() == b"a" * 10


@pytest.mark.asyncio
async def test_generic_upload_never_clobbers_preexisting_path(tmp_path):
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        await _stream_generic_upload(
            _FakeUpload([b"replacement"]), dest, max_bytes=100,
        )

    assert dest.read_bytes() == b"existing"


@pytest.mark.asyncio
async def test_generic_upload_cancellation_drains_then_removes_owned_path(tmp_path):
    dest = tmp_path / "out.bin"
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingUpload:
        calls = 0

        async def read(self, _size=-1):
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            entered.set()
            await release.wait()
            return b""

    upload = asyncio.create_task(
        _stream_generic_upload(BlockingUpload(), dest, max_bytes=100),
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    upload.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(upload, timeout=2)

    assert not dest.exists()
