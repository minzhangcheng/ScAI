"""S3: inline import (zip/json) must be size-capped to prevent OOM."""
import pytest
from fastapi import HTTPException

from backend.routes.export_import import (
    MAX_INLINE_IMPORT_BYTES,
    check_inline_import_size,
    read_capped,
)


def test_size_check_passes_under_limit():
    check_inline_import_size(1024, max_bytes=2048)  # no raise


def test_size_check_rejects_over_limit():
    with pytest.raises(HTTPException) as exc:
        check_inline_import_size(4096, max_bytes=2048)
    assert exc.value.status_code == 413


def test_size_check_allows_missing_content_length():
    check_inline_import_size(None, max_bytes=2048)  # no raise — fall through to capped read


class _FakeUpload:
    """Minimal UploadFile stand-in returning fixed-size chunks."""
    def __init__(self, total_bytes: int, chunk: int = 1024):
        self._remaining = total_bytes
        self._chunk = chunk

    async def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        n = min(self._chunk, self._remaining)
        self._remaining -= n
        return b"x" * n


@pytest.mark.asyncio
async def test_read_capped_under_limit_returns_bytes():
    up = _FakeUpload(total_bytes=3000)
    data = await read_capped(up, max_bytes=8192)
    assert len(data) == 3000


@pytest.mark.asyncio
async def test_read_capped_over_limit_raises_413():
    up = _FakeUpload(total_bytes=10000)
    with pytest.raises(HTTPException) as exc:
        await read_capped(up, max_bytes=4096)
    assert exc.value.status_code == 413
