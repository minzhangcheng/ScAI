"""S3: inline import (zip/json) must be size-capped to prevent OOM.

Inline read_capped() was removed when the import route switched to raw
octet-stream streaming. The size cap is now enforced during streaming inside
``import_state``. ``check_inline_import_size`` is still used as the fast-reject
Content-Length pre-check before reading the body.
"""
import pytest
from fastapi import HTTPException

from backend.routes.export_import import (
    MAX_INLINE_IMPORT_BYTES,
    check_inline_import_size,
)


def test_size_check_passes_under_limit():
    check_inline_import_size(1024, max_bytes=2048)  # no raise


def test_size_check_rejects_over_limit():
    with pytest.raises(HTTPException) as exc:
        check_inline_import_size(4096, max_bytes=2048)
    assert exc.value.status_code == 413


def test_size_check_allows_missing_content_length():
    check_inline_import_size(None, max_bytes=2048)  # no raise — fall through to capped read


def test_max_inline_import_bytes_constant_exposed():
    # Other code (tests, callers) imports this constant; pin that it stays an int >0.
    assert isinstance(MAX_INLINE_IMPORT_BYTES, int) and MAX_INLINE_IMPORT_BYTES > 0
