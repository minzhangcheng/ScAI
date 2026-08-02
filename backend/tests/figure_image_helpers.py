"""Small, real image/PDF payloads for artifact-integrity tests."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def valid_image_bytes(
    image_format: str = "PNG",
    *,
    color: tuple[int, int, int] = (12, 34, 56),
    min_bytes: int = 0,
    trailing: bytes = b"",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, image_format)
    payload = buffer.getvalue()
    if len(payload) < min_bytes:
        payload += b" " * (min_bytes - len(payload))
    return payload + trailing


def valid_pdf_bytes(*, color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, "PDF")
    return buffer.getvalue()


def write_valid_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (12, 34, 56),
    min_bytes: int = 0,
) -> bytes:
    image_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    payload = valid_image_bytes(
        image_format, color=color, min_bytes=min_bytes
    )
    path.write_bytes(payload)
    return payload
