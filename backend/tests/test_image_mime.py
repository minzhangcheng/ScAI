"""L4: detect_image_mime must verify the WEBP fourcc, not just the RIFF prefix,
so RIFF-container WAV/AVI are not misclassified as image/webp."""
from backend.utils.llm_client import detect_image_mime


def test_jpeg():
    assert detect_image_mime(b'\xff\xd8\xff\xe0\x00\x10') == "image/jpeg"


def test_png():
    assert detect_image_mime(b'\x89PNG\r\n\x1a\n\x00\x00\x00\x00') == "image/png"


def test_gif87a():
    assert detect_image_mime(b'GIF87a\x00\x00\x00\x00\x00\x00') == "image/gif"


def test_gif89a():
    assert detect_image_mime(b'GIF89a\x00\x00\x00\x00\x00\x00') == "image/gif"


def test_real_webp_is_detected():
    # RIFF....WEBP — bytes 0:4 == RIFF, bytes 8:12 == WEBP
    assert detect_image_mime(b'RIFF\x24\x00\x00\x00WEBP') == "image/webp"


def test_riff_wave_is_not_webp():
    # RIFF container that is actually WAV audio — must NOT be webp
    assert detect_image_mime(b'RIFF\x24\x00\x00\x00WAVE') == "image/png"


def test_riff_avi_is_not_webp():
    assert detect_image_mime(b'RIFF\x24\x00\x00\x00AVI ') == "image/png"


def test_too_short_riff_is_not_webp():
    # only 8 bytes — not enough to read the fourcc → fall back
    assert detect_image_mime(b'RIFF\x00\x00\x00\x00') == "image/png"


def test_unknown_defaults_to_png():
    assert detect_image_mime(b'\x00\x01\x02\x03') == "image/png"
