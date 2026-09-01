"""Validation helpers for user-configurable HTTP endpoints."""

from urllib.parse import urlparse


def validated_http_url(value: str, field_name: str, *, allow_empty: bool = False) -> str:
    """Return a trimmed HTTP(S) URL with an explicit host."""
    text = str(value or "").strip()
    if not text and allow_empty:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{field_name} must be a valid http or https URL")
    return text


def validated_proxy_url(value: str, field_name: str, *, allow_empty: bool = False) -> str:
    """Return a trimmed HTTP(S) or SOCKS5 proxy URL with an explicit host."""
    text = str(value or "").strip()
    if not text and allow_empty:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https", "socks5") or not parsed.netloc:
        raise ValueError(
            f"{field_name} must be a valid http, https, or socks5 URL"
        )
    return text
