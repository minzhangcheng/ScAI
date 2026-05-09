"""Tests for Client.export_session()."""
import httpx
import pytest

from scai_client import Client
from scai_client.errors import AuthError, ScaiError


def _make_client(handler, base="http://test"):
    transport = httpx.MockTransport(handler)
    c = Client(base_url=base, api_key="scai_test")
    c._http.close()  # close the real client before replacing with mock
    c._http = httpx.Client(
        base_url=base,
        headers={"Authorization": "Bearer scai_test"},
        transport=transport,
    )
    return c


def test_export_session_streams_to_file(tmp_path):
    captured = {}

    def handler(req):
        captured["url"] = str(req.url)
        return httpx.Response(200, content=b"chunk1chunk2")

    with _make_client(handler) as c:
        dest = tmp_path / "out.tar.gz"
        result = c.export_session("abc12345-xxxx", dest)
        assert result == dest
        assert dest.read_bytes() == b"chunk1chunk2"
        assert captured["url"].endswith("/api/session/abc12345-xxxx/export")
        assert not (tmp_path / "out.tar.gz.partial").exists()


def test_export_session_default_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def handler(req):
        return httpx.Response(200, content=b"data")

    with _make_client(handler) as c:
        result = c.export_session("abc12345-xxxx")
        assert result == tmp_path / "abc12345_export.tar.gz"
        assert result.read_bytes() == b"data"


def test_export_session_directory_arg(tmp_path):
    def handler(req):
        return httpx.Response(200, content=b"x")

    with _make_client(handler) as c:
        result = c.export_session("abc12345-xxxx", tmp_path)
        assert result == tmp_path / "abc12345_export.tar.gz"


def test_export_session_creates_parent_dirs(tmp_path):
    def handler(req):
        return httpx.Response(200, content=b"x")

    with _make_client(handler) as c:
        dest = tmp_path / "nested" / "sub" / "export.tar.gz"
        result = c.export_session("abc12345-xxxx", dest)
        assert result == dest
        assert dest.read_bytes() == b"x"


def test_export_session_auth_error(tmp_path):
    def handler(req):
        return httpx.Response(401)

    with _make_client(handler) as c:
        with pytest.raises(AuthError):
            c.export_session("abc12345-xxxx", tmp_path / "x.tar.gz")
        assert not (tmp_path / "x.tar.gz.partial").exists()


def test_export_session_forbidden(tmp_path):
    def handler(req):
        return httpx.Response(403)

    with _make_client(handler) as c:
        with pytest.raises(ScaiError, match="abc12345-xxxx"):
            c.export_session("abc12345-xxxx", tmp_path / "x.tar.gz")


def test_export_session_not_found(tmp_path):
    def handler(req):
        return httpx.Response(404)

    with _make_client(handler) as c:
        with pytest.raises(ScaiError, match="abc12345-xxxx"):
            c.export_session("abc12345-xxxx", tmp_path / "x.tar.gz")


def test_export_session_http_error(tmp_path):
    def handler(req):
        return httpx.Response(500)

    with _make_client(handler) as c:
        with pytest.raises(httpx.HTTPStatusError):
            c.export_session("abc12345-xxxx", tmp_path / "x.tar.gz")
        # partial cleaned up on failure
        assert not (tmp_path / "x.tar.gz.partial").exists()


def test_export_session_empty_archive(tmp_path):
    def handler(req):
        return httpx.Response(200, content=b"")

    with _make_client(handler) as c:
        dest = tmp_path / "empty.tar.gz"
        result = c.export_session("abc12345-xxxx", dest)
        assert result == dest
        assert dest.stat().st_size == 0
