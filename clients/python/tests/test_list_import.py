"""Tests for list_sessions / import_session client methods."""
import pytest
from unittest.mock import patch, MagicMock

from scai_client import Client


def test_import_failed_is_scai_error():
    from scai_client import ImportFailed, ScaiError
    assert issubclass(ImportFailed, ScaiError)


def test_import_failed_carries_message():
    from scai_client import ImportFailed
    e = ImportFailed("import_abc", "bad archive")
    assert "import_abc" in str(e)
    assert "bad archive" in str(e)


# --- list_sessions ---------------------------------------------------------


def test_list_sessions_returns_list():
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "sessions": [
                {"id": "s1", "name": "demo", "status": "completed", "current_round": 3},
                {"id": "s2", "name": "wip", "status": "running", "current_round": 1},
            ]
        }
        with patch.object(c._http, "get", return_value=mock_resp) as mock_get:
            sessions = c.list_sessions()
            mock_get.assert_called_once_with("/api/sessions")
            assert len(sessions) == 2
            assert sessions[0]["id"] == "s1"
            assert sessions[1]["status"] == "running"


def test_list_sessions_auth_error():
    from scai_client import AuthError
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch.object(c._http, "get", return_value=mock_resp):
            with pytest.raises(AuthError):
                c.list_sessions()


# --- import_session --------------------------------------------------------


def test_import_session_zip_returns_session_id(tmp_path):
    archive = tmp_path / "session.zip"
    archive.write_bytes(b"PK\x03\x04fake-zip-bytes")

    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "session_id": "imported-123",
            "data": {"name": "imported"},
        }
        with patch.object(c._http, "post", return_value=mock_resp) as mock_post:
            sid = c.import_session(archive)
            assert sid == "imported-123"
            args, kwargs = mock_post.call_args
            assert args[0] == "/api/session/import"
            assert "files" in kwargs


def test_import_session_missing_file_raises(tmp_path):
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        with pytest.raises(FileNotFoundError):
            c.import_session(tmp_path / "does-not-exist.tar.gz")


def test_import_session_tarball_polls_until_completed(tmp_path):
    archive = tmp_path / "session.tar.gz"
    archive.write_bytes(b"\x1f\x8bfake-tgz")

    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {
            "success": True,
            "data": {"import_id": "imp-1", "status": "pending"},
        }

        get_running = MagicMock()
        get_running.status_code = 200
        get_running.raise_for_status = MagicMock()
        get_running.json.return_value = {"import_id": "imp-1", "status": "pending"}

        get_done = MagicMock()
        get_done.status_code = 200
        get_done.raise_for_status = MagicMock()
        get_done.json.return_value = {
            "import_id": "imp-1", "status": "completed", "session_id": "sess-9",
        }

        with patch.object(c._http, "post", return_value=post_resp), \
             patch.object(c._http, "get", side_effect=[get_running, get_done]), \
             patch("time.sleep"):
            sid = c.import_session(archive, poll_interval=0.01)
            assert sid == "sess-9"


def test_import_session_tarball_failed_raises(tmp_path):
    from scai_client import ImportFailed
    archive = tmp_path / "session.tar.gz"
    archive.write_bytes(b"\x1f\x8bfake")

    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"success": True, "data": {"import_id": "imp-2"}}

        get_failed = MagicMock()
        get_failed.status_code = 200
        get_failed.raise_for_status = MagicMock()
        get_failed.json.return_value = {
            "import_id": "imp-2", "status": "failed", "error": "corrupt archive",
        }

        with patch.object(c._http, "post", return_value=post_resp), \
             patch.object(c._http, "get", return_value=get_failed), \
             patch("time.sleep"):
            with pytest.raises(ImportFailed):
                c.import_session(archive, poll_interval=0.01)


def test_import_session_wait_false_returns_import_id(tmp_path):
    archive = tmp_path / "s.tar.gz"
    archive.write_bytes(b"\x1f\x8bfake")
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"success": True, "data": {"import_id": "imp-7"}}
        with patch.object(c._http, "post", return_value=post_resp):
            assert c.import_session(archive, wait=False) == "imp-7"


def test_import_session_auth_error(tmp_path):
    from scai_client import AuthError
    archive = tmp_path / "s.zip"
    archive.write_bytes(b"PK")
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch.object(c._http, "post", return_value=mock_resp):
            with pytest.raises(AuthError):
                c.import_session(archive)


def test_import_session_auth_error_during_poll(tmp_path):
    from scai_client import AuthError
    archive = tmp_path / "s.tar.gz"
    archive.write_bytes(b"\x1f\x8bfake")

    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"success": True, "data": {"import_id": "imp-a"}}

        get_401 = MagicMock()
        get_401.status_code = 401

        with patch.object(c._http, "post", return_value=post_resp), \
             patch.object(c._http, "get", return_value=get_401):
            with pytest.raises(AuthError):
                c.import_session(archive, poll_interval=0.01)


def test_import_session_timeout_raises(tmp_path):
    archive = tmp_path / "s.tar.gz"
    archive.write_bytes(b"\x1f\x8bfake")

    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"success": True, "data": {"import_id": "imp-t"}}

        get_pending = MagicMock()
        get_pending.status_code = 200
        get_pending.raise_for_status = MagicMock()
        get_pending.json.return_value = {"import_id": "imp-t", "status": "pending"}

        with patch.object(c._http, "post", return_value=post_resp), \
             patch.object(c._http, "get", return_value=get_pending), \
             patch("time.sleep"), \
             patch("time.time", side_effect=[0, 0, 100]):
            with pytest.raises(TimeoutError):
                c.import_session(archive, poll_interval=0.01, timeout=1)
