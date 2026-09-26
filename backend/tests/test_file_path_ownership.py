"""Tests for hardened validate_file_path_ownership (P0-2)."""
import pytest
from unittest.mock import MagicMock

from fastapi import HTTPException

from backend.utils import agent_llm
from backend.config._config import get_session_dir, get_session_uploads, get_session_workspace


def _make_request(user_id: str = "user-1"):
    req = MagicMock()
    req.state.user_id = user_id
    return req


@pytest.fixture
def patch_ownership(monkeypatch):
    """Replace require_session_ownership with an async stub that records calls."""
    calls = []

    async def fake(session_id, user_id):
        calls.append((session_id, user_id))
        return None

    monkeypatch.setattr(agent_llm, "require_session_ownership", fake)
    return calls


@pytest.mark.asyncio
async def test_double_sessions_segment_rejected(patch_ownership):
    """Path with TWO 'sessions/' segments must be rejected (nested attack)."""
    sid_own = "own-sid"
    # Attacker tries: their own uploads dir containing a 'sessions/<victim>/' tail
    upload_dir = get_session_uploads(sid_own)
    upload_dir.mkdir(parents=True, exist_ok=True)
    malicious = str(upload_dir / "sessions" / "victim-sid" / "secret.h5ad")

    with pytest.raises(HTTPException) as exc:
        await agent_llm.validate_file_path_ownership(malicious, _make_request())
    assert exc.value.status_code == 403
    assert patch_ownership == []


@pytest.mark.asyncio
async def test_path_without_sessions_segment_rejected(patch_ownership):
    """Path without any 'sessions/' segment must be rejected."""
    with pytest.raises(HTTPException) as exc:
        await agent_llm.validate_file_path_ownership("/etc/passwd", _make_request())
    assert exc.value.status_code == 403
    assert patch_ownership == []


@pytest.mark.asyncio
async def test_path_outside_session_dir_rejected(patch_ownership, tmp_path):
    """Path that resolves outside the session's allowed dirs must be rejected."""
    # Path traversal: looks valid but resolves elsewhere
    malicious = str(get_session_dir("some-sid") / ".." / ".." / "etc" / "passwd")
    with pytest.raises(HTTPException) as exc:
        await agent_llm.validate_file_path_ownership(malicious, _make_request())
    assert exc.value.status_code == 403
    assert patch_ownership == []


@pytest.mark.asyncio
async def test_legit_path_with_expected_session_id(patch_ownership):
    """When caller passes expected_session_id and path is inside its dir, ownership is checked."""
    sid = "legit-sid"
    workspace = get_session_workspace(sid)
    workspace.mkdir(parents=True, exist_ok=True)
    legit_file = str(workspace / "figure.png")

    await agent_llm.validate_file_path_ownership(
        legit_file, _make_request("user-1"), expected_session_id=sid
    )
    assert patch_ownership == [(sid, "user-1")]


@pytest.mark.asyncio
async def test_expected_session_id_mismatched_path_rejected(patch_ownership):
    """expected_session_id provided but path is outside that session's dirs → reject."""
    sid = "claimed-sid"
    # Path points to a totally different session
    other = str(get_session_workspace("other-sid") / "x.txt")
    with pytest.raises(HTTPException) as exc:
        await agent_llm.validate_file_path_ownership(
            other, _make_request("user-1"), expected_session_id=sid
        )
    assert exc.value.status_code == 403
    assert patch_ownership == []


@pytest.mark.asyncio
async def test_legit_path_without_expected_session_id(patch_ownership):
    """Single 'sessions/<sid>/' segment with file inside that session's dir → ownership checked."""
    sid = "auto-sid"
    workspace = get_session_workspace(sid)
    workspace.mkdir(parents=True, exist_ok=True)
    legit_file = str(workspace / "result.txt")

    await agent_llm.validate_file_path_ownership(legit_file, _make_request("user-2"))
    assert patch_ownership == [(sid, "user-2")]


@pytest.mark.asyncio
async def test_unauthenticated_legit_path_delegates_uuid_guard(patch_ownership):
    """No auth token (HTML-embedded <img>/download): a legit session path is allowed
    and ownership is delegated to require_session_ownership with user_id=None
    (UUID-guard), mirroring the auth-optional figure/report endpoints."""
    sid = "noauth-sid"
    workspace = get_session_workspace(sid)
    workspace.mkdir(parents=True, exist_ok=True)
    legit_file = str(workspace / "graphic_abstract.jpg")

    req = MagicMock()
    req.state.user_id = None
    await agent_llm.validate_file_path_ownership(legit_file, req)
    assert patch_ownership == [(sid, None)]


@pytest.mark.asyncio
async def test_unauthenticated_invalid_path_still_rejected(patch_ownership):
    """No auth token does not relax path-containment checks: a path with no
    'sessions/<sid>/' segment is still rejected before any ownership delegation."""
    req = MagicMock()
    req.state.user_id = None
    with pytest.raises(HTTPException) as exc:
        await agent_llm.validate_file_path_ownership("/etc/passwd", req)
    assert exc.value.status_code == 403
    assert patch_ownership == []
