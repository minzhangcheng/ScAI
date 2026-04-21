"""Tests for session_store: save/load round-trip + list discovery."""
import asyncio
import json
import os

import pytest
import pytest_asyncio

os.environ.setdefault("SCAI_DATA_DIR", "/tmp/scai_test_data")

from backend.services.session_store import (
    save_session_to_disk, load_session_from_disk,
    async_save_session, ensure_session_loaded, list_saved_sessions,
)
from backend.models.schemas import AnalysisState
from backend.state import sessions
from backend.config import ensure_session_dirs


def _make_state(session_id: str, user_id: str = "u-1") -> AnalysisState:
    return AnalysisState(id=session_id, user_id=user_id, name="t")


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    sid = "sess-roundtrip"
    ensure_session_dirs(sid)
    state = _make_state(sid)
    save_session_to_disk(sid, state)
    loaded = load_session_from_disk(sid)
    assert loaded is not None
    assert loaded.id == sid
    assert loaded.user_id == "u-1"


def test_load_rejects_empty_user_id(tmp_path, monkeypatch):
    """Simulate a legacy session file with empty user_id (written before constraint was added)."""
    import tempfile
    from backend.config import SESSION_DIR, get_session_metadata_path
    sid = "sess-no-user"
    ensure_session_dirs(sid)
    # Write a raw JSON file directly (bypassing AnalysisState validation)
    session_file = get_session_metadata_path(sid)
    with open(session_file, "w") as f:
        json.dump({"id": sid, "user_id": "", "name": "t"}, f)
    assert load_session_from_disk(sid) is None


@pytest.mark.asyncio
async def test_ensure_session_loaded_caches():
    sid = "sess-cache"
    ensure_session_dirs(sid)
    save_session_to_disk(sid, _make_state(sid))
    sessions.pop(sid, None)
    first = await ensure_session_loaded(sid)
    assert first is not None
    assert sid in sessions
    second = await ensure_session_loaded(sid)
    assert second is first


def test_list_saved_sessions_sees_new_structure():
    sid = "sess-listed"
    ensure_session_dirs(sid)
    save_session_to_disk(sid, _make_state(sid))
    found = [s for s in list_saved_sessions() if s["id"] == sid]
    assert found and found[0]["user_id"] == "u-1"
