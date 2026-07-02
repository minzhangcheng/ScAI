"""Tests for partial report approval triggering report generation."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.utils import auth as auth_module
from backend.utils.auth import create_access_token, hash_password
from backend.utils.db import AppDB
from backend.state import sessions, state_lock
from backend.models.schemas import AnalysisState, TaskStatus
from backend.services import workflow_runner as wr


@pytest_asyncio.fixture
async def client_and_auth(tmp_path):
    old_db = auth_module._db
    db_path = tmp_path / "test_partial_report.db"
    new_db = AppDB(db_path)
    await new_db.init()
    auth_module._db = new_db

    pw = hash_password("x")
    uid = await new_db.create_user("tuser", pw, email="t@x.io", role="user", status="active")
    token = create_access_token(uid, "tuser")

    yield uid, {"Authorization": f"Bearer {token}"}

    auth_module._db = old_db


@pytest.mark.asyncio
async def test_approve_partial_triggers_report_generation(client_and_auth, monkeypatch):
    uid, headers = client_and_auth
    session_id = "test-partial-approve"

    called = {}
    async def fake_start_report(sid):
        called["sid"] = sid
    monkeypatch.setattr(wr, "start_report_generation", fake_start_report)

    state_obj = AnalysisState(
        id=session_id,
        user_id=uid,
        status=TaskStatus.COMPLETED,
        allow_partial_report=False,
    )
    async with state_lock:
        sessions[session_id] = state_obj

    try:
        from backend.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post(
                f"/api/session/{session_id}/report/partial/approve",
                headers=headers,
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data.get("success") is True
        assert data.get("data", {}).get("needs_reconnect") is True

        assert called.get("sid") == session_id, f"Expected {session_id}, got {called.get('sid')}"
    finally:
        async with state_lock:
            sessions.pop(session_id, None)
