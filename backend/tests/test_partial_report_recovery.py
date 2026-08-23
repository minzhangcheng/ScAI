"""Tests for partial report approval triggering report generation."""
import asyncio
import json
import uuid

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from backend.utils import auth as auth_module
from backend.utils.auth import create_access_token, hash_password
from backend.utils.db import AppDB
from backend.state import sessions, state_lock
from backend.models.schemas import AnalysisState, TaskStatus
from backend.services import workflow_runner as wr
from backend.routes.reports_parts import publication


def _state_bytes(state_obj: AnalysisState) -> bytes:
    return json.dumps(
        state_obj.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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
    session_id = f"test-partial-approve-{uuid.uuid4().hex}"

    called = {}
    async def fake_start_report(sid, *, expected_state=None, prepare=None):
        assert expected_state is state_obj
        if prepare is not None:
            await prepare()
        called["sid"] = sid
    monkeypatch.setattr(wr, "start_report_generation", fake_start_report)

    state_obj = AnalysisState(
        id=session_id,
        user_id=uid,
        status=TaskStatus.COMPLETED,
        allow_partial_report=False,
        lifecycle_generation=1,
        artifact_registry_version=1,
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


@pytest.mark.asyncio
async def test_partial_approval_without_bearer_is_401_and_does_not_mutate(
    client_and_auth, monkeypatch,
):
    uid, _headers = client_and_auth
    session_id = f"test-partial-no-bearer-{uuid.uuid4().hex}"
    state_obj = AnalysisState(
        id=session_id,
        user_id=uid,
        status=TaskStatus.COMPLETED,
        allow_partial_report=False,
    )
    async with state_lock:
        sessions[session_id] = state_obj
    before = _state_bytes(state_obj)
    runner = AsyncMock()
    monkeypatch.setattr(wr, "start_report_generation", runner)

    try:
        from backend.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            response = await c.post(
                f"/api/session/{session_id}/report/partial/approve",
            )

        assert response.status_code == 401
        assert _state_bytes(sessions[session_id]) == before
        runner.assert_not_awaited()
    finally:
        async with state_lock:
            sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_partial_approval_invalid_bearer_is_401_and_does_not_mutate(
    client_and_auth, monkeypatch,
):
    uid, _headers = client_and_auth
    session_id = f"test-partial-invalid-bearer-{uuid.uuid4().hex}"
    state_obj = AnalysisState(
        id=session_id,
        user_id=uid,
        status=TaskStatus.COMPLETED,
        allow_partial_report=False,
    )
    async with state_lock:
        sessions[session_id] = state_obj
    before = _state_bytes(state_obj)
    runner = AsyncMock()
    monkeypatch.setattr(wr, "start_report_generation", runner)

    try:
        from backend.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            response = await c.post(
                f"/api/session/{session_id}/report/partial/approve",
                headers={"Authorization": "Bearer definitely-invalid"},
            )

        assert response.status_code == 401
        assert _state_bytes(sessions[session_id]) == before
        runner.assert_not_awaited()
    finally:
        async with state_lock:
            sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_partial_approval_non_owner_jwt_is_denied_without_mutation(
    client_and_auth, monkeypatch,
):
    uid, _headers = client_and_auth
    other_uid = await auth_module._db.create_user(
        "other-user",
        hash_password("x"),
        email=f"other-{uuid.uuid4().hex}@x.io",
        role="user",
        status="active",
    )
    other_headers = {"Authorization": f"Bearer {create_access_token(other_uid, 'other-user')}"}
    session_id = f"test-partial-non-owner-{uuid.uuid4().hex}"
    state_obj = AnalysisState(
        id=session_id,
        user_id=uid,
        status=TaskStatus.COMPLETED,
        allow_partial_report=False,
        lifecycle_generation=1,
        artifact_registry_version=1,
    )
    async with state_lock:
        sessions[session_id] = state_obj
    before = _state_bytes(state_obj)
    runner = AsyncMock()
    monkeypatch.setattr(wr, "start_report_generation", runner)

    try:
        from backend.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            response = await c.post(
                f"/api/session/{session_id}/report/partial/approve",
                headers=other_headers,
            )

        assert response.status_code == 404
        after = _state_bytes(sessions[session_id])
        assert after == before, {
            key: (json.loads(before).get(key), json.loads(after).get(key))
            for key in json.loads(before)
            if json.loads(before).get(key) != json.loads(after).get(key)
        }
        runner.assert_not_awaited()
    finally:
        async with state_lock:
            sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_partial_approval_save_failure_leaves_original_state_and_no_owner(
    monkeypatch,
):
    session_id = "test-partial-save-failure"
    state_obj = AnalysisState(
        id=session_id,
        user_id="owner",
        status=TaskStatus.COMPLETED,
        allow_partial_report=False,
    )
    sessions[session_id] = state_obj
    monkeypatch.setattr(
        publication,
        "require_session_ownership",
        AsyncMock(return_value=state_obj),
    )
    monkeypatch.setattr(
        publication,
        "_async_save_session_under_lifecycle",
        AsyncMock(side_effect=OSError("disk full")),
    )
    monkeypatch.setattr(wr, "assert_generation_writable", lambda *_args: None)
    monkeypatch.setattr(
        publication, "assert_generation_writable", lambda *_args: None,
    )
    report_runner = AsyncMock()
    monkeypatch.setattr(wr, "_run_report_generation", report_runner)
    request = MagicMock()
    request.state.user_id = "owner"

    try:
        with pytest.raises(OSError, match="disk full"):
            await publication.approve_partial_report(session_id, request, "owner")
        await asyncio.sleep(0)

        assert sessions.get(session_id) is state_obj
        assert state_obj.status == TaskStatus.COMPLETED
        assert state_obj.allow_partial_report is False
        assert state_obj.partial_report_requested_at is None
        assert session_id not in wr._report_tasks
        report_runner.assert_not_awaited()
    finally:
        sessions.pop(session_id, None)
        task = wr._report_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
