import asyncio
from uuid import uuid4
import sys
from importlib.metadata import version as package_version
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.models.schemas import AnalysisState, TaskStatus
from backend.routes.websockets import router
from backend.services.session_lifecycle import (
    SessionGenerationError,
    SessionGoneError,
)
from backend.utils.auth import ws_authenticate_and_load, ws_require_auth


def _version_tuple(distribution: str) -> tuple[int, ...]:
    return tuple(
        int(part) for part in package_version(distribution).split(".")
        if part.isdigit()
    )


_OLD_ANYIO_PY313_WEBSOCKET = (
    sys.version_info >= (3, 13) and _version_tuple("anyio") < (4, 11)
)


@pytest.fixture
def lifecycle_ws_client(monkeypatch):
    async def authenticate(_token: str):
        return "request-owner"

    monkeypatch.setattr("backend.utils.auth.authenticate_token", authenticate)
    db = SimpleNamespace(
        get_user_by_id=AsyncMock(return_value=SimpleNamespace(status="active"))
    )
    monkeypatch.setattr("backend.utils.auth.get_db", AsyncMock(return_value=db))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _connect(client: TestClient, session_id: str):
    return client.websocket_connect(
        f"/ws/{session_id}",
        subprotocols=["scai.jwt.v1", "test-token"],
    )


class _RecordingWebSocket:
    def __init__(self):
        self.headers = {
            "sec-websocket-protocol": "scai.jwt.v1, test-token",
        }
        self.events = []

    async def accept(self, *, subprotocol=None):
        self.events.append(("accept", subprotocol))

    async def send_json(self, payload):
        self.events.append(("message", payload))

    async def close(self, *, code=1000, reason=None):
        self.events.append(("close", code, reason))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_code", "expected_close"),
    [
        ("unknown", "SESSION_NOT_FOUND", 4404),
        ("foreign", "SESSION_NOT_FOUND", 4404),
        ("tombstoned_owner", "SESSION_GONE", 4410),
    ],
)
async def test_websocket_lifecycle_helper_contract(
    monkeypatch, mode, expected_code, expected_close,
):
    session_id = str(uuid4())

    async def authenticate(_token: str):
        return "request-owner"

    async def ensure_loaded(_session_id: str):
        if mode == "unknown":
            return None
        if mode == "foreign":
            return AnalysisState(id=session_id, user_id="different-owner")
        raise SessionGoneError(session_id, "request-owner", 3)

    monkeypatch.setattr("backend.utils.auth.authenticate_token", authenticate)
    db = SimpleNamespace(
        get_user_by_id=AsyncMock(return_value=SimpleNamespace(status="active"))
    )
    monkeypatch.setattr("backend.utils.auth.get_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        "backend.services.session_store.ensure_session_loaded",
        ensure_loaded,
    )
    websocket = _RecordingWebSocket()

    assert await ws_authenticate_and_load(websocket, session_id) is None
    assert websocket.events[0] == ("accept", "scai.jwt.v1")
    assert websocket.events[1][0] == "message"
    assert websocket.events[1][1]["code"] == expected_code
    assert websocket.events[1][1]["session_id"] == session_id
    assert websocket.events[2] == ("close", expected_close, expected_code)


@pytest.mark.asyncio
async def test_websocket_auth_rejects_deleting_user_before_accept(monkeypatch):
    async def authenticate(_token: str):
        return "deleting-owner"

    db = SimpleNamespace(
        get_user_by_id=AsyncMock(
            return_value=SimpleNamespace(status="deleting")
        )
    )
    monkeypatch.setattr("backend.utils.auth.authenticate_token", authenticate)
    monkeypatch.setattr("backend.utils.auth.get_db", AsyncMock(return_value=db))
    websocket = _RecordingWebSocket()

    assert await ws_require_auth(websocket) is None
    assert websocket.events == [("close", 4401, "Account unavailable")]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_name", ["analysis", "report"])
@pytest.mark.parametrize(
    ("error_kind", "expected_close", "expected_reason"),
    [
        ("gone", 4410, "SESSION_GONE"),
        ("generation", 4404, "SESSION_NOT_FOUND"),
    ],
)
async def test_registration_lifecycle_race_uses_frontend_close_codes(
    monkeypatch, endpoint_name, error_kind, expected_close, expected_reason,
):
    from backend.routes import websockets as websocket_routes

    session_id = str(uuid4())
    state = AnalysisState(id=session_id, user_id="request-owner")
    websocket = _RecordingWebSocket()
    monkeypatch.setattr(
        websocket_routes,
        "ws_authenticate_and_load",
        AsyncMock(return_value=("request-owner", state)),
    )
    monkeypatch.setattr(
        websocket_routes,
        "resolve_config_for_user",
        AsyncMock(return_value=object()),
    )

    def fail_generation_guard(_session_id, _state):
        if error_kind == "gone":
            raise SessionGoneError(session_id, "request-owner", 3)
        raise SessionGenerationError("session generation changed")

    monkeypatch.setattr(
        websocket_routes, "assert_generation_writable", fail_generation_guard,
    )
    endpoint = (
        websocket_routes.websocket_endpoint
        if endpoint_name == "analysis"
        else websocket_routes.report_websocket
    )

    await endpoint(websocket, session_id)

    assert websocket.events == [("close", expected_close, expected_reason)]



@pytest.mark.asyncio
async def test_force_delete_cancellation_never_restarts_report_producer(
    monkeypatch,
):
    from backend import state as state_mod
    from backend import workflow as workflow_module
    from backend.routes import websockets as websocket_routes
    from backend.services.session_lifecycle import get_lifecycle_lock

    session_id = str(uuid4())
    state = AnalysisState(
        id=session_id,
        user_id="request-owner",
        status=TaskStatus.PENDING,
    )
    state_mod.sessions[session_id] = state
    run_started = asyncio.Event()
    report_calls = 0

    class BlockingWebSocket(_RecordingWebSocket):
        async def receive_text(self):
            await asyncio.Event().wait()

    class Workflow:
        async def run(self, current, _session_id):
            current.status = TaskStatus.GENERATING_REPORT
            run_started.set()
            await asyncio.Event().wait()

        async def resume(self, current, _session_id):
            return await self.run(current, _session_id)

        async def generate_report(self, current, _session_id):
            nonlocal report_calls
            report_calls += 1
            return current

        async def close(self, **_kwargs):
            return True

    workflow = Workflow()
    connection_ref = object()
    save = AsyncMock()
    update = AsyncMock()
    unregister = AsyncMock()
    monkeypatch.setattr(
        websocket_routes,
        "ws_authenticate_and_load",
        AsyncMock(return_value=("request-owner", state)),
    )
    monkeypatch.setattr(
        websocket_routes,
        "resolve_config_for_user",
        AsyncMock(return_value=SimpleNamespace(max_rounds=3)),
    )
    monkeypatch.setattr(
        websocket_routes, "register_analysis_websocket",
        AsyncMock(return_value=connection_ref),
    )
    monkeypatch.setattr(
        websocket_routes, "send_connection_json", AsyncMock(return_value=True),
    )
    monkeypatch.setattr(websocket_routes, "unregister_websocket", unregister)
    monkeypatch.setattr(websocket_routes, "_async_save_session", save)
    monkeypatch.setattr(websocket_routes, "_update_session_cache", update)
    monkeypatch.setattr(
        websocket_routes, "assert_generation_writable", lambda *_args: None,
    )
    monkeypatch.setattr(
        "backend.services.session_store.assert_recovery_clear",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        workflow_module, "SingleCellAnalysisWorkflow", lambda **_kwargs: workflow,
    )

    handler = asyncio.create_task(
        websocket_routes.websocket_endpoint(BlockingWebSocket(), session_id),
    )
    try:
        await asyncio.wait_for(run_started.wait(), timeout=1)
        async with get_lifecycle_lock(session_id):
            async with state_mod.state_lock:
                assert state_mod.workflows.pop(session_id) is workflow
                state_mod.sessions.pop(session_id, None)
                run_task = state_mod._run_tasks.pop(session_id)
                monitor_task = state_mod._monitor_tasks.pop(session_id)
            run_task.cancel()
            monitor_task.cancel()

        await asyncio.wait_for(handler, timeout=2)

        assert report_calls == 0
        save.assert_not_awaited()
        update.assert_not_awaited()
        assert session_id not in state_mod._run_tasks
        assert session_id not in state_mod.workflows
    finally:
        if not handler.done():
            handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
        state_mod.sessions.pop(session_id, None)
        state_mod.workflows.pop(session_id, None)
        state_mod._run_tasks.pop(session_id, None)
        state_mod._monitor_tasks.pop(session_id, None)
        state_mod.release_initializing_session(session_id)


@pytest.mark.parametrize(
    ("mode", "expected_code", "expected_close"),
    [
        ("unknown", "SESSION_NOT_FOUND", 4404),
        ("foreign", "SESSION_NOT_FOUND", 4404),
        ("tombstoned_owner", "SESSION_GONE", 4410),
    ],
)
@pytest.mark.skipif(
    _OLD_ANYIO_PY313_WEBSOCKET,
    reason=(
        "Python 3.13 with AnyIO <4.11 hangs TestClient's blocking portal; "
        "the supported project venv carries a newer AnyIO"
    ),
)
def test_analysis_websocket_lifecycle_message_and_close_contract(
    monkeypatch,
    lifecycle_ws_client,
    mode,
    expected_code,
    expected_close,
):
    session_id = str(uuid4())

    async def ensure_loaded(_session_id: str):
        assert _session_id == session_id
        if mode == "unknown":
            return None
        if mode == "foreign":
            return AnalysisState(id=session_id, user_id="different-owner")
        raise SessionGoneError(session_id, "request-owner", 3)

    monkeypatch.setattr(
        "backend.services.session_store.ensure_session_loaded",
        ensure_loaded,
    )

    with _connect(lifecycle_ws_client, session_id) as websocket:
        message = websocket.receive_json()
        assert message == {
            "type": "session_error",
            "code": expected_code,
            "message": (
                "This session was deleted and cannot be restored."
                if mode == "tombstoned_owner"
                else "Session not found"
            ),
            "session_id": session_id,
            **({"generation": 3} if mode == "tombstoned_owner" else {}),
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
        assert closed.value.code == expected_close
