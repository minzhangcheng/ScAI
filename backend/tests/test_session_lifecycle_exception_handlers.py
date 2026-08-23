from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from backend.main import (
    report_generation_exception_handler,
    session_generation_exception_handler,
    session_gone_exception_handler,
)
from backend.services.session_lifecycle import (
    SessionGenerationError,
    SessionGoneError,
)
from backend.services.report.output_generation import ReportGenerationError


def _request(user_id: str | None) -> Request:
    state = {} if user_id is None else {"user_id": user_id}
    return Request({"type": "http", "method": "POST", "path": "/", "state": state})


@pytest.mark.asyncio
async def test_session_gone_handler_returns_410_to_generation_owner():
    response = await session_gone_exception_handler(
        _request("owner"),
        SessionGoneError("session", "owner", 7),
    )

    assert response.status_code == 410
    assert json.loads(response.body)["detail"] == {
        "code": "SESSION_GONE",
        "message": "This session was deleted and cannot be restored.",
        "generation": 7,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("request_user_id", [None, "different-owner"])
async def test_session_gone_handler_keeps_tombstone_private(request_user_id):
    response = await session_gone_exception_handler(
        _request(request_user_id),
        SessionGoneError("session", "owner", 7),
    )

    assert response.status_code == 404
    assert json.loads(response.body)["detail"]["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_session_generation_handler_returns_reloadable_conflict():
    response = await session_generation_exception_handler(
        _request("owner"),
        SessionGenerationError("stale generation"),
    )

    assert response.status_code == 409
    assert json.loads(response.body)["detail"]["code"] == "SESSION_GENERATION_CHANGED"


@pytest.mark.asyncio
async def test_report_generation_handler_returns_stable_conflict_without_path_leak():
    response = await report_generation_exception_handler(
        _request("owner"),
        ReportGenerationError("unsafe path: /private/session/report"),
    )

    body = json.loads(response.body)
    assert response.status_code == 409
    assert body["detail"]["code"] == "REPORT_GENERATION_INVALID"
    assert "/private/session/report" not in body["detail"]["message"]


def test_owner_mismatch_has_precise_subtype_and_base_catch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from backend.services import session_lifecycle
    from backend.services.session_lifecycle import (
        SessionGenerationError,
        SessionOwnerMismatchError,
    )

    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_dir)

    state = SimpleNamespace(
        id="p1-owner-mismatch",
        user_id="owner-a",
        lifecycle_generation=0,
    )
    session_lifecycle.adopt_persisted_generation(state)
    intruder = SimpleNamespace(
        id=state.id,
        user_id="owner-b",
        lifecycle_generation=0,
    )

    with pytest.raises(SessionGenerationError) as caught:
        session_lifecycle.adopt_persisted_generation(intruder)
    assert type(caught.value) is SessionOwnerMismatchError
    assert isinstance(caught.value, SessionGenerationError)


@pytest.mark.asyncio
async def test_owner_mismatch_handler_is_distinct_and_non_disclosing():
    import backend.main as main
    from backend.services.session_lifecycle import SessionOwnerMismatchError

    response = await main.session_owner_mismatch_exception_handler(
        _request("owner-a"),
        SessionOwnerMismatchError("owner-a != owner-b"),
    )
    body = json.loads(response.body)
    assert response.status_code == 409
    assert body["detail"]["code"] == "SESSION_OWNER_MISMATCH"
    message = body["detail"]["message"]
    assert "owner-a" not in message
    assert "owner-b" not in message
    assert "generation" not in message.lower()
    assert main.app.exception_handlers[SessionOwnerMismatchError] is main.session_owner_mismatch_exception_handler
