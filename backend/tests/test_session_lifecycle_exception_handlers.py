from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from backend.main import (
    session_generation_exception_handler,
    session_gone_exception_handler,
)
from backend.services.session_lifecycle import (
    SessionGenerationError,
    SessionGoneError,
)


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
