from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from backend.middleware import auth as auth_middleware


def _install_pending_user(monkeypatch):
    async def authenticate(_token):
        return "pending-user"

    async def get_user_by_id(_user_id):
        return SimpleNamespace(id="pending-user", status="pending", role="user")

    async def get_db():
        return SimpleNamespace(get_user_by_id=get_user_by_id)

    monkeypatch.setattr(auth_middleware, "authenticate_token", authenticate)
    monkeypatch.setattr(auth_middleware, "get_db", get_db)


def _request(path: str, *, method: str = "GET") -> Request:
    return Request({
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"authorization", b"Bearer existing-jwt")],
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1),
        "state": {},
    })


async def _call_next(_request):
    return PlainTextResponse("ok")


@pytest.mark.asyncio
async def test_pending_status_revokes_an_existing_jwt(monkeypatch):
    _install_pending_user(monkeypatch)
    middleware = auth_middleware.AuthMiddleware(_call_next)

    response = await middleware.dispatch(
        _request("/api/v1/test"),
        _call_next,
    )

    assert response.status_code == 403
    assert b"Account is pending approval" in response.body


@pytest.mark.asyncio
async def test_pending_status_still_allows_logout(monkeypatch):
    _install_pending_user(monkeypatch)
    middleware = auth_middleware.AuthMiddleware(_call_next)

    response = await middleware.dispatch(
        _request("/api/auth/logout", method="POST"),
        _call_next,
    )

    assert response.status_code == 200
