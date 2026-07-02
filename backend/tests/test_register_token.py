import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.routes import auth as auth_routes
from backend.utils.db import AppDB


@pytest.mark.asyncio
async def test_register_returns_no_empty_token(app_db: AppDB):
    await app_db.set_meta("admin_initialized", "1")

    async def _get_db():
        return app_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        with patch.object(auth_routes, "get_db", side_effect=_get_db):
            r = await c.post(
                "/api/auth/register",
                json={
                    "username": "newuser",
                    "password": "password1",
                    "email": "newuser@x.io",
                },
            )
    assert r.status_code in (200, 201)
    body = r.json()
    assert body.get("access_token", None) not in ("",)
    assert (body.get("data", {}) or {}).get("status") == "pending"
