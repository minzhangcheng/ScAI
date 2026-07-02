import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.utils import auth as auth_module
from backend.utils.auth import create_access_token, hash_password
from backend.utils.db import AppDB

import backend.routes.sessions as sessions_mod
import backend.routes.lifecycle as lifecycle_mod

# Contains `!` which fails validate_session_id's whitelist regex ^[a-zA-Z0-9_-]+$
# but doesn't cause URL path normalization (avoids 404 before routing).
BAD_ID = "bad!id"


@pytest_asyncio.fixture
async def client_and_auth(tmp_path):
    """Wire up an isolated AppDB into the auth module so the middleware
    finds users in the test database, not the real one."""
    old_db = auth_module._db

    db_path = tmp_path / "test_sid_validation.db"
    new_db = AppDB(db_path)
    await new_db.init()
    auth_module._db = new_db

    pw = hash_password("x")
    uid = await new_db.create_user(
        "tuser", pw, email="t@x.io", role="user", status="active"
    )
    token = create_access_token(uid, "tuser")

    yield uid, {"Authorization": f"Bearer {token}"}

    auth_module._db = old_db


@pytest.fixture(autouse=True)
def _bypass_ownership(monkeypatch):
    async def _ok(session_id=None, user_id=None, **k):
        raise AssertionError("ownership reached before validate_session_id")
    monkeypatch.setattr(sessions_mod, "require_session_ownership", _ok)
    monkeypatch.setattr(lifecycle_mod, "require_session_ownership", _ok)


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", [
    ("delete", f"/api/session/{BAD_ID}",                          None),
    ("post",   f"/api/session/{BAD_ID}/run",                      None),
    ("post",   f"/api/session/{BAD_ID}/pause",                    None),
    ("post",   f"/api/session/{BAD_ID}/resume",                   None),
    ("post",   f"/api/session/{BAD_ID}/stop",                     {"action": "immediate"}),
    ("post",   f"/api/session/{BAD_ID}/resume-bioinformatics",    None),
    ("post",   f"/api/session/{BAD_ID}/llm-failure-action",      {"action": "retry", "node_name": "x", "generation": 0}),
    ("post",   f"/api/session/{BAD_ID}/retry-node",              {"node_name": "x", "skip": False}),
    ("get",    f"/api/session/{BAD_ID}/failed-node",             None),
])
async def test_endpoints_reject_bad_session_id(method, path, body, client_and_auth):
    _, headers = client_and_auth
    from backend.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        if body is not None:
            r = await getattr(c, method)(path, headers=headers, json=body)
        else:
            r = await getattr(c, method)(path, headers=headers)
    assert r.status_code == 400, f"{method} {path} -> {r.status_code} (expected 400 from validate_session_id)"
