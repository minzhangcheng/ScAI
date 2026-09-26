"""Integration tests: custom-username setup + role/status admin endpoints."""
import asyncio
import pytest
from fastapi.testclient import TestClient

from backend.routes.auth import _login_rate_limiter
from backend.utils import auth as auth_module
from backend.utils.db import AppDB


@pytest.fixture(autouse=True)
def _reset_login_limiter():
    _login_rate_limiter._events.clear()
    yield
    _login_rate_limiter._events.clear()


@pytest.fixture
def client(tmp_path):
    old_db = auth_module._db
    new_db = AppDB(tmp_path / "test_perms.db")
    loop = asyncio.new_event_loop()
    loop.run_until_complete(new_db.init())
    loop.close()
    auth_module._db = new_db
    from backend.main import app
    c = TestClient(app)
    yield c
    auth_module._db = old_db


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestStatusConstraints:
    def test_admin_cannot_be_set_pending(self, client):
        _setup_admin(client)
        token = _admin_token(client)
        # Create a normal user and promote to admin
        client.post("/api/admin/users", json={"username": "alice", "password": "alicepass123"}, headers=_auth(token))
        users = client.get("/api/admin/users", headers=_auth(token)).json()
        alice = next(u for u in users if u["username"] == "alice")
        client.patch(f"/api/admin/users/{alice['id']}/role", json={"role": "admin"}, headers=_auth(token))
        # Now alice is admin, setting to pending should be rejected
        resp = client.patch(f"/api/admin/users/{alice['id']}/status", json={"status": "pending"}, headers=_auth(token))
        assert resp.status_code == 400
        # Setting readonly / active should work
        assert client.patch(f"/api/admin/users/{alice['id']}/status", json={"status": "readonly"}, headers=_auth(token)).status_code == 200
        assert client.patch(f"/api/admin/users/{alice['id']}/status", json={"status": "active"}, headers=_auth(token)).status_code == 200


class TestRoleEndpoint:
    def _promote_setup(self, client):
        _setup_admin(client)
        token = _admin_token(client)
        client.post("/api/admin/users", json={"username": "carol", "password": "carolpass123"}, headers=_auth(token))
        carol = next(u for u in client.get("/api/admin/users", headers=_auth(token)).json() if u["username"] == "carol")
        return token, carol

    def test_admin_cannot_change_own_role(self, client):
        _setup_admin(client)
        token = _admin_token(client)
        me = next(u for u in client.get("/api/admin/users", headers=_auth(token)).json() if u["role"] == "admin")
        resp = client.patch(f"/api/admin/users/{me['id']}/role", json={"role": "user"}, headers=_auth(token))
        assert resp.status_code == 403


    def test_promote_pending_user_becomes_active(self, client):
        _setup_admin(client)
        token = _admin_token(client)
        # register creates pending users
        client.post("/api/auth/register", json={"username": "dave", "password": "davepass123"})
        dave = next(u for u in client.get("/api/admin/users", headers=_auth(token)).json() if u["username"] == "dave")
        assert dave["status"] == "pending"
        resp = client.patch(f"/api/admin/users/{dave['id']}/role", json={"role": "admin"}, headers=_auth(token))
        assert resp.status_code == 200
        dave2 = next(u for u in client.get("/api/admin/users", headers=_auth(token)).json() if u["username"] == "dave")
        assert dave2["role"] == "admin"
        assert dave2["status"] == "active"


    def test_cannot_delete_self(self, client):
        _setup_admin(client)
        token = _admin_token(client)
        me = next(u for u in client.get("/api/admin/users", headers=_auth(token)).json() if u["role"] == "admin")
        resp = client.delete(f"/api/admin/users/{me['id']}", headers=_auth(token))
        assert resp.status_code == 403


def _setup_admin(client, username="boss", password="adminpass123", email=None):
    body = {"username": username, "password": password}
    if email:
        body["email"] = email
    return client.post("/api/auth/setup/admin", json=body)


def _admin_token(client, username="boss", password="adminpass123"):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    return r.json()["access_token"]
