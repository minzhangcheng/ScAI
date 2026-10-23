"""Username changes preserve identity and enforce account authority atomically."""

import asyncio
from sqlite3 import IntegrityError

import pytest
from fastapi.testclient import TestClient

from backend.routes.auth import _login_rate_limiter
from backend.utils import auth
from backend.utils.db import AppDB


@pytest.fixture(scope="module")
def password_hash():
    return auth.hash_password("password123")


@pytest.fixture
def accounts(tmp_path, monkeypatch, password_hash):
    db = AppDB(tmp_path / "users.db")

    async def seed():
        await db.init()
        result = {}
        for name, role in (("boss", "admin"), ("alice", "user")):
            uid = await db.create_user(username=name, password_hash=password_hash, role=role, status="active")
            result[name] = await db.get_user_by_id(uid)
        return result

    users = asyncio.run(seed())
    monkeypatch.setattr(auth, "_db", db)
    _login_rate_limiter._events.clear()
    from backend.main import app
    yield TestClient(app), db, users
    _login_rate_limiter._events.clear()


def headers(user):
    return {"Authorization": f"Bearer {auth.create_access_token(user.id, user.username)}"}


@pytest.mark.parametrize("status", ["active", "readonly"])
def test_self_rename_preserves_credentials_and_user_data(accounts, status):
    client, db, users = accounts
    user = users["alice"]
    original_headers = headers(user)
    raw_key, hashed_key = auth.generate_api_key()
    asyncio.run(db.create_api_key(user.id, hashed_key, "existing-key"))
    asyncio.run(db.set_user_llm_config(user.id, "existing-secret", "https://example.test/v1", ""))
    config_before = asyncio.run(db.get_user_llm_config(user.id))
    asyncio.run(db.set_user_status(user.id, status))
    before = asyncio.run(db.get_user_by_id(user.id)).model_dump()

    response = client.post("/api/auth/profile/username", headers=original_headers,
                           json={"current_password": "password123", "new_username": "alice-renamed"})
    assert response.status_code == 200, response.text
    token = auth.decode_access_token(response.json()["access_token"])
    assert (token["sub"], token["username"]) == (user.id, "alice-renamed")
    assert asyncio.run(db.get_user_by_id(user.id)).model_dump() == {**before, "username": "alice-renamed"}
    assert asyncio.run(db.get_user_llm_config(user.id)) == config_before
    for credential in (original_headers, {"Authorization": f"Bearer {raw_key}"}):
        profile = client.get("/api/auth/me", headers=credential)
        assert profile.status_code == 200
        assert (profile.json()["id"], profile.json()["username"]) == (user.id, "alice-renamed")
    assert client.post("/api/auth/login", json={"username": "alice", "password": "password123"}).status_code == 401
    login = client.post("/api/auth/login", json={"username": "alice-renamed", "password": "password123"})
    assert login.status_code == 200
    assert login.json()["id"] == user.id


@pytest.mark.parametrize("username,password,code", [
    ("renamed", "wrong", 400), ("boss", "password123", 409),
    ("ab", "password123", 422), ("bad name", "password123", 422),
    ("x" * 65, "password123", 422),
])
def test_self_rename_rejects_invalid_requests(accounts, username, password, code):
    client, db, users = accounts
    response = client.post("/api/auth/profile/username", headers=headers(users["alice"]),
                           json={"new_username": username, "current_password": password})
    assert response.status_code == code
    assert asyncio.run(db.get_user_by_id(users["alice"].id)).username == "alice"


@pytest.mark.parametrize("role,status", [("user", "active"), ("user", "readonly"), ("user", "pending"), ("admin", "active")])
def test_admin_can_rename_other_accounts(accounts, role, status):
    client, db, users = accounts
    uid = users["alice"].id
    asyncio.run(db.set_user_role(uid, role))
    asyncio.run(db.set_user_status(uid, status))
    before = asyncio.run(db.get_user_by_id(uid)).model_dump()
    response = client.patch(f"/api/admin/users/{uid}/username", headers=headers(users["boss"]),
                            json={"new_username": "renamed"})
    assert response.status_code == 200
    assert response.json()["access_token"] is None
    assert asyncio.run(db.get_user_by_id(uid)).model_dump() == {**before, "username": "renamed"}


def test_admin_self_rename_and_name_conflict(accounts):
    client, db, users = accounts
    uid = users["boss"].id
    original_headers = headers(users["boss"])
    path = f"/api/admin/users/{uid}/username"
    assert client.patch(path, headers=original_headers, json={"new_username": "alice"}).status_code == 409
    assert client.patch(path, headers=original_headers, json={"new_username": "bad name"}).status_code == 422
    response = client.patch(path, headers=original_headers, json={"new_username": "new-boss"})
    assert response.status_code == 200
    assert auth.decode_access_token(response.json()["access_token"])["sub"] == uid
    assert client.get("/api/admin/users", headers=original_headers).status_code == 200
    assert asyncio.run(db.get_user_by_id(uid)).username == "new-boss"


def test_rename_requires_authentication_and_admin_role(accounts):
    client, db, users = accounts
    path = f"/api/admin/users/{users['boss'].id}/username"
    assert client.patch(path, json={"new_username": "renamed"}).status_code == 401
    assert client.patch(path, headers=headers(users["alice"]), json={"new_username": "renamed"}).status_code == 403
    assert client.post("/api/auth/profile/username", json={"new_username": "renamed", "current_password": "password123"}).status_code == 401
    assert client.patch("/api/admin/users/missing/username", headers=headers(users["boss"]), json={"new_username": "renamed"}).status_code == 404
    assert asyncio.run(db.get_user_by_id(users["boss"].id)).username == "boss"


@pytest.mark.parametrize("status", ["pending", "deleting"])
def test_unavailable_user_cannot_rename_self(accounts, status):
    client, db, users = accounts
    uid = users["alice"].id
    asyncio.run(db.set_user_status(uid, status))
    response = client.post("/api/auth/profile/username", headers=headers(users["alice"]),
                           json={"new_username": "renamed", "current_password": "password123"})
    assert response.status_code == 403
    if status == "deleting":
        response = client.patch(f"/api/admin/users/{uid}/username", headers=headers(users["boss"]),
                                json={"new_username": "renamed"})
        assert response.status_code == 409
    assert asyncio.run(db.get_user_by_id(uid)).username == "alice"


@pytest.mark.parametrize("change,code", [("password", 409), ("username", 409), ("deleting", 403)])
def test_self_rename_rechecks_verified_account_at_write(accounts, monkeypatch, change, code):
    client, db, users = accounts
    uid = users["alice"].id
    original = db.update_user_username

    async def raced_update(*args, **kwargs):
        if change == "password":
            await db.update_user_password(uid, "new-password-hash")
        elif change == "username":
            await original(uid, "changed-first", expected_username="alice", allowed_statuses=("active",))
        else:
            await db.set_user_status(uid, "deleting")
        return await original(*args, **kwargs)

    monkeypatch.setattr(db, "update_user_username", raced_update)
    response = client.post("/api/auth/profile/username", headers=headers(users["alice"]),
                           json={"new_username": "renamed", "current_password": "password123"})
    assert response.status_code == code
    assert asyncio.run(db.get_user_by_id(uid)).username != "renamed"


@pytest.mark.parametrize("change,code", [("admin_demoted", 403), ("target_deleting", 409)])
def test_admin_rename_rechecks_authority_at_write(accounts, monkeypatch, change, code):
    client, db, users = accounts
    original = db.update_user_username

    async def raced_update(*args, **kwargs):
        if change == "admin_demoted":
            await db.set_user_role(users["boss"].id, "user")
        else:
            await db.set_user_status(users["alice"].id, "deleting")
        return await original(*args, **kwargs)

    monkeypatch.setattr(db, "update_user_username", raced_update)
    response = client.patch(f"/api/admin/users/{users['alice'].id}/username", headers=headers(users["boss"]),
                            json={"new_username": "renamed"})
    assert response.status_code == code
    assert asyncio.run(db.get_user_by_id(users["alice"].id)).username == "alice"


@pytest.mark.asyncio
async def test_concurrent_renames_cannot_claim_same_username(app_db):
    ids = [await app_db.create_user(username=name, password_hash="unused", status="active") for name in ("first", "second")]
    results = await asyncio.gather(*(
        app_db.update_user_username(uid, "shared", expected_username=name, allowed_statuses=("active",))
        for uid, name in zip(ids, ("first", "second"))
    ), return_exceptions=True)
    assert sum(result is True for result in results) == 1
    assert sum(isinstance(result, IntegrityError) for result in results) == 1
