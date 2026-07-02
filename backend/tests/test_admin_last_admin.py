import pytest


@pytest.mark.asyncio
async def test_demote_last_admin_blocked(app_db):
    a = await app_db.create_user(username="admin1", password_hash="h", email="a@x.io", status="active")
    await app_db.set_user_role(a, "admin")
    changed = await app_db.demote_admin_if_not_last(a, "user")
    assert changed is False
    assert await app_db.count_admins() == 1


@pytest.mark.asyncio
async def test_demote_admin_allowed_when_others_exist(app_db):
    a = await app_db.create_user(username="a1", password_hash="h", email="a1@x.io", status="active")
    b = await app_db.create_user(username="a2", password_hash="h", email="a2@x.io", status="active")
    await app_db.set_user_role(a, "admin")
    await app_db.set_user_role(b, "admin")
    assert await app_db.demote_admin_if_not_last(a, "user") is True
    assert await app_db.count_admins() == 1
