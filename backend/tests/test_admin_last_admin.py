import asyncio

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


@pytest.mark.asyncio
async def test_concurrent_cross_delete_preserves_one_usable_admin(app_db):
    a = await app_db.create_user(
        username="delete-a", password_hash="h", role="admin", status="active",
    )
    b = await app_db.create_user(
        username="delete-b", password_hash="h", role="admin", status="active",
    )

    results = await asyncio.gather(
        app_db.mark_user_deleting_if_not_last_admin(b),
        app_db.mark_user_deleting_if_not_last_admin(a),
    )

    assert sorted(results) == ["last_admin", "updated"]
    users = [await app_db.get_user_by_id(a), await app_db.get_user_by_id(b)]
    usable = [
        user
        for user in users
        if user.role == "admin" and user.status in ("active", "readonly")
    ]
    assert len(usable) == 1
    assert await app_db.count_admins() == 1


@pytest.mark.asyncio
async def test_deleting_admin_does_not_license_last_usable_admin_demotion(app_db):
    active = await app_db.create_user(
        username="usable-admin", password_hash="h", role="admin", status="active",
    )
    await app_db.create_user(
        username="deleting-admin", password_hash="h", role="admin", status="deleting",
    )

    assert await app_db.demote_admin_if_not_last(active, "user") is False
    assert (await app_db.get_user_by_id(active)).role == "admin"
    assert await app_db.count_admins() == 1
