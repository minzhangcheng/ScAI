"""Database-level guards for account deletion state transitions."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.models.user import USER_STATUS_DELETING


@pytest.mark.asyncio
async def test_deleting_status_cannot_be_reactivated_by_guarded_update(app_db):
    uid = await app_db.create_user(
        username="deleting-user",
        password_hash="not-used",
        status=USER_STATUS_DELETING,
    )

    changed = await app_db.set_user_status(
        uid,
        "active",
        unless_current_status=USER_STATUS_DELETING,
    )

    assert changed is False
    assert (await app_db.get_user_by_id(uid)).status == USER_STATUS_DELETING


@pytest.mark.asyncio
async def test_deleting_status_cannot_be_reactivated_by_role_promotion(app_db):
    uid = await app_db.create_user(
        username="deleting-admin",
        password_hash="not-used",
        status=USER_STATUS_DELETING,
    )

    changed = await app_db.set_user_role(
        uid,
        "admin",
        status="active",
        unless_current_status=USER_STATUS_DELETING,
    )

    user = await app_db.get_user_by_id(uid)
    assert changed is False
    assert user.role == "user"
    assert user.status == USER_STATUS_DELETING


@pytest.mark.asyncio
async def test_deleting_status_rejects_guarded_password_update(app_db):
    uid = await app_db.create_user(
        username="deleting-password",
        password_hash="original",
        status=USER_STATUS_DELETING,
    )

    changed = await app_db.update_user_password(
        uid,
        "replacement",
        unless_current_status=USER_STATUS_DELETING,
    )

    assert changed is False
    assert (await app_db.get_user_by_id(uid)).password_hash == "original"


@pytest.mark.asyncio
async def test_deleting_admin_rejects_guarded_demotion(app_db):
    deleting_uid = await app_db.create_user(
        username="deleting-demotion",
        password_hash="not-used",
        role="admin",
        status=USER_STATUS_DELETING,
    )
    await app_db.create_user(
        username="active-admin",
        password_hash="not-used",
        role="admin",
        status="active",
    )

    changed = await app_db.demote_admin_if_not_last(
        deleting_uid,
        "user",
        unless_current_status=USER_STATUS_DELETING,
    )

    assert changed is False
    user = await app_db.get_user_by_id(deleting_uid)
    assert user.role == "admin"
    assert user.status == USER_STATUS_DELETING


@pytest.mark.asyncio
async def test_password_compare_and_swap_rejects_stale_verified_hash(app_db):
    uid = await app_db.create_user(
        username="password-cas",
        password_hash="old-hash",
        status="active",
    )

    assert await app_db.compare_and_swap_user_password(
        uid,
        "old-hash",
        "admin-reset",
        allowed_statuses=("active", "readonly"),
    ) is True
    assert await app_db.compare_and_swap_user_password(
        uid,
        "old-hash",
        "stale-self-service",
        allowed_statuses=("active", "readonly"),
    ) is False
    assert (await app_db.get_user_by_id(uid)).password_hash == "admin-reset"


@pytest.mark.asyncio
async def test_concurrent_promotion_and_pending_status_preserve_invariant(app_db):
    uid = await app_db.create_user(
        username="promotion-race",
        password_hash="not-used",
        status="active",
    )

    await asyncio.gather(
        app_db.set_user_status(
            uid,
            "pending",
            unless_current_status=USER_STATUS_DELETING,
            forbid_current_role="admin",
        ),
        app_db.set_user_role(
            uid,
            "admin",
            unless_current_status=USER_STATUS_DELETING,
            activate_pending_admin=True,
        ),
    )

    user = await app_db.get_user_by_id(uid)
    assert user.role == "admin"
    assert user.status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("revocation", ["demoted", "deleting"])
async def test_revoked_admin_cannot_commit_guarded_admin_writes(
    app_db,
    revocation,
):
    """Authorization is evaluated by the same statement/transaction as the write."""
    actor = await app_db.create_user(
        username=f"revoked-actor-{revocation}",
        password_hash="actor-hash",
        role="admin",
        status="active",
    )
    await app_db.create_user(
        username=f"keeper-{revocation}",
        password_hash="keeper-hash",
        role="admin",
        status="active",
    )
    target = await app_db.create_user(
        username=f"guarded-target-{revocation}",
        password_hash="target-hash",
        role="user",
        status="active",
    )
    admin_target = await app_db.create_user(
        username=f"guarded-admin-target-{revocation}",
        password_hash="target-hash",
        role="admin",
        status="active",
    )

    # This models revocation after require_admin() returned actor but before the
    # endpoint reached its database mutation.
    if revocation == "demoted":
        assert await app_db.set_user_role(actor, "user") is True
    else:
        assert await app_db.set_user_status(actor, "deleting") is True
    assert await app_db.is_usable_admin(actor) is False

    assert await app_db.set_user_status(
        target,
        "readonly",
        required_admin_id=actor,
    ) is False
    assert await app_db.set_user_role(
        target,
        "admin",
        activate_pending_admin=True,
        required_admin_id=actor,
    ) is False
    assert await app_db.update_user_password(
        target,
        "replacement",
        required_admin_id=actor,
    ) is False
    assert await app_db.demote_admin_if_not_last(
        admin_target,
        "user",
        required_admin_id=actor,
    ) is False
    assert await app_db.mark_user_deleting_if_not_last_admin(
        target,
        required_admin_id=actor,
    ) == "forbidden"
    assert await app_db.create_user(
        username=f"blocked-create-{revocation}",
        password_hash="new-hash",
        status="active",
        required_admin_id=actor,
    ) is None

    current = await app_db.get_user_by_id(target)
    assert current.role == "user"
    assert current.status == "active"
    assert current.password_hash == "target-hash"
    assert (await app_db.get_user_by_id(admin_target)).role == "admin"
    assert await app_db.get_user_by_username(
        f"blocked-create-{revocation}"
    ) is None


@pytest.mark.asyncio
async def test_route_rejects_admin_revoked_after_dependency_resolution(
    app_db,
    monkeypatch,
):
    from fastapi import HTTPException

    from backend.routes import admin

    actor = await app_db.create_user(
        username="resolved-admin",
        password_hash="actor-hash",
        role="admin",
        status="active",
    )
    target = await app_db.create_user(
        username="resolved-admin-target",
        password_hash="target-hash",
        role="user",
        status="active",
    )
    resolved_admin_id = actor  # Value already returned by require_admin().
    assert await app_db.set_user_role(actor, "user") is True
    monkeypatch.setattr(admin, "get_db", AsyncMock(return_value=app_db))

    with pytest.raises(HTTPException) as caught:
        await admin.set_user_status(
            target,
            admin.SetStatusRequest(status="readonly"),
            admin_id=resolved_admin_id,
        )

    assert caught.value.status_code == 403
    assert "revoked" in str(caught.value.detail).lower()
    assert (await app_db.get_user_by_id(target)).status == "active"
