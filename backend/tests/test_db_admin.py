"""Unit tests for AppDB.create_admin_if_missing."""
import logging

import pytest

from backend.utils.db import AppDB


@pytest.mark.asyncio
async def test_create_admin_inserts_when_missing(app_db):
    uid, created = await app_db.create_admin_if_missing(
        username="admin", password_hash="h1", email="a@x.com"
    )
    assert created is True
    user = await app_db.get_user_by_id(uid)
    assert user.username == "admin"
    assert user.email == "a@x.com"


@pytest.mark.asyncio
async def test_existing_admin_username_not_overwritten_but_email_updated(app_db, caplog):
    uid, _ = await app_db.create_admin_if_missing(
        username="admin", password_hash="h1", email="old@x.com"
    )
    with caplog.at_level(logging.WARNING):
        uid2, created = await app_db.create_admin_if_missing(
            username="alice", password_hash="h2", email="new@x.com"
        )
    assert created is False
    assert uid2 == uid
    user = await app_db.get_user_by_id(uid)
    assert user.username == "admin"
    assert user.email == "new@x.com"
    assert any("username" in r.message for r in caplog.records)