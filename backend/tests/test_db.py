"""Tests for AppDB — must set SCAI_DATA_DIR to a writable dir before importing backend."""
import os
import sys

# Point data dir to /tmp before any backend import

import pytest


@pytest.mark.asyncio
async def test_create_user_and_lookup(app_db):
    uid = await app_db.create_user(username="testuser", password_hash="h")
    user = await app_db.get_user_by_username("testuser")
    assert user is not None
    assert user.id == uid
    assert user.username == "testuser"


@pytest.mark.asyncio
async def test_get_user_by_id(app_db):
    uid = await app_db.create_user(username="testuser2", password_hash="h2")
    user = await app_db.get_user_by_id(uid)
    assert user is not None
    assert user.username == "testuser2"


@pytest.mark.asyncio
async def test_create_api_key_and_lookup(app_db):
    uid = await app_db.create_user(username="testuser3", password_hash="h")
    kid = await app_db.create_api_key(user_id=uid, hashed_key="hk", label="test")
    user_id = await app_db.lookup_user_by_hashed_key("hk")
    assert user_id == uid


@pytest.mark.asyncio
async def test_list_and_delete_api_keys(app_db):
    uid = await app_db.create_user(username="testuser4", password_hash="h")
    await app_db.create_api_key(user_id=uid, hashed_key="hk1", label="k1")
    await app_db.create_api_key(user_id=uid, hashed_key="hk2", label="k2")
    keys = await app_db.list_api_keys(uid)
    assert len(keys) == 2
    deleted = await app_db.delete_api_key(keys[0].id, uid)
    assert deleted is True
    keys_after = await app_db.list_api_keys(uid)
    assert len(keys_after) == 1


@pytest.mark.asyncio
async def test_lookup_nonexistent_key(app_db):
    result = await app_db.lookup_user_by_hashed_key("nope")
    assert result is None
