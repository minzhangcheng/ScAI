"""Tests for AppDB — must set SCAI_DATA_DIR to a writable dir before importing backend."""
import os
import sys

# Point data dir to /tmp before any backend import

import pytest
from pathlib import Path
from backend.utils.db import AppDB


@pytest.mark.asyncio
async def test_create_user_and_lookup(tmp_path):
    db = AppDB(tmp_path / "app.db")
    await db.init()
    uid = await db.create_user(email="a@b.com", password_hash="h")
    user = await db.get_user_by_email("a@b.com")
    assert user is not None
    assert user.id == uid
    assert user.email == "a@b.com"


@pytest.mark.asyncio
async def test_get_user_by_id(tmp_path):
    db = AppDB(tmp_path / "app.db")
    await db.init()
    uid = await db.create_user(email="x@y.com", password_hash="h2")
    user = await db.get_user_by_id(uid)
    assert user is not None
    assert user.email == "x@y.com"


@pytest.mark.asyncio
async def test_create_api_key_and_lookup(tmp_path):
    db = AppDB(tmp_path / "app.db")
    await db.init()
    uid = await db.create_user(email="a@b.com", password_hash="h")
    kid = await db.create_api_key(user_id=uid, hashed_key="hk", label="test")
    user_id = await db.lookup_user_by_hashed_key("hk")
    assert user_id == uid


@pytest.mark.asyncio
async def test_list_and_delete_api_keys(tmp_path):
    db = AppDB(tmp_path / "app.db")
    await db.init()
    uid = await db.create_user(email="a@b.com", password_hash="h")
    await db.create_api_key(user_id=uid, hashed_key="hk1", label="k1")
    await db.create_api_key(user_id=uid, hashed_key="hk2", label="k2")
    keys = await db.list_api_keys(uid)
    assert len(keys) == 2
    deleted = await db.delete_api_key(keys[0].id, uid)
    assert deleted is True
    keys_after = await db.list_api_keys(uid)
    assert len(keys_after) == 1


@pytest.mark.asyncio
async def test_lookup_nonexistent_key(tmp_path):
    db = AppDB(tmp_path / "app.db")
    await db.init()
    result = await db.lookup_user_by_hashed_key("nope")
    assert result is None
