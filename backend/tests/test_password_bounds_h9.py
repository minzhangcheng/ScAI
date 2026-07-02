"""Tests that admin/setup password paths enforce the 72-byte limit."""
import pytest
from pydantic import BaseModel
from backend.routes.admin import SetPasswordRequest
from backend.routes.setup import SetupAdminRequest
from backend.utils.auth import hash_password


def test_admin_setpassword_rejects_over_72_bytes():
    with pytest.raises(Exception):
        SetPasswordRequest(new_password="a" * 73)
    with pytest.raises(Exception):
        SetPasswordRequest(new_password="喵" * 25)


def test_setup_admin_rejects_over_72_bytes():
    with pytest.raises(Exception):
        SetupAdminRequest(username="admin", password="a" * 73)


def test_hash_password_rejects_over_72_bytes():
    with pytest.raises(ValueError):
        hash_password("a" * 73)
