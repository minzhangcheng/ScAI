"""Test byte-aware password validation (H9, H11)."""
import pytest
import bcrypt
from pydantic import BaseModel
from backend.models.user import PasswordStr
from backend.utils.auth import verify_password


class _M(BaseModel):
    pw: PasswordStr


def test_rejects_over_72_bytes():
    with pytest.raises(Exception):
        _M(pw="a" * 73)                 # 73 ASCII bytes
    with pytest.raises(Exception):
        _M(pw="喵" * 25)                # 25*3 = 75 bytes


def test_accepts_72_bytes():
    _M(pw="a" * 72)


def test_verify_backward_compatible():
    long_pw = "a" * 100
    legacy_hash = bcrypt.hashpw(long_pw.encode()[:72], bcrypt.gensalt()).decode()
    assert verify_password(long_pw, legacy_hash) is True
