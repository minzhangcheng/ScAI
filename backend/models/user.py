"""User and API Key Pydantic models."""
from pydantic import BaseModel, Field, AfterValidator
from typing import Annotated, Optional
from datetime import datetime


def _max_72_bytes(v: str) -> str:
    if len(v.encode("utf-8")) > 72:
        raise ValueError("Password must be at most 72 bytes")
    return v


UsernameStr = Annotated[str, Field(min_length=3, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')]
PasswordStr = Annotated[str, Field(min_length=8, max_length=128), AfterValidator(_max_72_bytes)]


class User(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    password_hash: str = ""
    created_at: str = ""
    role: str = "user"
    status: str = "pending"


class ApiKey(BaseModel):
    id: str
    user_id: str
    hashed_key: str
    label: str = ""
    created_at: str = ""
    last_used_at: Optional[str] = None
