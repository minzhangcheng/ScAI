"""User and API Key Pydantic models."""
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from datetime import datetime

UsernameStr = Annotated[str, Field(min_length=3, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')]
PasswordStr = Annotated[str, Field(min_length=8, max_length=128)]


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
