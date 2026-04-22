"""User and API Key Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    id: str
    email: Optional[str] = None
    username: Optional[str] = None
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
