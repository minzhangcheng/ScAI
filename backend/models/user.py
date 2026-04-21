"""User and API Key Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    id: str
    email: str
    password_hash: str = ""
    created_at: str = ""


class ApiKey(BaseModel):
    id: str
    user_id: str
    hashed_key: str
    label: str = ""
    created_at: str = ""
    last_used_at: Optional[str] = None
