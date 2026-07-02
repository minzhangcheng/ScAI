"""Setup routes: initial admin initialization."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

from backend.middleware.auth import invalidate_admin_initialized_cache
from backend.utils.auth import get_db, hash_password, check_user_uniqueness
from backend.models.user import UsernameStr, PasswordStr

router = APIRouter(prefix="/api/auth/setup", tags=["setup"])


class SetupStatusResponse(BaseModel):
    initialized: bool


class SetupAdminRequest(BaseModel):
    username: UsernameStr
    password: PasswordStr
    email: Optional[EmailStr] = None


class SetupAdminResponse(BaseModel):
    message: str


@router.get("/status", response_model=SetupStatusResponse)
async def get_status():
    """Check if admin has been initialized."""
    db = await get_db()
    meta = await db.get_meta("admin_initialized")
    return SetupStatusResponse(initialized=(meta == "1"))


@router.post("/admin", response_model=SetupAdminResponse, status_code=status.HTTP_201_CREATED)
async def setup_admin(req: SetupAdminRequest):
    """Create the admin account with a custom username and mark admin as initialized."""
    db = await get_db()
    meta = await db.get_meta("admin_initialized")
    if meta == "1":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Admin already initialized",
        )
    await check_user_uniqueness(db, req.username, req.email)
    pw_hash = hash_password(req.password)
    admin_id, created = await db.create_admin_if_missing(
        username=req.username, password_hash=pw_hash, email=req.email,
    )
    if not created:
        await db.update_user_password(admin_id, pw_hash)
    await db.set_meta("admin_initialized", "1")
    invalidate_admin_initialized_cache()
    return SetupAdminResponse(message="Admin account created successfully")
