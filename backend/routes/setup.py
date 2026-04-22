"""Setup routes: initial admin initialization."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.utils.auth import get_db, hash_password

router = APIRouter(prefix="/api/auth/setup", tags=["setup"])


class SetupStatusResponse(BaseModel):
    initialized: bool


class SetupAdminRequest(BaseModel):
    password: str = Field(min_length=8, max_length=128)


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
    """Set the admin password and mark admin as initialized."""
    db = await get_db()
    meta = await db.get_meta("admin_initialized")
    if meta == "1":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Admin already initialized",
        )
    # Create admin if not exists, then ensure password and status are set correctly
    pw_hash = hash_password(req.password)
    admin_id = await db.create_admin_if_missing("admin@scai.local", pw_hash)
    await db.update_user_password(admin_id, pw_hash)
    await db.set_user_status(admin_id, "active")
    await db.set_meta("admin_initialized", "1")
    return SetupAdminResponse(message="Admin password set successfully")