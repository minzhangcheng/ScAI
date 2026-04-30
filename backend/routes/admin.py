"""Admin management routes: user CRUD with cascade delete."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional

from backend.models.user import UsernameStr, PasswordStr
from backend.utils.db import AppDB
from backend.utils.auth import get_db, hash_password, require_admin, check_user_uniqueness
from backend.services.user_cleanup import delete_user_data

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Request / Response models ────────────────────────────────────────────

class AdminUserResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    role: str
    status: str
    created_at: str


class CreateUserRequest(BaseModel):
    username: UsernameStr
    password: PasswordStr
    email: Optional[EmailStr] = None


class SetStatusRequest(BaseModel):
    status: str  # "pending" | "readonly" | "active"


class SetPasswordRequest(BaseModel):
    new_password: str  # min 8 chars expected by caller


# ── Routes ────────────────────────────────────────────────────────────────

@router.get("/users", response_model=list[AdminUserResponse])
async def list_users(_: str = Depends(require_admin)):
    """List all users (password_hash excluded). Requires admin."""
    db: AppDB = await get_db()
    users = await db.list_users()
    return [
        AdminUserResponse(
            id=u.id, username=u.username, email=u.email,
            role=u.role, status=u.status, created_at=u.created_at,
        )
        for u in users
    ]


@router.post("/users", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(req: CreateUserRequest, _: str = Depends(require_admin)):
    """Create a new user with status='active'. Requires admin."""
    db: AppDB = await get_db()
    await check_user_uniqueness(db, req.username, req.email)
    pw_hash = hash_password(req.password)
    uid = await db.create_user(
        username=req.username,
        password_hash=pw_hash,
        email=req.email,
        role="user",  # Always 'user' — admin creation not allowed via this endpoint
        status="active",
    )
    user = await db.get_user_by_id(uid)
    if user is None:
        raise HTTPException(status_code=500, detail="User creation failed")
    return AdminUserResponse(
        id=user.id, username=user.username, email=user.email,
        role=user.role, status=user.status, created_at=user.created_at,
    )


@router.patch("/users/{uid}/status")
async def set_user_status(
    uid: str,
    req: SetStatusRequest,
    admin_id: str = Depends(require_admin),
):
    """Set a user's status. Admin cannot modify their own status."""
    if uid == admin_id:
        raise HTTPException(status_code=403, detail="Cannot modify your own status")
    if req.status not in ("pending", "readonly", "active"):
        raise HTTPException(status_code=400, detail="Invalid status value")
    db: AppDB = await get_db()
    target = await db.get_user_by_id(uid)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    success = await db.set_user_status(uid, req.status)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"User status updated to {req.status}"}


@router.post("/users/{uid}/password")
async def set_user_password(
    uid: str,
    req: SetPasswordRequest,
    _: str = Depends(require_admin),
):
    """Change any user's password. Requires admin."""
    db: AppDB = await get_db()
    target = await db.get_user_by_id(uid)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    pw_hash = hash_password(req.new_password)
    await db.update_user_password(uid, pw_hash)
    return {"message": "Password updated successfully"}


@router.delete("/users/{uid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    uid: str,
    admin_id: str = Depends(require_admin),
):
    """
    Delete a user and all their data (cascade delete).
    Admin cannot delete themselves.
    Cannot delete another admin.
    """
    if uid == admin_id:
        raise HTTPException(status_code=403, detail="Cannot delete yourself")
    db: AppDB = await get_db()
    target = await db.get_user_by_id(uid)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if target.role == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete another admin")
    await delete_user_data(uid)
    return None
