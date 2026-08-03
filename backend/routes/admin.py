"""Admin management routes: user CRUD with cascade delete."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional

from backend.models.user import (
    PasswordStr,
    USER_STATUS_DELETING,
    UsernameStr,
)
from backend.utils.db import AppDB
from backend.utils.auth import get_db, hash_password, require_admin, check_user_uniqueness
from backend.services.user_cleanup import UserCleanupPendingError, delete_user_data

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


class SetRoleRequest(BaseModel):
    role: str  # "admin" | "user"


class SetPasswordRequest(BaseModel):
    new_password: PasswordStr


# ── Routes ────────────────────────────────────────────────────────────────

def _reject_deleting_user(target) -> None:
    if target.status == USER_STATUS_DELETING:
        raise HTTPException(status_code=409, detail="User deletion is in progress")


async def _raise_guarded_user_update_failure(
    db: AppDB,
    uid: str,
    admin_id: str,
) -> None:
    """Distinguish deletion races from a user disappearing mid-update."""
    if not await db.is_usable_admin(admin_id):
        raise HTTPException(status_code=403, detail="Admin authorization was revoked")
    current = await db.get_user_by_id(uid)
    if current is None:
        raise HTTPException(status_code=404, detail="User not found")
    _reject_deleting_user(current)
    raise HTTPException(status_code=409, detail="User changed during the update")

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
async def create_user(
    req: CreateUserRequest,
    admin_id: str = Depends(require_admin),
):
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
        required_admin_id=admin_id,
    )
    if uid is None:
        raise HTTPException(status_code=403, detail="Admin authorization was revoked")
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
    """Set a user's status. Admin cannot modify their own status; admin users cannot be pending."""
    if uid == admin_id:
        raise HTTPException(status_code=403, detail="Cannot modify your own status")
    if req.status not in ("pending", "readonly", "active"):
        raise HTTPException(status_code=400, detail="Invalid status value")
    db: AppDB = await get_db()
    target = await db.get_user_by_id(uid)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    _reject_deleting_user(target)
    if target.role == "admin" and req.status == "pending":
        raise HTTPException(status_code=400, detail="Admin users cannot be set to pending")
    success = await db.set_user_status(
        uid,
        req.status,
        unless_current_status=USER_STATUS_DELETING,
        forbid_current_role="admin" if req.status == "pending" else None,
        required_admin_id=admin_id,
    )
    if not success:
        await _raise_guarded_user_update_failure(db, uid, admin_id)
    return {"message": f"User status updated to {req.status}"}


@router.patch("/users/{uid}/role")
async def set_user_role(
    uid: str,
    req: SetRoleRequest,
    admin_id: str = Depends(require_admin),
):
    """Change a user's role. Cannot change your own role. Promoting a pending user to admin activates them."""
    if uid == admin_id:
        raise HTTPException(status_code=403, detail="Cannot modify your own role")
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Invalid role value")
    db: AppDB = await get_db()
    target = await db.get_user_by_id(uid)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    _reject_deleting_user(target)
    if req.role != "admin" and target.role == "admin":
        if not await db.demote_admin_if_not_last(
            uid,
            req.role,
            unless_current_status=USER_STATUS_DELETING,
            required_admin_id=admin_id,
        ):
            if not await db.is_usable_admin(admin_id):
                raise HTTPException(
                    status_code=403,
                    detail="Admin authorization was revoked",
                )
            current = await db.get_user_by_id(uid)
            if current is None:
                raise HTTPException(status_code=404, detail="User not found")
            _reject_deleting_user(current)
            raise HTTPException(
                status_code=409,
                detail="Cannot demote the last remaining admin",
            )
    else:
        new_status = None
        if req.role == "admin" and target.status == "pending":
            new_status = "active"
        success = await db.set_user_role(
            uid,
            req.role,
            status=new_status,
            unless_current_status=USER_STATUS_DELETING,
            activate_pending_admin=req.role == "admin" and new_status is None,
            required_admin_id=admin_id,
        )
        if not success:
            await _raise_guarded_user_update_failure(db, uid, admin_id)
    return {"message": f"User role updated to {req.role}"}


@router.post("/users/{uid}/password")
async def set_user_password(
    uid: str,
    req: SetPasswordRequest,
    admin_id: str = Depends(require_admin),
):
    """Change any user's password. Requires admin."""
    db: AppDB = await get_db()
    target = await db.get_user_by_id(uid)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    _reject_deleting_user(target)
    pw_hash = hash_password(req.new_password)
    success = await db.update_user_password(
        uid,
        pw_hash,
        unless_current_status=USER_STATUS_DELETING,
        required_admin_id=admin_id,
    )
    if not success:
        await _raise_guarded_user_update_failure(db, uid, admin_id)
    return {"message": "Password updated successfully"}


@router.delete("/users/{uid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    uid: str,
    admin_id: str = Depends(require_admin),
):
    """
    Delete a user and all their data (cascade delete).
    Admin cannot delete themselves, but may delete other users (including other admins).
    """
    if uid == admin_id:
        raise HTTPException(status_code=403, detail="Cannot delete yourself")
    db: AppDB = await get_db()
    deletion_state = await db.mark_user_deleting_if_not_last_admin(
        uid,
        required_admin_id=admin_id,
    )
    if deletion_state == "forbidden":
        raise HTTPException(status_code=403, detail="Admin authorization was revoked")
    if deletion_state == "not_found":
        raise HTTPException(status_code=404, detail="User not found")
    if deletion_state == "last_admin":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete the last remaining admin",
        )
    try:
        await delete_user_data(uid)
    except UserCleanupPendingError as exc:
        raise HTTPException(status_code=409, detail=exc.as_detail()) from exc
    return None
