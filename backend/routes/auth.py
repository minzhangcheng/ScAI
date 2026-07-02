"""Authentication routes: register, login, me, API key CRUD."""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

from backend.models.user import UsernameStr, PasswordStr
from backend.models.schemas import APIResponse
from backend.utils.db import AppDB
from backend.utils.auth import (
    get_db,
    hash_password,
    verify_password,
    create_access_token,
    generate_api_key,
    hash_api_key,
    check_user_uniqueness,
    require_auth,
    require_active_user,
    require_non_pending_user,
    require_admin,
)

from backend.utils.rate_limit import SlidingWindowRateLimiter

# Login brute-force protection: max 5 failed attempts per (username, client-ip) / 60s
_login_rate_limiter = SlidingWindowRateLimiter(max_attempts=5, window_seconds=60)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Request / Response models ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: UsernameStr
    password: PasswordStr
    email: Optional[EmailStr] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    id: str
    username: str
    email: Optional[str] = None
    role: str
    status: str


class UserInfoResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    role: str
    status: str


class CreateApiKeyRequest(BaseModel):
    label: str = Field(min_length=1, max_length=100)


class ApiKeyResponse(BaseModel):
    id: str
    label: str
    created_at: str
    last_used_at: str | None = None


class CreateApiKeyResponse(ApiKeyResponse):
    key: str  # Raw key — shown only once!


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: PasswordStr


class ChangeEmailRequest(BaseModel):
    current_password: str
    new_email: EmailStr


class LogoutResponse(BaseModel):
    message: str


# ── Routes ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    db: AppDB = await get_db()
    meta = await db.get_meta("admin_initialized")
    if meta != "1":
        raise HTTPException(status_code=403, detail="Admin not initialized yet")
    await check_user_uniqueness(db, req.username, req.email)
    pw_hash = hash_password(req.password)
    uid = await db.create_user(username=req.username, password_hash=pw_hash, email=req.email, status="pending")
    return APIResponse(
        success=True,
        message="Registration submitted; awaiting admin approval.",
        data={"id": uid, "username": req.username, "email": req.email, "status": "pending"},
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    rl_key = f"{req.username}:{client_ip}"
    if not _login_rate_limiter.is_allowed(rl_key):
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please wait and try again.",
        )

    db: AppDB = await get_db()
    user = await db.get_user_by_username(req.username)
    if user is None or not verify_password(req.password, user.password_hash):
        _login_rate_limiter.record_failure(rl_key)
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Account is pending approval. Please contact the administrator to activate your account.")
    _login_rate_limiter.reset(rl_key)
    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token,
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        status=user.status,
    )


@router.get("/me", response_model=UserInfoResponse)
async def me(user_id: str = Depends(require_non_pending_user)):
    db: AppDB = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfoResponse(id=user.id, email=user.email, username=user.username,
                            role=user.role, status=user.status)


@router.get("/profile", response_model=UserInfoResponse)
async def get_profile(user_id: str = Depends(require_non_pending_user)):
    db: AppDB = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfoResponse(id=user.id, email=user.email, username=user.username,
                            role=user.role, status=user.status)


@router.post("/profile/password")
async def update_password(req: ChangePasswordRequest, user_id: str = Depends(require_non_pending_user)):
    db: AppDB = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid old password")
    pw_hash = hash_password(req.new_password)
    await db.update_user_password(user_id, pw_hash)
    return {"message": "Password updated successfully"}


@router.post("/profile/email")
async def update_email(req: ChangeEmailRequest, user_id: str = Depends(require_active_user)):
    db: AppDB = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid old password")
    existing = await db.get_user_by_email(req.new_email)
    if existing and existing.id != user_id:
        raise HTTPException(status_code=409, detail="Email already in use")
    await db.update_user_email(user_id, req.new_email)
    return {"message": "Email updated successfully"}


@router.post("/logout", response_model=LogoutResponse)
async def logout(user_id: str = Depends(require_non_pending_user)):
    # Token invalidation is handled client-side; this endpoint exists for completeness
    return LogoutResponse(message="Logged out successfully")


@router.post("/api-keys", response_model=CreateApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(req: CreateApiKeyRequest, user_id: str = Depends(require_active_user)):
    db: AppDB = await get_db()
    raw, hashed = generate_api_key()
    kid, created_at = await db.create_api_key(user_id=user_id, hashed_key=hashed, label=req.label)
    return CreateApiKeyResponse(
        id=kid, label=req.label, created_at=created_at,
        last_used_at=None, key=raw,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(user_id: str = Depends(require_active_user)):
    db: AppDB = await get_db()
    keys = await db.list_api_keys(user_id)
    return [
        ApiKeyResponse(id=k.id, label=k.label, created_at=k.created_at,
                        last_used_at=k.last_used_at)
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(key_id: str, user_id: str = Depends(require_active_user)):
    db: AppDB = await get_db()
    deleted = await db.delete_api_key(key_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
