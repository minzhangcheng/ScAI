"""Authentication routes: register, login, me, API key CRUD."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

from backend.utils.db import AppDB
from backend.utils.auth import (
    get_db,
    hash_password,
    verify_password,
    create_access_token,
    generate_api_key,
    hash_api_key,
    require_auth,
    require_active_user,
    require_non_pending_user,
    require_admin,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Request / Response models ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    identifier: str  # email or "admin"
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    id: str
    email: str
    username: str | None
    role: str
    status: str


class UserInfoResponse(BaseModel):
    id: str
    email: str
    username: str | None
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
    new_password: str = Field(min_length=8, max_length=128)


class ChangeEmailRequest(BaseModel):
    current_password: str
    new_email: EmailStr


class LogoutResponse(BaseModel):
    message: str


# ── Routes ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    db: AppDB = await get_db()
    meta = await db.get_meta("admin_initialized")
    if meta != "1":
        raise HTTPException(status_code=403, detail="Admin not initialized yet")
    existing = await db.get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    pw_hash = hash_password(req.password)
    uid = await db.create_user(email=req.email, password_hash=pw_hash, status="pending")
    # Intentionally return empty token: pending users must be approved by admin before receiving a valid session
    return TokenResponse(
        access_token="",
        id=uid,
        email=req.email,
        username=None,
        role="user",
        status="pending",
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    db: AppDB = await get_db()
    if req.identifier == "admin":
        user = await db.get_admin_user()
    else:
        user = await db.get_user_by_email(req.identifier)
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    if user.status == "pending":
        raise HTTPException(status_code=403, detail="Account is pending approval. Please contact the administrator to activate your account.")
    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token,
        id=user.id,
        email=user.email,
        username=user.username,
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
