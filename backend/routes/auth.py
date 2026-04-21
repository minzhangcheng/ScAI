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
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Request / Response models ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    id: str
    email: str


class UserInfoResponse(BaseModel):
    id: str
    email: str


class CreateApiKeyRequest(BaseModel):
    label: str = Field(min_length=1, max_length=100)


class ApiKeyResponse(BaseModel):
    id: str
    label: str
    created_at: str
    last_used_at: str | None = None


class CreateApiKeyResponse(ApiKeyResponse):
    key: str  # Raw key — shown only once!


# ── Routes ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    db: AppDB = await get_db()
    existing = await db.get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    pw_hash = hash_password(req.password)
    uid = await db.create_user(email=req.email, password_hash=pw_hash)
    token = create_access_token(uid, req.email)
    return TokenResponse(access_token=token, id=uid, email=req.email)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    db: AppDB = await get_db()
    user = await db.get_user_by_email(req.email)
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, id=user.id, email=user.email)


@router.get("/me", response_model=UserInfoResponse)
async def me(user_id: str = Depends(require_auth)):
    db: AppDB = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfoResponse(id=user.id, email=user.email)


@router.post("/api-keys", response_model=CreateApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(req: CreateApiKeyRequest, user_id: str = Depends(require_auth)):
    db: AppDB = await get_db()
    raw, hashed = generate_api_key()
    kid, created_at = await db.create_api_key(user_id=user_id, hashed_key=hashed, label=req.label)
    return CreateApiKeyResponse(
        id=kid, label=req.label, created_at=created_at,
        last_used_at=None, key=raw,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(user_id: str = Depends(require_auth)):
    db: AppDB = await get_db()
    keys = await db.list_api_keys(user_id)
    return [
        ApiKeyResponse(id=k.id, label=k.label, created_at=k.created_at,
                        last_used_at=k.last_used_at)
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(key_id: str, user_id: str = Depends(require_auth)):
    db: AppDB = await get_db()
    deleted = await db.delete_api_key(key_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
