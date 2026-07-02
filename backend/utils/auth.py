"""Authentication utilities: password hashing, JWT, API key, FastAPI dependencies."""
import os
import stat
import hashlib
import secrets
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.utils.db import AppDB
from backend.config import APP_DB_PATH


async def check_user_uniqueness(db: AppDB, username: str, email: Optional[str] = None):
    """Raise 409 if username or email is already taken."""
    existing = await db.get_user_by_username(username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    if email:
        existing_email = await db.get_user_by_email(email)
        if existing_email:
            raise HTTPException(status_code=409, detail="Email already registered")

# ── Password hashing ────────────────────────────────────────────────────

_BCRYPT_MAX_PW_BYTES = 72


def hash_password(password: str) -> str:
    if len(password.encode("utf-8")) > _BCRYPT_MAX_PW_BYTES:
        raise ValueError(f"Password exceeds {_BCRYPT_MAX_PW_BYTES} bytes")
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    # Truncate on verify only, to remain compatible with hashes created
    # before byte-length validation existed.
    return bcrypt.checkpw(plain.encode()[:_BCRYPT_MAX_PW_BYTES], hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────

JWT_SECRET = os.environ.get("SCAI_JWT_SECRET", "")
if not JWT_SECRET:
    # Generate a random secret on first boot and persist it
    _secret_path = APP_DB_PATH.parent / ".jwt_secret"
    if _secret_path.exists():
        JWT_SECRET = _secret_path.read_text().strip()
    else:
        JWT_SECRET = secrets.token_urlsafe(48)
        _secret_path.parent.mkdir(parents=True, exist_ok=True)
        _secret_path.write_text(JWT_SECRET)
        # Restrict file permissions to owner-only
        _secret_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 72  # 3 days


def create_access_token(user_id: str, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {"sub": user_id, "username": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ── API Key ──────────────────────────────────────────────────────────────

API_KEY_PREFIX = "scai_"
WS_AUTH_SUBPROTOCOL = "scai.jwt.v1"
WS_AUTH_FAILED_CODE = 4401
SESSION_NOT_FOUND = "Session not found"


def generate_api_key() -> tuple[str, str]:
    """Return (raw_key, hashed_key). The raw key is shown to the user ONCE."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def authenticate_token(token: str) -> Optional[str]:
    """Authenticate a token string. Returns user_id or None.
    Tries API key first (scai_ prefix), then JWT.
    """
    if token.startswith(API_KEY_PREFIX):
        db = await get_db()
        hashed = hash_api_key(token)
        user_id = await db.lookup_user_by_hashed_key(hashed)
        if user_id:
            return user_id

    payload = decode_access_token(token)
    if payload and "sub" in payload:
        return payload["sub"]

    return None


# ── DB singleton ──────────────────────────────────────────────────────────

_db: Optional[AppDB] = None
_db_lock = asyncio.Lock()


async def get_db() -> AppDB:
    global _db
    if _db is not None:
        return _db
    async with _db_lock:
        if _db is None:
            _db = AppDB(APP_DB_PATH)
            await _db.init()
        return _db


# ── FastAPI Dependencies ─────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """
    Return user_id if authenticated via JWT or API Key.
    Raises 401 if not.
    """
    if credentials and credentials.scheme.lower() == "bearer":
        user_id = await authenticate_token(credentials.credentials)
        if user_id:
            return user_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def optional_require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> Optional[str]:
    """
    Return user_id if authenticated via JWT or API Key, otherwise return None.
    Does not raise 401 - auth is optional.
    """
    if credentials and credentials.scheme.lower() == "bearer":
        user_id = await authenticate_token(credentials.credentials)
        if user_id:
            return user_id
    return None


async def require_active_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Return user_id for write operations. Requires active status (any role)."""
    user_id = await require_auth(request, credentials)
    db = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.status != "active":
        raise HTTPException(status_code=403, detail="Active user required for write operations")
    return user_id


async def require_non_pending_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Return user_id for read-only and profile operations. Requires user.status in ('active', 'readonly')."""
    user_id = await require_auth(request, credentials)
    db = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.status not in ("active", "readonly"):
        raise HTTPException(status_code=403, detail="Non-pending user required")
    return user_id


async def require_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Return user_id for admin operations. Requires user.role=='admin'."""
    user_id = await require_auth(request, credentials)
    db = await get_db()
    user = await db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    return user_id


async def require_session_ownership(
    session_id: str,
    user_id: Optional[str] = Depends(optional_require_auth),
) -> "AnalysisState":
    """
    Load session and verify ownership. Returns the session state.
    Raises 404 if not found or not owned (leak-oracle safe).
    Uses optional_require_auth - when user_id is None (no token), only verifies session exists.
    This allows HTML-embedded resources to be accessed without auth headers.
    """
    from backend.services.session_store import ensure_session_loaded
    from backend.models.schemas import AnalysisState

    state = await ensure_session_loaded(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND)
    # When user_id is None (no auth token provided), only verify session exists
    # This is safe because session_id is a UUID that attackers cannot guess
    if user_id is not None and state.user_id != user_id:
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND)
    return state


async def ws_require_auth(websocket: WebSocket) -> Optional[str]:
    """
    Extract JWT from Sec-WebSocket-Protocol header (scai.jwt.v1 subprotocol).
    Return user_id or close with 4401.

    Protocol format: Sec-WebSocket-Protocol: scai.jwt.v1, <jwt-token>
    The server accepts subprotocol "scai.jwt.v1" and extracts the token from
    the second protocol entry.
    """
    protocol_header = websocket.headers.get("sec-websocket-protocol", "")
    parts = [p.strip() for p in protocol_header.split(",")]

    # Require exactly "scai.jwt.v1" as first subprotocol, token as second
    if len(parts) < 2 or parts[0] != WS_AUTH_SUBPROTOCOL:
        await websocket.close(code=WS_AUTH_FAILED_CODE, reason="Authentication required")
        return None

    token = parts[1]
    user_id = await authenticate_token(token)
    if user_id:
        await websocket.accept(subprotocol=WS_AUTH_SUBPROTOCOL)
        return user_id

    await websocket.close(code=WS_AUTH_FAILED_CODE, reason="Invalid token")
    return None


async def ws_authenticate_and_load(
    websocket: WebSocket, session_id: str
) -> Optional[tuple[str, "AnalysisState"]]:
    """Authenticate WS connection and load session with ownership check.
    Returns (user_id, state) or None (closes websocket on failure).
    """
    from backend.services.session_store import ensure_session_loaded
    from backend.utils.files import validate_session_id

    validate_session_id(session_id)
    user_id = await ws_require_auth(websocket)
    if user_id is None:
        return None
    state = await ensure_session_loaded(session_id)
    if state is None or state.user_id != user_id:
        await websocket.send_json({"error": SESSION_NOT_FOUND})
        await websocket.close(code=WS_AUTH_FAILED_CODE)
        return None
    return (user_id, state)
