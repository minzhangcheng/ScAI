"""Authentication middleware for ScAI backend.

Enforces JWT/API-Key authentication and role/status-based access control.

Access rules:
- admin role: only /api/admin/*, /api/auth/profile/*, /api/auth/me, /api/auth/logout
- readonly status (pending/active but not user+active): non-GET blocked except
  /api/auth/logout, /api/auth/profile/password

All protected paths require authentication; user is loaded from DB per request.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.utils.auth import authenticate_token, get_db

# Paths that do NOT require authentication
PUBLIC_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/setup",
    "/api/config",
    "/docs",
    "/openapi",
    "/redoc",
)

# Paths that DO require authentication (prefix-matched)
PROTECTED_PREFIXES = (
    "/api/session/",
    "/api/v1/",
    "/api/file/",
    "/api/download/",
    "/api/auth/",
    "/api/admin/",
)

# Exact public paths
PUBLIC_EXACT = {"/", "/api"}

# Additional protected exact paths
PROTECTED_EXACT = {"/api/sessions"}

# Paths where auth is optional - authentication is delegated to route layer
# These are image/file endpoints embedded in HTML that browsers request without auth headers
AUTH_OPTIONAL_PATHS = (
    "/api/session/",  # report images and figures - route uses require_session_ownership
    "/api/file/",     # file serving with path-based ownership check
    "/api/download/",  # file download with path-based ownership check
)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in PUBLIC_EXACT:
            return await call_next(request)

        for prefix in PUBLIC_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        needs_auth = path in PROTECTED_EXACT
        if not needs_auth:
            for prefix in PROTECTED_PREFIXES:
                if path.startswith(prefix):
                    needs_auth = True
                    break

        if not needs_auth:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        user_id = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_id = await authenticate_token(token)

        if not user_id:
            # Check if path is auth-optional - delegate auth to route layer via require_session_ownership
            auth_optional = any(path.startswith(p) for p in AUTH_OPTIONAL_PATHS)
            if auth_optional:
                # No auth token provided, but path is auth-optional - let route handle auth
                request.state.user_id = None
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Load full user record from DB per request
        db = await get_db()
        user = await db.get_user_by_id(user_id)

        if user is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "User not found"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Admin role: restrict to specific paths
        if user.role == "admin":
            allowed = (
                path.startswith("/api/admin/") or
                path.startswith("/api/auth/profile/") or
                path == "/api/auth/me" or
                path == "/api/auth/logout"
            )
            if not allowed:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Admin access limited to specific paths"},
                )

        # Readonly status: block non-GET mutations (except logout)
        if user.status == "readonly":
            if request.method != "GET":
                if path not in ("/api/auth/logout", "/api/auth/profile/password"):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Readonly account cannot perform write operations"},
                    )

        # Attach user_id to request state for downstream use
        request.state.user_id = user_id

        return await call_next(request)
