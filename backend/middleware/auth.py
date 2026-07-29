"""Authentication middleware for ScAI backend.

Enforces JWT/API-Key authentication and role/status-based access control.

Access rules:
- admin role: only /api/admin/*, /api/auth/profile/*, /api/auth/me, /api/auth/logout,
  /api/config, /api/user/
- readonly status (pending/active but not user+active): non-GET blocked except
  /api/auth/logout, /api/auth/profile/password

All protected paths require authentication; user is loaded from DB per request.
"""
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.utils.auth import authenticate_token, get_db

# Paths that do NOT require authentication
PUBLIC_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/setup",
    "/docs",
    "/openapi",
    "/redoc",
)

# Paths that DO require authentication (prefix-matched)
PROTECTED_PREFIXES = (
    "/api/session/",
    "/api/v1/",
    # Leaf endpoints with no sub-paths — matched without a trailing slash so the
    # exact route path (/api/file, /api/download) is recognized, not just /api/file/*.
    "/api/file",
    "/api/download",
    "/api/auth/",
    "/api/admin/",
    "/api/user/",
)

# Exact public paths
PUBLIC_EXACT = {"/", "/api", "/api/health"}

# Additional protected exact paths
PROTECTED_EXACT = {"/api/sessions"}

# Paths where auth is optional - authentication is delegated to route layer
# These are image/file endpoints embedded in HTML that browsers request without auth headers
AUTH_OPTIONAL_PATHS = (
    "/api/file",     # file serving with path-based ownership check (<img src> / link, no auth header)
    "/api/download",  # file download with path-based ownership check
)

# Suffixes under /api/session/<uuid>/ that are auth-optional
# (content served to browsers/responses that can't send auth headers,
#  e.g. report HTML pages, figures, tables, exports)
SESSION_AUTH_OPTIONAL_SUFFIXES = (
    "/report/",    # /report/html, /report/figure/*, /report/table/*
    "-report/",    # /exploration-report/*, /annotation-report/*, /detailed-report/*
    # Only the token-redeeming download leg is auth-optional: the browser fetches
    # it through <a download>, which carries no Authorization header, and the
    # one-shot unguessable token minted by /export/prepare stands in for one.
    # /export and /export/prepare are NOT listed -- they hand out a full archive
    # of every upload, figure, report and message in the session, and their only
    # ownership check degrades to "does this session exist" when no token is
    # presented (see require_session_ownership).
    "/export/download",
    "/figure/",    # /figure/{fid}/png, /figure/{fid}/pdf (FigureGallery <img src> without auth headers)
)

# Cache for admin_initialized check (avoids DB query on every config request)
_admin_initialized_cache: dict = {"value": None, "expires": 0}
_ADMIN_INITIALIZED_CACHE_TTL = 60  # seconds


def _set_admin_cache(value: bool) -> None:
    """Set the admin-initialized cache to a known state with a long expiry."""
    _admin_initialized_cache["value"] = value
    _admin_initialized_cache["expires"] = time.time() + 3600


def invalidate_admin_initialized_cache() -> None:
    """Reset the admin-initialized cache to force a fresh DB query."""
    _admin_initialized_cache["value"] = None
    _admin_initialized_cache["expires"] = 0


def _is_auth_optional(path: str) -> bool:
    """Check if a path is auth-optional (auth delegated to route layer).

    Suffix matching is anchored to the path segment immediately after
    /api/session/{session_id} — so /export only matches at that position,
    not as a substring elsewhere in the URL.
    """
    # Defense in depth: reject relative-traversal segments before suffix matching.
    if "/../" in path or "/./" in path or path.endswith("/..") or path.endswith("/."):
        return False
    if any(path.startswith(p) for p in AUTH_OPTIONAL_PATHS):
        return True
    if path.startswith("/api/session/"):
        # Extract the remainder after /api/session/{session_id}
        rest = path[len("/api/session/"):]
        slash_idx = rest.find("/")
        if slash_idx == -1:
            return False
        # tail starts with "/" — e.g. "/export", "/report/html", "/exploration-report/x"
        tail = rest[slash_idx:]
        for suffix in SESSION_AUTH_OPTIONAL_SUFFIXES:
            if suffix.endswith("/"):
                # Prefix-of-subpath suffix (e.g. "/report/", "-report/", "/figure/")
                if tail.startswith(suffix) or _tail_has_dash_prefix(tail, suffix):
                    return True
            else:
                # Terminal suffix (e.g. "/export") — must be exact or followed by "/"
                if tail == suffix or tail.startswith(suffix + "/"):
                    return True
    return False


def _tail_has_dash_prefix(tail: str, suffix: str) -> bool:
    """Match suffixes like '-report/' that attach to the end of the first segment.

    e.g. tail '/exploration-report/html' matches suffix '-report/'.
    Only valid when suffix starts with '-' (not '/').
    """
    if not suffix.startswith("-"):
        return False
    # First segment of tail is everything between leading "/" and next "/"
    next_slash = tail.find("/", 1)
    if next_slash == -1:
        return False
    first_seg = tail[1:next_slash]
    rest_with_slash = tail[next_slash:]  # starts with "/"
    return first_seg.endswith(suffix.rstrip("/")) and rest_with_slash.startswith("/")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in PUBLIC_EXACT:
            return await call_next(request)

        for prefix in PUBLIC_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        # Config endpoints: allow unauthenticated access only before admin is initialized
        if path.startswith("/api/config"):
            if await self._admin_initialized():
                # After admin setup, config requires authentication
                needs_auth = True
            else:
                # Before admin setup, config is public (first-time configuration)
                request.state.user_id = None
                return await call_next(request)
        else:
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
            if _is_auth_optional(path):
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
            # Auth-optional paths: pass through with token user_id,
            # let route layer handle ownership checking
            if _is_auth_optional(path):
                request.state.user_id = user_id
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "User not found"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Admin role: full access (management paths + analysis paths).
        # Session ownership is still enforced at the route layer via
        # require_session_ownership, so admins only see their own sessions.

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

    @staticmethod
    async def _admin_initialized() -> bool:
        """Check if admin has been initialized, with short-lived cache."""
        global _admin_initialized_cache
        now = time.time()
        if _admin_initialized_cache["value"] is not None and now < _admin_initialized_cache["expires"]:
            return _admin_initialized_cache["value"]
        db = await get_db()
        result = await db.is_admin_initialized()
        _admin_initialized_cache["value"] = result
        _admin_initialized_cache["expires"] = now + _ADMIN_INITIALIZED_CACHE_TTL
        return result
