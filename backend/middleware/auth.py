"""Authentication middleware for ScAI backend.

Enforces JWT/API-Key authentication on protected paths:
- /api/session/*
- /api/v1/*
- /api/file/*
- /api/download/*

Allows unauthenticated access to:
- /api/auth/* (login, register)
- /api/config/* (initial setup)
- / (root)
- /docs, /openapi.json (FastAPI built-in)
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.utils.auth import authenticate_token

# Paths that do NOT require authentication
PUBLIC_PREFIXES = (
    "/api/auth/",
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
)

# Exact public paths
PUBLIC_EXACT = {"/", "/api"}

# Additional protected exact paths
PROTECTED_EXACT = {"/api/sessions"}


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
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Attach user_id to request state for downstream use
        request.state.user_id = user_id

        return await call_next(request)
