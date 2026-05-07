"""Per-user LLM configuration routes."""
import logging
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.models.schemas import APIResponse
from backend.utils.auth import get_db, require_auth, require_active_user, optional_require_auth
from backend.utils.config_resolver import invalidate_cache
from backend.utils.llm_probe import validate_llm_credentials
from backend.utils.db import AppDB
from cryptography.fernet import InvalidToken
from backend.utils import crypto
from backend.state import system_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["user_config"])


# ── Request models ───────────────────────────────────────────────────────────

class UserConfigRequest(BaseModel):
    api_key: str = ""
    llm_base_url: str = ""
    proxy_url: Optional[str] = None


class ValidateUserConfigRequest(BaseModel):
    api_key: str
    llm_base_url: str = "https://openrouter.ai/api/v1"
    proxy_url: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mask_api_key(key: str) -> str:
    if len(key) <= 12:
        return "****"
    return key[:6] + "..." + key[-4:]


def _validate_url(url: str, field_name: str) -> None:
    if not url:
        return
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"{field_name} must use http or https scheme")


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/config")
async def get_user_config(
    user_id: Optional[str] = Depends(optional_require_auth),
    db: AppDB = Depends(get_db),
):
    """Return the per-user LLM config (masked api_key). Returns {"configured": False} when unauthenticated."""
    if user_id is None:
        return {"configured": False}

    # Per-user config is meaningless without a global config to merge with.
    if system_config.get("current") is None:
        return {"configured": False, "has_api_key": False, "llm_base_url": ""}

    row = await db.get_user_llm_config(user_id)
    if row is None:
        return {"configured": False, "has_api_key": False, "llm_base_url": ""}

    # Decrypt api_key for masking (display only — never return plaintext)
    masked_key = ""
    has_api_key = False
    if row.get("encrypted_api_key"):
        try:
            plaintext_key = crypto.decrypt_value(row["encrypted_api_key"])
            has_api_key = bool(plaintext_key)
            masked_key = _mask_api_key(plaintext_key) if plaintext_key else ""
        except crypto.ConfigEncryptionUnavailable:
            logger.warning("Encryption unavailable; cannot decrypt api_key for user %s", user_id)
        except InvalidToken:
            logger.warning("Invalid token decrypting api_key for user %s", user_id, exc_info=True)

    return {
        "configured": True,
        "has_api_key": has_api_key,
        "llm_base_url": row.get("llm_base_url", ""),
        "masked_api_key": masked_key,
    }


@router.get("/config/status")
async def get_user_config_status(
    user_id: Optional[str] = Depends(optional_require_auth),
    db: AppDB = Depends(get_db),
):
    """Lightweight status check — no decryption. Returns {configured, has_api_key}."""
    if user_id is None:
        return {"configured": False}
    if system_config.get("current") is None:
        return {"configured": False, "has_api_key": False}
    row = await db.get_user_llm_config(user_id)
    if row is None:
        return {"configured": False, "has_api_key": False}
    return {"configured": True, "has_api_key": bool(row.get("encrypted_api_key"))}


@router.post("/config")
async def set_user_config(
    body: UserConfigRequest,
    user_id: str = Depends(require_active_user),
    db: AppDB = Depends(get_db),
):
    """Save per-user LLM config. Requires active user (pending/readonly users are rejected)."""
    if body.llm_base_url:
        _validate_url(body.llm_base_url, "llm_base_url")
    if body.proxy_url:
        _validate_url(body.proxy_url, "proxy_url")

    try:
        encrypted_api_key = crypto.encrypt_value(body.api_key) if body.api_key else ""
        encrypted_proxy_url = crypto.encrypt_value(body.proxy_url or "") if body.proxy_url else ""
    except crypto.ConfigEncryptionUnavailable as e:
        logger.error("Encryption unavailable when saving user config: %s", e)
        raise HTTPException(status_code=503, detail="Encryption service unavailable; cannot save config")

    await db.set_user_llm_config(
        user_id,
        encrypted_api_key,
        body.llm_base_url,
        encrypted_proxy_url,
    )

    invalidate_cache(user_id)

    return APIResponse(success=True)


@router.delete("/config")
async def delete_user_config(
    user_id: str = Depends(require_active_user),
    db: AppDB = Depends(get_db),
):
    """Delete per-user LLM config. Requires active user."""
    await db.delete_user_llm_config(user_id)

    invalidate_cache(user_id)

    return APIResponse(success=True)


@router.post("/config/validate")
async def validate_user_config(
    body: ValidateUserConfigRequest,
    user_id: str = Depends(require_active_user),
):
    """Validate the given api_key against the LLM provider's /models endpoint. Requires active user."""
    _validate_url(body.llm_base_url, "llm_base_url")
    if body.proxy_url:
        _validate_url(body.proxy_url, "proxy_url")

    result = await validate_llm_credentials(body.api_key, body.llm_base_url, body.proxy_url)
    return result
