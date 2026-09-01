"""Config routes for API key and LLM settings."""
import asyncio
import copy
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

from backend.models.api_requests import ConfigRequest, ValidateKeyRequest
from backend.models.schemas import APIResponse
from backend.config import is_snapshot_v2_enabled
from backend.state import system_config
from backend.utils.auth import require_admin, get_db
from backend.utils.llm_probe import validate_llm_credentials
from backend.utils.secret_redaction import mask_api_key, seal_system_config_payload
from backend.utils.url_validation import validated_http_url, validated_proxy_url

router = APIRouter(prefix="/api", tags=["config"])

_config_lock = asyncio.Lock()
MASKED_PROXY_URL = "•••••"


def _validated_url(value: str | None, field_name: str, *, allow_empty: bool = False) -> str:
    try:
        return validated_http_url(value or "", field_name, allow_empty=allow_empty)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _validated_proxy(value: str | None, *, allow_empty: bool = False) -> str:
    try:
        return validated_proxy_url(value or "", "proxy_url", allow_empty=allow_empty)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/config")
async def get_config(_: str = Depends(require_admin)):
    """Return the persisted global LLM configuration without exposing secrets."""
    cfg = system_config["current"]
    if cfg is None:
        return {
            "configured": False,
            "has_api_key": False,
            "llm_base_url": "https://openrouter.ai/api/v1",
            "masked_api_key": "",
            "has_proxy_url": False,
        }
    return {
        "configured": True,
        "has_api_key": bool(cfg.api.api_key),
        "llm_base_url": cfg.api.llm_base_url,
        "masked_api_key": mask_api_key(cfg.api.api_key),
        "has_proxy_url": bool(cfg.api.proxy_url),
    }


@router.post("/config")
async def set_config(request: ConfigRequest, _: str = Depends(require_admin)):
    """Set API configuration"""
    from backend.config import SystemConfig

    async with _config_lock:
        old_cfg = system_config["current"]
        cfg = copy.deepcopy(old_cfg) if old_cfg is not None else SystemConfig()
        old_key = cfg.api.api_key
        old_proxy = cfg.api.proxy_url
        cfg.api.api_key = (
            old_key
            if old_key and request.api_key == mask_api_key(old_key)
            else request.api_key
        )
        cfg.api.llm_base_url = _validated_url(request.llm_base_url, "llm_base_url")
        cfg.api.proxy_url = (
            old_proxy
            if old_proxy and request.proxy_url == MASKED_PROXY_URL
            else _validated_proxy(request.proxy_url, allow_empty=True) or None
        )

        system_config["current"] = cfg

        # Persist to database so config survives server restarts
        try:
            db = await get_db()
            persisted = seal_system_config_payload(cfg.model_dump(mode="json"))
            await db.set_meta(
                "global_system_config",
                json.dumps(persisted, ensure_ascii=False),
            )
        except Exception:
            system_config["current"] = old_cfg
            raise

    from backend.utils.config_resolver import invalidate_all
    invalidate_all()
    from backend.utils.settings_provider import refresh_active_providers
    await refresh_active_providers()

    logger.info(f"Config saved: base_url={cfg.api.llm_base_url}, proxy={'configured' if cfg.api.proxy_url else 'none'}")
    return APIResponse(
        success=True,
        message="Configuration saved successfully"
    )


@router.get("/config/status")
async def config_status():
    """Check if configuration is set"""
    cfg = system_config["current"]
    if cfg is None:
        return {
            "configured": False,
            "has_api_key": False,
            "snapshot_v2_enabled": is_snapshot_v2_enabled(),
        }

    return {
        "configured": True,
        "has_api_key": bool(cfg.api.api_key),
        "snapshot_v2_enabled": is_snapshot_v2_enabled(),
    }


@router.post("/config/validate")
async def validate_api_key(request: ValidateKeyRequest, _: str = Depends(require_admin)):
    """Validate API key by attempting to list models from the LLM endpoint"""
    llm_base_url = _validated_url(request.llm_base_url, "llm_base_url")
    proxy_url = _validated_proxy(request.proxy_url, allow_empty=True) or None
    result = await validate_llm_credentials(request.api_key, llm_base_url, proxy_url)
    if result["valid"]:
        return {"valid": True, "message": "API key is valid"}
    if result.get("uncertain"):
        return {"valid": False, "message": result["message"], "uncertain": True}
    return {"valid": False, "message": result["message"]}


@router.post("/config/validate-saved")
async def validate_saved_config(_: str = Depends(require_admin)):
    """Validate the LLM key already saved in the global config.

    Used when an admin has no local api_key in the browser (e.g. after the key
    stopped being persisted to localStorage). Validates the server-side saved
    key instead of one sent in the request body.
    """
    cfg = system_config["current"]
    if cfg is None or not cfg.api.api_key:
        return {"valid": False, "message": "No global API key configured"}
    result = await validate_llm_credentials(cfg.api.api_key, cfg.api.llm_base_url, cfg.api.proxy_url)
    if result["valid"]:
        return {"valid": True, "message": "Saved API key is valid"}
    if result.get("uncertain"):
        return {"valid": False, "message": result["message"], "uncertain": True}
    return {"valid": False, "message": result["message"]}
