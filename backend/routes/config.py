"""Config routes for API key and LLM settings."""
import asyncio
import copy
import logging

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

from backend.models.api_requests import ConfigRequest, ValidateKeyRequest
from backend.models.schemas import APIResponse
from backend.state import system_config
from backend.utils.auth import require_admin, get_db
from backend.utils.llm_probe import validate_llm_credentials

router = APIRouter(prefix="/api", tags=["config"])

_config_lock = asyncio.Lock()


@router.post("/config")
async def set_config(request: ConfigRequest, _: str = Depends(require_admin)):
    """Set API configuration"""
    from backend.config import SystemConfig

    async with _config_lock:
        old_cfg = system_config["current"]
        cfg = copy.deepcopy(old_cfg) if old_cfg is not None else SystemConfig()
        cfg.api.api_key = request.api_key
        cfg.api.llm_base_url = request.llm_base_url
        cfg.api.proxy_url = request.proxy_url

        system_config["current"] = cfg

        # Persist to database so config survives server restarts
        try:
            db = await get_db()
            await db.set_meta("global_system_config", cfg.model_dump_json())
        except Exception:
            system_config["current"] = old_cfg
            raise

    from backend.utils.config_resolver import invalidate_all
    invalidate_all()

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
        return {"configured": False}

    return {
        "configured": True,
        "has_api_key": bool(cfg.api.api_key),
    }


@router.post("/config/validate")
async def validate_api_key(request: ValidateKeyRequest, _: str = Depends(require_admin)):
    """Validate API key by attempting to list models from the LLM endpoint"""
    result = await validate_llm_credentials(request.api_key, request.llm_base_url, request.proxy_url)
    if result["valid"]:
        return {"valid": True, "message": "API key is valid"}
    if result.get("uncertain"):
        return {"valid": False, "message": result["message"], "uncertain": True}
    return {"valid": False, "message": result["message"]}
