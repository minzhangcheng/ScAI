"""Config routes for API key and LLM settings."""
from fastapi import APIRouter, HTTPException, status
import httpx

from backend.models.api_requests import ConfigRequest, ValidateKeyRequest
from backend.models.schemas import APIResponse
from backend.state import system_config

router = APIRouter(prefix="/api", tags=["config"])


@router.post("/config")
async def set_config(request: ConfigRequest):
    """Set API configuration"""
    from backend.config import SystemConfig

    cfg = SystemConfig()
    cfg.api.api_key = request.api_key
    cfg.api.llm_base_url = request.llm_base_url
    cfg.api.proxy_url = request.proxy_url

    system_config["current"] = cfg

    import logging
    logger = logging.getLogger(__name__)
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
async def validate_api_key(request: ValidateKeyRequest):
    """Validate API key by attempting to list models from the LLM endpoint"""
    base_url = request.llm_base_url.rstrip('/')
    models_url = f"{base_url}/models"

    headers = {
        "Authorization": f"Bearer {request.api_key}",
        "Content-Type": "application/json",
    }

    proxy = request.proxy_url if request.proxy_url else None

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            proxy=proxy,
        ) as client:
            response = await client.get(models_url, headers=headers)
            if response.status_code == 200:
                return {"valid": True, "message": "API key is valid"}
            else:
                return {"valid": False, "message": f"Invalid API key (HTTP {response.status_code})"}
    except httpx.HTTPError as e:
        return {"valid": False, "message": f"Failed to connect to LLM endpoint: {str(e)}", "uncertain": True}
