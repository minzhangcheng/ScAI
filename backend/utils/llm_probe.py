"""Shared LLM credentials validation — pings the /models endpoint."""
import httpx
import logging

logger = logging.getLogger(__name__)


async def validate_llm_credentials(
    api_key: str,
    llm_base_url: str,
    proxy_url: str | None = None,
) -> dict:
    """Validate ``api_key`` against ``llm_base_url/models``.

    Returns a dict with keys:
        valid (bool), uncertain (bool), message (str)
    """
    base_url = llm_base_url.rstrip("/")
    models_url = f"{base_url}/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    proxy = proxy_url if proxy_url else None

    try:
        async with httpx.AsyncClient(timeout=10.0, proxy=proxy) as client:
            response = await client.get(models_url, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("Network error validating LLM credentials: %s", exc)
        return {"valid": False, "uncertain": True, "message": f"Unable to verify (network error): {exc}"}
    except Exception as exc:
        logger.warning("Unexpected error validating LLM credentials: %s", exc)
        return {"valid": False, "uncertain": True, "message": f"Validation failed: {exc}"}

    if response.status_code in (401, 403):
        return {"valid": False, "uncertain": False, "message": "Invalid API key"}

    if response.status_code == 200:
        return {"valid": True, "uncertain": False}

    return {"valid": False, "uncertain": False, "message": f"Validation failed (HTTP {response.status_code})"}