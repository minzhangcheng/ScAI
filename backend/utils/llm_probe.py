"""Shared LLM credentials validation — pings the /models endpoint."""
import logging
from urllib.parse import urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

_LLM_ENDPOINT_SUFFIXES = ("/chat/completions", "/responses", "/messages")


def _models_endpoint_url(llm_base_url: str) -> str:
    """Build the models URL from either a base URL or a full LLM endpoint."""
    parsed = urlsplit(llm_base_url)
    path = parsed.path.rstrip("/")
    lower_path = path.lower()
    for suffix in _LLM_ENDPOINT_SUFFIXES:
        if lower_path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    models_path = f"{path}/models" if path else "/models"
    return urlunsplit(parsed._replace(path=models_path))


async def validate_llm_credentials(
    api_key: str,
    llm_base_url: str,
    proxy_url: str | None = None,
) -> dict:
    """Validate ``api_key`` against the provider's derived ``/models`` URL.

    Returns a dict with keys:
        valid (bool), uncertain (bool), message (str)
    """
    models_url = _models_endpoint_url(llm_base_url)

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
