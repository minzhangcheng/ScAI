"""Per-user config resolution: merges per-user API credentials over the global SystemConfig.

Resolution chain:
    per-user API creds -> global config (all other fields)

Only the per-user API portion is cached (not the full resolved SystemConfig).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from cryptography.fernet import InvalidToken

from backend.config._config import SystemConfig
from backend.state import system_config
from backend.utils.auth import get_db
from backend.utils.crypto import ConfigEncryptionUnavailable, decrypt_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache constants (R3/R10)
# ---------------------------------------------------------------------------

_TTL = 30          # seconds
_MAX_ENTRIES = 1000

# Cache stores (user_id -> (expiry_timestamp, Optional[dict]))
_user_cache: dict[str, tuple[float, Optional[dict]]] = {}

# Per-user locks to prevent thundering herd on cold cache entries.
_user_locks: dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def invalidate_cache(user_id: str) -> None:
    """Invalidate the cached per-user API config for a single user."""
    _user_cache.pop(user_id, None)
    _user_locks.pop(user_id, None)


def invalidate_all() -> None:
    """Clear the entire per-user cache (call when global config changes)."""
    _user_cache.clear()
    _user_locks.clear()


def _evict_if_needed() -> None:
    """Evict the oldest 25% of entries when cache exceeds _MAX_ENTRIES."""
    if len(_user_cache) < _MAX_ENTRIES:
        return
    count_to_evict = max(1, _MAX_ENTRIES // 4)
    # Sort by expiry ascending; entries expiring soonest are evicted first.
    sorted_keys = sorted(_user_cache.keys(), key=lambda k: _user_cache[k][0])
    for key in sorted_keys[:count_to_evict]:
        _user_cache.pop(key, None)
        _user_locks.pop(key, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _get_cached_user_api(user_id: str) -> Optional[dict]:
    """Return the per-user API config dict, using the cache when fresh.

    Fetches from DB on cache miss, decrypts stored secrets, and stores the
    result in the cache with a TTL of ``_TTL`` seconds.

    Returns ``None`` if the user has no row in ``user_llm_config``.
    """
    now = time.time()

    # Fast path: cache hit (no lock needed).
    if user_id in _user_cache:
        expiry, data = _user_cache[user_id]
        if now < expiry:
            return data

    # Slow path: acquire per-user lock to prevent thundering herd.
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()

    async with _user_locks[user_id]:
        # Double-check inside the lock: another coroutine may have filled the
        # cache while we were waiting to acquire it.
        now = time.time()
        if user_id in _user_cache:
            expiry, data = _user_cache[user_id]
            if now < expiry:
                return data

        # Cache miss: fetch from DB.
        db = await get_db()
        row = await db.get_user_llm_config(user_id)

        if row is None:
            result = None
        else:
            try:
                api_key = decrypt_value(row["encrypted_api_key"]) if row["encrypted_api_key"] else ""
                proxy_url = decrypt_value(row["encrypted_proxy_url"]) if row["encrypted_proxy_url"] else ""
            except (ConfigEncryptionUnavailable, InvalidToken) as e:
                logger.error(
                    "Failed to decrypt user %s LLM config: %s; falling back to global",
                    user_id,
                    e,
                )
                api_key = ""
                proxy_url = ""
            result = {
                "api_key": api_key,
                "llm_base_url": row["llm_base_url"],
                "proxy_url": proxy_url,
            }

        # Store in cache (evict stale entries first if needed).
        _evict_if_needed()
        _user_cache[user_id] = (now + _TTL, result)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_config_for_user(user_id: str) -> Optional[SystemConfig]:
    """Return the SystemConfig that the given user should use.

    Resolution rules:
    - If no ``system_config["current"]`` is set AND the user has no per-user
      ``api_key``, return ``None``.
    - If no global config is set but the user has per-user credentials, create
      a default ``SystemConfig`` and overlay the per-user API credentials.
    - If the user has no per-user ``api_key``, return a shallow copy of the
      global config with a fresh ``api`` sub-model to prevent callers from
      mutating the singleton.
    - Otherwise, copy the global config and overlay the per-user
      API credentials (``api_key``, ``llm_base_url``, ``proxy_url``).
    """
    base = system_config.get("current")
    user_api = await _get_cached_user_api(user_id) if user_id else None

    # Neither global nor per-user config available.
    if base is None and (not user_api or not user_api.get("api_key")):
        return None

    if base is None:
        base = SystemConfig()

    if not user_api or not user_api.get("api_key"):
        return base.model_copy(update={"api": base.api.model_copy()})

    return _overlay_user_api(base, user_api)


def _overlay_user_api(base: SystemConfig, user_api: dict) -> SystemConfig:
    """Return a copy of *base* with per-user API credentials overlaid."""
    return base.model_copy(update={"api": base.api.model_copy(update={
        "api_key": user_api["api_key"],
        "llm_base_url": user_api.get("llm_base_url") or base.api.llm_base_url,
        "proxy_url": user_api.get("proxy_url") or base.api.proxy_url,
    })})
