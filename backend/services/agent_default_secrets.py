"""Encrypted per-user credentials used when creating a new analysis."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from backend.utils import crypto
from backend.utils.db import AppDB
from backend.utils.secret_redaction import mask_api_key
from backend.utils.settings_provider import MASKED_SECRET_VALUE


_EMPTY: dict[str, Any] = {
    "agent_api_keys": {},
    "literature": {
        "deep_research_api_key": "",
        "google_api_key": "",
    },
}
_USER_LOCKS: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def agent_default_secrets_lock(user_id: str):
    """Serialize encrypted read-modify-write updates for one user."""
    lock = _USER_LOCKS.setdefault(user_id, asyncio.Lock())
    async with lock:
        yield


async def load_agent_default_secrets(db: AppDB, user_id: str) -> dict[str, Any]:
    encrypted = await db.get_user_agent_default_secrets(user_id)
    if not encrypted:
        return {"agent_api_keys": {}, "literature": dict(_EMPTY["literature"])}
    try:
        payload = json.loads(crypto.decrypt_value(encrypted))
    except (json.JSONDecodeError, TypeError):
        return {"agent_api_keys": {}, "literature": dict(_EMPTY["literature"])}
    if not isinstance(payload, dict):
        return {"agent_api_keys": {}, "literature": dict(_EMPTY["literature"])}
    agent_api_keys = payload.get("agent_api_keys")
    literature = payload.get("literature")
    return {
        "agent_api_keys": dict(agent_api_keys) if isinstance(agent_api_keys, dict) else {},
        "literature": dict(literature) if isinstance(literature, dict) else dict(_EMPTY["literature"]),
    }


async def save_agent_default_secrets(
    db: AppDB,
    user_id: str,
    payload: dict[str, Any],
) -> None:
    encrypted = crypto.encrypt_value(json.dumps(payload, ensure_ascii=False))
    await db.set_user_agent_default_secrets(user_id, encrypted)


def masked_agent_default_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_api_keys": {
            agent: mask_api_key(key)
            for agent, key in (payload.get("agent_api_keys") or {}).items()
            if key
        },
        "literature": {
            field: MASKED_SECRET_VALUE if (payload.get("literature") or {}).get(field) else ""
            for field in ("deep_research_api_key", "google_api_key")
        },
    }
