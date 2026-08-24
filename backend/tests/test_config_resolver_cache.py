"""A1 per-user configuration cache invalidation tests."""

import asyncio

import pytest

import backend.utils.config_resolver as resolver


@pytest.mark.asyncio
async def test_invalidation_during_read_cannot_publish_stale_config(monkeypatch):
    resolver.invalidate_all()
    entered = asyncio.Event()
    release = asyncio.Event()
    current = {"api_key": "old-key"}

    class _DB:
        async def get_user_llm_config(self, _user_id):
            entered.set()
            await release.wait()
            return {
                "encrypted_api_key": current["api_key"],
                "encrypted_proxy_url": "",
                "llm_base_url": "https://provider.example/v1",
            }

    async def get_db():
        return _DB()

    monkeypatch.setattr(resolver, "get_db", get_db)
    monkeypatch.setattr(resolver, "decrypt_value", lambda value: value)

    first = asyncio.create_task(resolver._get_cached_user_api("u1"))
    await entered.wait()
    resolver.invalidate_cache("u1")
    current["api_key"] = "new-key"
    release.set()
    await first

    refreshed = await resolver._get_cached_user_api("u1")
    assert refreshed["api_key"] == "new-key"
