"""Tests for the admin /config/validate-saved endpoint (L9).

Admins whose browser has no local api_key (localStorage no longer persists it)
must be able to validate the key already saved in the global config, instead of
POSTing an empty key to /config/validate.
"""
import pytest

from backend.routes import config as config_mod
from backend.config._config import SystemConfig


@pytest.mark.asyncio
async def test_validate_saved_no_global_config(monkeypatch):
    monkeypatch.setitem(config_mod.system_config, "current", None)
    result = await config_mod.validate_saved_config(_="admin")
    assert result["valid"] is False
    assert "No global API key" in result["message"]


@pytest.mark.asyncio
async def test_validate_saved_empty_key(monkeypatch):
    cfg = SystemConfig()
    cfg.api.api_key = ""
    monkeypatch.setitem(config_mod.system_config, "current", cfg)
    result = await config_mod.validate_saved_config(_="admin")
    assert result["valid"] is False


@pytest.mark.asyncio
async def test_validate_saved_valid_key(monkeypatch):
    cfg = SystemConfig()
    cfg.api.api_key = "sk-real"
    monkeypatch.setitem(config_mod.system_config, "current", cfg)

    async def _fake_validate(api_key, base_url, proxy_url):
        assert api_key == "sk-real"  # validates the SAVED key, not a request body
        return {"valid": True}

    monkeypatch.setattr(config_mod, "validate_llm_credentials", _fake_validate)
    result = await config_mod.validate_saved_config(_="admin")
    assert result["valid"] is True
