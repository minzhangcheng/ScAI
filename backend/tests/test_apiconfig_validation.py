import pytest
from backend.config._config import APIConfig


def test_proxy_url_rejects_garbage():
    with pytest.raises(Exception):
        APIConfig(api_key="k", proxy_url="not-a-url")


def test_proxy_url_accepts_http():
    APIConfig(api_key="k", proxy_url="http://127.0.0.1:8080")


def test_require_key_raises_on_empty():
    cfg = APIConfig(api_key="")
    with pytest.raises(ValueError, match="LLM api_key is not configured"):
        cfg.require_key()


def test_require_key_ok_with_key():
    cfg = APIConfig(api_key="sk-xxx")
    cfg.require_key()  # should not raise


@pytest.mark.asyncio
async def test_resolve_returns_none_when_no_key(monkeypatch):
    from backend.utils import config_resolver as cr
    from backend.config._config import SystemConfig
    base = SystemConfig()
    base.api.api_key = ""
    monkeypatch.setitem(cr.system_config, "current", base)
    async def _no_user(_uid): return None
    monkeypatch.setattr(cr, "_get_cached_user_api", _no_user)
    result = await cr.resolve_config_for_user("u1")
    assert result is None
