import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.tests.split_facade import setattr_facade


class _Provider:
    def __init__(self, literature_config):
        self._literature_config = literature_config
        self._retry_callback = None

    def get_literature_config(self):
        return self._literature_config

    def get_model(self, agent_name):
        return "model"

    def get_system_prompt(self, agent_name):
        return "prompt"

    def get_temperature(self, agent_name):
        return 0.5

    def get_vision_setting(self, agent_name):
        return True

    def get_llm_client(self, agent_name):
        return None

    def get_config_revision(self, agent_name):
        return 1

    def clear_agent_client_cache(self, agent_name):
        return None

    def apply_retry_callback(self, client):
        if client is not None and self._retry_callback:
            client.on_retry = self._retry_callback


class _ClosableClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _GoogleClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _OpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.on_retry = None

    async def close(self):
        return None


def test_sync_literature_config_missing_google_key_preserves_existing_google_client():
    provider = _Provider({"mode": "deep_research"})
    agent = LiteratureAgent(settings_provider=provider, google_api_key="existing-google")
    existing_client = _GoogleClient()
    agent._deep_research = existing_client

    agent._sync_literature_config_from_provider()

    assert agent._google_api_key == "existing-google"
    assert agent._deep_research is existing_client


def test_make_google_client_returns_google_deep_research_client(monkeypatch):
    created = []

    class FakeGoogleDeepResearchClient:
        def __init__(self, api_key, proxy_url=None):
            self.api_key = api_key
            self.proxy_url = proxy_url
            created.append(self)

    # Patch where the name is looked up: _make_google_client lives in the
    # literature mixin package, so binding it on the facade module has no effect.
    monkeypatch.setattr(
        "backend.agents.literature.clients.GoogleDeepResearchClient",
        FakeGoogleDeepResearchClient,
    )
    provider = _Provider({})
    agent = LiteratureAgent(settings_provider=provider, google_api_key="")
    agent._google_api_key = "google-key"
    agent.proxy_url = "http://proxy.example"

    client = agent._make_google_client()

    assert client is created[0]
    assert client.api_key == "google-key"
    assert client.proxy_url == "http://proxy.example"


def test_make_deep_research_client_inherits_provider_retry_callback(monkeypatch):
    def retry_callback(*args, **kwargs):
        return None

    provider = _Provider({})
    provider._retry_callback = retry_callback
    setattr_facade(
        monkeypatch,
        "backend.agents.literature_agent",
        "OpenAICompatibleClient",
        _OpenAIClient,
    )
    agent = LiteratureAgent(
        settings_provider=provider,
        deep_research_llm_base_url="https://dr.example/v1",
        deep_research_api_key="dr-key",
    )

    client = agent._make_deep_research_client()

    assert client.on_retry is retry_callback


def test_close_runtime_clients_closes_current_literature_owned_clients():
    import asyncio
    provider = _Provider({})
    agent = LiteratureAgent(settings_provider=provider)
    dr_client = _ClosableClient()
    google_client = _ClosableClient()
    agent._deep_research_llm_client = dr_client
    agent._deep_research = google_client

    asyncio.run(agent.close_runtime_clients())

    assert dr_client.closed is True
    assert google_client.closed is True
    assert agent._deep_research_llm_client is None
    assert agent._deep_research is None


def test_sync_literature_config_closes_replaced_clients(monkeypatch):
    # _make_deep_research_client lives in the literature mixin package, so the
    # replacement has to be bound across the whole split, not just the facade.
    setattr_facade(
        monkeypatch,
        "backend.agents.literature_agent",
        "OpenAICompatibleClient",
        _OpenAIClient,
    )
    provider = _Provider({
        "deep_research_llm_base_url": "https://new.example/v1",
        "deep_research_api_key": "new-key",
        "google_api_key": "new-google",
    })
    agent = LiteratureAgent(
        settings_provider=provider,
        deep_research_llm_base_url="https://old.example/v1",
        deep_research_api_key="old-key",
        google_api_key="old-google",
    )
    old_dr_client = _ClosableClient()
    old_google_client = _ClosableClient()
    agent._deep_research_llm_client = old_dr_client
    agent._deep_research = old_google_client

    closed_clients = []

    def fake_close_stale_client(client):
        closed_clients.append(client)

    monkeypatch.setattr(agent, "_close_stale_client", fake_close_stale_client)

    agent._sync_literature_config_from_provider()

    assert old_dr_client in closed_clients
    assert old_google_client in closed_clients
    assert agent._deep_research_llm_client is not old_dr_client
    assert agent._deep_research is not old_google_client


@pytest.mark.asyncio
async def test_close_stale_client_tracks_pending_task_until_runtime_cleanup():
    provider = _Provider({})
    agent = LiteratureAgent(settings_provider=provider)
    stale_client = _ClosableClient()

    agent._close_stale_client(stale_client)

    assert len(agent._client_close_tasks) == 1

    await agent.close_runtime_clients()

    assert stale_client.closed is True
    assert agent._client_close_tasks == set()
