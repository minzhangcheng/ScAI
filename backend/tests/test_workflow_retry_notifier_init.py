from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.config._config import SystemConfig
from backend.models.schemas import AgentType, AnalysisState
from backend.tests.split_facade import setattr_facade
from backend.workflow.core import SingleCellAnalysisWorkflow


class _FakeOpenRouter:
    def __init__(self):
        self.on_retry = None
        self.before_retry = None


class _FakeLLMManager:
    def __init__(self, *args, **kwargs):
        self.openrouter = _FakeOpenRouter()


class _FakeSandbox:
    last_runtime_budgets = None

    def __init__(self, *args, **kwargs):
        type(self).last_runtime_budgets = kwargs.get("runtime_budgets")


class _FakeCheckpointer:
    @classmethod
    def from_conn_string(cls, *args, **kwargs):
        return cls()

    async def setup(self):
        return None


class _FakeDeepResearchClient:
    def __init__(self, *args, **kwargs):
        self.on_retry = None
        self.before_retry = None

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_regular_runtime_budgets_are_kept(
    monkeypatch,
    tmp_path,
):
    setattr_facade(monkeypatch, "backend.workflow.core", "LLMManager", _FakeLLMManager)
    setattr_facade(monkeypatch, "backend.workflow.core", "LocalJupyterSandbox", _FakeSandbox)
    setattr_facade(monkeypatch, "backend.workflow.core", "AsyncSqliteSaver", _FakeCheckpointer)
    setattr_facade(
        monkeypatch, "backend.workflow.core", "get_session_states",
        lambda session_id: tmp_path / "states",
    )
    setattr_facade(
        monkeypatch, "backend.workflow.core", "get_session_workspace",
        lambda session_id: tmp_path / "workspace",
    )
    gib = 1024 ** 3
    monkeypatch.setattr(
        "backend.utils.runtime_budget.detect_memory_capacity_bytes",
        lambda: (64 * gib, "test_capacity"),
    )
    monkeypatch.setattr(
        SingleCellAnalysisWorkflow,
        "ensure_agent_ready",
        lambda self, agent_name: None,
    )
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "_build_graph", lambda self: None)

    state = AnalysisState(
        id="snapshot-disabled-runtime-budgets",
        name="Snapshot Disabled Runtime Budgets",
        user_id="user-1",
        initial_instruction="test",
    )
    config = SystemConfig(
        runtime_max_rss_bytes=2 * gib,
    )
    workflow = SingleCellAnalysisWorkflow(config=config, session_id=state.id)
    workflow._state = state
    workflow._db_conn = MagicMock()
    workflow._db_conn.close = AsyncMock()

    await workflow.initialize()

    assert _FakeSandbox.last_runtime_budgets["max_rss_bytes"] == 2 * gib
    assert "max_namespace_count" not in _FakeSandbox.last_runtime_budgets
    assert "max_single_object_bytes" not in _FakeSandbox.last_runtime_budgets
    assert "max_pkl_bytes" not in _FakeSandbox.last_runtime_budgets
    assert "max_save_seconds" not in _FakeSandbox.last_runtime_budgets
    assert _FakeSandbox.last_runtime_budgets["effective_limit_basis"] == "test_capacity"


@pytest.mark.asyncio
async def test_preconfigured_deep_research_client_gets_retry_callback_before_agent_creation(monkeypatch, tmp_path):
    setattr_facade(monkeypatch, "backend.workflow.core", "LLMManager", _FakeLLMManager)
    setattr_facade(monkeypatch, "backend.workflow.core", "LocalJupyterSandbox", _FakeSandbox)
    setattr_facade(monkeypatch, "backend.workflow.core", "AsyncSqliteSaver", _FakeCheckpointer)
    setattr_facade(monkeypatch, "backend.workflow.core", "get_session_states", lambda session_id: tmp_path / "states")
    setattr_facade(monkeypatch, "backend.workflow.core", "get_session_workspace", lambda session_id: tmp_path / "workspace")
    setattr_facade(monkeypatch, "backend.agents.literature_agent", "OpenAICompatibleClient", _FakeDeepResearchClient)
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "ensure_agent_ready", lambda self, agent_name: None)
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "_build_graph", lambda self: None)

    state = AnalysisState(
        id="retry-init-session",
        name="Retry Init Session",
        user_id="user-1",
        initial_instruction="test",
    )
    state.literature_config = {
        "mode": "deep_research",
        "deep_research_model": "dr-model",
        "deep_research_llm_base_url": "https://dr.example/v1",
        "deep_research_api_key": "dr-key",
        "google_api_key": "",
        "pubmed_email": "",
        "max_search_rounds": 5,
    }

    workflow = SingleCellAnalysisWorkflow(config=SystemConfig(), session_id=state.id)
    workflow._state = state
    workflow._db_conn = MagicMock()
    workflow._db_conn.close = AsyncMock()

    await workflow.initialize()

    assert workflow.literature_agent._deep_research_llm_client is not None
    assert workflow.literature_agent._deep_research_llm_client.on_retry is workflow.llm_manager.openrouter.on_retry
    assert workflow.literature_agent._deep_research_llm_client.before_retry is workflow.llm_manager.openrouter.before_retry


@pytest.mark.asyncio
async def test_retry_notice_is_labelled_with_the_agent_that_stalled(monkeypatch, tmp_path):
    """The retry notice is the only trace of an upstream stall the user sees.

    It used to be hardcoded to BIOINFORMATICS, so a pathology vision request
    timing out sent the user to the wrong agent's logs.
    """
    setattr_facade(monkeypatch, "backend.workflow.core", "LLMManager", _FakeLLMManager)
    setattr_facade(monkeypatch, "backend.workflow.core", "LocalJupyterSandbox", _FakeSandbox)
    setattr_facade(monkeypatch, "backend.workflow.core", "AsyncSqliteSaver", _FakeCheckpointer)
    setattr_facade(monkeypatch, "backend.workflow.core", "get_session_states", lambda session_id: tmp_path / "states")
    setattr_facade(monkeypatch, "backend.workflow.core", "get_session_workspace", lambda session_id: tmp_path / "workspace")
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "ensure_agent_ready", lambda self, agent_name: None)
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "_build_graph", lambda self: None)

    state = AnalysisState(
        id="retry-label-session",
        name="Retry Label Session",
        user_id="user-1",
        initial_instruction="test",
    )

    workflow = SingleCellAnalysisWorkflow(config=SystemConfig(), session_id=state.id)
    workflow._state = state
    workflow._db_conn = MagicMock()
    workflow._db_conn.close = AsyncMock()

    await workflow.initialize()

    sent = []

    async def capture(msg):
        sent.append(msg)

    workflow.message_callback = capture
    on_retry = workflow.llm_manager.openrouter.on_retry

    await on_retry(0, 4, 2.0, "TimeoutError", "pathology")
    await on_retry(1, 4, 4.0, "TimeoutError", "biochemistry")
    await on_retry(2, 4, 6.0, "TimeoutError", "")

    assert [msg.agent for msg in sent] == [
        AgentType.PATHOLOGY,
        AgentType.BIOCHEMISTRY,
        None,
    ]
    assert "LLM request retry 1/5 (TimeoutError, waiting 2s)" in sent[0].content
