from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.config._config import SystemConfig
from backend.workflow.core import SingleCellAnalysisWorkflow


class _FakeLLMManager:
    def __init__(self, *args, **kwargs):
        self.openrouter = MagicMock()
        self.openrouter.on_retry = None


class _FakeSandbox:
    def __init__(self, *args, **kwargs):
        return None


class _FakeCheckpointer:
    @classmethod
    def from_conn_string(cls, *args, **kwargs):
        return cls()

    async def setup(self):
        return None


@pytest.mark.asyncio
async def test_pause_keeps_plan_review_action(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.workflow.core.LLMManager", _FakeLLMManager)
    monkeypatch.setattr("backend.workflow.core.LocalJupyterSandbox", _FakeSandbox)
    monkeypatch.setattr("backend.workflow.core.AsyncSqliteSaver", _FakeCheckpointer)
    monkeypatch.setattr("backend.workflow.core.get_session_states", lambda session_id: tmp_path / "states")
    monkeypatch.setattr("backend.workflow.core.get_session_workspace", lambda session_id: tmp_path / "workspace")
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "ensure_agent_ready", lambda self, agent_name: None)
    monkeypatch.setattr(SingleCellAnalysisWorkflow, "_build_graph", lambda self: None)

    from backend.models.schemas import AnalysisState

    workflow = SingleCellAnalysisWorkflow(config=SystemConfig(), session_id="test-pause-session")

    state = AnalysisState(
        id="test-pause-session",
        name="Test Pause Session",
        user_id="user-1",
        initial_instruction="test",
    )
    workflow._state = state
    workflow._db_conn = MagicMock()
    workflow._db_conn.close = AsyncMock()

    # Mock the compiled_graph.aget_state to return a minimal state
    fake_state = MagicMock()
    fake_state.values = {}
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=fake_state)
    workflow.compiled_graph = mock_graph

    await workflow.initialize()

    workflow._plan_review_action = {"action": "approve"}
    await workflow.pause(thread_id="test-thread")

    assert workflow._plan_review_action == {"action": "approve"}
