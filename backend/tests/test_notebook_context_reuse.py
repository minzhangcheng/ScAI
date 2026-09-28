"""Existing notebook context survives a new Agent and Chat's early startup."""

import logging
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics.context_mixin import ContextMixin
from backend.agents.bioinformatics.executor_restore import restore_executor_state
from backend.config import SystemConfig
from backend.models.schemas import AnalysisState, AnalysisStep, CodeExecution
from backend.routes.chat_parts import delegation, source_selection


def _notebook_cells():
    return [{
        "code": "print(adata.shape, list(adata.obs), list(adata.obsm))",
        "outputs": [{
            "type": "stream",
            "text": "(120, 30) ['cell_type', 'condition'] ['X_umap']",
        }],
    }]


@pytest.mark.asyncio
async def test_chat_prestarted_sandbox_supplies_history_without_new_receipts(
    monkeypatch, tmp_path,
):
    from backend.agents import bioinformatics_agent, literature_agent, pi_agent
    from backend.utils import sandbox as sandbox_module

    state = AnalysisState(id="chat-existing-notebook", user_id="owner")
    cells = []
    sandbox = SimpleNamespace(km=None, get_code_history=lambda: cells)

    async def start():
        sandbox.km = object()
        cells.extend(_notebook_cells())
        return {"started": True, "session_restored": True, "restore_failed": False}

    async def stop(**kwargs):
        sandbox.km = None

    sandbox.start = AsyncMock(side_effect=start)
    sandbox.stop = AsyncMock(side_effect=stop)

    class RecordingAgent(ContextMixin):
        def __init__(self, **kwargs):
            self.sandbox = kwargs["sandbox"]
            self.code_history = []

        def set_session(self, session_id):
            self.session_id = session_id

        async def run_autonomous_analysis(self, actual_state, instruction, max_steps, **options):
            assert actual_state is state
            assert options["run_kind"] == "chat"
            assert options["resume_from_checkpoint"] is False
            assert options["run_scope_id"].startswith("chat_")
            assert self.sandbox.km is not None
            restored = restore_executor_state(
                self, state, checkpoint=None, checkpoint_valid=False,
                static_system_prompt="test",
                dynamic_context_message={
                    "role": "user", "content": self._build_notebook_context(),
                },
                initial_instruction=instruction, max_steps=max_steps,
                bindings=SimpleNamespace(
                    logger=logging.getLogger(__name__),
                    tool_calls_are_paired=lambda _: True,
                    AnalysisStep=AnalysisStep, CodeExecution=CodeExecution,
                    build_approved_plan_context=MagicMock(),
                ),
            )
            return {
                "messages": restored.messages,
                "code_history": list(self.code_history),
                "findings": restored.findings, "figures": restored.figures,
            }

    monkeypatch.setattr("backend.utils.agent_llm.get_agent_llm_client", lambda *a, **kw: AsyncMock())
    monkeypatch.setattr(delegation, "resolve_config_for_user", AsyncMock(return_value=SystemConfig()))
    monkeypatch.setattr(delegation, "get_session_states", lambda _: tmp_path / "states")
    monkeypatch.setattr(delegation, "get_session_workspace", lambda _: tmp_path / "workspace")
    monkeypatch.setattr(delegation, "_collect_session_figures", lambda *a, **kw: [])
    monkeypatch.setattr(sandbox_module, "LocalJupyterSandbox", MagicMock(return_value=sandbox))
    monkeypatch.setattr(literature_agent, "LiteratureAgent", MagicMock())
    monkeypatch.setattr(delegation, "_persist_chat_terminal", AsyncMock(return_value=True))
    monkeypatch.setattr(pi_agent, "PIAgent", MagicMock(return_value=MagicMock(
        review_chat_execution=AsyncMock(return_value={"decision": "accept"}),
    )))
    monkeypatch.setattr(source_selection, "select_chat_source", AsyncMock(return_value={
        "selection": {"mode": "current"}, "active_adata_var": "adata",
        "rationale": "Use the restored result",
    }))
    monkeypatch.setattr(bioinformatics_agent, "BioinformaticsAgent", RecordingAgent)

    result, _ = await delegation._execute_chat_agent_inner(
        state.id, state, "bioinformatics", "Extend the existing analysis",
        MagicMock(), AsyncMock(), [],
    )

    context = result["messages"][1]["content"]
    assert "(120, 30) ['cell_type', 'condition'] ['X_umap']" in context
    assert "historical notebook cells" in context
    assert "Notebook is empty" not in context
    assert result["code_history"] == []
    assert result["findings"] == []
    assert result["figures"] == []
    sandbox.start.assert_awaited_once()
    sandbox.stop.assert_awaited_once_with(save=True)
    assert state.id not in delegation.chat_sandboxes


@pytest.mark.parametrize("saved_history", [[], [CodeExecution(id="original-id", code="x = 1").model_dump()]])
def test_checkpoint_restore_replaces_history_and_preserves_receipt_identity(saved_history):
    agent = SimpleNamespace(code_history=[CodeExecution(code="unrelated old cell")])
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "context"},
        {"role": "user", "content": "Analysis instruction: Resume this task"},
    ]
    restored = restore_executor_state(
        agent, AnalysisState(user_id="owner"),
        checkpoint={"messages": deepcopy(messages), "code_history": saved_history, "step_number": 3},
        checkpoint_valid=True, static_system_prompt="new system",
        dynamic_context_message={"role": "user", "content": "new context"},
        initial_instruction="unused on ordinary resume", max_steps=30,
        bindings=SimpleNamespace(
            logger=logging.getLogger(__name__), tool_calls_are_paired=lambda _: True,
            AnalysisStep=AnalysisStep, CodeExecution=CodeExecution,
            build_approved_plan_context=MagicMock(),
        ),
    )
    assert restored.messages == messages
    assert restored.step_count == 3
    assert [record.id for record in agent.code_history] == [item["id"] for item in saved_history]
