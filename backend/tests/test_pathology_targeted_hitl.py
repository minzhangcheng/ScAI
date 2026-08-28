"""targeted_analysis must route its LLM call through _call_llm_with_hitl
so that LLM failures surface proper HITL intervention (retry/skip/terminate)
instead of silently failing or propagating raw exceptions."""
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
import os

import backend.agents.pathology_agent as mod
from backend.agents.pathology_agent import PathologyAgent, _ImageSkipRequested
from backend.models.schemas import AgentType, DataFile
from backend.workflow.helpers import WorkflowTerminatedError
from backend.tests.split_facade import setattr_facade


def _bare_agent() -> PathologyAgent:
    agent = object.__new__(PathologyAgent)
    agent.llm_intervention_callback = None
    agent.agent_type = AgentType.PATHOLOGY
    agent._provider = MagicMock()
    agent._provider.get_config_revision = MagicMock(return_value=0)
    agent._last_config_revision = 0
    agent.tile_size = 224
    return agent


@pytest.mark.asyncio
async def test_targeted_analysis_uses_hitl(monkeypatch):
    agent = _bare_agent()

    called = {"hitl": False}

    async def fake_hitl(call_factory, image_label, step_label):
        called["hitl"] = True
        return SimpleNamespace(content="mocked response")

    # Bypass setup calls so the test reaches the actual LLM call site
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(agent, "refresh_runtime_config_if_needed", lambda: None)
    monkeypatch.setattr(agent, "_call_llm_with_hitl", fake_hitl)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    setattr_facade(
        monkeypatch,
        mod.__name__,
        "process_pathology_image",
        lambda *a, **k: ["tile1", "tile2"],
    )

    image_info = DataFile(
        id="img-1",
        filename="test.tif",
        file_path="/fake/test.tif",
        section="pathology",
        data_type="H&E",
    )

    result = await agent.targeted_analysis(
        image_info=image_info,
        specific_question="Is there fibrosis?"
    )

    assert called["hitl"], "_call_llm_with_hitl was never called"


def _targeted_image():
    return DataFile(
        id="img-1",
        filename="test.tif",
        file_path="/fake/test.tif",
        section="pathology",
        data_type="H&E",
    )


def _patch_targeted_setup(monkeypatch, agent):
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(agent, "refresh_runtime_config_if_needed", lambda: None)
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    setattr_facade(
        monkeypatch,
        mod.__name__,
        "process_pathology_image",
        lambda *args, **kwargs: ["tile"],
    )


@pytest.mark.asyncio
async def test_targeted_analysis_records_skip_without_generic_error(monkeypatch):
    agent = _bare_agent()
    _patch_targeted_setup(monkeypatch, agent)

    async def raise_skip(*args, **kwargs):
        raise _ImageSkipRequested("test.tif")

    monkeypatch.setattr(agent, "_call_llm_with_hitl", raise_skip)
    result = await agent.targeted_analysis(_targeted_image(), "question")

    assert result["status"] == "skipped"
    assert result["skipped"] is True
    assert "error" not in result


@pytest.mark.asyncio
async def test_targeted_analysis_propagates_terminate(monkeypatch):
    agent = _bare_agent()
    _patch_targeted_setup(monkeypatch, agent)

    async def raise_terminate(*args, **kwargs):
        raise WorkflowTerminatedError("user terminated")

    monkeypatch.setattr(agent, "_call_llm_with_hitl", raise_terminate)
    with pytest.raises(WorkflowTerminatedError):
        await agent.targeted_analysis(_targeted_image(), "question")
