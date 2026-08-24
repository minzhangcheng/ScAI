"""Biochemistry image outcomes preserve the distinction between no data and failure."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.biochemistry_agent import BiochemistryAgent
from backend.models.schemas import AgentType, AnalysisState, DataFile
from backend.utils.llm_client import UpstreamExhaustedError


def _agent(llm_client=None):
    agent = BiochemistryAgent.__new__(BiochemistryAgent)
    agent.agent_type = AgentType.BIOCHEMISTRY
    agent.session_id = "s"
    provider = MagicMock()
    provider.get_llm_client.return_value = llm_client
    provider.get_model.return_value = "m"
    provider.get_system_prompt.return_value = "sys"
    agent._provider = provider
    return agent


def _state():
    return AnalysisState(user_id="u", current_round=1)


@pytest.mark.asyncio
async def test_no_image_all_failed_and_partial_are_distinct(monkeypatch):
    agent = _agent()
    assert await agent._step2_process_images([], "instruction") is None

    monkeypatch.setattr(
        BiochemistryAgent,
        "_analyze_single_image",
        AsyncMock(side_effect=[None, "EXTRACTION_STATUS: usable\nfinding"]),
    )
    images = [
        DataFile(filename="failed.png", file_path="/nope/failed.png", section="biochemistry", data_type="image"),
        DataFile(filename="ok.png", file_path="/nope/ok.png", section="biochemistry", data_type="image"),
    ]
    agent.chat = AsyncMock(return_value=json.dumps({
        "image_findings": ["finding"],
        "observed_values": [],
        "image_mechanisms": [],
        "image_suggested_experiments": [],
    }))
    partial = await agent._step2_process_images(images, "instruction")
    assert partial["status"] == "partial"
    assert partial["failed_files"] == ["failed.png"]

    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", AsyncMock(return_value=None))
    failed = await agent._step2_process_images(images, "instruction")
    assert failed == {"status": "failed", "failed_files": ["failed.png", "ok.png"]}


@pytest.mark.asyncio
async def test_all_images_failing_reports_failed_not_no_data():
    agent = _agent()
    agent.chat = AsyncMock(side_effect=AssertionError("failure must not become no-data advice"))
    result = await agent._step3_summarize(
        None, {"status": "failed", "failed_files": ["biochem.png"]}, "instruction", _state()
    )
    assert result["status"] == "failed"
    assert result["image_analysis_failed"] is True
    assert result["findings"] == []
    assert "No biochemistry data provided" not in json.dumps(result)
    agent.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_indicators_survive_partial_image_failure_and_gap_is_stated():
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps({
        "findings": ["ALT elevated"],
        "abnormal_markers": [],
        "interpretation": "i",
        "suggested_experiments": [],
        "conclusions": [],
    }))
    result = await agent._step3_summarize(
        {"findings": ["ALT elevated"], "abnormal_markers": [], "mechanisms": [], "suggested_experiments": []},
        {"status": "partial", "failed_files": ["blot.png"], "image_findings": ["f"]},
        "instruction",
        _state(),
    )
    assert result["status"] == "completed"
    assert result["image_analysis_partial"] is True
    assert result["failed_files"] == ["blot.png"]
    assert "MISSING evidence" in agent.chat.await_args.args[0]


@pytest.mark.asyncio
async def test_step2_marks_failure_when_every_image_fails(monkeypatch):
    agent = _agent()
    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", AsyncMock(return_value=None))
    images = [DataFile(filename="a.png", file_path="/nope/a.png", section="biochemistry", data_type="image")]
    assert await agent._step2_process_images(images, "instruction") == {
        "status": "failed", "failed_files": ["a.png"]
    }


@pytest.mark.asyncio
async def test_upstream_exhaustion_propagates(tmp_path):
    from PIL import Image

    image_path = tmp_path / "biochem.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    record = DataFile(filename=image_path.name, file_path=str(image_path), section="biochemistry", data_type="image")
    client = MagicMock()
    client.chat_with_vision = AsyncMock(side_effect=UpstreamExhaustedError(
        "upstream gone", agent_name="biochemistry", model="m", llm_server="https://x/v1"
    ))
    agent = _agent(client)
    with pytest.raises(UpstreamExhaustedError):
        await agent._analyze_single_image(record, "instruction")
    assert record.extracted_data["error"]
