"""Regression tests for pathology tile-batch validation and retry."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.agents.pathology_agent as mod
from backend.agents.pathology_agent import (
    PathologyAgent,
)
from backend.models.schemas import AgentType, DataFile
from backend.tests.split_facade import setattr_facade


def _bare_agent() -> PathologyAgent:
    agent = object.__new__(PathologyAgent)
    agent.agent_type = AgentType.PATHOLOGY
    agent._provider = MagicMock()
    agent._provider.get_config_revision.return_value = 0
    agent._provider.get_model.return_value = "vision-model"
    agent._last_config_revision = 0
    agent.llm_intervention_callback = None
    agent.tile_size = 224
    agent.max_tiles = 2
    return agent


def _patch_image_setup(monkeypatch, agent: PathologyAgent) -> DataFile:
    monkeypatch.setattr(agent, "refresh_runtime_config_if_needed", lambda: None)
    monkeypatch.setattr(agent, "_set_llm_context", lambda: None)
    monkeypatch.setattr(mod.os.path, "exists", lambda path: True)
    setattr_facade(monkeypatch, mod.__name__, "load_image", lambda path: object())
    setattr_facade(
        monkeypatch,
        mod.__name__,
        "process_pathology_image",
        lambda *args, **kwargs: ["tile-a", "tile-b"],
    )
    setattr_facade(
        monkeypatch, mod.__name__, "create_overview_image", lambda image: "overview"
    )

    async def direct_hitl(call_factory, image_label, step_label):
        return await call_factory()

    monkeypatch.setattr(agent, "_call_llm_with_hitl", direct_hitl)
    return DataFile(
        id="img-1",
        filename="slide.tif",
        file_path="/fake/slide.tif",
        section="pathology",
        data_type="H&E",
    )


@pytest.mark.asyncio
async def test_invalid_tile_batch_retries_then_uses_only_validated_entries(monkeypatch):
    agent = _bare_agent()
    image = _patch_image_setup(monkeypatch, agent)
    prompts = []
    stream_preferences = []
    responses = [
        {"overall_structure": "Preserved", "severity": "normal"},
        {
            "tile_findings": [
                {"tile_index": 1, "findings": "Tile 1 findings..."},
                {"tile_index": 2, "findings": "Tile 2 findings..."},
            ]
        },
        {
            "tile_findings": [
                {"tile_index": 99, "findings": "No abnormality detected."},
                {"tile_index": 100, "findings": "Mild stromal expansion is visible."},
            ]
        },
    ]

    class FakeClient:
        async def chat_with_vision(self, **kwargs):
            prompts.append(kwargs["text_prompt"])
            stream_preferences.append(kwargs.get("prefer_stream"))
            return SimpleNamespace(content=json.dumps(responses.pop(0)))

    agent._provider.get_llm_client.return_value = FakeClient()

    async def synthesize(prompt, temperature=None):
        assert prompt.index("Reply in JSON format:") < prompt.index("## Overview Analysis")
        assert "Mild stromal expansion is visible." in prompt
        return json.dumps({
            "findings": ["Validated finding"],
            "severity_score": 1,
            "further_analysis": [],
            "suggested_staining": [],
        })

    monkeypatch.setattr(agent, "chat_without_history", synthesize)
    result = await agent._analyze_image(image, "Analyze")

    assert prompts[0].index("Please reply in JSON format:") < prompts[0].index("Analysis instruction:")
    assert len(prompts) == 3  # overview + invalid batch + corrected retry
    assert stream_preferences == [True, True, True]
    assert "previous response failed validation" in prompts[-1]
    assert [entry["tile_index"] for entry in result["tile_findings"]] == [0, 1]
    assert result["tile_validation"]["status"] == "completed"
    assert result["tile_validation"]["invalid_batches"] == []
