import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import ExplorationReport

SRC = Path("backend/agents/pi_agent.py").read_text()


def _make_pi_agent():
    from unittest.mock import MagicMock
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def test_score_prompt_research_candidates_carry_scored_fields():
    # The output schema for research_direction_candidates must include scored fields,
    # not the legacy expected_innovation/expected_novelty triple.
    block = SRC.split("def score_directions_with_evidence", 1)[1].split("async def", 1)[0]
    template = block.split('"research_direction_candidates"', 1)[1].split('"raw_direction_candidates"', 1)[0]
    assert "direction_id" in template
    assert "overall_priority" in template
    assert "evidence_ref" in template
    assert "functional_impact" in template


def test_score_prompt_uses_spec_biology_expectedness_enum():
    block = SRC.split("def score_directions_with_evidence", 1)[1].split("async def", 1)[0]
    assert "partially_expected" in block
    assert "unexpected" in block


def test_score_prompt_contains_spec_scoring_rules():
    block = SRC.split("def score_directions_with_evidence", 1)[1].split("async def", 1)[0]
    for phrase in [
        "complementary",
        "high data reliability + high data novelty",
        "reconcile",
        "negative_validation_confidence=medium",
    ]:
        assert phrase in block, f"missing scoring-rule phrase: {phrase}"


@pytest.mark.asyncio
async def test_scored_candidates_land_in_research_direction_candidates(monkeypatch):
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        # Simulate an LLM that (wrongly) puts scored candidates under raw_direction_candidates
        # and leaves research_direction_candidates legacy-shaped. Normalization must recover.
        return json.dumps({
            "cluster_overview": "c",
            "research_direction_candidates": [
                {"direction": "IFN network", "expected_innovation": "High"}
            ],
            "raw_direction_candidates": [
                {"direction_id": "dir_001", "direction": "IFN network",
                 "overall_priority": "high", "data_reliability": "high",
                 "literature_novelty": "high", "functional_impact": "high"}
            ],
            "direction_dependency_graph": {},
        })

    monkeypatch.setattr(agent, "chat", fake_chat)
    state = SimpleNamespace(name="s", initial_instruction="x", species="mouse",
                            annotation_state=None)
    round_result = SimpleNamespace(
        bioinformatics_findings=[], pathology_findings=[], biochemistry_findings=[],
        literature_findings=[], pi_summary="")
    raw = [{"direction_id": "dir_001", "direction": "IFN network", "key_genes": ["Stat1"]}]
    evidence = [{"direction_id": "dir_001", "direction": "IFN network", "status": "completed"}]

    report = await agent.score_directions_with_evidence(state, round_result, raw, evidence)

    final = report.research_direction_candidates[0]
    assert final.get("overall_priority") == "high"
    assert final.get("direction_id") == "dir_001"
    assert final.get("evidence_ref") == "dir_001"
    # raw_direction_candidates must hold the ORIGINAL raw input, not the scored set
    assert report.raw_direction_candidates[0].get("key_genes") == ["Stat1"]
