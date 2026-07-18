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


@pytest.mark.asyncio
async def test_integration_direction_candidates_parse_json_array(monkeypatch):
    agent = _make_pi_agent()
    captured_prompt = ""

    async def fake_chat(prompt, temperature=0.3):
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps([
            {
                "direction_id": "int_001",
                "direction": "Cross-direction stress program",
                "core_observation": "Round 2 and 3 share NRF2-linked signals.",
                "supporting_data": "Round 2 and 3 both show NRF2 activation.",
                "artifact_risks": "Single-library composition uncertainty.",
                "suggested_validation_analysis": "Residualize the shared score.",
            }
        ])

    monkeypatch.setattr(agent, "chat", fake_chat)
    state = SimpleNamespace(
        name="s",
        initial_instruction="x",
        species="mouse",
        user_chosen_directions=[],
        build_experimental_design_summary=lambda: "",
    )
    rr = SimpleNamespace(
        round_number=2,
        direction="D1",
        bioinformatics_findings=[],
        pathology_findings=[],
        biochemistry_findings=[],
        literature_findings=[],
        pi_summary="",
    )

    candidates = await agent.generate_integration_direction_candidates(state, [rr], "report", "lit")

    assert candidates[0]["direction_id"] == "int_001"
    assert candidates[0]["direction"] == "Cross-direction stress program"
    assert candidates[0]["supporting_data"] == [
        "Round 2 and 3 both show NRF2 activation."
    ]
    assert candidates[0]["artifact_risks"] == [
        "Single-library composition uncertainty."
    ]
    assert candidates[0]["validation_plan"]["next_analysis"] == [
        "Residualize the shared score."
    ]
    assert "arrays of strings (never a single string)" in captured_prompt
    assert "supporting_data: an array of objects" in captured_prompt


@pytest.mark.asyncio
async def test_integration_direction_scoring_parses_json_array(monkeypatch):
    agent = _make_pi_agent()

    async def fake_chat(prompt, temperature=0.3):
        return json.dumps([
            {
                "direction_id": "int_001",
                "direction": "Cross-direction stress program",
                "novelty_score": 8,
                "priority_rank": 1,
            }
        ])

    monkeypatch.setattr(agent, "chat", fake_chat)
    state = SimpleNamespace(name="s", initial_instruction="x")
    raw_candidates = [{
        "direction_id": "int_001",
        "direction": "Cross-direction stress program",
        "supporting_data": "Observed in rounds 2 and 3.",
        "artifact_risks": "Batch confounding.",
    }]

    scored = await agent.score_integration_directions_with_evidence(state, raw_candidates, [])

    assert scored[0]["direction_id"] == "int_001"
    assert scored[0]["novelty_score"] == 8
    # A score-only LLM response must retain and normalize the original evidence fields.
    assert scored[0]["direction"] == "Cross-direction stress program"
    assert scored[0]["supporting_data"] == ["Observed in rounds 2 and 3."]
    assert scored[0]["artifact_risks"] == ["Batch confounding."]


def test_candidate_specific_core_observation_requires_data_element():
    assert PIAgent._candidate_has_specific_core_observation({
        "core_observation": "Cluster 7 hepatocytes show Mt2 log2FC=1.3 and NRF2 score correlation rho=0.38."
    }) is True
    assert PIAgent._candidate_has_specific_core_observation({
        "core_observation": "The data suggest interesting patterns."
    }) is False


def test_direction_guardrails_downgrade_generic_core_observation_and_premise_conflict():
    scored = [{
        "direction_id": "dir_001",
        "direction": "Generic direction",
        "core_observation": "The data suggest interesting patterns.",
        "data_reliability": "high",
        "artifact_risk": "low",
        "data_sufficiency": "sufficient",
        "recommendation": "prioritize",
        "overall_priority": "high",
        "contradicts_stated_premise": True,
        "risk_flags": [],
    }]

    guarded = PIAgent._apply_direction_guardrails(scored, [], [])

    assert guarded[0]["recommendation"] == "needs_more_data"
    assert "generic_core_observation" in guarded[0]["risk_flags"]
    assert "contradicts_premise" in guarded[0]["risk_flags"]


def test_direction_guardrails_conservative_when_literature_timed_out():
    scored = [{
        "direction_id": "dir_001",
        "direction": "Timed out direction",
        "core_observation": "Cluster 7 hepatocytes show Mt2 log2FC=1.3.",
        "data_reliability": "high",
        "artifact_risk": "low",
        "data_sufficiency": "sufficient",
        "literature_novelty": "high",
        "recommendation": "prioritize",
        "risk_flags": [],
    }]
    evidence = [{"direction_id": "dir_001", "novelty_status": "unknown_due_to_timeout"}]

    guarded = PIAgent._apply_direction_guardrails(scored, [], evidence)

    assert guarded[0]["literature_novelty"] == "unknown"
    assert guarded[0]["recommendation"] == "needs_more_data"
    assert "literature_evidence_timeout" in guarded[0]["risk_flags"]


def test_critical_direction_fields_missing_lists_absent_fields():
    missing = PIAgent._critical_direction_fields_missing({
        "core_observation": "Cluster 7 shows Mt2 log2FC=1.3.",
        "recommendation": "prioritize",
    })

    assert "alternative_hypotheses" in missing
    assert "artifact_risks" in missing
    assert "validation_plan" in missing
