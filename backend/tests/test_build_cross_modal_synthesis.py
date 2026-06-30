import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import AnalysisState, RoundResult, CrossModalSynthesis
from backend.agents.pi_agent import PIAgent


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _state():
    s = AnalysisState(user_id="u1", name="DKO liver", initial_instruction="analyze")
    s.round_results.append(RoundResult(round_number=1, pi_summary="round1 summary"))
    return s


_LLM_OK = json.dumps({
    "matrix_markdown": "| theme | biochem |\n|---|---|\n| iron | Fe 2.2x |",
    "candidates": [{
        "theme": "iron-associated oxidative injury",
        "mechanism_statement": "Iron accumulation drives lipid peroxidation",
        "supporting_evidence": [
            {"source": "biochemistry", "evidence": "Fe 2.2x", "strength": "moderate"},
            {"source": "bioinformatics", "evidence": "Hmox1 up", "strength": "weak"},
        ],
        "evidence_consistency": "medium",
        "confidence": "low",
        "novelty_relative_to_current_direction": "high",
        "relation_to_current_direction": "uncovered",
        "missing_evidence": ["4-HNE not measured"],
        "report_recommendation": "future_direction",
    }],
    "excluded_or_low_support_themes": ["unrelated theme"],
})


@pytest.mark.asyncio
async def test_parses_matrix_and_candidates():
    agent = _make_pi_agent()
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=_LLM_OK):
        cms = await agent.build_cross_modal_synthesis(
            state=_state(),
            current_direction="copper metabolic collapse",
            current_agent_raw={"biochemistry": {"findings": ["Fe 2.2x"]}},
            literature_context="PMID 38963329 links Cu deficiency to ferroptosis",
            biochemistry_raw="Iron: 2.2x (ref 1.0)",
        )
    assert isinstance(cms, CrossModalSynthesis)
    assert len(cms.candidates) == 1
    assert cms.candidates[0].confidence == "low"
    assert cms.excluded_or_low_support_themes == ["unrelated theme"]


@pytest.mark.asyncio
async def test_wider_inputs_in_prompt():
    agent = _make_pi_agent()
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=_LLM_OK) as mock_chat:
        await agent.build_cross_modal_synthesis(
            state=_state(),
            current_direction="dir",
            current_agent_raw={"bioinformatics": {"findings": ["Hmox1 up"]}},
            literature_context="UNIQUE_LIT_MARKER",
            biochemistry_raw="UNIQUE_BIOCHEM_MARKER",
        )
    prompt = mock_chat.call_args[0][0]
    assert "UNIQUE_LIT_MARKER" in prompt
    assert "UNIQUE_BIOCHEM_MARKER" in prompt
    assert "Hmox1 up" in prompt  # raw agent dict, not just findings string


@pytest.mark.asyncio
async def test_empty_inputs_graceful():
    agent = _make_pi_agent()
    state = AnalysisState(user_id="u1", name="n", initial_instruction="i")
    with patch.object(agent, "chat", new_callable=AsyncMock,
                      return_value=json.dumps({"matrix_markdown": "", "candidates": [], "excluded_or_low_support_themes": []})):
        cms = await agent.build_cross_modal_synthesis(
            state=state, current_direction="", current_agent_raw={},
            literature_context="", biochemistry_raw="",
        )
    assert cms.candidates == []
    assert cms.matrix_markdown == ""


@pytest.mark.asyncio
async def test_unparseable_response_keeps_raw_capped():
    agent = _make_pi_agent()
    junk = "x" * 30000
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=junk):
        cms = await agent.build_cross_modal_synthesis(
            state=_state(), current_direction="d", current_agent_raw={},
            literature_context="", biochemistry_raw="",
        )
    assert cms.candidates == []
    assert len(cms.matrix_markdown) == 20000


@pytest.mark.asyncio
async def test_skips_unparseable_candidate_entries():
    agent = _make_pi_agent()
    mixed = json.dumps({
        "matrix_markdown": "| t |\n|---|\n| x |",
        "candidates": [
            {"theme": "ok", "mechanism_statement": "m", "confidence": "low",
             "relation_to_current_direction": "uncovered", "report_recommendation": "secondary"},
            {"theme": 123},  # invalid — missing required fields / wrong types
        ],
        "excluded_or_low_support_themes": [],
    })
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=mixed):
        cms = await agent.build_cross_modal_synthesis(
            state=_state(), current_direction="d", current_agent_raw={},
            literature_context="", biochemistry_raw="",
        )
    assert len(cms.candidates) == 1
    assert cms.candidates[0].theme == "ok"
    assert cms.matrix_markdown.startswith("| t |")
