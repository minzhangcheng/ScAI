"""Synthesized-fixture regression: verify honest evidence grading shape.
Uses anonymized round_results + a mocked LLM response (no real session IDs/PMIDs).
Asserts grading contract, NOT specific mechanism content."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import AnalysisState, RoundResult, AnalysisPhase
from backend.agents.pi_agent import PIAgent


def _make_pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _fixture_state():
    s = AnalysisState(user_id="u1", name="Anon DKO study", initial_instruction="analyze")
    s.analysis_phase = AnalysisPhase.INNOVATION
    s.current_round_direction = "metal-import failure metabolic collapse"
    s.round_results.append(RoundResult(
        round_number=1,
        biochemistry_findings=["Metal A depleted ~80%", "Metal B accumulated ~2.2x"],
        bioinformatics_findings=["Oxidative-stress proxy score elevated", "GeneX/GeneY upregulated"],
        pathology_findings=["Cell ballooning", "Fibrosis"],
        literature_findings=["A pathway linking metal imbalance to a cell-death program"],
        next_steps=["[Pathology] measure lipid peroxidation marker", "[Bioinformatics] assay cell-death markers"],
        pi_summary="metabolic collapse narrative",
    ))
    return s


# Mock LLM output that grades honestly: a low/medium candidate built largely from
# suggested future experiments → must carry missing_evidence + future-direction/secondary.
_HONEST = json.dumps({
    "matrix_markdown": "| theme | biochem | bioinfo | path | lit | consistency |\n|---|---|---|---|---|---|\n| metal-associated oxidative injury | B 2.2x | proxy up | ballooning | pathway | medium |",
    "candidates": [{
        "theme": "metal-associated oxidative injury",
        "mechanism_statement": "Accumulated metal B may drive oxidative cell death",
        "supporting_evidence": [
            {"source": "biochemistry", "evidence": "Metal B 2.2x", "strength": "moderate"},
            {"source": "bioinformatics", "evidence": "oxidative-stress proxy elevated", "strength": "weak"},
        ],
        "evidence_consistency": "medium",
        "confidence": "low",
        "novelty_relative_to_current_direction": "high",
        "relation_to_current_direction": "uncovered",
        "missing_evidence": ["lipid peroxidation not measured", "cell-death markers not assayed"],
        "report_recommendation": "future_direction"
    }],
    "excluded_or_low_support_themes": []
})


@pytest.mark.asyncio
async def test_honest_grading_shape():
    agent = _make_pi_agent()
    with patch.object(agent, "chat", new_callable=AsyncMock, return_value=_HONEST):
        cms = await agent.build_cross_modal_synthesis(
            state=_fixture_state(),
            current_direction="metal-import failure metabolic collapse",
            current_agent_raw={"biochemistry": {"findings": ["Metal B 2.2x"]}},
            literature_context="A pathway linking metal imbalance to a cell-death program",
            biochemistry_raw="Metal A: -80%; Metal B: 2.2x",
        )
    # (a) at least one low/medium candidate
    graded = [c for c in cms.candidates if c.confidence in ("low", "medium")]
    assert graded, "expected at least one low/medium candidate"
    c = graded[0]
    # (b) non-empty missing_evidence
    assert c.missing_evidence
    # (c) recommendation is secondary or future_direction
    assert c.report_recommendation in ("secondary", "future_direction")
    # Do NOT assert on theme/mechanism content.
