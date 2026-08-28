import asyncio

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import AnalysisState


class _PromptHarness:
    def __init__(self):
        self.prompt = ""

    def reset_history(self):
        return None

    async def chat(self, prompt, temperature=None):
        self.prompt = prompt
        return "Integrated result"


def test_partial_integration_prompt_receives_findings_and_evidence_gaps():
    harness = _PromptHarness()
    state = AnalysisState(name="t", user_id="u", initial_instruction="o")
    directions = [{
        "direction": "Direction A",
        "notes": "Selected at the gate",
        "round_number": 2,
        "round_summary": "Retained summary",
        "round_status": "partial",
        "evidence_status": "substantive",
        "direction_coverage_eligible": False,
        "available_findings": ["Observed signal"],
        "evidence_gaps": ["Independent replication unavailable"],
    }]

    result = asyncio.run(
        PIAgent.analyze_direction_correlations(harness, state, directions)
    )

    assert result == "Integrated result"
    assert "Observed signal" in harness.prompt
    assert "Independent replication unavailable" in harness.prompt
    assert "Evidence Coverage and Limitations" in harness.prompt
