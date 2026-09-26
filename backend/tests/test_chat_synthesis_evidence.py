"""Chat synthesis receives failures and historical provenance alongside summaries."""

import pytest

from backend.models.schemas import AnalysisStep
from backend.routes.chat_parts.delegation import _build_synthesis_prompt


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("status", ["max_steps_reached", "completed"])
@pytest.mark.parametrize("terminal_error", [False, True])
def test_synthesis_preserves_failed_attempts_stop_status_and_historical_scores(
    serialized, status, terminal_error,
):
    steps = [
        AnalysisStep(step_number=3, step_type="report_findings",
                     findings_summary="The final computation used normalized data.",
                     evidence_execution_ids=["retry"]),
        AnalysisStep(step_number=4, step_type="tool_error", text_content="Annotation callback failed"),
        AnalysisStep(step_number=5, step_type="execute_code", code_error="Code validation failed"),
    ]
    result = {
        "status": status, "success": True, "steps_executed": 30,
        "attempt_execution_ids": ["attempt", "retry"], "figures": [],
        "findings": [], "summary": "The analyst says the input is counts.",
        "analysis_log": [step.model_dump() for step in steps] if serialized else steps,
        "code_history": [
            {"id": "attempt", "code": "use_counts()", "execution_status": "failed",
             "error": "Counts unavailable"},
            {"id": "retry", "code": "use_normalized()", "execution_status": "succeeded"},
        ],
        "score_definitions": [{"score_name": "historical_score", "execution_id": "old-execution"}],
        "scoring_plans": [{"id": "old-plan"}],
    }
    if terminal_error:
        result["error"] = "Delegate stopped before returning final findings."
    elif status == "completed":
        result["findings"] = ["A usable partial observation."]
    prompt = _build_synthesis_prompt("Explain methods", {}, {"bioinformatics": result})
    for expected in (
        result.get("error", result["summary"]),
        "The final computation used normalized data.", "use_counts()", "use_normalized()",
        '"execution_status": "failed"', '"execution_status": "succeeded"', "Counts unavailable",
        f'"status": "{status}"', '"success": true', '"steps_executed": 30',
        '"attempt_execution_ids": [', "Annotation callback failed", "Code validation failed",
        "historical_score", "old-execution", "old-plan", "may predate this delegation",
        "Figure artifacts returned for this delegation:\n[]",
    ):
        assert expected in prompt
    if result["findings"]:
        assert result["findings"][0] in prompt
    assert result["status"] == status
