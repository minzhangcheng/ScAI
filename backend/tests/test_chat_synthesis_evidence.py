"""Executed methods must survive the executor-to-PI chat handoff."""

import pytest

from backend.models.schemas import AnalysisStep, CodeExecution
from backend.routes.chat_parts.delegation import _build_synthesis_prompt


@pytest.mark.parametrize("serialized", [False, True])
def test_synthesis_preserves_executor_summary_and_method_records(serialized):
    summary = "Panel: GeneA, GeneB. The input contains log-normalized expression."
    code = "score(data.raw, genes=['GeneA', 'GeneB'], bins=25, controls=50)"
    step = AnalysisStep(
        step_number=3, step_type="report_findings", findings_summary=summary,
        evidence_execution_ids=["score-execution"],
    )
    execution = CodeExecution(
        id="score-execution", code=code, output="input_scale=log_normalized",
        figures=["IMAGE_PAYLOAD_MUST_NOT_ENTER_PROMPT"],
    )
    if serialized:
        step = step.model_dump(mode="json")
        execution = execution.model_dump(mode="json")
    prompt = _build_synthesis_prompt("Write methods", {}, {
        "bioinformatics": {
            "findings": ["A two-feature score changed."],
            "analysis_log": [step], "code_history": [execution],
            "score_definitions": [{
                "score_name": "demo_score", "method": "sc.tl.score_genes",
                "gene_list": ["GeneA", "GeneB"], "parameters": {"n_bins": 25},
            }],
            "scoring_plans": [{
                "id": "plan-1", "score_name": "demo_score",
                "panel_genes": ["GeneA", "GeneB"],
            }],
        },
    })
    assert summary in prompt
    assert code in prompt
    assert "score-execution" in prompt
    assert "input_scale=log_normalized" in prompt
    assert "IMAGE_PAYLOAD_MUST_NOT_ENTER_PROMPT" not in prompt
    assert "Registered score definitions" in prompt
    assert "sc.tl.score_genes" in prompt
    assert '"n_bins": 25' in prompt
    assert "Frozen scoring plans" in prompt
    assert "plan-1" in prompt
    assert "A two-feature score changed." in prompt
    assert "reconcile agent summaries" in prompt
    assert "unverified rather than inventing" in prompt


@pytest.mark.parametrize("terminal_error", [False, True])
def test_synthesis_keeps_failed_attempt_visible_even_after_delegate_error(terminal_error):
    result = {
        "analysis_log": [{
            "step_number": 4,
            "findings_summary": "The final computation used normalized data.",
            "evidence_execution_ids": ["retry"],
        }],
        "code_history": [
            {"id": "attempt", "code": "use_counts()",
             "execution_status": "failed", "error": "Counts unavailable"},
            {"id": "retry", "code": "use_normalized()",
             "execution_status": "succeeded"},
        ],
    }
    if terminal_error:
        result["error"] = "Delegate stopped before returning final findings."
    else:
        result["summary"] = "The analyst says the input is counts."
    prompt = _build_synthesis_prompt("Explain methods", {}, {
        "bioinformatics": result,
    })
    assert result.get("error", result.get("summary")) in prompt
    for text in (
        "The final computation used normalized data.",
        '"execution_status": "failed"', "Counts unavailable",
        '"execution_status": "succeeded"', "use_normalized()",
        "If records conflict, explain the discrepancy",
    ):
        assert text in prompt


def test_synthesis_preserves_long_executions_and_discloses_summary_truncation():
    long_code = "c" * 49000
    final_output = "GeneA log2FC=-0.28, padj=0.60"
    prompt = _build_synthesis_prompt("Review", {}, {
        "bioinformatics": {
            "analysis_log": [{"findings_summary": "s" * 17000}],
            "code_history": [
                {"id": "long-execution", "code": long_code},
                {"id": "final-result", "code": "print(results)", "output": final_output},
            ],
        },
        "literature": {"report": "Independent literature result."},
    })
    assert prompt.count("[Evidence truncated:") == 1
    assert "Missing details cannot be verified" in prompt
    assert long_code in prompt
    assert final_output in prompt
    assert "1 report summaries; 2 execution records; 0 score definitions; 0 scoring plans" in prompt
    assert "Independent literature result." in prompt


def test_findings_without_execution_records_do_not_imply_verified_methods():
    prompt = _build_synthesis_prompt("Write methods", {}, {
        "bioinformatics": {"findings": ["A score changed."]},
    })
    assert "A score changed." in prompt
    assert "library default" in prompt
    assert "cannot be verified" in prompt
    assert "Executor report summaries" not in prompt


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("status", ["max_steps_reached", "completed"])
def test_synthesis_preserves_stop_status_and_tool_errors_with_historical_scores(serialized, status):
    error = "annotate_callback() got an unexpected keyword argument 'annotation_source'"
    step = AnalysisStep(step_number=8, step_type="tool_error", text_content=error)
    validation = AnalysisStep(
        step_number=9, step_type="execute_code", code_error="Code validation failed",
    )
    steps = [step, validation]
    if serialized:
        steps = [item.model_dump(mode="json") for item in steps]
    result = {
        "status": status,
        # The executor uses this flag for non-LLM-error exits, including exhaustion.
        "success": True,
        "steps_executed": 30,
        "findings": [] if status == "max_steps_reached" else ["A usable partial observation."],
        "summary": "The analyst claims the requested analysis is complete.",
        "figures": [],
        "analysis_log": steps,
        "code_history": [{"id": "preprocessing", "code": "prepare_data()"}],
        "attempt_execution_ids": ["preprocessing"],
        "score_definitions": [{"score_name": "historical_score", "execution_id": "old-execution"}],
        "scoring_plans": [{"id": "old-plan"}],
    }

    prompt = _build_synthesis_prompt("Analyze and plot", {}, {"bioinformatics": result})

    assert f'"status": "{status}"' in prompt
    assert '"success": true' in prompt
    assert '"steps_executed": 30' in prompt
    assert '"attempt_execution_ids": [\n    "preprocessing"' in prompt
    assert error in prompt
    assert "Code validation failed" in prompt
    if result["findings"]:
        assert "A usable partial observation." in prompt
    assert "prepare_data()" in prompt
    assert "Figure artifacts returned for this delegation:\n[]" in prompt
    assert "historical_score" in prompt and "old-plan" in prompt
    assert "may predate this delegation" in prompt
    assert "later recovery is possible" in prompt
    assert result["status"] == status


def test_synthesis_exposes_real_figure_references_without_image_payloads():
    prompt = _build_synthesis_prompt("Explain the figure", {}, {
        "bioinformatics": {
            "status": "completed", "figures": [{
                "name": "observed_result", "path": "/workspace/figures/observed_result.jpg",
                "pdfPath": "/workspace/figures/observed_result.pdf",
                "image_base64": "IMAGE_PAYLOAD_MUST_NOT_ENTER_PROMPT",
            }],
        },
    })
    assert '"name": "observed_result"' in prompt
    assert "/workspace/figures/observed_result.pdf" in prompt
    assert "IMAGE_PAYLOAD_MUST_NOT_ENTER_PROMPT" not in prompt
