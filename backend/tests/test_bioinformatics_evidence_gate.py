import traceback

import pytest
from unittest.mock import AsyncMock

from backend.agents.bioinformatics_agent import (
    BioinformaticsAgent,
    TOOLS,
    _build_exec_autosave_wrapper,
    execution_supports_evidence,
    successful_evidence_executions,
)
from backend.models.schemas import CodeExecution
from backend.models.schemas import AnalysisState, RoundResult
from backend.services.report.analysis_log import build_round_summaries


def test_failed_or_explicitly_failed_validation_cannot_support_findings():
    good = CodeExecution(code="print(metric)", output="metric=0.81", round_number=2)
    exception = CodeExecution(code="run()", error="TypeError: bad argument")
    failed_validation = CodeExecution(
        code="bulk_validation()",
        output="Matched in bulk: []\nCRITICAL: No signature genes found. Validation failed.",
    )

    assert execution_supports_evidence(good) is True
    assert execution_supports_evidence(exception) is False
    assert execution_supports_evidence(failed_validation) is False
    assert successful_evidence_executions([exception, good, failed_validation]) == [good]
    assert successful_evidence_executions(
        [exception, good, failed_validation], round_number=1
    ) == []
    assert successful_evidence_executions(
        [exception, good, failed_validation], round_number=2
    ) == [good]


def test_report_findings_requires_execution_evidence_ids():
    tool = next(
        item for item in TOOLS
        if item["function"]["name"] == "report_findings"
    )
    parameters = tool["function"]["parameters"]
    assert "evidence_execution_ids" in parameters["required"]
    assert parameters["properties"]["evidence_execution_ids"]["items"] == {
        "type": "string"
    }


def test_user_exception_traceback_points_to_user_source_not_cleanup(tmp_path):
    wrapper = _build_exec_autosave_wrapper(
        "raise TypeError('user failure')",
        str(tmp_path),
        "traceback-source",
    )

    with pytest.raises(TypeError) as caught:
        exec(wrapper, {}, {})
    rendered = "".join(traceback.format_exception(caught.value))

    assert '<scai-user-code>' in rendered
    assert "_scai_cleanup_plt.close('all')" not in rendered


def test_report_round_summary_preserves_evidence_qualification():
    round_result = RoundResult(
        round_number=7,
        phase="innovation",
        bioinformatics_findings=["candidate signal"],
        round_status="partial",
        evidence_status="substantive",
        bioinformatics_evidence_verified=False,
        evidence_execution_ids=["exec-good"],
        evidence_notes=["Validation did not complete."],
    )
    state = AnalysisState(user_id="owner", round_results=[round_result])

    summary = build_round_summaries(state)[0]
    assert summary["round_status"] == "partial"
    assert summary["bioinformatics_evidence_verified"] is False
    assert summary["evidence_execution_ids"] == ["exec-good"]
    assert summary["evidence_notes"] == ["Validation did not complete."]


@pytest.mark.asyncio
async def test_process_preserves_execution_evidence_metadata():
    agent = object.__new__(BioinformaticsAgent)
    agent.refresh_runtime_config_if_needed = lambda: None
    agent.run_autonomous_analysis = AsyncMock(return_value={
        "status": "completed",
        "findings": ["supported finding"],
        "figures": [],
        "analysis_log": [],
        "success": True,
        "steps_executed": 1,
        "mechanism_programs": [],
        "evidence_verified": True,
        "evidence_execution_ids": ["exec-good"],
    })
    state = AnalysisState(
        user_id="owner",
        current_round=1,
        data_files=[{
            "id": "file-1",
            "filename": "matrix.h5ad",
            "file_path": "/tmp/matrix.h5ad",
            "data_type": "scRNA-seq",
            "section": "sequencing",
        }],
    )

    result = await agent.process(state, "Analyze")

    assert result["evidence_verified"] is True
    assert result["evidence_execution_ids"] == ["exec-good"]
