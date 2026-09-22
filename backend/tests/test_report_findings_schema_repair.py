"""A malformed report retries without replacing the intended final findings."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents import bioinformatics_agent as facade
from backend.agents.bioinformatics import executor_report, executor_report_gates
from backend.agents.bioinformatics.executor_dispatch import ToolAction
from backend.models.schemas import AnalysisPhase, AnalysisState, CodeExecution
from backend.utils.llm_client import ToolCall


@pytest.mark.asyncio
@pytest.mark.parametrize("finding_count", [1, 3])
async def test_schema_repair_keeps_all_resubmitted_findings_and_links(finding_count):
    executor_report.bind_facade(facade)
    executor_report_gates.bind_facade(facade)
    agent = facade.BioinformaticsAgent.__new__(facade.BioinformaticsAgent)
    agent._notify = AsyncMock()
    agent.code_history = [CodeExecution(
        id=f"execution-{index}", code="print('result')", output="result",
        round_number=1,
    ) for index in range(finding_count)]
    context = SimpleNamespace(
        agent=agent, state=AnalysisState(
            user_id="schema-repair", current_round=1,
            analysis_phase=AnalysisPhase.INNOVATION,
        ),
        messages=[], analysis_log=[], step_count=1, agent_reasoning="Submit report",
        annotation_completed=True, planning_stage="full", completed=False,
        evidence_gate_degraded=False, evidence_gate_findings=[], all_findings=[],
        accepted_finding_records=[], reported_evidence_ids=[],
        mechanism_programs_registered=True,
    )
    rejected = ToolCall(id="malformed", name="report_findings", arguments={
        "summary": 'Complete report</summary><findings>[{"text":"result"}]</findings>',
    })
    assert await executor_report.handle_report_findings(context, rejected) == ToolAction.CONTINUE
    assert context.completed is False
    assert context.accepted_finding_records == []
    assert context.all_findings == []
    feedback = context.messages[-1]["content"]
    assert "Received top-level keys: summary" in feedback
    assert "sibling JSON fields" in feedback
    assert "Retry with ALL findings" in feedback
    assert "ends this analysis pass" in feedback

    findings = [{"text": f"Result {index}", "evidence_execution_ids": [f"execution-{index}"]}
                for index in range(finding_count)]
    accepted = ToolCall(id="corrected", name="report_findings", arguments={
        "summary": "Complete report", "findings": findings,
        "suggestions": ["Review the complete evidence"],
    })
    assert await executor_report.handle_report_findings(context, accepted) == ToolAction.BREAK
    assert context.completed is True
    assert [item["text"] for item in context.accepted_finding_records] == [
        item["text"] for item in findings
    ]
    assert context.reported_evidence_ids == [f"execution-{index}" for index in range(finding_count)]
    assert [item["candidate_evidence_execution_ids"] for item in context.accepted_finding_records] == [
        [f"execution-{index}"] for index in range(finding_count)
    ]
    assert context.analysis_log[-1].findings_suggestions == ["Review the complete evidence"]
