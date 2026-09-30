"""Terminal result helpers for autonomous analysis reporting."""
from __future__ import annotations

from .executor_dispatch import ToolAction
from ...utils.bioinformatics_evidence import attach_evidence_sources, record_evidence_context_failure
from .executor_report_gates import (
    attach_report_observations,
    prepare_report_attempt,
    record_report_evidence_candidates,
)


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


def terminal_status(
    *,
    llm_error: bool,
    plan_revision_needed: bool,
    completed: bool,
    evidence_gate_degraded: bool,
) -> str:
    if llm_error:
        return "error"
    if plan_revision_needed:
        return "paused_for_revision"
    if completed:
        return "partial" if evidence_gate_degraded else "completed"
    return "max_steps_reached"


async def handle_report_findings(context, tool_call) -> ToolAction:
    attempt = await prepare_report_attempt(context, tool_call)
    if attempt is None:
        return ToolAction.CONTINUE
    await attach_report_observations(context, attempt)
    await record_report_evidence_candidates(context, tool_call, attempt)
    try:
        attach_evidence_sources(context.state, attempt.finding_records)
    except Exception as exc:
        record_evidence_context_failure(
            attempt.finding_records, stage="source attachment", error=exc,
        )
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    context.has_substantive_work = True
    context.accepted_finding_records = list(attempt.finding_records)
    findings = [record["text"] for record in context.accepted_finding_records]
    self._last_finding_records = context.accepted_finding_records
    self._last_findings = findings
    summary = args.get("summary", "")
    suggestions = args.get("suggestions", [])
    for note in attempt.escaped_notes:
        if note not in findings:
            findings.append(note)
    context.all_findings.extend(findings)
    context.reported_evidence_ids = sorted(attempt.requested_evidence_ids)
    context.evidence_gate_degraded = attempt.gate_degraded
    context.evidence_gate_findings = list(attempt.gate_findings)
    if context.evidence_gate_degraded:
        state.evidence_gate_degraded = True
        known_gate_records = {
            (
                item.get("gate"), item.get("rule_id"),
                item.get("subject"), item.get("round_number"),
            )
            for item in state.evidence_gate_findings
            if isinstance(item, dict)
        }
        for item in context.evidence_gate_findings:
            if not isinstance(item, dict):
                continue
            enriched = {
                **item,
                "round_number": item.get(
                    "round_number", state.current_round,
                ),
            }
            key = (
                enriched.get("gate"), enriched.get("rule_id"),
                enriched.get("subject"), enriched.get("round_number"),
            )
            if key not in known_gate_records:
                state.evidence_gate_findings.append(enriched)
                known_gate_records.add(key)
    await self._notify("findings", json.dumps({
        "findings": findings,
        "summary": summary,
        "suggestions": suggestions
    }))
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": "Findings recorded. Continue or wait for instructions."
    })
    analysis_log.append(AnalysisStep(
        step_number=step_count, step_type="report_findings",
        agent_reasoning=agent_reasoning,
        findings=findings, findings_summary=summary,
        finding_records=context.accepted_finding_records,
        findings_suggestions=suggestions or [],
        evidence_execution_ids=context.reported_evidence_ids,
    ))
    context.completed = True
    return ToolAction.BREAK
