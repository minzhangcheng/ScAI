"""Statistical and evidence-gate input construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid

from backend.services.sample_design import samples_per_group_by_modality


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


@dataclass
class ReportAttempt:
    evidence_executions: list
    lowered_codes: list[str]
    has_scrna_data: bool
    has_bulk_data: bool
    finding_records: list[dict] = field(default_factory=list)
    escaped_notes: list[str] = field(default_factory=list)
    gate_degraded: bool = False
    gate_findings: list[dict] = field(default_factory=list)
    requested_evidence_ids: set[str] = field(default_factory=set)


def gate_score_definitions(state, *, run_gates) -> List[Any]:
    """Keep current/pre-tracking scores and older evidence-valid records."""
    carried: List[Any] = []
    for score in (state.score_definitions or []):
        if getattr(score, "round_number", 0) in (0, state.current_round):
            carried.append(score)
            continue
        if run_gates(
            None,
            score_definitions=[score],
            scoring_plans=state.scoring_plans,
        ).passed:
            carried.append(score)
    return carried


def gate_samples_per_group_by_modality(state) -> Dict[str, Dict[str, int]]:
    """Replicate counts keyed by sequencing modality, then treatment group."""
    return samples_per_group_by_modality(state)


def evidence_gate_record(
    *,
    gate: str,
    detail: str,
    severity: str,
    rule_id: str = "",
    subject: str = "",
    rejection_count: int = 0,
    rejection_limit: int = 0,
    forced_release: bool = False,
):
    return {
        "gate": str(gate),
        "rule_id": str(rule_id),
        "detail": str(detail),
        "subject": str(subject),
        "severity": str(severity),
        "rejection_count": max(0, int(rejection_count)),
        "rejection_limit": max(0, int(rejection_limit)),
        "forced_release": bool(forced_release),
    }


async def prepare_report_attempt(context, tool_call) -> ReportAttempt | None:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    if state.analysis_phase == AnalysisPhase.EXPLORATION and not context.annotation_completed:
        logger.warning(
            f"[Bioinformatics] report_findings rejected: annotate_clusters not yet called"
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                "cell type annotation via annotate_clusters. "
                "You MUST first run sc.tl.rank_genes_groups(), then call annotate_clusters, "
                "before reporting findings."
            )
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count, step_type="report_findings",
            agent_reasoning=agent_reasoning,
            findings=[], findings_summary="REJECTED: annotate_clusters not completed",
        ))
        return None
    has_scrna_data = state.has_scrnaseq()
    has_bulk_data = state.has_bulk_rnaseq()
    finding_records = self._normalize_report_finding_records(args)
    if not finding_records:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                "[REJECTED] findings must contain at least one non-empty finding "
                "object. A single object is accepted for compatibility, but it "
                "must contain text."
            ),
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count,
            step_type="report_findings",
            agent_reasoning=agent_reasoning,
            findings=[],
            findings_summary="REJECTED: empty findings payload",
        ))
        return None
    active_plan_id = str(getattr(state.current_plan, "id", "") or "") or None
    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    cohort_identity = active_cohort_identity(state)
    active_cohort_digest = (
        str(cohort_identity.get("cohort_digest") or "") or None
    )
    active_cohort_generation = int(
        cohort_identity.get("cohort_generation", 0) or 0
    ) or None
    evidence_executions = successful_evidence_executions(
        self.code_history,
        round_number=state.current_round,
        plan_id=active_plan_id,
        lineage_id=active_lineage,
        cohort_digest=active_cohort_digest,
        cohort_generation=active_cohort_generation,
        require_plan_binding=False,
    )
    for record in finding_records:
        record.update({
            "id": str(record.get("id") or uuid.uuid4()),
            "round_number": int(state.current_round or 0),
            "lineage_id": active_lineage,
            "plan_id": active_plan_id,
            "analysis_generation": int(
                (state.sandbox_state or {}).get("analysis_generation", 0) or 0
            ),
            "cohort_generation": int(
                cohort_identity.get("cohort_generation", 0) or 0
            ),
            "cohort_digest": active_cohort_digest,
            "candidate_plan_step_ids": [],
            "confirmed_plan_step_ids": [],
            "evidence_status": "unverified",
            "semantic_status": "unverified",
        })
    lowered_codes = [ce.code.lower() for ce in evidence_executions if ce.code]
    _escaped_notes = []
    _attempt_gate_degraded = context.evidence_gate_degraded
    _attempt_gate_findings: List[Dict[str, Any]] = list(
        context.evidence_gate_findings
    )
    from ...services.figure_contract import (
        audit_figure_contract,
        format_missing_figure_feedback,
    )
    _figure_contract = audit_figure_contract(
        state,
        plan=state.current_plan,
        planning_stage=context.planning_stage,
        include_stage_baseline=False,
        defer_platform_baseline=True,
    )
    if not _figure_contract.passed:
        missing_text = format_missing_figure_feedback(
            _figure_contract
        )
        logger.warning(
            "[Bioinformatics] %d required figure deliverable(s) "
            "unmet at report_findings; recorded for PI review",
            len(_figure_contract.missing),
        )
        _figure_gap_detail = (
            "Required result figures were not complete at "
            "report submission. They are unrendered, stale, "
            "or held back by review/layout gates, so the PI "
            "must decide whether to repair them, accept the "
            "result with a limitation, or drop the "
            "deliverable:\n"
            f"{missing_text}"
        )
        _escaped_notes.append(
            "[PI review observation; non-blocking] "
            + _figure_gap_detail
        )
        _attempt_gate_findings.append(_evidence_gate_record(
            gate="required-figure-observation",
            rule_id="scai-native/required-figure-observation",
            detail=_figure_gap_detail,
            subject="required-publication-figures",
            severity="advise",
        ))
    _stats_report = run_statistical_gates(
        finding_records,
        score_definitions=_gate_score_definitions(state),
        scoring_plans=state.scoring_plans,
        samples_per_group_by_modality=_gate_samples_per_group_by_modality(state),
    )
    _stats_advice = _stats_report.advisory_message()
    if _stats_advice:
        _escaped_notes.append(_stats_advice)
        _attempt_gate_findings.extend(
            _evidence_gate_record(
                gate=finding.gate,
                rule_id=finding.rule_id,
                detail=finding.detail,
                subject=finding.subject,
                severity="advise",
            )
            for finding in _stats_report.advisories
        )
    _unsatisfiable = {"cell-level-inference"}
    _cap = (1 if all(f.gate in _unsatisfiable
                     for f in _stats_report.blocking)
            else context.max_stats_gate_rejections)
    if not _stats_report.passed:
        if context.stats_gate_reject_count < _cap:
            context.stats_gate_reject_count += 1
            logger.warning(
                f"[Bioinformatics] report_findings rejected: "
                f"statistical validity "
                f"({context.stats_gate_reject_count}/{_cap})"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": _stats_report.message(),
            })
            return None
        logger.warning(
            "[Bioinformatics] statistical gate: max rejections "
            "reached, allowing report_findings"
        )
        _escaped_notes.append(
            "Statistical validity checks did not pass after "
            + str(_cap)
            + " attempt(s) and were force-released: "
            + "; ".join(f.detail for f in _stats_report.blocking)
        )
        _attempt_gate_degraded = True
        _attempt_gate_findings.extend(
            _evidence_gate_record(
                gate=finding.gate,
                rule_id=finding.rule_id,
                detail=finding.detail,
                subject=finding.subject,
                severity="block",
                rejection_count=_cap,
                rejection_limit=_cap,
                forced_release=True,
            )
            for finding in _stats_report.blocking
        )
    return ReportAttempt(
        evidence_executions=evidence_executions,
        lowered_codes=lowered_codes,
        has_scrna_data=has_scrna_data,
        has_bulk_data=has_bulk_data,
        finding_records=finding_records,
        escaped_notes=_escaped_notes,
        gate_degraded=_attempt_gate_degraded,
        gate_findings=_attempt_gate_findings,
    )


async def check_mandatory_report_gates(context, tool_call, attempt) -> bool:
    self = context.agent
    state = context.state
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    lowered_codes = attempt.lowered_codes
    has_scrna_data = attempt.has_scrna_data
    has_bulk_data = attempt.has_bulk_data
    scrna_has_replicates = has_replicated_condition_design(
        state, SCRNA_MODALITY,
    )
    bulk_has_replicates = has_replicated_condition_design(
        state, BULK_RNASEQ_MODALITY,
    )
    _escaped_notes = attempt.escaped_notes
    _attempt_gate_degraded = attempt.gate_degraded
    _attempt_gate_findings = attempt.gate_findings
    if (state.analysis_phase == AnalysisPhase.EXPLORATION
            and context.planning_stage != "stage1"
            and not context.mechanism_programs_registered):
        if context.mechanism_gate_reject_count < context.max_mechanism_rejections:
            context.mechanism_gate_reject_count += 1
            logger.warning(
                f"[Bioinformatics] report_findings rejected: "
                f"register_mechanism_programs not called "
                f"({context.mechanism_gate_reject_count}/{context.max_mechanism_rejections})"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    "[REJECTED] Before reporting findings you must call "
                    "register_mechanism_programs. After completing downstream "
                    "analysis, identify up to 8 data-driven mechanism programs that "
                    "are genuinely supported by observed genes, pathways, scores, "
                    "enrichment terms, or cross-validated patterns (do NOT invent "
                    "weak programs to reach a count). "
                    + MECHANISM_EVIDENCE_POLICY
                    + " If none are "
                    "supportable, call register_mechanism_programs with an empty "
                    "programs list and explain why in no_program_reason."
                ),
            })
            analysis_log.append(AnalysisStep(
                step_number=step_count, step_type="report_findings",
                agent_reasoning=agent_reasoning,
                findings=[], findings_summary="REJECTED: register_mechanism_programs not called",
            ))
            return False
        else:
            _escaped_notes.append(build_gate_skip_note("data-driven mechanism program capture"))
            _attempt_gate_degraded = True
            _attempt_gate_findings.append(_evidence_gate_record(
                gate="mechanism-program-capture",
                rule_id="scai-native/mechanism-program-capture",
                detail="register_mechanism_programs was not completed before reporting findings.",
                severity="block",
                rejection_count=context.max_mechanism_rejections,
                rejection_limit=context.max_mechanism_rejections,
                forced_release=True,
            ))
    if has_bulk_data and context.planning_stage != "stage1":
        eligible_ids = {item.id for item in attempt.evidence_executions}
        current_decisions = [
            item for item in state.method_decisions
            if item.round_number == state.current_round
            and item.execution_id in eligible_ids
            and any(profile.id == item.input_profile_id
                    for profile in state.bulk_input_profiles)
        ]
        if not current_decisions:
            _attempt_gate_degraded = True
            _attempt_gate_findings.append(_evidence_gate_record(
                gate="bulk-method-decision",
                rule_id="scai-native/bulk-method-decision",
                detail=(
                    "No current bulk method decision was linked to eligible evidence. "
                    "The findings were retained as unverified; the workflow was not "
                    "blocked."
                ),
                severity="advise",
            ))
    attempt.gate_degraded = _attempt_gate_degraded
    attempt.gate_findings = _attempt_gate_findings
    return True


async def check_report_evidence(context, tool_call, attempt) -> bool:
    args = tool_call.arguments
    evidence_executions = attempt.evidence_executions
    requested_evidence_ids = {
        str(value).strip()
        for value in _ensure_list(args.get("evidence_execution_ids"))
        if str(value).strip()
    }
    eligible_evidence_ids = {execution.id for execution in evidence_executions}
    evidence_by_id = {execution.id: execution for execution in evidence_executions}
    valid_requested_ids = requested_evidence_ids & eligible_evidence_ids
    attempt.requested_evidence_ids = set(valid_requested_ids)
    for record in attempt.finding_records:
        claimed_ids = {
            str(value).strip()
            for value in record.get("evidence_execution_ids", [])
            if str(value).strip()
        }
        if not claimed_ids:
            claimed_ids = set(requested_evidence_ids)
        record_ids = claimed_ids & eligible_evidence_ids
        record_executions = [evidence_by_id[value] for value in record_ids]
        record["claimed_evidence_execution_ids"] = sorted(claimed_ids)[:40]
        record["candidate_evidence_execution_ids"] = sorted(record_ids)[:40]
        # Producer citations nominate evidence. They do not prove the semantic
        # relation between an execution and a scientific finding.
        record["evidence_execution_ids"] = []
        record["candidate_plan_step_ids"] = sorted({
            int(step_id)
            for execution in record_executions
            for step_id in getattr(execution, "plan_step_ids", [])
        })
        record["evidence_status"] = "unverified"
        record["semantic_status"] = "unverified"
        attempt.requested_evidence_ids.update(record_ids)
    return True
