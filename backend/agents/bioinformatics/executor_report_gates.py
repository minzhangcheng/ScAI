"""Statistical and evidence-gate input construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


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


def gate_samples_per_group(state) -> Dict[str, int]:
    numbered: Dict[Any, set] = {}
    for data_file in (state.data_files or []):
        if getattr(data_file, "sample_number", None) is None:
            continue
        numbered.setdefault(data_file.group_id, set()).add(data_file.sample_number)

    detected: Dict[Any, int] = {}
    for record in (getattr(state, "detected_sample_counts", None) or []):
        group_id = getattr(record, "group_id", None)
        count = int(getattr(record, "detected_count", 0) or 0)
        if group_id is None or count < 1:
            continue
        prior = detected.get(group_id)
        detected[group_id] = count if prior is None else min(prior, count)

    return {
        group.name: (
            detected[group.id]
            if group.id in detected
            else max(group.sample_count, len(numbered.get(group.id, ())))
        )
        for group in (state.treatment_groups or [])
        if getattr(group, "sample_count", None) is not None
    }


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
    evidence_executions = successful_evidence_executions(
        self.code_history, round_number=state.current_round
    )
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
        args.get("findings"),
        score_definitions=_gate_score_definitions(state),
        scoring_plans=state.scoring_plans,
        samples_per_group=_gate_samples_per_group(state),
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
    _unsatisfiable = {"cell-level-inference", "unidentifiable-design"}
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
    if state.analysis_phase == AnalysisPhase.EXPLORATION and context.planning_stage != "stage1":
        if has_bulk_data:
            context.bulk_reject_count, rejected, escaped = self._check_gate_keyword(
                lowered_codes,
                ["bulk", "merged_counts", "pydeseq", "deseqdataset"],
                context.bulk_reject_count, context.max_bulk_rejections,
                "bulk RNA-seq data exists but was not analyzed",
            )
            if rejected:
                bulk_paths = ", ".join(
                    f.file_path for f in state.get_files_by_section("sequencing")
                    if f.data_type == DATA_TYPE_BULK_RNASEQ
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                        "bulk RNA-seq analysis. Bulk RNA-seq data is available and MUST be analyzed. "
                        f"Bulk data files: {bulk_paths}. "
                        "You MUST: (1) Load bulk count matrix with pd.read_csv(), "
                        "(2) Perform DEG analysis using pydeseq2 (if ImportError, fallback to scipy.stats.ttest_ind + statsmodels FDR correction), "
                        "(3) Compare bulk DEGs with scRNA-seq pseudobulk DEGs (Venn diagram), "
                        "THEN call report_findings."
                    )
                })
                analysis_log.append(AnalysisStep(
                    step_number=step_count, step_type="report_findings",
                    agent_reasoning=agent_reasoning,
                    findings=[], findings_summary="REJECTED: bulk RNA-seq not analyzed",
                ))
                return False
            if escaped:
                _escaped_notes.append(build_gate_skip_note("bulk RNA-seq analysis"))
                _attempt_gate_degraded = True
                _attempt_gate_findings.append(_evidence_gate_record(
                    gate="bulk-rnaseq-analysis",
                    rule_id="scai-native/bulk-rnaseq-analysis",
                    detail="Bulk RNA-seq data was available but was never analyzed.",
                    severity="block",
                    rejection_count=context.max_bulk_rejections,
                    rejection_limit=context.max_bulk_rejections,
                    forced_release=True,
                ))
        if has_scrna_data and has_bulk_data:
            context.integration_reject_count, rejected, escaped = self._check_gate_keyword(
                lowered_codes,
                ["venn", "pseudobulk", "overlap", "intersect",
                 "matplotlib_venn", "venn2", "venn3"],
                context.integration_reject_count, context.max_integration_rejections,
                "bulk/scRNA-seq cross-validation not performed",
            )
            if rejected:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                        "bulk/scRNA-seq cross-validation. "
                        "Both bulk RNA-seq and scRNA-seq data are available — you MUST "
                        "compare their DEG results before reporting. Specifically:\n"
                        "(1) Aggregate scRNA-seq to pseudobulk DEGs (or use your existing "
                        "condition-level DEG results across all cells)\n"
                        "(2) Find overlapping DEGs between bulk and scRNA-seq using set intersection\n"
                        "(3) Visualize overlap with a Venn diagram (from matplotlib_venn import venn2)\n"
                        "(4) For overlapping DEGs, show cell-type-specific expression patterns\n"
                        "THEN call report_findings."
                    )
                })
                analysis_log.append(AnalysisStep(
                    step_number=step_count, step_type="report_findings",
                    agent_reasoning=agent_reasoning,
                    findings=[], findings_summary="REJECTED: bulk/scRNA-seq cross-validation not performed",
                ))
                return False
            if escaped:
                _escaped_notes.append(build_gate_skip_note("bulk/scRNA-seq cross-validation"))
                _attempt_gate_degraded = True
                _attempt_gate_findings.append(_evidence_gate_record(
                    gate="bulk-scrna-cross-validation",
                    rule_id="scai-native/bulk-scrna-cross-validation",
                    detail="Bulk and single-cell results were not cross-validated.",
                    severity="block",
                    rejection_count=context.max_integration_rejections,
                    rejection_limit=context.max_integration_rejections,
                    forced_release=True,
                ))
        if has_scrna_data and has_bulk_data:
            context.deconv_reject_count, rejected, escaped = self._check_gate_keyword(
                lowered_codes,
                ["deconvolution", "nnls", "scaden", "least_squares"],
                context.deconv_reject_count, context.max_deconv_rejections,
                "cell-type deconvolution not performed",
            )
            if rejected:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                        "cell-type deconvolution. "
                        "Both bulk RNA-seq and scRNA-seq data are available — you MUST "
                        "perform deconvolution to estimate cell-type proportions in the bulk samples.\n"
                        "(1) Use scRNA-seq as a reference profile\n"
                        "(2) Prioritize pure Python methods like scipy.optimize.nnls or sklearn Non-negative Least Squares\n"
                        "(3) Visualize inferred cell-type proportions across bulk conditions\n"
                        "THEN call report_findings."
                    )
                })
                analysis_log.append(AnalysisStep(
                    step_number=step_count, step_type="report_findings",
                    agent_reasoning=agent_reasoning,
                    findings=[], findings_summary="REJECTED: cell-type deconvolution not performed",
                ))
                return False
            if escaped:
                _escaped_notes.append(build_gate_skip_note("cell-type deconvolution"))
                _attempt_gate_degraded = True
                _attempt_gate_findings.append(_evidence_gate_record(
                    gate="cell-type-deconvolution",
                    rule_id="scai-native/cell-type-deconvolution",
                    detail="Cell-type deconvolution was not completed.",
                    severity="block",
                    rejection_count=context.max_deconv_rejections,
                    rejection_limit=context.max_deconv_rejections,
                    forced_release=True,
                ))
    elif state.analysis_phase == AnalysisPhase.INNOVATION and has_bulk_data and has_scrna_data:
        context.innovation_bulk_reject_count, rejected, escaped = self._check_gate_keyword(
            lowered_codes,
            ["bulk_", "merged_counts"],
            context.innovation_bulk_reject_count, context.max_innovation_bulk_rejections,
            "innovation-phase bulk direction validation missing",
        )
        if rejected:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": (
                    "[REJECTED] Cannot report findings yet. Mandatory step not completed: "
                    "bulk RNA-seq direction validation. Both bulk and scRNA-seq data are "
                    "available — you MUST validate this direction's findings on the bulk data. "
                    "Reuse the bulk variables already loaded in Round 1 (see the data context "
                    "section for their exact names); do NOT re-run the bulk DEG pipeline from "
                    "scratch. Use bulk data as an independent validation layer for your new "
                    "findings, choosing methods appropriate to your specific results and the "
                    "bulk sample design. THEN call report_findings."
                )
            })
            analysis_log.append(AnalysisStep(
                step_number=step_count, step_type="report_findings",
                agent_reasoning=agent_reasoning,
                findings=[],
                findings_summary="REJECTED: innovation-phase bulk direction validation missing",
            ))
            return False
        if escaped:
            _escaped_notes.append(build_gate_skip_note("innovation-phase bulk direction validation"))
            _attempt_gate_degraded = True
            _attempt_gate_findings.append(_evidence_gate_record(
                gate="innovation-bulk-validation",
                rule_id="scai-native/innovation-bulk-validation",
                detail="Innovation findings were not independently validated in bulk data.",
                severity="block",
                rejection_count=context.max_innovation_bulk_rejections,
                rejection_limit=context.max_innovation_bulk_rejections,
                forced_release=True,
            ))
    attempt.gate_degraded = _attempt_gate_degraded
    attempt.gate_findings = _attempt_gate_findings
    return True


async def check_report_evidence(context, tool_call, attempt) -> bool:
    messages = context.messages
    analysis_log = context.analysis_log
    step_count = context.step_count
    agent_reasoning = context.agent_reasoning
    args = tool_call.arguments
    evidence_executions = attempt.evidence_executions
    _attempt_gate_findings = attempt.gate_findings
    requested_evidence_ids = {
        str(value).strip()
        for value in _ensure_list(args.get("evidence_execution_ids"))
        if str(value).strip()
    }
    eligible_evidence_ids = {execution.id for execution in evidence_executions}
    invalid_evidence_ids = requested_evidence_ids - eligible_evidence_ids
    if not requested_evidence_ids or invalid_evidence_ids:
        context.evidence_citation_reject_count += 1
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": (
                "[REJECTED] Findings must cite at least one successful current-run "
                "execute_code result using evidence_execution_ids. Failed executions "
                "or outputs that explicitly say validation failed cannot support a "
                f"claim. Invalid IDs: {sorted(invalid_evidence_ids)}. "
                f"Available IDs: {sorted(eligible_evidence_ids)[-20:]}"
            ),
        })
        analysis_log.append(AnalysisStep(
            step_number=step_count,
            step_type="report_findings",
            agent_reasoning=agent_reasoning,
            findings=[],
            findings_summary="REJECTED: missing or invalid execution evidence",
            evidence_execution_ids=sorted(requested_evidence_ids),
        ))
        if context.evidence_citation_reject_count == context.max_evidence_citation_refusals:
            _attempt_gate_findings.append(_evidence_gate_record(
                gate="evidence-citation",
                rule_id="scai-native/evidence-citation",
                subject="evidence_execution_ids",
                detail=(
                    "report_findings was refused "
                    f"{context.evidence_citation_reject_count} times for citing no "
                    "successful current-run execution. The gate does not "
                    "release; the run continues to spend its step budget "
                    "until a successful cell is cited or the budget ends."
                ),
                severity="block",
                rejection_count=context.evidence_citation_reject_count,
                rejection_limit=0,
            ))
        return False
    attempt.requested_evidence_ids = requested_evidence_ids
    return True
