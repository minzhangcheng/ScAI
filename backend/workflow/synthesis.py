"""PI synthesis, user input wait, and convergence check."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constants import (
    CONTINUE_SIGNAL,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
    LlmFailureAction,
    PI_DECIDE_SIGNAL,
    StateSyncChangeType,
    WaitInputReason,
)
from ..models.schemas import AgentType, AnalysisPhase, AnalysisState, RoundResult, TaskStatus
from ..services.interaction_record_capture import get_or_open_direction_record
from ..utils.direction_evidence import build_fallback_direction_evidence
from ..utils.llm_client import UpstreamExhaustedError
from .direction_coverage import (
    at_integration_gate,
    build_result_identity_key,
    count_innovation_rounds_done,
    current_lineage_id,
    direction_coverage_status,
    has_substantive_findings,
    normalize_direction_key,
    normalize_legacy_round_result,
)
from .helpers import ack_restore_safe_point, wait_with_skip_time
from .provenance import consume_restore_provenance, restore_provenance_id

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


def build_disabled_pathology_warning(files: list[dict], pathology_enabled: bool) -> str:
    """Return a modality-boundary warning when image files exist but pathology is disabled."""
    if pathology_enabled:
        return ""
    # Only genuine pathology-modality inputs matter here. Images belonging to
    # other modalities (e.g. a biochemistry panel PNG) or generic PI-chat
    # uploads are not pathology data and must not trigger this warning. The
    # canonical modality signal is the section: data_files carry it under
    # "section", section-based pending uploads under "type".
    has_pathology_input = any(
        (f.get("section") or f.get("type") or "").lower() == "pathology"
        for f in files
    )
    if not has_pathology_input:
        return ""
    return (
        "Pathology image inputs are present while the pathology agent is disabled. "
        "These images will be treated as non-pathology descriptive evidence unless pathology analysis is enabled."
    )


def classify_exploration_failure(bio) -> str:
    """Map the internal bioinformatics result to a STABLE reason code for the UI.

    Never leak raw internal status strings (e.g. 'max_steps_reached',
    'completed_with_fallback') to the frontend — the dialog localizes off this
    code. Returns one of: 'analysis_error', 'did_not_converge'."""
    if isinstance(bio, dict) and bio.get("error"):
        return "analysis_error"
    return "did_not_converge"


FAILED_AGENT_STATUSES = frozenset({
    "failed",
    "error",
    "timeout",
    "max_steps_reached",
})

SKIPPED_AGENT_STATUSES = frozenset({
    "skipped",
    "not_run",
})

PARTIAL_AGENT_STATUSES = frozenset({
    "partial",
    "completed_with_fallback",
})

CORE_ROUND_AGENT_RESULT_KEYS = (
    "bioinformatics_result",
    "literature_result",
)

OPTIONAL_ROUND_AGENT_RESULT_KEYS = (
    "pathology_result",
    "biochemistry_result",
)

CORE_FAILED_NODES = frozenset({
    "bioinformatics_analysis",
    "literature_research",
})


def _agent_result_status(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("error"):
        return "failed"
    return str(result.get("status") or "").strip().lower()


def _statuses_for_keys(
    state: Dict[str, Any],
    keys: tuple[str, ...],
    *,
    missing_status: str = "",
) -> list[str]:
    statuses = []
    for key in keys:
        result = state.get(key)
        if isinstance(result, dict) and result:
            statuses.append(_agent_result_status(result))
        elif missing_status:
            statuses.append(missing_status)
    return statuses


def derive_round_completion_status(state: Dict[str, Any], result: RoundResult) -> RoundResult:
    """Populate RoundResult completion fields from actual agent execution state.

    Core bioinformatics/literature failures block direction completion. Optional
    pathology/biochemistry skipped/no-data results do not block completion when
    core evidence exists, but optional hard failures downgrade the round to
    partial so the report can disclose degraded modality coverage.
    """
    failed_node = state.get("failed_node")
    # This assignment is the provenance boundary for report-quality consumers.
    # Older persisted RoundResult objects contain the same field defaults but no
    # evidence that those defaults were ever derived from node execution.
    result.evidence_schema_version = 1

    # Integration is a first-class evidence-producing phase, but it deliberately
    # does not execute the ordinary bioinformatics/literature nodes.  Judge it by
    # its own output instead of manufacturing two ``not_run`` core-agent states.
    if result.phase == "integration":
        has_integration_evidence = bool(
            (result.integration_report_summary or "").strip()
            or result.integration_candidates
        )
        if failed_node == "integration_synthesis":
            result.round_status = "partial" if has_integration_evidence else "failed"
            result.evidence_status = (
                "substantive" if has_integration_evidence else "failed"
            )
            result.direction_coverage_eligible = None
            return result
        result.round_status = "completed" if has_integration_evidence else "partial"
        result.evidence_status = (
            "substantive" if has_integration_evidence else "empty"
        )
        # Direction coverage is an innovation-batch concept. Integration has a
        # valid completion/evidence verdict of its own but is not applicable to
        # that ledger.
        result.direction_coverage_eligible = None
        return result

    has_evidence = has_substantive_findings(result)
    has_core_evidence = bool(
        result.bioinformatics_findings
        or result.literature_findings
        or result.data_driven_mechanism_programs
    )

    if failed_node in CORE_FAILED_NODES:
        result.round_status = "partial" if has_evidence else "skipped"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    # Core agents are required for a completed research round. During
    # stop-and-summarize, the graph snapshot legitimately lacks results for
    # agents that had not run yet; absence is therefore an explicit not_run,
    # never evidence that the round completed successfully.
    core_statuses = _statuses_for_keys(
        state,
        CORE_ROUND_AGENT_RESULT_KEYS,
        missing_status="not_run",
    )
    optional_statuses = _statuses_for_keys(state, OPTIONAL_ROUND_AGENT_RESULT_KEYS)

    has_core_failed = any(status in FAILED_AGENT_STATUSES for status in core_statuses)
    has_core_skipped = any(status in SKIPPED_AGENT_STATUSES for status in core_statuses)
    has_core_partial = any(status in PARTIAL_AGENT_STATUSES for status in core_statuses)
    has_optional_failed = any(status in FAILED_AGENT_STATUSES for status in optional_statuses)
    has_optional_partial = any(status in PARTIAL_AGENT_STATUSES for status in optional_statuses)

    # New bioinformatics runs must bind findings to successful current-run
    # execution IDs. A deliberately degraded mandatory gate or an invalid
    # evidence reference may still yield a partial narrative, but it must not
    # satisfy direction coverage or be reported as validated evidence.
    bioinformatics_result = state.get("bioinformatics_result")
    if (
        isinstance(bioinformatics_result, dict)
        and bioinformatics_result.get("findings")
        and bioinformatics_result.get("evidence_verified") is False
    ):
        result.round_status = "partial"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    if has_core_failed:
        result.round_status = "partial" if has_evidence else "failed"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    if has_core_skipped:
        result.round_status = "partial" if has_evidence else "skipped"
        result.evidence_status = "substantive" if has_evidence else "failed"
        result.direction_coverage_eligible = False
        return result

    if has_core_partial:
        result.round_status = "partial"
        result.evidence_status = "substantive" if has_evidence else "empty"
        result.direction_coverage_eligible = False
        return result

    if has_optional_failed or has_optional_partial:
        result.round_status = "partial"
        result.evidence_status = "substantive" if has_evidence else "empty"
        # Optional modality degradation must remain visible in round_status,
        # but it must not keep a direction batch permanently incomplete when
        # both core agents completed and substantive evidence exists.
        result.direction_coverage_eligible = has_core_evidence
        return result

    result.round_status = "completed"
    result.evidence_status = "substantive" if has_evidence else "empty"
    result.direction_coverage_eligible = has_evidence
    return result


def _upsert_round_result(
    results: list[RoundResult],
    result: RoundResult,
    *,
    lineage_id: str,
    branch_id: Optional[str],
    branch_label: Optional[str],
    restored_from_snapshot_id: Optional[str],
) -> None:
    result.lineage_id = lineage_id or "main"
    result.branch_id = branch_id
    result.branch_label = branch_label
    result.restored_from_snapshot_id = restored_from_snapshot_id
    result.direction_key = normalize_direction_key(result.direction)
    result.result_identity_key = build_result_identity_key(
        result.lineage_id,
        result.phase,
        result.round_number,
        result.direction_key,
    )
    result = normalize_legacy_round_result(result)
    normalized: list[RoundResult] = []
    identity_indexes: dict[str, int] = {}
    for raw in results:
        existing = normalize_legacy_round_result(raw)
        key = existing.result_identity_key
        if key in identity_indexes:
            # A retry written before identity-aware upsert may have appended a
            # second legacy record. Keep the later retained value in the first
            # stable slot so API/WS ordering does not jump on migration.
            normalized[identity_indexes[key]] = existing
            continue
        identity_indexes[key] = len(normalized)
        normalized.append(existing)
    target_index = identity_indexes.get(result.result_identity_key)
    if target_index is None:
        normalized.append(result)
    else:
        normalized[target_index] = result
    results[:] = normalized


def _should_force_next_round(analysis_state: AnalysisState) -> bool:
    """Return True when workflow must honor pending work instead of stopping on convergence.

    Priority order:
    1. Pending integration round
    2. Real human text input (not synthetic continue markers)
    3. Newly uploaded files
    4. Remaining multi-direction queue entries, except for the stale final
       queue entry left after completed multi-direction analysis
    """
    if getattr(analysis_state, "pending_integration", False):
        return True
    has_real_text = bool(
        analysis_state.pending_human_input
        and analysis_state.pending_human_input.strip() not in ("", CONTINUE_SIGNAL, PI_DECIDE_SIGNAL)
    )
    has_uploaded_files = bool(analysis_state.pending_uploaded_files)
    has_remaining_directions = bool(
        analysis_state.directions_queue
        and any(d.get("direction", "").strip() for d in analysis_state.directions_queue)
    )
    if has_remaining_directions and _multi_direction_analysis_completed(analysis_state):
        has_remaining_directions = False
    return has_real_text or has_uploaded_files or has_remaining_directions


def _consume_signal_then_clear(event: asyncio.Event) -> None:
    event.clear()


def _multi_direction_analysis_completed(analysis_state) -> bool:
    """True once the active direction batch is fully covered (spec §14)."""
    from .direction_coverage import active_batch_complete
    return active_batch_complete(analysis_state)



def _should_build_innovation_segment(analysis_state, round_result) -> bool:
    """Whether to build a detailed-report segment for the just-completed round.

    Keys off round_result.phase (captured before the exploration→innovation
    transition) rather than analysis_state.analysis_phase, which is already
    flipped to INNOVATION at the end of the exploration round and would
    otherwise spawn a spurious 'innovation_round_1' segment for round 1.

    Deduplicates by phase to stay idempotent across node retries.
    """
    if round_result.phase != "innovation":
        return False
    phase = f"innovation_round_{analysis_state.current_round}"
    return not any(
        s.phase == phase for s in analysis_state.detailed_report_segments
    )


def _build_directions_with_results_for_correlation(analysis_state: AnalysisState) -> list[dict]:
    target_lineage = current_lineage_id(analysis_state)
    coverage = direction_coverage_status(analysis_state, lineage_id=target_lineage)
    direction_summaries: Dict[str, str] = {}

    for raw in analysis_state.round_results:
        result = normalize_legacy_round_result(raw)
        if result.lineage_id != target_lineage:
            continue
        if result.direction_key in coverage.completed_keys:
            direction_summaries[result.direction_key] = result.pi_summary or ""

    directions_with_results = []
    for chosen_dir in analysis_state.user_chosen_directions:
        key = normalize_direction_key(chosen_dir.get("direction", ""))
        if key not in coverage.completed_keys:
            continue
        directions_with_results.append({
            "direction": chosen_dir.get("direction", ""),
            "notes": chosen_dir.get("notes", ""),
            "round_summary": direction_summaries.get(key, ""),
        })
    return directions_with_results


def build_round_summary_payload(
    round_result,
    analysis_state,
    effective_max_rounds: int,
    data_requests: dict | None = None,
    cross_modal_candidates: list | None = None,
) -> dict:
    """Build the round_summary WebSocket payload from a RoundResult.

    Shared between _pi_synthesis (exploration/innovation) and
    _integration_synthesis (integration) so the shape stays consistent.
    """
    return {
        "round": round_result.round_number,
        "max_rounds": effective_max_rounds,
        "summary": round_result.pi_summary,
        "pathology_findings": round_result.pathology_findings,
        "bioinformatics_findings": round_result.bioinformatics_findings,
        "biochemistry_findings": round_result.biochemistry_findings,
        "literature_findings": round_result.literature_findings,
        "findings": [
            *round_result.pathology_findings,
            *round_result.bioinformatics_findings,
            *round_result.biochemistry_findings,
            *round_result.literature_findings,
        ],
        "cross_modal_candidates": cross_modal_candidates or [],
        "data_requests": data_requests or {"pathology": [], "biochemistry": [], "other": []},
        "status": "waiting_input",
        "phase": round_result.phase,
        "integration_candidates": round_result.integration_candidates,
        "integration_report_summary": round_result.integration_report_summary,
        "round_status": round_result.round_status,
        "evidence_status": round_result.evidence_status,
        "direction_coverage_eligible": round_result.direction_coverage_eligible,
        "bioinformatics_evidence_verified": round_result.bioinformatics_evidence_verified,
        "evidence_gate_degraded": round_result.evidence_gate_degraded,
        "evidence_execution_ids": list(round_result.evidence_execution_ids or []),
        "evidence_notes": list(round_result.evidence_notes or []),
        "evidence_gate_findings": list(round_result.evidence_gate_findings or []),
        "lineage_id": round_result.lineage_id,
        "branch_id": round_result.branch_id,
        "branch_label": round_result.branch_label,
        "restored_from_snapshot_id": round_result.restored_from_snapshot_id,
        "direction": round_result.direction,
        "direction_key": round_result.direction_key,
        "result_identity_key": round_result.result_identity_key,
        "evidence_schema_version": round_result.evidence_schema_version,
    }


class SynthesisMixin:
    """Mixin providing synthesis, user-input, and convergence methods. Composed into SingleCellAnalysisWorkflow."""

    _LIT_AUDIT_CHAR_CAP = 32000  # ~8K tokens

    def _build_literature_audit_context(self, state) -> str:
        """Bounded literature context for the cross-modal audit. Priority:
        (1) persisted mechanism_signals, (2) reference titles/abstracts,
        (3) report body excerpt. Hard-capped.

        IMPORTANT (E2): references and abstracts are read from the persisted
        `analysis_state.literature_reports` objects, NOT from
        `state["literature_result"]`. The combined `literature_result` dict built by
        `_literature_research` holds only topics/findings/reports/search_modes — it has
        NO "references" key (verified). The per-topic references are persisted on each
        LiteratureReport, which also survive snapshot restore / session reload.
        """
        analysis_state = state["analysis_state"]
        parts = []

        # (1) mechanism_signals from persisted LiteratureReports (last 8)
        for rep in analysis_state.literature_reports[-8:]:
            if rep.mechanism_signals:
                for sig in rep.mechanism_signals:
                    parts.append(
                        f"- Mechanism: {sig.mechanism} "
                        f"(PMIDs: {', '.join(sig.supporting_pmids) or 'N/A'}; "
                        f"type: {sig.evidence_type}); relevance: {sig.relevance_to_current_study}; "
                        f"links: {', '.join(sig.observed_data_links) or 'N/A'}; "
                        f"limitations: {', '.join(sig.limitations) or 'N/A'}"
                    )

        # (2) reference titles/abstracts from persisted LiteratureReports.
        # Abstracts are NOT guaranteed (PubMed mode may drop them; Deep Research has none),
        # so degrade gracefully. Bound to the most recent reports to approximate the
        # current round; the hard char cap below is the real bound.
        ref_count = 0
        for rep in analysis_state.literature_reports[-4:]:
            for ref in (rep.references or []):
                if not isinstance(ref, dict):
                    continue
                title = ref.get("title", "")
                abstract = ref.get("abstract", "")  # may be absent — degrade gracefully
                pmid = ref.get("pmid", "")
                line = f"- {title}"
                if pmid:
                    line += f" (PMID: {pmid})"
                if abstract:
                    line += f": {abstract[:400]}"
                parts.append(line)
                ref_count += 1
                if ref_count >= 30:
                    break
            if ref_count >= 30:
                break

        # (3) current round's report body excerpts (this key DOES exist on literature_result)
        lit_result = state.get("literature_result") or {}
        for report_body in (lit_result.get("reports") or [])[:3]:
            if report_body:
                parts.append(f"Report excerpt: {report_body[:2000]}")

        text = "\n".join(parts)
        return text[:self._LIT_AUDIT_CHAR_CAP]

    def _build_biochemistry_raw_context(self, state) -> str:
        """Raw biochemistry measurements + manual indicators for the current round."""
        analysis_state = state["analysis_state"]
        parts = []
        bio_result = state.get("biochemistry_result") or {}
        findings = bio_result.get("findings") or []
        if findings:
            parts.append("Findings: " + "; ".join(str(f) for f in findings))
        if analysis_state.biochemistry_indicators:
            for ind in analysis_state.biochemistry_indicators[:50]:
                parts.append(
                    f"- {ind.indicator}: {ind.value}{ind.unit or ''} "
                    f"(ref: {ind.reference_range or 'N/A'}; {ind.interpretation or ''})"
                )
        return ("\n".join(parts))[:16000]

    def _exploration_round_succeeded(self, state) -> bool:
        """P1-12: whitelist — exploration succeeded ONLY when bioinformatics
        explicitly reported status='completed'. All other statuses
        (error, failed, timeout, max_steps_reached, completed_with_fallback,
        waiting_data, waiting_input, skipped, no_data, unavailable, empty,
        missing) mean the exploration was incomplete and we must NOT
        advance EXPLORATION -> INNOVATION; the user retains the option to
        retry stage 1 in the next round.

        Also honors @with_retry's failed_node signal (HITL Skip path).
        """
        if state.get("failed_node") == "bioinformatics_analysis":
            return False
        bio = state.get("bioinformatics_result") or {}
        return bio.get("status") == "completed"

    def _skipped_direction_evidence(self, raw_candidates, reason):
        return build_fallback_direction_evidence(raw_candidates, status="skipped", reason=reason)

    async def _generate_evidence_gated_exploration_report(self, graph_state, analysis_state, round_result):
        raw_candidates = await self.pi_agent.generate_raw_direction_candidates(analysis_state, round_result)
        literature_result = (graph_state or {}).get("literature_result") or {}
        if literature_result.get("status") in {"skipped", "disabled"}:
            reason = "Direction-level literature verification was skipped because the Literature Agent was skipped or disabled earlier in the workflow."
            direction_evidence = self._skipped_direction_evidence(raw_candidates, reason)
            report = await self.pi_agent.score_directions_with_evidence(analysis_state, round_result, raw_candidates, direction_evidence)
            self.pi_agent._sanitize_direction_dependency_graph(report)
            return report

        try:
            direction_evidence = await self.literature_agent.evaluate_direction_novelty_candidates(raw_candidates, state=analysis_state)
        except Exception as lit_err:
            logger.exception(f"[{self.session_id}] Direction literature evidence gate failed")
            direction_evidence = build_fallback_direction_evidence(
                raw_candidates,
                status="failed",
                reason=f"Direction-level literature verification failed: {lit_err}",
                limitations=["Literature verification failed; PI scoring must be conservative."],
            )
        report = await self.pi_agent.score_directions_with_evidence(analysis_state, round_result, raw_candidates, direction_evidence)
        self.pi_agent._sanitize_direction_dependency_graph(report)
        return report

    async def _pi_synthesis(self, state: WorkflowState) -> WorkflowState:
        """PI synthesizes all results and prepares round summary for user"""
        await ack_restore_safe_point(self)
        analysis_state = state["analysis_state"]

        provider = getattr(self, "_settings_provider", None)
        pathology_enabled = True
        if provider is not None:
            model = provider.get_model("pathology")
            pathology_enabled = bool(model and model.strip())
        file_dicts = [f.model_dump() for f in analysis_state.data_files]
        file_dicts.extend(analysis_state.pending_uploaded_files)
        modality_warning = build_disabled_pathology_warning(file_dicts, pathology_enabled)
        if modality_warning:
            await self._send_message("modality_warning", modality_warning, AgentType.PI)

        skip = await self._require_core_agent(state, "pi", "pi_synthesis")
        if skip is not None:
            return skip

        logger.info(f"[{self.session_id}] PI synthesis for round {analysis_state.current_round}")

        await self._send_message("status", "PI is synthesizing analysis results...", AgentType.PI)
        
        # Extract findings from each agent, including error/status info for PI awareness
        agent_results = {}
        for agent_name, result_key in [
            ("biochemistry", "biochemistry_result"),
            ("pathology", "pathology_result"),
            ("bioinformatics", "bioinformatics_result"),
            ("literature", "literature_result"),
        ]:
            raw_result = state.get(result_key)
            result = raw_result if isinstance(raw_result, dict) else {}
            findings = result.get("findings", [])
            error = result.get("error")
            status = result.get("status")

            # Build result string with error/status context so PI knows what happened
            if not result:
                agent_results[agent_name] = json.dumps(
                    {"status": "not_run", "findings": []}, ensure_ascii=False
                )
            elif error:
                agent_results[agent_name] = json.dumps(
                    {"error": error, "findings": findings}, ensure_ascii=False
                )
                logger.warning(f"[{self.session_id}] {agent_name} had error during this round: {error}")
            elif status in ("skipped", "no_data", "waiting_data"):
                agent_results[agent_name] = json.dumps(
                    {"status": status, "findings": findings}, ensure_ascii=False
                )
            elif agent_name == "bioinformatics":
                # The plan audit and the executor's own suggestions are included so
                # the PI synthesizes against what the round actually ran, not only
                # against the findings it chose to report. An unmet approved step is
                # the strongest signal that a conclusion should stay hedged.
                review = result.get("post_execution_review") or {}
                audit = result.get("plan_audit") or {}
                # Three states, not two. `None` means the binding status was never
                # recorded — a cached result, a restored snapshot, or a path that did
                # not carry it — and it must not read as verified. Treating unknown
                # as verified is what told the PI its findings were bound to
                # successful executions on every round, whatever actually happened.
                evidence_verified = result.get("evidence_verified")
                if evidence_verified is True:
                    evidence_note = "Findings are linked to successful execution records."
                elif evidence_verified is False:
                    evidence_note = (
                        "Treat these findings as partial/unverified and do not use "
                        "validated, confirmed, or dominant wording."
                    )
                else:
                    evidence_note = (
                        "Evidence binding for these findings is unknown; treat them as "
                        "unverified and do not use validated, confirmed, or dominant wording."
                    )
                agent_results[agent_name] = json.dumps(
                    {
                        "status": status or "completed",
                        "findings": findings,
                        "evidence_verified": evidence_verified,
                        "evidence_execution_ids": result.get("evidence_execution_ids", []),
                        "evidence_note": evidence_note,
                        "agent_suggestions": result.get("agent_suggestions", []),
                        # n_unverifiable belongs here even though nothing acts on
                        # it: without it the counts do not sum to n_steps, and a
                        # coverage report that silently omits what it did not
                        # check reads as a completeness claim the audit never made.
                        "plan_coverage": {
                            k: audit.get(k)
                            for k in ("n_steps", "n_completed", "n_skipped", "n_unmet",
                                      "n_unverifiable", "unmet_titles", "unmet_families")
                            if audit.get(k) is not None
                        } or None,
                        "supplementary_pass": {
                            "status": review.get("review_status"),
                            "verdict": (review.get("assessment") or {}).get("verdict"),
                            "remaining_gaps": (review.get("assessment") or {}).get("remaining_gaps", []),
                        } if review.get("review_status") in ("completed", "partial", "failed") else None,
                    },
                    ensure_ascii=False,
                )
            else:
                agent_results[agent_name] = json.dumps(findings, ensure_ascii=False)

        # Cross-modal audit (Gap 0/1/2 fix) — innovation rounds only, non-critical.
        cms = None
        if analysis_state.analysis_phase == AnalysisPhase.INNOVATION:
            try:
                current_agent_raw = {
                    "biochemistry": state.get("biochemistry_result") or {},
                    "pathology": state.get("pathology_result") or {},
                    "bioinformatics": state.get("bioinformatics_result") or {},
                }
                literature_context = self._build_literature_audit_context(state)
                biochemistry_raw = self._build_biochemistry_raw_context(state)
                cms = await self.pi_agent.build_cross_modal_synthesis(
                    state=analysis_state,
                    current_direction=analysis_state.current_round_direction,
                    current_agent_raw=current_agent_raw,
                    literature_context=literature_context,
                    biochemistry_raw=biochemistry_raw,
                )
            except Exception as e:
                logger.warning(f"[{self.session_id}] cross-modal synthesis failed (non-critical): {e}")
                cms = None

        round_result = await self.pi_agent.synthesize_round(
            analysis_state,
            pathology_result=agent_results["pathology"],
            bioinformatics_result=agent_results["bioinformatics"],
            biochemistry_result=agent_results["biochemistry"],
            literature_result=agent_results["literature"],
            research_direction=analysis_state.current_round_direction,
            cross_modal_context=(cms.matrix_markdown if cms else ""),
        )
        if cms is not None:
            round_result.cross_modal_synthesis = cms

        logger.info(f"[{self.session_id}] Round {round_result.round_number} synthesis complete: new_findings={round_result.new_findings}, {len(round_result.next_steps)} next steps")

        # Tag round with current phase and direction
        round_result.phase = analysis_state.analysis_phase
        round_result.direction = analysis_state.current_round_direction or ""

        # Exploration phase: generate structured Report A and transition to innovation.
        # Skip both the Report A generation and the phase transition when this
        # round's bioinformatics analysis failed — otherwise we'd produce an
        # exploration report from empty findings and lock the user into
        # INNOVATION phase next round with no path to retry stage 1.
        if (
            analysis_state.analysis_phase == AnalysisPhase.EXPLORATION
            and self._exploration_round_succeeded(state)
        ):
            await self._send_message(
                "status",
                "Generating exploration report (Report A)...",
                AgentType.PI
            )
            report_generated = False
            try:
                exploration_report = await self._generate_evidence_gated_exploration_report(
                    state, analysis_state, round_result
                )
                analysis_state.exploration_report = exploration_report
                round_result.exploration_report = exploration_report

                # Generate exploration report HTML
                try:
                    from ..services.report import generate_exploration_report_html
                    html_path = generate_exploration_report_html(
                        analysis_state, exploration_report, round_result
                    )
                    analysis_state.exploration_report_html_path = html_path
                    logger.info(f"[{self.session_id}] Exploration report HTML saved to: {html_path}")
                except Exception as html_err:
                    logger.error(f"[{self.session_id}] Exploration report HTML generation failed: {html_err}")

                html_url = f"/api/session/{self.session_id}/exploration-report/html" if analysis_state.exploration_report_html_path else None

                # Generate detailed report segment for exploration completion
                await self._build_and_save_detailed_segment(
                    state=analysis_state,
                    phase="exploration_complete",
                    title="Exploration Phase Complete - Analysis Summary",
                    graph_state=state,
                    round_result=round_result,
                )
                detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if analysis_state.detailed_report_html_path else None

                # Auto-snapshot before direction selection HITL
                await self._create_auto_snapshot(analysis_state, "pre_direction")

                await self._send_message("exploration_report", {
                    "report_a": exploration_report.model_dump(mode="json"),
                    "round": round_result.round_number,
                    "summary": round_result.pi_summary,
                    "html_url": html_url,
                    "detailed_report_url": detailed_report_url,
                }, AgentType.PI)
                report_generated = True
            except Exception as e:
                logger.exception(f"[{self.session_id}] Exploration report generation failed")
                await self._send_message(
                    "error",
                    f"Exploration report generation failed: {e}. Staying in the exploration phase — please fix the issue and retry generating the exploration report and direction candidates.",
                    AgentType.PI
                )

            # Only advance to innovation when an exploration report was actually produced.
            if report_generated:
                analysis_state.analysis_phase = AnalysisPhase.INNOVATION
                logger.info(f"[{self.session_id}] Transitioning to innovation phase")
            else:
                logger.warning(
                    f"[{self.session_id}] Exploration report not generated; staying in EXPLORATION phase for retry"
                )

        elif analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
            # Exploration did NOT succeed (bioinformatics stage1 failed/incomplete).
            # Most often this means the uploaded data has a problem. Surface an
            # explicit prompt so the user can fix/re-upload data and retry stage 1,
            # rather than silently advancing rounds toward max_rounds.
            bio = state.get("bioinformatics_result") or {}
            reason_code = classify_exploration_failure(bio)
            await self._send_message(
                "exploration_failed",
                {
                    "round": round_result.round_number,
                    "reason": reason_code,
                },
                AgentType.PI,
            )
            logger.warning(
                f"[{self.session_id}] Exploration failed (reason={reason_code}; "
                f"raw_status={bio.get('status')!r} raw_error={bio.get('error')!r}); "
                f"staying in EXPLORATION, prompting user to check data"
            )

        bioinformatics_result = state.get("bioinformatics_result") or {}
        round_result.data_driven_mechanism_programs = bioinformatics_result.get("mechanism_programs", [])
        round_result.bioinformatics_evidence_verified = bioinformatics_result.get("evidence_verified")
        round_result.evidence_gate_degraded = bool(
            bioinformatics_result.get("evidence_gate_degraded")
            or any(
                finding.get("forced_release")
                for finding in bioinformatics_result.get("evidence_gate_findings") or []
                if isinstance(finding, dict)
            )
        )
        round_result.evidence_execution_ids = list(
            bioinformatics_result.get("evidence_execution_ids") or []
        )
        round_result.evidence_gate_findings = list(
            bioinformatics_result.get("evidence_gate_findings") or []
        )
        for gate_finding in round_result.evidence_gate_findings:
            if not gate_finding.get("forced_release"):
                continue
            gate = gate_finding.get("gate") or "mandatory-evidence-gate"
            rule_id = gate_finding.get("rule_id") or "unregistered-rule"
            count = gate_finding.get("rejection_count", 0)
            limit = gate_finding.get("rejection_limit", 0)
            round_result.evidence_notes.append(
                f"Mandatory gate {gate} ({rule_id}) was force-released after "
                f"{count}/{limit} rejection attempts; affected findings are degraded."
            )
        if round_result.bioinformatics_evidence_verified is False:
            round_result.evidence_notes.append(
                "Bioinformatics findings were not fully bound to successful execution evidence; "
                "treat them as partial and unvalidated."
            )
        derive_round_completion_status(state, round_result)
        lineage_id = current_lineage_id(analysis_state)
        restored_from = restore_provenance_id(analysis_state)
        _upsert_round_result(
            analysis_state.round_results,
            round_result,
            lineage_id=lineage_id,
            branch_id=None if lineage_id == "main" else lineage_id,
            branch_label=analysis_state.restore_branch_label,
            restored_from_snapshot_id=restored_from,
        )
        if restored_from is not None:
            consume_restore_provenance(analysis_state)
        from .direction_coverage import active_batch_complete
        if (
            analysis_state.analysis_phase == AnalysisPhase.INNOVATION
            and active_batch_complete(analysis_state)
        ):
            analysis_state.pending_integration = True
            logger.info(f"[{self.session_id}] Direction batch complete -> pending_integration=True")
        # Round is now canonical in round_results; clear the per-round cache
        # so a later snapshot-restored round can't replay stale agent outputs.
        analysis_state.current_round_cache = None



        # P1: innovation rounds also get an incremental detailed-report segment.
        # Exploration round is already covered at the exploration_complete call above.
        if _should_build_innovation_segment(analysis_state, round_result):
            try:
                await self._build_and_save_detailed_segment(
                    analysis_state,
                    f"innovation_round_{analysis_state.current_round}",
                    f"Innovation Round {analysis_state.current_round} — Analysis Results",
                    graph_state=state,
                    round_result=round_result,
                )
            except Exception as e:
                logger.exception(
                    f"[{self.session_id}] Innovation detailed segment generation failed"
                )

        # Save state after each round
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.exception("Failed to save state after round")

        # Bound long-running checkpoint growth at a semantic round boundary.
        # Pruning includes associated writes and dangling parent links; VACUUM
        # remains close-time maintenance because it takes a stronger SQLite
        # lock and is unnecessary on every round.
        try:
            await self._compact_checkpoints(vacuum=False)
        except Exception:
            logger.exception(
                "[%s] Failed to compact checkpoints at round boundary",
                self.session_id,
            )

        # Extract data requests from PI
        try:
            data_requests = await self.pi_agent.identify_data_needs(
                analysis_state,
                round_result
            )
        except Exception as e:
            logger.warning(f"Failed to identify data needs: {e}")
            data_requests = []
        
        # Send round summary with data requests
        effective_max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        cross_modal_candidates = []
        if round_result.cross_modal_synthesis:
            for c in round_result.cross_modal_synthesis.candidates:
                cross_modal_candidates.append(c.to_summary_dict())

        round_summary = build_round_summary_payload(
            round_result, analysis_state, effective_max_rounds,
            data_requests=data_requests,
            cross_modal_candidates=cross_modal_candidates,
        )
        
        await self._send_message(
            "round_summary",
            round_summary,
            AgentType.PI
        )
        
        # Also send traditional result message for compatibility
        await self._send_message(
            "result",
            {
                "round": round_result.round_number,
                "summary": round_result.pi_summary,
                "next_steps": round_result.next_steps,
                "new_findings": round_result.new_findings
            },
            AgentType.PI
        )
        
        # Auto-snapshot before round wait HITL
        await self._create_auto_snapshot(analysis_state, "pre_round_wait")

        # Clear continue event BEFORE announcing waiting status to prevent race
        # condition where frontend signals between status message and event.wait()
        self._continue_event.clear()

        # Set status to waiting for user input
        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)

        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        is_last_round = analysis_state.current_round >= max_rounds
        if is_last_round:
            await self._send_message(
                "status",
                f"Round {round_result.round_number} complete (final round). You can provide additional data to extend analysis, or skip to generate the final report.",
                AgentType.PI
            )
        else:
            await self._send_message(
                "status",
                f"Round {round_result.round_number} complete. Waiting for your input - you can provide additional data or instructions.",
                AgentType.PI
            )

        # Send state sync for round completion
        await self._send_state_sync(analysis_state, "round_complete")

        return {
            **state,
            "analysis_state": analysis_state,
            "round_result": round_result,
            "data_requests": data_requests
        }

    async def _wait_for_user_input(self, state: WorkflowState) -> WorkflowState:
        """Block between rounds, waiting for user input or continue signal from frontend.

        Behavior depends on analysis_state.round_wait_seconds (the user-facing
        "skip time" setting, unified across all HITL gates):
          -1  = manual mode: wait indefinitely for user signal
           0  = no-wait mode: proceed immediately without waiting
          >0  = countdown mode: frontend shows countdown timer, backend waits
                `wait_seconds + 60s` then auto-skips (60s buffer for client lag)

        Auto-skip semantics for the post-exploration direction-selection gate:
        equivalent to "let PI decide" (sets user_chosen_direction = [PI_DECIDE]).
        """
        await ack_restore_safe_point(self)
        analysis_state = state["analysis_state"]

        # Integration gate: auto-skip / timeout now triggers PI_DECIDE_SIGNAL
        # (instead of final_report_requested), so PI gets a chance to select
        # directions from integration candidates. User explicit finalize (via
        # /finalize-from-integration) still sets final_report_requested and
        # is handled first.
        is_integration_gate = (
            state.get("semantic_gate") == GATE_INTEGRATION_SELECTION
            or at_integration_gate(analysis_state)
        )
        if is_integration_gate:
            wait_seconds = analysis_state.round_wait_seconds
            if wait_seconds == 0:
                analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                analysis_state.pending_human_input = PI_DECIDE_SIGNAL
            else:
                signaled = await wait_with_skip_time(
                    self._continue_event, wait_seconds, self.session_id, "integration_gate",
                    safe_point_hook=getattr(self, "await_restore_safe_point", None),
                )
                if not signaled:
                    analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                    analysis_state.pending_human_input = PI_DECIDE_SIGNAL
                else:
                    _consume_signal_then_clear(self._continue_event)

            # User explicit finalize (via /finalize-from-integration API) takes priority.
            if analysis_state.consume_final_report_request():
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "integration_finalize")
                return {
                    **state,
                    "analysis_state": analysis_state,
                    "should_continue": False,
                    "semantic_gate": None,
                }

            # Robustness: a generic continue (e.g. /human-input with no direction,
            # or any signal that did not go through /choose-direction) leaves the
            # gate unresolved — phase still INTEGRATION with candidates populated.
            # Routing would then send the graph to initialize_round and start a
            # fresh round with no integration direction selected. Treat such a
            # continue as "let PI decide" (same default as auto-skip / no-wait) so
            # the gate always resolves. An explicit selection clears the candidates
            # (at_integration_gate becomes False) and is left untouched here.
            if analysis_state.user_chosen_direction != PI_DECIDE_SIGNAL and at_integration_gate(analysis_state):
                analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                analysis_state.pending_human_input = PI_DECIDE_SIGNAL

            # Auto-skip set PI_DECIDE_SIGNAL, or user chose directions via API.
            # If PI_DECIDE_SIGNAL, switch to INNOVATION so pi_planning handles it.
            if analysis_state.user_chosen_direction == PI_DECIDE_SIGNAL:
                analysis_state.analysis_phase = AnalysisPhase.INNOVATION
            self._set_status(analysis_state, TaskStatus.RUNNING)
            await self._send_state_sync(analysis_state, "integration_continue")
            return {
                **state,
                "analysis_state": analysis_state,
                "should_continue": True,
                "semantic_gate": None,
            }

        # Note: _continue_event is cleared in _pi_synthesis before the waiting
        # status is announced, to prevent a race condition with signal_continue().

        wait_seconds = analysis_state.round_wait_seconds  # -1=manual, 0=no wait, >0=countdown
        is_direction_gate = (
            analysis_state.current_round == 1
            and analysis_state.analysis_phase == AnalysisPhase.INNOVATION
        )

        # Filter out empty/whitespace-only directions from queue
        if analysis_state.directions_queue:
            analysis_state.directions_queue = [
                d for d in analysis_state.directions_queue if d and d.get("direction", "").strip()
            ]

        if wait_seconds == 0:
            # No-wait mode: proceed immediately (with let-PI-decide for direction gate)
            logger.info(f"[{self.session_id}] No-wait mode, proceeding immediately (round {analysis_state.current_round})")
            if is_direction_gate and not analysis_state.user_chosen_direction:
                analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                analysis_state.pending_human_input = PI_DECIDE_SIGNAL
        else:
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")

            # P0-6: Do NOT clear _continue_event here. It was already cleared in
            # _pi_synthesis before WAITING_INPUT was announced; clearing again
            # would discard signals delivered during LangGraph's checkpoint I/O
            # window between the two nodes, forcing the user to click twice.
            while True:
                signaled = await wait_with_skip_time(
                    self._continue_event,
                    wait_seconds,
                    self.session_id,
                    GATE_DIRECTION_SELECTION if is_direction_gate else GATE_ROUND_WAIT,
                    safe_point_hook=getattr(self, "await_restore_safe_point", None),
                )
                if not signaled:
                    # Auto-skip on timeout
                    if is_direction_gate and not analysis_state.user_chosen_direction:
                        analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
                        analysis_state.pending_human_input = PI_DECIDE_SIGNAL
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "auto_continue")
                    break
                _consume_signal_then_clear(self._continue_event)

                if self._supplementary_queue and self._supplementary_queue[0][1] == "direction":
                    req, _target = self._supplementary_queue.popleft()
                    _phase = "integration" if analysis_state.analysis_phase == AnalysisPhase.INTEGRATION else "exploration"
                    _rec = get_or_open_direction_record(
                        analysis_state, round_number=analysis_state.current_round,
                        phase=_phase,
                        integration_cycle=analysis_state.integration_cycle if _phase == "integration" else None,
                    )
                    _rec.supplementary_requests.append(req)
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "supplementary_running")

                    await self._run_supplementary_analysis(analysis_state, req, "direction")

                    try:
                        last_round = analysis_state.round_results[-1] if analysis_state.round_results else None
                        if last_round and analysis_state.exploration_report:
                            exploration_report = await self._generate_evidence_gated_exploration_report(
                                state, analysis_state, last_round
                            )
                            analysis_state.exploration_report = exploration_report

                            from ..services.report import generate_exploration_report_html
                            html_path = generate_exploration_report_html(
                                analysis_state, exploration_report, last_round
                            )
                            analysis_state.exploration_report_html_path = html_path
                            html_url = f"/api/session/{self.session_id}/exploration-report/html"
                            detailed_report_url = f"/api/session/{self.session_id}/detailed-report/html" if analysis_state.detailed_report_html_path else None

                            await self._send_message("exploration_report", {
                                "report_a": exploration_report.model_dump(mode="json"),
                                "round": last_round.round_number,
                                "summary": last_round.pi_summary,
                                "html_url": html_url,
                                "detailed_report_url": detailed_report_url,
                            }, AgentType.PI)
                    except Exception as e:
                        logger.error(f"[{self.session_id}] Exploration report regeneration failed: {e}")
                        await self._send_message("error", f"Report regeneration failed: {e}", AgentType.PI)

                    self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                    await self._send_state_sync(analysis_state, "waiting_input")
                    continue
                elif self._supplementary_queue and self._supplementary_queue[0][1] == "round":
                    req, _target = self._supplementary_queue.popleft()
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "supplementary_running")

                    await self._run_supplementary_analysis(analysis_state, req, "round")

                    self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                    await self._send_state_sync(analysis_state, "waiting_input")
                    continue
                else:
                    logger.info(f"[{self.session_id}] Continue signal received")
                    break

        # If user provided real input (text or files) on the last round, extend max_rounds to allow one more
        has_real_text = (
            analysis_state.pending_human_input
            and analysis_state.pending_human_input.strip() not in ('', '[CONTINUE]', '[PI_DECIDE]')
        )
        has_uploaded_files = bool(analysis_state.pending_uploaded_files)
        max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        if (has_real_text or has_uploaded_files) and analysis_state.current_round >= max_rounds:
            # Don't extend if multi-direction mode has completed all directions
            # AND the user didn't provide real new input/files (only synthetic signals).
            # If the user explicitly typed instructions or uploaded files, honor their intent.
            innovation_rounds_done = count_innovation_rounds_done(analysis_state)
            multi_dir_completed = (
                len(analysis_state.user_chosen_directions) > 1
                and innovation_rounds_done >= len(analysis_state.user_chosen_directions)
            )
            if multi_dir_completed and not has_real_text:
                logger.info(f"[{self.session_id}] Multi-direction completed, not extending max_rounds")
            else:
                analysis_state.max_rounds = analysis_state.current_round + 1
                logger.info(f"[{self.session_id}] User input on last round (text={has_real_text}, files={has_uploaded_files}), extending max_rounds to {analysis_state.max_rounds}")

        return {**state, "analysis_state": analysis_state}

    async def _check_convergence(self, state: WorkflowState) -> WorkflowState:
        """Check if analysis has converged.

        Hard-stop conditions are handled before this node. Here we only decide
        whether soft convergence should end the workflow, unless pending user work
        explicitly requires another round.
        """
        analysis_state = state["analysis_state"]

        if _should_force_next_round(analysis_state):
            logger.info(
                f"[{self.session_id}] Convergence overridden by pending user work: "
                f"text={bool(analysis_state.pending_human_input)}, "
                f"files={bool(analysis_state.pending_uploaded_files)}, "
                f"queue={len(analysis_state.directions_queue)}"
            )
            return {
                **state,
                "should_continue": True,
            }

        should_stop = analysis_state.should_stop(self.config.convergence_rounds)

        logger.info(
            f"[{self.session_id}] Convergence check: "
            f"round={analysis_state.current_round}, should_stop={should_stop}"
        )

        return {
            **state,
            "should_continue": not should_stop,
        }
    
