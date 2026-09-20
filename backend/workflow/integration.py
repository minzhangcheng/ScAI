"""Integration round: cross-direction synthesis + new-direction gate.

Orchestrates PI (cumulative integration + candidate generation + scoring) and
Literature (open search + novelty gate). Does NOT run bioinformatics/pathology/
biochemistry agents. See docs/superpowers/specs/2026-07-03-integration-round-loop-design.md.
"""
import asyncio
import logging
from typing import List

from ..models.schemas import AnalysisPhase, TaskStatus, RoundResult, AgentType
from ..utils.direction_payload import merge_unpursued_candidates
from ..models.schema_parts.round_report import RoundReportBlock, RoundReportChapter
from .helpers import build_integration_gate_payload
from ..utils.direction_evidence import build_fallback_direction_evidence
from .retry import with_retry
from .state import WorkflowState
from ..utils.llm_client import UpstreamExhaustedError
from .helpers import WorkflowSupersededError, WorkflowTerminatedError

logger = logging.getLogger(__name__)


class IntegrationMixin:
    """integration_synthesis node + _after_integration routing decision."""

    session_id: str

    def _cumulative_innovation_rounds(self, analysis_state) -> List[RoundResult]:
        from .direction_coverage import (
            current_lineage_id,
            normalize_legacy_round_result,
        )

        lineage = current_lineage_id(analysis_state)
        by_identity: dict[str, RoundResult] = {}
        for raw in analysis_state.round_results:
            result = normalize_legacy_round_result(raw)
            if result.phase != "innovation" or result.lineage_id != lineage:
                continue
            # Replayed/retried synthesis can leave duplicate legacy records.
            # The last retained record is the canonical upsert result.
            by_identity[result.result_identity_key] = result
        return list(by_identity.values())

    @with_retry("integration_synthesis")
    async def _integration_synthesis(self, state: WorkflowState) -> WorkflowState:
        """Integration round: cross-direction synthesis + new-direction gate.

        When the user chooses new directions at the integration gate and those
        directions complete, a subsequent integration cycle will have fresh
        round_results to analyze. If the user does NOT choose new directions
        (or the PI cannot generate novel ones), the integration cycle may
        produce a report identical to the previous one because the underlying
        data has not changed. This is by design — the cycle cap
        (max_integration_cycles) prevents infinite repetition.
        """
        analysis_state = state["analysis_state"]
        session_id = self.session_id
        logger.info(f"[{session_id}] Integration round (cycle {analysis_state.integration_cycle + 1})")
        await self._send_message("status", "PI is integrating results across all research directions...", AgentType.PI)

        from .synthesis import (
            _build_directions_with_results_for_correlation,
            _integration_inputs_are_partial,
        )
        directions_with_results = _build_directions_with_results_for_correlation(analysis_state)
        source_inputs_partial = _integration_inputs_are_partial(directions_with_results)
        integration_report = await self.pi_agent.analyze_direction_correlations(
            analysis_state, directions_with_results
        )

        try:
            new_literature = await self.literature_agent.integration_literature_search(
                analysis_state,
                integration_report,
                pi_plan_fn=self.pi_agent.plan_literature_search,
                pi_review_fn=self.pi_agent.review_literature_search,
            )
        except (UpstreamExhaustedError, WorkflowSupersededError, WorkflowTerminatedError, asyncio.CancelledError):
            raise
        except Exception as e:
            logger.warning(f"[{session_id}] Integration literature search failed (non-critical): {e}")
            new_literature = ""

        cumulative_rounds = self._cumulative_innovation_rounds(analysis_state)
        raw_candidates = await self.pi_agent.generate_integration_direction_candidates(
            analysis_state, cumulative_rounds, integration_report, new_literature
        )
        # An empty candidate list has two causes and only one of them is a
        # scientific statement. Capture the tooling failure immediately: further
        # down, an empty list is otherwise reported as "no sufficiently novel
        # research direction was found".
        candidate_failure = getattr(
            self.pi_agent, "last_candidate_generation_error", None
        )
        if candidate_failure:
            latest_round = analysis_state.get_latest_round()
            if latest_round is not None:
                notes = list(getattr(latest_round, "evidence_notes", []) or [])
                if candidate_failure not in notes:
                    notes.append(candidate_failure)
                    latest_round.evidence_notes = notes

        # Only now bring back what earlier gates left unpursued. A direction
        # deferred at an earlier gate is otherwise never reconsidered: generation
        # sees only the integration report, and the prerequisite that blocked it
        # may have completed rounds ago. Recycling before the check above would
        # have made a generation failure look like a successful proposal.
        raw_candidates = merge_unpursued_candidates(analysis_state, raw_candidates)
        carried_only = bool(raw_candidates) and bool(candidate_failure)
        if carried_only:
            logger.warning(
                f"[{session_id}] Candidate generation failed; the gate is being offered "
                f"{len(raw_candidates)} candidate(s) carried from earlier gates only"
            )

        scored_candidates = []
        literature_evidence = []
        if raw_candidates:
            try:
                literature_evidence = await self.literature_agent.evaluate_direction_literature_search(
                    raw_candidates,
                    state=analysis_state,
                    pi_plan_fn=self.pi_agent.plan_direction_literature_search,
                    pi_review_fn=self.pi_agent.review_direction_literature_search,
                    intervention_fn=self._wait_for_llm_intervention,
                    phase="integration",
                )
            except (UpstreamExhaustedError, WorkflowSupersededError, WorkflowTerminatedError, asyncio.CancelledError):
                raise
            except Exception as e:
                logger.warning(f"[{session_id}] Integration novelty gate failed: {e}")
                checkpoint_projection = getattr(
                    self.literature_agent, "direction_evidence_from_checkpoint", None
                )
                literature_evidence = (
                    checkpoint_projection(raw_candidates, analysis_state, phase="integration")
                    if callable(checkpoint_projection)
                    else build_fallback_direction_evidence(
                        raw_candidates,
                        status="failed",
                        reason=f"Direction-level literature verification failed technically: {e}",
                        limitations=["PI-directed literature assessment was unavailable; candidates were retained."],
                    )
                )
            scored_candidates = await self.pi_agent.score_integration_directions_with_evidence(
                analysis_state, raw_candidates, literature_evidence
            )
            scoring_failure = getattr(
                self.pi_agent, "last_candidate_scoring_error", None
            )
            if scoring_failure:
                latest_round = analysis_state.get_latest_round()
                if latest_round is not None:
                    notes = list(getattr(latest_round, "evidence_notes", []) or [])
                    if scoring_failure not in notes:
                        notes.append(scoring_failure)
                        latest_round.evidence_notes = notes

        analysis_state.current_round += 1
        analysis_state.integration_cycle += 1
        analysis_state.last_integrated_round = analysis_state.current_round
        analysis_state.integration_reports.append(integration_report)
        analysis_state.multi_direction_correlation = integration_report
        analysis_state.pending_integration = False

        # Create RoundResult for the integration cycle
        from .synthesis import _upsert_round_result, build_round_summary_payload, derive_round_completion_status, current_lineage_id
        summary_preview = integration_report or "Integration analysis completed."
        if len(summary_preview) > 2000:
            summary_preview = (
                summary_preview[:2000]
                + "\n\n[Condensed preview; the complete integration report is retained.]"
            )
        integration_result = RoundResult(
            round_number=analysis_state.current_round,
            phase="integration",
            pi_summary=summary_preview,
            integration_report_summary=integration_report or "",
            report_chapter=RoundReportChapter(
                title=f"Integration Cycle {analysis_state.integration_cycle}",
                blocks=[RoundReportBlock(text=integration_report or summary_preview)],
            ),
            integration_candidates=list(scored_candidates) if scored_candidates else [],
            round_status="completed",
            evidence_status="substantive" if integration_report else "empty",
            wait_policy=(
                "no_wait" if analysis_state.round_wait_seconds == 0
                else "manual" if analysis_state.round_wait_seconds < 0
                else "countdown"
            ),
        )
        derive_round_completion_status({"analysis_state": analysis_state}, integration_result)
        if source_inputs_partial:
            integration_result.round_status = "partial"
            integration_result.evidence_notes.append(
                "This integration includes one or more partial direction attempts; "
                "its synthesis is limited to available findings and recorded gaps."
            )
        lineage_id = current_lineage_id(analysis_state)
        from .provenance import consume_restore_provenance, restore_provenance_id
        restored_from = restore_provenance_id(analysis_state)
        integration_result = _upsert_round_result(
            analysis_state.round_results,
            integration_result,
            lineage_id=lineage_id,
            branch_id=None if lineage_id == "main" else lineage_id,
            branch_label=analysis_state.restore_branch_label,
            restored_from_snapshot_id=restored_from,
        )
        if restored_from is not None:
            consume_restore_provenance(analysis_state)

        # Integration is a recorded round even when it produces only text.
        # Capture its canonical boundary before any gate/report exit; no kernel
        # is started merely to create a backup.
        from .adata_checkpoint_hooks import capture_and_persist_boundary

        _capture_outcome = await capture_and_persist_boundary(
            self,
            analysis_state,
            boundary="round_complete",
            event_scope=str(integration_result.result_identity_key or "integration"),
            round_number=int(integration_result.round_number or 0),
            planning_stage="full",
            phase="integration",
            source_plan_id=None,
            round_result=integration_result,
            direction=integration_result.direction or "integration",
            summary_at_capture=str(integration_result.pi_summary or ""),
            outcome_at_capture=str(integration_result.round_status or ""),
            source_preparation_ids_at_capture=tuple(
                integration_result.adata_source_preparation_ids or []
            ),
        )
        if not _capture_outcome.available and _capture_outcome.disclosure:
            integration_result.evidence_notes = list(
                integration_result.evidence_notes or []
            ) + [
                "No canonical AnnData checkpoint was captured at this integration "
                f"boundary ({_capture_outcome.disclosure})."
            ]

        try:
            await self._build_and_save_detailed_segment(
                analysis_state,
                f"integration_cycle_{analysis_state.integration_cycle}",
                f"Integration Cycle {analysis_state.integration_cycle} — Cross-Direction Analysis",
                graph_state=state,
                round_result=integration_result,
            )
        except Exception as e:
            logger.warning(f"[{session_id}] Integration detailed segment failed: {e}")

        # Send round_summary so the frontend shows this integration round
        effective_max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        no_wait = analysis_state.round_wait_seconds == 0
        wait_policy = (
            "no_wait" if no_wait
            else "manual" if analysis_state.round_wait_seconds < 0
            else "countdown"
        )
        round_summary = build_round_summary_payload(
            integration_result, analysis_state, effective_max_rounds,
            status="completed" if no_wait else "waiting_input",
            wait_policy=wait_policy,
        )
        await self._send_message("round_summary", round_summary, AgentType.PI)

        try:
            from ..services.report import generate_integration_report_html
            html_path = generate_integration_report_html(
                analysis_state,
                candidates=list(scored_candidates) if scored_candidates else [],
                literature_evidence=list(literature_evidence or []),
            )
            analysis_state.integration_report_html_path = html_path
            logger.info(f"[{session_id}] Integration report HTML saved to: {html_path}")
        except Exception as html_err:
            logger.error(f"[{session_id}] Integration report HTML generation failed: {html_err}")

        cap_reached = analysis_state.integration_cycle >= analysis_state.max_integration_cycles
        # The integration node has completed regardless of whether it opens a
        # new-direction gate or proceeds directly to the final report. Persist
        # the actual terminal phase before either branch returns.
        analysis_state.analysis_phase = AnalysisPhase.INTEGRATION
        if cap_reached or not scored_candidates:
            if cap_reached:
                reason = "Reached the integration analysis limit; proceeding to the final report."
            elif candidate_failure:
                reason = (
                    "Direction-candidate generation was unavailable, so no new direction could "
                    "be proposed; proceeding to the final report. This is a tooling failure, "
                    "not a finding that no novel direction remains."
                )
            else:
                reason = (
                    "No sufficiently novel research direction was found; proceeding to the "
                    "final report."
                )
            analysis_state.integration_gate_candidates = []
            analysis_state.integration_gate_literature_evidence = []
            html_url = f"/api/session/{session_id}/integration-report/html" if analysis_state.integration_report_html_path else None
            await self._send_message("integration_complete", {
                "cycle": analysis_state.integration_cycle, "reason": reason,
                "html_url": html_url,
            }, AgentType.PI)
            return {**state, "analysis_state": analysis_state,
                    "integration_next": "stop", "should_continue": False}

        analysis_state.integration_gate_candidates = scored_candidates
        analysis_state.integration_gate_literature_evidence = list(literature_evidence or [])
        self._continue_event.clear()
        self._set_status(
            analysis_state,
            TaskStatus.RUNNING if no_wait else TaskStatus.WAITING_INPUT,
        )
        if no_wait:
            await self._send_message(
                "status",
                "Integration complete. No-wait mode is enabled; PI selection "
                "will continue automatically.",
                AgentType.PI,
            )
        else:
            await self._send_message(
                "integration_gate",
                build_integration_gate_payload(analysis_state, session_id),
                AgentType.PI,
            )
            await self._send_state_sync(analysis_state, "integration_gate")
        return {**state, "analysis_state": analysis_state, "integration_next": "gate"}

    def _after_integration(self, state: WorkflowState) -> str:
        return "gate" if state.get("integration_next") == "gate" else "stop"
