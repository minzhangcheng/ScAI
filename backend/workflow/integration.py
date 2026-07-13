"""Integration round: cross-direction synthesis + new-direction gate.

Orchestrates PI (cumulative integration + candidate generation + scoring) and
Literature (open search + novelty gate). Does NOT run bioinformatics/pathology/
biochemistry agents. See docs/superpowers/specs/2026-07-03-integration-round-loop-design.md.
"""
import logging
from typing import List

from ..models.schemas import AnalysisPhase, TaskStatus, RoundResult, AgentType
from .helpers import build_integration_gate_payload
from .retry import with_retry
from .state import WorkflowState

logger = logging.getLogger(__name__)


class IntegrationMixin:
    """integration_synthesis node + _after_integration routing decision."""

    session_id: str

    def _cumulative_innovation_rounds(self, analysis_state) -> List[RoundResult]:
        return [rr for rr in analysis_state.round_results if getattr(rr, "phase", None) == "innovation"]

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

        from .synthesis import _build_directions_with_results_for_correlation
        directions_with_results = _build_directions_with_results_for_correlation(analysis_state)
        integration_report = await self.pi_agent.analyze_direction_correlations(
            analysis_state, directions_with_results
        )

        try:
            new_literature = await self.literature_agent.integration_literature_search(
                analysis_state, integration_report
            )
        except Exception as e:
            logger.warning(f"[{session_id}] Integration literature search failed (non-critical): {e}")
            new_literature = ""

        cumulative_rounds = self._cumulative_innovation_rounds(analysis_state)
        raw_candidates = await self.pi_agent.generate_integration_direction_candidates(
            analysis_state, cumulative_rounds, integration_report, new_literature
        )

        scored_candidates = []
        literature_evidence = []
        if raw_candidates:
            try:
                literature_evidence = await self.literature_agent.evaluate_direction_novelty_candidates(
                    raw_candidates, state=analysis_state
                )
            except Exception as e:
                logger.warning(f"[{session_id}] Integration novelty gate failed: {e}")
                literature_evidence = []
            scored_candidates = await self.pi_agent.score_integration_directions_with_evidence(
                analysis_state, raw_candidates, literature_evidence
            )

        analysis_state.current_round += 1
        analysis_state.integration_cycle += 1
        analysis_state.last_integrated_round = analysis_state.current_round
        analysis_state.integration_reports.append(integration_report)
        analysis_state.multi_direction_correlation = integration_report
        analysis_state.pending_integration = False

        # Create RoundResult for the integration cycle
        from .synthesis import _upsert_round_result, build_round_summary_payload, derive_round_completion_status, current_lineage_id
        integration_result = RoundResult(
            round_number=analysis_state.current_round,
            phase="integration",
            pi_summary=integration_report[:2000] if integration_report else "Integration analysis completed.",
            integration_report_summary=integration_report[:4000] if integration_report else "",
            integration_candidates=list(scored_candidates) if scored_candidates else [],
            round_status="completed",
            evidence_status="substantive" if integration_report else "empty",
        )
        derive_round_completion_status({"analysis_state": analysis_state}, integration_result)
        lineage_id = current_lineage_id(analysis_state)
        from .provenance import consume_restore_provenance, restore_provenance_id
        restored_from = restore_provenance_id(analysis_state)
        _upsert_round_result(
            analysis_state.round_results,
            integration_result,
            lineage_id=lineage_id,
            branch_id=None if lineage_id == "main" else lineage_id,
            branch_label=analysis_state.restore_branch_label,
            restored_from_snapshot_id=restored_from,
        )
        if restored_from is not None:
            consume_restore_provenance(analysis_state)

        # Send round_summary so the frontend shows this integration round
        effective_max_rounds = analysis_state.max_rounds if analysis_state.max_rounds > 0 else self.config.max_rounds
        round_summary = build_round_summary_payload(
            integration_result, analysis_state, effective_max_rounds,
        )
        await self._send_message("round_summary", round_summary, AgentType.PI)

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
        if cap_reached or not scored_candidates:
            reason = ("Reached the integration analysis limit; proceeding to the final report."
                      if cap_reached else
                      "No sufficiently novel research direction was found; proceeding to the final report.")
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
        analysis_state.analysis_phase = AnalysisPhase.INTEGRATION
        self._continue_event.clear()
        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
        await self._create_auto_snapshot(analysis_state, "pre_integration_gate")
        await self._send_message(
            "integration_gate",
            build_integration_gate_payload(analysis_state, session_id),
            AgentType.PI,
        )
        await self._send_state_sync(analysis_state, "integration_gate")
        return {**state, "analysis_state": analysis_state, "integration_next": "gate"}

    def _after_integration(self, state: WorkflowState) -> str:
        return "gate" if state.get("integration_next") == "gate" else "stop"
