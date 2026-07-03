"""Integration round: cross-direction synthesis + new-direction gate.

Orchestrates PI (cumulative integration + candidate generation + scoring) and
Literature (open search + novelty gate). Does NOT run bioinformatics/pathology/
biochemistry agents. See docs/superpowers/specs/2026-07-03-integration-round-loop-design.md.
"""
import logging
from typing import List

from ..models.schemas import AnalysisPhase, TaskStatus, RoundResult, AgentType
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
        try:
            await self._build_and_save_detailed_segment(
                analysis_state,
                f"integration_cycle_{analysis_state.integration_cycle}",
                f"Integration Cycle {analysis_state.integration_cycle} — Cross-Direction Analysis",
                graph_state=state,
                round_result=None,
            )
        except Exception as e:
            logger.warning(f"[{session_id}] Integration detailed segment failed: {e}")

        cap_reached = analysis_state.integration_cycle >= analysis_state.max_integration_cycles
        if cap_reached or not scored_candidates:
            reason = ("Reached the integration analysis limit; proceeding to the final report."
                      if cap_reached else
                      "No sufficiently novel research direction was found; proceeding to the final report.")
            analysis_state.integration_gate_candidates = []
            await self._send_message("integration_complete", {
                "cycle": analysis_state.integration_cycle, "reason": reason,
            }, AgentType.PI)
            return {**state, "analysis_state": analysis_state,
                    "integration_next": "stop", "should_continue": False}

        analysis_state.integration_gate_candidates = scored_candidates
        analysis_state.analysis_phase = AnalysisPhase.INTEGRATION
        self._continue_event.clear()
        self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
        await self._create_auto_snapshot(analysis_state, "pre_integration_gate")
        detailed_report_url = (
            f"/api/session/{session_id}/detailed-report/html"
            if analysis_state.detailed_report_html_path else None
        )
        await self._send_message("integration_gate", {
            "cycle": analysis_state.integration_cycle,
            "integration_analysis": integration_report,
            "candidates": scored_candidates,
            "detailed_report_url": detailed_report_url,
        }, AgentType.PI)
        await self._send_state_sync(analysis_state, "integration_gate")
        return {**state, "analysis_state": analysis_state, "integration_next": "gate"}

    def _after_integration(self, state: WorkflowState) -> str:
        return "gate" if state.get("integration_next") == "gate" else "stop"
