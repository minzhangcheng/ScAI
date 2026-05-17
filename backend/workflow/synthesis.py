"""PI synthesis, user input wait, and convergence check."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constants import LlmFailureAction, WaitInputReason, StateSyncChangeType
from ..models.schemas import AgentType, AnalysisPhase, TaskStatus
from ..utils.llm_client import UpstreamExhaustedError

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


class SynthesisMixin:
    """Mixin providing synthesis, user-input, and convergence methods. Composed into SingleCellAnalysisWorkflow."""

    async def _pi_synthesis(self, state: WorkflowState) -> WorkflowState:
        """PI synthesizes all results and prepares round summary for user"""
        analysis_state = state["analysis_state"]

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
            result = state.get(result_key, {}) or {}
            findings = result.get("findings", [])
            error = result.get("error")
            status = result.get("status")

            # Build result string with error/status context so PI knows what happened
            if error:
                agent_results[agent_name] = json.dumps(
                    {"error": error, "findings": findings}, ensure_ascii=False
                )
                logger.warning(f"[{self.session_id}] {agent_name} had error during this round: {error}")
            elif status in ("skipped", "no_data", "waiting_data"):
                agent_results[agent_name] = json.dumps(
                    {"status": status, "findings": findings}, ensure_ascii=False
                )
            else:
                agent_results[agent_name] = json.dumps(findings, ensure_ascii=False)

        round_result = await self.pi_agent.synthesize_round(
            analysis_state,
            pathology_result=agent_results["pathology"],
            bioinformatics_result=agent_results["bioinformatics"],
            biochemistry_result=agent_results["biochemistry"],
            literature_result=agent_results["literature"],
            research_direction=analysis_state.current_round_direction,
        )

        logger.info(f"[{self.session_id}] Round {round_result.round_number} synthesis complete: new_findings={round_result.new_findings}, {len(round_result.next_steps)} next steps")

        # Tag round with current phase and direction
        round_result.phase = analysis_state.analysis_phase
        round_result.direction = analysis_state.current_round_direction or ""

        # Exploration phase: generate structured Report A and transition to innovation
        if analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
            await self._send_message(
                "status",
                "Generating exploration report (Report A)...",
                AgentType.PI
            )
            try:
                exploration_report = await self.pi_agent.generate_exploration_report(
                    analysis_state, round_result
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
            except Exception as e:
                logger.exception(f"[{self.session_id}] Exploration report generation failed")
                await self._send_message(
                    "error",
                    f"Exploration report generation failed: {e}. You can still select a research direction based on the analysis findings above.",
                    AgentType.PI
                )

            # Transition to innovation phase for next round
            analysis_state.analysis_phase = AnalysisPhase.INNOVATION
            logger.info(f"[{self.session_id}] Transitioning to innovation phase")

        analysis_state.round_results.append(round_result)

        # Multi-direction correlation analysis: trigger when queue is empty and multiple directions were selected
        if (
            not analysis_state.directions_queue
            and len(analysis_state.user_chosen_directions) > 1
            and analysis_state.analysis_phase == AnalysisPhase.INNOVATION
            and analysis_state.multi_direction_correlation is None
        ):
            try:
                await self._send_message("status", "Analyzing correlations between research directions...", AgentType.PI)
                # Build directions_with_results by matching direction names (not index)
                directions_with_results = []
                innovation_rounds = [r for r in analysis_state.round_results if r.phase == "innovation"]
                # Build lookup: direction name -> latest round summary
                direction_summaries: Dict[str, str] = {}
                for r in innovation_rounds:
                    if r.direction:
                        direction_summaries[r.direction] = r.pi_summary or ""
                for chosen_dir in analysis_state.user_chosen_directions:
                    dir_name = chosen_dir.get("direction", "")
                    round_summary = direction_summaries.get(dir_name, "")
                    directions_with_results.append({
                        "direction": dir_name,
                        "notes": chosen_dir.get("notes", ""),
                        "round_summary": round_summary
                    })
                correlation = await self.pi_agent.analyze_direction_correlations(
                    analysis_state, directions_with_results
                )
                analysis_state.multi_direction_correlation = correlation
                await self._send_message("direction_correlation", {
                    "correlation": correlation,
                    "directions": [d.get("direction", "") for d in analysis_state.user_chosen_directions]
                }, AgentType.PI)
                logger.info(f"[{self.session_id}] Multi-direction correlation analysis complete")
            except Exception as e:
                logger.exception(f"[{self.session_id}] Direction correlation analysis failed")

        # Save state after each round
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.exception("Failed to save state after round")
        
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
        round_summary = {
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
            "data_requests": data_requests,
            "status": "waiting_input"
        }
        
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

        Behavior depends on analysis_state.round_wait_seconds:
          -1  = manual mode: wait indefinitely for user signal
           0  = no-wait mode: proceed immediately without waiting
          >0  = countdown mode: frontend shows countdown timer, backend uses
                countdown + 60s buffer as safety timeout
        """
        analysis_state = state["analysis_state"]

        # Note: _continue_event is cleared in _pi_synthesis before the waiting
        # status is announced, to prevent a race condition with signal_continue().

        wait_seconds = analysis_state.round_wait_seconds  # -1=manual, 0=no wait, >0=countdown

        # Filter out empty/whitespace-only directions from queue
        if analysis_state.directions_queue:
            analysis_state.directions_queue = [
                d for d in analysis_state.directions_queue if d and d.get("direction", "").strip()
            ]
        # Note: no auto-skip for multi-direction mode — fall through to normal wait logic

        # After exploration phase (Round 1), force manual wait so user must choose direction
        if analysis_state.current_round == 1 and analysis_state.analysis_phase == AnalysisPhase.INNOVATION:
            wait_seconds = -1
            logger.info(f"[{self.session_id}] Exploration phase complete, forcing manual wait for direction selection")

        if wait_seconds == 0:
            # No-wait mode: proceed immediately
            logger.info(f"[{self.session_id}] No-wait mode, proceeding immediately (round {analysis_state.current_round})")
        elif wait_seconds == -1:
            # Manual mode: wait for user signal (with supplementary analysis support)
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")

            while True:
                self._continue_event.clear()
                logger.info(f"[{self.session_id}] Manual mode, waiting for user input (round {analysis_state.current_round})")
                await self._continue_event.wait()

                # Check: supplementary analysis requested?
                if self._supplementary_request and self._supplementary_target == "direction":
                    req = self._supplementary_request
                    self._supplementary_request = None
                    self._supplementary_target = None
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "supplementary_running")

                    await self._run_supplementary_analysis(analysis_state, req, "direction")

                    # Regenerate exploration report with supplementary data
                    try:
                        last_round = analysis_state.round_results[-1] if analysis_state.round_results else None
                        if last_round and analysis_state.exploration_report:
                            exploration_report = await self.pi_agent.generate_exploration_report(
                                analysis_state, last_round
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
                elif self._supplementary_request and self._supplementary_target == "round":
                    req = self._supplementary_request
                    self._supplementary_request = None
                    self._supplementary_target = None
                    self._set_status(analysis_state, TaskStatus.RUNNING)
                    await self._send_state_sync(analysis_state, "supplementary_running")

                    await self._run_supplementary_analysis(analysis_state, req, "round")

                    self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                    await self._send_state_sync(analysis_state, "waiting_input")
                    continue
                else:
                    logger.info(f"[{self.session_id}] Continue signal received")
                    break
        else:
            # Countdown mode: backend safety timeout = frontend countdown + 60s buffer
            # Guard: ensure WAITING_INPUT status is synced before blocking
            if analysis_state.status != TaskStatus.WAITING_INPUT:
                self._set_status(analysis_state, TaskStatus.WAITING_INPUT)
                await self._send_state_sync(analysis_state, "waiting_input")
            backend_timeout = wait_seconds + 60
            logger.info(f"[{self.session_id}] Waiting for user input (round {analysis_state.current_round}, timeout={backend_timeout}s)")
            try:
                await asyncio.wait_for(self._continue_event.wait(), timeout=backend_timeout)
                logger.info(f"[{self.session_id}] Continue signal received")
            except asyncio.TimeoutError:
                logger.info(f"[{self.session_id}] Wait timeout ({backend_timeout}s), auto-proceeding")
                self._set_status(analysis_state, TaskStatus.RUNNING)
                await self._send_state_sync(analysis_state, "auto_continue")

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
            multi_dir_completed = (
                not analysis_state.directions_queue
                and len(analysis_state.user_chosen_directions) > 1
            )
            if multi_dir_completed and not has_real_text:
                logger.info(f"[{self.session_id}] Multi-direction completed, not extending max_rounds")
            else:
                analysis_state.max_rounds = analysis_state.current_round + 1
                logger.info(f"[{self.session_id}] User input on last round (text={has_real_text}, files={has_uploaded_files}), extending max_rounds to {analysis_state.max_rounds}")

        return {**state, "analysis_state": analysis_state}

    async def _check_convergence(self, state: WorkflowState) -> WorkflowState:
        """Check if analysis has converged.

        Note: This node is only reached when _should_continue() returns "continue",
        meaning hard-stop conditions (human stop, max_rounds, paused) have already
        been checked. Here we only evaluate soft convergence (repeated findings).
        """
        analysis_state = state["analysis_state"]

        should_stop = analysis_state.should_stop(self.config.convergence_rounds)

        logger.info(f"[{self.session_id}] Convergence check: round={analysis_state.current_round}, should_stop={should_stop}")

        return {
            **state,
            "should_continue": not should_stop
        }
    

