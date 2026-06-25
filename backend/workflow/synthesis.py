"""PI synthesis, user input wait, and convergence check."""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constants import (
    CONTINUE_SIGNAL,
    LlmFailureAction,
    PI_DECIDE_SIGNAL,
    StateSyncChangeType,
    WaitInputReason,
)
from ..models.schemas import AgentType, AnalysisPhase, AnalysisState, TaskStatus
from ..utils.llm_client import UpstreamExhaustedError
from .helpers import wait_with_skip_time

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


def classify_exploration_failure(bio) -> str:
    """Map the internal bioinformatics result to a STABLE reason code for the UI.

    Never leak raw internal status strings (e.g. 'max_steps_reached',
    'completed_with_fallback') to the frontend — the dialog localizes off this
    code. Returns one of: 'analysis_error', 'did_not_converge'."""
    if isinstance(bio, dict) and bio.get("error"):
        return "analysis_error"
    return "did_not_converge"


def _should_force_next_round(analysis_state: AnalysisState) -> bool:
    """Return True when workflow must honor pending work instead of stopping on convergence.

    Priority order:
    1. Real human text input (not synthetic continue markers)
    2. Newly uploaded files
    3. Remaining multi-direction queue entries
    """
    has_real_text = bool(
        analysis_state.pending_human_input
        and analysis_state.pending_human_input.strip() not in ("", CONTINUE_SIGNAL, PI_DECIDE_SIGNAL)
    )
    has_uploaded_files = bool(analysis_state.pending_uploaded_files)
    has_remaining_directions = bool(
        analysis_state.directions_queue
        and any(d.get("direction", "").strip() for d in analysis_state.directions_queue)
    )
    return has_real_text or has_uploaded_files or has_remaining_directions


def _consume_signal_then_clear(event: asyncio.Event) -> None:
    event.clear()


def _should_run_direction_correlation(analysis_state) -> bool:
    """Whether to run cross-direction correlation analysis.

    Criterion: all chosen directions have completed their innovation rounds,
    measured by 'completed innovation rounds >= number of chosen directions'
    rather than 'directions_queue is empty'. Under P1-1 peek-commit the final
    direction's queue entry is only popped on the next round's init, but the
    terminal round has no next init, so the old empty-queue criterion would
    silently skip this analysis.

    Called after round_results.append(round_result), so the current round
    is already counted.
    """
    chosen = analysis_state.user_chosen_directions
    if len(chosen) <= 1:
        return False
    if analysis_state.analysis_phase != AnalysisPhase.INNOVATION:
        return False
    if analysis_state.multi_direction_correlation is not None:
        return False
    innovation_rounds_done = sum(
        1 for r in analysis_state.round_results if r.phase == "innovation"
    )
    return innovation_rounds_done >= len(chosen)


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


class SynthesisMixin:
    """Mixin providing synthesis, user-input, and convergence methods. Composed into SingleCellAnalysisWorkflow."""

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
        return [
            {
                "direction_id": c.get("direction_id", f"dir_{idx + 1:03d}"),
                "direction": c.get("direction", ""),
                "status": "skipped",
                "search_tier": "failed",
                "literature_novelty": "unknown",
                "confidence": "failed",
                "summary": reason,
                "limitations": [reason],
            }
            for idx, c in enumerate(raw_candidates)
        ]

    async def _generate_evidence_gated_exploration_report(self, graph_state, analysis_state, round_result):
        raw_candidates = await self.pi_agent.generate_raw_direction_candidates(analysis_state, round_result)
        literature_result = (graph_state or {}).get("literature_result") or {}
        if literature_result.get("status") in {"skipped", "disabled"}:
            reason = "Direction-level literature verification was skipped because the Literature Agent was skipped or disabled earlier in the workflow."
            direction_evidence = self._skipped_direction_evidence(raw_candidates, reason)
            report = await self.pi_agent.score_directions_with_evidence(analysis_state, round_result, raw_candidates, direction_evidence)
            self.pi_agent._validate_direction_dependency_graph(getattr(report, "direction_dependency_graph", {}) or {})
            return report

        try:
            direction_evidence = await self.literature_agent.evaluate_direction_novelty_candidates(raw_candidates, state=analysis_state)
        except Exception as lit_err:
            logger.exception(f"[{self.session_id}] Direction literature evidence gate failed")
            direction_evidence = [
                {
                    "direction_id": c.get("direction_id", f"dir_{idx + 1:03d}"),
                    "direction": c.get("direction", ""),
                    "status": "failed",
                    "search_tier": "failed",
                    "literature_novelty": "unknown",
                    "confidence": "failed",
                    "summary": f"Direction-level literature verification failed: {lit_err}",
                    "limitations": ["Literature verification failed; PI scoring must be conservative."],
                }
                for idx, c in enumerate(raw_candidates)
            ]
        report = await self.pi_agent.score_directions_with_evidence(analysis_state, round_result, raw_candidates, direction_evidence)
        self.pi_agent._validate_direction_dependency_graph(getattr(report, "direction_dependency_graph", {}) or {})
        return report

    async def _pi_synthesis(self, state: WorkflowState) -> WorkflowState:
        """PI synthesizes all results and prepares round summary for user"""
        analysis_state = state["analysis_state"]

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

        analysis_state.round_results.append(round_result)
        # Round is now canonical in round_results; clear the per-round cache
        # so a later snapshot-restored round can't replay stale agent outputs.
        analysis_state.current_round_cache = None

        # Multi-direction correlation analysis: trigger once all chosen
        # directions have completed their innovation rounds. NOTE: do NOT gate
        # on `directions_queue` being empty — P1-1 peek-commit leaves the final
        # direction in the queue at the terminal round (no next-round init to
        # pop it), which silently suppressed this analysis. See
        # _should_run_direction_correlation.
        if _should_run_direction_correlation(analysis_state):
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

        Behavior depends on analysis_state.round_wait_seconds (the user-facing
        "skip time" setting, unified across all HITL gates):
          -1  = manual mode: wait indefinitely for user signal
           0  = no-wait mode: proceed immediately without waiting
          >0  = countdown mode: frontend shows countdown timer, backend waits
                `wait_seconds + 60s` then auto-skips (60s buffer for client lag)

        Auto-skip semantics for the post-exploration direction-selection gate:
        equivalent to "let PI decide" (sets user_chosen_direction = [PI_DECIDE]).
        """
        analysis_state = state["analysis_state"]

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
                    "direction_selection" if is_direction_gate else "round_wait",
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
            innovation_rounds_done = sum(
                1 for r in analysis_state.round_results if r.phase == "innovation"
            )
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
    

