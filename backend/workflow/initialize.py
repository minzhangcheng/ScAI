"""Round initialization logic."""
import asyncio
import logging
from typing import TYPE_CHECKING

from ..constants import PI_DECIDE_SIGNAL
from ..models.schemas import AgentType, AnalysisPhase, BiochemistryIndicator, TaskStatus

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


class InitializeMixin:
    """Mixin providing round-initialization method. Composed into SingleCellAnalysisWorkflow."""

    async def _initialize_round(self, state: WorkflowState) -> WorkflowState:
        """Initialize a new analysis round, processing any pending user input"""
        analysis_state = state["analysis_state"]
        # P1-1: commit the previously peeked direction. If current_round_direction
        # matches the head of the queue, the prior round completed successfully
        # (otherwise LangGraph would have replayed from checkpoint without
        # current_round_direction set) → consume it now.
        if (
            analysis_state.current_round_direction
            and analysis_state.directions_queue
            and analysis_state.directions_queue[0].get("direction") == analysis_state.current_round_direction
        ):
            consumed = analysis_state.directions_queue.pop(0)
            logger.info(
                f"[{self.session_id}] Direction consumed at round entry: "
                f"{consumed.get('direction', '')[:80]} (remaining queue: {len(analysis_state.directions_queue)})"
            )
        analysis_state.current_round_direction = None

        # P3-5: business default — if resuming in INNOVATION after round 1 with
        # no direction chosen and no pending input, default to PI_DECIDE so the
        # workflow doesn't run directionless. This belongs in workflow (not the
        # resume route) because it's a workflow-state invariant, not a transport
        # concern. The route just signals "resume"; defaults live here.
        if (
            analysis_state.analysis_phase == AnalysisPhase.INNOVATION
            and analysis_state.current_round >= 1
            and not analysis_state.user_chosen_direction
            and not analysis_state.directions_queue
            and not analysis_state.pending_human_input
            and not analysis_state.user_chosen_directions
        ):
            analysis_state.user_chosen_direction = PI_DECIDE_SIGNAL
            analysis_state.pending_human_input = PI_DECIDE_SIGNAL
            logger.info(f"[{self.session_id}] Round init without direction: defaulting to PI_DECIDE")
            effective_max = analysis_state.max_rounds if analysis_state.max_rounds > 0 else 2
            if analysis_state.current_round >= effective_max:
                analysis_state.max_rounds = analysis_state.current_round + 1
                logger.info(
                    f"[{self.session_id}] Extended max_rounds to {analysis_state.max_rounds} for PI_DECIDE default"
                )

        # Update current analysis state for callbacks (e.g., literature_callback)
        self._current_analysis_state = analysis_state
        self._current_graph_state = state
        logger.info(f"[{self.session_id}] Initializing round {analysis_state.current_round + 1}")
        
        # Process pending human input (if any)
        human_guidance = None
        user_action = None
        new_data_files = []
        adjusted_instructions = None
        
        if analysis_state.pending_human_input or analysis_state.pending_uploaded_files:
            human_message = analysis_state.pending_human_input or ""
            uploaded_files = analysis_state.pending_uploaded_files or []
            
            await self._send_message(
                "human_input_received",
                f"Processing user input: {human_message[:100]}{'...' if len(human_message) > 100 else ''}" + 
                (f" ({len(uploaded_files)} files)" if uploaded_files else ""),
                AgentType.PI
            )
            
            # Let PI Agent process the user input
            pi_response = await self.pi_agent.process_user_input(
                analysis_state,
                human_message,
                uploaded_files
            )
            
            user_action = pi_response.get("action", "continue")
            human_guidance = human_message
            
            await self._send_message(
                "response",
                pi_response.get("response", "Acknowledged"),
                AgentType.PI
            )
            
            # Check if user requested to stop
            if user_action == "stop":
                await self._send_message(
                    "status",
                    "User requested to conclude analysis. Generating final report...",
                    AgentType.PI
                )
                # Clear pending input
                analysis_state.pending_human_input = None
                analysis_state.pending_uploaded_files = []
                
                return {
                    **state,
                    "analysis_state": analysis_state,
                    "should_continue": False,
                    "human_stop_requested": True
                }
            
            # Process file assignments from PI
            file_assignments = pi_response.get("file_assignments", {})

            # Track new data files for prioritized analysis
            if file_assignments.get("pathology"):
                for file_info in file_assignments["pathology"]:
                    if file_info:
                        new_data_files.append({
                            "section": "pathology",
                            "filename": file_info.get("filename"),
                            "data_type": file_info.get("image_type", "user_upload"),
                            "reason": file_info.get("reason", "")
                        })

            # Extract and add biochemistry values from user input
            biochemistry_values = pi_response.get("biochemistry_values", {})
            if biochemistry_values and isinstance(biochemistry_values, dict):
                for indicator, data in biochemistry_values.items():
                    if isinstance(data, dict) and "value" in data:
                        try:
                            value = float(data["value"])
                            result = BiochemistryIndicator(
                                indicator=indicator,
                                value=value,
                                unit=data.get("unit", ""),
                                interpretation=data.get("interpretation")
                            )
                            analysis_state.biochemistry_indicators.append(result)
                            logger.info(f"[{self.session_id}] Added biochemistry value: {indicator}={value} {data.get('unit', '')}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"[{self.session_id}] Failed to parse biochemistry value for {indicator}: {e}")

            # Track biochemistry data in new_data_files for agent activation logic
            if biochemistry_values:
                new_data_files.append({
                    "section": "biochemistry",
                    "filename": None,
                    "data_type": "indicator_values",
                    "reason": "User-provided biochemistry indicators"
                })

            # Track biochemistry files from file_assignments (reuse from line 1793)
            if file_assignments.get("biochemistry"):
                for file_info in file_assignments["biochemistry"]:
                    if file_info:
                        new_data_files.append({
                            "section": "biochemistry",
                            "filename": file_info.get("filename"),
                            "data_type": file_info.get("image_type", "user_upload"),
                            "reason": file_info.get("reason", "")
                        })

            # Get adjusted instructions
            adjusted_instructions = pi_response.get("adjusted_instructions")
            
            # Clear pending input
            analysis_state.pending_human_input = None
            analysis_state.pending_uploaded_files = []
        
        # Increment round counter (skip if resuming from snapshot restore,
        # since current_round is already correct from the snapshot)
        if not analysis_state.last_restored_snapshot_id:
            analysis_state.current_round += 1
            # Clear the previous round's approved plan so the new round
            # actually runs planning_dialogue. Without this, _pi_planning's
            # "already approved" early-return short-circuits the next round.
            # Skip on snapshot restore — the restored plan must be preserved.
            analysis_state.current_plan = None
            # Snapshot-restore path preserves it by skipping this whole block.
            analysis_state.current_round_cache = None
        self._set_status(analysis_state, TaskStatus.RUNNING)

        # Clear residual should_continue from previous convergence check
        # to prevent stale False value from terminating an extended round
        state["should_continue"] = True

        await self._send_message(
            "status",
            f"Starting round {analysis_state.current_round} analysis"
        )

        # Send state sync for round initialization
        await self._send_state_sync(analysis_state, "round_start")

        return {
            **state,
            "analysis_state": analysis_state,
            "current_round": analysis_state.current_round,
            "human_guidance": human_guidance,
            "user_action": user_action,
            "new_data_files": new_data_files,
            "adjusted_instructions": adjusted_instructions,
            "pathology_result": None,
            "bioinformatics_result": None,
            "biochemistry_result": None,
            "literature_result": None,
            "round_result": None,
            # Clear retry-related state for new round
            "node_retry_count": {},
            "node_errors": {},
            "failed_node": None,
            # Reset skip_nodes so stale values from previous round do not carry over
            "skip_nodes": [],
            # Reset planning_context so stale stage1 data from previous round doesn't leak
            "planning_context": None,
            # Reset plan approval so the new round runs planning_dialogue;
            # paired with clearing analysis_state.current_plan above.
            # On snapshot restore, current_plan is preserved but plan_approved
            # was unconditionally reset — causing planning_dialogue to re-run
            # for an already-approved plan.  Guard: if current_plan exists and
            # is approved on snapshot restore, preserve plan_approved = True.
            "plan_approved": (
                bool(analysis_state.last_restored_snapshot_id
                     and analysis_state.current_plan
                     and analysis_state.current_plan.status == "approved")
                if analysis_state.last_restored_snapshot_id
                else False
            ),
            "plan_revision_needed": False,
            "plan_revision_resume": False,
            # Reset continuation flags for new round
            "should_continue": True,
            "human_stop_requested": False,
            "human_guidance_processed": False
        }

