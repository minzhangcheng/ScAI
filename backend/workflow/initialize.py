"""Round initialization logic."""
import asyncio
import logging
from typing import TYPE_CHECKING

from ..constants import PI_DECIDE_SIGNAL
from ..models.schemas import AgentType, AnalysisPhase, BiochemistryIndicator, TaskStatus
from .retry import with_retry

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

logger = logging.getLogger(__name__)


class InitializeMixin:
    """Mixin providing round-initialization method. Composed into SingleCellAnalysisWorkflow."""

    @with_retry("initialize_round")
    async def _initialize_round(self, state: WorkflowState) -> WorkflowState:
        """Initialize a new analysis round, processing any pending user input"""
        analysis_state = state["analysis_state"]
        # Only an explicit Start on an idle/manual restore preserves the round
        # captured by the snapshot. Durable restore origin (`last_restored_*`)
        # is lineage, never round arithmetic: a semantic direction action that
        # routes through check_convergence must initialize the next round.
        preserve_restore = bool(state.get("_preserve_round_on_initialize", False))
        # P1-1: commit the previously peeked direction. If current_round_direction
        # matches the head of the queue, the prior round completed successfully
        # (otherwise LangGraph would have replayed from checkpoint without
        # current_round_direction set) → consume it now.
        if (
            not preserve_restore
            and
            analysis_state.current_round_direction
            and analysis_state.directions_queue
            and analysis_state.directions_queue[0].get("direction") == analysis_state.current_round_direction
        ):
            consumed = analysis_state.directions_queue.pop(0)
            logger.info(
                f"[{self.session_id}] Direction consumed at round entry: "
                f"{consumed.get('direction', '')[:80]} (remaining queue: {len(analysis_state.directions_queue)})"
            )
        if not preserve_restore:
            analysis_state.current_round_direction = ""

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
                    "_preserve_round_on_initialize": False,
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
                                origin="extracted",
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
        
        # Increment the round and reset the previous plan/cache boundary
        # unless this is an explicit Start on an idle/manual restore, whose
        # one-shot flag keeps the captured round in place.
        if not preserve_restore:
            analysis_state.current_round += 1
            # Clear the previous round's approved plan so the new round
            # actually runs planning_dialogue. Without this, _pi_planning's
            # "already approved" early-return short-circuits the next round.
            # Both are cleared together: the next round must plan anew and
            # must not replay a cached result from the round just completed.
            analysis_state.current_plan = None
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

        result = {
            **state,
            "analysis_state": analysis_state,
            # Read-and-consume in the returned channel itself. Popping the
            # local state dict is not a graph channel update: LangGraph merges
            # the node's return, so the explicit False is what consumes the
            # one-shot flag for every later node and invocation.
            "_preserve_round_on_initialize": False,
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
            # An explicit Start restores an already-approved plan; every
            # semantic continuation clears it so the new round plans afresh.
            "plan_approved": (
                bool(
                    analysis_state.current_plan
                    and analysis_state.current_plan.status == "approved"
                )
                if preserve_restore
                else False
            ),
            "plan_revision_needed": False,
            "plan_revision_resume": False,
            # Reset continuation flags for new round
            "should_continue": True,
            "human_stop_requested": False,
            "human_guidance_processed": False,
        }

        # Carry forward the graph-node resume target from the initial workflow
        # state so the routing decision survives the initialize_round node.
        #
        # Note: `_restore_resume_node` is not a declared `WorkflowState` channel,
        # so LangGraph drops it from both the invoke input and every node write.
        # There is no node→branch passthrough to rescue it either: a router whose
        # state parameter is type-annotated -- `_should_continue_after_init` is --
        # receives the declared channels rather than the node's raw return. This
        # re-emit and the branch it feeds are therefore inert; they are kept
        # because making them live would require declaring the channel, and a
        # declared one-shot target must then also be consumed here or every
        # later round would re-route to it.
        #
        # The graph-node jump does not depend on any of that. It is performed by
        # `_route_from_restore_entry` off `_restore_entry_plan`, which *is*
        # declared, for exactly the reason above.
        restore_node = state.get("_restore_resume_node")
        if restore_node is not None:
            result["_restore_resume_node"] = restore_node

        return result
