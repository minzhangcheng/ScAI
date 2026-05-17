"""Planning dialogue: PI ↔ Bioinformatics plan design, review, and user gate."""
import asyncio
import json
import logging
from datetime import datetime
from html import escape as _html_escape
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constants import LlmFailureAction
from ..models.schemas import AgentType, AnalysisPhase, TaskStatus
from ..utils.llm_client import UpstreamExhaustedError
from .helpers import _diff_plan_steps, _get_task_text, _normalize_tasks, _DIFF_LABELS
from .retry import MAX_NODE_RETRIES, _retry_backoff_delay, with_retry

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict  # runtime fallback for type annotations

logger = logging.getLogger(__name__)

MAX_PLANNING_ROUNDS = 5


class PlanningMixin:
    """Mixin providing planning-dialogue methods. Composed into SingleCellAnalysisWorkflow."""
    @with_retry("pi_planning")
    async def _pi_planning(self, state: WorkflowState) -> WorkflowState:
        """PI agent plans the round, incorporating human guidance and adjusted instructions"""
        analysis_state = state["analysis_state"]
        human_guidance = state.get("human_guidance")
        adjusted_instructions = state.get("adjusted_instructions")

        # If a plan is already approved (e.g., from no-workflow fallback resume),
        # preserve the approval and skip re-planning so downstream routing
        # (_should_plan, _after_bioinformatics) can act on it.
        if analysis_state.current_plan and analysis_state.current_plan.status == "approved":
            logger.info(
                f"[{self.session_id}] Plan already approved "
                f"(status={analysis_state.current_plan.status}), skipping re-planning"
            )
            state["plan_approved"] = True
            state["planning_result"] = {
                "plan": analysis_state.current_plan.model_dump(mode="json"),
                "dialogue": analysis_state.plan_dialogue_history or [],
            }
            return state

        # Reset plan-related state for new round
        analysis_state.current_plan = None
        state["plan_approved"] = False
        state["plan_revision_needed"] = False
        # Do NOT reset plan_revision_resume here — it's set by _user_review_gate
        # and must survive until _after_bioinformatics routing reads it
        if not state.get("plan_revision_resume"):
            state["plan_revision_resume"] = False
        state["planning_result"] = None
        state["user_plan_feedback"] = None

        # Reset PI message history at start of each round's planning phase.
        # Dynamic planning calls (plan_agent_tasks) will accumulate context
        # within this round via self.chat(), building a coherent chain of thought.
        self.pi_agent.reset_history()

        logger.info(f"[{self.session_id}] PI planning for round {analysis_state.current_round}")
        if human_guidance:
            logger.info(f"[{self.session_id}] Human guidance received: {human_guidance[:100]}...")

        await self._send_message("status", "PI is creating analysis plan...", AgentType.PI)
        
        # If we already have adjusted instructions from user input processing, use them
        # BUT: in innovation phase (round 2+), don't short-circuit direction selection —
        # treat adjusted_instructions as supplementary guidance alongside the direction
        if adjusted_instructions and analysis_state.current_round <= 1:
            await self._send_message(
                "message",
                f"Using adjusted instructions from user input:\n{json.dumps(adjusted_instructions, ensure_ascii=False, indent=2)}",
                AgentType.PI
            )
            return {
                **state,
                "pi_instructions": adjusted_instructions,
                "skip_nodes": [],
                "human_guidance_processed": True
            }
        
        # Normal planning without pre-processed instructions
        skip_nodes = []  # Empty for round 1; populated for round 2+
        if analysis_state.current_round == 1:
            # Round 1: Exploration phase — unbiased comprehensive exploration
            instructions = await self.pi_agent.initial_exploration_instruction(
                analysis_state,
                analysis_state.initial_instruction
            )
        else:
            # Round 2+: Innovation phase — hypothesis-driven targeted analysis
            if analysis_state.analysis_phase != "innovation":
                analysis_state.analysis_phase = AnalysisPhase.INNOVATION

            # Check directions_queue first (multi-direction mode), then single direction
            user_direction = analysis_state.user_chosen_direction
            if analysis_state.directions_queue:
                # Multi-direction mode: peek first, pop only on success
                current_dir = analysis_state.directions_queue[0]
                dir_text = current_dir.get("direction", "")
                dir_notes = current_dir.get("notes", "")
                logger.info(f"[{self.session_id}] Innovation phase: direction from queue → {dir_text[:100]} (remaining: {len(analysis_state.directions_queue) - 1})")
                # Set current direction immediately so retry has correct context
                analysis_state.current_round_direction = dir_text
                try:
                    instructions = await self.pi_agent.innovation_instruction(
                        analysis_state, dir_text, notes=dir_notes
                    )
                except Exception as e:
                    # Direction stays in queue (never popped) — no re-insert needed
                    analysis_state.current_round_direction = None
                    logger.exception(f"[{self.session_id}] innovation_instruction failed, direction stays in queue")
                    raise
                # Pop only after successful instruction generation
                analysis_state.directions_queue.pop(0)
                # Persist state after instructions are successfully generated
                if self.state_callback:
                    try:
                        await self.state_callback(analysis_state)
                    except Exception as e:
                        logger.exception(f"[{self.session_id}] Failed to persist state after direction pop")
                # Clear single direction field
                analysis_state.user_chosen_direction = None
            elif user_direction and user_direction not in ("", "[PI_DECIDE]"):
                logger.info(f"[{self.session_id}] Innovation phase: user-directed → {user_direction[:100]}")
                instructions = await self.pi_agent.innovation_instruction(
                    analysis_state, user_direction
                )
                # Preserve direction for bio agent phase_instruction and synthesis prompt
                analysis_state.current_round_direction = user_direction
                # Consume the direction to avoid reusing in next round
                analysis_state.user_chosen_direction = None
            elif user_direction == "[PI_DECIDE]" or (not user_direction and analysis_state.current_round == 2):
                # PI autonomously selects 2 best directions
                logger.info(f"[{self.session_id}] Innovation phase: PI selecting best directions")
                selected_dirs = await self.pi_agent.select_best_directions(analysis_state)
                # Set up multi-direction queue
                analysis_state.user_chosen_directions = selected_dirs
                # Pop first direction, rest goes to queue
                first_dir = selected_dirs[0] if selected_dirs else {"direction": "Continue analysis", "notes": ""}
                analysis_state.directions_queue = selected_dirs[1:] if len(selected_dirs) > 1 else []
                analysis_state.max_rounds = len(selected_dirs) + 1  # Round 1 + N direction rounds
                dir_text = first_dir.get("direction", "")
                dir_notes = first_dir.get("notes", "")
                instructions = await self.pi_agent.innovation_instruction(
                    analysis_state, dir_text, notes=dir_notes
                )
                analysis_state.current_round_direction = dir_text
                analysis_state.user_chosen_direction = None
            else:
                # Subsequent innovation rounds (3+): use previous round's next_steps
                last_round = analysis_state.get_latest_round()
                instructions = {
                    "pathology_tasks": [],
                    "bioinformatics_tasks": [],
                    "biochemistry_tasks": [],
                    "literature_topics": []
                }

                # Extract tasks for each agent
                prefix_map = {
                    "[Pathology]": "pathology_tasks",
                    "[Bioinformatics]": "bioinformatics_tasks",
                    "[Biochemistry]": "biochemistry_tasks",
                    "[Literature]": "literature_topics",
                }
                for step in (last_round.next_steps if last_round else []):
                    for prefix, key in prefix_map.items():
                        if step.startswith(prefix):
                            instructions[key].append(step.split("] ", 1)[1].strip() if "] " in step else step[len(prefix):].strip())
                            break

                # Add overview for round 3+ so PI dynamic planning has strategic context
                if last_round:
                    summary = last_round.pi_summary or "Continue analysis based on previous round findings."
                    instructions["overview"] = summary[:500]

            last_round = analysis_state.get_latest_round()

            # Fallback: if prefix matching extracted no tasks at all
            all_empty = not any([
                instructions["pathology_tasks"],
                instructions["bioinformatics_tasks"],
                instructions["biochemistry_tasks"],
                instructions["literature_topics"]
            ])
            if all_empty and last_round:
                if last_round.next_steps:
                    # First fallback: assign unprefixed next_steps to bioinformatics
                    instructions["bioinformatics_tasks"] = last_round.next_steps
                    logger.warning(
                        f"[{self.session_id}] No prefixed tasks found in next_steps, "
                        f"falling back to bioinformatics-only with raw next_steps"
                    )
                elif last_round.pi_summary:
                    # Second fallback: use PI summary
                    fallback_context = last_round.pi_summary[:500]
                    fallback = f"Continue analysis based on previous round findings. Context: {fallback_context}"
                    instructions["bioinformatics_tasks"] = [fallback]
                    instructions["literature_topics"] = [f"Literature search related to: {fallback_context[:200]}"]
                    logger.warning(
                        f"[{self.session_id}] No next_steps available, "
                        f"using PI summary as fallback for bioinformatics and literature"
                    )

            # Build skip_nodes from PI's agent activation decision
            agents_to_activate = set(last_round.agents_for_next_round) if last_round else set()
            reanalysis_agents = set(last_round.reanalysis_agents) if last_round else set()
            agents_to_activate.add("bioinformatics")  # Always activate bioinformatics

            # Default disable logic (from round 2 onwards): skip pathology/biochemistry if no new data
            if analysis_state.current_round >= 2:
                new_pathology_files = [f for f in state.get("new_data_files", []) if f.get("section") == "pathology"]
                new_biochemistry_files = [f for f in state.get("new_data_files", []) if f.get("section") == "biochemistry"]

                # Pathology: default disabled unless new data or PI explicitly requests reanalysis
                if not new_pathology_files and "pathology" not in reanalysis_agents:
                    agents_to_activate.discard("pathology")
                    logger.info(f"[{self.session_id}] Default disabling pathology: no new data and no reanalysis request")

                # Biochemistry: default disabled unless new data or PI explicitly requests reanalysis
                if not new_biochemistry_files and "biochemistry" not in reanalysis_agents:
                    agents_to_activate.discard("biochemistry")
                    logger.info(f"[{self.session_id}] Default disabling biochemistry: no new data and no reanalysis request")

            # Reconcile: if prefix parsing found tasks for an agent, ensure it's activated
            task_to_agent = {
                "pathology_tasks": "pathology",
                "bioinformatics_tasks": "bioinformatics",
                "biochemistry_tasks": "biochemistry",
                "literature_topics": "literature",
            }
            for task_key, agent_key in task_to_agent.items():
                if instructions.get(task_key) and agent_key not in agents_to_activate:
                    agents_to_activate.add(agent_key)
                    logger.info(
                        f"[{self.session_id}] Auto-activated {agent_key}: "
                        f"has tasks but was not in agents_for_next_round"
                    )

            skip_nodes = []
            agent_node_map = {
                "pathology": "pathology_analysis",
                "biochemistry": "biochemistry_analysis",
                "literature": "literature_research",
            }
            for agent_key, node_name in agent_node_map.items():
                if agent_key not in agents_to_activate:
                    skip_nodes.append(node_name)

            if skip_nodes:
                activated = [k for k in agent_node_map if k in agents_to_activate]
                logger.info(
                    f"[{self.session_id}] Round {analysis_state.current_round}: "
                    f"PI activated agents: bioinformatics + {activated}, "
                    f"skipping: {skip_nodes}"
                )

        # In innovation phase, merge adjusted_instructions as supplementary human guidance
        # (not replacing direction-based planning, but augmenting it)
        if adjusted_instructions and analysis_state.current_round > 1:
            for key in ("bioinformatics_tasks", "pathology_tasks", "biochemistry_tasks", "literature_topics"):
                extra = adjusted_instructions.get(key, [])
                if extra:
                    existing = instructions.get(key, [])
                    instructions[key] = _normalize_tasks(existing) + [f"[User guidance] {t}" for t in _normalize_tasks(extra)]
            logger.info(f"[{self.session_id}] Merged adjusted_instructions into innovation planning")

        await self._send_message(
            "message",
            json.dumps(instructions, ensure_ascii=False, indent=2),
            AgentType.PI
        )

        # Determine planning stage based on round and phase
        if analysis_state.analysis_phase == AnalysisPhase.EXPLORATION:
            state["planning_stage"] = "stage1"
        else:
            state["planning_stage"] = "full"

        return {
            **state,
            "pi_instructions": instructions,
            "skip_nodes": skip_nodes,
            "human_guidance_processed": True,
        }
    
    async def _planning_dialogue(self, state: WorkflowState) -> WorkflowState:
        """PI ↔ Bioinformatics dialogue to design an analysis plan (max 5 rounds)."""
        analysis_state = state["analysis_state"]
        pi_instructions = state.get("pi_instructions", {})
        bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
        overview = pi_instructions.get("overview", "")
        stage = state.get("planning_stage", "full")  # "full", "stage1", "stage2"
        existing_plan = analysis_state.current_plan  # may have a partial plan from revision
        if existing_plan and existing_plan.stage and existing_plan.stage != stage and stage != "full":
            existing_plan = None

        phase = analysis_state.analysis_phase.value
        combined_previous = self._build_planning_context(state)
        dialogue_history = []

        round_num = 1
        while round_num <= MAX_PLANNING_ROUNDS:
            await self._send_message(
                "status",
                f"Planning dialogue: round {round_num}/{MAX_PLANNING_ROUNDS}",
                AgentType.PI,
            )

            # --- Bioinformatics agent call (with LLM failure intervention) ---
            while True:
                try:
                    if round_num == 1 and not existing_plan:
                        plan = await self.bioinformatics_agent.generate_analysis_plan(
                            state=analysis_state,
                            pi_tasks=bioinf_tasks,
                            overview=overview,
                            phase=phase,
                            planning_stage=stage,
                            previous_results=combined_previous,
                        )
                    elif existing_plan and existing_plan.status == "rejected":
                        plan = await self.bioinformatics_agent.generate_analysis_plan(
                            state=analysis_state,
                            pi_tasks=bioinf_tasks,
                            overview=overview,
                            phase=phase,
                            planning_stage=stage,
                            previous_results=combined_previous,
                            existing_plan=existing_plan,
                        )
                    else:
                        # Collect PI feedback from all dialogue rounds, not just the last one
                        all_pi_feedback = "\n".join(
                            f"Round {d['round']}: {d['pi_feedback']}"
                            for d in dialogue_history
                            if d.get("pi_feedback")
                        )
                        user_feedback = state.get("user_plan_feedback", "")
                        pi_feedback = all_pi_feedback or (dialogue_history[-1]["pi_feedback"] if dialogue_history else "")
                        combined_feedback = f"{pi_feedback}\n\nUser feedback: {user_feedback}" if user_feedback else pi_feedback
                        plan = await self.bioinformatics_agent.revise_plan_incrementally(
                            state=analysis_state,
                            current_plan=existing_plan,
                            pi_feedback=combined_feedback,
                        )
                    break  # success, exit inner retry loop
                except UpstreamExhaustedError as e:
                    action = await self._wait_for_llm_intervention(
                        e, "planning_dialogue", analysis_state
                    )
                    if action == LlmFailureAction.RETRY:
                        continue  # retry same bioinfo call
                    # SKIP
                    await self._send_message(
                        "error",
                        "Planning dialogue: bioinformatics agent skipped due to LLM failure.",
                        AgentType.PI,
                    )
                    if existing_plan:
                        # Don't reinstate a PI-rejected plan as pi_reviewing
                        if existing_plan.status != "rejected":
                            existing_plan.status = "pi_reviewing"
                        analysis_state.current_plan = existing_plan
                        preserved_history = analysis_state.plan_dialogue_history or dialogue_history
                        analysis_state.plan_dialogue_history = preserved_history
                        state["planning_result"] = {
                            "plan": existing_plan.model_dump(mode="json"),
                            "dialogue": preserved_history,
                        }
                    else:
                        state["failed_node"] = "planning_dialogue"
                        state["planning_result"] = {
                            "error": str(e),
                            "findings": [],
                            "status": "failed",
                        }
                    return state
                # Bioinfo plan generation: no retry on generic Exception.
                # Unlike PI review below (which retries), bioinfo failures
                # are typically deterministic — retrying won't help.
                except Exception as e:
                    logger.exception(
                        f"[{self.session_id}] Bioinformatics plan generation failed "
                        f"in planning dialogue round {round_num}"
                    )
                    await self._send_message(
                        "error",
                        f"Bioinformatics plan generation failed: {str(e)}",
                        AgentType.PI,
                    )
                    state["failed_node"] = "planning_dialogue"
                    state["planning_result"] = {
                        "error": str(e),
                        "findings": [],
                        "status": "failed",
                    }
                    return state

            # --- PI review call (with LLM failure intervention) ---
            pi_review_attempt = 0
            while True:
                try:
                    review = await self.pi_agent.review_bioinfo_plan(
                        state=analysis_state,
                        plan=plan,
                        dialogue_round=round_num,
                    )
                    break  # success, exit inner retry loop
                except UpstreamExhaustedError as e:
                    action = await self._wait_for_llm_intervention(
                        e, "planning_dialogue", analysis_state
                    )
                    if action == LlmFailureAction.RETRY:
                        continue  # retry same PI review call
                    # SKIP — plan exists but unreviewed; auto-accept it
                    await self._send_message(
                        "error",
                        "Planning dialogue: PI review skipped due to LLM failure, auto-accepting plan.",
                        AgentType.PI,
                    )
                    plan.status = "pi_reviewing"
                    analysis_state.current_plan = plan
                    plan_json = plan.model_dump(mode="json")
                    analysis_state.plan_dialogue_history = dialogue_history + [{
                        "round": round_num,
                        "plan": plan_json,
                        "pi_decision": "approved",
                        "pi_feedback": "PI review skipped due to LLM failure — auto-accepted",
                        "pi_modifications": [],
                    }]
                    state["planning_result"] = {
                        "plan": plan_json,
                        "dialogue": analysis_state.plan_dialogue_history,
                    }
                    return state
                except Exception as e:
                    pi_review_attempt += 1
                    if pi_review_attempt <= MAX_NODE_RETRIES:
                        delay = _retry_backoff_delay(pi_review_attempt)
                        logger.warning(
                            f"[{self.session_id}] PI review failed in round {round_num} "
                            f"(attempt {pi_review_attempt}/{MAX_NODE_RETRIES + 1}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        await self._send_message(
                            "status",
                            f"PI review failed, retrying ({pi_review_attempt}/{MAX_NODE_RETRIES + 1})...",
                            AgentType.PI,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # All retries exhausted — return failed (matches @with_retry)
                    logger.exception(
                        f"[{self.session_id}] PI review failed after "
                        f"{MAX_NODE_RETRIES + 1} attempts in round {round_num}"
                    )
                    await self._send_message(
                        "error",
                        f"PI review failed after {MAX_NODE_RETRIES + 1} attempts: {str(e)}",
                        AgentType.PI,
                    )
                    state["failed_node"] = "planning_dialogue"
                    state["planning_result"] = {
                        "error": str(e),
                        "findings": [],
                        "status": "failed",
                    }
                    return state

            # --- Process PI review decision ---
            dialogue_history.append({
                "round": round_num,
                "plan": plan.model_dump(mode="json"),
                "pi_decision": review["decision"],
                "pi_feedback": review["feedback"],
                "pi_modifications": review.get("modifications", []),
            })

            if review["decision"] == "approved":
                plan.status = "pi_reviewing"
                analysis_state.current_plan = plan
                analysis_state.plan_dialogue_history = dialogue_history
                state["planning_result"] = {"plan": dialogue_history[-1]["plan"], "dialogue": dialogue_history}
                await self._send_message("status", "Plan approved by PI", AgentType.PI)
                break
            elif review["decision"] == "reject":
                plan.status = "rejected"
                existing_plan = plan
                await self._send_message(
                    "status",
                    f"Plan rejected by PI, redesigning... ({review['feedback'][:200]})",
                    AgentType.PI,
                )
            else:  # revise
                existing_plan = plan
                existing_plan.status = "draft"
                await self._send_message(
                    "status",
                    f"Plan needs revision (round {round_num}): {review['feedback'][:200]}",
                    AgentType.PI,
                )

            round_num += 1

        # Max rounds reached without approval — take last plan as-is
        if round_num > MAX_PLANNING_ROUNDS:
            logger.warning(
                f"[{self.session_id}] Planning dialogue reached max rounds ({MAX_PLANNING_ROUNDS})"
            )
            if existing_plan:
                existing_plan.status = "pi_reviewing"
                analysis_state.current_plan = existing_plan
                analysis_state.plan_dialogue_history = dialogue_history
                state["planning_result"] = {
                    "plan": existing_plan.model_dump(mode="json"),
                    "dialogue": dialogue_history,
                }
            await self._send_message(
                "status",
                "Planning dialogue: max rounds reached, proceeding with last version",
                AgentType.PI,
            )

        return state

    async def _approve_plan(
        self, state: WorkflowState, plan, analysis_state, action: str
    ) -> None:
        """Mark plan as approved and notify frontend."""
        plan.status = "approved"
        plan.approved_at = datetime.now()
        analysis_state.current_plan = plan
        self._set_status(analysis_state, TaskStatus.RUNNING)
        await self._send_state_sync(analysis_state, "plan_approved")
        await self._send_message("plan_approved", {
            "plan_id": plan.id,
            "action": action,
        }, AgentType.PI)
        state["plan_approved"] = True
        if state.get("plan_revision_needed"):
            state["plan_revision_resume"] = True
            state["plan_revision_needed"] = False

    async def _user_review_gate(self, state: WorkflowState) -> WorkflowState:
        """Present the analysis plan to the user for review and approval.

        Blocks until the user responds (approve/skip/modify).
        For modification requests, loops back to planning_dialogue.
        """
        analysis_state = state["analysis_state"]
        plan = analysis_state.current_plan

        if not plan:
            logger.warning(f"[{self.session_id}] No plan to review, skipping gate")
            return state

        # Main review loop — continues until user approves or skips
        # Generate dialogue HTML once — it doesn't change during the review loop
        dialogue_html = self._build_plan_dialogue_html(
            analysis_state.plan_dialogue_history
        )
        while True:
            # Set plan status and notify frontend
            plan.status = "user_reviewing"

            # Send plan to frontend
            await self._send_message("plan_review", {
                "plan": plan.model_dump(mode="json"),
                "dialogue_html": dialogue_html,
                "phase": analysis_state.analysis_phase.value,
                "round": analysis_state.current_round,
                "stage": state.get("planning_stage", "full"),
            }, AgentType.PI)

            # Set waiting status and sync
            self._set_status(analysis_state, TaskStatus.WAITING_PLAN_REVIEW)
            await self._send_state_sync(analysis_state, "plan_review")

            # Block and wait for user (with 1-hour timeout to prevent permanent hang)
            # Save and reset action before clearing event to close race window
            # where signal_plan_review arrives between clarify continue and event.clear()
            #
            # TOCTOU note: The clear() → None → wait() sequence (lines below) is safe
            # in asyncio because lines 589-591 are synchronous (no `await`), so no
            # other coroutine can interleave between event.clear() and event.wait().
            # The only yield point is the `await` in wait_for() on the line after.
            prev_action = self._plan_review_action
            self._plan_review_event.clear()
            self._plan_review_action = None
            if not prev_action:
                logger.info(f"[{self.session_id}] Waiting for user plan review...")
                try:
                    await asyncio.wait_for(self._plan_review_event.wait(), timeout=3600)
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.session_id}] Plan review timed out, auto-skipping")
                    self._plan_review_action = {"action": "skip", "user_input": ""}
                action = self._plan_review_action or {}
            else:
                logger.info(f"[{self.session_id}] Plan review action already received, skipping wait")
                action = prev_action
            action_type = action.get("action", "skip")
            user_input = action.get("user_input", "")

            if action_type == "approve":
                await self._approve_plan(state, plan, analysis_state, "approved")
                break

            elif action_type == "skip":
                await self._approve_plan(state, plan, analysis_state, "skipped")
                break

            elif action_type == "modify" and user_input:
                # User wants modifications → PI judges intent
                pi_judgment = await self.pi_agent.process_user_plan_feedback(
                    user_input=user_input,
                    current_plan=plan,
                )

                if pi_judgment["action"] == "proceed_to_execution":
                    await self._approve_plan(state, plan, analysis_state, "approved_after_clarification")
                    break

                elif pi_judgment["action"] == "clarify_with_user":
                    # PI needs to clarify — notify user and wait again
                    await self._send_message("status", 
                        pi_judgment.get("clarification_question", "Could you clarify?"),
                        AgentType.PI)
                    continue  # Re-show the review gate

                else:  # revise_with_bioinfo
                    # PI will incorporate user feedback into plan revision
                    guidance = pi_judgment.get("guidance_for_bioinfo", user_input)
                    state["user_plan_feedback"] = guidance
                    state["plan_approved"] = False
                    if state.get("plan_revision_needed"):
                        state["plan_revision_resume"] = True
                        state["plan_revision_needed"] = False
                    break

        return state

    def _build_plan_dialogue_html(self, dialogue_history: List[Dict]) -> str:
        """Build an HTML document from the PI↔Bioinfo planning dialogue.

        Includes step-level diff between consecutive rounds to highlight changes.
        """
        html_parts = [
            "<html><head><meta charset='utf-8'><title>Analysis Plan Dialogue</title>",
            "<style>body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}",
            ".round{border:1px solid #ccc;margin:10px 0;padding:15px;border-radius:8px;}",
            ".approved{background:#e8f5e9}.revise{background:#fff3e0}.reject{background:#ffebee}",
            ".step{margin:5px 0;padding:8px;border-radius:4px;}",
            ".step-added{background:#e8f5e9;border-left:4px solid #4caf50}",
            ".step-removed{background:#ffebee;border-left:4px solid #f44336;text-decoration:line-through}",
            ".step-modified{background:#fff3e0;border-left:4px solid #ff9800}",
            ".step-unchanged{background:#f5f5f5}",
            "</style></head><body><h1>Analysis Plan Dialogue</h1>",
        ]

        prev_steps = None
        for entry in dialogue_history:
            round_num = entry["round"]
            decision = _html_escape(entry.get("pi_decision", ""))
            css_class = {"approved": "approved", "revise": "revise", "reject": "reject"}.get(decision, "")

            html_parts.append(f"<div class='round {css_class}'>")
            html_parts.append(f"<h2>Round {round_num} — PI Decision: {decision.upper()}</h2>")
            html_parts.append(f"<p><strong>PI Feedback:</strong> {_html_escape(entry.get('pi_feedback', ''))}</p>")

            plan = entry.get("plan", {})
            curr_steps = plan.get("steps", [])
            step_diffs = _diff_plan_steps(prev_steps, curr_steps) if prev_steps else {}

            if prev_steps is not None and curr_steps:
                has_changes = any(d != 'unchanged' for d in step_diffs.values())
                if has_changes:
                    html_parts.append("<h3>Changes from Previous Round:</h3>")
                    changes = []
                    for sid, status in step_diffs.items():
                        if status == 'added':
                            step_data = next((s for s in curr_steps if s.get('id') == sid), {})
                            changes.append(f"<span style='color:#4caf50'>+ Added Step {sid}: {_html_escape(step_data.get('title',''))}</span>")
                        elif status == 'removed':
                            step_data = next((s for s in prev_steps if s.get('id') == sid), {})
                            changes.append(f"<span style='color:#f44336'>- Removed Step {sid}: {_html_escape(step_data.get('title',''))}</span>")
                        elif status == 'modified':
                            changes.append(f"<span style='color:#ff9800'>~ Modified Step {sid}</span>")
                    html_parts.append("<br>".join(changes))

            if curr_steps:
                html_parts.append("<h3>Plan Steps:</h3>")
                for step in curr_steps:
                    sid = step.get('id', '')
                    diff_status = step_diffs.get(sid, 'unchanged')
                    css = f"step-{diff_status}"
                    label = _DIFF_LABELS.get(diff_status, '')
                    html_parts.append(
                        f"<div class='step {css}'><strong>Step {_html_escape(str(sid))}: {_html_escape(step.get('title', ''))}</strong> {label}<br>"
                        f"Method: {_html_escape(step.get('method', ''))}<br>"
                        f"Rationale: {_html_escape(step.get('rationale', ''))}</div>"
                    )

            html_parts.append("</div>")
            prev_steps = curr_steps

        html_parts.append("</body></html>")
        return "\n".join(html_parts)
