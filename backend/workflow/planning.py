"""Planning dialogue: PI ↔ Bioinformatics plan design, review, and user gate."""
import asyncio
import difflib
import json
import logging
import re
from datetime import datetime
from html import escape as _html_escape
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constants import LlmFailureAction, PI_DECIDE_SIGNAL, WaitInputReason, StateSyncChangeType
from ..models.schemas import AgentType, AnalysisPhase, AnalysisPlan, AnalysisState, PlanStep, TaskStatus
from ..services.interaction_record_capture import (
    find_open_integration_record,
    find_open_pi_decide_record,
    get_or_open_direction_record,
)
from ..services.figure_contract import record_plan_figure_inheritance
from ..services.sample_design import plan_sample_design_violations
from ..utils.llm_client import UpstreamExhaustedError
from ..utils.formatting import truncate_for_llm
from .direction_coverage import (
    current_lineage_id,
    normalize_direction_key,
    uncovered_direction_requirements,
)
from .helpers import (
    _diff_plan_steps, _get_task_text, _normalize_tasks, _DIFF_LABELS,
    ack_restore_safe_point, wait_with_skip_time,
)
from .provenance import restore_provenance_id
from .retry import MAX_NODE_RETRIES, _retry_backoff_delay, with_retry

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict  # runtime fallback for type annotations

logger = logging.getLogger(__name__)

MAX_PLANNING_ROUNDS = 5
MAX_SPECIES_INFERENCE_ATTEMPTS = 3
PLAN_STAGNATION_TOKEN_SIMILARITY = 0.97


def _plan_semantic_text(plan: Optional[AnalysisPlan]) -> str:
    """Canonical text used only to detect a revision with no material change."""
    if plan is None:
        return ""
    payload = {
        "overview": plan.overview,
        "steps": [
            {
                "title": step.title,
                "method": step.method,
                "input": step.input,
                "expected_output": step.expected_output,
                "figure_requirements": [
                    requirement.model_dump(mode="json")
                    for requirement in step.figure_requirements
                ],
                "allow_text_only": step.allow_text_only,
                "text_only_reason": step.text_only_reason,
                "rationale": step.rationale,
                "depends_on": sorted(step.depends_on),
            }
            for step in plan.steps
        ],
        "key_decision_points": plan.key_decision_points,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True).casefold()
    return re.sub(r"\s+", " ", text).strip()


def plans_materially_equivalent(
    previous: Optional[AnalysisPlan], current: Optional[AnalysisPlan]
) -> bool:
    """Conservatively identify a revision that only rephrases the same plan."""
    if previous is None or current is None:
        return False
    if len(previous.steps) != len(current.steps):
        return False
    before = _plan_semantic_text(previous)
    after = _plan_semantic_text(current)
    if not before or not after:
        return False
    if before == after:
        return True
    before_tokens = set(re.findall(r"[a-z0-9_]+", before))
    after_tokens = set(re.findall(r"[a-z0-9_]+", after))
    union = before_tokens | after_tokens
    token_similarity = len(before_tokens & after_tokens) / len(union) if union else 1.0
    sequence_similarity = difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).ratio()
    return (
        token_similarity >= PLAN_STAGNATION_TOKEN_SIMILARITY
        and sequence_similarity >= 0.94
    )


def _consume_user_plan_feedback(state: dict) -> str:
    value = state.get("user_plan_feedback") or ""
    state["user_plan_feedback"] = None
    return value


def resolve_round_phase(current_round: int, current_phase: str,
                        has_exploration_report: bool) -> str:
    """Decide the analysis phase when entering round 2+.

    Only advance EXPLORATION -> INNOVATION once exploration actually produced a
    report (i.e. stage1 succeeded). If exploration failed last round there is no
    report; stay in EXPLORATION so the user can retry stage 1 next round —
    honoring the guard documented in synthesis._exploration_round_succeeded."""
    if current_phase == "innovation":
        return "innovation"
    return "innovation" if has_exploration_report else "exploration"


def _selected_directions_from_decision(selection_decision):
    if isinstance(selection_decision, dict):
        selected = selection_decision.get("selected_directions", [])
        return selected if isinstance(selected, list) else []
    return selection_decision if isinstance(selection_decision, list) else []


def populate_plan_scope(plan: AnalysisPlan, state: AnalysisState) -> AnalysisPlan:
    direction = state.current_round_direction or ""
    plan.round_number = state.current_round
    plan.direction = direction
    plan.direction_key = normalize_direction_key(direction)
    lineage_id = current_lineage_id(state)
    plan.branch_id = None if lineage_id == "main" else lineage_id
    plan.created_from_snapshot_id = restore_provenance_id(state)
    return plan


def resolve_plan_review_wait(state: AnalysisState) -> int:
    if state.auto_approve_plans:
        return max(0, state.plan_review_wait_seconds)
    # Inherit round_wait_seconds semantics so plan review follows the same
    # auto-skip / countdown / manual behaviour as all other HITL gates.
    return state.round_wait_seconds


def approved_plan_matches_current_context(state: AnalysisState) -> bool:
    plan = state.current_plan
    if not plan or plan.status != "approved":
        return False
    direction = state.current_round_direction or ""
    lineage_id = current_lineage_id(state)
    branch_id = None if lineage_id == "main" else lineage_id
    return (
        plan.round_number == state.current_round
        and plan.direction_key == normalize_direction_key(direction)
        and plan.branch_id == branch_id
    )


class PlanningMixin:
    """Mixin providing planning-dialogue methods. Composed into SingleCellAnalysisWorkflow."""
    @with_retry("pi_planning")
    async def _pi_planning(self, state: WorkflowState) -> WorkflowState:
        """PI agent plans the round, incorporating human guidance and adjusted instructions"""
        analysis_state = state["analysis_state"]

        skip = await self._require_core_agent(state, "pi", "pi_planning")
        if skip is not None:
            return skip

        human_guidance = state.get("human_guidance")
        adjusted_instructions = state.get("adjusted_instructions")

        if approved_plan_matches_current_context(analysis_state):
            logger.info("[%s] Plan already approved for current context, skipping re-planning", self.session_id)
            plan = analysis_state.current_plan
            state["plan_approved"] = True
            state["planning_stage"] = plan.stage
            state["planning_result"] = {
                "plan": plan.model_dump(mode="json"),
                "dialogue": [],
            }
            return state
        elif analysis_state.current_plan and analysis_state.current_plan.status == "approved":
            logger.info("[%s] Approved plan is stale for current context; regenerating", self.session_id)
            analysis_state.current_plan = None

        # ── Species inference (Round 1 only) ─────────────────────────────
        if analysis_state.current_round == 1 and not analysis_state.species:
            await self._send_message("status", "Inferring species from project information...", AgentType.PI)

            inference = await self.pi_agent.infer_species(analysis_state)

            if inference.get("confidence") == "high" and inference.get("species"):
                analysis_state.species = inference["species"]
                analysis_state.species_resolved = True
                await self._send_message(
                    "status",
                    f"Species inferred: {inference['species']} ({inference.get('reasoning', '')})",
                    AgentType.PI,
                )
                await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_RESOLVED.value)
            else:
                # Low confidence or failure — HITL loop (max 3 iterations)
                original_status = analysis_state.status
                try:
                    for attempt in range(MAX_SPECIES_INFERENCE_ATTEMPTS):
                        analysis_state.species_input_info = {
                            "source": "pi_inference",
                            "suggestions": inference.get("suggestions", []),
                            "context": inference.get("reasoning", "Unable to determine species automatically"),
                        }

                        self._species_event.clear()
                        self._species_user_response = None
                        analysis_state.wait_input_reason = WaitInputReason.SPECIES_INPUT.value
                        self._set_status(analysis_state, TaskStatus.WAITING_SPECIES_INPUT)
                        await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_INPUT.value)

                        signaled = await wait_with_skip_time(
                            self._species_event,
                            analysis_state.round_wait_seconds,
                            self.session_id,
                            "species_input",
                            safe_point_hook=getattr(self, "await_restore_safe_point", None),
                        )

                        if not signaled:
                            suggestions = inference.get("suggestions", [])
                            if suggestions:
                                analysis_state.species = suggestions[0]
                                analysis_state.species_resolved = True
                                await self._send_message(
                                    "status",
                                    f"Species auto-selected: {suggestions[0]} (timeout)",
                                    AgentType.PI,
                                )
                            break

                        user_input = self._species_user_response or ""
                        inference = await self.pi_agent.infer_species_from_hint(analysis_state, user_input)

                        if inference.get("confidence") == "high" and inference.get("species"):
                            analysis_state.species = inference["species"]
                            analysis_state.species_resolved = True
                            await self._send_message(
                                "status",
                                f"Species confirmed: {inference['species']}",
                                AgentType.PI,
                            )
                            break
                    else:
                        suggestions = inference.get("suggestions", [])
                        if suggestions:
                            analysis_state.species = suggestions[0]
                            analysis_state.species_resolved = True
                            await self._send_message(
                                "status",
                                f"Species set to {suggestions[0]} (max attempts reached)",
                                AgentType.PI,
                            )

                    if analysis_state.species_resolved:
                        await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_RESOLVED.value)
                    else:
                        await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_AUTO_SKIPPED.value)
                        await self._send_message(
                            "status",
                            "Species unresolved; later gene validation will attempt correction.",
                            AgentType.PI,
                        )
                        logger.info(
                            f"[{self.session_id}] Species not resolved after HITL — "
                            f"will rely on gene validation"
                        )
                finally:
                    analysis_state.species_input_info = None
                    analysis_state.wait_input_reason = None
                    self._set_status(analysis_state, original_status)

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
                "human_guidance_processed": True,
                "planning_stage": "stage1",
                "plan_approved": False,
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
            # Round 2+: advance to INNOVATION only if exploration produced a report;
            # otherwise stay in EXPLORATION so the user can retry stage 1.
            resolved = resolve_round_phase(
                current_round=analysis_state.current_round,
                current_phase=analysis_state.analysis_phase.value,
                has_exploration_report=analysis_state.exploration_report is not None,
            )
            analysis_state.analysis_phase = (
                AnalysisPhase.INNOVATION if resolved == "innovation" else AnalysisPhase.EXPLORATION
            )

            if resolved == "exploration":
                # Retry stage 1: re-run the exploration instruction, no direction selection.
                instructions = await self.pi_agent.initial_exploration_instruction(
                    analysis_state, analysis_state.initial_instruction
                )
            else:
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
                        analysis_state.current_round_direction = ""
                        logger.exception(f"[{self.session_id}] innovation_instruction failed, direction stays in queue")
                        raise
                    # P1-1: do NOT pop the queue here. The direction is committed
                    # at the start of the NEXT round via _initialize_round, only if
                    # this round completes successfully (LangGraph checkpoint confirms).
                    # current_round_direction (set above) is the in-flight marker.
                    # Clear single direction field
                    analysis_state.user_chosen_direction = None
                elif user_direction and user_direction not in ("", PI_DECIDE_SIGNAL):
                    logger.info(f"[{self.session_id}] Innovation phase: user-directed → {user_direction[:100]}")
                    instructions = await self.pi_agent.innovation_instruction(
                        analysis_state, user_direction
                    )
                    # Preserve direction for bio agent phase_instruction and synthesis prompt
                    analysis_state.current_round_direction = user_direction
                    # Consume the direction to avoid reusing in next round
                    analysis_state.user_chosen_direction = None
                elif user_direction == PI_DECIDE_SIGNAL or (not user_direction and analysis_state.current_round == 2):
                    # Distinguish integration-gate PI_DECIDE from Round-1 PI_DECIDE.
                    # At this point, _wait_for_user_input has already switched phase to
                    # INNOVATION, so we use integration_cycle > 0 as the signal that
                    # this PI_DECIDE came from the integration gate.
                    is_integration_decide = getattr(analysis_state, 'integration_cycle', 0) > 0

                    if is_integration_decide:
                        # Integration gate auto-skip: PI selects from integration candidates
                        logger.info(f"[{self.session_id}] Integration gate: PI selecting directions")
                        selection_decision = await self.pi_agent.select_integration_directions(
                            analysis_state,
                            analysis_state.integration_gate_candidates,
                            analysis_state.multi_direction_correlation or "",
                        )
                        selected_dirs = _selected_directions_from_decision(selection_decision)

                        # Close or create the integration pi_decide record.
                        # IMPORTANT: Do NOT use populate_direction_record_from_state here!
                        # At this point analysis_phase is INNOVATION (switched by
                        # _wait_for_user_input / interaction.py), so populate_* would
                        # read exploration_report candidates instead of integration gate
                        # candidates. Direct assignment from gate fields is correct.
                        _rec = find_open_integration_record(analysis_state)
                        if _rec is not None:
                            _rec.candidates = list(analysis_state.integration_gate_candidates)
                            _rec.literature_evidence = list(analysis_state.integration_gate_literature_evidence)
                        else:
                            _rec = get_or_open_direction_record(
                                analysis_state,
                                round_number=analysis_state.current_round - 1,
                                phase="integration",
                                integration_cycle=analysis_state.integration_cycle,
                            )
                            _rec.candidates = list(analysis_state.integration_gate_candidates)
                            _rec.literature_evidence = list(analysis_state.integration_gate_literature_evidence)
                            _rec.let_pi_decide = True
                        _rec.pi_selected = [
                            {"direction": d.get("direction", ""), "notes": d.get("notes", "")}
                            for d in selected_dirs
                        ]
                        _rec.event_type = "pi_decide"

                        # Clear gate state now that PI has decided
                        analysis_state.integration_gate_candidates = []
                        analysis_state.integration_gate_literature_evidence = []

                        if not selected_dirs:
                            # PI chose 0 directions → proceed to final report
                            logger.info(f"[{self.session_id}] PI selected 0 integration directions; finalizing")
                            analysis_state.final_report_requested = True
                            analysis_state.user_chosen_direction = None
                            # Return minimal instructions; routing will short-circuit
                            # to generate_final_report via _after_pi_planning.
                            instructions = {
                                "pathology_tasks": [],
                                "bioinformatics_tasks": [],
                                "biochemistry_tasks": [],
                                "literature_topics": [],
                            }
                            skip_nodes = ["pathology_analysis", "biochemistry_analysis", "literature_research"]
                        else:
                            # Set up multi-direction queue
                            analysis_state.user_chosen_directions = selected_dirs
                            analysis_state.active_direction_batch = list(selected_dirs)
                            analysis_state.pending_integration = False
                            first_dir = selected_dirs[0]
                            analysis_state.directions_queue = selected_dirs[1:] if len(selected_dirs) > 1 else []
                            analysis_state.max_rounds = analysis_state.current_round + len(selected_dirs)
                            dir_text = first_dir.get("direction", "")
                            dir_notes = first_dir.get("notes", "")
                            instructions = await self.pi_agent.innovation_instruction(
                                analysis_state, dir_text, notes=dir_notes
                            )
                            analysis_state.current_round_direction = dir_text
                            analysis_state.user_chosen_direction = None
                            skip_nodes = []
                    else:
                        # Round 1 direction selection (exploration gate PI_DECIDE)
                        logger.info(f"[{self.session_id}] Innovation phase: PI selecting best directions")
                        selection_decision = await self.pi_agent.select_best_directions(analysis_state)
                        selected_dirs = _selected_directions_from_decision(selection_decision)
                        rec = find_open_pi_decide_record(analysis_state)
                        if rec is None:
                            rec = get_or_open_direction_record(
                                analysis_state, round_number=analysis_state.current_round - 1,
                                phase="exploration",
                            )
                        if analysis_state.exploration_report:
                            rec.candidates = list(analysis_state.exploration_report.research_direction_candidates)
                            rec.literature_evidence = list(analysis_state.exploration_report.direction_literature_evidence)
                        rec.let_pi_decide = True
                        rec.pi_selected = [
                            {"direction": d.get("direction", ""), "notes": d.get("notes", "")}
                            for d in selected_dirs
                        ]
                        rec.event_type = "pi_decide"
                        if analysis_state.exploration_report and isinstance(selection_decision, dict):
                            analysis_state.exploration_report.direction_selection_summary = selection_decision
                        # Set up multi-direction queue
                        analysis_state.user_chosen_directions = selected_dirs
                        analysis_state.active_direction_batch = list(selected_dirs)
                        analysis_state.pending_integration = False
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
                        skip_nodes = []
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
    
    def _select_fallback_plan_from_history(
        self,
        dialogue_history: list,
    ) -> Optional["AnalysisPlan"]:
        """Walk dialogue history in reverse and return the first plan that
        passes model_validate.  Entries with pi_decision=='reject' or empty
        steps are skipped.  Returns None if no valid plan is found.

        Formerly, a manual PlanStep(**s) + AnalysisPlan(**plan_kwargs)
        fallback existed in the except branch.  That fallback was removed
        because it would fail for the same reasons model_validate failed
        (type errors in step fields), and it created a divergence from the
        schema that could silently accept malformed plans.

        Note: phase/stage parameters were removed along with the manual
        construction branch — they only served that deleted code path.
        """
        for entry in reversed(dialogue_history):
            if entry.get("pi_decision") == "reject":
                continue
            plan_dict = entry.get("plan", {})
            if not plan_dict.get("steps"):
                continue
            try:
                fallback_plan = AnalysisPlan.model_validate(plan_dict)
            except Exception:
                logger.warning(
                    "[%s] Fallback: model_validate failed for dialogue entry, skipping",
                    self.session_id,
                )
                continue
            logger.info(
                f"[{self.session_id}] Fallback: using plan from dialogue round "
                f"{entry.get('round', '?')} with {len(fallback_plan.steps)} steps"
            )
            return fallback_plan
        return None

    async def _planning_dialogue(self, state: WorkflowState) -> WorkflowState:
        """PI ↔ Bioinformatics dialogue to design an analysis plan (max 5 rounds)."""
        await ack_restore_safe_point(self)
        analysis_state = state["analysis_state"]

        # Planning calls use explicit, stateless prompts. Clear legacy BaseAgent
        # history here so in-memory sessions created before context isolation do
        # not retain multi-round planning/annotation transcripts indefinitely.
        reset_bioinfo_history = getattr(self.bioinformatics_agent, "reset_history", None)
        if callable(reset_bioinfo_history):
            reset_bioinfo_history()

        for agent_name, node in [("pi", "planning_dialogue"), ("bioinformatics", "planning_dialogue")]:
            skip = await self._require_core_agent(state, agent_name, node)
            if skip is not None:
                return skip

        pi_instructions = state.get("pi_instructions", {})
        bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
        overview = pi_instructions.get("overview", "")
        stage = state.get("planning_stage", "full")  # "full", "stage1", "stage2"
        existing_plan = analysis_state.current_plan  # may have a partial plan from revision
        if existing_plan and existing_plan.stage and existing_plan.stage != stage and stage != "full":
            existing_plan = None

        phase = analysis_state.analysis_phase.value
        combined_previous = self._build_planning_context(state)
        execution_revision_feedback = ""
        if state.get("plan_revision_needed"):
            revision_context = state.get("plan_revision_context") or {}
            revision_text = json.dumps(
                revision_context,
                ensure_ascii=False,
                default=str,
            )[:4000]
            results_text = json.dumps(
                combined_previous or {},
                ensure_ascii=False,
                default=str,
            )[:4000]
            execution_revision_feedback = (
                "Execution-triggered revision context:\n"
                f"{revision_text or '(no explicit trigger details)'}\n\n"
                "Bounded summary of results available at the pause:\n"
                f"{results_text or '(no result summary available)'}"
            )
        dialogue_history = []
        finalization_reason: Optional[str] = None

        round_num = 1
        while round_num <= MAX_PLANNING_ROUNDS:
            await self._send_message(
                "status",
                f"Planning dialogue: round {round_num}/{MAX_PLANNING_ROUNDS}",
                AgentType.PI,
            )

            # Keep the exact input revision so convergence is based on the two
            # adjacent plans, not on volatile AnalysisState references.
            previous_plan = existing_plan.model_copy(deep=True) if existing_plan else None

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
                        all_pi_feedback = truncate_for_llm("\n".join(
                            f"Round {d['round']}: {d['pi_feedback']}"
                            for d in dialogue_history
                            if d.get("pi_feedback")
                        ), max_chars=12_000)
                        # P2-6: read user_plan_feedback WITHOUT clearing it.
                        # If the revise call below fails, the feedback must
                        # survive so the user doesn't have to re-type it.
                        user_feedback = truncate_for_llm(
                            state.get("user_plan_feedback") or "",
                            max_chars=4_000,
                        )
                        pi_feedback = all_pi_feedback or (dialogue_history[-1]["pi_feedback"] if dialogue_history else "")
                        feedback_parts = [part for part in (
                            execution_revision_feedback,
                            pi_feedback,
                            f"User feedback: {user_feedback}" if user_feedback else "",
                        ) if part]
                        combined_feedback = "\n\n".join(feedback_parts)
                        # Feedback accumulates every round; the plan is re-derived
                        # every round. Without the tasks and the step-level asks
                        # travelling alongside it, the caution compounds while the
                        # content it was written about decays out of the plan.
                        all_pi_modifications = [
                            {"round": d.get("round"), "modifications": d.get("pi_modifications") or []}
                            for d in dialogue_history
                            if d.get("pi_modifications")
                        ]
                        plan = await self.bioinformatics_agent.revise_plan_incrementally(
                            state=analysis_state,
                            current_plan=existing_plan,
                            pi_feedback=combined_feedback,
                            pi_tasks=bioinf_tasks,
                            overview=overview,
                            pi_modifications=all_pi_modifications,
                        )
                        # Clear only after a successful revise call.
                        if user_feedback:
                            state["user_plan_feedback"] = None
                    populate_plan_scope(plan, analysis_state)
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
                        state["should_continue"] = False
                        analysis_state.status = TaskStatus.FAILED
                        await self._send_state_sync(analysis_state, "planning_aborted")
                        await self._send_message(
                            "error",
                            "Planning aborted: bioinformatics LLM unavailable. Workflow terminating.",
                            AgentType.PI,
                        )
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
                    state["should_continue"] = False
                    analysis_state.status = TaskStatus.FAILED
                    await self._send_state_sync(analysis_state, "planning_aborted")
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
                    unavailable_review_violations = plan_sample_design_violations(
                        plan, analysis_state,
                    )
                    if unavailable_review_violations:
                        review = {
                            "decision": "revise",
                            "feedback": (
                                "PI review was unavailable, but the deterministic "
                                "sample-design check rejected this plan."
                            ),
                            "modifications": unavailable_review_violations,
                        }
                        break
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
                    state["should_continue"] = False
                    analysis_state.status = TaskStatus.FAILED
                    await self._send_state_sync(analysis_state, "planning_aborted")
                    return state

            design_violations = plan_sample_design_violations(
                plan, analysis_state,
            )
            if design_violations:
                review = {
                    "decision": "revise",
                    "feedback": (
                        "The plan contradicts the measured modality-specific "
                        "replicate design. Unreplicated single-cell conditions "
                        "cannot support inferential condition testing, and replicate "
                        "counts alone do not verify a separable condition-by-batch "
                        "design for correction."
                    ),
                    "modifications": design_violations,
                }

            # --- Process PI review decision ---
            dialogue_history.append({
                "round": round_num,
                "plan": plan.model_dump(mode="json"),
                "pi_decision": review["decision"],
                "pi_feedback": review["feedback"],
                "pi_modifications": review.get("modifications", []),
                "materially_unchanged": plans_materially_equivalent(previous_plan, plan),
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

            if (
                review["decision"] != "approved"
                and dialogue_history[-1]["materially_unchanged"]
            ):
                finalization_reason = "no_material_plan_change"
                logger.warning(
                    "[%s] Planning dialogue converged without approval at round %s; "
                    "routing to unilateral PI finalization",
                    self.session_id,
                    round_num,
                )
                break

            round_num += 1

        # No agreement at the hard limit, or an unchanged revision proves the
        # dialogue has converged without approval. In both cases PI must actually
        # revise and commit the final plan; never relabel a draft as approved.
        if finalization_reason or round_num > MAX_PLANNING_ROUNDS:
            if finalization_reason is None:
                finalization_reason = "max_rounds"
                logger.warning(
                    f"[{self.session_id}] Planning dialogue reached max rounds "
                    f"({MAX_PLANNING_ROUNDS})"
                )

            seed_plan = (
                existing_plan
                if existing_plan is not None and existing_plan.steps
                else self._select_fallback_plan_from_history(dialogue_history)
            )
            if seed_plan is not None:
                analysis_state.current_plan = seed_plan
            await self._send_message(
                "status",
                "Planning dialogue converged without approval; PI is finalizing "
                "the plan unilaterally...",
                AgentType.PI,
            )
            try:
                final_plan = await self.pi_agent.generate_final_plan_from_dialogue(
                    analysis_state,
                    dialogue_history,
                    current_plan=seed_plan,
                )
                populate_plan_scope(final_plan, analysis_state)
                final_design_violations = plan_sample_design_violations(
                    final_plan, analysis_state,
                )
                if final_design_violations:
                    raise ValueError(
                        "Final plan violates the measured sequencing sample "
                        "design: " + "; ".join(
                            str(item.get("suggestion") or "")
                            for item in final_design_violations
                        )
                    )
                analysis_state.current_plan = final_plan
                analysis_state.plan_dialogue_history = dialogue_history
                state["plan_approved"] = True
                state["planning_result"] = {
                    "plan": final_plan.model_dump(mode="json"),
                    "dialogue": dialogue_history,
                    "finalization_reason": finalization_reason,
                }
                await self._send_message(
                    "status",
                    "PI has finalized the plan unilaterally.",
                    AgentType.PI,
                )
            except Exception as e:
                logger.exception(
                    f"[{self.session_id}] PI failed to generate final plan "
                    f"from dialogue: {e}"
                )
                await self._send_message(
                    "error",
                    f"PI failed to finalize plan after dialogue convergence: {str(e)}",
                    AgentType.PI,
                )
                state["failed_node"] = "planning_dialogue"
                state["planning_result"] = {
                    "error": str(e),
                    "findings": [],
                    "status": "failed",
                }
                state["should_continue"] = False
                analysis_state.status = TaskStatus.FAILED
                await self._send_state_sync(analysis_state, "planning_aborted")

        await self._report_direction_coverage_gaps(analysis_state)
        return state

    async def _report_direction_coverage_gaps(self, analysis_state) -> None:
        """Surface direction requirements the settled plan does not address.

        Direction coverage elsewhere answers "did a round run for this
        direction". It cannot tell the user that the plan about to execute
        dropped the questions the direction was selected to resolve — which is
        what a long revision dialogue tends to produce. Reported at the plan
        review gate so the user can add them back before execution starts.
        """
        plan = analysis_state.current_plan
        if plan is None or not getattr(plan, "steps", None):
            return
        try:
            gaps = uncovered_direction_requirements(analysis_state, plan)
        except Exception:
            logger.warning(
                "[%s] Direction coverage check failed", self.session_id, exc_info=True
            )
            return
        if not gaps:
            return
        logger.warning(
            "[%s] Settled plan does not address %d direction requirement(s)",
            self.session_id, len(gaps),
        )
        preview = "\n".join(f"- {item}" for item in gaps[:8])
        await self._send_message(
            "status",
            "Plan review: the following questions this direction was selected to "
            f"resolve are not addressed by the plan ({len(gaps)} total):\n{preview}",
            AgentType.PI,
        )

    async def _approve_plan(
        self, state: WorkflowState, plan, analysis_state, action: str
    ) -> None:
        """Mark plan as approved and notify frontend."""
        populate_plan_scope(plan, analysis_state)
        plan.status = "approved"
        plan.approved_at = datetime.now()
        # Preserve reviewed figures for completed steps carried unchanged into a
        # revision. The lineage lives beside runtime state; artifact.plan_id keeps
        # naming the plan that actually rendered and reviewed the bytes.
        record_plan_figure_inheritance(analysis_state, plan)
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
        await ack_restore_safe_point(self)
        analysis_state = state["analysis_state"]
        plan = analysis_state.current_plan

        if not plan or not plan.steps:
            logger.error(
                f"[{self.session_id}] No plan (or empty plan with {len(plan.steps) if plan else 0} "
                f"steps) to review; planning aborted, routing to final report"
            )
            await self._send_message(
                "error",
                "Planning failed: no valid plan is available.",
                AgentType.PI,
            )
            state["failed_node"] = "planning_dialogue"
            state["planning_result"] = {
                "error": "No valid analysis plan is available",
                "findings": [],
                "status": "failed",
            }
            state["should_continue"] = False
            analysis_state.status = TaskStatus.FAILED
            await self._send_state_sync(analysis_state, "planning_aborted")
            return state

        review_design_violations = plan_sample_design_violations(
            plan, analysis_state,
        )
        if review_design_violations:
            state["plan_approved"] = False
            state["user_plan_feedback"] = (
                "Deterministic sample-design validation requires revision: "
                + "; ".join(
                    str(item.get("suggestion") or "")
                    for item in review_design_violations
                )
            )
            await self._send_message(
                "status",
                "Plan review was returned for revision because it conflicts "
                "with the measured modality-specific replicate counts.",
                AgentType.PI,
            )
            return state

        # Wipe any action/event left behind by a previous gate invocation in the
        # same workflow (e.g. stage1's approve still sitting in the slot when
        # stage2's gate is entered). Without this, the `prev_action` capture
        # below would consume the stale value and skip waiting for the user.
        self._plan_review_action = None
        self._plan_review_event.clear()

        # Main review loop — continues until user approves or skips.
        # Generate dialogue HTML once — it doesn't change during the review loop.
        dialogue_html = self._build_plan_dialogue_html(
            analysis_state.plan_dialogue_history
        )
        while True:
            # Set plan status and notify frontend
            plan.status = "user_reviewing"

            # Send plan to frontend
            wait_seconds = resolve_plan_review_wait(analysis_state)
            await self._send_message("plan_review", {
                "plan": plan.model_dump(mode="json"),
                "dialogue_html": dialogue_html,
                "phase": analysis_state.analysis_phase.value,
                "round": analysis_state.current_round,
                "stage": state.get("planning_stage", "full"),
                "wait_seconds": wait_seconds,
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
                wait_seconds = resolve_plan_review_wait(analysis_state)
                signaled = await wait_with_skip_time(
                    self._plan_review_event,
                    wait_seconds,
                    self.session_id,
                    "plan_review",
                    safe_point_hook=getattr(self, "await_restore_safe_point", None),
                )
                if not signaled:
                    # Auto-skip on timeout regardless of auto_approve_plans,
                    # since the user chose a finite wait via round_wait_seconds
                    # or plan_review_wait_seconds — same semantics as round-wait.
                    self._plan_review_action = {"action": "skip", "user_input": ""}
                action = self._plan_review_action or {}
            else:
                logger.info(f"[{self.session_id}] Plan review action already received, skipping wait")
                action = prev_action

            # Consume: clear the slot + event so the next loop iteration (clarify
            # path) or the next gate invocation starts from a clean state.
            self._plan_review_action = None
            self._plan_review_event.clear()

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
