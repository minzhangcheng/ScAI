# Analysis-plan generation, incremental revision, and checkpoint discovery.
# Moved verbatim out of backend/agents/bioinformatics_agent.py; composed into BioinformaticsAgent.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import asyncio
import functools
import json
import logging
import base64
import io
import uuid
import re
import ast
import textwrap
import time
from pathlib import Path
from datetime import datetime

import aiofiles

from ..base import BaseAgent, AgentConfig
from ...utils.settings_provider import AgentSettingsProvider
from ...utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult, _TPL_NOOP_INSTALL,
)
from ...utils.checkpoint import CheckpointManager
from ...constants import PROFILED_OBJECTS_KEY
import inspect
import json as _json
from backend.services import ora_kernel
from backend.services.marker_priors import MarkerDB, summarize_database_priors

from ...utils.formatting import (
    as_list,
    coerce_json_text,
    format_revision_history,
    render_plan_steps,
    truncate_for_llm,
)
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...utils.direction_payload import (
    DirectionContext,
    build_direction_context_block,
    resolve_direction_context,
)

# Configure logger
logger = logging.getLogger(__name__)

# Prompt budgets for rendered plan steps. Generous on purpose: a revision that
# cannot see what a step consumed and why will regenerate those fields from the
# title alone, which is how a plan's biology decays into pure methodology over
# successive PI review rounds.
_COMPLETED_STEPS_BUDGET = 16_000
_AFFECTED_STEPS_BUDGET = 28_000


from .planning_support import render_pi_modifications, _normalize_plan_boundary


from .shared import (  # noqa: F401
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
    Tuple,
    asyncio,
    functools,
    json,
    logging,
    base64,
    io,
    uuid,
    re,
    ast,
    textwrap,
    time,
    Path,
    datetime,
    aiofiles,
    BaseAgent,
    AgentConfig,
    AgentSettingsProvider,
    validate_code_safety,
    DANGEROUS_PATTERNS,
    extract_code_from_markdown,
    LocalJupyterSandbox,
    CodeExecutionResult,
    _TPL_NOOP_INSTALL,
    CheckpointManager,
    inspect,
    _json,
    ora_kernel,
    MarkerDB,
    summarize_database_priors,
    _ORA_KERNEL_SRC,
    format_revision_history,
    truncate_for_llm,
    MECHANISM_EVIDENCE_POLICY,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _OVERCLAIM_WORDS,
    _NEGATION_WORDS,
    _MAX_PROGRAMS,
    _MAX_GENES,
    _MAX_OBS_LEN,
    _DEFAULT_MISSING,
    _REQUIRED_FIELDS,
    _ensure_list,
    _overclaim_unnegated,
    validate_mechanism_program_payload,
    TOOLS,
    KNOWN_TOOL_NAMES,
    build_gate_skip_note,
    count_saved_figures,
    AgentType,
    AnalysisPhase,
    AnalysisState,
    MessageRole,
    CodeExecution,
    DataDrivenMechanismProgram,
    PublicationFigure,
    FigureReviewStatus,
    AnalysisStep,
    PlanStage,
    DATA_TYPE_SCRNASEQ,
    DATA_TYPE_BULK_RNASEQ,
    OpenAICompatibleClient,
    ToolCall,
    UpstreamExhaustedError,
    EMPTY_CONTENT_PLACEHOLDER,
    tool_calls_are_paired,
    append_missing_tool_results,
    PLAN_REVISION_NEEDED_TAG,
    strip_ansi,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_invalidated_file_stems,
    record_representation_change,
    unchanged_artifact_stems,
    workspace_artifact_versions,
    SupervisorEvalResult,
    _DYNAMIC_CONTEXT_PREFIX,
    _APPROVED_PLAN_PREFIX,
    _CELL_TYPE_ASSIGN_RE,
    _FAILED_EVIDENCE_OUTPUT_RE,
    build_approved_plan_context,
    inject_revised_plan_context,
    execution_supports_evidence,
    successful_evidence_executions,
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
    _build_swallowed_error_feedback,
    _MARKER_DB,
    _get_marker_db,
    marker_priors_write,
)


class PlanGenerationFailure(RuntimeError):
    """Bounded, typed signal that a planner produced no executable plan."""

    def __init__(self, detail: str, *, phase: str = "", stage: str = "full"):
        self.detail = str(detail or "missing or invalid steps")
        self.phase = str(phase)
        self.stage = str(stage)
        super().__init__(
            "Bioinformatics plan generation produced no executable steps after "
            f"the bounded JSON repair attempt (phase={self.phase}, "
            f"stage={self.stage}): {self.detail}"
        )


class PlanningMixin:
    """Analysis-plan generation, incremental revision, and checkpoint discovery."""

    @staticmethod
    def _prepare_plan_data(
        plan_data: dict,
        *,
        reserved_step_ids: Optional[set[int]] = None,
        require_complete_steps: bool = False,
    ) -> dict:
        """Normalize LLM plan identifiers and validate every step.

        Models occasionally emit label-style identifiers such as
        ``"S2_Step_0"`` even though :class:`PlanStep` uses integer IDs.  Map
        those labels to deterministic positive integers and rewrite dependency
        references before schema validation.  Invalid structures are converted
        into the same ``parse_error`` shape used by JSON parsing so the caller
        can request one bounded repair instead of leaking a Pydantic exception
        into workflow routing.
        """
        from ...models.schemas import PlanStep

        plan_data = _normalize_plan_boundary(plan_data)
        if not isinstance(plan_data, dict):
            return {
                "steps": [],
                "parse_error": "Invalid plan schema: response must be a JSON object",
            }
        if "parse_error" in plan_data:
            return {
                **plan_data,
                "steps": [],
                "parse_error": (
                    "Invalid plan schema: parse_error is reserved for host parsing; "
                    f"model signal: {plan_data.get('parse_error')}"
                ),
            }

        raw_steps = plan_data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return {
                **plan_data,
                "steps": [],
                "parse_error": "Invalid plan schema: steps must be a non-empty JSON array",
            }

        reserved = set(reserved_step_ids or set())
        used_ids: set[int] = set()
        raw_to_normalized: Dict[str, int] = {}
        normalized_ids: List[Optional[int]] = []

        def _raw_key(value: Any) -> str:
            if isinstance(value, bool):
                raise ValueError("boolean step IDs are not supported")
            return str(value)

        def _positive_int(value: Any) -> Optional[int]:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value if value > 0 else None
            if isinstance(value, str) and value.strip().isdigit():
                parsed = int(value.strip())
                return parsed if parsed > 0 else None
            return None

        try:
            # Reserve every valid numeric ID before assigning label-style IDs,
            # so a label that appears first cannot steal a later numeric ID.
            for index, raw_step in enumerate(raw_steps):
                if not isinstance(raw_step, dict):
                    raise ValueError(f"step {index + 1} must be an object")
                raw_id = raw_step.get("id", f"__missing_{index}")
                key = _raw_key(raw_id)
                if key in raw_to_normalized:
                    raise ValueError(f"duplicate step ID: {raw_id!r}")

                normalized_id = _positive_int(raw_id)
                if normalized_id is not None:
                    if normalized_id in used_ids:
                        raise ValueError(f"duplicate numeric step ID: {normalized_id}")
                    used_ids.add(normalized_id)
                    raw_to_normalized[key] = normalized_id
                normalized_ids.append(normalized_id)

            # Label-style or missing IDs receive the smallest available
            # integer, excluding IDs of locked completed steps during revision.
            next_id = 1
            for index, normalized_id in enumerate(normalized_ids):
                if normalized_id is not None:
                    continue
                while next_id in used_ids or next_id in reserved:
                    next_id += 1
                normalized_id = next_id
                next_id += 1
                raw_id = raw_steps[index].get("id", f"__missing_{index}")
                used_ids.add(normalized_id)
                raw_to_normalized[_raw_key(raw_id)] = normalized_id
                normalized_ids[index] = normalized_id

            missing_reserved = reserved - used_ids
            if missing_reserved:
                raise ValueError(
                    "revised plan omitted locked completed step IDs: "
                    f"{sorted(missing_reserved)}"
                )

            normalized_steps = []
            for index, raw_step in enumerate(raw_steps):
                step = dict(raw_step)
                step["id"] = normalized_ids[index]
                raw_dependencies = step.get("depends_on", [])
                if not isinstance(raw_dependencies, list):
                    raise ValueError(
                        f"depends_on for step {step['id']} must be an array"
                    )

                dependencies: List[int] = []
                for dependency in raw_dependencies:
                    key = _raw_key(dependency)
                    if key in raw_to_normalized:
                        dependencies.append(raw_to_normalized[key])
                        continue
                    numeric_dependency = _positive_int(dependency)
                    if numeric_dependency is not None and (
                        numeric_dependency in used_ids
                        or numeric_dependency in reserved
                    ):
                        dependencies.append(numeric_dependency)
                        continue
                    raise ValueError(
                        f"step {step['id']} references unknown dependency "
                        f"{dependency!r}"
                    )

                step["depends_on"] = dependencies
                # Validate all fields now, while a repair retry is still
                # possible. The canonical objects are constructed by callers.
                PlanStep.model_validate(step)
                normalized_steps.append(step)

            return {**plan_data, "steps": normalized_steps}
        except Exception as exc:
            logger.warning("[Bioinformatics] Invalid plan schema: %s", exc)
            return {
                **plan_data,
                "steps": [],
                "parse_error": f"Invalid plan schema: {exc}",
            }

    async def _retry_on_parse_failure(
        self,
        prompt: str,
        plan_data: dict,
        *,
        warn_on_double_failure: bool = True,
        caller_name: str = "",
        reserved_step_ids: Optional[set[int]] = None,
        require_complete_steps: bool = False,
    ) -> dict:
        """Normalize and validate a plan, retrying one malformed response."""
        plan_data = self._prepare_plan_data(
            plan_data,
            reserved_step_ids=reserved_step_ids,
            require_complete_steps=require_complete_steps,
        )
        if not isinstance(plan_data, dict):
            plan_data = {
                "steps": [],
                "parse_error": "Invalid plan schema: normalizer returned a non-object",
            }
        if "parse_error" not in plan_data and plan_data.get("steps"):
            return plan_data

        validation_error = str(plan_data.get("parse_error") or "")[:4000]
        retry_prompt = (
            prompt
            + f"\n\nIMPORTANT: Use valid JSON only. Numeric fields must be numbers, "
            f"e.g. \"id\": 1 and \"depends_on\": [1, 2]. "
            f"Do not add stray quotes after numbers. "
            f"Double-check JSON syntax before responding. "
            f"Validation error to correct: {validation_error or 'missing or invalid steps'}"
            f"\n[retry:{uuid.uuid4().hex[:8]}]"
        )
        # The retry prompt contains the complete original request, so it must not
        # inherit unrelated planning/validation turns from BaseAgent history.
        resp2 = await self.chat_without_history(retry_prompt, temperature=0.3)
        data2 = self._safe_parse_json(resp2, default={})
        data2 = self._prepare_plan_data(
            data2,
            reserved_step_ids=reserved_step_ids,
            require_complete_steps=require_complete_steps,
        )
        if not isinstance(data2, dict):
            data2 = {
                "steps": [],
                "parse_error": "Invalid plan schema: normalizer returned a non-object",
            }
        if "parse_error" not in data2 and data2.get("steps"):
            return data2

        if warn_on_double_failure:
            logger.warning(
                "[Bioinformatics] %s: retry also failed to parse; "
                "returning original result (caller will handle)",
                caller_name or self.__class__.__name__,
            )
        return plan_data

    async def generate_analysis_plan(
        self,
        state: AnalysisState,
        pi_tasks: List[str],
        overview: str,
        phase: str,
        planning_stage: PlanStage = "full",
        previous_results: Optional[Dict[str, Any]] = None,
        existing_plan: Optional['AnalysisPlan'] = None,
    ) -> 'AnalysisPlan':
        """Generate a method-level analysis plan based on PI's tasks.

        Args:
            state: Current analysis state
            pi_tasks: PI's assigned tasks (strings or dicts with task/rationale/expected_outcome)
            overview: PI's overall strategy description
            phase: "exploration" or "innovation"
            planning_stage: "full", "stage1" (QC+clustering+annotation), or "stage2" (downstream)
            previous_results: Results from stage1 (only for stage2 planning)
            existing_plan: Existing plan to revise (for mid-execution revision)

        Returns:
            AnalysisPlan with method-level steps
        """
        from ...models.schemas import AnalysisPlan, PlanStep

        tasks_text = truncate_for_llm(
            "\n".join(f"- {t}" for t in pi_tasks),
            max_chars=8_000,
        )
        bounded_overview = truncate_for_llm(str(overview), max_chars=4_000)

        previous_context = ""
        if previous_results:
            prev_text = truncate_for_llm(
                json.dumps(previous_results, indent=2, default=str),
                max_chars=4_000,
            )
            previous_context = f"""
## Previous Stage Results
{prev_text}

Use these results to plan the downstream analysis steps."""

        revision_context = ""
        if existing_plan:
            if existing_plan.status == "rejected":
                revision_context = f"""
## Previous Plan was REJECTED by PI

The following plan was completely rejected. Generate a COMPLETELY NEW approach. Do NOT reuse any structure or methods from the rejected plan. The PI's feedback will be provided in the dialogue — use it to understand what to avoid. Rejecting the approach does NOT reject the biology: every entity, hypothesis, and named mechanism the rejected plan covered must still be covered by the replacement.

Rejected plan overview: {existing_plan.overview}
Rejected plan steps (DO NOT COPY the structure):
""" + render_plan_steps(
                    existing_plan.steps, "REJECTED", budget=_AFFECTED_STEPS_BUDGET
                )
            else:
                # No tag: each step is labelled with its own status, which is
                # what "completed steps MUST NOT be changed" below refers to.
                completed_text = render_plan_steps(
                    existing_plan.steps, budget=_COMPLETED_STEPS_BUDGET
                )
                revision_context = f"""
## Existing Plan (for incremental revision)
Completed steps MUST NOT be changed:
{completed_text}

You are REVISING the remaining steps based on new findings."""

        stage_guidance = _stage_guidance_text(planning_stage)

        data_desc = truncate_for_llm(
            state.build_experimental_design_summary(),
            max_chars=8_000,
        )

        # The pre-planning profiling pass already read these files and left the
        # objects in the kernel. Without this note the planner writes a load
        # step that re-reads them, and -- worse -- an acceptance condition that
        # restates the declared sample count, which is exactly the condition the
        # data could not satisfy. Empty when no profiling pass ran, which is
        # every session already on disk.
        #
        # Gated to a genuine first plan of round 1: this function also serves
        # round 2+ and mid-round revisions, and neither should see it. The
        # profiling pass runs once, in round 1, but its sandbox_state entry is
        # never cleared, so `profiled` alone stays truthy in every later round
        # too -- current_round has to gate it. And existing_plan is only ever
        # None on the very first call for a round (see the three branches in
        # _planning_dialogue); a revision already carries "Completed steps
        # MUST NOT be changed" from revision_context, which this note's "the
        # first step should verify the loaded object" directly contradicts.
        profiled = (getattr(state, "sandbox_state", None) or {}).get(
            PROFILED_OBJECTS_KEY
        ) or []
        is_first_plan_of_round_one = (
            existing_plan is None
            and int(getattr(state, "current_round", 0) or 0) == 1
        )
        loaded_objects_block = ""
        if profiled and is_first_plan_of_round_one:
            loaded_objects_block = (
                "\n\nThe sequencing data is ALREADY LOADED in the analysis "
                f"kernel as: {', '.join(profiled)}. Replicate counts per group "
                "and data type were established by reading it and appear above."
                "\nDo NOT plan a step that re-reads these files from disk. Do "
                "NOT plan a step whose acceptance condition restates how many "
                "samples are expected -- the counts above are the measurement, "
                "not a target. The first step should verify the loaded object: "
                "its dimensions, its obs columns, and the group labels it "
                "carries."
            )

        research_context = self._research_context_block(state)

        prompt = f"""You are planning a single-cell RNA-seq bioinformatics analysis. Generate a method-level analysis plan.

## Data Context
{data_desc}{loaded_objects_block}
{research_context}

## PI's Analysis Goals
Overview: {bounded_overview}

Tasks:
{tasks_text}
{stage_guidance}
{previous_context}
{revision_context}

## Evidence Policy
{MECHANISM_EVIDENCE_POLICY}

## Instructions
Generate a detailed, method-level analysis plan. For each step specify:
- **title**: Short, descriptive name
- **method**: The specific Python library/method to use (e.g., "scanpy.pp.filter_cells with min_genes=200")
- **input**: What data/result this step consumes
- **expected_output**: What this step produces
- **figure_requirements**: A JSON list of publication figure deliverables. Each
  item has ``figure_type``, ``description``, and ``minimum_count``. Result-bearing
  analytical steps should normally have at least one; use multiple entries for
  complementary views such as a UMAP plus a violin plot, or a volcano plot plus
  a heatmap. Technical setup steps may use an empty list. In Stage 1, a standard
  platform-owned panel may additionally name ``platform_baseline`` as exactly
  ``annotation``, ``batch``, ``cluster``, or ``markers``; never use that field
  for comparisons, subclustering, uncertainty, filtering, or other decision views.
- **allow_text_only**: ``true`` only for a genuinely narrative or special task
  for which a meaningful result figure cannot be produced. Do not use it merely
  to save execution time. It cannot override a non-empty figure_requirements list.
- **text_only_reason**: Required when allow_text_only is true; explain why a
  meaningful data-result visualization is not applicable to this special task.
- **rationale**: Why this step is needed (biological/analytical justification)
- **depends_on**: List of step IDs that must complete before this step

Rules:
- Each step should be ONE distinct analytical action
- Methods should reference actual scanpy/numpy/pandas/liana/gseapy functions
- Include ALL steps needed to complete the PI's tasks
- This session has NO route to any external dataset. The inputs are the data already
  loaded plus the pinned reference collections reachable through the provided retrieval
  tool — nothing else can be downloaded, and there is no step that can acquire it. A step
  that needs data from outside this session is not executable here: say so in its rationale
  and drop it, rather than restating it as something that looks runnable. A step written
  that way is not skipped when it cannot run; it is silently satisfied by whatever the
  executor can improvise, and the substitution then carries the weight of the real thing
- Steps should form a logical, dependency-ordered pipeline
- Identify 2-4 key decision points where PI input may be needed
- Carry every entity, hypothesis, and named mechanism from the research context and the PI's tasks into the step that tests it. Name them in that step's input, expected_output, or rationale — a category label such as "relevant gene modules" is not a substitute for the specific set the tasks asked for

Return the plan as a JSON object:
{{
    "overview": "Brief summary of the analytical strategy",
    "steps": [
        {{
            "id": 1,
            "title": "...",
            "method": "...",
            "input": "...",
            "expected_output": "...",
            "figure_requirements": [
                {{"figure_type": "umap", "description": "...", "minimum_count": 1,
                  "platform_baseline": ""}}
            ],
            "allow_text_only": false,
            "text_only_reason": "",
            "rationale": "...",
            "depends_on": []
        }},
        ...
    ],
    "key_decision_points": ["..."],
    "expected_runtime": "~N analysis steps"
}}"""

        # Planning is a bounded, explicit context assembled by the workflow.
        # Keep it isolated from the autonomous executor and prior rounds.
        response = await self.chat_without_history(prompt, temperature=0.3)
        plan_data = self._safe_parse_json(response, default={})

        plan_data = await self._retry_on_parse_failure(prompt, plan_data, warn_on_double_failure=True, caller_name="generate_analysis_plan")

        if plan_data.get("parse_error") or not plan_data.get("steps"):
            raise PlanGenerationFailure(
                str(plan_data.get("parse_error") or "response contained no executable steps"),
                phase=phase,
                stage=planning_stage,
            )

        steps = []
        for s in plan_data.get("steps", []):
            steps.append(PlanStep(
                id=s.get("id", len(steps) + 1),
                title=s.get("title", ""),
                method=s.get("method", ""),
                input=s.get("input", ""),
                expected_output=s.get("expected_output", ""),
                figure_requirements=s.get("figure_requirements", []) or [],
                allow_text_only=s.get("allow_text_only", False),
                text_only_reason=s.get("text_only_reason", ""),
                rationale=s.get("rationale", ""),
                depends_on=s.get("depends_on", []),
            ))

        plan = AnalysisPlan(
            phase=phase,
            stage=planning_stage,
            overview=plan_data.get("overview") or overview,
            steps=steps,
            key_decision_points=plan_data.get("key_decision_points") or [],
            expected_runtime=(
                plan_data.get("expected_runtime") or f"~{len(steps)} steps"
            ),
            status="draft",
            revision_of=existing_plan.id if existing_plan else None,
        )

        logger.info(f"[Bioinformatics] Generated plan: {len(steps)} steps, phase={phase}, stage={planning_stage}")
        return plan

    def _research_context_block(self, state: AnalysisState) -> str:
        """Objective, active direction, and that direction's own open questions.

        Planning prompts previously carried only the PI's task list. When a
        revision round re-derived the plan, nothing in its prompt stated what
        the study was for, so the model had no basis for keeping content the
        tasks had introduced.

        """
        parts: List[str] = []
        objective = truncate_for_llm(
            str(getattr(state, "initial_instruction", "") or ""),
            max_chars=3_000,
        )
        if objective:
            parts.append(f"\n## Research Objective\n{objective}")
        direction = str(getattr(state, "current_round_direction", "") or "").strip()
        if direction:
            parts.append(f"\n## Current Research Direction\n{direction}")
        try:
            context = resolve_direction_context(state, direction)
        except Exception:  # defensive: malformed history must not block planning
            logger.warning("[Bioinformatics] Could not resolve direction payload", exc_info=True)
            context = DirectionContext()
        block = build_direction_context_block(context)
        if block:
            parts.append(block)
        return "\n".join(parts)

    async def revise_plan_incrementally(
        self,
        state: AnalysisState,
        current_plan: 'AnalysisPlan',
        pi_feedback: str,
        completed_step_ids: Optional[List[int]] = None,
        pi_tasks: Optional[List[Any]] = None,
        overview: str = "",
        pi_modifications: Optional[List[Any]] = None,
    ) -> 'AnalysisPlan':
        """Incrementally revise a plan based on PI feedback, preserving completed steps.

        The prompt carries the plan's full step payload, not just titles and
        methods. Rendering a step as ``title (method)`` and then asking for a
        FULL plan back forces the model to regenerate input, expected_output,
        and rationale from a name alone — so whatever biology those fields held
        is replaced by generic methodology. PI feedback accumulates across
        rounds while the plan is re-derived each time, so the caution compounds
        and the content decays; after four or five review rounds nothing
        specific survives.

        Args:
            state: Current analysis state
            current_plan: The existing plan to revise
            pi_feedback: PI's feedback on what to change
            completed_step_ids: IDs of steps that have been successfully executed (must not change)
            pi_tasks: PI's assigned tasks for this round — still binding on the revision
            overview: PI's overall strategy description
            pi_modifications: PI's structured per-step suggestions from plan review

        Returns:
            Revised AnalysisPlan with revision_of pointing to current_plan
        """
        from ...models.schemas import AnalysisPlan, PlanStep

        if completed_step_ids is None:
            completed_step_ids = [s.id for s in current_plan.steps if s.status == "completed"]

        affected_steps = [s for s in current_plan.steps if s.id not in completed_step_ids]
        completed_steps = [s for s in current_plan.steps if s.id in completed_step_ids]

        completed_text = render_plan_steps(
            completed_steps, "COMPLETED - DO NOT CHANGE", budget=_COMPLETED_STEPS_BUDGET
        )
        affected_text = render_plan_steps(
            affected_steps, "TO REVISE", budget=_AFFECTED_STEPS_BUDGET
        )
        bounded_feedback = truncate_for_llm(str(pi_feedback), max_chars=12_000)

        tasks_text = truncate_for_llm(
            "\n".join(f"- {t}" for t in (pi_tasks or [])),
            max_chars=8_000,
        )
        goals_context = ""
        if tasks_text or overview:
            goals_context = "\n## PI's Analysis Goals (still binding on this revision)\n"
            if overview:
                goals_context += f"Overview: {truncate_for_llm(str(overview), max_chars=4_000)}\n"
            if tasks_text:
                goals_context += f"\nTasks:\n{tasks_text}\n"

        modifications_text = render_pi_modifications(pi_modifications)
        modifications_context = ""
        if modifications_text:
            modifications_context = (
                "\n## PI's Step-Level Modifications (apply each one)\n" + modifications_text + "\n"
            )

        research_context = self._research_context_block(state)
        stage_guidance = _stage_guidance_text(current_plan.stage or "full")

        prompt = f"""Revise an analysis plan based on PI feedback. Only modify affected steps — completed steps are locked.
{research_context}
{goals_context}
## PI's Feedback
{bounded_feedback}
{modifications_context}
## Completed Steps (LOCKED)
{completed_text}

## Steps To Revise
{affected_text}
{stage_guidance}

## Evidence Policy
{MECHANISM_EVIDENCE_POLICY}

## Revision Rules
- Stay strictly within the stage scope above — do NOT introduce steps that belong to a different stage.
- COMPLETED steps: Do NOT change title, method, or rationale. Only update depends_on if a new step is inserted before them.
- You may modify, remove, or reorder the "TO REVISE" steps.
- You may insert NEW steps between completed steps and revised steps.
- Every step ID and every depends_on entry MUST be a positive integer. Do not use labels such as "S2_Step_1".
- Updated plan must maintain logical dependency order.
- The feedback tells you how to analyse, not what to stop analysing. Every entity, gene, pathway, module, hypothesis, and named mechanism present in the current plan, the PI's tasks, or the research context MUST still appear in the revised plan unless the feedback explicitly says to drop it. Dropping one because the evidence for it is weak is not permitted — keep the step and state the limitation.
- Return every step with a populated input, expected_output, and rationale. Do not replace a concrete list with a category label: "curated gene sets for A, B, C" must not become "relevant gene modules".
- Preserve or explicitly revise every step's figure_requirements. Result-bearing
  steps should normally retain at least one publication figure. Set
  allow_text_only=true only for genuinely narrative or special tasks with no
  meaningful visual representation, keep figure_requirements empty, and provide
  a concrete text_only_reason.
- A caution about how strongly a result may be interpreted is not a reason to stop measuring it.

Return the FULL revised plan as JSON (include ALL steps, both completed and revised):
{{
    "overview": "Updated strategy summary",
    "steps": [
        // ALL steps (completed + revised + new), in execution order
    ],
    "key_decision_points": ["updated list"],
    "expected_runtime": "~N steps"
}}"""

        response = await self.chat_without_history(prompt, temperature=0.3)
        plan_data = self._safe_parse_json(response, default={})

        plan_data = await self._retry_on_parse_failure(
            prompt,
            plan_data,
            warn_on_double_failure=False,
            caller_name="revise_plan_incrementally",
            reserved_step_ids=set(completed_step_ids),
            require_complete_steps=True,
        )

        steps = []
        for s in plan_data.get("steps", []):
            sid = s.get("id", len(steps) + 1)
            original = next((cs for cs in current_plan.steps if cs.id == sid), None)
            if original and original.status == "completed":
                # Completed work is immutable. Only dependency edges may be
                # updated when the revision inserts prerequisite steps.
                steps.append(original.model_copy(deep=True, update={
                    "depends_on": list(s.get("depends_on", [])),
                    "status": "completed",
                }))
                continue

            steps.append(PlanStep(
                id=sid,
                title=s.get("title", ""),
                method=s.get("method", ""),
                input=s.get("input", ""),
                expected_output=s.get("expected_output", ""),
                figure_requirements=(s.get(
                    "figure_requirements",
                    getattr(original, "figure_requirements", []) if original else [],
                ) or []),
                allow_text_only=s.get(
                    "allow_text_only",
                    getattr(original, "allow_text_only", False) if original else False,
                ),
                text_only_reason=s.get(
                    "text_only_reason",
                    getattr(original, "text_only_reason", "") if original else "",
                ),
                rationale=s.get("rationale", ""),
                depends_on=s.get("depends_on", []),
                status="pending",
            ))

        if not steps:
            logger.warning(
                "[Bioinformatics] revise_plan_incrementally produced 0 steps; "
                "retaining current_plan steps unchanged"
            )
            retained = current_plan.model_copy(deep=True, update={
                "status": "draft",
                "revision_of": current_plan.id,
                "revision_reason": f"Retained previous plan (revision parse failed): {pi_feedback[:500]}",
                "id": str(uuid.uuid4()),
                "approved_at": None,
            })
            return retained

        revised_plan = AnalysisPlan(
            phase=current_plan.phase,
            stage=current_plan.stage,
            overview=plan_data.get("overview") or current_plan.overview,
            steps=steps,
            key_decision_points=(
                plan_data.get("key_decision_points") or current_plan.key_decision_points
            ),
            expected_runtime=(
                plan_data.get("expected_runtime") or current_plan.expected_runtime
            ),
            status="draft",
            revision_of=current_plan.id,
            revision_reason=pi_feedback[:500],
        )

        return revised_plan

    def _checkpoint_state_dir(self) -> str:
        """Resolve the directory where this session's checkpoint lives.

        Mirrors the resolution used inside run_autonomous_analysis so both
        agree on the checkpoint path.
        """
        if hasattr(self.sandbox, 'state_dir') and self.sandbox.state_dir:
            return str(self.sandbox.state_dir)
        state_dir = str(Path(self.sandbox.working_dir) / "states")
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        return state_dir

    async def has_resumable_checkpoint(
        self, state: AnalysisState, planning_stage: str = "full"
    ) -> bool:
        """True when a non-completed checkpoint for the current round+stage exists.

        The workflow node uses this to decide whether a re-entry (e.g. the user
        clicking Retry after an LLM failure) should resume the autonomous loop
        directly. On resume, run_autonomous_analysis restores the conversation
        from the checkpoint and discards the freshly-built instruction, so the
        PI planning prelude (goal definition, task refinement, pre-execution
        snapshot) is wasted work — and worse, re-runs PI LLM calls that confuse
        the user and can fail the same way. The deeper message-shape validation
        still happens inside run_autonomous_analysis; this is only the cheap
        gate that distinguishes "fresh start" from "resume".

        The planning_stage check mirrors run_autonomous_analysis: stage1 and
        stage2 share a round_number, so a not-yet-cleared stage1 checkpoint must
        NOT be treated as resumable when stage2 begins. Only an explicit stage
        mismatch disqualifies it (pre-fix checkpoints carry no planning_stage).
        """
        try:
            checkpoint_mgr = CheckpointManager(
                session_id=state.id, state_dir=self._checkpoint_state_dir()
            )
            checkpoint = await checkpoint_mgr.load()
        except Exception:
            return False
        if not checkpoint or checkpoint.get('completed', False):
            return False
        if checkpoint.get('round_number') != state.current_round:
            return False
        cp_stage = checkpoint.get('planning_stage')
        if cp_stage is not None and cp_stage != planning_stage:
            return False
        return True

    async def _handle_report_findings(self, args: dict) -> list:
        findings = _ensure_list(args.get("findings"))
        self._last_findings = findings
        return findings
