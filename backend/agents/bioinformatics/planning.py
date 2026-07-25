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
import inspect
import json as _json
from backend.services import ora_kernel
from backend.services.marker_priors import MarkerDB, summarize_database_priors

from ...utils.formatting import format_revision_history, truncate_for_llm
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)


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
    MAX_WORKSPACE_H5AD_FILES,
    _HIGH_MEMORY_ANALYSIS_RE,
    _high_memory_referenced_names,
    _build_resource_guard_preamble,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _static_h5ad_path,
    workspace_h5ad_write_guard,
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


class PlanningMixin:
    """Analysis-plan generation, incremental revision, and checkpoint discovery."""

    @staticmethod
    def _prepare_plan_data(
        plan_data: dict,
        *,
        reserved_step_ids: Optional[set[int]] = None,
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

        if not isinstance(plan_data, dict) or "parse_error" in plan_data:
            return plan_data

        raw_steps = plan_data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return plan_data

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
    ) -> dict:
        """Normalize and validate a plan, retrying one malformed response."""
        plan_data = self._prepare_plan_data(
            plan_data,
            reserved_step_ids=reserved_step_ids,
        )
        if "parse_error" not in plan_data and plan_data.get("steps"):
            return plan_data

        retry_prompt = (
            prompt
            + f"\n\nIMPORTANT: Use valid JSON only. Numeric fields must be numbers, "
            f"e.g. \"id\": 1 and \"depends_on\": [1, 2]. "
            f"Do not add stray quotes after numbers. "
            f"Double-check JSON syntax before responding.\n[retry:{uuid.uuid4().hex[:8]}]"
        )
        # The retry prompt contains the complete original request, so it must not
        # inherit unrelated planning/validation turns from BaseAgent history.
        resp2 = await self.chat_without_history(retry_prompt, temperature=0.3)
        data2 = self._safe_parse_json(resp2, default={})
        data2 = self._prepare_plan_data(
            data2,
            reserved_step_ids=reserved_step_ids,
        )
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

The following plan was completely rejected. Generate a COMPLETELY NEW approach. Do NOT reuse any structure or methods from the rejected plan. The PI's feedback will be provided in the dialogue — use it to understand what to avoid.

Rejected plan overview: {existing_plan.overview}
Rejected plan steps (DO NOT COPY):
""" + truncate_for_llm("\n".join(
                    f"  Step {s.id}: {s.title} ({s.method})"
                    for s in existing_plan.steps
                ), max_chars=12_000)
            else:
                completed_text = truncate_for_llm(
                    "\n".join(
                        f"  [{s.status}] Step {s.id}: {s.title} ({s.method})"
                        for s in existing_plan.steps
                    ),
                    max_chars=12_000,
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
        
        prompt = f"""You are planning a single-cell RNA-seq bioinformatics analysis. Generate a method-level analysis plan.

## Data Context
{data_desc}

## PI's Analysis Goals
Overview: {bounded_overview}

Tasks:
{tasks_text}
{stage_guidance}
{previous_context}
{revision_context}

## Instructions
Generate a detailed, method-level analysis plan. For each step specify:
- **title**: Short, descriptive name
- **method**: The specific Python library/method to use (e.g., "scanpy.pp.filter_cells with min_genes=200")
- **input**: What data/result this step consumes
- **expected_output**: What this step produces
- **rationale**: Why this step is needed (biological/analytical justification)
- **depends_on**: List of step IDs that must complete before this step

Rules:
- Each step should be ONE distinct analytical action
- Methods should reference actual scanpy/numpy/pandas/liana/gseapy functions
- Include ALL steps needed to complete the PI's tasks
- Steps should form a logical, dependency-ordered pipeline
- Identify 2-4 key decision points where PI input may be needed

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

        steps = []
        for s in plan_data.get("steps", []):
            steps.append(PlanStep(
                id=s.get("id", len(steps) + 1),
                title=s.get("title", ""),
                method=s.get("method", ""),
                input=s.get("input", ""),
                expected_output=s.get("expected_output", ""),
                rationale=s.get("rationale", ""),
                depends_on=s.get("depends_on", []),
            ))

        plan = AnalysisPlan(
            phase=phase,
            stage=planning_stage,
            overview=plan_data.get("overview", overview),
            steps=steps,
            key_decision_points=plan_data.get("key_decision_points", []),
            expected_runtime=plan_data.get("expected_runtime", f"~{len(steps)} steps"),
            status="draft",
            revision_of=existing_plan.id if existing_plan else None,
        )

        logger.info(f"[Bioinformatics] Generated plan: {len(steps)} steps, phase={phase}, stage={planning_stage}")
        return plan

    async def revise_plan_incrementally(
        self,
        state: AnalysisState,
        current_plan: 'AnalysisPlan',
        pi_feedback: str,
        completed_step_ids: Optional[List[int]] = None,
    ) -> 'AnalysisPlan':
        """Incrementally revise a plan based on PI feedback, preserving completed steps.

        Args:
            state: Current analysis state
            current_plan: The existing plan to revise
            pi_feedback: PI's feedback on what to change
            completed_step_ids: IDs of steps that have been successfully executed (must not change)

        Returns:
            Revised AnalysisPlan with revision_of pointing to current_plan
        """
        from ...models.schemas import AnalysisPlan, PlanStep

        if completed_step_ids is None:
            completed_step_ids = [s.id for s in current_plan.steps if s.status == "completed"]

        affected_steps = [s for s in current_plan.steps if s.id not in completed_step_ids]
        completed_steps = [s for s in current_plan.steps if s.id in completed_step_ids]

        completed_text = truncate_for_llm(
            "\n".join(
                f"  [COMPLETED - DO NOT CHANGE] Step {s.id}: {s.title} ({s.method})"
                for s in completed_steps
            ),
            max_chars=12_000,
        )
        affected_text = truncate_for_llm(
            "\n".join(
                f"  [TO REVISE] Step {s.id}: {s.title} ({s.method})"
                for s in affected_steps
            ),
            max_chars=16_000,
        )
        bounded_feedback = truncate_for_llm(str(pi_feedback), max_chars=12_000)

        stage_guidance = _stage_guidance_text(current_plan.stage or "full")

        prompt = f"""Revise an analysis plan based on PI feedback. Only modify affected steps — completed steps are locked.

## PI's Feedback
{bounded_feedback}

## Completed Steps (LOCKED)
{completed_text}

## Steps To Revise
{affected_text}
{stage_guidance}

## Revision Rules
- Stay strictly within the stage scope above — do NOT introduce steps that belong to a different stage.
- COMPLETED steps: Do NOT change title, method, or rationale. Only update depends_on if a new step is inserted before them.
- You may modify, remove, or reorder the "TO REVISE" steps.
- You may insert NEW steps between completed steps and revised steps.
- Every step ID and every depends_on entry MUST be a positive integer. Do not use labels such as "S2_Step_1".
- Updated plan must maintain logical dependency order.

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
            overview=plan_data.get("overview", current_plan.overview),
            steps=steps,
            key_decision_points=plan_data.get("key_decision_points", current_plan.key_decision_points),
            expected_runtime=plan_data.get("expected_runtime", current_plan.expected_runtime),
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
