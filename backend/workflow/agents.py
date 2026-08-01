"""Agent node methods: pathology, bioinformatics, biochemistry, literature."""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models.schemas import AgentType, AnalysisPhase, AnalysisStep, MessageRole, TaskStatus
from ..utils.llm_client import UpstreamExhaustedError
from ..utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ..utils.expression_contract import record_round_start
from ..agents.bioinformatics_agent import count_saved_figures
from ..agents.bioinformatics.exploration_diagnostics import (
    ensure_exploration_diagnostic_figure,
)
from .helpers import _get_task_text, _has_stage1_keyword, _normalize_tasks, cache_hit, cache_invalidate, cache_store, decide_bioinfo_post_execution, prepend_species, safe_pi_brief, STAGE1_KEYWORDS
from ..utils.blob_store import BLOB_REF_PREFIX
from .retry import with_retry

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

# Re-exported so `backend.workflow.agents.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .agent_nodes.shared import (  # noqa: F401
    asyncio,
    hashlib,
    json,
    logging,
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    AgentType,
    AnalysisPhase,
    AnalysisStep,
    MessageRole,
    TaskStatus,
    UpstreamExhaustedError,
    MECHANISM_EVIDENCE_POLICY,
    count_saved_figures,
    _get_task_text,
    _has_stage1_keyword,
    _normalize_tasks,
    cache_hit,
    cache_invalidate,
    cache_store,
    decide_bioinfo_post_execution,
    prepend_species,
    safe_pi_brief,
    STAGE1_KEYWORDS,
    BLOB_REF_PREFIX,
    with_retry,
    _reset_pi_instructions_for_stage2,
    _stable_code_hash,
    _STAGE1_INSTRUCTION,
    _MECHANISM_CAPTURE_INSTRUCTION,
    _build_exploration_instruction,
    _build_innovation_instruction,
    _build_stage2_instruction,
    build_phase_instruction,
)
from .agent_nodes import (  # noqa: F401
    ModalityNodesMixin,
    LiteratureNodeMixin,
)


logger = logging.getLogger(__name__)


class AgentNodesMixin(
    ModalityNodesMixin,
    LiteratureNodeMixin,
):
    """Mixin providing agent-node execution methods. Composed into SingleCellAnalysisWorkflow."""


    
    @with_retry("bioinformatics_analysis")
    async def _bioinformatics_analysis(self, state: WorkflowState) -> WorkflowState:
        """
        Bioinformatics analysis with PI-controlled iterative execution

        Flow:
        1. PI provides high-level analysis goals
        2. Bioinformatics agent runs autonomously for technical steps
        3. Agent consults PI only for biological decisions
        4. Agent reports findings when analysis phase is complete
        """
        self._current_graph_state = state
        analysis_state = state["analysis_state"]

        skip = await self._require_core_agent(state, "bioinformatics", "bioinformatics_analysis")
        if skip is not None:
            return skip

        cached = cache_hit(analysis_state, AgentType.BIOINFORMATICS)
        if cached is not None:
            logger.info(f"[{self.session_id}] bioinformatics cache hit, skipping re-execution")
            await self._send_message("result", cached, AgentType.BIOINFORMATICS)
            # If the cached result was from stage2, the normal path would have
            # transitioned planning_stage to "full". Carry that forward here so
            # downstream routing (_after_bioinformatics) sees the correct stage.
            result_state = {
                **state,
                "analysis_state": analysis_state,
                "bioinformatics_result": cached,
                "failed_node": None,
            }
            planning_stage = state.get("planning_stage", "full")
            if planning_stage == "stage2":
                result_state["planning_stage"] = "full"
                result_state["planning_context"] = None
            return result_state

        logger.info(f"[{self.session_id}] Starting bioinformatics analysis")
        seq_files = analysis_state.get_files_by_section("sequencing")
        logger.info(f"[{self.session_id}] Sequencing files: {len(seq_files)}")

        await self._send_message("status", "Starting bioinformatics analysis...", AgentType.BIOINFORMATICS)
        
        # Check if data is available
        if not analysis_state.has_sequencing_data():
            waiting_result = {"findings": [], "status": "waiting_data"}
            await self._send_message(
                "result",
                {"message": "Please upload sequencing data first", "findings": []},
                AgentType.BIOINFORMATICS
            )
            cache_store(analysis_state, AgentType.BIOINFORMATICS, waiting_result)
            return {
                **state,
                "analysis_state": analysis_state,
                "bioinformatics_result": waiting_result
            }
        
        # Planning stage support: filter tasks and adjust instructions based on stage
        planning_stage = state.get("planning_stage", "full")

        # Detect resume: a non-completed checkpoint for this round+stage means a
        # prior run already started the autonomous loop (e.g. it failed and the
        # user clicked Retry on the LLM-failure gate). On resume the agent restores
        # the conversation from the checkpoint and ignores the instruction we build
        # here, so the PI planning prelude — goal definition, task refinement, and
        # the heavy pre-execution snapshot — is pure waste and re-triggers PI LLM
        # calls. Skip it and go straight back into the autonomous loop. The
        # planning_stage argument prevents a stale stage1 checkpoint (same round)
        # from being treated as resumable once stage2 begins.
        resuming = await self.bioinformatics_agent.has_resumable_checkpoint(
            analysis_state, planning_stage
        )
        if resuming:
            logger.info(
                f"[{self.session_id}] Resumable checkpoint detected — "
                f"skipping PI planning prelude and resuming autonomous analysis"
            )

        # Incorporate human guidance from workflow state (set by _initialize_round)
        human_guidance = ""
        if state.get("human_guidance"):
            human_guidance = f"\n\nHuman researcher guidance: {state['human_guidance']}"

        # Create snapshot before execution (skip on resume — it already exists)
        if not resuming:
            await self._create_auto_snapshot(analysis_state, f"pre_execution_{planning_stage}")

        # Step 1: PI provides analysis goals
        if not resuming:
            await self._send_message("status", "PI is defining analysis goals...", AgentType.PI)
        
        # Get PI instructions from workflow state
        pi_instructions = state.get("pi_instructions", {})
        bioinf_tasks = pi_instructions.get("bioinformatics_tasks", [])
        merge_strategy = pi_instructions.get("merge_strategy", "")
        cluster_comparison_strategy = pi_instructions.get("cluster_comparison_strategy", "")

        # Stage1 task filtering: only QC + clustering + annotation tasks
        if planning_stage == "stage1":
            stage1_tasks = [t for t in bioinf_tasks if _has_stage1_keyword(_get_task_text(t))]
            if not stage1_tasks:
                stage1_tasks = ["Execute scRNA-seq QC, clustering, and cell type annotation"]
            bioinf_tasks = stage1_tasks
        elif planning_stage == "stage2":
            # The user-reviewed stage2 plan lives in current_plan; pi_instructions
            # bioinformatics_tasks were cleared by _reset_pi_instructions_for_stage2.
            # Recover the executable tasks from the approved plan so the agent runs
            # what the user reviewed instead of falling back to default goals.
            plan = analysis_state.current_plan
            if plan and getattr(plan, "steps", None):
                bioinf_tasks = [
                    (f"{s.title}: {s.method}" if s.method else s.title)
                    for s in plan.steps if getattr(s, "title", "")
                ]

        # Dynamic PI planning: refine bioinformatics tasks based on completed results
        # (skipped on resume — the refined instruction would be discarded anyway)
        overview = pi_instructions.get("overview", "")
        if overview and not resuming:
            completed = self._collect_current_round_results(state)
            if completed:
                await self._send_message("status", "PI is refining bioinformatics tasks...", AgentType.PI)
                bioinf_tasks = await self.pi_agent.plan_agent_tasks(
                    agent_name="bioinformatics",
                    original_tasks=bioinf_tasks,
                    completed_results=completed,
                    overall_strategy=overview,
                    research_objective=analysis_state.initial_instruction,
                    state=analysis_state,
                )

        # Build previous findings and code history context for round 2+
        previous_bio_findings = self._build_previous_findings_context(analysis_state, "bioinformatics")
        code_history_context = ""
        if analysis_state.current_round > 1:
            code_history_context = self._build_code_history_summary(analysis_state)

        # Include current round pathology findings if available
        pathology_context = ""
        pathology_result = state.get("pathology_result")
        if pathology_result and pathology_result.get("findings"):
            path_findings = pathology_result["findings"][:5]
            pathology_context = (
                "\n\nCurrent round pathology findings:\n"
                + "\n".join(f"- {f}" for f in path_findings)
            )

        # Include current round biochemistry findings if available
        biochemistry_context = ""
        biochemistry_result = state.get("biochemistry_result")
        if biochemistry_result and biochemistry_result.get("findings"):
            biochem_findings = biochemistry_result["findings"][:5]
            biochemistry_context = (
                "\n\nCurrent round biochemistry findings:\n"
                + "\n".join(f"- {f}" for f in biochem_findings)
            )

        # Build comprehensive instruction for bioinformatics agent
        # Phase-aware instruction for bioinformatics agent
        # Detect data types for dynamic instruction (single pass)
        has_scrna = analysis_state.has_scrnaseq()
        has_bulk = analysis_state.has_bulk_rnaseq()

        phase_instruction = build_phase_instruction(
            planning_stage=planning_stage,
            analysis_phase=analysis_state.analysis_phase.value,
            has_bulk=has_bulk, has_scrna=has_scrna,
            direction=analysis_state.current_round_direction or "",
        )

        # Build merge strategy context from PI instructions
        merge_context = ""
        if merge_strategy:
            merge_context += f"\n\nPI's cluster merge strategy:\n{merge_strategy}"
        if cluster_comparison_strategy:
            merge_context += f"\n\nPI's cluster comparison strategy:\n{cluster_comparison_strategy}"

        if has_scrna and has_bulk:
            data_desc = "single-cell RNA-seq and bulk RNA-seq"
        elif has_bulk:
            data_desc = "bulk RNA-seq"
        else:
            data_desc = "single-cell RNA-seq"

        if bioinf_tasks:
            goals_text = "\n".join(f"- {task}" for task in _normalize_tasks(bioinf_tasks))
        elif has_bulk and not has_scrna:
            goals_text = "- Perform standard bulk RNA-seq analysis (pydeseq2, pathway enrichment)"
        elif has_bulk and has_scrna:
            goals_text = ("- Perform standard scRNA-seq analysis pipeline\n"
                          "- Perform bulk RNA-seq DEG analysis (pydeseq2) and cross-validate with scRNA-seq pseudobulk DEGs")
        else:
            goals_text = "- Perform standard scRNA-seq analysis pipeline"

        # Ensure bulk direction validation appears in goals even if PI didn't assign a bulk-specific task
        if analysis_state.analysis_phase == AnalysisPhase.INNOVATION and has_bulk and has_scrna:
            goals_lower = goals_text.lower()
            bulk_validation_mentioned = any(
                kw in goals_lower for kw in ["bulk", "ssgsea", "deconvolution", "cross-validat", "pseudobulk"]
            )
            if not bulk_validation_mentioned:
                goals_text += (
                    "\n- Validate direction-level findings on bulk RNA-seq data "
                    "(reuse the bulk variables already loaded in Round 1; see the data context section "
                    "for their exact names)"
                )

        instruction = f"""Analyze {data_desc} data for project: {analysis_state.name}

Research context: {analysis_state.initial_instruction}
{phase_instruction}

PI's analysis goals for this round:
{goals_text}
{merge_context}
{human_guidance}
{pathology_context}
{biochemistry_context}
{previous_bio_findings}
{code_history_context}

This is Round {analysis_state.current_round}. The Jupyter kernel retains all variables from previous rounds.
Focus on PI's specific goals above — do NOT re-run completed steps (QC, clustering, cell annotation, etc.).
Call report_findings ONLY after completing ALL assigned tasks above."""

        # On resume the agent restores its own conversation from the checkpoint,
        # so showing this freshly-built (and discarded) instruction would mislead.
        if not resuming:
            await self._send_message("prompt", instruction, AgentType.PI)

        # Step 2: Bioinformatics agent runs autonomously with figure review support
        await self._send_message("status", "Bioinformatics agent is running autonomous analysis...", AgentType.BIOINFORMATICS)

        # Reset supervisor state for this round
        self.ensure_agent_ready("supervisor")
        if self.supervisor_agent:
            self.supervisor_agent.reset_for_round(planning_stage=planning_stage)

        # Set current state references for _pi_callback plan deviation detection
        self._current_analysis_state = analysis_state
        self._current_graph_state = state

        # Open the round for artifact-staleness purposes. Everything that writes
        # a figure or table into the workspace runs below this line, so this is
        # the boundary `unchanged_artifact_stems` needs in order to blacklist
        # only what this round produced. It cannot be derived from
        # analysis_log: the agent accumulates its steps locally and they are
        # merged and stamped with round_number further down this same method,
        # i.e. never while the round is still running. Idempotent per round, so
        # stage 2 keeps stage 1's stamp — a stage-2 normalization does
        # invalidate a figure stage 1 drew.
        record_round_start(analysis_state)

        max_steps = self.config.bioinfo_max_steps
        result = await self.bioinformatics_agent.run_autonomous_analysis(
            analysis_state,
            instruction,
            max_steps=max_steps,
            figure_review_callback=self._figure_review_callback,
            supervisor_callback=self._supervisor_callback,
            planning_stage=planning_stage,  # Pass stage context to agent
        )

        # Collect supervisor stats
        if self.supervisor_agent:
            supervision_stats = self.supervisor_agent.get_summary_stats()
            await self._send_message("supervisor_summary", json.dumps(supervision_stats), AgentType.SUPERVISOR)
            logger.info(
                f"[{self.session_id}] Supervisor: {supervision_stats['total_steps_monitored']} steps, "
                f"{supervision_stats['steps_with_technical_issues']} issues, "
                f"checkpoints: {supervision_stats['checkpoints_reached']}"
            )

        # Check result status and log accordingly
        result_status = result.get("status", "unknown")
        if result_status == "error":
            logger.error(f"[{self.session_id}] Bioinformatics analysis failed: {result.get('error', 'Unknown error')}")
            await self._send_message("error", f"Bioinformatics analysis encountered an error: {result.get('error', 'Unknown error')}", AgentType.BIOINFORMATICS)
        elif result_status in ("timeout", "max_steps_reached"):
            logger.warning(f"[{self.session_id}] Bioinformatics analysis reached step limit after {result.get('steps_executed', 0)} steps")
            await self._send_message("status", "Bioinformatics analysis reached step limit, proceeding with partial results", AgentType.BIOINFORMATICS)

        # Collect results
        all_findings = result.get("findings", [])
        all_figures = result.get("figures", [])

        existing_code_ids = {c.id for c in analysis_state.code_history}
        for code_exec in result.get("code_history", []):
            # Execution identity, not source text, defines evidence provenance.
            # The same code may fail first and succeed after runtime state changes;
            # collapsing by code hash would discard the successful evidence ID.
            if code_exec.id not in existing_code_ids:
                analysis_state.code_history.append(code_exec)
                existing_code_ids.add(code_exec.id)

        # Store analysis log (dedup by step id to handle node retries)
        existing_step_ids = {s.id for s in analysis_state.analysis_log}
        for step in result.get("analysis_log", []):
            if isinstance(step, dict):
                step = AnalysisStep(**step)
            if step.id not in existing_step_ids:
                step.round_number = max(analysis_state.current_round, 1)
                analysis_state.analysis_log.append(step)
                existing_step_ids.add(step.id)
        # Cap log size to prevent unbounded memory growth
        _MAX_ANALYSIS_LOG = 2000
        if len(analysis_state.analysis_log) > _MAX_ANALYSIS_LOG:
            analysis_state.analysis_log = analysis_state.analysis_log[-_MAX_ANALYSIS_LOG:]

        # Dedup publication_figures by name to handle node retries
        seen_fig_names = {}
        for i, fig in enumerate(analysis_state.publication_figures):
            seen_fig_names[fig.name] = i  # later index wins (keeps latest)
        if len(seen_fig_names) < len(analysis_state.publication_figures):
            analysis_state.publication_figures = [
                analysis_state.publication_figures[i]
                for i in sorted(seen_fig_names.values())
            ]

        # Step 2a: the exploration round's quality-control panel, rendered by the
        # platform rather than requested from the model. Measured over seven real
        # sessions the agent ran a full QC pass every time and plotted it zero
        # times, and the audit below cannot see that gap: METHOD_SIGNATURES holds
        # tool families and no plotting token, so a step promising a plot is
        # unverifiable and by design never counted as unmet. The instruction now
        # asks for the figure and this guarantees it, because a prompt-only figure
        # rule has already failed once in this codebase. Unconditional for the
        # exploration round: quality control is the evidence every later claim
        # rests on, and "the agent drew something else" is not a reason to leave it
        # unillustrated. Upserted by name, so a node retry replaces rather than
        # duplicates. Cannot fail the round.
        # Both stages of the exploration round, not just the first. Stage 1 and
        # Stage 2 re-enter this node with the same round number, and every
        # successful Stage-2 cell that mutates the expression representation calls
        # invalidate_current_artifacts for round 1 — which would set is_current=False
        # on a panel rendered at the end of Stage 1 and drop it from every surface
        # before `diagnostic` is ever consulted (that check is downstream of
        # is_current in is_report_eligible). Observed in session 86a9255a, where a
        # Stage-2 normalization cell invalidated the round's only round-1 figure.
        # Rendering again after Stage 2 also means the panel describes the
        # representation the round actually ended on. Upserted by name, so the
        # second render replaces the first rather than duplicating it.
        if (
            planning_stage in ("stage1", "stage2", "full")
            and int(analysis_state.current_round or 0) <= 1
        ):
            try:
                await ensure_exploration_diagnostic_figure(
                    self.bioinformatics_agent, analysis_state
                )
            except Exception as diag_err:
                logger.warning(
                    f"[{self.session_id}] QC diagnostic panel skipped: {diag_err!r}"
                )

        # Step 2b: audit the approved plan against what actually ran, then — only
        # on hard structural signals — let the PI commission a bounded repair pass.
        # Both run inside this node so the round still produces exactly one
        # RoundResult for one direction: max_rounds, the direction queue and the
        # integration bookkeeping are untouched. Neither can fail the round.
        audit_summary = self._audit_round_plan_execution(analysis_state, result)
        review_meta, review_merge = await self._run_post_execution_review(
            analysis_state, result, audit_summary, planning_stage
        )
        if review_merge:
            all_findings, all_figures = self._merge_supplementary_into_round(
                analysis_state, result, all_findings, all_figures, review_merge
            )
            # Re-audit so the persisted plan reflects the repair, not the state
            # that triggered it.
            audit_summary = self._audit_round_plan_execution(analysis_state, result)
            review_meta["plan_audit"] = audit_summary

        # Step 3: PI summarizes results
        await self._send_message("status", "PI is reviewing analysis results...", AgentType.PI)

        summary = await self.pi_agent.summarize_bioinformatics_analysis(
            analysis_state,
            [{"findings": all_findings, "figures": all_figures}]
        )

        final_result = {
            "findings": all_findings,
            "figures": all_figures,
            "summary": summary,
            "steps_executed": result.get("steps_executed", 0),
            "status": result.get("status", "completed"),
            "mechanism_programs": result.get("mechanism_programs", []),
            # Evidence binding must survive this dict. It is read by three
            # consumers that all fail silently without it: the completeness gate in
            # derive_round_completion_status (which withholds direction-coverage
            # eligibility from a round whose findings are not bound to successful
            # executions), the evidence note handed to the PI at synthesis, and the
            # report's hedging. Dropping the keys made every round look verified,
            # because each of those consumers tests `is False` and an absent key
            # reads as None.
            "evidence_verified": result.get("evidence_verified"),
            "evidence_execution_ids": list(result.get("evidence_execution_ids") or []),
            # The executing agent's own view of what is left undone. It is present
            # in every real report_findings call and used to be dropped here, so
            # the PI synthesizing the round never saw it.
            "agent_suggestions": self._collect_agent_suggestions(result),
            "plan_audit": audit_summary,
            "post_execution_review": review_meta,
        }

        _saved_fig_count = count_saved_figures(result.get("analysis_log", []))
        logger.info(f"[{self.session_id}] Bioinformatics analysis completed: {len(all_findings)} findings, {len(all_figures)} inline figures, {_saved_fig_count} saved figures, {result.get('steps_executed', 0)} steps")

        await self._send_message(
            "result",
            {
                "findings": all_findings,
                "summary": summary,
                "steps_executed": result.get("steps_executed", 0)
            },
            AgentType.BIOINFORMATICS
        )

        analysis_state.add_message(
            AgentType.BIOINFORMATICS,
            MessageRole.ASSISTANT,
            json.dumps({"findings": all_findings, "summary": summary}, ensure_ascii=False)
        )

        # Mid-execution plan revision is signalled on the raw agent `result`, not
        # on `final_result` (rebuilt above with only a fixed set of keys, so it
        # never carries plan_revision_needed). A "paused_for_revision" return means
        # the agent stopped partway through the current plan — NOT a stage
        # completion — so the stage must not advance here.
        plan_revision_requested = bool(result.get("plan_revision_needed"))
        action = decide_bioinfo_post_execution(planning_stage, plan_revision_requested)

        # One-shot: the recompute prompt has now been delivered and acted on.
        analysis_state.runtime_recompute_pending = False

        result_state = {
            **state,
            "analysis_state": analysis_state,
            "bioinformatics_result": final_result
        }

        if action == "revision":
            # Stay in the current stage and keep current_plan intact: after the
            # revision is approved the agent resumes this same plan from its
            # checkpoint (e.g. stage1's remaining annotation steps). Do NOT cache
            # the partial result and do NOT advance the stage.
            result_state["plan_revision_needed"] = True
            result_state["plan_revision_context"] = result.get("plan_revision_context", {})
            # P0-8: invalidate cache so post-revision re-execution actually runs
            # instead of returning the stale (pre-revision) cached result.
            cache_invalidate(analysis_state, AgentType.BIOINFORMATICS)
            logger.info(
                f"[{self.session_id}] Mid-execution plan revision triggered in stage "
                f"'{planning_stage}'; staying in stage, bioinfo cache invalidated"
            )
            # Required boundary: plan_revision semantic gate (pre_major_revision).
            # Fail closed via capture_required_snapshot → RequiredSnapshotBlocked.
            await self._create_auto_snapshot(analysis_state, "pre_major_revision")
        elif action == "stage1_complete":
            # Stage transition: stage1 genuinely completed → set up for stage2.
            # Don't cache the stage1 result — stage2 re-enters this node and must
            # run fresh. Biochem/pathology/literature cache survives so stage2
            # reuses them.
            result_state["planning_stage"] = "stage2"
            result_state["plan_approved"] = False
            result_state["pi_instructions"] = _reset_pi_instructions_for_stage2(
                state.get("pi_instructions", {})
            )
            result_state["planning_context"] = {
                "stage1_summary": final_result.get("summary", ""),
                "stage1_findings": final_result.get("findings", []),
                "stage1_figures": final_result.get("figures", []),
                "stage1_steps_executed": final_result.get("steps_executed", 0),
            }
            analysis_state.current_plan = None
            cache_invalidate(analysis_state, AgentType.BIOINFORMATICS)
        else:  # "complete"
            cache_store(analysis_state, AgentType.BIOINFORMATICS, final_result)
            if planning_stage == "stage2":
                result_state["planning_stage"] = "full"
                result_state["planning_context"] = None  # Clear stage1 context after stage2 completes

        return result_state
    


    
