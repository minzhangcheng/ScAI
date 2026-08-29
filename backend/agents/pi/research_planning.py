"""Plan review, plan-dialogue convergence, and per-agent task planning.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List, Optional
import json
import re
import logging


# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState, RoundResult, DATA_TYPE_SCRNASEQ
from ...constants import PI_REVIEW_FALLBACK_DECISION
from ...services.sample_design import sequencing_design_instruction
from ...utils.formatting import as_list, coerce_json_text, render_plan_steps

from .shared import (  # noqa: F401
    _CONTROL_SIGNAL_RESULTS,
    _BRIEF_PREFERRED_TYPES,
    _BRIEF_QC_MARKERS,
    _DIRECTION_CANDIDATE_LIST_FIELDS,
    _normalize_supporting_data,
    _normalize_direction_candidate,
    _normalize_integration_scored_candidates,
    _bounded_brief_fallback,
    _format_database_priors_block,
    _build_default_bio_tasks,
)
from .planning_response import (
    build_validated_final_plan,
    normalize_data_needs,
    normalize_plan_deviation_result,
    normalize_plan_review_result,
    normalize_supplementary_plan,
    normalize_user_plan_feedback_result,
    select_valid_plan_seed,
)


class ResearchPlanningMixin:
    """Plan review, plan-dialogue convergence, and per-agent task planning."""

    async def review_bioinfo_plan(
        self,
        state: AnalysisState,
        plan: 'AnalysisPlan',
        dialogue_round: int = 1,
    ) -> Dict[str, Any]:
        """Review bioinformatics agent's analysis plan.

        Returns:
            Dict with keys: decision ("approved"/"revise"/"reject"), feedback (str),
            modifications (List[Dict] with step_id + suggestion)
        """
        # Shared with the bioinformatics planner so the two sides of the
        # dialogue read the same rendering, and so a long plan shrinks its
        # fields to fit rather than losing its tail.
        steps_text = render_plan_steps(plan.steps)

        decision_points = "\n".join(f"  - {d}" for d in plan.key_decision_points)
        sample_design_facts = sequencing_design_instruction(state)

        stage_label = (plan.stage or "full")
        stage_scope_section = ""
        if stage_label == "stage1":
            stage_scope_section = """
## Stage Scope (Stage 1)
This plan is for **Stage 1** only. Stage 1 covers: data loading of count matrices, QC, doublet handling, normalization / HVG, batch assessment, dimensionality reduction, clustering, and cell type annotation. Use the measured design facts below when deciding whether a correction/integration step is scientifically appropriate.

Stage 1 must NOT contain steps whose only purpose is to feed Stage 2 analyses, including:
- Loading Loom files or assigning spliced / unspliced layers (RNA velocity belongs to Stage 2)
- Trajectory / pseudotime inputs (e.g. diffusion maps staged for Slingshot / Monocle)
- Cell-cell communication preprocessing (CellChat / liana / NicheNet ligand-receptor tables)
- Pre-staging DEG contrasts, gene signatures, or pathway databases for Stage 2 enrichment

If any step's method, input, or rationale references "for downstream", "to support Stage 2", "for velocity", "for trajectory", "for communication", or any Stage 2 analysis by name, you MUST mark the plan as **revise** and list that step under modifications with the suggestion to defer it to Stage 2."""
        elif stage_label == "stage2":
            stage_scope_section = """
## Stage Scope (Stage 2)
This plan is for **Stage 2**. Stage 2 covers downstream analyses (DEG, GO/KEGG, cell communication, RNA velocity, trajectory, etc.) and may load any data those analyses need. Do NOT flag Stage 2-specific steps as out of scope here."""

        prompt = f"""You are reviewing a bioinformatics analysis plan for a single-cell RNA-seq study.

## Plan Overview
{plan.overview}

## Plan Stage
{stage_label}

## Planned Steps
{steps_text}

## Key Decision Points
{decision_points}

## Context
- Phase: {plan.phase}
- Round: {state.current_round}
- Data: {state.build_experimental_design_summary()}
- Measured sequencing-design facts: {sample_design_facts or "No additional design facts."}
{stage_scope_section}

## Review Criteria
1. **Stage scope**: Does every step belong in the declared stage? See "Stage Scope" above. Steps that prep data for a later stage are out of scope and require **revise**.
2. **Scientific validity**: Are the planned methods appropriate for the data and biological questions?
3. **Completeness**: Are any critical steps missing for THIS stage?
4. **Logical order**: Do dependencies make sense? Are steps correctly ordered?
5. **Redundancy**: Are there unnecessary or duplicate steps?
6. **Visual coverage**: Does every result-bearing analytical step declare useful
   ``figure_requirements``? In Stage 1 the platform itself renders the standard
   annotation, batch, cluster and marker baselines, so do not request duplicate
   copies; preserve additional decision plots such as doublet diagnostics or
   batch-correction before/after comparisons. Complementary views are preferred
   when they reveal different aspects of a result. An empty list is appropriate
   only for a non-result setup step or a genuinely narrative/special task. The
   latter exception requires ``allow_text_only=true``, an empty requirements
   list, and a concrete ``text_only_reason``. Missing visual coverage or contradictory
   fields require **revise**.
7. **Risk**: Are there steps likely to fail or produce unreliable results?

## Decision Options
- **approved**: Plan is in-scope and scientifically sound, no changes needed
- **revise**: Plan needs specific modifications (list them; scope-leak steps must be called out by step_id)
- **reject**: Plan is fundamentally flawed and needs complete redesign (explain why)

Respond in JSON:
{{
    "decision": "approved|revise|reject",
    "feedback": "Overall assessment and reasoning",
    "modifications": [
        {{"step_id": 3, "suggestion": "Use a sample-level method supported by the measured replicate design because..."}},
        ...
    ]
}}

IMPORTANT: Be specific in your feedback. Reference exact step IDs. If you approve, modifications must be empty. If you reject, explain what the bioinformatics agent should rethink."""

        response = await self.chat(prompt, temperature=0.2)
        result = self._safe_parse_json(response, default={})
        
        result = normalize_plan_review_result(result)
        
        logger.info(f"[PI] Plan review decision: {result['decision']} (round {dialogue_round})")
        return result

    async def generate_final_plan_from_dialogue(
        self,
        analysis_state: AnalysisState,
        dialogue_history: List[Dict[str, Any]],
        current_plan: Optional['AnalysisPlan'] = None,
    ) -> 'AnalysisPlan':
        """P1-10: Unilaterally generate a final analysis plan based on the
        planning dialogue so far. Used when the round limit is exhausted or
        successive revisions stop changing materially — PI decides alone
        instead of looping further.

        Returns an AnalysisPlan whose status is "pi_reviewing"; the workflow
        will then mark plan_approved=True so the user-review gate sees a
        concrete plan.
        """
        from ...models.schemas import AnalysisPlan, PlanStep

        phase = analysis_state.analysis_phase.value
        stage = "full"
        seed_plan = current_plan or analysis_state.current_plan
        stopped_for_stagnation = bool(
            dialogue_history
            and dialogue_history[-1].get("materially_unchanged")
        )
        termination_reason = (
            "Successive planning revisions stopped changing materially"
            if stopped_for_stagnation
            else "MAX_PLANNING_ROUNDS exhausted"
        )
        if seed_plan is not None and seed_plan.stage:
            stage = seed_plan.stage

        recent = dialogue_history[-5:] if dialogue_history else []
        history_summary = "\n\n".join(
            f"Round {entry.get('round', '?')}: PI decision={entry.get('pi_decision', '?')}\n"
            f"  PI feedback: {str(entry.get('pi_feedback', ''))[:1200]}\n"
            f"  Required modifications: "
            f"{json.dumps(entry.get('pi_modifications') or [], ensure_ascii=False)[:3000]}"
            for entry in recent
        ) or "(no prior dialogue)"
        current_plan_json = (
            json.dumps(seed_plan.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if seed_plan is not None else "(no valid current plan)"
        )

        prompt = f"""You are the PI deciding on a FINAL analysis plan unilaterally because
the bioinformatics agent and PI did not reach agreement before the dialogue
stopped ({termination_reason}). You must now commit to a plan based on the
dialogue history below — no more discussion is permitted.

## Phase
{phase}

## Stage
{stage}

## Experimental Design
{analysis_state.build_experimental_design_summary()}

## Measured Sequencing-Design Facts
{sequencing_design_instruction(analysis_state) or "No additional design facts."}

## Current Complete Plan
{current_plan_json}

## Dialogue History (most recent rounds)
{history_summary}

## Your Task
Revise the complete current plan above and produce the final AnalysisPlan in
JSON form. Resolve every still-material PI concern, preserve already-correct
steps, and keep the plan executable within the current stage's scope.

Respond ONLY with JSON in this schema:
{{
    "overview": "Short summary of the plan",
    "steps": [
        {{"id": 1, "title": "...", "method": "...", "input": "...",
          "expected_output": "...",
          "figure_requirements": [{{"figure_type": "umap", "description": "...", "minimum_count": 1, "platform_baseline": ""}}],
          "allow_text_only": false,
          "text_only_reason": "",
          "rationale": "...", "depends_on": []}},
        ...
    ],
    "key_decision_points": ["...", "..."],
    "expected_runtime": "~N steps"
}}"""

        response = await self.chat(prompt, temperature=0.2)
        seed_plan = select_valid_plan_seed(
            current_plan,
            analysis_state.current_plan,
            dialogue_history,
        )
        last_validation_error: Optional[Exception] = None
        for attempt in range(2):
            parsed = self._safe_parse_json(response, default={})
            try:
                plan = build_validated_final_plan(
                    parsed,
                    phase=phase,
                    stage=stage,
                    seed_plan=seed_plan,
                    termination_reason=termination_reason,
                )
            except Exception as exc:
                last_validation_error = exc
                if attempt == 1:
                    break
                logger.warning(
                    "[PI] Final plan failed schema validation; requesting one corrected response: %s",
                    exc,
                )
                repair_prompt = f"""Your previous FINAL AnalysisPlan JSON failed
schema validation. Correct the complete plan and return JSON only. Preserve all
scientific content and change only fields needed to satisfy the schema.

Validation error:
{str(exc)[:4000]}

Remember: ``platform_baseline`` must be exactly ``\"\"``, ``\"annotation\"``,
``\"batch\"``, ``\"cluster\"``, or ``\"markers\"``. It is not a figure type."""
                response = await self.chat(repair_prompt, temperature=0.2)
                continue

            logger.info(
                "[PI] Generated final plan unilaterally from dialogue: %s steps, phase=%s, stage=%s",
                len(plan.steps), phase, stage,
            )
            return plan

        if seed_plan is not None and seed_plan.steps and seed_plan.status != "rejected":
            fallback = seed_plan.model_copy(deep=True)
            fallback.status = "pi_reviewing"
            fallback.revision_reason = (
                f"{termination_reason} — PI finalization fallback retained the "
                "last valid plan after two schema-invalid responses"
            )
            return fallback

        from ..bioinformatics.planning import PlanGenerationFailure

        raise PlanGenerationFailure(
            "PI failed to generate a schema-valid final plan after one repair attempt",
            phase=phase,
            stage=stage,
        ) from last_validation_error

    async def process_user_plan_feedback(
        self,
        user_input: str,
        current_plan: 'AnalysisPlan',
    ) -> Dict[str, Any]:
        """Process user feedback on an analysis plan.

        PI first judges the user's intent, then decides next action.

        Returns:
            Dict with keys: 
            - intent: "confirm" | "modify" | "unclear"
            - action: "proceed_to_execution" | "revise_with_bioinfo" | "clarify_with_user"
            - guidance_for_bioinfo: str (only if action=revise_with_bioinfo)
            - clarification_question: str (only if action=clarify_with_user)
        """
        steps_summary = "\n".join(
            f"  Step {s.id}: {s.title} — {s.method}"
            for s in current_plan.steps
        )

        prompt = f"""A user has reviewed a bioinformatics analysis plan and provided feedback.

## Current Plan Summary
{steps_summary}

## User's Feedback
{user_input}

## Your Task
First, determine the user's INTENT:
- **confirm**: User explicitly approves the plan (keywords: "确认", "同意", "没问题", "可以", "approve", "confirm", "ok", "good", "looks good") or says to proceed/start
- **modify**: User wants specific changes to the plan (adding/removing/changing steps, different methods, etc.)
- **unclear**: You cannot confidently determine if the user is confirming or requesting changes

Then, based on intent, decide your ACTION:
- If **confirm** → action="proceed_to_execution", no further dialogue with bioinformatics needed
- If **modify** → action="revise_with_bioinfo", provide guidance_for_bioinfo summarizing what to change
- If **unclear** → action="clarify_with_user", provide a clarification_question to ask

## CRITICAL RULES
- If the user says ANYTHING that means "yes/go ahead/proceed/start/execute", treat it as **confirm**
- Do NOT overthink — if user says "好的" or "可以" or "go", just confirm
- Only classify as **modify** if the user gives SPECIFIC change requests
- Only classify as **unclear** if genuinely ambiguous

Respond in JSON:
{{
    "intent": "confirm|modify|unclear",
    "action": "proceed_to_execution|revise_with_bioinfo|clarify_with_user",
    "guidance_for_bioinfo": "Specific changes to make (only for revise_with_bioinfo, empty otherwise)",
    "clarification_question": "Question to ask user (only for clarify_with_user, empty otherwise)"
}}"""

        response = await self.chat(prompt, temperature=0.1)
        result = self._safe_parse_json(response, default={})
        
        if not result:
            result = {
                "intent": "confirm",
                "action": "proceed_to_execution",
                "guidance_for_bioinfo": "",
                "clarification_question": "",
            }
        
        return normalize_user_plan_feedback_result(
            result,
            user_input=user_input,
        )

    async def judge_plan_deviation(
        self,
        deviation_description: str,
        current_plan: 'AnalysisPlan',
        affected_step_ids: List[int],
    ) -> Dict[str, Any]:
        """Judge whether a mid-execution deviation is minor or major.

        Returns:
            Dict with keys:
            - severity: "minor" | "major"
            - reasoning: str
            - suggested_action: str
            - needs_user_popup: bool
        """
        affected_steps = [s for s in current_plan.steps if s.id in affected_step_ids]
        affected_text = "\n".join(
            f"  Step {s.id}: {s.title} ({s.method})"
            for s in affected_steps
        )

        prompt = f"""During bioinformatics analysis execution, a deviation from the plan was discovered.

## Deviation Description
{deviation_description}

## Affected Plan Steps
{affected_text}

## Remaining Plan Overview
{current_plan.overview}

## Your Task
Classify this deviation as **minor** or **major**:

**Minor** (no user popup needed):
- Switching between equivalent analysis methods (e.g., gseapy → clusterProfiler)
- Adjusting parameters based on data characteristics
- Adding/removing a visualization step
- Skipping a step that data genuinely doesn't support
- Adding a small complementary analysis

**Major** (requires user popup):
- The core hypothesis/direction needs to change (e.g., "thought it was T cell driven, actually myeloid")
- A completely new analysis branch is needed (e.g., "need to add trajectory analysis")
- A core objective of the plan must be removed
- The fundamental experimental design interpretation has changed

Respond in JSON:
{{
    "severity": "minor|major",
    "reasoning": "Brief explanation of your classification",
    "suggested_action": "What the bioinformatics agent should do next",
    "needs_user_popup": true|false
}}"""

        response = await self.chat(prompt, temperature=0.1)
        result = self._safe_parse_json(response, default={})
        
        return normalize_plan_deviation_result(result)

    async def plan_supplementary_analysis(
        self, state: AnalysisState, user_request: str, context: str
    ) -> Dict[str, Any]:
        """Plan supplementary analysis tasks based on user's request at a HITL point.

        Args:
            state: Current analysis state
            user_request: User's free-text description of what additional analysis they want
            context: "annotation", "direction" or "round" - which HITL checkpoint

        Returns:
            Dict with keys: bioinformatics_tasks, literature_topics, summary
        """
        # The round checkpoint used to fall into the `direction` branch, so the PI
        # was told it was choosing research directions while the user was in fact
        # asking for more work on a round that had just finished — and it received
        # none of that round's results to plan against.
        if context == "annotation":
            context_desc = ("cell type annotation confirmation (the user wants more information "
                            "before confirming cluster annotations)")
        elif context == "round":
            context_desc = ("review of a completed analysis round (the round's results are already "
                            "reported; the user is asking for additional analysis on the SAME "
                            "research topic before the workflow moves on)")
        else:
            context_desc = ("research direction selection (the user wants more information before "
                            "choosing innovation directions)")

        round_context = ""
        if context == "round":
            last_round = state.round_results[-1] if state.round_results else None
            direction = (getattr(state, "current_round_direction", "") or "").strip()
            bits = []
            if direction:
                bits.append(f"- Research direction this round: {direction}")
            if last_round is not None:
                if last_round.pi_summary:
                    bits.append(f"- What this round concluded: {last_round.pi_summary[:1200]}")
                findings = list(last_round.bioinformatics_findings or [])[:10]
                if findings:
                    bits.append("- Bioinformatics findings already reported:\n"
                                + "\n".join(f"    - {str(f)[:300]}" for f in findings))
            if bits:
                round_context = "\n\nResults this request follows on from:\n" + "\n".join(bits)

        prompt = f"""You are the PI Agent coordinating a supplementary analysis requested by the user.

Current analysis checkpoint: {context_desc}

User's request for additional analysis:
"{user_request}"

Current analysis state:
- Round: {state.current_round}
- Phase: {state.analysis_phase}
- Clusters annotated: {len(state.annotation_state.cluster_annotations) if state.annotation_state else 0}
{round_context}

Based on the user's request, plan specific supplementary analysis tasks.
Return a JSON object with these keys (empty lists for agents that don't need to run):
{{
    "bioinformatics_tasks": ["task1", "task2"],
    "literature_topics": ["topic1"],
    "summary": "Brief explanation of what will be analyzed"
}}

Keep tasks focused and actionable. The user is waiting for results to make a decision,
so prioritize speed - plan only the minimum necessary analysis."""

        response = await self.chat(prompt)
        plan = self._safe_parse_json(response, default={})
        if not plan or not isinstance(plan, dict):
            plan = {
                "bioinformatics_tasks": [user_request],
                "literature_topics": [],
                "summary": user_request,
            }
        normalized_plan = normalize_supplementary_plan(
            plan,
            user_request=user_request,
        )
        logger.info(
            "[PI] Supplementary analysis plan: %s",
            normalized_plan["summary"][:100],
        )
        return normalized_plan

    async def plan_agent_tasks(
        self,
        agent_name: str,
        original_tasks: List[str],
        completed_results: Dict[str, List[str]],
        overall_strategy: str = "",
        research_objective: str = "",
        state: Optional[AnalysisState] = None,
    ) -> List[str]:
        """Dynamically refine tasks for an agent based on already-completed results.

        Uses self.chat() to accumulate context within the round, building a
        coherent chain of thought across successive agent planning calls.

        Args:
            agent_name: Name of the target agent (e.g., "pathology", "bioinformatics")
            original_tasks: Tasks originally assigned by PI planning
            completed_results: Dict of findings from already-completed agents
            overall_strategy: Overview/strategy string for context
            research_objective: The project's research objective
            state: Optional AnalysisState for data availability context

        Returns:
            Refined list of task strings. Falls back to original_tasks on error.
        """
        if not completed_results:
            return original_tasks

        results_summary = self._format_completed_results(completed_results)

        data_summary = state.build_experimental_design_summary() if state else "Not available"

        prompt = f"""You are the PI coordinating a multi-agent analysis.

## Research Objective
{research_objective}

## Species
{state.species if state else "Not specified"}

## Available Data
{data_summary}

## Overall Strategy
{overall_strategy or "Not available"}

## Completed Agent Results (this round)
{results_summary}

## Original Tasks for {agent_name.capitalize()} Agent
{chr(10).join(f"- {t}" for t in original_tasks)}

Based on the completed results, refine the tasks for the {agent_name} agent.
You may: keep tasks unchanged, add specific focus areas, modify tasks to leverage
insights, or add cross-referencing instructions.
Ensure refined tasks are grounded in the available data. Do NOT add tasks requiring data that is not available.

Return a JSON object:
{{
    "refined_tasks": ["task1", "task2", ...]
}}

Keep tasks actionable and specific. Do NOT remove core tasks unless truly unnecessary."""

        try:
            response = await self.chat(prompt, temperature=0.3)
            result = self._safe_parse_json(response, default={})
            refined = result.get("refined_tasks", [])
            if refined and isinstance(refined, list) and all(isinstance(t, str) for t in refined):
                logger.info(
                    f"[PI] Dynamic planning refined {len(original_tasks)} -> "
                    f"{len(refined)} tasks for {agent_name}"
                )
                return refined
        except Exception as e:
            logger.warning(f"[PI] Dynamic planning failed for {agent_name}: {e}")

        return original_tasks

    async def plan_deep_research_tasks(
        self,
        topics: List[str],
        completed_results: Dict[str, List[str]],
        overall_strategy: str = "",
        research_objective: str = "",
        state: Optional[AnalysisState] = None,
    ) -> Dict[str, Any]:
        """Decide whether literature research is complex enough to split into
        multiple independent Deep Research (DR) sub-tasks.

        Returns ``{"decompose": bool, "tasks": [str, ...]}``. When ``decompose``
        is True the caller issues one DR call per task (serial, capped at 2).
        When False the caller merges all topics into a single DR call (legacy
        behavior). Falls back to ``{"decompose": False, "tasks": topics}`` on any
        parse/LLM error.
        """
        self.refresh_runtime_config_if_needed()
        MAX_DR_TASKS = 2

        results_summary = self._format_completed_results(completed_results) or "None yet."

        prompt = f"""You are the PI coordinating a literature review that will be \
executed with a Deep Research engine (each call autonomously runs multi-round \
web/literature search and synthesis).

## Research Objective
{research_objective}

## Overall Strategy
{overall_strategy or "Not available"}

## Completed Agent Results (this round)
{results_summary}

## Candidate Literature Topics
{chr(10).join(f"- {t}" for t in topics)}

        Decide whether this literature work truly REQUIRES splitting into TWO independent \
Deep Research tasks, or whether ONE comprehensive Deep Research query is sufficient.

DEFAULT to a single merged query (decompose=false). A single Deep Research call \
already runs multi-round web/literature search and synthesis internally, so it \
covers the vast majority of cases.

Set decompose=true ONLY when it is GENUINELY NECESSARY — i.e. the topics span TWO \
clearly distinct questions that each demand their own separate, deep investigation \
and cannot be adequately covered together. The bar for splitting is HIGH; when in \
doubt, prefer a single merged query.

Return a JSON object:
{{
    "decompose": true/false,
    "tasks": ["focused task 1", "focused task 2", ...]
}}

If decompose is true, provide EXACTLY 2 self-contained tasks (each a complete \
standalone research brief). If decompose is false, "tasks" may echo the candidate \
topics."""

        try:
            response = await self.chat(prompt, temperature=0.3)
            result = self._safe_parse_json(response, default={})
            decompose = bool(result.get("decompose", False))
            tasks = result.get("tasks", [])
            if not (isinstance(tasks, list) and all(isinstance(t, str) for t in tasks) and tasks):
                tasks = list(topics)
                decompose = False
            tasks = tasks[:MAX_DR_TASKS]
            # Decomposition only makes sense with 2+ tasks.
            if decompose and len(tasks) < 2:
                decompose = False
            logger.info(
                f"[PI] Deep Research planning: decompose={decompose}, "
                f"{len(tasks)} task(s)"
            )
            return {"decompose": decompose, "tasks": tasks}
        except Exception as e:
            logger.warning(f"[PI] plan_deep_research_tasks failed: {e}")
            return {"decompose": False, "tasks": list(topics)}

    async def identify_data_needs(
        self,
        state: AnalysisState,
        round_result: RoundResult
    ) -> Dict[str, List[str]]:
        """
        Identify what additional data would help the analysis
        
        Args:
            state: Current analysis state
            round_result: Results from the current round
            
        Returns:
            Dict with lists of needed data by category
        """
        prompt = f"""
## Current Analysis State
Project: {state.name}
Round: {round_result.round_number}
Research Objective: {state.initial_instruction}

## Round Summary
{round_result.pi_summary}

## Key Findings
Pathology: {round_result.pathology_findings}
Bioinformatics: {round_result.bioinformatics_findings}
Biochemistry: {round_result.biochemistry_findings}

## Next Steps Planned
{round_result.next_steps}

Based on the current findings and planned next steps, what additional data would help deepen the analysis?

Please identify specific data needs in JSON format:
{{
    "pathology": [
        "Special staining selected from the stated image type and the feature that the study asks to measure",
        "Additional tissue sections from different regions",
        ...
    ],
    "biochemistry": [
        "Quantitative readouts directly tied to the stated research question",
        "Independent measurements for the main observed change",
        "Assay controls and orthogonal validation measurements",
        ...
    ],
    "other": [
        "Western blot for specific proteins",
        "qPCR validation of key genes",
        ...
    ]
}}

Only include data that would significantly help answer the research question.
If no additional data is needed, return empty lists.
"""
        
        response = await self.chat(prompt, temperature=0.3)

        # Use base class safe JSON parser
        data = self._safe_parse_json(response, default={})

        return normalize_data_needs(data)

    async def plan_bioinformatics_workflow(
        self,
        state: AnalysisState
    ) -> List[str]:
        """
        Plan the bioinformatics analysis workflow based on research goals
        
        Returns:
            List of high-level analysis steps to perform
        """
        # Build data availability info from unified data_files
        seq_files = [f for f in state.data_files if f.section == "sequencing"]
        data_info = []
        for f in seq_files:
            group = state.get_group_by_id(f.group_id)
            group_label = group.name if group else "Ungrouped"
            sample_label = f"Sample {f.sample_number}" if f.sample_number else ""
            label = f"{group_label} {sample_label}".strip()
            data_info.append(f"- {label} ({f.data_type}): {f.file_path}")

        # Determine analysis characteristics
        seq_group_ids = set(f.group_id for f in seq_files if f.data_type == DATA_TYPE_SCRNASEQ and f.group_id)
        has_multiple_groups = len(seq_group_ids) > 1
        has_velocity = any(f.data_type == "loom" for f in seq_files)

        # Experimental design summary
        design_summary = state.build_experimental_design_summary()
        sample_design_facts = sequencing_design_instruction(state)

        prompt = f"""
## Project Information
Project Name: {state.name}
Research Objective: {state.initial_instruction}

## Experimental Design
{design_summary}

## Measured Sequencing-Design Facts
{sample_design_facts or "No additional design facts."}

## Uploaded Sequencing Data
{chr(10).join(data_info) if data_info else "No sequencing data uploaded"}

## Data Characteristics
- Multiple groups for comparison: {"Yes (" + str(len(seq_group_ids)) + " groups), differential/comparative analysis required" if has_multiple_groups else "Single group only"}
- RNA velocity data: {"Yes, cell trajectory analysis possible" if has_velocity else "No"}

Please plan the single-cell sequencing analysis workflow for this project. Return an ordered list of analysis steps.
Note: The first step must be loading data.
{"If multiple groups exist, preserve explicit group/sample labels, assess the uncorrected representation, and let the PI decide whether integration or correction is appropriate for the measured design." if has_multiple_groups else ""}

Format requirement: Return a JSON array, each element is a brief description of an analysis step.
Example: ["Load data", "Quality control", "Normalization", ...]

Return only the JSON array, no other content.
"""

        response = await self.chat(prompt, temperature=0.3)

        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                steps = json.loads(json_match.group())
                if isinstance(steps, list):
                    return steps
        except json.JSONDecodeError:
            pass

        # Default workflow if parsing fails
        default_steps = [
            "Load data and view basic information",
            "Quality control: filter low-quality cells and genes",
            "Data normalization and log transformation",
            "Identify highly variable genes",
            "PCA dimensionality reduction",
            "Build neighbor graph and UMAP visualization",
            "Leiden clustering",
            "Identify marker genes for each cluster",
            "Cell type annotation",
        ]

        if has_multiple_groups:
            default_steps.extend([
                "Multi-group differential expression analysis",
                "Pathway enrichment analysis",
                "Cell proportion comparison across groups"
            ])

        if has_velocity:
            default_steps.append("RNA velocity analysis")

        return default_steps
