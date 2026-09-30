"""Post-execution review of a finished round: plan-versus-evidence triage and
assessment of the bounded supplementary pass.

Composed into PIAgent. The deterministic half of the feature lives in
backend/workflow/post_execution_review.py — this module holds only the two
judgements that need the PI's model, and both are written to fail soft: an
unparseable answer returns {} and the caller proceeds as if nothing was proposed.
"""
from typing import Any, Dict, List, Optional
import json
import logging

from ...models.schemas import AnalysisState
from ...utils.direction_payload import (
    build_direction_projection,
    build_direction_projection_block,
)
from ...services.supplementary_contract import (
    render_supplementary_capability_contract,
)

logger = logging.getLogger(__name__)

# Prompt-side cap on every enumerated list. The review is a triage over a handful
# of audit facts; a longer list buys no accuracy and crowds out the instructions.
_MAX_PROMPT_ITEMS = 12
_MAX_PROMPT_ITEM_LEN = 400

# The name inventory is the one block where breadth beats brevity: a task that
# names an identifier the session does not hold is dead on arrival, and the
# executor's budget is spent finding that out. Capped per category rather than
# in total so a wide `obs` cannot crowd out the score names.
_MAX_NAMES_PER_CATEGORY = 60
_MAX_NAME_LEN = 80


class ExecutionReviewMixin:
    """Post-execution plan review and supplementary-result assessment."""

    async def select_chat_adata_source(
        self, request: str, inventory: Dict[str, Any], catalog: str,
        previous_source: Dict[str, Any], correction: str = "",
    ) -> Dict[str, Any]:
        """Choose the scientific source for one Chat analysis before execution."""
        prompt = f"""Select the AnnData source for this Chat request before Bioinformatics executes code.

Request and delegated task:
{request}

Live kernel AnnData inventory (observed now):
{json.dumps(inventory, ensure_ascii=False)[:16000]}

Registered source catalog:
{catalog or '(none)'}

Previously recorded working source (verify against the live inventory):
{json.dumps(previous_source, ensure_ascii=False)[:2000]}

If the task uses only non-AnnData results (for example a bulk table), set
uses_adata=false and adata_source=null. Otherwise choose the object that
matches the user's task and its analysis lineage. Reuse a
matching live object by choosing mode=current and its exact variable name; this
does not reload data. To use a registered checkpoint or live binding instead,
choose checkpoint_copy, checkpoint_branch, live_binding, or promote_live and
provide its exact catalog ID. Choose load_registered_inputs=true only when the
task requires loading new registered inputs rather than reusing any live AnnData.
If an AnnData source is scientifically ambiguous, set uses_adata=true,
explain the ambiguity in rationale, and return no selection; do not guess from
a variable name alone.

Return only JSON:
{{"uses_adata": true, "adata_source": {{"mode": "current",
"checkpoint_id": null, "live_binding_id": null}},
"active_adata_var": "exact live variable name or null",
"load_registered_inputs": false, "rationale": "scientific reason"}}
Replace mode with exactly one of current, checkpoint_copy, checkpoint_branch,
live_binding, or promote_live. Fill only the ID required by that mode.
For an ambiguous source, use adata_source=null and active_adata_var=null.
{correction}
"""
        response = await self.chat_without_history(prompt)
        parsed = self._safe_parse_json(response, default={})
        return parsed if isinstance(parsed, dict) else {}

    async def review_chat_execution(
        self, request: str, evidence: str, remaining_steps: int,
        remaining_revisions: int,
    ) -> Dict[str, Any]:
        """Review a Chat request against executed evidence, not round plans."""
        prompt = f"""Review the completed Chat analysis before its final answer.

User request and delegated scope:
{request}

Execution evidence and review history:
{evidence}

Assess whether the original question was answered, whether code and outputs
support the claims, and whether figures represent the computed results. Check
numerical logic, method assumptions, circular reasoning, uncertainty, and any
disagreement between records and narrative. Treat Supervisor feedback as advice.
Distinguish failed, unattempted, unsupported, and completed analyses. Do not
assume that successful execution or a visually acceptable figure establishes
scientific validity. Unreviewed figures are not visually approved. Earlier
findings may have been superseded; cite the relevant execution/figure identities.

You own the scientific decision. Choose accept, limited, or revise. Negative or
ambiguous results are valid outcomes; do not request work to obtain a preferred
conclusion. For revise, provide one actionable instruction restricted to this
request and existing inputs. Preserve prior evidence; correct or supplement only
what is needed. Do not restart the whole analysis or expand the research scope.
Remaining shared execution steps: {remaining_steps}.
Remaining correction cycles: {remaining_revisions}.
If no correction budget remains, disclose unresolved issues; do not imply repair.

Return JSON:
{{"decision": "accept|limited|revise", "rationale": "reason",
  "issues": ["issue with evidence reference"],
  "revision_instruction": "instruction, only for revise",
  "limitations": ["limitation to disclose in the final answer"]}}
"""
        unavailable = {
            "decision": "unavailable", "review_unavailable": True,
            "rationale": "PI execution review was unavailable or unparseable.",
        }
        try:
            response = await self.chat_without_history(prompt)
            data = self._safe_parse_json(response, default={})
            if not isinstance(data, dict) or data.get("decision") not in {
                "accept", "limited", "revise",
            }:
                return unavailable
            if not isinstance(data.get("rationale"), str):
                return unavailable
            for key in ("issues", "limitations"):
                if not isinstance(data.get(key, []), list) or any(
                    not isinstance(item, str) for item in data.get(key, [])
                ):
                    return unavailable
            if data["decision"] == "revise" and (
                not isinstance(data.get("revision_instruction"), str)
                or not data["revision_instruction"].strip()
            ):
                return unavailable
            return {key: data[key] for key in (
                "decision", "rationale", "issues", "limitations", "revision_instruction",
            ) if key in data}
        except Exception:
            logger.warning("Chat PI execution review failed", exc_info=True)
            return unavailable

    async def review_round_execution(
        self,
        state: AnalysisState,
        plan_audit: Dict[str, Any],
        admissions: List[str],
        findings: List[str],
        agent_suggestions: List[str],
        direction: str,
        result_status: str,
        is_terminal_round: bool,
        max_tasks: int,
        trigger_reason: str,
        session_names: Optional[Dict[str, Any]] = None,
        direction_context: str = "",
    ) -> Dict[str, Any]:
        """PI triages a completed round into at most `max_tasks` supplementary
        analyses.

        `session_names` is the inventory of identifiers this session actually
        holds — object columns, layers, registered score names, artifact names.
        It is optional so the review still runs when no live kernel is attached,
        but supplying it is what keeps a task addressable: the executor resolves
        the names verbatim, and one that does not exist costs an iteration from
        a budget measured in single digits.

        Returns the raw dict for `sanitize_review_plan` (gaps, step_decisions,
        supplementary_tasks, out_of_scope, verdict, rationale); {} on failure,
        which the caller treats as "no action".
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Reviewing round {state.current_round} execution ({trigger_reason})")

        audit_block = self._format_execution_audit(plan_audit, admissions)
        names_block = self._format_session_names(session_names)
        phase_value = getattr(getattr(state, "analysis_phase", ""), "value", None)
        phase_value = str(phase_value or getattr(state, "analysis_phase", "") or "").lower()
        innovation_review = bool(str(direction or "").strip()) and (
            phase_value == "innovation" or bool(direction_context)
        )
        if innovation_review and not direction_context:
            direction_context = build_direction_projection_block(
                build_direction_projection(state, direction)
            )
        direction_context_section = direction_context if innovation_review else ""
        review_role = (
            "Your job here is to assess this direction's completed evidence and make one "
            "PI decision about useful in-scope deepening. You may combine repair of "
            "unfinished plan work with a bounded analysis that advances the selected "
            "question. Keep other unresolved questions explicit in `followups` for the "
            "report and user Chat handoff."
            if innovation_review else
            "Your job here is TRIAGE, not brainstorming: decide which of this round's own "
            "unfinished business can still be repaired right now inside the live analysis "
            "session, and return a short executable list. Everything else must be set aside "
            "explicitly."
        )
        direction_guidance = (
            """
Use the selected-direction evidence context above to distinguish what this round
already answered from remaining questions. A clean technical audit does not imply
that the scientific question is exhausted. Null, opposing, heterogeneous, or
ambiguous results may still justify a useful follow-up, while repeated analyses of
overlapping data are not independent validation.

For each remaining question choose one disposition: `deepen_now` means it is useful,
in scope, and executable against loaded data under the shared supplementary task,
iteration, and timeout budget; `offer_in_chat` means it needs user prioritization,
broader tools, or more work than that budget; `needs_new_data` means the required
measurement/input is absent; `closed` means you have a reasoned decision that no
useful increment remains. These are PI decisions, not host keyword verdicts.
"""
            if innovation_review else ""
        )
        followups_schema = (
            """
  \"followups\": [
    {{\"question\": \"<remaining question>\",
      \"why_it_matters\": \"<information this would add>\",
      \"disposition\": \"deepen_now\" | \"offer_in_chat\" | \"needs_new_data\" | \"closed\",
      \"reason\": \"<why this disposition fits the current scope>\",
      \"evidence_references\": [\"<execution or evidence id>\"],
      \"suggested_instruction\": \"<editable Chat instruction, when useful>\",
      \"required_input\": \"<new input needed, when applicable>\"}}
  ],
"""
            if innovation_review else ""
        )
        findings_block = self._format_review_items(findings) or "None reported."
        suggestions_block = self._format_review_items(agent_suggestions) or "None."
        supplementary_contract = render_supplementary_capability_contract()
        # Mirror sanitize_review_plan's own clamp so the prompt never advertises a
        # budget the caller will then trim away.
        cap = max(0, int(max_tasks or 0))

        if is_terminal_round:
            horizon_block = (
                "## Automatic-pass boundary\n"
                "This is the only automatic supplementary pass for this selected direction. "
                "A question that is not executable within the shared budget must remain in "
                "`followups` with its PI disposition; do not describe it as dropped and do "
                "not create a renamed direction to bypass this boundary.\n"
                if innovation_review else
                "## This is the terminal round\n"
                "No later round will exist, so anything you defer is dropped for good. "
                "Prioritize what would most change this analysis's conclusions if it came "
                "out differently — not what would open new ground.\n"
            )
        else:
            horizon_block = (
                "## Automatic-pass boundary\n"
                "This is the only automatic supplementary pass for this selected direction. "
                "Keep broader or not-yet-executable questions in `followups` for the report "
                "and Chat; do not requeue them as a new direction or imply that another "
                "automatic pass will be commissioned later.\n"
                if innovation_review else
                "## A later round remains\n"
                "Work that genuinely belongs to a fresh line of investigation can wait for "
                "the next round. Keep this pass to what is unfinished here.\n"
            )

        prompt = f"""You are the PI auditing a FINISHED EXECUTION ATTEMPT before writing its summary.
{review_role}

## Project
Project: {state.name}
Research objective: {state.initial_instruction}
Round: {state.current_round}
Research direction this round: {direction or "N/A"}
Execution process ended as: {result_status or "unknown"} (this does not establish that every plan step ran)
Why this review was triggered: {trigger_reason or "N/A"}

## Plan-versus-evidence observations
The Host validated only the receipt and artifact identities below. A plan-step
association labelled `agent_nominated` is producer intent only: the Host verified the
receipt identity, not that its code, method, or output matches the nominated step. Do not
mark a step completed solely because an agent nominated a receipt or because execution
returned successfully. Compare the planned method and output with the bounded receipt and
artifact observations; an explicitly reported missing method or artifact remains a gap
unless you give an evidence-based reason otherwise. You, the PI, decide scientific
sufficiency and whether each step is completed, pending, or scientifically skipped; the
Host validates only objective identities plus step-ID and status syntax.
{audit_block}

Missing execution evidence, `not_nominated` linkage, and a method named in the
plan do not establish dependency unavailability. Describe an import or runtime
failure only when the supplied execution evidence records that failure, and
keep its method, cause, and execution scope explicit. If no such failure is
recorded, say the method was not executed or its execution was not evidenced.

## Findings this round reported
{findings_block}

## Follow-ups the analysis agent itself suggested
{suggestions_block}

{names_block}
{direction_context_section}
## What may become a supplementary task
A supplementary task is handed to a restricted code generator that runs against THE DATA
ALREADY AVAILABLE TO THIS SESSION, in this round. Registered session inputs may be read;
unregistered or external inputs may not be acquired.
Only work satisfiable that way may enter `supplementary_tasks`.

{supplementary_contract}

{direction_guidance}

This restricted pass cannot register or PI-review a PublicationFigure. A missing
publication-figure deliverable must NOT become a supplementary task: leave that gap open
for the main bioinformatics executor, which owns the publication-figure and review
pipeline. An inline diagnostic plot cannot close such a gap.

Anything requiring new laboratory measurement, additional sequencing or profiling, new
imaging or histology, a new sample or cohort, or any external or public dataset CANNOT be
executed here. Put those in `out_of_scope`, each with a one-line reason. That list is not
a consolation prize — it is where most worthwhile follow-ups honestly belong, and routing
one of them into the executor spends the whole budget on work it cannot perform.
Deferral due only to this pass's budget, tools, or figure restrictions remains
unfinished work for Chat or a later main plan. Describe that specific restriction
in `gaps` or `followups` when available; do not label it environment unavailability
or a permanent session-wide exclusion. Pathway activity is explicitly allowed
by the shared contract when its required inputs and method are available.

## Order of preference
1. Approved steps the audit found with no execution evidence — repairing these outranks
   everything else.
2. The not-run or failed items the agent recorded itself.
3. Contradictions between this round's findings, or claims in them that the evidence
   above does not support.
4. The agent's own suggested follow-ups, where they are runnable on the loaded data.
5. Genuinely new analyses — last, and only once the above are exhausted.

For a recorded import failure, assess whether a supported alternative can serve
the task within this pass. Preserve the observed failure and any unresolved work
for PI review; a method-family label alone does not rule out every implementation.

## Hard limits
- At most {cap} task(s). Return the highest-value ones and stop.
- Each task is ONE self-contained analysis instruction, phrased so a code generator can
  execute it with no further clarification: name the concrete method or function and the
  exact comparison, grouping, or object it applies to. No multi-part tasks, no "and then",
  and nothing that merely asks to investigate or explore a topic.
- Do not re-request work the audit already evidenced as executed.
- When a task refers to something that already exists, use its identifier VERBATIM from the
  inventory above. A plausible-looking name that the session does not hold fails on the
  first line and costs one of the {cap} task(s). When a task asks for something new to be
  computed, say so explicitly and describe it from the identifiers that do exist.

## "No action" is a correct answer
If the approved plan was carried out and nothing runnable on the loaded data would change
this round's conclusions, return verdict "no_action" with an empty `supplementary_tasks`.
That is the expected outcome for a clean round. Do not invent work to fill the quota, and
do not reword an out-of-scope wish into something that merely looks runnable.

{horizon_block}
Reply with STRICT JSON only:
{{
  "gaps": ["<what is unfinished or unsupported in this round>"],
  "step_decisions": [
    {{"plan_step_id": 1, "status": "completed" | "pending" | "skipped", "reason": "<PI rationale>"}}
  ],
  "supplementary_tasks": ["<one executable analysis instruction>"],
  "out_of_scope": ["<follow-up needing new data or new experiments — one-line reason>"],
{followups_schema}
  "verdict": "supplement" | "no_action",
  "rationale": "<2-4 sentences: what you are repairing, and what you are deliberately leaving>"
}}"""

        result = self._safe_parse_json(await self.chat(prompt, temperature=0.2), default={})
        if not isinstance(result, dict) or not result or "parse_error" in result:
            logger.warning("[PI] Post-execution review returned no structured plan")
            return {}
        return result

    async def assess_supplementary_result(
        self,
        state: AnalysisState,
        review_plan: Dict[str, Any],
        supplementary_result: Dict[str, Any],
        direction: str,
    ) -> Dict[str, Any]:
        """PI re-assesses the supplementary pass against the gaps it raised.

        Returns verdict / summary / resolved_gaps / remaining_gaps; {} on failure,
        leaving the caller with the raw supplementary result only.
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Assessing supplementary pass for round {state.current_round}")

        prompt = f"""You are the PI assessing a bounded SUPPLEMENTARY analysis pass you requested
after reviewing this round's execution. Judge only whether the gaps you raised are now
closed by evidence, and write the assessment so it can be folded into this round's summary.

## Project
Project: {state.name}
Research objective: {state.initial_instruction}
Round: {state.current_round}
Research direction this round: {direction or "N/A"}

## The gaps you raised and the tasks you asked for
{json.dumps(review_plan, ensure_ascii=False)[:3000]}

## What the supplementary pass produced
{json.dumps(supplementary_result, ensure_ascii=False)[:6000]}

Rules:
- `evidence_records` are bounded observations of individual execution requests. Their
  `execution_id` is the join key to the raw provenance record, `status` says only whether
  the request returned an error, and `stdout_excerpt` / `error` are the available context;
  they are not host-generated findings or scientific verdicts.
- A gap is resolved only when a completed record's bounded output contains a concrete
  result that addresses it and you judge that result sufficient. A completed status alone
  does not resolve a gap; a task that errored, was skipped, or returned no informative
  output leaves its gap open.
- An inline notebook plot is not a registered or PI-reviewed PublicationFigure and cannot
  resolve a required publication-figure gap. Leave that gap in `remaining_gaps` for the
  main executor.
- Judge the evidence produced here; do not restate the round's earlier conclusions and do
  not raise a claim's confidence beyond what this pass actually shows.
- If the pass produced nothing usable, say so plainly. "unresolved" with the gaps intact
  is a valid and useful outcome.
- Keep any remaining-question records from the review plan linked by their existing
  ``followup_id``. You may update their outcome or explanation, but this assessment
  cannot commission another automatic pass.

Reply with STRICT JSON only:
{{
  "verdict": "resolved" | "partial" | "unresolved",
  "summary": "<2-5 sentences of report-ready prose: what these analyses showed and what it changes about this round's interpretation>",
  "resolved_gaps": ["<gap now closed, with the evidence that closed it>"],
  "remaining_gaps": ["<gap still open, and why>"],
  "followups": [
    {{"followup_id": "<existing id>", "question": "<remaining question>",
      "why_it_matters": "<information this would add>",
      "disposition": "deepen_now" | "offer_in_chat" | "needs_new_data" | "closed",
      "reason": "<updated PI reason>",
      "evidence_references": ["<execution or evidence id>"],
      "suggested_instruction": "<editable Chat instruction, when useful>",
      "required_input": "<new input needed, when applicable>",
      "execution_outcome": "<factual outcome for this supplementary pass, when this question was explicitly assessed>",
      "execution_reference": "<known execution or evidence id only when explicitly tied to this question>"}}
  ]
}}"""

        parsed = self._safe_parse_json(await self.chat(prompt, temperature=0.2), default={})
        if not isinstance(parsed, dict) or not parsed or "parse_error" in parsed:
            logger.warning("[PI] Supplementary assessment returned no structured result")
            return {}

        verdict = str(parsed.get("verdict", "") or "").strip().lower()
        if verdict not in ("resolved", "partial", "unresolved"):
            # Derive it from the gap lists rather than discarding an otherwise usable
            # assessment: the prose summary is what the round's PI summary absorbs.
            resolved = parsed.get("resolved_gaps") or []
            remaining = parsed.get("remaining_gaps") or []
            verdict = "resolved" if resolved and not remaining else ("partial" if resolved else "unresolved")
        parsed["verdict"] = verdict
        return parsed

    @staticmethod
    def _format_review_items(items: Any) -> str:
        """Bounded bullet list of free-text items for a review prompt."""
        return "\n".join(
            f"- {str(item)[:_MAX_PROMPT_ITEM_LEN]}"
            for item in (items or [])[:_MAX_PROMPT_ITEMS]
            if str(item).strip()
        )

    #: Rendered in this order, so the identifiers a repair task is most likely to
    #: need survive the per-category cap even on a wide object.
    _NAME_CATEGORIES = (
        ("registered_scores", "Registered score names"),
        ("obs_columns", "Columns on the active object"),
        ("layers", "Layers on the active object"),
        ("artifacts", "Files this round wrote"),
        ("variables", "Kernel variables holding the analysis objects"),
    )

    @classmethod
    def _format_session_names(cls, session_names: Optional[Dict[str, Any]]) -> str:
        """Render the inventory of identifiers this session actually holds.

        Returns "" when nothing is known, which is the honest rendering: an empty
        heading would read as "the session holds nothing" and push the review
        toward `out_of_scope` for work that is in fact runnable.
        """
        names = session_names or {}
        lines: List[str] = []
        source = names.get("analysis_source") or {}
        if isinstance(source, dict) and source:
            lines.append("- Resolved analysis source: " + json.dumps(source, ensure_ascii=False))
            lines.append("- Frame supplementary tasks against this source and its exact cluster_key; do not reuse cluster IDs from another partition. A selection_error is a technical source limitation, not missing scientific evidence.")
        for key, label in cls._NAME_CATEGORIES:
            values = [
                str(v)[:_MAX_NAME_LEN] for v in (names.get(key) or []) if str(v).strip()
            ]
            if not values:
                continue
            shown = values[:_MAX_NAMES_PER_CATEGORY]
            suffix = f" (+{len(values) - len(shown)} more)" if len(values) > len(shown) else ""
            lines.append(f"- {label}: {', '.join(shown)}{suffix}")
        if not lines:
            return ""
        return (
            "## Identifiers this session actually holds\n"
            "These are read from the live analysis objects and this round's registrations, "
            "not from prose. Anything not listed here does not exist yet.\n"
            + "\n".join(lines)
            + "\n"
        )

    @classmethod
    def _format_execution_audit(cls, plan_audit: Dict[str, Any], admissions: List[str]) -> str:
        """Render `PlanAudit.summary()` plus the agent's own admissions."""
        audit = plan_audit or {}
        awaiting_pi = audit.get("n_awaiting_pi_decision")
        if awaiting_pi is None:
            awaiting_pi = audit.get("n_unverifiable", 0)
        lines = [
            f"- Approved steps: {audit.get('n_steps', 0)} "
            f"(evidenced {audit.get('n_completed', 0)}, "
            f"legitimately skipped {audit.get('n_skipped', 0)}, "
            f"no execution evidence {audit.get('n_unmet', 0)}, "
            f"awaiting PI completion decision {awaiting_pi})"
        ]
        figure_steps = audit.get("n_figure_steps") or 0
        if figure_steps:
            lines.append(
                f"- Required figures: {figure_steps} approved step(s) declared one; "
                f"{audit.get('n_figure_unmet', 0)} produced no linked figure this "
                "round. Artifact linkage is a technical observation; whether the "
                "missing figures matter, and whether to ask for that work again, "
                "is your decision."
            )
        observations = audit.get("step_observations") or []
        if observations:
            lines.append("- Per-step technical observations (use these plan_step_id values):")
            for item in observations:
                if not isinstance(item, dict):
                    continue
                linkage = str(
                    item.get("technical_linkage") or "linkage_not_recorded"
                )[:40]
                if linkage == "unverified":
                    linkage = "linkage_not_recorded"
                lines.append(
                    "  - plan_step_id={step_id}; linkage={linkage}; "
                    "verifiable={verifiable}; title={title}; reason={reason}".format(
                        step_id=item.get("plan_step_id", "?"),
                        linkage=linkage,
                        verifiable=bool(item.get("verifiable", False)),
                        title=str(item.get("title") or "")[:_MAX_PROMPT_ITEM_LEN],
                        reason=str(item.get("reason") or "")[:_MAX_PROMPT_ITEM_LEN],
                    )
                )
        unmet = cls._format_review_items(audit.get("unmet_titles"))
        if unmet:
            lines.append("- Approved steps with NO execution evidence:")
            lines.append(unmet)
        for label, key in (
            ("Planned method families without execution evidence", "unmet_families"),
            ("Required publication figure types still missing", "unmet_figure_kinds"),
            ("Methods evidenced by a successful execution", "executed_families"),
            ("Method families with recorded import failures", "unavailable_families"),
        ):
            values = audit.get(key) or []
            if values:
                lines.append(f"- {label}: {', '.join(str(v) for v in values[:_MAX_PROMPT_ITEMS])}")
        if not audit.get("unavailable_families"):
            lines.append("- No method import failures are recorded in this audit; "
                         "this does not guarantee that every method is runnable.")
        admitted = cls._format_review_items(admissions)
        if admitted:
            lines.append("- The analysis agent itself recorded these as not run or failed:")
            lines.append(admitted)
        return "\n".join(lines)
