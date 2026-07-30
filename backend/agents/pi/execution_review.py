"""Post-execution review of a finished round: plan-versus-evidence triage and
assessment of the bounded supplementary pass.

Composed into PIAgent. The deterministic half of the feature lives in
backend/workflow/post_execution_review.py — this module holds only the two
judgements that need the PI's model, and both are written to fail soft: an
unparseable answer returns {} and the caller proceeds as if nothing was proposed.
"""
from typing import Any, Dict, List
import json
import logging

from ...models.schemas import AnalysisState

logger = logging.getLogger(__name__)

# Prompt-side cap on every enumerated list. The review is a triage over a handful
# of audit facts; a longer list buys no accuracy and crowds out the instructions.
_MAX_PROMPT_ITEMS = 12
_MAX_PROMPT_ITEM_LEN = 400


class ExecutionReviewMixin:
    """Post-execution plan review and supplementary-result assessment."""

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
    ) -> Dict[str, Any]:
        """PI triages a completed round into at most `max_tasks` supplementary
        analyses.

        Returns the raw dict for `sanitize_review_plan` (gaps,
        supplementary_tasks, out_of_scope, verdict, rationale); {} on failure,
        which the caller treats as "no action".
        """
        self.refresh_runtime_config_if_needed()
        logger.info(f"[PI] Reviewing round {state.current_round} execution ({trigger_reason})")

        audit_block = self._format_execution_audit(plan_audit, admissions)
        findings_block = self._format_review_items(findings) or "None reported."
        suggestions_block = self._format_review_items(agent_suggestions) or "None."
        # Mirror sanitize_review_plan's own clamp so the prompt never advertises a
        # budget the caller will then trim away.
        cap = max(0, int(max_tasks or 0))

        if is_terminal_round:
            horizon_block = (
                "## This is the terminal round\n"
                "No later round will exist, so anything you defer is dropped for good. "
                "Prioritize what would most change this analysis's conclusions if it came "
                "out differently — not what would open new ground.\n"
            )
        else:
            horizon_block = (
                "## A later round remains\n"
                "Work that genuinely belongs to a fresh line of investigation can wait for "
                "the next round. Keep this pass to what is unfinished here.\n"
            )

        prompt = f"""You are the PI auditing a COMPLETED analysis round before writing its summary.
Your job here is TRIAGE, not brainstorming: decide which of this round's own unfinished
business can still be repaired right now inside the live analysis session, and return a
short executable list. Everything else must be set aside explicitly.

## Project
Project: {state.name}
Research objective: {state.initial_instruction}
Round: {state.current_round}
Research direction this round: {direction or "N/A"}
Execution ended as: {result_status or "unknown"}
Why this review was triggered: {trigger_reason or "N/A"}

## Plan-versus-evidence audit (machine-checked against executed code, not the agent's prose)
{audit_block}

## Findings this round reported
{findings_block}

## Follow-ups the analysis agent itself suggested
{suggestions_block}

## What may become a supplementary task
A supplementary task is handed to a restricted code generator that runs against THE DATA
ALREADY LOADED IN THIS SESSION'S KERNEL, in this round, with no new inputs of any kind.
Only work satisfiable that way may enter `supplementary_tasks`.

Anything requiring new laboratory measurement, additional sequencing or profiling, new
imaging or histology, a new sample or cohort, or any external or public dataset CANNOT be
executed here. Put those in `out_of_scope`, each with a one-line reason. That list is not
a consolation prize — it is where most worthwhile follow-ups honestly belong, and routing
one of them into the executor spends the whole budget on work it cannot perform.

## Order of preference
1. Approved steps the audit found with no execution evidence — repairing these outranks
   everything else.
2. The not-run or failed items the agent recorded itself.
3. Contradictions between this round's findings, or claims in them that the evidence
   above does not support.
4. The agent's own suggested follow-ups, where they are runnable on the loaded data.
5. Genuinely new analyses — last, and only once the above are exhausted.

Never propose a method the audit lists as unavailable in this kernel. If its intent still
matters, name an available method that serves it, or record it in `out_of_scope`.

## Hard limits
- At most {cap} task(s). Return the highest-value ones and stop.
- Each task is ONE self-contained analysis instruction, phrased so a code generator can
  execute it with no further clarification: name the concrete method or function and the
  exact comparison, grouping, or object it applies to. No multi-part tasks, no "and then",
  and nothing that merely asks to investigate or explore a topic.
- Do not re-request work the audit already evidenced as executed.

## "No action" is a correct answer
If the approved plan was carried out and nothing runnable on the loaded data would change
this round's conclusions, return verdict "no_action" with an empty `supplementary_tasks`.
That is the expected outcome for a clean round. Do not invent work to fill the quota, and
do not reword an out-of-scope wish into something that merely looks runnable.

{horizon_block}
Reply with STRICT JSON only:
{{
  "gaps": ["<what is unfinished or unsupported in this round>"],
  "supplementary_tasks": ["<one executable analysis instruction>"],
  "out_of_scope": ["<follow-up needing new data or new experiments — one-line reason>"],
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
- A gap is resolved only when the pass produced a concrete result that addresses it. A
  task that errored, was skipped, or returned nothing leaves its gap open.
- Judge the evidence produced here; do not restate the round's earlier conclusions and do
  not raise a claim's confidence beyond what this pass actually shows.
- If the pass produced nothing usable, say so plainly. "unresolved" with the gaps intact
  is a valid and useful outcome.

Reply with STRICT JSON only:
{{
  "verdict": "resolved" | "partial" | "unresolved",
  "summary": "<2-5 sentences of report-ready prose: what these analyses showed and what it changes about this round's interpretation>",
  "resolved_gaps": ["<gap now closed, with the evidence that closed it>"],
  "remaining_gaps": ["<gap still open, and why>"]
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

    @classmethod
    def _format_execution_audit(cls, plan_audit: Dict[str, Any], admissions: List[str]) -> str:
        """Render `PlanAudit.summary()` plus the agent's own admissions.

        Only counts and the identifiers that failed the audit are shown. The PI is
        triaging machine-checked facts, so replaying the whole approved plan here
        would only invite it to re-litigate steps the audit already cleared.
        """
        audit = plan_audit or {}
        lines = [
            f"- Approved steps: {audit.get('n_steps', 0)} "
            f"(evidenced {audit.get('n_completed', 0)}, "
            f"legitimately skipped {audit.get('n_skipped', 0)}, "
            f"no execution evidence {audit.get('n_unmet', 0)}, "
            f"not machine-checkable {audit.get('n_unverifiable', 0)})"
        ]
        unmet = cls._format_review_items(audit.get("unmet_titles"))
        if unmet:
            lines.append("- Approved steps with NO execution evidence:")
            lines.append(unmet)
        for label, key in (
            ("Methods the plan named that nothing ran", "unmet_families"),
            ("Methods evidenced by a successful execution", "executed_families"),
            ("Methods this kernel cannot run (import failure observed)", "unavailable_families"),
        ):
            values = audit.get(key) or []
            if values:
                lines.append(f"- {label}: {', '.join(str(v) for v in values[:_MAX_PROMPT_ITEMS])}")
        admitted = cls._format_review_items(admissions)
        if admitted:
            lines.append("- The analysis agent itself recorded these as not run or failed:")
            lines.append(admitted)
        return "\n".join(lines)
