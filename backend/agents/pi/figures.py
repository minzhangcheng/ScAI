"""Publication-figure review, graphic abstracts, and text diagrams.

Moved verbatim out of backend/agents/pi_agent.py; composed into PIAgent.
"""
from typing import Dict, Any, List, Optional
import asyncio
import logging
import json

from ...utils.formatting import normalize_text_list, safe_parse_json_array
from ...services.figure_layout import format_violations_for_prompt

# Configure logger
logger = logging.getLogger(__name__)
from ...models.schemas import AnalysisState, PublicationFigure
from ...constants import NO_OUTPUT_TOKEN_CAP
from ...utils.llm_client import is_vision_capability_mismatch
from ..vision_replay import replay_safe_vision_call
from .figure_adoption_prompt import build_figure_adoption_prompt

from .figure_text import build_ga_revision_history_text, describe_candidate
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


class FiguresMixin:
    """Publication-figure review, graphic abstracts, and text diagrams."""

    @staticmethod
    def _unavailable_publication_figure_review(reason: str) -> Dict[str, Any]:
        """Describe absence of a visual verdict without inventing a score."""
        feedback = str(reason or "Figure review unavailable").strip()[:500]
        return {
            "approved": False,
            "quality_score": None,
            "review_unavailable": True,
            "feedback": feedback or "Figure review unavailable",
            "strengths": [],
            "issues": [],
            "suggestions": [],
            "must_fix": [],
            "requires_reanalysis": False,
        }

    def _build_revision_history_text(self, figure: PublicationFigure) -> str:
        """Build a text summary of all revision history for the PI review prompt."""
        if figure.revision_history:
            lines = []
            for entry in figure.revision_history:
                round_num = entry.get('round', '?')
                hist_feedback = entry.get('feedback', '')
                lines.append(f"  Round {round_num}: {hist_feedback}")
            return "- Previous revision feedback:\n" + "\n".join(lines)
        elif figure.pi_feedback:
            return f"- Previous feedback: {figure.pi_feedback}"
        return ""

    @staticmethod
    def _figure_pi_decision_text(
        state: AnalysisState,
        figure: PublicationFigure,
    ) -> str:
        """Return later PI guidance applicable to this figure's plan and round."""
        plan_id = str(getattr(figure, "plan_id", "") or "")
        round_number = int(getattr(figure, "round_number", 0) or 0)
        records = []
        for record in getattr(state, "pi_decision_records", None) or []:
            record_plan = str(getattr(record, "plan_id", "") or "")
            record_round = int(getattr(record, "round_number", 0) or 0)
            if plan_id and record_plan != plan_id:
                continue
            if round_number and record_round != round_number:
                continue
            records.append(record)
        lines = []
        for record in records[-8:]:
            lines.append(
                f"- Question: {str(getattr(record, 'question', '') or '')[:1000]}\n"
                f"  PI decision: {str(getattr(record, 'response', '') or '')[:4000]}"
            )
        for task in state.figure_revision_tasks:
            if (task.plan_id or "") != plan_id or not task.pi_instruction:
                continue
            if task.logical_figure_id != figure.logical_figure_id:
                continue
            if task.round_number and task.round_number != round_number:
                continue
            lines.append(f"- PI-directed figure follow-up: {task.pi_instruction[:4000]}")
        if not lines:
            return ""
        return (
            "**Later PI decisions applicable to this plan and round:**\n"
            + "\n".join(lines)
            + "\nTreat these as authoritative refinements of the linked plan. If a "
              "decision conflicts with the original plan text, follow the later PI "
              "decision and disclose any remaining ambiguity."
        )

    async def decide_figure_followups(
        self,
        state: AnalysisState,
        tasks: List[Any],
        candidate_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Give every unresolved figure task an explicit PI disposition."""
        if not tasks:
            return []
        rows = []
        for task in tasks[:12]:
            rows.append(
                f"- task_id={task.id}; logical_figure_id={task.logical_figure_id}; "
                f"requirement={task.plan_requirement_key or 'supplementary'}; "
                f"step={task.plan_step_id}; coverage={task.found_count}/{task.required_count}; "
                f"reason={task.reason}; candidates={candidate_counts.get(task.id, 0)}; "
                f"must_fix={'; '.join(task.must_fix[:6]) or 'none recorded'}"
            )
        plan = state.current_plan
        evidence = {
            "plan": plan.model_dump(mode="json") if plan else None,
            "later_pi_decisions": [
                {"question": record.question, "context": record.context,
                 "decision": record.response}
                for record in state.pi_decision_records
                if record.plan_id == getattr(plan, "id", None)
                and record.round_number == state.current_round
            ][-8:],
            "executions": [
                {"id": item.id, "code": item.code[:1500],
                 "status": item.execution_status}
                for item in state.code_history
                if item.round_number == state.current_round
                and item.plan_id == getattr(plan, "id", None)
            ][-12:],
            "execution_notes": [
                item.model_dump(mode="json", include={
                    "step_type", "code_description", "findings_summary", "error"
                }) for item in state.analysis_log
                if item.round_number == state.current_round
            ][-12:],
        }
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)[:28000]
        prompt = f"""You are the PI closing the publication-figure work for one completed
analysis round. Assign exactly one disposition to every listed task.

Research objective: {state.initial_instruction}
Round: {state.current_round}
Plan: {getattr(getattr(state, 'current_plan', None), 'id', '')}

Available execution evidence and later PI decisions (missing facts are unknown):
{evidence_text}

Unresolved figure tasks:
{chr(10).join(rows)}

Allowed dispositions:
- targeted_redraw: the loaded session can produce or repair the deliverable now.
  Supply one bounded, executable instruction for the main Bioinformatics executor.
  Each task owns one distinct output; several tasks may fulfill the same requirement.
- accept_best: at least one existing candidate is useful enough for the report with
  explicit reservations. The candidates will be shown to you in a separate visual
  selection call; do not pick an id here.
- needs_new_analysis: a new measurement, source load, filtering, model fit, or other
  analytical decision is required before the requested figure can be made. This
  records a later planning request; it does not run analysis now.
- omit: no current candidate should enter the report and targeted work is not useful
  or feasible now. Explain the omission for the report handoff.

Scientific adequacy, usefulness, and whether another analysis is worthwhile are PI
judgments. Preserve limitations. Do not turn a missing measurement into a plotting
task. Use targeted_redraw only when the instruction is executable from the loaded
session and names the required visual output.

Return ONLY a JSON array with one item per task:
[{{"task_id":"...","disposition":"targeted_redraw|accept_best|omit|needs_new_analysis",
"instruction":"required only for targeted_redraw","rationale":"PI reasoning",
"report_note":"concise limitation or omission note"}}]"""
        valid_ids = {task.id for task in tasks[:12]}
        results = []
        seen = set()
        request = prompt
        for attempt in range(2):
            try:
                parsed = safe_parse_json_array(
                    (await self.chat_without_history(request, temperature=0.1)).strip()
                )
            except Exception as exc:
                logger.warning("[PI] figure follow-up disposition failed: %s", exc)
                break
            for item in parsed if isinstance(parsed, list) else []:
                if not isinstance(item, dict):
                    continue
                task_id = str(item.get("task_id") or "")
                disposition = str(item.get("disposition") or "").strip().lower()
                if task_id not in valid_ids or task_id in seen:
                    continue
                if disposition not in {
                    "targeted_redraw", "accept_best", "omit", "needs_new_analysis",
                }:
                    continue
                if disposition == "accept_best" and not candidate_counts.get(task_id, 0):
                    continue
                instruction = str(item.get("instruction") or "").strip()[:4000]
                if disposition == "targeted_redraw" and not instruction:
                    continue
                results.append({
                    "task_id": task_id,
                    "disposition": disposition,
                    "instruction": instruction,
                    "rationale": str(item.get("rationale") or "").strip()[:4000],
                    "report_note": str(item.get("report_note") or "").strip()[:4000],
                })
                seen.add(task_id)
            missing_ids = sorted(valid_ids - seen)
            if not missing_ids or attempt == 1:
                break
            request = (
                prompt + "\n\n"
                "Your previous response did not give a valid disposition for every "
                "task. Return ONLY a JSON array for these omitted task IDs, using the "
                "same schema and decision rules:\n" + "\n".join(missing_ids)
            )
        return results

    async def select_fallback_figure(
        self,
        state: AnalysisState,
        task: Any,
        candidates: List[PublicationFigure],
    ) -> Dict[str, Any]:
        """Visually choose one failed-review candidate and explain its limitations."""
        if not candidates:
            return {}
        import base64
        import io
        from PIL import Image
        from ...services.report.artifacts import verified_artifact_bytes, _preview_bytes

        considered = candidates[-6:]
        shown, images = [], []
        if self.has_vision:
            for candidate in considered:
                try:
                    payload = verified_artifact_bytes(
                        candidate.png_path, candidate, kind="image"
                    ) or _preview_bytes(candidate)
                    if not payload:
                        continue
                    with Image.open(io.BytesIO(payload)) as raw:
                        raw.thumbnail((1200, 1200))
                        buffer = io.BytesIO()
                        raw.convert("RGB").save(buffer, format="JPEG", quality=85)
                    images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
                    shown.append(candidate)
                except Exception:
                    logger.warning(
                        "[PI] fallback candidate %s is unreadable", candidate.id
                    )
        displayed = shown if shown else considered
        rows = []
        for index, candidate in enumerate(displayed, 1):
            status = getattr(candidate.review_status, "value", candidate.review_status)
            scores = [
                item.get("quality_score") for item in candidate.revision_history
                if isinstance(item, dict)
                and isinstance(item.get("quality_score"), (int, float))
            ]
            rows.append(
                f"[image {index}] figure_id={candidate.id}; name={candidate.name}; "
                f"status={status}; revisions={candidate.revision_count}; "
                f"best_recorded_score={max(scores) if scores else 'unavailable'}; "
                f"feedback={str(candidate.pi_feedback or '')[:1200]}; "
                f"unresolved={'; '.join(candidate.unresolved_must_fix[:6])}"
            )
        prompt = f"""Ordinary review did not clear this planned figure deliverable. As PI,
choose the single most useful existing candidate for a report with reservations, or
decline all candidates. Judge the attached images when present together with the
recorded limitations. Do not claim that unresolved issues were fixed.

Task: {task.logical_figure_id}
Required deliverable: {task.plan_requirement_key or 'supplementary figure'}
Candidates:
{chr(10).join(rows)}

Return ONLY JSON:
{{"selected_figure_id":"<id or empty>","rationale":"why this is the best available",
"report_note":"one concise statement of unresolved limitations and permitted use"}}"""
        try:
            if shown:
                self._set_llm_context()
                response = await replay_safe_vision_call(
                    self.llm_client,
                    agent_name="pi",
                    model=self.model,
                    text_prompt=prompt,
                    images=images,
                    temperature=0.1,
                    system_prompt=self.system_prompt,
                    prefer_stream=True,
                )
                response_text = response.content or ""
                valid_ids = {item.id for item in shown}
                selection_basis = "visual PI comparison"
            else:
                response_text = await self.chat_without_history(
                    prompt
                    + "\n\nThe PI model cannot inspect the candidate pixels. Base the "
                      "choice only on recorded review evidence, and state this "
                      "limitation in report_note.",
                    temperature=0.1,
                )
                valid_ids = {item.id for item in considered}
                selection_basis = "recorded review evidence; PI vision unavailable"
            parsed = self._safe_parse_json(response_text.strip(), default={})
        except Exception as exc:
            logger.warning("[PI] fallback figure selection failed: %s", exc)
            return {}
        selected_id = str(parsed.get("selected_figure_id") or "")
        if selected_id not in valid_ids:
            return {}
        report_note = str(parsed.get("report_note") or "").strip()[:4000]
        if not shown:
            report_note = (
                f"Selection basis: {selection_basis}. {report_note}"
            ).strip()[:4000]
        return {
            "selected_figure_id": selected_id,
            "rationale": str(parsed.get("rationale") or "").strip()[:4000],
            "report_note": report_note,
        }

    @staticmethod
    def _figure_plan_requirement_text(
        state: AnalysisState,
        figure: PublicationFigure,
    ) -> str:
        """Show the reviewer the deliverable this render claims to satisfy."""
        plan = getattr(state, "current_plan", None)
        step_id = getattr(figure, "plan_step_id", None)
        if plan is None or step_id is None:
            return "- Plan deliverable: no approved plan-step link recorded"
        step = next(
            (
                candidate for candidate in (getattr(plan, "steps", None) or [])
                if int(getattr(candidate, "id", 0) or 0) == int(step_id)
            ),
            None,
        )
        if step is None:
            return f"- Plan deliverable: linked step {step_id} is absent from the current plan"
        requirements = []
        for requirement in getattr(step, "figure_requirements", None) or []:
            figure_type = str(getattr(requirement, "figure_type", "other") or "other")
            description = str(getattr(requirement, "description", "") or "").strip()
            count = int(getattr(requirement, "minimum_count", 1) or 1)
            requirements.append(
                f"{figure_type} x{count}"
                + (f": {description}" if description else "")
            )
        required_text = "; ".join(requirements) or "legacy/inferred visual deliverable"
        return (
            f"- Linked plan: {getattr(plan, 'id', '')}\n"
            f"- Linked step {step_id}: {getattr(step, 'title', '')}\n"
            f"- Step method: {getattr(step, 'method', '')}\n"
            f"- Required visual output: {required_text}"
        )

    async def review_figure(
        self,
        state: AnalysisState,
        figure: PublicationFigure,
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Review a figure generated by Bioinformatics Agent for publication quality.

        Args:
            state: Current analysis state
            figure: PublicationFigure object with metadata
            image_base64: Base64 encoded PNG image for review

        Returns:
            Dict with:
            - approved: bool - whether the figure passes review
            - feedback: str - overall feedback
            - suggestions: List[str] - specific improvement suggestions
            - quality_score: int - score from 1-10
        """
        logger.info(f"[{self.session_id}][PI] Reviewing figure: {figure.name}")

        # Renderer-measured collisions in the image being reviewed, so the
        # reviewer is not left to infer from a downscaled JPEG what the renderer
        # already knows exactly.
        layout_block = ""
        measured = format_violations_for_prompt(
            getattr(figure, "layout_violations", None)
        )
        if measured:
            layout_block = f"""
**Measured layout defects in this image (from the rendering engine — pixel
overlaps between artists, not matters of taste):**
{measured}

Anything listed as BLOCKING above means text is covered by another element and
cannot be read. List those in `must_fix` and do NOT approve the figure while
they are present, regardless of revision count.
"""

        plan_requirement_block = self._figure_plan_requirement_text(state, figure)
        pi_decision_block = self._figure_pi_decision_text(state, figure)

        # Keep review rules and project context ahead of per-figure changes.
        prompt = f"""## Figure Review for Publication Quality

You are reviewing a figure for publication in a high-impact scientific journal.

**Review Criteria:**
1. **Scientific Accuracy**: Does the figure accurately represent the data?
2. **Clarity**: Is the figure easy to understand? Are labels clear and readable?
3. **Aesthetics**: Is the color scheme appropriate? Is the layout clean?
4. **Completeness**: Are all necessary elements present (legend, axis labels, scale bars, titles)?
5. **Publication Quality**: Would this be acceptable in a high-impact journal (Nature, Cell, etc.)?
6. **Data Fidelity**: Does the figure show ONLY the actual experimental groups listed above? Are there any fabricated or non-existent groups/conditions?
7. **Plan Deliverable Fidelity**: Does the image actually answer the linked plan step and satisfy its required visual output? Do not approve a merely attractive plot of the wrong result.

**Respond in JSON format:**
```json
{{
    "approved": true/false,
    "quality_score": 1-10,
    "feedback": "Detailed assessment: describe EXACTLY what is wrong and what the expected result should look like. Reference specific visual elements (axis labels, color scheme, legend, font size, layout, etc.)",
    "strengths": ["strength 1", "strength 2"],
    "issues": ["Specific issue with exact description, e.g., 'X-axis label font size is too small (appears ~6pt, should be >=10pt)'"],
    "suggestions": ["Actionable code-level fix, e.g., 'Add plt.xlabel(..., fontsize=12)' or 'Change colormap from viridis to tab20 for discrete clusters'"],
    "must_fix": ["Critical issue that MUST be fixed. Be specific: what is wrong AND what it should look like instead"],
    "requires_reanalysis": false
}}
```

**requires_reanalysis**: Set to true ONLY if the issue cannot be fixed by modifying the plotting code alone — e.g., wrong clustering results, incorrect differential expression, data needs re-processing upstream. When true, the figure revision loop will be skipped and the issue will be deferred to the next analysis step.

**Critical review instructions:**
- Your feedback will be used to automatically revise the plotting code. Every issue MUST be described precisely enough for a programmer to fix it.
- BAD feedback: "The figure is not clear enough"
- GOOD feedback: "The cluster labels overlap in the upper-right quadrant. Use adjustText or reduce font size from 12 to 8."
- BAD feedback: "Colors are not good"
- GOOD feedback: "Clusters 3 and 7 use nearly identical shades of blue, making them indistinguishable. Use a qualitative palette like tab20."
- Be strict but fair. Only approve figures that meet publication standards.
- If this is a revision (revision_count > 0), focus on whether previous issues were actually addressed. Be more lenient on minor aesthetic issues after 2+ revisions.
- For UMAP/clustering plots: check color distinctness, label clarity, legend placement.
- For heatmaps: check color scale, row/column labels, clustering dendrograms if applicable.
- For violin/dotplots: check group comparisons, statistical annotations if needed.

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research'}

**Experimental Design (ACTUAL uploaded groups):**
{state.build_experimental_design_summary()}
Group names: {', '.join(g.name for g in state.treatment_groups) if state.treatment_groups else 'No groups defined'}

**CRITICAL CONSTRAINT:** The experiment contains ONLY the groups listed above. Do NOT suggest adding, referencing, or comparing groups that are not in the experimental design. If the figure shows groups not listed above, flag this as a critical data accuracy issue.

**Figure Information:**
- Name: {figure.name}
- Type: {figure.figure_type}
- Description: {figure.description}
- Revision count: {figure.revision_count}
{self._build_revision_history_text(figure)}
{plan_requirement_block}
{pi_decision_block}
{layout_block}
"""

        # Apply progressive leniency for multiple revisions
        if figure.revision_count >= 3:
            prompt += """
**FINAL REVISION - MUST APPROVE unless critical issues exist:**
- This is the final revision attempt. APPROVE if quality_score >= 5.
- Only reject for: fabricated data, completely missing labels, scientific errors, or text that is covered by another element and cannot be read.
- Aesthetic issues at this stage should be noted in feedback but should NOT prevent approval.
"""
        elif figure.revision_count >= 2:
            prompt += """
**IMPORTANT - REVISION LENIENCY POLICY (revision count >= 2):**
- This figure has been revised multiple times. Focus ONLY on critical issues.
- Approve if quality_score >= 6 AND there are no data accuracy/fabrication issues.
- Minor aesthetic preferences (colour choices, exact font sizes, generous or tight spacing) should NOT block approval.
- Leniency does not extend to legibility. Text drawn on top of other text, a legend covering a neighbouring panel's labels, tick labels running into each other, or a label so small it cannot be read are not preferences — the reader cannot recover that information from the figure. Keep rejecting those.
- Otherwise reject only for scientific accuracy issues, missing essential labels, or data fidelity problems.
"""

        # Use vision capability to analyze the figure
        if self.has_vision:
            try:
                self._set_llm_context()
                response = await replay_safe_vision_call(
                    self.llm_client, agent_name="pi", model=self.model,
                    text_prompt=prompt,
                    images=[image_base64],
                    temperature=0.3,
                    system_prompt=self.system_prompt,
                    prefer_stream=True,
                )
                response_text = response.content if response and response.content else ""
            except Exception as e:
                # Only a host-classified capability mismatch changes the
                # session setting.  Transport, quota, and transient upstream
                # failures retain vision for the next call.
                if is_vision_capability_mismatch(e):
                    logger.warning(
                        f"[PI] Vision API capability mismatch, disabling vision for this session: {e}"
                    )
                    self.has_vision = False
                else:
                    logger.warning(f"[PI] Vision API transient failure; retaining vision: {e}")
                return self._unavailable_publication_figure_review(
                    f"PI vision review failed: {str(e)[:400]}"
                )
        else:
            return self._unavailable_publication_figure_review(
                "Figure review unavailable: PI model has no vision capability"
            )

        # Use base class safe JSON parser
        data = self._safe_parse_json(response_text, default={})

        if data and "approved" in data:
            try:
                quality_score = int(data.get("quality_score", 5))
            except (TypeError, ValueError):
                return self._unavailable_publication_figure_review(
                    "Figure review response could not be parsed"
                )
            result = {
                "approved": bool(data.get("approved", False)),
                "quality_score": quality_score,
                "feedback": data.get("feedback", ""),
                "strengths": data.get("strengths", []),
                "issues": data.get("issues", []),
                "suggestions": data.get("suggestions", []),
                "must_fix": normalize_text_list(data.get("must_fix"), limit=20),
                "requires_reanalysis": bool(data.get("requires_reanalysis", False))
            }
            logger.info(f"[PI] Figure review result: approved={result['approved']}, score={result['quality_score']}, requires_reanalysis={result['requires_reanalysis']}")
            return result

        return self._unavailable_publication_figure_review(
            "Figure review response could not be parsed"
        )

    def _build_ga_revision_history_text(self, revision_history: List[Dict[str, Any]]) -> str:
        return build_ga_revision_history_text(revision_history)

    @staticmethod
    def _unavailable_graphic_abstract_review(reason: str) -> Dict[str, Any]:
        """Record that no graphic-abstract visual verdict was obtained."""
        bounded_reason = str(reason or "Graphic abstract review unavailable").strip()[:500]
        bounded_reason = bounded_reason or "Graphic abstract review unavailable"
        return {
            "review_unavailable": True,
            "review_reason": bounded_reason,
            "quality_score": None,
            "feedback": bounded_reason,
            "strengths": [],
            "issues": [],
            "suggestions": [],
            "must_fix": [],
        }

    async def review_graphic_abstract(
        self,
        state: AnalysisState,
        image_base64: str,
        revision_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review a graphic abstract for publication quality using vision API.

        Args:
            state: Current analysis state
            image_base64: Base64 encoded graphic abstract image
            revision_history: All previous rounds' feedback

        Returns:
            Dict with feedback, suggestions, must_fix, quality_score
        """
        logger.info(f"[{self.session_id}][PI] Reviewing graphic abstract (round {len(revision_history) + 1})")

        revision_text = self._build_ga_revision_history_text(revision_history)

        prompt = f"""## Graphic Abstract Review

You are reviewing a graphic abstract for a biomedical research publication.

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:1000] if state.final_summary else 'Not available'}

**Review Round:** {len(revision_history) + 1}
{revision_text}

**Review Criteria:**
1. **Scientific Accuracy**: Does the graphic abstract correctly represent the key findings?
2. **Visual Communication**: Does it tell the research story at a glance?
3. **Layout & Flow**: Is there a clear visual flow (left-to-right or top-to-bottom)?
4. **Labeling**: Are key elements labeled clearly and legibly?
5. **Color Scheme**: Are colors professional and publication-appropriate?
6. **Completeness**: Are the main findings, cell types, pathways, and conclusions represented?
7. **Publication Quality**: Would this be acceptable for a high-impact journal?

**Respond in JSON format:**
```json
{{
    "quality_score": 1-10,
    "feedback": "Detailed assessment of the graphic abstract. Describe what works well and what needs improvement.",
    "strengths": ["strength 1", "strength 2"],
    "issues": ["specific issue 1", "specific issue 2"],
    "suggestions": ["Describe the visual change needed, e.g., 'Add arrows showing the pathway from treatment to outcome' or 'Use distinct colors for different cell types'"],
    "must_fix": ["Critical visual issue that MUST be addressed, e.g., 'The key finding about X pathway is not represented at all'"]
}}
```

**Review instructions:**
- Your feedback will be used by an image generation model to revise the graphic abstract. Describe visual changes clearly.
- BAD: "The figure is not informative enough"
- GOOD: "The graphic abstract should include a visual representation of the <key signaling pathway> connecting <source cell type> to <target cell type> as identified in the analysis"
- BAD: "Colors need improvement"
- GOOD: "Use distinct colors for tumor cells (red/pink) vs immune cells (blue/green) to improve visual differentiation"
{"- Since this is round " + str(len(revision_history) + 1) + ", focus on whether previous issues were addressed. Be more lenient on minor aesthetic issues." if revision_history else ""}
"""

        if not self.has_vision:
            # No vision capability — cannot meaningfully review graphic abstract
            logger.info(f"[PI] Skipping graphic abstract vision review (has_vision=False)")
            return self._unavailable_graphic_abstract_review(
                "Graphic abstract review unavailable: PI model has no vision capability"
            )

        try:
            self._set_llm_context()
            response = await replay_safe_vision_call(
                self.llm_client, agent_name="pi", model=self.model,
                text_prompt=prompt,
                images=[image_base64],
                temperature=0.3,
                system_prompt=self.system_prompt,
                prefer_stream=True,
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[PI] Vision API failed for graphic abstract review: {e}")
            # Cannot meaningfully review without seeing the image
            return self._unavailable_graphic_abstract_review(
                f"Graphic abstract vision review failed: {str(e)[:400]}"
            )

        data = self._safe_parse_json(response_text, default={})

        if not isinstance(data, dict):
            return self._unavailable_graphic_abstract_review(
                "Graphic abstract review response could not be parsed"
            )

        raw_score = data.get("quality_score")
        feedback = data.get("feedback")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, int)
            or not 1 <= raw_score <= 10
            or not isinstance(feedback, str)
            or not feedback.strip()
        ):
            return self._unavailable_graphic_abstract_review(
                "Graphic abstract review response was missing valid score or feedback"
            )

        result = {
            "review_unavailable": False,
            "review_reason": None,
            "quality_score": raw_score,
            "feedback": feedback,
            "strengths": normalize_text_list(data.get("strengths"), limit=20),
            "issues": normalize_text_list(data.get("issues"), limit=20),
            "suggestions": normalize_text_list(data.get("suggestions"), limit=20),
            "must_fix": normalize_text_list(data.get("must_fix"), limit=20),
        }
        logger.info(f"[PI] Graphic abstract review: score={result['quality_score']}")
        return result

    async def select_best_graphic_abstract(
        self,
        state: AnalysisState,
        candidates: List[str],
        revision_history: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Select the best graphic abstract from all candidates using vision API.

        Args:
            state: Current analysis state
            candidates: List of base64 encoded candidate images
            revision_history: All rounds' feedback history

        Returns:
            Index of the best candidate (0-based), or ``None`` when no
            technically valid visual selection was obtained.
        """
        logger.info(f"[{self.session_id}][PI] Selecting best graphic abstract from {len(candidates)} candidates")

        if not candidates:
            return None

        revision_text = self._build_ga_revision_history_text(revision_history)

        prompt = f"""## Select Best Graphic Abstract

You are reviewing {len(candidates)} versions of a graphic abstract for publication.

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:500] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:800] if state.final_summary else 'Not available'}

**Revision History:**
{revision_text if revision_text else 'No previous feedback'}

**Images provided (in order):**
{chr(10).join(f'- Image {i+1}: {"Original" if i == 0 else f"Revision {i}"}' for i in range(len(candidates)))}

**Task:**
Compare ALL {len(candidates)} images and select the ONE that is best suited for journal publication.

Consider:
1. Scientific accuracy and completeness of key findings
2. Visual communication clarity
3. Layout, flow, and visual hierarchy
4. Labeling and readability
5. Professional appearance and color scheme
6. Overall impact and storytelling

**Respond in JSON format:**
```json
{{
    "selected_index": <0-based index of the best image>,
    "reason": "Brief explanation of why this version is the best"
}}
```
"""

        if not self.has_vision:
            logger.info("[PI] Skipping graphic abstract selection (has_vision=False)")
            return None

        try:
            self._set_llm_context()
            response = await replay_safe_vision_call(
                self.llm_client, agent_name="pi", model=self.model,
                text_prompt=prompt,
                images=candidates,
                temperature=0.2,
                system_prompt=self.system_prompt,
                prefer_stream=True,
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[PI] Vision API failed for graphic abstract selection: {e}")
            return None

        data = self._safe_parse_json(response_text, default={})

        if not isinstance(data, dict) or "selected_index" not in data:
            return None
        try:
            idx = data["selected_index"]
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise ValueError("selection index must be an integer")
        except (TypeError, ValueError, OverflowError):
            logger.warning("[PI] Graphic abstract selection response had an invalid index")
            return None
        if 0 <= idx < len(candidates):
            reason = data.get("reason", "No reason provided")
            logger.info(f"[PI] Selected graphic abstract {idx + 1}/{len(candidates)}: {reason}")
            return idx
        logger.warning(f"[PI] Invalid selected_index {idx}")
        return None

    async def generate_graphic_abstract_prompt(
        self,
        state: AnalysisState
    ) -> str:
        """
        Generate a summary prompt for the Painter Agent to create a graphic abstract.

        Args:
            state: Analysis state with all findings

        Returns:
            Summary text for graphic abstract generation
        """
        logger.info(f"[{self.session_id}][PI] Generating graphic abstract prompt")

        # Extract key findings
        all_findings = []
        for result in state.round_results:
            all_findings.extend(result.bioinformatics_findings[:3])
            all_findings.extend(result.pathology_findings[:2])
            all_findings.extend(result.biochemistry_findings[:2])

        prompt = f"""## Generate Graphic Abstract Summary

**Research Project:**
- Name: {state.name}
- Objective: {state.initial_instruction}

**Key Findings:**
{chr(10).join(f'- {f}' for f in all_findings[:10]) if all_findings else "No specific findings"}

**Final Summary:**
{state.final_summary[:1500] if state.final_summary else "Summary not yet generated"}

**Task:**
Write a concise summary (100-150 words, STRICTLY under 200 words) that captures the essence of this research for a graphic abstract.

IMPORTANT: The downstream image generation model has a 30K token input limit.
Keep your summary brief and focused — long summaries will be truncated and lose important context.

The summary should:
1. State the main research question or hypothesis
2. Report the findings the analysis actually established — however many that is. If the study's
   main result is negative or descriptive, say that; do not manufacture discoveries to fill a count
3. Describe the main conclusion and its significance, at the claim level the evidence supports
4. Be written in a way that can guide visual illustration

Draw ONLY on what the supplied results contain. Depending on the study that may include:
- Key populations or groups identified in the supplied analysis
- Important molecular pathways or genes
- Differences between the conditions the design actually compared
- Mechanism, where the results support one
Omit any of these the study did not measure. In particular, do not describe a treatment effect,
a disease mechanism, or a control-vs-treatment comparison unless the experimental design contains
one — the downstream image model will draw whatever this text names, including things that do
not exist.

Return ONLY the summary text, no additional formatting.
"""

        # Try up to 2 times if response is empty
        max_retries = 2
        for attempt in range(max_retries):
            summary = await self.chat(prompt, temperature=0.4)
            if summary and summary.strip():
                return summary.strip()
            logger.warning(f"[PI] Empty response on graphic abstract prompt attempt {attempt + 1}, retrying...")
            self.rollback_last_turn()
            await asyncio.sleep(1)

        # Fallback if all retries failed
        logger.error(f"[PI] Failed to generate graphic abstract prompt after {max_retries} attempts")
        # Generate a basic summary from available data
        findings = []
        for r in state.round_results:
            findings.extend(r.bioinformatics_findings[:2])
            findings.extend(r.pathology_findings[:1])
        findings_text = (
            "; ".join(findings[:5]) if findings
            else "No findings were available to summarize for this figure"
        )
        return f"Study: {state.name}. Objective: {state.initial_instruction[:200]}. Key findings: {findings_text}."

    async def generate_text_diagram_prompt(
        self,
        state: AnalysisState
    ) -> str:
        """
        Generate an instruction for the Bioinformatics Agent to draw an ASCII/Unicode
        text-mode pathway diagram. Used when Painter Agent is disabled or fails.

        Returns empty string on failure so the caller can fall back to a hardcoded template.
        """
        logger.info(f"[{self.session_id}][PI] Generating text diagram prompt")

        all_findings = []
        for result in state.round_results:
            all_findings.extend(result.bioinformatics_findings[:3])
            all_findings.extend(result.pathology_findings[:2])
            all_findings.extend(result.biochemistry_findings[:2])

        prompt = f"""## Generate Text-Mode Diagram Instruction

You are directing a bioinformatics collaborator to draw an ASCII/Unicode flow diagram
(not a real image) that summarizes the key signaling pathways or cellular processes
discovered in this study.

**Research Project:**
- Name: {state.name}
- Objective: {state.initial_instruction[:400] if state.initial_instruction else 'Biomedical research'}

**Key Findings:**
{chr(10).join(f'- {f}' for f in all_findings[:10]) if all_findings else 'No specific findings'}

**Final Summary:**
{state.final_summary[:1200] if state.final_summary else 'Summary not yet generated'}

**Task:**
Write a 300-500 word instruction that tells the bioinformatics agent EXACTLY what to
draw. The output will be rendered inside a <pre> block on a web page, so it must be
pure ASCII/Unicode text.

Your instruction MUST:
1. List the nodes the findings actually name — however many that is. They may be
   ligands, receptors, transcription factors, cell states or phenotypes, but equally
   sample groups, measured quantities, or observed structures; use whatever this study
   produced. Do not add nodes to reach a count
2. Use a directional arrow ONLY where the findings establish direction (a measured
   perturbation, a time course, an explicitly causal result). For descriptive or
   associative findings use grouping, adjacency, or a dashed association link, and
   label direction as unresolved. Drawing a correlation as an arrow asserts a
   causal claim the study did not make
3. Mention only cellular processes supported by the supplied analysis results
4. Use English labels only (non-ASCII characters will break alignment)
5. NOT include drawing symbols — the diagram author will choose them

Return ONLY the instruction text, no markdown headers, no code fences.
"""

        try:
            text = await self.chat_without_history(prompt, temperature=0.4)
        except Exception as e:
            logger.warning(f"[PI] generate_text_diagram_prompt failed: {e}")
            return ""

        return (text or "").strip()

    async def review_text_diagram(
        self,
        state: AnalysisState,
        text_diagram: str,
        revision_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Review an ASCII/Unicode text-mode graphic abstract.

        Pure text, so vision capability is NOT required. Raises on any failure so the
        workflow can fall back to a lower-priority reviewer.
        """
        logger.info(f"[{self.session_id}][PI] Reviewing text diagram (round {len(revision_history) + 1})")

        revision_text = self._build_ga_revision_history_text(revision_history)

        system_prompt = (
            "You are a principal investigator reviewing a text-mode (ASCII/Unicode) "
            "pathway diagram prepared for a publication's graphic abstract. "
            "You read the diagram character by character; it is not an image."
        )

        prompt = f"""## Text Diagram Review

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:400] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:1000] if state.final_summary else 'Not available'}

**Review Round:** {len(revision_history) + 1}
{revision_text}

**Diagram to review:**
```
{text_diagram}
```

**Review Criteria:**
1. Scientific accuracy — do the nodes and arrows match the key findings?
2. Completeness — are critical pathways, cell states, or processes missing?
3. Clarity — is the directional flow (arrows) unambiguous?
4. Layout — do lines align cleanly (watch for mis-aligned arrows, stray characters)?
5. Labels — are node labels in English, short, scientifically precise?

**Respond in JSON:**
```json
{{
    "quality_score": 1-10,
    "feedback": "Plain-English assessment of the diagram",
    "strengths": ["..."],
    "issues": ["..."],
    "suggestions": ["Describe text edits, e.g., 'Add the missing intermediate node between the upstream signal and the observed response'"],
    "must_fix": ["Critical omissions that block publication"]
}}
```

Describe edits in terms of nodes and arrows, NOT visual styling (color, size, etc.).
"""

        response_text = await self.chat_without_history(
            prompt, temperature=0.3, system_prompt=system_prompt
        )
        if not response_text or not response_text.strip():
            raise ValueError("PI text diagram review returned empty response")

        data = self._safe_parse_json(response_text, default={})
        if not data or "feedback" not in data:
            raise ValueError("PI text diagram review did not return parseable JSON with 'feedback'")

        result = {
            "quality_score": int(data.get("quality_score", 5)),
            "feedback": data.get("feedback", ""),
            "strengths": data.get("strengths", []),
            "issues": data.get("issues", []),
            "suggestions": data.get("suggestions", []),
            "must_fix": normalize_text_list(data.get("must_fix"), limit=20)
        }
        logger.info(f"[PI] Text diagram review: score={result['quality_score']}")
        return result

    async def select_best_text_diagram(
        self,
        state: AnalysisState,
        candidates: List[str],
        revision_history: List[Dict[str, Any]]
    ) -> int:
        """
        Select the best text-mode diagram from multiple ASCII candidates.

        Pure text, no vision needed. On failure returns the last (most revised) candidate.
        """
        logger.info(f"[{self.session_id}][PI] Selecting best text diagram from {len(candidates)} candidates")

        revision_text = self._build_ga_revision_history_text(revision_history)

        candidates_block = "\n\n".join(
            f"--- Candidate {i+1} ---\n{c}" for i, c in enumerate(candidates)
        )

        system_prompt = (
            "You are a principal investigator comparing multiple text-mode pathway "
            "diagrams and selecting the best one for a publication graphic abstract."
        )

        prompt = f"""## Select Best Text Diagram

**Research Context:**
Project: {state.name}
Objective: {state.initial_instruction[:400] if state.initial_instruction else 'Biomedical research'}

**Final Summary:**
{state.final_summary[:800] if state.final_summary else 'Not available'}

**Revision History:**
{revision_text if revision_text else 'No previous feedback'}

**Candidates ({len(candidates)} versions):**
{candidates_block}

**Task:**
Select the ONE candidate that is most scientifically accurate, complete, and cleanly
laid out. Prefer later revisions only if they genuinely addressed feedback; otherwise
an earlier version may win.

**Respond in JSON:**
```json
{{
    "selected_index": <0-based index>,
    "reason": "Brief justification"
}}
```
"""

        try:
            response_text = await self.chat_without_history(
                prompt, temperature=0.2, system_prompt=system_prompt
            )
        except Exception as e:
            logger.warning(f"[PI] select_best_text_diagram failed: {e}")
            return len(candidates) - 1

        data = self._safe_parse_json(response_text or "", default={})
        if data and "selected_index" in data:
            idx = int(data["selected_index"])
            if 0 <= idx < len(candidates):
                logger.info(f"[PI] Selected text diagram {idx + 1}/{len(candidates)}")
                return idx

        return len(candidates) - 1

    #: Bounded so one call cannot carry an unbounded image payload. Anything
    #: dropped is logged by name rather than silently truncated.
    MAX_ADOPTION_CANDIDATES = 24

    async def bind_workspace_figures(
        self,
        state: AnalysisState,
        requirements: List[Any],
        candidates: List[Any],
    ) -> List[Any]:
        """Decide which workspace render answers which plan figure requirement.

        This is a vision call by necessity, not by preference. ``adopt_bindings``
        records ACCEPTED_WITH_RESERVATIONS, and ``is_report_eligible`` admits a
        non-diagnostic record only at a reviewed status -- so the attestation has
        to be real. Seeing the images is also the only dependable way to tell a
        superseded draft from the render that replaced it; filenames must never
        be read for that (see ``PublicationFigure.logical_figure_id``).

        Returns ``[]`` when vision is unavailable or the call fails. An empty
        result simply leaves every requirement missing, which is the honest
        outcome and exactly what the audit already reports today.
        """
        from backend.services.figure_adoption import AdoptionCandidate, FigureBinding
        from backend.utils.image_utils import compress_image_to_base64

        if not requirements or not candidates:
            return []
        if not self.has_vision:
            logger.info(
                "[PI] bind_workspace_figures skipped: no vision model, so no "
                "review can be attested and nothing may be adopted"
            )
            return []

        offered = list(candidates)[: self.MAX_ADOPTION_CANDIDATES]
        if len(candidates) > len(offered):
            dropped = [c.name for c in list(candidates)[len(offered):]]
            logger.warning(
                "[PI] bind_workspace_figures capped at %d candidates; not shown: %s",
                self.MAX_ADOPTION_CANDIDATES, ", ".join(dropped),
            )

        images: List[str] = []
        shown: List[Any] = []
        for candidate in offered:
            try:
                images.append(compress_image_to_base64(candidate.path, max_width=900))
                shown.append(candidate)
            except Exception:
                logger.warning(
                    "[PI] could not read candidate %r for review; not offered",
                    candidate.path, exc_info=True,
                )
        if not shown:
            return []

        prompt = build_figure_adoption_prompt(state, requirements, shown)

        try:
            self._set_llm_context()
            response = await replay_safe_vision_call(
                self.llm_client, agent_name="pi", model=self.model,
                text_prompt=prompt,
                images=images,
                temperature=0.1,
                system_prompt=self.system_prompt,
                prefer_stream=True,
            )
            text = (response.content or "") if response else ""
        except Exception as e:
            logger.warning(f"[PI] bind_workspace_figures vision call failed: {e}")
            return []

        parsed = safe_parse_json_array(text.strip())
        if not isinstance(parsed, list):
            logger.warning("[PI] bind_workspace_figures returned no usable JSON array")
            return []

        bindings: List[Any] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                step = int(item["plan_step_id"])
            except (KeyError, TypeError, ValueError):
                continue
            filename = str(item.get("filename") or "").strip()
            figure_type = str(item.get("figure_type") or "").strip()
            if not filename or not figure_type:
                continue
            bindings.append(FigureBinding(
                plan_step_id=step,
                figure_type=figure_type,
                filename=filename,
                caption=str(item.get("caption") or "").strip(),
            ))
        logger.info(
            "[PI] bind_workspace_figures proposed %d binding(s) over %d candidate(s)",
            len(bindings), len(shown),
        )
        return bindings

    _describe_candidate = staticmethod(describe_candidate)

    async def select_brief_figures(
        self,
        state: AnalysisState,
        all_figures: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Select figure names to display in the Brief Report (Result page).
        Uses AI judgment based on figure names/types and analysis context.

        Args:
            state: Analysis state (for research context)
            all_figures: All available figures from workspace, each with "name" and "type"

        Returns:
            List of figure names that should appear in the brief report
        """
        if not all_figures:
            return []

        # Render every field the caller supplies, defensively: callers that pass a
        # bare name/type dict must keep working, so nothing here is subscripted.
        figures_list = "\n".join(
            self._describe_candidate(f) for f in all_figures
        )
        rounds_present = sorted({
            int(f["round"]) for f in all_figures
            if f.get("round") and not f.get("diagnostic")
        })
        coverage_line = (
            f"\nThis analysis produced result figures in rounds: "
            f"{', '.join(str(r) for r in rounds_present)}."
            if len(rounds_present) > 1 else ""
        )

        prompt = f"""You are reviewing figures generated during a bioinformatics analysis.

Research objective: {state.initial_instruction}

All available figures:
{figures_list}
{coverage_line}

Select which figures should appear in the Brief Results display (shown to the researcher immediately after analysis).
Include figures that are:
- Key results: cell type UMAP with annotations, differential expression volcano plots, cell composition comparisons, key marker dotplots/violins
- Essential visualizations that directly answer the research question

Exclude figures that are:
- QC / technical diagnostics (qc_violin, pca_variance, hvg_plot, and similar) — EXCEPT keep
  a single quality-control overview panel when one exists, since the composition of the
  data every later result rests on is itself a result the researcher needs to see
- Intermediate/exploratory plots superseded by a final version (e.g. exclude umap_initial if umap_cell_types exists)
- Redundant views of the same scientific result. The caller has already resolved
  explicit logical-figure lineage; never infer revision order from filename suffixes.

How many:
- Select 8-15 figures when that many qualify. Select fewer only when fewer qualify.
- Cover the whole analysis: include at least one figure from every round that produced result figures.
- Where two figures show the same result, prefer the one from the later round.

Return ONLY a JSON array of the selected figure names, with no other text.
Example: ["umap_cell_types", "cell_type_proportions", "volcano_<celltype>_<condition>_vs_<control>"]
"""

        try:
            self._set_llm_context()
            response = await self._budgeted_text_call(
                self.llm_client.chat,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
                prefer_stream=True,
            )
            text = (response.content or "").strip()
            parsed = safe_parse_json_array(text)
            if isinstance(parsed, list):
                # Only keep names that are real candidates. A hallucinated or
                # json-repaired name would otherwise suppress the reporting
                # publication fallback and empty the Results gallery.
                valid_names = {f["name"] for f in all_figures}
                names = [
                    x for x in parsed
                    if isinstance(x, str) and x in valid_names
                ]
                if names:
                    return names
            logger.warning("[PI] select_brief_figures returned no usable names; using bounded fallback")
        except Exception as e:
            logger.warning(f"[PI] select_brief_figures failed: {e}")

        # Bounded, deterministic fallback -- never dump every candidate.
        return _bounded_brief_fallback(all_figures)
