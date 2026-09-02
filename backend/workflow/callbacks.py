"""Domain callbacks invoked by sub-agents (literature search, PI consult, figure review)."""
import logging
from typing import Any, Dict, List, Optional

from ..constants import PLAN_REVISION_NEEDED_TAG, NO_OUTPUT_TOKEN_CAP, StateSyncChangeType, WaitInputReason
from ..models.schemas import AgentType, AnalysisState, ClusterAnnotation, PublicationFigure, TaskStatus
from ..utils.formatting import safe_parse_json
from ..services.report.artifacts import has_blocking_layout_defects
from .helpers import wait_with_skip_time

logger = logging.getLogger(__name__)


def _unavailable_figure_review(reason: str) -> Dict[str, Any]:
    """Return a bounded factual result when no visual verdict was obtained."""
    feedback = str(reason or "Figure review unavailable").strip()[:500]
    return {
        "approved": False,
        "quality_score": None,
        "review_unavailable": True,
        "feedback": feedback or "Figure review unavailable",
        "suggestions": [],
        "must_fix": [],
    }


def parse_plan_revision_signal(response_text: str) -> bool:
    text = response_text or ""
    if PLAN_REVISION_NEEDED_TAG in text:
        return True
    parsed = safe_parse_json(text)
    if isinstance(parsed, dict):
        return parsed.get("plan_revision_needed") is True
    return False


def compact_figure_event_payload(
    figure: PublicationFigure,
    *,
    review_status: Optional[str] = None,
    pi_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the live-event figure record without its potentially huge preview.

    Review events used to send only ``figure_id`` and ``figure_name`` at the top
    level.  The Results gallery consumes a nested figure record, so a newly
    rendered figure remained invisible until a later session reload or the
    integrated-report event.  Keep the nested record compact, but carry the
    approved-plan provenance needed to explain why the figure exists.
    """
    raw_status = review_status
    if raw_status is None:
        status = getattr(figure, "review_status", None)
        raw_status = str(getattr(status, "value", status) or "pending")
    return {
        "id": figure.id,
        "name": figure.name,
        "description": figure.description,
        "figure_type": figure.figure_type,
        "png_path": figure.png_path,
        "pdf_path": figure.pdf_path,
        "review_status": raw_status,
        "review_unavailable": bool(
            getattr(figure, "review_unavailable", False)
        ),
        "pi_degraded_acceptance": bool(
            getattr(figure, "pi_degraded_acceptance", False)
        ),
        "pi_degraded_acceptance_reason": getattr(
            figure, "pi_degraded_acceptance_reason", None
        ),
        "pi_feedback": (
            pi_feedback if pi_feedback is not None
            else getattr(figure, "pi_feedback", None)
        ),
        "revision_count": figure.revision_count,
        "plan_step_id": getattr(figure, "plan_step_id", None),
        "plan_id": getattr(figure, "plan_id", None),
        "plan_requirement_key": getattr(figure, "plan_requirement_key", None),
        "submitted_plan_step_id": getattr(
            figure, "submitted_plan_step_id", None
        ),
        "supervisor_binding_decision": getattr(
            figure, "supervisor_binding_decision", None
        ),
        "source_execution_id": getattr(figure, "source_execution_id", None),
        "purpose": getattr(figure, "purpose", "result") or "result",
        "round": getattr(figure, "round_number", 0) or None,
        "diagnostic": bool(getattr(figure, "diagnostic", False)),
        "layout_blocking": has_blocking_layout_defects(figure),
        "created_at": figure.created_at.isoformat(),
    }


class CallbacksMixin:
    """Mixin providing callback methods invoked by sub-agents.

    Composed into SingleCellAnalysisWorkflow. Relies on self.session_id,
    self.pi_agent, self.literature_agent, self.bioinformatics_agent,
    self.supervisor_agent, self._current_analysis_state, self._current_graph_state,
    self._vision_settings, self._settings_provider, and messaging methods.
    """

    async def _literature_callback(self, topic: str) -> str:
        """Callback for bioinformatics agent to request literature support"""
        if self._current_analysis_state is None:
            raise RuntimeError(
                "_current_analysis_state must be set before _literature_callback; "
                "this fallback was previously a silent ValidationError because user_id is required."
            )
        state = self._current_analysis_state
        self.literature_agent.refresh_runtime_config_if_needed()
        result = await self.literature_agent.process(
            state,
            topic,
            use_deep_research=True
        )
        return result.get("report", "")

    async def _runtime_persistence_commit_callback(
        self, event: Dict[str, Any]
    ) -> None:
        """Publish a committed NotebookState generation into session state."""
        if str(event.get("session_id") or "") != self.session_id:
            logger.error(
                "[%s] Ignored runtime projection for foreign session %s",
                self.session_id,
                event.get("session_id"),
            )
            return
        analysis_state = getattr(self, "_current_analysis_state", None)
        if analysis_state is None:
            analysis_state = getattr(self, "_state", None)
        if analysis_state is None:
            # The sidecar remains authoritative. A later session load projects
            # it even when no live workflow state is available during shutdown.
            return

        persistence = dict(event.get("runtime_persistence") or {})
        event_generation = int(event.get("lifecycle_generation") or 0)
        state_generation = int(
            getattr(analysis_state, "lifecycle_generation", 0) or 0
        )
        if (
            event_generation
            and state_generation
            and event_generation != state_generation
        ):
            from ..services.runtime_persistence_gate import (
                enter_runtime_persistence_gate,
            )

            mismatch = dict(
                getattr(analysis_state, "runtime_persistence", None) or {}
            )
            mismatch.update({
                "status": "failed",
                "requires_recompute": True,
                "recommended_action": "restore_matching_lifecycle",
                "error_summary": (
                    f"Runtime commit lifecycle generation {event_generation} "
                    f"does not match session generation {state_generation}."
                ),
                "projection_status": "lifecycle_mismatch",
            })
            enter_runtime_persistence_gate(analysis_state, mismatch)
            await self._send_state_sync(
                analysis_state,
                StateSyncChangeType.RUNTIME_PERSISTENCE_FAILURE.value,
            )
            return

        analysis_state.runtime_persistence = persistence
        sandbox_state = dict(analysis_state.sandbox_state or {})
        sandbox_state.update({
            "runtime_revision": int(event.get("current_revision") or 0),
            "saved_runtime_revision": int(event.get("saved_revision") or 0),
            "runtime_persistence": persistence,
        })
        analysis_state.sandbox_state = sandbox_state
        await self._send_state_sync(
            analysis_state,
            StateSyncChangeType.RUNTIME_PERSISTENCE_COMMITTED.value,
        )

    async def _pi_callback(self, question: str, context: str) -> str:
        """Callback for bioinformatics agent to consult PI for biological decisions"""
        await self._send_message("status", "Bioinformatics consulting PI...", AgentType.BIOINFORMATICS)
        await self._send_message("pi_question", f"Question: {question}\nContext: {context}", AgentType.BIOINFORMATICS)

        # Build plan context if an approved plan exists
        plan_context = ""
        analysis_state = getattr(self, '_current_analysis_state', None)
        if analysis_state and analysis_state.current_plan and analysis_state.current_plan.status == "approved":
            plan = analysis_state.current_plan
            steps_summary = "\n".join(
                f"  [{s.status}] Step {s.id}: {s.title}"
                for s in plan.steps
            )
            plan_context = f"""

## Current Analysis Plan (for reference)
{steps_summary}

If the question reveals that the current plan cannot be executed as-is, you should:
1. First, determine if this is a MINOR adjustment (method switch, parameter change, skip step)
   or MAJOR revision (hypothesis change, new analysis branch, core objective removed)
2. For MINOR: Provide direct actionable guidance
3. For MAJOR: State clearly that a plan revision is needed and explain why
"""
        # Build stage context so PI knows the agent's current constraints
        stage_context = ""
        graph_state = self._current_graph_state
        if graph_state:
            ps = graph_state.get("planning_stage", "full")
            if ps == "stage1":
                stage_context = (
                    "\n\nCURRENT STAGE: Stage 1 (QC + Clustering + Cell Type Annotation ONLY). "
                    "The bioinformatics agent should ONLY perform basic QC, clustering, and annotation. "
                    "Downstream analyses (DEG, GO/KEGG, cell communication, trajectory) are reserved "
                    "for Stage 2 and should NOT be performed now. "
                    "If the agent asks about downstream analysis, advise them to complete Stage 1 first "
                    "by calling report_findings."
                )
            elif ps == "stage2":
                stage_context = (
                    "\n\nCURRENT STAGE: Stage 2 (Downstream Analysis). "
                    "Stage 1 (QC + clustering + annotation) has been completed. "
                    "The agent is now executing the approved Stage 2 analysis plan."
                )

        # Use PI agent to answer the biological question
        prompt = f"""The bioinformatics analyst is asking for your guidance on a biological decision:

QUESTION: {question}

CONTEXT FROM ANALYSIS:
{context}
{plan_context}
{stage_context}

Please provide clear guidance on:
1. Your recommended decision/direction
2. The reasoning behind your recommendation
3. Any specific parameters, genes, or pathways to focus on
4. Potential pitfalls to avoid

Be specific and actionable in your response.

IMPORTANT: Respond in English only. Do not use any other language."""

        try:
            response = await self.pi_agent.chat_without_history(prompt, temperature=self._settings_provider.get_temperature('pi'))
            await self._send_message("pi_response", response, AgentType.PI)

            # Check if plan revision is needed based on PI's response
            if analysis_state and analysis_state.current_plan and analysis_state.current_plan.status == "approved":
                if parse_plan_revision_signal(response):
                    # PI hints plan has issues — further judge severity
                    plan = analysis_state.current_plan
                    affected_ids = []
                    for step in plan.steps:
                        if step.status != "completed":
                            step_keywords = [step.title.lower(), step.method.lower()]
                            if any(kw in question.lower() or kw in context.lower() for kw in step_keywords if len(kw) > 3):
                                affected_ids.append(step.id)
                    if not affected_ids:
                        pending = [s.id for s in plan.steps if s.status != "completed"]
                        if pending:
                            affected_ids = [pending[0]]

                    try:
                        judgment = await self.pi_agent.judge_plan_deviation(
                            deviation_description=f"QUESTION: {question}\nCONTEXT: {context}",
                            current_plan=plan,
                            affected_step_ids=affected_ids,
                        )

                        if judgment.get("needs_user_popup"):
                            response += "\n\n" + PLAN_REVISION_NEEDED_TAG
                            # Pre-major-revision snapshot is handled by the
                            # post-PI-synthesis snapshot in core.py. No need
                            # for a dedicated snapshot here.
                    except Exception as e:
                        logger.warning(f"[{self.session_id}] Plan deviation judgment failed: {e}")

            return response
        except Exception as e:
            error_msg = f"PI consultation failed: {str(e)}"
            await self._send_message("error", error_msg, AgentType.PI)
            return "Unable to get PI guidance. Please proceed with standard bioinformatics practices."

    async def _figure_review_callback(self, figure: PublicationFigure) -> Dict[str, Any]:
        """
        Callback for bioinformatics agent to get PI review of generated figures.
        Routes to different reviewers based on vision capability settings.

        Args:
            figure: PublicationFigure to review

        Returns:
            Dict with review results (approved, feedback, suggestions)
        """
        pi_vision = self._vision_settings.get('pi', True)
        bio_vision = self._vision_settings.get('bioinformatics', True)
        self.ensure_agent_ready("supervisor")
        sup_vision = (
            self._vision_settings.get('supervisor', True)
            if self.supervisor_agent else False
        )

        # This callback supplies a review opinion only.  The producer owns the
        # final state because deterministic layout checks can legitimately
        # overrule an approving reviewer.  Emit a pending event here so the UI
        # can show that review started, then let ``FiguresMixin`` publish the
        # authoritative approved/revision event after all gates have run.
        await self._send_message(
            "figure_review",
            {
                "figure_id": figure.id,
                "figure_name": figure.name,
                "figure_type": figure.figure_type,
                "description": figure.description,
                "revision_count": figure.revision_count,
                "status": "reviewing",
                "figure": compact_figure_event_payload(
                    figure, review_status="pending"
                ),
            },
            AgentType.PI,
        )

        # 4.1: All models in the chain lack vision → no review verdict exists
        if not pi_vision and not bio_vision and not sup_vision:
            logger.info(
                f"[{self.session_id}] No vision-capable figure reviewer: {figure.name}"
            )
            return _unavailable_figure_review(
                "Figure review unavailable: no vision-capable model in pipeline"
            )

        # 4.3.2: PI has no vision
        if not pi_vision:
            if bio_vision:
                # Bioinformatics agent self-reviews its own figure
                logger.info(f"[{self.session_id}] PI lacks vision, bio agent self-reviewing figure: {figure.name}")
                review = await self._bio_self_figure_review(figure)
            elif sup_vision and self.supervisor_agent:
                # 4.3.3: Both PI and bio lack vision → supervisor guides review
                logger.info(f"[{self.session_id}] PI and bio lack vision, supervisor reviewing figure: {figure.name}")
                review = await self._supervisor_figure_review(figure)
            else:
                review = _unavailable_figure_review(
                    "Figure review unavailable: no vision-capable reviewer"
                )

            return review

        # Default: PI reviews with vision (existing flow)
        logger.info(f"[{self.session_id}] PI reviewing figure: {figure.name}")

        if self._current_analysis_state is None:
            raise RuntimeError(
                "_current_analysis_state must be set before figure review callback; "
                "this fallback was previously a silent ValidationError because user_id is required."
            )
        state = self._current_analysis_state

        try:
            return await self.pi_agent.review_figure(
                state,
                figure,
                figure.png_base64
            )
        except Exception as exc:
            logger.warning(
                "[%s] PI figure review failed: %s", self.session_id, exc
            )
            return _unavailable_figure_review(
                f"PI vision review failed: {str(exc)[:400]}"
            )

    async def _figure_plan_binding_callback(
        self, state: AnalysisState, proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask the Supervisor to judge semantic binding before rendering."""
        self.ensure_agent_ready("supervisor")
        if self.supervisor_agent is None:
            return {
                "decision": "reject",
                "plan_requirement_key": None,
                "reason": "Supervisor is unavailable for semantic plan binding.",
                "parse_failed": True,
            }
        return await self.supervisor_agent.review_figure_plan_binding(
            state, proposal
        )

    async def _vision_figure_review(
        self,
        agent,
        figure: PublicationFigure,
        prompt: str,
        image_b64: str,
        temperature: float = 0.3,
        label: str = "agent",
        **vision_kwargs,
    ) -> Dict[str, Any]:
        """Shared helper: call an agent's vision API to review a figure and parse the result."""
        try:
            agent._set_llm_context()
            response = await agent.llm_client.chat_with_vision(
                model=agent.model,
                text_prompt=prompt,
                images=[image_b64],
                temperature=temperature,
                prefer_stream=True,
                **vision_kwargs,
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[{self.session_id}] {label} figure review vision failed: {e}")
            return _unavailable_figure_review(
                f"{label} vision review failed: {str(e)[:400]}"
            )

        if hasattr(agent, '_safe_parse_json'):
            data = agent._safe_parse_json(response_text, default={})
        elif hasattr(agent, '_parse_supervisor_response'):
            data = agent._parse_supervisor_response(response_text) if response_text else {}
        else:
            data = {}
        if data and ("approved" in data or "quality_score" in data):
            try:
                score = int(data.get("quality_score", 7))
            except (TypeError, ValueError):
                return _unavailable_figure_review(
                    "Figure review response could not be parsed"
                )
            return {
                "approved": data.get("approved", score >= 7),
                "quality_score": score,
                "feedback": data.get("feedback", ""),
                "suggestions": data.get("suggestions", []),
                "must_fix": data.get("must_fix", []),
                "requires_reanalysis": data.get("requires_reanalysis", False),
            }
        return _unavailable_figure_review(
            "Figure review response could not be parsed"
        )

    async def _bio_self_figure_review(self, figure: PublicationFigure) -> Dict[str, Any]:
        """Bioinformatics agent self-reviews its own figure using vision."""
        prompt = f"""## Figure Self-Review for Publication Quality

You generated this figure and now need to review it for publication quality.

**Figure Information:**
- Name: {figure.name}
- Type: {figure.figure_type}
- Description: {figure.description}
- Revision count: {figure.revision_count}

**Review Criteria:**
1. Scientific Accuracy: Does the figure accurately represent the data?
2. Clarity: Are labels clear and readable?
3. Aesthetics: Is the color scheme appropriate and layout clean?
4. Completeness: Are all necessary elements present (legend, axis labels, titles)?
5. Publication Quality: Would this be acceptable in a high-impact journal?

**Respond in JSON format:**
```json
{{
    "approved": true/false,
    "quality_score": 1-10,
    "feedback": "Detailed assessment",
    "suggestions": ["suggestion 1"],
    "must_fix": ["critical issue"],
    "requires_reanalysis": false
}}
```
Be strict but fair. Approve if quality_score >= 7."""

        return await self._vision_figure_review(
            self.bioinformatics_agent, figure, prompt, figure.png_base64,
            temperature=0.3, label="Bio self-review",
        )

    async def _supervisor_figure_review(self, figure: PublicationFigure) -> Dict[str, Any]:
        """Supervisor agent reviews figure using vision.

        When the supervisor is the *only* reviewer (PI and bio both lack vision),
        it uses a relaxed approval threshold (score >= 5) instead of the normal 7,
        since a small model shouldn't block the pipeline.  Suggestions are still
        forwarded as non-blocking feedback.

        Retry exhaustion is NOT decided here.  The producer's own review loop
        (``figures.py``: ``revision_count = 0`` then ``while revision_count <= 3``)
        owns it, and it is the only scope that can: this method is re-entered per
        render, so a counter kept on the orchestrator outlives the loop it would
        bound and merely starves later renders of revisions they never used.
        A low visual score is also not evidence about the upstream analysis, so
        it is never promoted to ``requires_reanalysis`` -- that stamps the figure
        REVISION_REQUESTED, which is not report-eligible, and the figure can then
        never satisfy the plan's visual contract.
        """
        compressed = self.supervisor_agent._compress_base64_figure(figure.png_base64)

        # Determine if supervisor is the sole reviewer (PI and bio both lack vision)
        pi_vision = self._vision_settings.get('pi', True)
        bio_vision = self._vision_settings.get('bioinformatics', True)
        supervisor_is_sole_reviewer = not pi_vision and not bio_vision

        prompt = f"""## Figure Quality Review

Review this figure for publication quality.
- Name: {figure.name}
- Type: {figure.figure_type}
- Description: {figure.description}

Check for: blank/empty plots, missing labels, overlapping text, poor color choices,
missing legends, unreadable fonts. Score 1-10.

Respond in JSON:
```json
{{
    "approved": true/false,
    "quality_score": 1-10,
    "feedback": "assessment",
    "suggestions": ["suggestion"],
    "must_fix": ["critical issue"]
}}
```
Approve if quality_score >= 7."""

        review = await self._vision_figure_review(
            self.supervisor_agent, figure, prompt, compressed,
            temperature=0.2, label="Supervisor", max_tokens=NO_OUTPUT_TOKEN_CAP,
        )

        if review.get("review_unavailable") is True:
            return review

        # --- Supervisor-as-sole-reviewer degradation logic ---
        if supervisor_is_sole_reviewer and not review.get("approved", False):
            score = review.get("quality_score", 0)

            # Relaxed threshold: score >= 5 is acceptable when supervisor is the
            # only reviewer, since the small model's judgment is less reliable.
            if score >= 5:
                logger.info(
                    f"[{self.session_id}] Supervisor-only mode: accepting figure "
                    f"{figure.name} at score={score} (relaxed threshold)"
                )
                review["approved"] = True
                review["feedback"] = (
                    f"[Accepted with relaxed threshold (supervisor-only mode, score={score}/10)] "
                    + review.get("feedback", "")
                )
            # Below the relaxed threshold the review stays an ordinary
            # rejection and the producer loop decides what to do with it.

        return review

    async def _species_validation_callback(
        self,
        gene_names: List[str],
        species: str,
        assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """HITL callback for gene-species mismatch. Returns action dict."""
        analysis_state = self._current_analysis_state
        if analysis_state is None:
            return {"action": "keep", "user_input": ""}

        analysis_state.species_input_info = {
            "source": "gene_validation",
            "species": species,
            "sample_genes": gene_names[:20],
            "pi_assessment": assessment.get("reasoning", ""),
            "likely_species": assessment.get("likely_species", ""),
            "suggestions": [
                f"Change species to {assessment.get('likely_species', 'unknown')}",
                "Keep current species and continue",
            ],
        }

        original_status = analysis_state.status
        try:
            self._species_event.clear()
            self._species_user_response = None
            analysis_state.wait_input_reason = WaitInputReason.GENE_SPECIES_MISMATCH.value
            self._set_status(analysis_state, TaskStatus.WAITING_SPECIES_INPUT)
            await self._send_state_sync(analysis_state, StateSyncChangeType.GENE_SPECIES_MISMATCH.value)

            signaled = await wait_with_skip_time(
                self._species_event,
                analysis_state.round_wait_seconds,
                self.session_id,
                "gene_species_mismatch",
                safe_point_hook=getattr(self, "await_restore_safe_point", None),
            )

            if not signaled:
                analysis_state.species_input_info = None
                analysis_state.wait_input_reason = None
                self._set_status(analysis_state, original_status)
                await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_AUTO_SKIPPED.value)
                return {"action": "keep", "user_input": ""}

            user_input = self._species_user_response or ""
            analysis_state.species_input_info = None
            analysis_state.wait_input_reason = None
            self._set_status(analysis_state, original_status)
            await self._send_state_sync(
                analysis_state,
                "species_input_received",
            )

            try:
                action = await self.pi_agent.decide_species_action(user_input, species, assessment)
            except Exception:
                action = {"action": "keep", "user_input": user_input}

            if action.get("action") == "change_species" and action.get("new_species"):
                analysis_state.species = action["new_species"]
                analysis_state.species_resolved = True
                await self._send_message(
                    "status",
                    f"Species updated to {action['new_species']} (per user input)",
                    AgentType.PI,
                )

            await self._send_state_sync(
                analysis_state,
                StateSyncChangeType.SPECIES_RESOLVED.value,
            )
            return {**action, "user_input": user_input}
        finally:
            analysis_state.species_input_info = None
            analysis_state.wait_input_reason = None
            self._set_status(analysis_state, original_status)
