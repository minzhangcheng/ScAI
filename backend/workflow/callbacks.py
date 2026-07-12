"""Domain callbacks invoked by sub-agents (literature search, PI consult, figure review)."""
import logging
from typing import Any, Dict, List, Optional

from ..constants import PLAN_REVISION_NEEDED_TAG, REASONING_MODEL_MIN_TOKENS, StateSyncChangeType, WaitInputReason
from ..models.schemas import AgentType, AnalysisState, ClusterAnnotation, PublicationFigure, TaskStatus
from ..utils.formatting import safe_parse_json
from .helpers import wait_with_skip_time

logger = logging.getLogger(__name__)


def parse_plan_revision_signal(response_text: str) -> bool:
    text = response_text or ""
    if PLAN_REVISION_NEEDED_TAG in text:
        return True
    parsed = safe_parse_json(text)
    if isinstance(parsed, dict):
        return parsed.get("plan_revision_needed") is True
    return False


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

        # 4.1: All models in the chain lack vision → skip image review
        if not pi_vision and not bio_vision and not sup_vision:
            logger.info(f"[{self.session_id}] All agents lack vision, auto-approving figure: {figure.name}")
            review = {
                "approved": True,
                "quality_score": 7,
                "feedback": "Auto-approved (no vision-capable model in pipeline)",
                "suggestions": [],
                "must_fix": [],
            }
            await self._send_message(
                "figure_approved",
                {
                    "figure_id": figure.id,
                    "figure_name": figure.name,
                    "approved": True,
                    "quality_score": 7,
                    "feedback": review["feedback"],
                    "suggestions": [],
                    "must_fix": []
                },
                AgentType.PI
            )
            return review

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
                # No one can see the figure — auto-approve
                review = {
                    "approved": True,
                    "quality_score": 7,
                    "feedback": "Auto-approved (no vision-capable reviewer available)",
                    "suggestions": [],
                    "must_fix": [],
                }

            await self._send_message(
                "figure_approved" if review.get("approved") else "figure_revision",
                {
                    "figure_id": figure.id,
                    "figure_name": figure.name,
                    "approved": review.get("approved", False),
                    "quality_score": review.get("quality_score", 5),
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", [])
                },
                AgentType.BIOINFORMATICS if bio_vision else AgentType.SUPERVISOR
            )
            return review

        # Default: PI reviews with vision (existing flow)
        logger.info(f"[{self.session_id}] PI reviewing figure: {figure.name}")

        await self._send_message(
            "figure_review",
            {
                "figure_id": figure.id,
                "figure_name": figure.name,
                "figure_type": figure.figure_type,
                "description": figure.description,
                "revision_count": figure.revision_count,
                "status": "reviewing"
            },
            AgentType.PI
        )

        if self._current_analysis_state is None:
            raise RuntimeError(
                "_current_analysis_state must be set before figure review callback; "
                "this fallback was previously a silent ValidationError because user_id is required."
            )
        state = self._current_analysis_state

        review = await self.pi_agent.review_figure(
            state,
            figure,
            figure.png_base64
        )

        await self._send_message(
            "figure_approved" if review.get("approved") else "figure_revision",
            {
                "figure_id": figure.id,
                "figure_name": figure.name,
                "approved": review.get("approved", False),
                "quality_score": review.get("quality_score", 5),
                "feedback": review.get("feedback", ""),
                "suggestions": review.get("suggestions", []),
                "must_fix": review.get("must_fix", [])
            },
            AgentType.PI
        )

        return review

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
                **vision_kwargs,
            )
            response_text = response.content if response and response.content else ""
        except Exception as e:
            logger.warning(f"[{self.session_id}] {label} figure review vision failed: {e}")
            return {
                "approved": True,
                "quality_score": 7,
                "feedback": f"{label} vision review failed, auto-approved",
                "suggestions": [],
                "must_fix": [],
            }

        if hasattr(agent, '_safe_parse_json'):
            data = agent._safe_parse_json(response_text, default={})
        elif hasattr(agent, '_parse_supervisor_response'):
            data = agent._parse_supervisor_response(response_text) if response_text else {}
        else:
            data = {}
        if data and ("approved" in data or "quality_score" in data):
            score = int(data.get("quality_score", 7))
            return {
                "approved": data.get("approved", score >= 7),
                "quality_score": score,
                "feedback": data.get("feedback", ""),
                "suggestions": data.get("suggestions", []),
                "must_fix": data.get("must_fix", []),
                "requires_reanalysis": data.get("requires_reanalysis", False),
            }
        return {
            "approved": True,
            "quality_score": 7,
            "feedback": "Could not parse review response, auto-approved",
            "suggestions": [],
            "must_fix": [],
        }

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

        A per-figure rejection counter (_sup_rejections) tracks consecutive
        rejections.  After 2 consecutive rejections of the same figure, the
        review auto-sets ``requires_reanalysis=True`` so the bioinformatics
        agent breaks out of the revision loop instead of retrying endlessly.
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
            temperature=0.2, label="Supervisor", max_tokens=REASONING_MODEL_MIN_TOKENS,  # reasoning headroom
        )

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
            else:
                # Track consecutive rejections per figure
                if not hasattr(self, "_sup_rejections"):
                    self._sup_rejections: Dict[str, int] = {}
                self._sup_rejections[figure.name] = self._sup_rejections.get(figure.name, 0) + 1
                reject_count = self._sup_rejections[figure.name]

                if reject_count >= 2:
                    logger.warning(
                        f"[{self.session_id}] Supervisor rejected figure {figure.name} "
                        f"{reject_count} times consecutively, escalating to requires_reanalysis"
                    )
                    review["requires_reanalysis"] = True
                    review["feedback"] = (
                        f"[Escalated after {reject_count} consecutive rejections] "
                        + review.get("feedback", "")
                    )

        # Reset rejection counter on approval
        if review.get("approved", False) and hasattr(self, "_sup_rejections"):
            self._sup_rejections.pop(figure.name, None)

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
            )

            if not signaled:
                await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_AUTO_SKIPPED.value)
                return {"action": "keep", "user_input": ""}

            user_input = self._species_user_response or ""
            await self._send_state_sync(analysis_state, StateSyncChangeType.SPECIES_RESOLVED.value)

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

            return {**action, "user_input": user_input}
        finally:
            analysis_state.species_input_info = None
            analysis_state.wait_input_reason = None
            self._set_status(analysis_state, original_status)
