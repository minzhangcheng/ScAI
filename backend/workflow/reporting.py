"""Reporting mixin — final report, graphic abstract, and detailed report generation."""
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .state import WorkflowState
else:
    WorkflowState = dict

from ..models.schemas import (
    AnalysisState, AgentType, TaskStatus, GraphicAbstract,
)
from ..utils.llm_client import UpstreamExhaustedError
from ..constants import LlmFailureAction
from ..config import get_session_reports, get_session_workspace
from ..services.report import DetailedReportGenerator, compress_image_to_base64

logger = logging.getLogger(__name__)


class ReportingMixin:
    """Mixin providing report generation node methods.

    Composed into SingleCellAnalysisWorkflow. Relies on self.session_id,
    self.pi_agent, self.painter_agent, self.bioinformatics_agent,
    self.supervisor_agent, self._settings_provider, self._vision_settings,
    self._send_message, self._send_state_sync, self._set_status,
    self._wait_for_llm_intervention, self._create_auto_snapshot,
    self.sandbox, and self.llm_manager.
    """

    # ------------------------------------------------------------------
    # Node: generate_final_report
    # ------------------------------------------------------------------

    async def _generate_final_report(self, state: WorkflowState) -> WorkflowState:
        """Generate final analysis report including integrated report with figures"""
        analysis_state = state["analysis_state"]

        # Mark status so interrupted report generation can be detected on resume
        self._set_status(analysis_state, TaskStatus.GENERATING_REPORT)
        await self._send_state_sync(analysis_state, "report_generation_start")

        logger.info(f"[{self.session_id}] Generating final report after {analysis_state.current_round} rounds")

        # Step 1: Generate final summary (with LLM failure intervention)
        await self._send_message("status", "Generating final summary...", AgentType.PI)

        while True:
            try:
                final_summary = await self.pi_agent.generate_final_summary(analysis_state)
                analysis_state.final_summary = final_summary
                await self._send_message(
                    "result",
                    {"final_summary": final_summary},
                    AgentType.PI
                )
                break
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_final_report", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue
                # Skip: use minimal fallback summary
                analysis_state.final_summary = (
                    "Analysis completed. Summary generation was skipped due to LLM failure."
                )
                await self._send_message(
                    "error",
                    "Final summary generation skipped due to LLM failure.",
                    AgentType.PI,
                )
                break
            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to generate final summary")
                analysis_state.final_summary = f"Summary generation failed: {str(e)}"
                await self._send_message(
                    "error",
                    f"Final summary generation failed: {str(e)}",
                    AgentType.PI,
                )
                break

        # Step 2: Generate integrated report with figure/table references
        await self._send_message("status", "Generating integrated report with figures...", AgentType.PI)

        try:
            integrated_report = await self.pi_agent.generate_integrated_report(analysis_state)
            analysis_state.integrated_report = integrated_report

            # --- Collect all workspace figures (figures/ subdir + workspace root) ---
            workspace_dir = get_session_workspace(self.session_id)
            pub_figure_names = {fig.name for fig in analysis_state.publication_figures}
            all_workspace_figs: List[Dict[str, Any]] = []
            seen_stems: set = set()

            def _infer_fig_type(stem: str) -> str:
                s = stem.lower()
                if "umap" in s: return "umap"
                if "volcano" in s: return "volcano"
                if "violin" in s: return "violin"
                if "dotplot" in s or "dot_plot" in s: return "dotplot"
                if "heatmap" in s: return "heatmap"
                if "barplot" in s or "bar_plot" in s or "composition" in s or "proportion" in s: return "barplot"
                return "other"

            for scan_dir in [workspace_dir / "figures", workspace_dir]:
                if not scan_dir.exists():
                    continue
                for img in sorted(
                    list(scan_dir.glob("*.png")) + list(scan_dir.glob("*.jpg")),
                    key=lambda p: p.stat().st_mtime,
                ):
                    if img.stem in seen_stems:
                        continue
                    seen_stems.add(img.stem)
                    pdf = img.with_suffix(".pdf")
                    all_workspace_figs.append({
                        "name": img.stem,
                        "type": _infer_fig_type(img.stem),
                        "path": str(img),
                        "pdf_path": str(pdf) if pdf.exists() else None,
                    })

            # Build full figure list for AI selection (workspace + existing pub figures)
            all_figs_for_ai = [{"name": f["name"], "type": f["type"]} for f in all_workspace_figs]
            for fig in analysis_state.publication_figures:
                if fig.name not in seen_stems:
                    all_figs_for_ai.append({"name": fig.name, "type": fig.figure_type})

            # Ask AI to curate the brief report figure set
            selected_names = set(
                await self.pi_agent.select_brief_figures(analysis_state, all_figs_for_ai)
            )
            # Fallback: if AI returned nothing, show all publication figures
            if not selected_names:
                selected_names = pub_figure_names

            # Build figures_data: publication figures first (already have pngBase64),
            # then AI-selected workspace figures not covered by publication figures
            figures_data: List[Dict[str, Any]] = []
            for fig in analysis_state.publication_figures:
                if fig.name in selected_names:
                    figures_data.append({
                        "id": fig.id,
                        "name": fig.name,
                        "description": fig.description,
                        "figure_type": fig.figure_type,
                        "png_base64": fig.png_base64,
                        "pdf_path": fig.pdf_path,
                        "review_status": fig.review_status.value,
                        "revision_count": fig.revision_count
                    })

            for wf in all_workspace_figs:
                if wf["name"] in selected_names and wf["name"] not in pub_figure_names:
                    try:
                        b64 = compress_image_to_base64(wf["path"], max_width=800)
                    except Exception:
                        logger.warning("Failed to compress image", exc_info=True)
                        b64 = None
                    figures_data.append({
                        "id": str(uuid.uuid4()),
                        "name": wf["name"],
                        "description": "",
                        "figure_type": wf["type"],
                        "png_base64": b64,
                        "png_path": wf["path"],
                        "pdf_path": wf["pdf_path"],
                        "review_status": "pending",
                        "revision_count": 0
                    })

            logger.info(
                f"[{self.session_id}] Brief figures: {len(figures_data)} selected "
                f"({len(pub_figure_names)} pub + {len(all_workspace_figs)} workspace scanned)"
            )

            tables_data = [
                {
                    "id": tbl.id,
                    "name": tbl.name,
                    "description": tbl.description,
                    "csv_path": tbl.csv_path,
                    "review_status": tbl.review_status.value
                }
                for tbl in analysis_state.publication_tables
            ]

            await self._send_message(
                "integrated_report",
                {
                    "report": integrated_report,
                    "figures": figures_data,
                    "tables": tables_data,
                    "total_figures": len(figures_data),
                    "approved_figures": sum(1 for f in figures_data if f.get("review_status") == "approved")
                },
                AgentType.PI
            )

            logger.info(f"[{self.session_id}] Integrated report generated with {len(figures_data)} brief figures")

        except UpstreamExhaustedError as e:
            logger.warning(f"[{self.session_id}] Integrated report LLM failed, skipping: {e}")
            await self._send_message(
                "error",
                "Integrated report generation skipped due to LLM failure.",
                AgentType.PI,
            )
        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to generate integrated report")
            await self._send_message("error", f"Integrated report generation failed: {str(e)}", AgentType.PI)
            # Continue anyway - final summary is still available

        logger.info(f"[{self.session_id}] Final report generated")

        return {
            **state,
            "analysis_state": analysis_state
        }

    # ------------------------------------------------------------------
    # Node: generate_graphic_abstract (dispatcher)
    # ------------------------------------------------------------------

    async def _generate_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Dispatcher: try image mode first when Painter is available; on failure (or when
        Painter is disabled) fall back to text-mode ASCII diagram drawn by the Bio agent.
        """
        if self.painter_agent:
            try:
                return await self._generate_image_graphic_abstract(state)
            except Exception as e:
                # Any failure (including UpstreamExhaustedError) falls back to text mode
                logger.exception(
                    f"[{self.session_id}] Image graphic abstract failed; "
                    f"falling back to text-mode diagram"
                )
                await self._send_message(
                    "status",
                    "Image generation failed, falling back to text-mode diagram...",
                    AgentType.PAINTER,
                )
        else:
            logger.info(
                f"[{self.session_id}] Painter agent disabled; generating text-mode diagram"
            )

        # Text mode — with LLM failure intervention
        # (UpstreamExhaustedError propagates from _generate_text_graphic_abstract
        #  thanks to the except UpstreamExhaustedError: raise added above)
        while True:
            try:
                return await self._generate_text_graphic_abstract(state)
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_graphic_abstract", state["analysis_state"]
                )
                if action == LlmFailureAction.RETRY:
                    continue
                # Skip: graphic abstract is optional
                await self._send_message(
                    "error",
                    "Graphic abstract generation skipped due to LLM failure.",
                    AgentType.PI,
                )
                return {**state, "analysis_state": state["analysis_state"]}

    # ------------------------------------------------------------------
    # Image-mode graphic abstract
    # ------------------------------------------------------------------

    async def _generate_image_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Generate image graphic abstract with revision loop (max 2 revisions, PI selects best).

        NOTE: must only write `analysis_state.graphic_abstract` on successful completion
        so that exceptions leave the state untouched and the dispatcher can cleanly fall
        back to text mode.
        """
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] Generating graphic abstract")
        await self._send_message("status", "Generating graphic abstract...", AgentType.PAINTER)

        try:
            # Step 1: PI generates summary prompt for graphic abstract
            summary_prompt = await self.pi_agent.generate_graphic_abstract_prompt(analysis_state)
            logger.info(f"[{self.session_id}] Generated graphic abstract prompt")

            # Step 2: Painter generates initial graphic abstract
            graphic = await self.painter_agent.generate_graphic_abstract(
                analysis_state,
                summary_prompt
            )

            candidates = [graphic.image_base64]
            revision_history = []

            # Send initial image to frontend
            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": graphic.image_base64,
                    "prompt": graphic.prompt,
                    "created_at": graphic.created_at.isoformat()
                },
                AgentType.PAINTER
            )
            logger.info(f"[{self.session_id}] Initial graphic abstract generated (candidate 1/3)")

            # Step 3: Revision loop - up to 2 revisions
            for revision_round in range(2):
                # PI reviews the latest image (sees image + all previous feedback)
                await self._send_message(
                    "status",
                    f"PI reviewing graphic abstract (round {revision_round + 1}/2)...",
                    AgentType.PI
                )

                review = await self.pi_agent.review_graphic_abstract(
                    analysis_state,
                    candidates[-1],
                    revision_history
                )

                # Record feedback
                revision_history.append({
                    "round": revision_round + 1,
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", []),
                    "quality_score": review.get("quality_score")
                })

                logger.info(
                    f"[{self.session_id}] PI review round {revision_round + 1}: "
                    f"score={review.get('quality_score', 'N/A')}"
                )

                # Painter revises (sees current image + all feedback)
                try:
                    await self._send_message(
                        "status",
                        f"Revising graphic abstract (revision {revision_round + 1}/2)...",
                        AgentType.PAINTER
                    )

                    new_image = await self.painter_agent.revise_graphic_abstract(
                        analysis_state,
                        candidates[-1],
                        graphic.prompt,
                        review.get("feedback", ""),
                        review.get("suggestions", []),
                        review.get("must_fix", []),
                        revision_history
                    )
                    candidates.append(new_image)

                    # Send revised image to frontend
                    await self._send_message(
                        "graphic_abstract",
                        {
                            "image_base64": new_image,
                            "prompt": graphic.prompt,
                            "created_at": graphic.created_at.isoformat()
                        },
                        AgentType.PAINTER
                    )
                    logger.info(
                        f"[{self.session_id}] Graphic abstract revision {revision_round + 1} "
                        f"successful (candidate {len(candidates)}/3)"
                    )

                except Exception as e:
                    logger.warning(
                        f"[{self.session_id}] Graphic abstract revision "
                        f"{revision_round + 1} failed: {e}"
                    )
                    break  # Stop revisions, PI will pick from available candidates

            # Step 4: PI selects the best from all candidates
            if len(candidates) > 1:
                await self._send_message(
                    "status",
                    f"PI selecting best graphic abstract from {len(candidates)} candidates...",
                    AgentType.PI
                )
                selected_idx = await self.pi_agent.select_best_graphic_abstract(
                    analysis_state,
                    candidates,
                    revision_history
                )
            else:
                selected_idx = 0

            # Update graphic abstract with selected image and metadata
            graphic.image_base64 = candidates[selected_idx]
            graphic.revision_count = len(candidates) - 1
            graphic.revision_history = revision_history
            graphic.all_candidates = candidates
            graphic.selected_index = selected_idx
            analysis_state.graphic_abstract = graphic

            # Send final selected image to frontend
            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": candidates[selected_idx],
                    "prompt": graphic.prompt,
                    "created_at": graphic.created_at.isoformat()
                },
                AgentType.PAINTER
            )

            logger.info(
                f"[{self.session_id}] Graphic abstract completed: "
                f"selected candidate {selected_idx + 1}/{len(candidates)}, "
                f"{len(candidates) - 1} revisions"
            )

        except Exception:
            logger.exception(f"[{self.session_id}] Failed to generate image graphic abstract")
            # Re-raise so the dispatcher (_generate_graphic_abstract) can fall back to text mode.
            # `analysis_state.graphic_abstract` is only assigned on success, so state is untouched.
            raise

        logger.info(f"[{self.session_id}] Graphic abstract step completed")

        return {
            **state,
            "analysis_state": analysis_state
        }

    # ------------------------------------------------------------------
    # Text-mode graphic abstract (Painter unavailable fallback)
    # ------------------------------------------------------------------

    _DEFAULT_TEXT_DIAGRAM_PROMPT = (
        "Draw an ASCII/Unicode flow diagram summarizing the study's key signaling "
        "pathway and cellular processes. Reason from the final summary provided and "
        "highlight ligand -> receptor -> downstream effector -> transcription factor "
        "-> cellular outcome relationships. Include at least 5 nodes and use "
        "directional arrows. English labels only.\n\n"
        "Study summary:\n{summary}"
    )

    async def _select_diagram_reviewer(
        self,
        state: AnalysisState,
        diagram: str,
        revision_history: List[Dict[str, Any]],
    ):
        """Pick a text-diagram reviewer by exception-based cascade: PI -> Bio -> Supervisor.

        Returns (review_dict, reviewer_name) or (None, None) if all reviewers fail.
        """
        try:
            r = await self.pi_agent.review_text_diagram(state, diagram, revision_history)
            return r, "pi"
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI text diagram review failed: {e}")

        if self.bioinformatics_agent is not None:
            try:
                r = await self.bioinformatics_agent.review_text_diagram(
                    state, diagram, revision_history
                )
                return r, "bioinformatics"
            except Exception as e:
                logger.warning(f"[{self.session_id}] Bio text diagram review failed: {e}")

        if self.supervisor_agent is not None:
            try:
                r = await self.supervisor_agent.review_text_diagram(
                    state, diagram, revision_history
                )
                return r, "supervisor"
            except Exception as e:
                logger.warning(f"[{self.session_id}] Supervisor text diagram review failed: {e}")

        return None, None

    async def _select_best_diagram_across_reviewers(
        self,
        state: AnalysisState,
        candidates: List[str],
        revision_history: List[Dict[str, Any]],
    ) -> int:
        """Pick best text diagram with same PI -> Bio -> Supervisor cascade."""
        try:
            return await self.pi_agent.select_best_text_diagram(
                state, candidates, revision_history
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] PI best-diagram selection failed: {e}")

        # Bio and Supervisor don't have select_best_text_diagram; use PI's as the only
        # LLM-driven selection. On failure, fall back to the most recent candidate.
        return len(candidates) - 1

    async def _generate_text_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Generate ASCII/Unicode text-mode graphic abstract with tiered review."""
        analysis_state = state["analysis_state"]

        if self.bioinformatics_agent is None:
            logger.error(
                f"[{self.session_id}] Bioinformatics agent missing; cannot generate "
                f"text-mode graphic abstract"
            )
            return state

        await self._send_message(
            "status",
            "Generating text-mode graphic abstract (ASCII diagram)...",
            AgentType.BIOINFORMATICS,
        )

        # Explicit fresh containers — must not inherit any image-mode residue.
        candidates: List[str] = []
        revision_history: List[Dict[str, Any]] = []
        pi_prompt: str = ""

        try:
            # Step 1: PI generates drawing instruction (or hardcoded fallback)
            try:
                pi_prompt = await self.pi_agent.generate_text_diagram_prompt(analysis_state)
            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] PI generate_text_diagram_prompt raised: {e}"
                )
                pi_prompt = ""

            if not pi_prompt:
                summary = (analysis_state.final_summary or "")[:1500]
                pi_prompt = self._DEFAULT_TEXT_DIAGRAM_PROMPT.format(summary=summary)

            # Step 2: Bio draws initial diagram
            initial = await self.bioinformatics_agent.generate_text_diagram(
                analysis_state, pi_prompt
            )
            candidates.append(initial)

            await self._send_message(
                "graphic_abstract",
                {
                    "image_base64": "",
                    "text_diagram": initial,
                    "is_text_mode": True,
                    "prompt": pi_prompt,
                    "created_at": datetime.now().isoformat(),
                },
                AgentType.BIOINFORMATICS,
            )

            # Step 3: revision loop (max 2 rounds)
            for round_idx in range(2):
                review, reviewer_name = await self._select_diagram_reviewer(
                    analysis_state, candidates[-1], revision_history
                )
                if review is None:
                    logger.info(
                        f"[{self.session_id}] No reviewer available for text diagram; "
                        f"ending revision loop at round {round_idx + 1}"
                    )
                    break

                revision_history.append({
                    "round": round_idx + 1,
                    "reviewer": reviewer_name,
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", []),
                    "quality_score": review.get("quality_score"),
                })

                try:
                    await self._send_message(
                        "status",
                        f"Revising text diagram (round {round_idx + 1}/2, reviewer={reviewer_name})...",
                        AgentType.BIOINFORMATICS,
                    )
                    revised = await self.bioinformatics_agent.revise_text_diagram(
                        analysis_state,
                        candidates[-1],
                        pi_prompt,
                        review,
                        revision_history,
                    )
                    candidates.append(revised)

                    await self._send_message(
                        "graphic_abstract",
                        {
                            "image_base64": "",
                            "text_diagram": revised,
                            "is_text_mode": True,
                            "prompt": pi_prompt,
                            "created_at": datetime.now().isoformat(),
                        },
                        AgentType.BIOINFORMATICS,
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.session_id}] Text diagram revision round {round_idx + 1} failed: {e}"
                    )
                    break

            # Step 4: pick best candidate
            if len(candidates) > 1:
                selected_idx = await self._select_best_diagram_across_reviewers(
                    analysis_state, candidates, revision_history
                )
            else:
                selected_idx = 0
            if not (0 <= selected_idx < len(candidates)):
                logger.warning(
                    f"[{self.session_id}] Selected candidate index {selected_idx} is out of bounds, "
                    f"falling back to last candidate."
                )
                selected_idx = len(candidates) - 1

            # Step 5: commit state (LAST, after everything succeeded)
            graphic = GraphicAbstract(
                prompt=pi_prompt,
                image_base64="",
                text_diagram=candidates[selected_idx],
                revision_count=len(candidates) - 1,
                revision_history=revision_history,
                all_candidates=candidates,
                selected_index=selected_idx,
            )
            analysis_state.graphic_abstract = graphic

            if revision_history:
                await self._send_message(
                    "graphic_abstract",
                    {
                        "image_base64": "",
                        "text_diagram": candidates[selected_idx],
                        "is_text_mode": True,
                        "prompt": pi_prompt,
                        "created_at": graphic.created_at.isoformat(),
                    },
                    AgentType.BIOINFORMATICS,
                )

            logger.info(
                f"[{self.session_id}] Text-mode graphic abstract completed: "
                f"selected candidate {selected_idx + 1}/{len(candidates)}, "
                f"{len(candidates) - 1} revisions"
            )

        except UpstreamExhaustedError:
            raise  # let dispatcher show intervention dialog
        except Exception as e:
            logger.exception(
                f"[{self.session_id}] Failed to generate text-mode graphic abstract"
            )
            await self._send_message(
                "error",
                f"Text-mode graphic abstract failed: {type(e).__name__}: {e or 'see logs'}. Continuing with detailed report...",
                AgentType.BIOINFORMATICS,
            )
            # Don't fail the workflow - graphic abstract is optional

        return {**state, "analysis_state": analysis_state}

    # ------------------------------------------------------------------
    # Node: generate_detailed_report
    # ------------------------------------------------------------------

    async def _generate_detailed_report(self, state: WorkflowState) -> WorkflowState:
        """Generate detailed report with figures and tables"""
        analysis_state = state["analysis_state"]

        logger.info(f"[{self.session_id}] Generating detailed report")
        await self._send_message(
            "status", "Generating detailed report with figures and tables...", AgentType.PI
        )

        output_dir = str(get_session_reports(self.session_id))
        os.makedirs(output_dir, exist_ok=True)

        async def progress_callback(data: dict):
            await self._send_message(
                "progress",
                {
                    "step": data.get("step", ""),
                    "progress": data.get("progress", 0),
                    "status": data.get("status", "generating"),
                },
                AgentType.PI,
            )

        llm_client = self._settings_provider.get_llm_client('pi')
        pi_model = self._settings_provider.get_model("pi")
        bio_model = self._settings_provider.get_model("bioinformatics")
        workspace_dir = str(get_session_workspace(self.session_id))

        # === LLM call with retry + user intervention ===
        while True:
            try:
                if self.supervisor_agent:
                    report = await self.supervisor_agent.generate_detailed_report(
                        analysis_state=analysis_state,
                        report_model=pi_model,
                        output_dir=output_dir,
                        workspace_dir=workspace_dir,
                        progress_callback=progress_callback,
                        report_llm_client=llm_client,
                        chat_messages=None,
                    )
                else:
                    supervisor_model = self._settings_provider.get_model("supervisor")
                    generator = DetailedReportGenerator(
                        llm_client=llm_client,
                        pi_model=pi_model,
                        bio_model=bio_model,
                        output_dir=output_dir,
                        workspace_dir=workspace_dir,
                        progress_callback=progress_callback,
                        session_id=self.session_id,
                        vision_settings=self._vision_settings,
                        supervisor_model=supervisor_model,
                    )
                    report = await generator.generate_report(analysis_state)

                await self._send_message(
                    "detailed_report",
                    {
                        "status": "completed",
                        "report_id": report.get("id", ""),
                        "title": report.get("title", ""),
                        "sections_count": len(report.get("sections", [])),
                        "figures_count": len(report.get("figures", [])),
                        "tables_count": len(report.get("tables", [])),
                    },
                    AgentType.PI,
                )

                logger.info(f"[{self.session_id}] Detailed report generated successfully")
                break  # success

            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_detailed_report", analysis_state
                )
                if action == LlmFailureAction.RETRY:
                    continue
                # Skip: send error and fall through to post-processing
                await self._send_message(
                    "error",
                    "Detailed report generation skipped due to LLM failure. "
                    "Analysis is still complete.",
                    AgentType.PI,
                )
                break

            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to generate detailed report")
                await self._send_message(
                    "error",
                    f"Detailed report generation failed: {str(e)}. "
                    "Analysis is still complete.",
                    AgentType.PI,
                )
                break

        # === POST-PROCESSING: ALWAYS RUNS (even on Skip/Error) ===
        await self._create_auto_snapshot(analysis_state, "post_report")

        self._set_status(analysis_state, TaskStatus.COMPLETED)
        await self._send_message("completed", "Analysis completed", AgentType.PI)

        await self._send_state_sync(analysis_state, "analysis_complete")

        logger.info(f"[{self.session_id}] Analysis workflow completed")

        return {
            **state,
            "analysis_state": analysis_state,
        }
