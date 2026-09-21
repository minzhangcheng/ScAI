# Graphic abstract: brief figure gallery, image and text variants, reviewer selection.
# Moved verbatim out of backend/workflow/reporting.py; composed into ReportingMixin.
"""Reporting mixin — final report, graphic abstract, and detailed report generation."""
import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..state import WorkflowState
else:
    WorkflowState = dict

from ...models.schemas import (
    AnalysisState, AgentType, TaskStatus, GraphicAbstract,
    IntegratedReportFigureRef,
)
from ...utils.llm_client import UpstreamExhaustedError
from ...constants import LlmFailureAction
from ..helpers import reset_unattended_gate_budget
from ...config import get_session_reports, get_session_workspace
from ...services.report import DetailedReportGenerator, compress_image_to_base64
from ...services.report.figure_discovery import (
    discover_workspace_figures,
    resolve_state_figure_candidates,
)
from ...services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
    has_blocking_layout_defects,
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
    render_is_intact,
)
from ...services.report.figure_attribution import attribute_figures_to_rounds  # noqa: F401
from ..direction_coverage import current_lineage_id, direction_coverage_status
from ...services.report.paths import resolve_report_output_dir
from ...services.report.lineage import build_lineage_report_state
from ...services.report.integrated_report import integrated_report_catalog_fingerprint

from .shared import (  # noqa: F401
    logging,
    os,
    uuid,
    datetime,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    AnalysisState,
    AgentType,
    TaskStatus,
    GraphicAbstract,
    UpstreamExhaustedError,
    LlmFailureAction,
    get_session_reports,
    get_session_workspace,
    DetailedReportGenerator,
    compress_image_to_base64,
    discover_workspace_figures,
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
    has_blocking_layout_defects,
    attribute_figures_to_rounds,
    current_lineage_id,
    direction_coverage_status,
    resolve_report_output_dir,
    build_lineage_report_state,
    report_coverage_observation,
)


logger = logging.getLogger(__name__)

# A candidate line is rendered into the PI's prompt once per figure against a
# 200-candidate cap, so the description is an excerpt, not the whole field.
_MAX_CANDIDATE_DESCRIPTION = 200

def _attribute_rounds(
    analysis_state, publication_figures, workspace_figs,
) -> Dict[str, Optional[int]]:
    """Map candidate stem -> producing round, omitting what cannot be resolved.

    Reuses ``attribute_figures_to_rounds`` (supervision records, then publication
    round_number, then mtime against round boundaries) rather than inferring a
    second time; it keys on basename, so translate to stems for the caller.
    """
    paths: Dict[str, str] = {}
    for f in workspace_figs:
        path = f.get("path")
        if path:
            paths[os.path.basename(path)] = path
    for fig in publication_figures:
        png_path = getattr(fig, "png_path", None)
        if png_path:
            paths.setdefault(os.path.basename(png_path), png_path)
    if not paths:
        return {}
    try:
        attributed = attribute_figures_to_rounds(
            list(paths),
            state=analysis_state,
            records=None,
            figure_paths=paths,
        )
    except Exception:
        logger.warning("Figure round attribution failed", exc_info=True)
        return {}
    return {
        os.path.splitext(basename)[0]: round_number
        for basename, round_number in attributed.items()
    }


def _round_coverage_floor(candidates, selected_names) -> set:
    """Admit one figure per round that produced candidates but won none.

    A gallery that narrates rounds 1 and 5 of an eight-round analysis is not a
    summary of it. The PI's judgement decides *which* figure represents a round;
    this only decides that a round which produced results is represented at all.
    Diagnostics never satisfy the floor and never trigger it.
    """
    by_round: Dict[int, List[dict]] = {}
    for index, cand in enumerate(candidates):
        round_number = cand.get("round")
        if not round_number or cand.get("diagnostic"):
            continue
        cand = {**cand, "_index": index}
        by_round.setdefault(int(round_number), []).append(cand)

    admitted = set()
    for round_number, entries in sorted(by_round.items()):
        if any(c["name"] in selected_names for c in entries):
            continue
        entries.sort(key=lambda c: (not c.get("is_publication"), c["_index"]))
        admitted.add(entries[0]["name"])
        logger.info(
            "Brief figures: round %s produced %d candidates but won none; "
            "admitting %s for coverage",
            round_number, len(entries), entries[0]["name"],
        )
    return admitted


class GraphicAbstractMixin:
    """Graphic abstract: brief figure gallery, image and text variants, reviewer selection."""

    async def _prepare_brief_figures(
        self, analysis_state,
    ) -> tuple[List[IntegratedReportFigureRef], List[str], List[Dict[str, Any]]]:
        """Select and freeze the numbered Integrated Report catalogue."""
        DISCOVERY_CAP = 200
        # The report catalogue uses the existing report-eligibility and
        # published-head contract. Browsable pending/rejected revisions remain
        # visible in the ordinary gallery but cannot acquire report numbers.
        #
        # The one thing that breadth must not carry with it is an entry whose
        # file is no longer the figure it advertises: see ``render_is_intact``.
        # Dropping it here also feeds ``excluded_artifact_stems`` below, so the
        # same blank file is not re-offered by workspace discovery.
        publication_figures = [
            fig for fig in eligible_artifacts(
                analysis_state.publication_figures, state=analysis_state,
            )
            if render_is_intact(fig)
            and (getattr(fig, "lineage_id", None) or "main")
            == current_lineage_id(analysis_state)
        ]
        pub_figure_names = {fig.name for fig in publication_figures}
        excluded_stems = excluded_artifact_stems(
            analysis_state.publication_figures,
            publication_figures,
            invalidated_stems=getattr(
                analysis_state, "invalidated_artifact_stems", None
            ),
        )

        discovered = resolve_state_figure_candidates(
            analysis_state,
            get_session_workspace(self.session_id),
            report_only=False,
            lineage_id=current_lineage_id(analysis_state),
        )
        discovered = [f for f in discovered if f["name"] not in excluded_stems]
        all_workspace_figs = discovered[:DISCOVERY_CAP]
        if len(discovered) > DISCOVERY_CAP:
            logger.info(
                f"[{self.session_id}] Figure discovery: {len(discovered)} found, "
                f"passing first {DISCOVERY_CAP} to PI selection"
            )
        seen_stems = {f["name"] for f in all_workspace_figs}

        # Build the selection input: workspace candidates + publication figures
        # not already discovered. ``is_publication`` drives the bounded fallback.
        # ``round`` and ``description`` exist on the record and were never passed;
        # the PI was choosing "key results" from a name and a plot type alone.
        round_by_stem = _attribute_rounds(
            analysis_state, publication_figures, all_workspace_figs
        )
        descriptions = {
            fig.name: (fig.description or "") for fig in publication_figures
        }
        diagnostic_by_name = {
            fig.name: bool(getattr(fig, "diagnostic", False))
            for fig in publication_figures
        }
        all_figs_for_ai = [
            {
                "name": f["name"],
                "type": f["type"],
                "is_publication": f["name"] in pub_figure_names,
                "round": round_by_stem.get(f["name"]),
                "description": descriptions.get(f["name"], "")[:_MAX_CANDIDATE_DESCRIPTION],
                # Discovery only knows a figure is diagnostic from the directory it
                # sits in; the record knows because the platform drew it. The
                # platform's QC panel is written into the scanned workspace, so it
                # arrives through this branch and would otherwise lose the flag.
                "diagnostic": bool(f.get("diagnostic")) or diagnostic_by_name.get(f["name"], False),
            }
            for f in all_workspace_figs
        ]
        for fig in publication_figures:
            if fig.name not in seen_stems:
                all_figs_for_ai.append({
                    "name": fig.name,
                    "type": fig.figure_type,
                    "is_publication": True,
                    "round": round_by_stem.get(fig.name),
                    "description": (fig.description or "")[:_MAX_CANDIDATE_DESCRIPTION],
                    # Read from the record, not assumed: a platform-rendered panel
                    # is diagnostic no matter which directory it landed in, and the
                    # per-round coverage floor and the PI rubric both key on this.
                    "diagnostic": bool(getattr(fig, "diagnostic", False)),
                })

        selected_names = list(
            await self.pi_agent.select_brief_figures(analysis_state, all_figs_for_ai)
        )
        # Defensive filter: drop any selected name that is not a real candidate,
        # so a bogus non-empty selection cannot suppress the publication fallback
        # below and silently empty the Results gallery.
        known_names = pub_figure_names | {f["name"] for f in all_workspace_figs}
        selected_names = list(dict.fromkeys(
            n for n in selected_names if n in known_names
        ))
        # Fallback: if selection is empty, show all publication figures.
        if not selected_names:
            selected_names = [
                fig.name for fig in publication_figures
            ]
        coverage_names = _round_coverage_floor(all_figs_for_ai, set(selected_names))
        selected_names.extend(
            candidate["name"] for candidate in all_figs_for_ai
            if candidate["name"] in coverage_names
            and candidate["name"] not in selected_names
        )

        figures_data: List[Dict[str, Any]] = []
        publication_by_name = {fig.name: fig for fig in publication_figures}
        workspace_by_name = {
            fig["name"]: fig for fig in all_workspace_figs
            if fig["name"] not in pub_figure_names
        }
        for selected_name in selected_names:
            fig = publication_by_name.get(selected_name)
            if fig is not None:
                figures_data.append({
                    "id": fig.id,
                    "name": fig.name,
                    "description": fig.description,
                    "figure_type": fig.figure_type,
                    "png_base64": fig.png_base64,
                    "png_path": fig.png_path,
                    "pdf_path": fig.pdf_path,
                    "render_sha256": getattr(fig, "render_sha256", None),
                    "preview_sha256": getattr(fig, "preview_sha256", None),
                    "pdf_sha256": getattr(fig, "pdf_sha256", None),
                    "review_status": fig.review_status.value,
                    "review_unavailable": bool(
                        getattr(fig, "review_unavailable", False)
                    ),
                    "revision_count": fig.revision_count,
                    "plan_step_id": getattr(fig, "plan_step_id", None),
                    "plan_id": getattr(fig, "plan_id", None),
                    "purpose": getattr(fig, "purpose", "result") or "result",
                    "logical_figure_id": getattr(fig, "logical_figure_id", None),
                    "render_attempt_id": getattr(fig, "render_attempt_id", None),
                    "revision_of_id": getattr(fig, "revision_of_id", None),
                    "supersedes_id": getattr(fig, "supersedes_id", None),
                    "is_published_head": bool(
                        getattr(fig, "is_published_head", False)
                    ),
                    "round": round_by_stem.get(fig.name),
                    # The renderer's verdict travels with every eligible head;
                    # "Final" over an illegible figure would be a lie.
                    "layout_blocking": has_blocking_layout_defects(fig),
                    "diagnostic": bool(getattr(fig, "diagnostic", False)),
                    "is_publication": True,
                })

                continue
            wf = workspace_by_name.get(selected_name)
            if wf is not None:
                try:
                    b64 = compress_image_to_base64(wf["path"], max_width=800)
                except Exception:
                    logger.warning("Failed to compress image", exc_info=True)
                    b64 = None
                figures_data.append({
                    "id": str(wf.get("id") or wf.get("legacy_artifact_id") or ""),
                    "name": wf["name"],
                    "description": "",
                    "figure_type": wf["type"],
                    "png_base64": b64,
                    "png_path": wf["path"],
                    "pdf_path": wf["pdf_path"],
                    # Workspace-only entries have no PublicationFigure record,
                    # so the persisted integrated-report payload owns the bytes
                    # needed by the download routes.  Bind those routes to the
                    # exact image/PDF selected here rather than trusting a later
                    # same-name file in the workspace.
                    "render_sha256": image_file_sha256(wf["path"]),
                    "preview_sha256": preview_render_sha256(b64),
                    "pdf_sha256": pdf_file_sha256(wf["pdf_path"]),
                    "review_status": "pending",
                    "review_unavailable": False,
                    "revision_count": 0,
                    "plan_step_id": wf.get("plan_step_id"),
                    "plan_id": wf.get("plan_id"),
                    "purpose": wf.get("purpose"),
                    "round": round_by_stem.get(wf["name"]),
                    "layout_blocking": False,
                    "diagnostic": bool(wf.get("diagnostic")),
                    "is_publication": False,
                })

        task_notes_by_figure_id: Dict[str, List[str]] = {}
        reporting_limitations: List[str] = []
        selected_ids = {str(item.get("id") or "") for item in figures_data}
        report_lineage_id = current_lineage_id(analysis_state)
        for task in getattr(analysis_state, "figure_revision_tasks", None) or []:
            task_lineage_id = getattr(task, "lineage_id", None)
            if (
                task_lineage_id is not None
                and str(task_lineage_id or "main") != report_lineage_id
            ):
                continue
            note = str(getattr(task, "pi_report_note", None) or "").strip()
            if not note:
                continue
            candidate_ids = [
                str(getattr(task, "selected_figure_id", None) or ""),
                str(getattr(task, "current_figure_id", None) or ""),
            ]
            attached_id = next((item for item in candidate_ids if item in selected_ids), None)
            if attached_id:
                notes = task_notes_by_figure_id.setdefault(attached_id, [])
                if note not in notes:
                    notes.append(note)
            else:
                label = (
                    getattr(task, "plan_requirement_key", None)
                    or getattr(task, "logical_figure_id", None)
                    or "Planned visual"
                )
                disposition = (
                    getattr(task, "pi_disposition", None)
                    or getattr(task, "status", None)
                    or "not selected"
                )
                reporting_limitations.append(f"{label}: {disposition} — {note}")

        def combined_pi_note(figure_id: str, figure_name: str) -> Optional[str]:
            notes = [
                *task_notes_by_figure_id.get(figure_id, []),
                str(getattr(
                    publication_by_name.get(figure_name),
                    "pi_degraded_acceptance_reason", "",
                ) or "").strip(),
            ]
            combined = "; ".join(note for note in dict.fromkeys(notes) if note)
            return combined or None

        refs: List[IntegratedReportFigureRef] = []
        for item in figures_data:
            figure_id = str(item.get("id") or "").strip()
            render_sha256 = str(item.get("render_sha256") or "").strip()
            # A numbered new reference must have a stable identity and verified
            # render digest. Unbound compatibility candidates remain available
            # to the ordinary gallery but cannot be cited as a frozen figure.
            if not figure_id or not render_sha256:
                continue
            refs.append(IntegratedReportFigureRef(
                number=len(refs) + 1,
                figure_id=figure_id,
                name=str(item.get("name") or ""),
                description=str(item.get("description") or ""),
                figure_type=str(item.get("figure_type") or "other"),
                render_sha256=render_sha256,
                pdf_sha256=item.get("pdf_sha256"),
                logical_figure_id=item.get("logical_figure_id"),
                review_status=str(item.get("review_status") or "pending"),
                review_unavailable=bool(item.get("review_unavailable", False)),
                revision_count=int(item.get("revision_count") or 0),
                round_number=item.get("round"),
                diagnostic=bool(item.get("diagnostic", False)),
                is_publication=bool(item.get("is_publication", True)),
                pi_report_note=combined_pi_note(
                    figure_id, str(item.get("name") or ""),
                ),
            ))

        logger.info(
            f"[{self.session_id}] Brief figures: {len(figures_data)} selected "
            f"({len(pub_figure_names)} pub + {len(all_workspace_figs)} workspace scanned)"
        )

        return refs, reporting_limitations, figures_data

    async def _emit_integrated_report(self, analysis_state) -> None:
        """Emit the already-persisted publication as one versioned message."""
        publication = analysis_state.integrated_report_state.published
        if publication is None:
            return
        figures_data = [
            {
                "id": ref.figure_id,
                "name": ref.name,
                "description": ref.description,
                "figure_type": ref.figure_type,
                "render_sha256": ref.render_sha256,
                "pdf_sha256": ref.pdf_sha256,
                "review_status": ref.review_status,
                "review_unavailable": ref.review_unavailable,
                "revision_count": ref.revision_count,
                "logical_figure_id": ref.logical_figure_id,
                "round": ref.round_number,
                "diagnostic": ref.diagnostic,
                "is_publication": ref.is_publication,
                "report_number": ref.number,
                "pi_report_note": ref.pi_report_note,
                "report_generation_id": publication.generation_id,
            }
            for ref in publication.figure_refs
        ]
        tables_data = [
            {
                "id": tbl.id,
                "name": tbl.name,
                "description": tbl.description,
                "csv_path": tbl.csv_path,
                "review_status": tbl.review_status.value,
            }
            for tbl in analysis_state.publication_tables
        ]

        await self._send_message(
            "integrated_report",
            {
                "report": publication.markdown,
                "figures": figures_data,
                "tables": tables_data,
                "generation_id": publication.generation_id,
                "publication_revision": publication.publication_revision,
                "catalog_fingerprint": publication.catalog_fingerprint,
                "total_figures": len(figures_data),
                "approved_figures": sum(
                    1 for f in figures_data if f.get("review_status") == "approved"
                ),
            },
            AgentType.PI,
        )
        logger.info(
            f"[{self.session_id}] Integrated report generated with "
            f"{len(figures_data)} brief figures"
        )

    async def _assemble_brief_figures(
        self, analysis_state, integrated_report
    ) -> None:
        """Compatibility wrapper for direct callers of the former one-step API."""
        refs, limitations, figures_data = await self._prepare_brief_figures(
            analysis_state
        )
        if not hasattr(analysis_state, "integrated_report_state"):
            tables_data = [
                {
                    "id": tbl.id,
                    "name": tbl.name,
                    "description": tbl.description,
                    "csv_path": tbl.csv_path,
                    "review_status": tbl.review_status.value,
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
                    "approved_figures": sum(
                        1 for item in figures_data
                        if item.get("review_status") == "approved"
                    ),
                },
                AgentType.PI,
            )
            return
        from ...models.schemas import IntegratedReportPublication
        from ...services.report.integrated_report import sanitize_integrated_report

        previous = analysis_state.integrated_report_state.published
        revision = (previous.publication_revision if previous else 0) + 1
        generation_id = str(uuid.uuid4())
        fingerprint = integrated_report_catalog_fingerprint(refs)
        analysis_state.integrated_report_state.published = IntegratedReportPublication(
            generation_id=generation_id,
            publication_revision=revision,
            lineage_id=current_lineage_id(analysis_state),
            markdown=sanitize_integrated_report(
                str(integrated_report), refs, limitations,
            ),
            figure_refs=refs,
            reporting_limitations=limitations,
            catalog_fingerprint=fingerprint,
        )
        analysis_state.integrated_report = (
            analysis_state.integrated_report_state.published.markdown
        )
        await self._emit_integrated_report(analysis_state)

    async def _generate_graphic_abstract(self, state: WorkflowState) -> WorkflowState:
        """Dispatcher: try image mode first when Painter is available; on failure (or when
        Painter is disabled) fall back to text-mode ASCII diagram drawn by the Bio agent.
        """
        analysis_state = state["analysis_state"]

        if not analysis_state.analysis_log and not analysis_state.code_history:
            logger.info(f"[{self.session_id}] No analysis work recorded, skipping graphic abstract")
            return {**state, "analysis_state": analysis_state}

        if self.ensure_agent_ready("painter"):
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
        reset_unattended_gate_budget(self, "generate_graphic_abstract")
        while True:
            try:
                return await self._generate_text_graphic_abstract(state)
            except UpstreamExhaustedError as e:
                action = await self._wait_for_llm_intervention(
                    e, "generate_graphic_abstract", state["analysis_state"]
                )
                if action == LlmFailureAction.RETRY:
                    continue
                if action == LlmFailureAction.TERMINATE:
                    from ..helpers import WorkflowTerminatedError
                    self._set_status(analysis_state, TaskStatus.STOPPED)
                    await self._send_state_sync(analysis_state, "graphic_abstract_terminated")
                    raise WorkflowTerminatedError("User terminated graphic abstract generation") from e
                # Skip: graphic abstract is optional
                await self._send_message(
                    "error",
                    "Graphic abstract generation skipped due to LLM failure.",
                    AgentType.PI,
                )
                return {**state, "analysis_state": state["analysis_state"]}

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
            review_by_candidate: Dict[int, Optional[bool]] = {}
            review_path_unavailable = False

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
                candidate_index = len(candidates) - 1
                review_unavailable = review.get("review_unavailable") is True
                review_by_candidate[candidate_index] = (
                    False if review_unavailable else True
                )

                # Record feedback
                revision_history.append({
                    "round": revision_round + 1,
                    "candidate_index": candidate_index,
                    "feedback": review.get("feedback", ""),
                    "suggestions": review.get("suggestions", []),
                    "must_fix": review.get("must_fix", []),
                    "quality_score": review.get("quality_score"),
                    "review_unavailable": review_unavailable,
                    "review_reason": review.get("review_reason"),
                })

                logger.info(
                    f"[{self.session_id}] PI review round {revision_round + 1}: "
                    f"score={review.get('quality_score', 'N/A')}"
                )

                if review_unavailable:
                    review_path_unavailable = True
                    logger.info(
                        f"[{self.session_id}] Graphic abstract review unavailable for "
                        f"candidate {candidate_index + 1}; retaining the latest valid image"
                    )
                    break

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
            selection_available: Optional[bool]
            if len(candidates) > 1 and not review_path_unavailable:
                await self._send_message(
                    "status",
                    f"PI selecting best graphic abstract from {len(candidates)} candidates...",
                    AgentType.PI
                )
                try:
                    selected_idx = await self.pi_agent.select_best_graphic_abstract(
                        analysis_state,
                        candidates,
                        revision_history
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.session_id}] Graphic abstract selection unavailable: {e}"
                    )
                    selected_idx = None
                selection_available = selected_idx is not None
                if selected_idx is None:
                    selected_idx = len(candidates) - 1
            elif len(candidates) > 1:
                # No visual selection is attempted after an unavailable review;
                # retaining the newest valid image is deterministic and carries
                # no PI selection verdict.
                selected_idx = len(candidates) - 1
                selection_available = None
            else:
                selected_idx = 0
                selection_available = None

            if not (0 <= selected_idx < len(candidates)):
                logger.warning(
                    f"[{self.session_id}] Selected graphic abstract index {selected_idx} "
                    "was out of bounds; retaining the latest candidate"
                )
                selected_idx = len(candidates) - 1
                selection_available = False

            selected_review_available = review_by_candidate.get(selected_idx)

            # Update graphic abstract with selected image and metadata
            graphic.image_base64 = candidates[selected_idx]
            graphic.revision_count = len(candidates) - 1
            graphic.revision_history = revision_history
            graphic.all_candidates = candidates
            graphic.selected_index = selected_idx
            graphic.review_available = selected_review_available
            graphic.selection_available = selection_available
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
