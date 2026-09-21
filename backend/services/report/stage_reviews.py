"""Freeze independent review exports at the two Round 1 decision points."""
from __future__ import annotations

import html
import logging

from backend.models.schema_parts.round_report import StageReviewReport
from backend.models.schema_parts.session import DetailedReportSegment
from .round_chapters import assemble_round_chapter
from .decision_reviews import decision_artifact_catalog
from .variants_parts.detailed import _render_saved_chapter, render_detailed_report_document

logger = logging.getLogger(__name__)


def latest_stage_review(state, stage):
    return next((report for report in reversed(state.stage_review_reports)
                 if report.stage == stage and report.session_id == state.id
                 and report.lineage_id == (state.restore_branch_id or "main")), None)


def latest_stage_review_url(state, stage):
    report = latest_stage_review(state, stage)
    if report is None:
        path = (state.annotation_report_html_path if stage == "stage1"
                else state.exploration_report_html_path)
        # Legacy paths are active-session exports. Do not borrow ancestor records
        # when a restored session has never produced a review of its own.
        if path:
            endpoint = "annotation-report" if stage == "stage1" else "exploration-report"
            return f"/api/session/{state.id}/{endpoint}/html"
        return None
    if report.status == "failed" or not report.html:
        return None
    return f"/api/session/{state.id}/stage-review-report/{report.id}/html"


def record_stage_review_failure(state, stage):
    """Invalidate the current link when upstream review preparation fails."""
    previous = latest_stage_review(state, stage)
    state.stage_review_reports.append(StageReviewReport(
        session_id=state.id, lineage_id=state.restore_branch_id or "main", stage=stage,
        revision=previous.revision + 1 if previous else 1,
        error="Stage review preparation failed; no current export is available.",
    ))


async def build_stage_review(state, pi_agent, stage, *, summary="", annotations=(),
                             graph_state=None, round_result=None, validation=None, annotation_preview=None):
    """Append a version even on failure, so no caller advertises an older export."""
    previous = latest_stage_review(state, stage)
    report = StageReviewReport(
        session_id=state.id, lineage_id=state.restore_branch_id or "main",
        stage=stage, revision=previous.revision + 1 if previous else 1,
    )
    state.stage_review_reports.append(report)
    catalog = decision_artifact_catalog(state)
    annotation_data = [item.model_dump(mode="json") for item in annotations]
    summary = summary or (round_result.pi_summary if round_result else "")
    context = {
        "summary": summary,
        "candidate_annotations": annotation_data,
        "candidate_preview_note": (annotation_preview or {}).get("note"),
        "annotation_validation": validation,
        "analysis_log": state.analysis_log,
        "annotation_confirmation": (state.annotation_state.model_dump(mode="json")
                                    if stage == "stage2" and state.annotation_state else None),
        "bioinformatics_result": (graph_state or {}).get("bioinformatics_result"),
        "exploration_report": (state.exploration_report.model_dump(mode="json")
                               if stage == "stage2" and state.exploration_report else None),
        "full_round_chapter": (round_result.report_chapter.model_dump(mode="json")
                               if round_result and round_result.report_chapter else None),
        "stage1_reviewed_artifact_ids": [a.artifact_id for a in prior.chapter.artifacts]
            if (prior := latest_stage_review(state, "stage1")) and prior.chapter else [],
    }
    try:
        try:
            raw = await pi_agent.generate_stage_review_chapter(state, stage, catalog, context)
            # The PI orders and explains evidence; it cannot hide catalog results.
            chapter = assemble_round_chapter(raw.get("report_chapter"), catalog, summary, 1)
        except Exception:
            logger.warning("Stage review PI composition unavailable for %s/%s", state.id, stage,
                           exc_info=True)
            chapter = assemble_round_chapter(None, catalog, summary, 1)
            chapter.assembly_notes.append(
                "Stage-specific narrative composition was unavailable. All available "
                "artifacts are retained with their recorded captions."
            )
        report.chapter = chapter
        report.status = "partial" if chapter.source == "fallback" or chapter.assembly_notes else "ready"
        title = f"Round 1 Stage {1 if stage == 'stage1' else 2} Review Report"
        notice = ("Candidate annotations awaiting user confirmation; these are not accepted labels."
                  if stage == "stage1" else
                  "Exploration results awaiting research-direction selection. The final report "
                  "places confirmed annotation figures in the QC/annotation foundation section.")
        content = f'<section id="{stage}" class="report-segment"><h2>{title}</h2><p>{notice}</p>'
        content += f'<p>Review version {report.revision}; status: {report.status}.</p>'
        content += _render_saved_chapter(state, chapter, decision_review=True)
        if annotation_data:
            # The exact candidates being confirmed must survive narrative summarization.
            content += '<h3>Candidate Annotations Awaiting Confirmation</h3>'
            if annotation_preview:
                if annotation_preview.get("image"):
                    content += '<figure><img style="max-width:100%" alt="Candidate annotation UMAP" src="'
                    content += html.escape(annotation_preview["image"], quote=True) + '" /></figure>'
                content += '<p>' + html.escape(annotation_preview.get("note", "")) + '</p>'
            content += '<table><thead><tr>'
            columns = [("cluster_id", "Cluster"), ("cell_type", "Candidate label"),
                       ("key_markers", "Markers"), ("confidence", "Confidence"),
                       ("evidence", "Evidence"), ("alternative_cell_types", "Alternatives"),
                       ("low_confidence_reason", "Uncertainty")]
            content += "".join(f"<th>{label}</th>" for _, label in columns) + '</tr></thead><tbody>'
            for annotation in annotation_data:
                content += '<tr>'
                for key, _ in columns:
                    value = annotation.get(key, "")
                    if isinstance(value, list):
                        value = ", ".join(value)
                    content += '<td>' + html.escape(str(value)) + '</td>'
                content += '</tr>'
            content += '</tbody></table>'
            if validation:
                content += '<h3>Annotation Validation</h3>'
                for key in ("validation_status", "validation_summary", "validation_text_summary"):
                    if validation.get(key):
                        content += '<p>' + html.escape(str(validation[key])) + '</p>'

        content += '</section>'
        segment = DetailedReportSegment(phase=stage, title=title, content_html=content, round_number=1)
        report.html = render_detailed_report_document(
            state, segment=segment, document_title=title, deduplicate_figures=False,
        )
    except Exception:
        report.status = "failed"
        report.html = ""
        report.error = "Stage review rendering failed; no current export is available."
        logger.exception("Stage review rendering failed for %s/%s", state.id, stage)
    return latest_stage_review_url(state, stage)
