"""Complete result inventories for human decisions, independent of publication eligibility."""
from pathlib import Path
import html

from backend.models.schema_parts.round_report import RoundReportArtifact


def decision_artifact_catalog(state, *, round_number=None, lineage_id=None):
    """Retain every registered version in the decision's round and lineage."""
    number = state.current_round if round_number is None else round_number
    lineage = lineage_id or state.restore_branch_id or "main"
    rounds = {0, 1} if number == 1 else {number}
    return [
        RoundReportArtifact(
            artifact_id=record.id, kind=kind,
            filename=Path(getattr(record, path_field) or record.name).name,
            caption=record.description or record.interpretation or "",
        )
        for kind, records, path_field in (
            ("figure", state.publication_figures, "png_path"),
            ("table", state.publication_tables, "csv_path"),
        )
        for record in records
        if record.round_number in rounds and record.lineage_id == lineage
    ]


def decision_artifact_notice(record, records):
    """Disclose review, layout and revision facts without withholding the result."""
    fields = []
    for label, field in (
        ("Review status", "review_status"), ("Producer status", "producer_status"),
        ("PI feedback", "pi_feedback"), ("Remaining issues", "unresolved_must_fix"),
        ("Layout findings", "layout_violations"),
        ("Provenance issue", "provenance_validation_reason"),
        ("Invalidation reason", "invalidation_reason"),
        ("Revision of", "revision_of_id"), ("Supersedes", "supersedes_id"),
    ):
        value = getattr(record, field, None)
        if value:
            value = getattr(value, "value", value)
            fields.append((label, str(value)))
    replacements = [item.id for item in records if record.id in (
        getattr(item, "supersedes_id", None), getattr(item, "revision_of_id", None),
    )]
    if replacements:
        fields.append(("Later versions", ", ".join(replacements)))
    if not getattr(record, "is_current", True):
        fields.append(("Current result", "No; retained historical result"))
    return '<div class="decision-artifact-status">' + ''.join(
        '<p><strong>' + html.escape(label) + ':</strong> ' + html.escape(value) + '</p>'
        for label, value in fields
    ) + '</div>'


def render_round_decision_content(state, result, phase, title):
    """Freeze the complete review separately from the publication chapter."""
    from .round_chapters import assemble_round_chapter
    from .variants_parts.detailed import generate_detailed_report_segment_html

    catalog = decision_artifact_catalog(
        state, round_number=result.round_number, lineage_id=result.lineage_id,
    )
    raw = result.report_chapter.model_dump(mode="json") if result.report_chapter else None
    chapter = assemble_round_chapter(raw, catalog, result.pi_summary, result.round_number)
    return generate_detailed_report_segment_html(
        state, phase, title, round_result=result.model_copy(update={"report_chapter": chapter}),
        decision_review=True,
    )
