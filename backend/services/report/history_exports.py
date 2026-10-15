"""Export retained round and decision reports without regenerating findings."""
import json
from pathlib import Path

from .variants_parts.detailed import render_detailed_report_document


def render_round_report(state, result, *, segments_by_id=None):
    """Use the same settled content and supplements for single and bulk exports."""
    by_id = (segments_by_id if segments_by_id is not None else
             {item.id: item for item in state.detailed_report_segments})
    segment = by_id.get(result.report_segment_id)
    if segment is None or not segment.content_html:
        return None
    supplements = [by_id[sid] for sid in dict.fromkeys(result.report_supplement_ids)
                   if sid in by_id and sid != segment.id]
    if segment.decision_content_html:
        segment = segment.model_copy(update={"content_html": segment.decision_content_html})
    return render_detailed_report_document(
        state, segment=segment, supplements=supplements,
        deduplicate_figures=not bool(segment.decision_content_html),
    )


def snapshot_report_history(state):
    """Detach the fields read by the export worker from live session updates."""
    return state.model_copy(update={
        field: [item.model_copy(deep=True) for item in getattr(state, field)]
        for field in ("round_results", "detailed_report_segments", "stage_review_reports",
                      "publication_figures", "publication_tables")
    })


def iter_report_history_members(state, lineage):
    """Yield one standalone document at a time, plus an identity/availability index."""
    entries = []
    by_id = {item.id: item for item in state.detailed_report_segments}
    for index, result in enumerate(state.round_results, 1):
        if result.lineage_id != lineage:
            continue
        content = render_round_report(state, result, segments_by_id=by_id)
        path = f"history/rounds/round_{result.round_number}_{index}.html"
        segment = by_id.get(result.report_segment_id)
        entries.append({
            "kind": "round", "round": result.round_number, "phase": result.phase,
            "segment_id": result.report_segment_id,
            "supplement_ids": result.report_supplement_ids,
            "title": segment.title if segment else "",
            "created_at": segment.timestamp.isoformat() if segment else None,
            "status": "available" if content else "unavailable",
            "path": path if content else None,
        })
        if content:
            yield Path(path), content.encode("utf-8")

    for index, report in enumerate(state.stage_review_reports, 1):
        if report.session_id != state.id or report.lineage_id != lineage:
            continue
        available = report.status != "failed" and bool(report.html)
        path = f"history/stages/round_1_{report.stage}_v{report.revision}_{index}.html"
        entries.append({
            "kind": "stage", "round": 1, "stage": report.stage,
            "report_id": report.id, "revision": report.revision,
            "created_at": report.created_at.isoformat(), "status": report.status,
            "path": path if available else None, "error": report.error,
        })
        if available:
            yield Path(path), report.html.encode("utf-8")

    if entries:
        yield Path("history/index.json"), json.dumps({
            "session_id": state.id, "lineage_id": lineage, "reports": entries,
        }, ensure_ascii=False, indent=2).encode("utf-8")
