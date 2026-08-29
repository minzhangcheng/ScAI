"""Deterministic analysis-quality disclosures shared by every report surface."""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional

from backend.utils.formatting import normalize_text_list


_SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = {1}
_ROUND_STATUSES = {"completed", "partial", "failed", "skipped"}
_EVIDENCE_STATUSES = {"substantive", "empty", "failed"}

_REVIEW_SUCCESS_STATUSES = {"completed"}
_REVIEW_FAILURE_STATUSES = {
    "failed",
    "error",
    "max_steps_reached",
    "max_iterations",
    "timeout",
}
_REVIEW_PARTIAL_STATUSES = {"partial"}
_ASSESSMENT_VERDICTS = {"resolved", "partial", "unresolved"}
_MISSING = object()


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _texts(values: Iterable[Any] | None, *, limit: int = 20) -> List[str]:
    return normalize_text_list(values, limit=limit)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _round_disclosures(state: Any) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    rounds: List[Dict[str, Any]] = []
    blocking: List[str] = []
    cautions: List[str] = []
    ordered = sorted(
        list(_value(state, "round_results", []) or []),
        key=lambda item: _integer(_value(item, "round_number", 0)),
    )
    for result in ordered:
        evidence_schema_version = _value(
            result, "evidence_schema_version", None
        )
        legacy_evidence_schema = evidence_schema_version is None
        current_evidence_schema = (
            isinstance(evidence_schema_version, int)
            and not isinstance(evidence_schema_version, bool)
            and evidence_schema_version in _SUPPORTED_EVIDENCE_SCHEMA_VERSIONS
        )
        number = _integer(_value(result, "round_number", 0))
        raw_status = _value(result, "round_status", _MISSING)
        raw_evidence = _value(result, "evidence_status", _MISSING)
        status = (
            "completed" if legacy_evidence_schema and raw_status is _MISSING
            else str("" if raw_status is _MISSING else raw_status or "").lower()
        )
        evidence = (
            "empty" if legacy_evidence_schema and raw_evidence is _MISSING
            else str("" if raw_evidence is _MISSING else raw_evidence or "").lower()
        )
        coverage = _value(result, "direction_coverage_eligible", None)
        raw_phase = _value(result, "phase", "exploration") or "exploration"
        phase = str(getattr(raw_phase, "value", raw_phase)).lower()
        verified = _value(result, "bioinformatics_evidence_verified", None)
        notes = _texts(_value(result, "evidence_notes", []))
        pathology_coverage: List[Dict[str, Any]] = []
        for raw_coverage in list(
            _value(result, "pathology_coverage", []) or []
        )[:100]:
            expected_tiles = _value(raw_coverage, "expected_tiles", None)
            valid_tiles = _value(raw_coverage, "valid_tiles", None)
            pathology_coverage.append({
                "image_id": str(_value(raw_coverage, "image_id", "") or ""),
                "image_type": str(_value(raw_coverage, "image_type", "") or ""),
                "status": str(_value(raw_coverage, "status", "error") or "error"),
                "expected_tiles": (
                    _integer(expected_tiles) if expected_tiles is not None else None
                ),
                "valid_tiles": (
                    _integer(valid_tiles) if valid_tiles is not None else None
                ),
                "coverage_ratio": _value(raw_coverage, "coverage_ratio", None),
                "invalid_batch_count": len(
                    list(_value(raw_coverage, "invalid_batches", []) or [])
                ),
                "error": str(_value(raw_coverage, "error", "") or "")[:1000],
            })
        entry = {
            "round_number": number,
            "phase": phase,
            "direction": str(_value(result, "direction", "") or ""),
            "round_status": status,
            "evidence_status": evidence,
            "evidence_schema_version": evidence_schema_version,
            "direction_coverage_eligible": coverage,
            "bioinformatics_evidence_verified": verified,
            "evidence_execution_ids": _texts(
                _value(result, "evidence_execution_ids", []), limit=100
            ),
            "evidence_notes": notes,
            "pathology_coverage": pathology_coverage,
        }
        rounds.append(entry)
        label = f"Round {number}"
        if not legacy_evidence_schema and not current_evidence_schema:
            blocking.append(
                f"{label} uses unsupported evidence schema version "
                f"{evidence_schema_version!r}."
            )
        if current_evidence_schema:
            if status not in _ROUND_STATUSES:
                blocking.append(f"{label} has unsupported round status {status!r}.")
            elif status in {"failed", "skipped"}:
                blocking.append(f"{label} ended with status {status}.")
            elif status == "partial":
                cautions.append(f"{label} completed only partially.")
            if evidence not in _EVIDENCE_STATUSES:
                blocking.append(
                    f"{label} has unsupported evidence status {evidence!r}."
                )
            elif evidence in {"failed", "empty"}:
                blocking.append(f"{label} has {evidence} evidence status.")
        if current_evidence_schema and phase != "integration":
            if coverage is False:
                blocking.append(f"{label} is not eligible for direction coverage.")
            elif coverage is not True:
                blocking.append(
                    f"{label} has no verifiable direction-coverage verdict."
                )
        if current_evidence_schema and phase != "integration":
            if verified is False:
                blocking.append(f"{label} bioinformatics evidence is unverified.")
            elif verified is not True:
                blocking.append(
                    f"{label} has no verifiable bioinformatics evidence verdict."
                )
        elif legacy_evidence_schema:
            cautions.append(
                f"{label} predates explicit execution-derived evidence verdicts."
            )
        for coverage_entry in pathology_coverage:
            expected_tiles = coverage_entry["expected_tiles"]
            valid_tiles = coverage_entry["valid_tiles"]
            image_label = (
                coverage_entry["image_type"]
                or coverage_entry["image_id"]
                or "pathology image"
            )
            if (
                isinstance(expected_tiles, int)
                and isinstance(valid_tiles, int)
                and valid_tiles < expected_tiles
            ):
                ratio = (100.0 * valid_tiles / expected_tiles) if expected_tiles else 0.0
                cautions.append(
                    f"{label} pathology coverage for {image_label} was "
                    f"{valid_tiles}/{expected_tiles} validated tiles "
                    f"({ratio:.1f}%); omitted tiles were not assessed."
                )
            elif coverage_entry["status"] in {"error", "failed", "skipped"}:
                cautions.append(
                    f"{label} pathology coverage for {image_label} ended with "
                    f"status {coverage_entry['status']}."
                )
        cautions.extend(f"{label}: {note}" for note in notes)
    return rounds, blocking, cautions


def _review_disclosures(state: Any) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    reviews: List[Dict[str, Any]] = []
    blocking: List[str] = []
    cautions: List[str] = []
    for index, raw in enumerate(_value(state, "post_execution_reviews", []) or []):
        if not isinstance(raw, dict):
            reviews.append({
                "review_attempt_id": f"invalid-{index + 1}",
                "round_number": 0,
                "review_status": "invalid",
                "review_reason": "",
                "final_gap_status": "unverifiable",
                "plan_audit": {},
                "supplementary_execution": {
                    "status": "unverifiable",
                    "summary": "",
                    "warnings": [],
                },
                "assessment": {
                    "verdict": "not_assessed",
                    "summary": "",
                    "resolved_gaps": [],
                    "remaining_gaps": [],
                },
            })
            blocking.append(
                f"Post-execution review {index + 1} has an invalid retained record."
            )
            continue
        audit = raw.get("plan_audit") if isinstance(raw.get("plan_audit"), dict) else {}
        result = raw.get("review_result") if isinstance(raw.get("review_result"), dict) else {}
        assessment = raw.get("assessment") if isinstance(raw.get("assessment"), dict) else {}
        round_number = _integer(raw.get("round_number", 0))
        remaining = _texts(assessment.get("remaining_gaps"), limit=20)
        warnings = _texts(result.get("warnings"), limit=20)
        review_status = str(raw.get("review_status") or "unknown").lower()
        execution_status = str(result.get("status") or "unknown").lower()
        verdict = str(assessment.get("verdict") or "not_assessed").lower()
        execution_status_explicit = "status" in result
        verdict_explicit = "verdict" in assessment
        audit_unmet = _integer(audit.get("n_unmet", 0))
        audit_unverifiable = _integer(audit.get("n_unverifiable", 0))
        review_plan = (
            raw.get("review_plan")
            if isinstance(raw.get("review_plan"), dict)
            else {}
        )

        # ``no_action`` and ``skipped`` do not claim that a supplementary pass
        # repaired anything. Preserve those producer states without laundering
        # them into a successful execution. A no-action decision is complete
        # only when its stored plan says so and the audit contained no gap.
        no_action_complete = (
            review_status == "no_action"
            and str(review_plan.get("verdict") or "").lower() == "no_action"
            and not remaining
            and not audit_unmet
            and not audit_unverifiable
        )
        skipped_without_gap = (
            review_status == "skipped"
            and not remaining
            and not audit_unmet
            and not audit_unverifiable
        )

        if review_status in _REVIEW_FAILURE_STATUSES:
            final_gap_status = "failed"
        elif execution_status in _REVIEW_FAILURE_STATUSES:
            final_gap_status = "failed"
        elif (
            execution_status_explicit
            and execution_status
            not in (
                _REVIEW_SUCCESS_STATUSES
                | _REVIEW_FAILURE_STATUSES
                | _REVIEW_PARTIAL_STATUSES
            )
        ):
            final_gap_status = "unverifiable"
        elif verdict_explicit and verdict not in _ASSESSMENT_VERDICTS:
            final_gap_status = "unverifiable"
        elif remaining:
            final_gap_status = "unresolved"
        elif no_action_complete:
            final_gap_status = "not_applicable"
        elif skipped_without_gap:
            final_gap_status = "not_assessed"
        elif (
            review_status not in _REVIEW_SUCCESS_STATUSES
            or execution_status not in _REVIEW_SUCCESS_STATUSES
            or verdict not in _ASSESSMENT_VERDICTS
        ):
            final_gap_status = "unverifiable"
        elif verdict in {"partial", "unresolved"}:
            final_gap_status = "unresolved"
        elif verdict == "resolved":
            final_gap_status = "resolved"
        else:  # Defensive fail-closed branch for future producer values.
            final_gap_status = "unverifiable"
        entry = {
            "review_attempt_id": str(raw.get("review_attempt_id") or f"legacy-{index + 1}"),
            "round_number": round_number,
            "review_status": review_status,
            "review_reason": str(raw.get("review_reason") or ""),
            "final_gap_status": final_gap_status,
            "plan_audit": {
                "n_steps": _integer(audit.get("n_steps", 0)),
                "n_completed": _integer(audit.get("n_completed", 0)),
                "n_skipped": _integer(audit.get("n_skipped", 0)),
                "n_unmet": audit_unmet,
                "n_unverifiable": audit_unverifiable,
                "n_figure_steps": _integer(audit.get("n_figure_steps", 0)),
                "n_figure_unmet": _integer(audit.get("n_figure_unmet", 0)),
                "unmet_titles": _texts(audit.get("unmet_titles"), limit=20),
                "unmet_families": _texts(audit.get("unmet_families"), limit=20),
                "unmet_figure_kinds": _texts(
                    audit.get("unmet_figure_kinds"), limit=20
                ),
            },
            "supplementary_execution": {
                "status": execution_status,
                "summary": str(result.get("summary") or "")[:4000],
                "warnings": warnings,
            },
            "assessment": {
                "verdict": verdict,
                "summary": str(assessment.get("summary") or "")[:4000],
                "resolved_gaps": _texts(assessment.get("resolved_gaps"), limit=20),
                "remaining_gaps": remaining,
            },
        }
        reviews.append(entry)
        label = f"Round {round_number} post-execution review"
        # The plan audit is historical input to the supplementary pass. It is
        # disclosed verbatim above, but only the final assessment may decide
        # whether a gap remains. A completed pass with verdict=resolved and no
        # remaining gaps must not stay partial merely because the pre-pass audit
        # correctly recorded what triggered the repair.
        if final_gap_status == "unresolved":
            blocking.append(f"{label} retained unresolved supplementary gaps.")
        elif final_gap_status == "failed":
            blocking.append(f"{label} failed to resolve its supplementary gaps.")
        elif final_gap_status == "unverifiable":
            blocking.append(f"{label} has no verifiable final gap assessment.")
        if final_gap_status == "not_assessed":
            cautions.append(f"{label} was skipped and has no final gap assessment.")
        if review_status in (_REVIEW_FAILURE_STATUSES | _REVIEW_PARTIAL_STATUSES):
            cautions.append(f"{label} ended with status {review_status}.")
        if execution_status in (_REVIEW_FAILURE_STATUSES | _REVIEW_PARTIAL_STATUSES):
            cautions.append(
                f"{label} supplementary execution ended with status {execution_status}."
            )
        cautions.extend(f"{label}: {warning}" for warning in warnings)
    return reviews, blocking, cautions


def _annotation_disclosure(state: Any) -> tuple[Dict[str, Any], List[str], List[str]]:
    record = _value(state, "annotation_record", None)
    annotation_state = _value(state, "annotation_state", None)
    source = record or annotation_state
    clusters = list(_value(source, "clusters", None) or [])
    if not clusters:
        clusters = list(_value(source, "cluster_annotations", []) or [])

    if record is not None:
        confirmation_status = (
            "auto_confirmed" if bool(_value(record, "auto_confirmed", False))
            else "user_confirmed"
        )
    elif clusters and bool(_value(annotation_state, "confirmed", False)):
        confirmation_status = "confirmed_without_record"
    elif clusters:
        confirmation_status = "pending"
    else:
        confirmation_status = "not_applicable"

    validation_status = str(
        _value(record, "validation_status", None)
        or _value(annotation_state, "validation_status", None)
        or "not_run"
    )
    validation_summary = str(
        _value(record, "validation_summary", None)
        or _value(annotation_state, "validation_summary", None)
        or ""
    )
    validation_text_summary = str(
        _value(record, "validation_text_summary", None)
        or _value(annotation_state, "validation_text_summary", None)
        or ""
    )

    cluster_entries: List[Dict[str, Any]] = []
    known_total = 0
    known_clusters = 0
    for cluster in clusters:
        raw_count = int(_value(cluster, "n_cells", 0) or 0)
        known = bool(_value(cluster, "n_cells_known", False))
        source_name = str(_value(cluster, "n_cells_source", "unknown") or "unknown")
        if raw_count > 0 and not known:
            # Backward-compatible interpretation for old persisted positive
            # counts. Zero remains unknown because it was the historical default.
            known = True
            source_name = "legacy_record"
        if known:
            known_clusters += 1
            known_total += raw_count
        cluster_entries.append({
            "cluster_id": str(_value(cluster, "cluster_id", "") or ""),
            "final_cell_type": str(
                _value(cluster, "final_cell_type", None)
                or _value(cluster, "user_override", None)
                or _value(cluster, "cell_type", "")
                or ""
            ),
            "n_cells": raw_count if known else None,
            "n_cells_known": known,
            "n_cells_source": source_name if known else "unknown",
        })

    annotation = {
        "confirmation": {
            "status": confirmation_status,
            "auto_confirmed": confirmation_status == "auto_confirmed",
        },
        "validation": {
            "status": validation_status,
            "summary": validation_summary,
            "text_summary": validation_text_summary,
        },
        "cell_counts": {
            "known_clusters": known_clusters,
            "unknown_clusters": len(cluster_entries) - known_clusters,
            "known_cell_total": known_total,
            "clusters": cluster_entries,
        },
    }
    blocking: List[str] = []
    cautions: List[str] = []
    if confirmation_status == "pending":
        blocking.append("Cell-type annotation has not been confirmed.")
    elif confirmation_status == "confirmed_without_record":
        cautions.append("Cell-type annotation is confirmed but lacks an immutable record.")
    if clusters and validation_status in {"failed", "partial"}:
        cautions.append(f"Annotation validation ended with status {validation_status}.")
    elif clusters and validation_status in {"not_run", "unknown"}:
        cautions.append(f"Annotation validation status is {validation_status}.")
    if cluster_entries and known_clusters < len(cluster_entries):
        cautions.append(
            f"Cell counts are unknown for {len(cluster_entries) - known_clusters} "
            "annotation cluster(s)."
        )
    return annotation, blocking, cautions


def build_quality_disclosures(state: Any) -> Dict[str, Any]:
    """Build a stable, non-LLM quality record from retained workflow evidence."""
    rounds, round_blocking, round_cautions = _round_disclosures(state)
    reviews, review_blocking, review_cautions = _review_disclosures(state)
    annotation, annotation_blocking, annotation_cautions = _annotation_disclosure(state)
    blocking = _texts(round_blocking + review_blocking + annotation_blocking, limit=100)
    cautions = _texts(round_cautions + review_cautions + annotation_cautions, limit=100)
    status = "incomplete" if blocking else ("qualified" if cautions else "complete")
    return {
        "schema_version": 1,
        "status": status,
        "rounds": rounds,
        "post_execution_reviews": reviews,
        "annotation": annotation,
        "blocking_issues": blocking,
        "cautions": cautions,
    }


def quality_requires_partial(quality: Dict[str, Any] | None) -> bool:
    return str((quality or {}).get("status") or "") == "incomplete"


def build_report_partial_fields(
    *,
    degraded_fallback: bool = False,
    unreported_figures: Iterable[Any] | None = None,
    figureless_sections: Iterable[Any] | None = None,
    approved_partial: bool = False,
    quality_disclosures: Dict[str, Any] | None = None,
) -> tuple[str, str, Optional[str], List[str]]:
    """Return one canonical status/current-step/reason tuple for every writer."""
    unreported = _texts(unreported_figures, limit=100)
    figureless = _texts(figureless_sections, limit=100)
    reasons: List[str] = []
    details: List[str] = []
    if degraded_fallback:
        reasons.append("degraded_fallback")
        details.append("Degraded fallback report")
    if figureless or unreported:
        reasons.append("figure_coverage")
        if unreported and not figureless:
            coverage_text = "unreported figures: " + ", ".join(unreported)
            details.append(
                coverage_text
                if details else "Complete with " + coverage_text
            )
        else:
            coverage: List[str] = []
            if figureless:
                coverage.append("figureless sections: " + ", ".join(figureless))
            if unreported:
                coverage.append("unreported figures: " + ", ".join(unreported))
            coverage_text = "figure coverage gaps: " + "; ".join(coverage)
            details.append(
                coverage_text
                if details else "Complete with " + coverage_text
            )
    if approved_partial:
        reasons.append("approved_partial")
        details.append("partial report approved" if details else "Partial report approved")
    if quality_requires_partial(quality_disclosures):
        reasons.append("analysis_quality")
        details.append(
            "unresolved analysis-quality gaps"
            if details else "Complete with unresolved analysis-quality gaps"
        )
    if not reasons:
        return "completed", "Complete", None, []
    # Keep the first reason as the backward-compatible primary while the list is
    # the complete contract consumed by JSON, WebSocket, completion text and HTML.
    return "partial", "; ".join(details), reasons[0], reasons


def render_report_status_line_html(report: Dict[str, Any] | None) -> str:
    """One always-visible line saying the report is qualified, or nothing.

    Deliberately minimal and deliberately NOT collapsible: the detail belongs in
    the folded block near the conclusions, but whether a reader should trust the
    conclusions at all has to be legible before they read them. Returns "" for a
    report with nothing to qualify, so a clean report gains no banner.
    """
    if not isinstance(report, dict):
        return ""
    reasons = _texts(
        report.get("partial_reasons")
        or ([report.get("partial_reason")] if report.get("partial_reason") else [])
    )
    quality = report.get("quality_disclosures") or {}
    if not isinstance(quality, dict):
        quality = {}
    blocking = _texts(quality.get("blocking_issues"), limit=100)
    cautions = _texts(quality.get("cautions"), limit=100)
    is_partial = str(report.get("status") or "") == "partial" or bool(reasons)
    if not (is_partial or blocking or cautions):
        return ""

    if blocking or str(quality.get("status") or "") == "incomplete":
        tone, headline = "status-blocking", "Partial report — unresolved evidence gaps"
    elif is_partial:
        tone, headline = "status-partial", "Partial report"
    else:
        tone, headline = "status-qualified", "Report carries evidence qualifications"

    counts = []
    if blocking:
        counts.append(f"{len(blocking)} unresolved issue" + ("s" if len(blocking) != 1 else ""))
    if cautions:
        counts.append(f"{len(cautions)} caution" + ("s" if len(cautions) != 1 else ""))
    detail = "; ".join(counts) if counts else "see the qualifications below"
    return (
        f'<div class="report-status-line {tone}">'
        f"<strong>{html.escape(headline)}.</strong> "
        f"{html.escape(detail)} — read the conclusions in that light; the itemised "
        "record is at the end of this report."
        "</div>"
    )


def render_report_limitations_html(report: Dict[str, Any] | None) -> str:
    """Render one collapsed deterministic qualification block at report end.

    The machine-readable report status and quality record remain unchanged. This
    presentation avoids repeating the same partial state in multiple banners
    while keeping every retained issue available for audit.
    """
    if not isinstance(report, dict):
        return ""

    labels = {
        "figure_coverage": "Figure coverage is incomplete.",
        "degraded_fallback": "The report writer used a degraded summary.",
        "approved_partial": "A qualified report was explicitly approved.",
        "analysis_quality": "Retained analysis-quality gaps limit interpretation.",
    }
    reasons = _texts(
        report.get("partial_reasons")
        or ([report.get("partial_reason")] if report.get("partial_reason") else [])
    )
    details = [
        labels.get(reason, reason.replace("_", " ").capitalize() + ".")
        for reason in reasons
    ]
    figureless = _texts(report.get("figureless_sections"), limit=100)
    unreported = _texts(report.get("unreported_figures"), limit=100)
    if figureless:
        details.append("Figureless result sections: " + ", ".join(figureless) + ".")
    if unreported:
        details.append("Unreported figures: " + ", ".join(unreported) + ".")

    quality = report.get("quality_disclosures") or {}
    if not isinstance(quality, dict):
        quality = {}
    details.extend(list(quality.get("blocking_issues") or []))
    details.extend(list(quality.get("cautions") or []))
    details = _texts(details, limit=200)

    # The prose above summarises; it does not name. A caution reads "Round 1
    # post-execution review failed to resolve its supplementary gaps" without
    # saying WHICH step went unmet, and says nothing at all about annotation
    # confirmation or per-round evidence. Those three groups were rendered by the
    # two banners this block replaced in 0.49.3; the machine-readable record kept
    # them but the report stopped showing them, so an audit could no longer see
    # which obligation was missed. They belong here, inside the one collapsed
    # block, which is what "keeping every retained issue available" meant.
    review_items = []
    for review in quality.get("post_execution_reviews") or []:
        if not isinstance(review, dict):
            continue
        audit = review.get("plan_audit") or {}
        assessment = review.get("assessment") or {}
        unmet = _texts(audit.get("unmet_titles"), limit=20)
        remaining = _texts(assessment.get("remaining_gaps"), limit=20)
        if not unmet and not remaining:
            continue
        parts = []
        if unmet:
            parts.append("unmet plan steps: " + ", ".join(unmet))
        if remaining:
            parts.append("remaining gaps: " + ", ".join(remaining))
        label = f"Round {review.get('round_number', 0)} post-execution review"
        review_items.append(
            f"<li><strong>{html.escape(label)}:</strong> "
            f"{html.escape('; '.join(parts))}</li>"
        )

    # A block with nothing to disclose stays absent: annotation state and the
    # round table are context for a qualification, never a qualification on
    # their own, so they must not make a clean report grow an empty section.
    if not details and not review_items:
        return ""

    body = ""
    if details:
        body += "<ul>" + "".join(
            f"<li>{html.escape(item)}</li>" for item in details
        ) + "</ul>"
    if review_items:
        body += (
            "<h3>Post-execution review gaps</h3><ul>"
            + "".join(review_items)
            + "</ul>"
        )

    rows = []
    for result in quality.get("rounds") or []:
        if not isinstance(result, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('round_number', '')))}</td>"
            f"<td>{html.escape(str(result.get('round_status', '')))}</td>"
            f"<td>{html.escape(str(result.get('evidence_status', '')))}</td>"
            f"<td>{html.escape(str(result.get('direction_coverage_eligible')))}</td>"
            f"<td>{html.escape(str(result.get('bioinformatics_evidence_verified')))}</td>"
            "</tr>"
        )
    if rows:
        body += (
            "<table><thead><tr><th>Round</th><th>Status</th><th>Evidence</th>"
            "<th>Coverage eligible</th><th>Evidence verified</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    annotation = quality.get("annotation") or {}
    if isinstance(annotation, dict) and annotation:
        confirmation = (annotation.get("confirmation") or {}).get("status", "unknown")
        validation = (annotation.get("validation") or {}).get("status", "unknown")
        counts = annotation.get("cell_counts") or {}
        body += (
            "<p><strong>Annotation confirmation:</strong> "
            f"{html.escape(str(confirmation))}; "
            "<strong>validation:</strong> "
            f"{html.escape(str(validation))}; "
            "<strong>cluster cell counts:</strong> "
            f"{int(counts.get('known_clusters', 0) or 0)} known / "
            f"{int(counts.get('unknown_clusters', 0) or 0)} unknown.</p>"
        )
        validation_summary = str(
            (annotation.get("validation") or {}).get("summary") or ""
        )
        if validation_summary:
            body += (
                "<p><strong>Annotation validation summary:</strong> "
                f"{html.escape(validation_summary)}</p>"
            )

    return (
        '<details class="report-limitations-details">'
        "<summary>Detailed evidence qualifications</summary>"
        "<p>The PI summary above consolidates the principal limitations; "
        "the retained machine-derived details are listed here.</p>"
        f"{body}</details>"
    )
