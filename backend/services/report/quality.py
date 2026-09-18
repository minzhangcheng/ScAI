"""Deterministic analysis-quality disclosures shared by every report surface."""
from __future__ import annotations

import html as html_module
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
                cautions.append(f"{label} has {evidence} evidence status.")
        if current_evidence_schema and phase != "integration":
            if coverage is False:
                cautions.append(f"{label} is not eligible for direction coverage.")
            elif coverage is None:
                cautions.append(
                    f"{label} has no verifiable direction-coverage verdict."
                )
            elif coverage is not True:
                blocking.append(
                    f"{label} has malformed direction-coverage value {coverage!r}."
                )
        if current_evidence_schema and phase != "integration":
            if verified is False:
                cautions.append(
                    f"{label} has bioinformatics findings that are not linked to "
                    "a current successful execution receipt."
                )
            elif verified is None:
                cautions.append(
                    f"{label} has no recorded bioinformatics execution-linkage verdict."
                )
            elif verified is not True:
                blocking.append(
                    f"{label} has malformed bioinformatics evidence value {verified!r}."
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
        audit_steps = _integer(audit.get("n_steps", 0))
        technical_snapshots = (
            raw.get("technical_snapshots")
            if isinstance(raw.get("technical_snapshots"), dict)
            else {}
        )
        review_snapshot = (
            technical_snapshots.get("at_review")
            if isinstance(technical_snapshots.get("at_review"), dict)
            else {}
        )
        final_snapshot = (
            technical_snapshots.get("after_figure_followup")
            if isinstance(technical_snapshots.get("after_figure_followup"), dict)
            else {}
        )
        figure_at_review = (
            review_snapshot.get("figure_contract")
            if isinstance(review_snapshot.get("figure_contract"), dict)
            else {}
        )
        final_figure = (
            final_snapshot.get("figure_contract")
            if isinstance(final_snapshot.get("figure_contract"), dict)
            else {}
        )
        if final_figure:
            figure_gap_status = (
                "fulfilled" if final_figure.get("passed") is True else "missing"
            )
        else:
            figure_gap_status = "unverifiable"
        review_plan = (
            raw.get("review_plan")
            if isinstance(raw.get("review_plan"), dict)
            else {}
        )
        step_decisions = {
            _integer(item.get("plan_step_id", 0)): str(
                item.get("status") or "pending"
            ).lower()
            for item in (review_plan.get("step_decisions") or [])
            if isinstance(item, dict)
            and _integer(item.get("plan_step_id", 0)) > 0
        }
        audited_step_ids = {
            _integer(item.get("plan_step_id", 0))
            for item in (audit.get("step_observations") or [])
            if isinstance(item, dict)
            and _integer(item.get("plan_step_id", 0)) > 0
        }
        missing_pi_step_ids = sorted(audited_step_ids - set(step_decisions))
        pending_pi_step_ids = sorted(
            step_id
            for step_id, status in step_decisions.items()
            if status == "pending"
        )
        missing_pi_decision_count = (
            len(missing_pi_step_ids)
            if audited_step_ids
            else max(audit_steps - len(step_decisions), 0)
        )
        pi_steps_resolved = (
            len(step_decisions) == audit_steps
            and (not audited_step_ids or set(step_decisions) == audited_step_ids)
            and all(
                status in {"completed", "skipped"}
                for status in step_decisions.values()
            )
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
            and pi_steps_resolved
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
        elif (
            review_status == "no_action"
            and str(review_plan.get("verdict") or "").lower() == "no_action"
            and not remaining
            and not pi_steps_resolved
        ):
            final_gap_status = "pending_pi_decision"
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
            "lifecycle": str(raw.get("lifecycle") or raw.get("lifecycle_status") or ""),
            "lifecycle_status": str(raw.get("lifecycle_status") or ""),
            "persistence_status": str(raw.get("persistence_status") or ""),
            "persistence_failure": str(raw.get("persistence_failure") or "")[:4000],
            "final_gap_status": final_gap_status,
            "figure_gap_status": figure_gap_status,
            "figure_contract_at_review": {
                "passed": figure_at_review.get("passed"),
                "n_requirements": _integer(figure_at_review.get("n_requirements", 0)),
                "n_missing": _integer(figure_at_review.get("n_missing", 0)),
            },
            "final_figure_contract": {
                "passed": final_figure.get("passed"),
                "n_requirements": _integer(final_figure.get("n_requirements", 0)),
                "n_missing": _integer(final_figure.get("n_missing", 0)),
                "eligible_figure_names": _texts(
                    final_figure.get("eligible_figure_names"), limit=100
                ),
            },
            "plan_audit": {
                "n_steps": audit_steps,
                "n_completed": _integer(audit.get("n_completed", 0)),
                "n_skipped": _integer(audit.get("n_skipped", 0)),
                "n_unmet": audit_unmet,
                "n_unverifiable": audit_unverifiable,
                "n_awaiting_pi_decision": _integer(
                    audit.get("n_awaiting_pi_decision", 0)
                ),
                "n_figure_steps": _integer(audit.get("n_figure_steps", 0)),
                "n_figure_unmet": _integer(audit.get("n_figure_unmet", 0)),
                "n_family_unmet": _integer(audit.get("n_family_unmet", 0)),
                "unmet_titles": _texts(audit.get("unmet_titles"), limit=20),
                "unmet_families": _texts(audit.get("unmet_families"), limit=20),
                "unmet_figure_kinds": _texts(
                    audit.get("unmet_figure_kinds"), limit=20
                ),
            },
            "pi_completion": {
                "decisions": [
                    {
                        "plan_step_id": step_id,
                        "status": status,
                    }
                    for step_id, status in sorted(step_decisions.items())[:100]
                ],
                "missing_decision_count": missing_pi_decision_count,
                "missing_step_ids": missing_pi_step_ids[:100],
                "pending_step_ids": pending_pi_step_ids[:100],
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
            cautions.append(f"{label} retained unresolved supplementary gaps.")
        elif final_gap_status == "failed":
            cautions.append(f"{label} failed to resolve its supplementary gaps.")
        elif final_gap_status == "unverifiable":
            cautions.append(f"{label} has no usable final gap assessment record.")
        elif final_gap_status == "pending_pi_decision":
            if missing_pi_decision_count:
                cautions.append(
                    f"{label} has no recorded PI completion decision for "
                    f"{missing_pi_decision_count} plan step(s)."
                )
            if pending_pi_step_ids:
                cautions.append(
                    f"{label} retained PI-pending plan step(s): "
                    + ", ".join(str(value) for value in pending_pi_step_ids)
                    + "."
                )
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
    missing_sections: Iterable[Any] | None = None,
    approved_partial: bool = False,
    quality_disclosures: Dict[str, Any] | None = None,
) -> tuple[str, str, Optional[str], List[str]]:
    """Return one canonical status/current-step/reason tuple for every writer."""
    unreported = _texts(unreported_figures, limit=100)
    figureless = _texts(figureless_sections, limit=100)
    missing = _texts(missing_sections, limit=100)
    reasons: List[str] = []
    details: List[str] = []
    if degraded_fallback:
        reasons.append("degraded_fallback")
        details.append("Degraded fallback report")
    # A planned chapter that never made it into the body is a larger gap than a
    # chapter that made it without a figure, so it is named first. Naming is all
    # this does: the report is written, published and downloadable either way.
    if missing:
        reasons.append("section_coverage")
        section_text = "planned sections absent from the report: " + ", ".join(missing)
        details.append(
            section_text if details else "Complete with " + section_text
        )
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
    """Compatibility hook for the retired report-level status banner.

    Report status and qualification data remain machine-readable and continue to
    inform the ordinary Limitations prose. They are no longer repeated as a
    prominent deterministic HTML block.
    """
    return ""


def render_report_limitations_html(report: Dict[str, Any] | None) -> str:
    """Compatibility hook for the retired detailed qualification block.

    The underlying partial reasons and quality disclosures remain in report data
    for API, persistence, regeneration, and audit consumers.
    """
    return ""


def render_figure_reservation_notice_html(notice: Any) -> str:
    """Render one persisted reservation without relying on LLM-authored prose."""
    if not isinstance(notice, dict) or not notice:
        return ""
    summary = " ".join(str(notice.get("summary") or "").split()).strip()
    details = _texts(notice.get("details"), limit=12)
    if not summary and not details:
        return ""
    body = "".join(f"<li>{html_module.escape(item)}</li>" for item in details)
    detail_html = f"<ul>{body}</ul>" if body else ""
    return (
        '<div class="figure-reservation-notice" role="note">'
        f"<strong>{html_module.escape(summary or 'PI accepted this figure with reservations.')}</strong>"
        f"{detail_html}</div>"
    )


def render_technical_review_notice_html(report: Dict[str, Any] | None) -> str:
    """Render explicit technical-review and retained-reservation observations.

    This presentation notice deliberately reads retained report metadata rather
    than scientific cautions or the generic report status.  Missing/legacy
    availability fields remain unknown and therefore produce no notice.
    """
    if not isinstance(report, dict):
        return ""

    observations: List[str] = []
    seen_figures: set[str] = set()
    figure_records: List[Any] = list(report.get("figures") or [])
    for section in report.get("sections") or []:
        if isinstance(section, dict):
            figure_records.extend(section.get("figures") or [])

    for figure in figure_records:
        if not isinstance(figure, dict):
            continue
        # Workspace-only/legacy figures have no publication review contract.
        if figure.get("is_publication") is False:
            continue
        identity = str(
            figure.get("id")
            or figure.get("logical_figure_id")
            or figure.get("filename")
            or figure.get("png_path")
            or figure.get("name")
            or "figure"
        )
        if identity in seen_figures:
            continue
        seen_figures.add(identity)
        review_status = str(figure.get("review_status") or "").lower()
        review_unavailable = figure.get("review_unavailable") is True
        reservation = figure.get("reservation_notice")
        if isinstance(reservation, dict) and reservation:
            label = str(
                figure.get("filename")
                or figure.get("name")
                or "selected publication figure"
            ).strip()[:160]
            summary = " ".join(str(reservation.get("summary") or "").split())
            observations.append(
                f"Figure review for {label}: "
                f"{summary or 'PI accepted this figure with reservations.'}"
            )
        if review_unavailable or review_status == "pending":
            label = str(
                figure.get("filename")
                or figure.get("name")
                or "selected publication figure"
            ).strip()[:160]
            if review_unavailable:
                observations.append(f"Visual review unavailable for {label}.")
            else:
                observations.append(f"Visual review is pending for {label}.")

    graphic = report.get("graphic_abstract")
    if not isinstance(graphic, dict):
        graphic = report.get("graphic_abstract_metadata")
    if isinstance(graphic, dict):
        if graphic.get("review_available") is False:
            observations.append("Graphic abstract visual review was unavailable.")
        if graphic.get("selection_available") is False:
            observations.append("Graphic abstract candidate selection was unavailable.")

    quality = report.get("quality_disclosures")
    post_reviews = quality.get("post_execution_reviews", []) if isinstance(quality, dict) else []
    review_statuses = {
        "unavailable", "failed", "error", "interrupted", "pending", "reserved",
        "reservation_failed",
    }
    lifecycle_statuses = {
        "reserved", "running", "pending", "interrupted", "reservation_failed",
        "terminal_persistence_pending",
    }
    persistence_statuses = {"pending", "failed", "error"}
    execution_failure_statuses = {
        "failed", "error", "timeout", "max_steps_reached",
        "completed_with_fallback", "interrupted",
    }
    no_execution_statuses = {"", "unknown", "not_run", "pending", "reserved"}
    pre_execution_failure_statuses = {
        "failed", "error", "unavailable", "interrupted", "reservation_failed",
    }
    for review in post_reviews or []:
        if not isinstance(review, dict):
            continue
        statuses = {
            key: str(review.get(key) or "").lower()
            for key in (
                "review_status", "lifecycle", "lifecycle_status",
                "persistence_status",
            )
        }
        supplementary = review.get("supplementary_execution")
        supplementary = supplementary if isinstance(supplementary, dict) else {}
        execution_status = str(supplementary.get("status") or "").lower()
        review_reason = str(review.get("review_reason") or "").strip()
        persistence_failure = str(review.get("persistence_failure") or "").strip()
        explicit = (
            statuses["review_status"] in review_statuses
            or statuses["lifecycle"] in lifecycle_statuses
            or statuses["lifecycle_status"] in lifecycle_statuses
            or statuses["persistence_status"] in persistence_statuses
        )
        if not explicit:
            continue
        if persistence_failure:
            # This field is populated by the persistence boundary itself and is
            # therefore a retained technical reason, independent of PI wording.
            detail = persistence_failure[:240]
        elif execution_status in execution_failure_statuses:
            # Once supplementary execution has a terminal result, review_reason
            # describes why PI requested it and may contain scientific gaps.
            detail = "Supplementary technical review execution failed"
        elif statuses["persistence_status"] in persistence_statuses:
            detail = "Technical review persistence status is unavailable"
        elif (
            statuses["review_status"] in {"pending", "reserved"}
            or statuses["lifecycle"] in {"pending", "reserved", "running"}
            or statuses["lifecycle_status"] in {"pending", "reserved", "running"}
        ):
            detail = "Technical review remains pending before a terminal result"
        elif (
            statuses["review_status"] == "reservation_failed"
            or statuses["lifecycle"] == "reservation_failed"
            or statuses["lifecycle_status"] == "reservation_failed"
        ):
            detail = "Technical review reservation failed"
        elif (
            execution_status in no_execution_statuses
            and statuses["review_status"] in pre_execution_failure_statuses
        ):
            # API/unavailable/interrupted/reservation failures before a
            # supplementary pass may retain their technical boundary reason.
            detail = review_reason[:240] if review_reason else (
                "Technical review was unavailable before supplementary execution"
            )
        else:
            detail = "Technical review availability could not be confirmed"
        observations.append(f"Post-execution technical review: {detail}.")

    if not observations:
        return ""
    rendered = []
    for observation in observations[:8]:
        rendered.append(f"<li>{html_module.escape(observation)}</li>")
    if len(observations) > 8:
        rendered.append(
            "<li>Additional technical review observations: "
            f"{len(observations) - 8}.</li>"
        )
    return (
        '<div class="technical-review-notice" role="note">'
        "<strong>Technical review observations:</strong>"
        "<ul>"
        + "".join(rendered)
        + "</ul><p>Report generation continued; these notices disclose technical "
        "review state and retained figure reservations only.</p></div>"
    )


def render_artifact_drift_notice_html(artifact):
    """Disclose recorded data changes without rewriting the saved caption."""
    from .artifacts import data_drift_warning
    from html import escape
    provenance = getattr(artifact, "expression_provenance", None) or {}
    notices = list(dict.fromkeys(str(provenance[k]) for k in (
        "cohort_change_warning", "representation_change_warning"
    ) if provenance.get(k)))
    legacy_notice = data_drift_warning(artifact)
    if legacy_notice and legacy_notice not in notices:
        notices.append(legacy_notice)
    return "".join(f'<p class="artifact-drift-notice" role="note">{escape(n)}</p>' for n in notices)
