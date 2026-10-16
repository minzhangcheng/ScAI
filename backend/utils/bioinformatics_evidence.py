"""Compact, source-linked Bioinformatics briefs for PI decisions.

Interpretations belong to the agents. These helpers retain submitted summaries
and copy nominated method records/table rows without assessing their meaning.
"""
import csv
import hashlib
import io
import json
import logging
from itertools import islice
from pathlib import Path

logger = logging.getLogger(__name__)


def _strings(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, (list, tuple)) else []


def _failure_note(stage, error, reference=""):
    # Import lazily: workflow.__init__ imports agents that use this module.
    from ..workflow.helpers import WorkflowSupersededError, WorkflowTerminatedError

    if isinstance(error, (WorkflowSupersededError, WorkflowTerminatedError)):
        raise error
    reference = reference[:120] if isinstance(reference, str) else "unknown"
    logger.warning("Optional evidence %s failed for %s: %s", stage, reference, type(error).__name__)
    return f"Optional evidence {stage} failed ({type(error).__name__}); original findings and execution references are retained."


def record_evidence_context_failure(records, *, stage, error):
    """Persist an optional-context failure without changing execution status."""
    note = _failure_note(stage, error)
    for record in records:
        if isinstance(record, dict):
            notes = _strings(record.get("evidence_context_notes"))
            if note not in notes:
                notes.append(note)
            record["evidence_context_notes"] = notes[-8:]


EVIDENCE_REVIEW_GUIDANCE = """Use the Bioinformatics-authored evidence briefs below
to distinguish performed comparisons, supporting results, opposing results,
alternative explanations, and open questions. Source excerpts are recorded
methods and values, not Host confirmation of a scientific interpretation.
Read a score's recorded genes and method before interpreting its name; distinguish
a mixed response score from a specific biological program using the actual panel.
Missing comparisons or measurements are unresolved questions, not negative results.
Decide which unresolved candidates merit follow-up using all available modalities.
When materially revising an earlier PI conclusion, explain in the summary and
conclusions: the original claim, whether it is retained, qualified, superseded,
or unresolved, the current claim, and the reason with source references. Preserve
the measured result even when its interpretation changes. If there is no new
measurement, distinguish reinterpretation from new contradictory evidence.
"""

_LIST_FIELDS = (
    "supporting_results", "opposing_results", "alternative_explanations",
    "open_questions", "score_ids",
)


def normalize_evidence_summary(value):
    """Bound optional report metadata; absent briefs never reject a finding."""
    if not isinstance(value, dict):
        return {}
    result = {}
    omitted = {}
    malformed = set()
    if isinstance(value.get("comparison"), str):
        result["comparison"] = value["comparison"][:600]
    elif value.get("comparison") is not None:
        malformed.add("comparison")
    for key in _LIST_FIELDS:
        values = value.get(key)
        if isinstance(values, list):
            strings = [item for item in values if isinstance(item, str) and item.strip()]
            result[key] = [item[:480] for item in strings[:8]]
            if any(not isinstance(item, str) for item in values):
                malformed.add(key)
            if len(strings) > 8:
                omitted[key] = len(strings) - 8
        elif values is not None:
            malformed.add(key)
    references = value.get("table_rows")
    if isinstance(references, list):
        result["table_rows"] = []
        for reference in references[:3]:
            if not isinstance(reference, dict):
                malformed.add("table_rows")
                continue
            match = reference.get("row_match")
            columns = reference.get("columns")
            if not isinstance(reference.get("table_id"), str) or not isinstance(match, dict):
                malformed.add("table_rows")
                continue
            if any(not isinstance(key, str) or not isinstance(item, (str, int, float, bool)) for key, item in match.items()):
                malformed.add("table_rows")
                # Dropping one selector would broaden the requested row identity.
                continue
            if columns is not None and (not isinstance(columns, list) or any(not isinstance(item, str) for item in columns)):
                malformed.add("table_rows")
            result["table_rows"].append({
                "table_id": reference["table_id"][:120],
                "row_match": {
                    key[:200]: str(item)[:200]
                    for key, item in list(match.items())[:4]
                    if isinstance(key, str) and isinstance(item, (str, int, float, bool))
                },
                "columns": [item[:200] for item in columns[:8] if isinstance(item, str)] if isinstance(columns, list) else [],
            })
        if len(references) > 3:
            omitted["table_rows"] = len(references) - 3
    elif references is not None:
        malformed.add("table_rows")
    if omitted:
        result["omitted_items"] = omitted
    if malformed:
        result["format_notes"] = ["Malformed optional metadata omitted: " + ", ".join(sorted(malformed)) + "."]
    return result


def _table_excerpt(table, reference):
    source = {key: getattr(table, key, None) for key in (
        "id", "csv_sha256", "round_number", "lineage_id", "source_execution_id",
        "cohort_digest", "is_current", "producer_status",
    )}
    source["kind"] = "table"
    source["row_match"] = reference["row_match"]
    source["rows"] = []
    try:
        path = Path(table.csv_path)
        with path.open("rb") as handle:
            payload = handle.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            source["note"] = "Table exceeds the bounded excerpt read; CSV remains available."
            return source
        actual_hash = hashlib.sha256(payload).hexdigest()
        source["observed_sha256"] = actual_hash
        if table.csv_sha256 and table.csv_sha256 != actual_hash:
            source["note"] = "CSV differs from the registered hash; no source rows copied."
            return source
        source["note"] = "Registered hash matched." if table.csv_sha256 else "Registered hash absent."
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        match = reference["row_match"]
        if not match or not all(key in (reader.fieldnames or []) for key in match):
            source["note"] += " No addressable row match was supplied."
            return source
        columns = reference["columns"] or (reader.fieldnames or [])[:8]
        for index, row in enumerate(islice(reader, 100000)):
            if all(row.get(key) == value for key, value in match.items()):
                source["rows"].append({
                    "row_index": index,
                    "values": {key: row.get(key) for key in columns},
                })
                if len(source["rows"]) == 3:
                    source["note"] += " At most three matching rows are shown."
                    break
        if not source["rows"]:
            source["note"] += " No matching row found within the bounded read."
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        source["note"] = f"Source excerpt unavailable: {type(exc).__name__}."
    return source


def attach_evidence_sources(state, records):
    """Freeze the submitted brief's exact references before later score updates."""
    scores = {score.id: score for score in state.score_definitions}
    plans = {plan.id: plan for plan in state.scoring_plans}
    tables = {table.id: table for table in state.publication_tables}
    for record in records:
        summary = normalize_evidence_summary(record.get("evidence_summary"))
        if not summary:
            continue
        sources = []
        record["evidence_sources"] = sources
        for score_id in summary.get("score_ids") or []:
            source = {"kind": "score", "id": score_id}
            try:
                score = scores.get(score_id)
                if score is None:
                    source["note"] = "Referenced score ID not found."
                else:
                    source.update(score.model_dump(mode="json", include={
                        "id", "score_name", "method", "gene_list", "gene_list_source",
                        "definition_kind", "definition_status", "definition_artifacts",
                        "n_genes", "n_genes_matched", "panel_hash", "plan_id",
                        "plan_drift", "round_number", "parameters",
                    }))
                    plan = plans.get(score.plan_id)
                    if plan is not None:
                        try:
                            source["frozen_plan"] = plan.model_dump(mode="json", include={
                                "id", "hypothesis", "falsifier", "panel_genes", "panel_source",
                                "baseline_genes", "baseline_source", "edit_operations",
                            })
                        except Exception as exc:
                            source["note"] = _failure_note("frozen-plan snapshot", exc, plan.id)
            except Exception as exc:
                source["note"] = _failure_note("score snapshot", exc, score_id)
            sources.append(source)
        for reference in summary.get("table_rows") or []:
            table_id = reference["table_id"]
            try:
                table = tables.get(table_id)
                source = _table_excerpt(table, reference) if table is not None else {
                    "kind": "table", "id": table_id, "note": "Referenced table ID not found.",
                }
            except Exception as exc:
                source = {
                    "kind": "table", "id": table_id, "rows": [],
                    "note": _failure_note("table excerpt", exc, table_id),
                }
            sources.append(source)


def evidence_brief_references(records):
    """Keep exact join keys available when a parent prompt omits brief details."""
    references = []
    for record in records if isinstance(records, (list, tuple)) else []:
        if not isinstance(record, dict) or not (record.get("evidence_summary") or record.get("evidence_context_notes")):
            continue
        references.extend(_strings(record.get("id")))
        references.extend(_strings(record.get("evidence_execution_ids")) or _strings(record.get("candidate_evidence_execution_ids")))
        sources = record.get("evidence_sources")
        for source in sources if isinstance(sources, (list, tuple)) else []:
            if isinstance(source, dict):
                references.extend(_strings(source.get("id")))
                references.extend(_strings(source.get("source_execution_id")))
    return list(dict.fromkeys(references))


def _fallback_card(record, note, share):
    """Keep the original claim and exact references under the existing budget."""
    references = _strings(record.get("evidence_execution_ids")) or _strings(record.get("candidate_evidence_execution_ids"))
    identity = record.get("id") if isinstance(record.get("id"), str) else "?"
    prefix = f"Finding {identity}\n{note}\nExecution references: " + ", ".join(references[:8]) + "\nAgent interpretation: "
    text = record.get("text") if isinstance(record.get("text"), str) else "Original finding text unavailable."
    return (prefix + text[:max(0, share - len(prefix))])[:share]


def render_evidence_briefs(records, *, max_chars=6000):
    """Optional context cannot prevent a PI decision or report generation."""
    try:
        if records is None:
            return ""
        if not isinstance(records, (list, tuple)):
            raise TypeError("Finding records must be a sequence")
        return _render_evidence_briefs(records, max_chars=max_chars)
    except Exception as exc:
        note = _failure_note("brief rendering", exc)
        valid = [record for record in records if isinstance(record, dict)] if isinstance(records, (list, tuple)) else []
        if not valid:
            return note[:max_chars]
        shown = valid if len(valid) <= 4 else valid[:2] + valid[-2:]
        share = max(0, (max_chars - len(note) - len(shown)) // len(shown))
        return (note + "\n" + "\n".join(_fallback_card(record, note, share) for record in shown))[:max_chars]


def _render_card(record, share):
    summary = record.get("evidence_summary") or {}
    if not isinstance(summary, dict):
        raise TypeError("Evidence summary must be an object")
    lines = [f"Finding {record.get('id', '?')}; round {record.get('round_number', '?')}; "
             f"lineage {record.get('lineage_id', 'unknown')}"]
    parts = [("Agent interpretation", str(record.get("text") or ""), 1)]
    for field in ("comparison", *_LIST_FIELDS[:-1]):
        values = summary.get(field) or []
        strings = _strings(values)
        text = "; ".join(strings)
        if (not isinstance(values, (str, list, tuple))) or (
            isinstance(values, (list, tuple)) and any(not isinstance(item, str) for item in values)
        ):
            text += " [Malformed metadata omitted from this field.]"
        parts.append((field, text or "not recorded", 1))
    notes = _strings(record.get("evidence_context_notes")) + _strings(summary.get("format_notes"))
    if notes:
        parts.insert(0, ("Technical context notes", "; ".join(notes[:8]), 2))
    ids = _strings(record.get("evidence_execution_ids")) or _strings(record.get("candidate_evidence_execution_ids"))
    lines.append("Execution references: " + ", ".join(ids[:8]))
    sources = record.get("evidence_sources") or []
    if not isinstance(sources, (list, tuple)):
        parts.append(("Source notes", "Malformed source metadata omitted.", 1))
        sources = []
    for source in sources:
        if not isinstance(source, dict):
            parts.append(("Source notes", "Malformed source metadata omitted.", 1))
            continue
        try:
            encoded = _source_preview(source)
        except Exception as exc:
            encoded = _failure_note("source rendering", exc, source.get("id", ""))
        label = f"Recorded {source.get('kind')} source {source.get('id')}"
        if source.get("source_execution_id"):
            label += f" (execution {source['source_execution_id']})"
        parts.append((label, encoded, 2))
    if summary.get("omitted_items"):
        lines.append("Submission omissions: " + json.dumps(summary["omitted_items"]))
    note = "Details shortened; full brief and frozen sources remain in this finding record."
    fixed = len("\n".join(lines)) + sum(len(label) + 3 for label, _, _ in parts) + len(note) + 2
    if fixed >= share:
        return (lines[0] + "\nContext capped; full brief and sources remain in this finding record.")[:share]
    width = max(0, (share - fixed) // sum(weight for _, _, weight in parts))
    shortened = False
    for label, text, weight in parts:
        cap = width * weight
        shortened = shortened or len(text) > cap
        lines.append(f"{label}: " + (text if len(text) <= cap else text[:max(0, cap - 3)] + "..."))
    if shortened:
        lines.append(note)
    return "\n".join(lines)


def _source_preview(source):
    preview = dict(source)
    if source.get("kind") == "score":
        genes = source.get("gene_list") or []
        parameters = source.get("parameters") or {}
        preview = {
            "method": source.get("method"),
            "definition_kind": source.get("definition_kind"),
            "definition_status": source.get("definition_status"),
            "definition_artifacts": source.get("definition_artifacts") or [],
            "gene_list": genes[:16],
            "omitted_genes": max(0, len(genes) - 16),
            "n_genes_matched": source.get("n_genes_matched"),
            "matched_genes": parameters.get("matched_genes", "not recorded"),
            "missing_genes": parameters.get("missing_genes", "not recorded"),
            "gene_list_source": source.get("gene_list_source"),
            "round_number": source.get("round_number"),
        }
        if source.get("note"):
            preview = {"note": source["note"], **preview}
        if source.get("frozen_plan"):
            preview["frozen_panel_genes"] = source["frozen_plan"].get("panel_genes", [])[:16]
    elif source.get("kind") == "table":
        # Put nominated values before metadata, so a compact view still
        # shows the numbers rather than only a table's existence.
        preview = {key: source.get(key) for key in (
            "rows", "note", "source_execution_id", "csv_sha256",
            "round_number", "lineage_id", "cohort_digest", "is_current",
        )}
    return json.dumps(preview, ensure_ascii=False)


def _render_evidence_briefs(records, *, max_chars=6000):
    """Render balanced briefs, with explicit omissions and no notebook dump."""
    selected = [record for record in records
                if isinstance(record, dict)
                and (record.get("evidence_summary") or record.get("evidence_context_notes"))]
    if not selected:
        return ""
    header = ("## Bioinformatics evidence briefs\n"
              "Bioinformatics-authored summaries with recorded source excerpts; "
              "source linkage is not scientific confirmation. "
              "Untested comparisons remain open questions, not negative results.\n")
    # Retain both early observations and later counter-evidence under the cap.
    shown = selected if len(selected) <= 4 else selected[:2] + selected[-2:]
    omitted = len(selected) - len(shown)
    trailer = (
        f"\nContext capped; {omitted} brief(s) omitted. Full briefs and sources remain in finding records.\n"
        + "Finding references: " + ", ".join(str(item.get("id") or "?") for item in selected[:12])
    ) if omitted else ""
    trailer = trailer[:500]
    share = (max_chars - len(header) - len(trailer) - 1) // len(shown)
    if share < 250:
        return ("Bioinformatics brief context capped; consult persisted finding records. "
                + ", ".join(str(item.get("id") or "?") for item in selected[:12]))[:max_chars]
    cards = []
    for record in shown:
        try:
            cards.append(_render_card(record, share))
        except Exception as exc:
            note = _failure_note("brief rendering", exc, record.get("id", ""))
            cards.append(_fallback_card(record, note, share))
    return header + "\n" + "\n".join(cards) + trailer
