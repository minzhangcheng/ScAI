"""Read-only HTML renderers for user-interaction records + a file writer.

Standalone HTML (embedded <style>, native <details>, no JS, no Tailwind).
Reproduces the look of AnnotationConfirmationDialog and ExplorationReportDialog.
"""
from html import escape as _esc
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...models.schemas import AnalysisState, DirectionSelectionRecord, AnnotationRecord
from ...utils.formatting import as_list

_STYLE = """
<style>
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#1e293b;
  max-width:900px;margin:24px auto;padding:0 16px;line-height:1.5;}
.hdr{border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin-bottom:16px;}
.hdr h1{font-size:18px;margin:0;} .hdr .sub{font-size:13px;color:#64748b;margin-top:4px;}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:9999px;}
.b-high{background:#d1fae5;color:#047857;} .b-med{background:#fef3c7;color:#b45309;}
.b-low{background:#fee2e2;color:#b91c1c;} .b-slate{background:#f1f5f9;color:#475569;}
.card{border:2px solid #a78bfa;background:#f5f3ff80;border-radius:12px;padding:14px;margin-bottom:10px;}
.card.unsel{border:2px dashed #e2e8f0;background:#f8fafc;opacity:.8;}
.mark{width:22px;height:22px;border-radius:5px;display:inline-flex;align-items:center;
  justify-content:center;font-weight:700;font-size:15px;flex-shrink:0;}
.mark.on{background:#7c3aed;color:#fff;} .mark.off{background:#fef2f2;color:#dc2626;border:1px solid #fecaca;}
.row{display:flex;gap:12px;align-items:flex-start;}
.note{margin-top:10px;background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:8px 10px;font-size:13px;}
.note .lbl{font-size:11px;font-weight:600;color:#7c3aed;text-transform:uppercase;}
.tag{background:#f1f5f9;color:#475569;font-size:10px;font-weight:600;padding:1px 6px;border-radius:6px;margin-left:6px;}
.banner{background:#ede9fe;border:1px solid #c4b5fd;border-radius:10px;padding:12px 14px;margin-bottom:14px;}
table{width:100%;border-collapse:collapse;font-size:13px;}
th{text-align:left;color:#64748b;border-bottom:1px solid #e2e8f0;padding:8px;}
td{padding:8px;border-bottom:1px solid #f1f5f9;vertical-align:top;}
details>summary{cursor:pointer;} .muted{color:#94a3b8;font-size:12px;}
</style>
"""


def _doc(title: str, body: str) -> str:
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{_esc(title)}</title>{_STYLE}</head><body>{body}</body></html>")


def _conf_badge(conf: str) -> str:
    cls = {"high": "b-high", "medium": "b-med", "low": "b-low"}.get(conf, "b-slate")
    return f"<span class='badge {cls}'>{_esc(conf)}</span>"


def _score_badge(label: str, val: Any) -> str:
    if not val:
        return ""
    return f"<span class='muted'>{_esc(label)}:</span> <span class='badge b-slate'>{_esc(str(val))}</span> "


def _candidate_detail(c: Dict[str, Any]) -> str:
    parts: List[str] = []
    badges = "".join([
        _score_badge("Priority", c.get("overall_priority") or c.get("expected_innovation")),
        _score_badge("Data", c.get("data_reliability")),
        _score_badge("Lit novelty", c.get("literature_novelty") or c.get("expected_novelty")),
        _score_badge("Impact", c.get("functional_impact")),
        _score_badge("Recommendation", c.get("recommendation")),
        _score_badge("Plausibility", c.get("biological_plausibility")),
        _score_badge("Artifact risk", c.get("artifact_risk")),
    ])
    if badges:
        parts.append(f"<div style='margin:6px 0'>{badges}</div>")
    for key, label in [("rationale", "Rationale"), ("core_observation", "Core observation"),
                       ("main_hypothesis", "Main hypothesis"), ("pi_review_summary", "PI review")]:
        if c.get(key):
            parts.append(f"<div><b>{label}:</b> {_esc(str(c[key]))}</div>")
    supporting_data = as_list(c.get("supporting_data"))
    if supporting_data:
        rows = []
        for item in supporting_data:
            if isinstance(item, dict):
                source = item.get("source") or "unknown"
                evidence = item.get("evidence") or ""
                strength = item.get("strength") or ""
                suffix = f" (strength: {strength})" if strength else ""
                rows.append(f"<li>[{_esc(str(source))}] {_esc(str(evidence))}{_esc(suffix)}</li>")
            else:
                rows.append(f"<li>{_esc(str(item))}</li>")
        parts.append(f"<div><b>Supporting data:</b><ul>{''.join(rows)}</ul></div>")
    for key, label in [("alternative_hypotheses", "Alternative hypotheses"),
                       ("artifact_risks", "Artifact risks"),
                       ("contradictory_or_missing_evidence", "Contradictory / missing evidence")]:
        items = as_list(c.get(key))
        if items:
            lis = "".join(f"<li>{_esc(str(x))}</li>" for x in items)
            parts.append(f"<div><b>{label}:</b><ul>{lis}</ul></div>")
    vp = c.get("validation_plan") or {}
    if isinstance(vp, dict):
        for key, label in [("next_analysis", "Next analysis"),
                           ("wet_lab_validation", "Wet-lab validation"),
                           ("decision_criteria", "Decision criteria")]:
            items = as_list(vp.get(key))
            if items:
                lis = "".join(f"<li>{_esc(str(x))}</li>" for x in items)
                parts.append(f"<div><b>{label}:</b><ul>{lis}</ul></div>")
    return "<div style='font-size:12px;color:#475569'>" + "".join(parts) + "</div>"


def _evidence_for(c: Dict[str, Any], i: int,
                  evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not evidence:
        return {}
    ref = c.get("evidence_ref") or c.get("direction_id")
    if ref:
        for ev in evidence:
            if ev.get("direction_id") == ref or ev.get("direction") == c.get("direction"):
                return ev
    return evidence[i] if i < len(evidence) else {}


def _pubmed_list(items: List[Dict[str, Any]]) -> str:
    lis = []
    for it in items or []:
        title = _esc(it.get("title", ""))
        pmid = _esc(str(it.get("pmid", "")))
        why = _esc(it.get("why_it_matters") or it.get("supports") or "")
        lis.append(f"<li><b>{title}</b> (PMID {pmid})<br><span class='muted'>{why}</span></li>")
    return f"<ul>{''.join(lis)}</ul>" if lis else ""


def _evidence_panel(ev: Dict[str, Any]) -> str:
    if not ev:
        return ""
    blocks = []
    for key, label in [("direct_evidence", "Direct evidence"),
                       ("analogous_evidence", "Analogous evidence"),
                       ("contradictory_evidence", "Contradictory evidence"),
                       ("methodological_precedent", "Methodological precedent")]:
        html = _pubmed_list(ev.get(key) or [])
        if html:
            blocks.append(f"<div><b>{label}</b>{html}</div>")
    if ev.get("summary"):
        blocks.append(f"<div><b>Summary:</b> {_esc(str(ev['summary']))}</div>")
    if not blocks:
        return ""
    return ("<details><summary class='muted'>Literature evidence</summary>"
            f"<div style='margin-top:6px;font-size:12px'>{''.join(blocks)}</div></details>")


def _candidate_card(c: Dict[str, Any], selected: bool, note: str, is_custom: bool,
                    evidence: Dict[str, Any] = None) -> str:
    mark = "<div class='mark on'>&#10003;</div>" if selected else "<div class='mark off'>&#10007;</div>"
    tag = "<span class='tag'>User custom</span>" if is_custom else ""
    title = _esc(c.get("direction", ""))
    note_html = (f"<div class='note'><div class='lbl'>User note</div>{_esc(note)}</div>"
                 if (selected and note) else "")
    detail = _candidate_detail(c) + _evidence_panel(evidence or {})
    if selected:
        inner = f"<div>{detail}</div>{note_html}"
    else:
        inner = (f"<details><summary class='muted'>Unselected — show detail</summary>"
                 f"<div style='margin-top:8px'>{detail}</div></details>")
    cls = "card" if selected else "card unsel"
    return (f"<div class='{cls}'><div class='row'>{mark}"
            f"<div style='flex:1'><div style='font-weight:600'>{title}{tag}</div>{inner}</div></div></div>")


def _direction_record_label(rec: DirectionSelectionRecord) -> str:
    return (
        f"Integration Cycle {rec.integration_cycle} · integration"
        if rec.phase == "integration"
        else f"Round {rec.round_number} · {rec.phase}"
    )


def build_direction_selection_history(state: AnalysisState) -> List[Dict[str, str]]:
    """Return every completed direction choice in chronological record order.

    ``user_chosen_directions`` represents only the current/most recent batch and
    is therefore not an audit trail.  DirectionSelectionRecord is the durable
    source for the report appendix.  The legacy field remains as a fallback for
    sessions created before interaction records were introduced.
    """
    rows: List[Dict[str, str]] = []
    for rec in state.direction_selection_records:
        if rec.event_type == "pi_decide":
            selections = rec.pi_selected
            decision_mode = "PI-selected"
        elif rec.event_type == "select":
            selections = rec.selections
            decision_mode = "User-selected"
        else:
            # "open" is unfinished and "finalize" adds no direction.
            continue

        for selection in selections:
            if hasattr(selection, "model_dump"):
                selection = selection.model_dump()
            direction = str(selection.get("direction", "") or "").strip()
            if not direction:
                continue
            rows.append({
                "selection_event": _direction_record_label(rec),
                "decision_mode": decision_mode,
                "direction": direction,
                "notes": str(selection.get("notes", "") or ""),
            })

    if rows:
        return rows

    # Backward compatibility for sessions that predate selection records.
    for selection in state.user_chosen_directions or []:
        direction = str(selection.get("direction", "") or "").strip()
        if not direction:
            continue
        rows.append({
            "selection_event": "Current selection",
            "decision_mode": "Legacy state",
            "direction": direction,
            "notes": str(selection.get("notes", "") or ""),
        })
    return rows


def render_direction_selection_history_html(rows: List[Dict[str, str]]) -> str:
    """Render the canonical direction-history table used by all report paths."""
    if not rows:
        return ""
    body = "".join(
        "<tr>"
        f"<td>{_esc(str(row.get('selection_event', '-') or '-'))}</td>"
        f"<td>{_esc(str(row.get('decision_mode', '-') or '-'))}</td>"
        f"<td>{_esc(str(row.get('direction', '-') or '-'))}</td>"
        f"<td>{_esc(str(row.get('notes', '-') or '-'))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<h4>Research Direction Selections</h4>"
        "<table><thead><tr>"
        "<th>Selection Event</th><th>Decision Mode</th>"
        "<th>Direction</th><th>Rationale / Notes</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def render_direction_record_html(rec: DirectionSelectionRecord) -> str:
    label = _direction_record_label(rec)
    head = (f"<div class='hdr'><h1>Direction Selection Record</h1>"
            f"<div class='sub'>{_esc(label)} · {rec.timestamp:%Y-%m-%d %H:%M}</div></div>")

    if rec.event_type == "finalize":
        body = (head + "<div class='banner'>User ended the analysis at the integration "
                "gate without adding new directions.</div>")
        if rec.supplementary_requests:
            body += _supp_block(rec.supplementary_requests)
        return _doc("Direction Selection Record", body)

    parts = [head]
    if rec.event_type == "pi_decide":
        picks = "".join(f"<li>{_esc(p.get('direction',''))}"
                        + (f" — {_esc(p.get('notes',''))}" if p.get('notes') else "")
                        + "</li>" for p in rec.pi_selected)
        parts.append(f"<div class='banner'>User delegated the decision to the PI."
                     f"<ul>{picks}</ul></div>")

    selected_idx = {s.candidate_index for s in rec.selections if s.candidate_index is not None}
    note_by_idx = {s.candidate_index: s.notes for s in rec.selections if s.candidate_index is not None}
    if rec.event_type == "pi_decide":
        pi_by_direction = {
            (p.get("direction") or "").strip(): p.get("notes", "") or ""
            for p in rec.pi_selected
        }
        for i, c in enumerate(rec.candidates):
            direction = (c.get("direction") or "").strip()
            if direction in pi_by_direction:
                selected_idx.add(i)
                if pi_by_direction[direction]:
                    note_by_idx[i] = pi_by_direction[direction]
    parts.append("<h3>Recommended directions</h3>")
    for i, c in enumerate(rec.candidates):
        is_sel = i in selected_idx
        ev = _evidence_for(c, i, rec.literature_evidence)
        parts.append(_candidate_card(c, is_sel, note_by_idx.get(i, ""),
                                     is_custom=False, evidence=ev))

    customs = [s for s in rec.selections if s.candidate_index is None]
    if customs:
        parts.append("<h3>Custom directions</h3>")
        for s in customs:
            parts.append(_candidate_card({"direction": s.direction}, True, s.notes, is_custom=True))

    if rec.supplementary_requests:
        parts.append(_supp_block(rec.supplementary_requests))
    return _doc("Direction Selection Record", "".join(parts))


def _supp_block(requests: List[str]) -> str:
    lis = "".join(f"<li>{_esc(r)}</li>" for r in requests)
    return f"<h3>Supplementary analysis requests</h3><ul>{lis}</ul>"


def render_annotation_record_html(rec: AnnotationRecord) -> str:
    edited = sum(1 for c in rec.clusters if c.user_override)
    mode = "auto-accepted PI labels" if rec.auto_confirmed else "confirmed by user"
    head = (f"<div class='hdr'><h1>Cell Annotation Record</h1>"
            f"<div class='sub'>{_esc(rec.organism)} — {_esc(rec.tissue)} · "
            f"{rec.timestamp:%Y-%m-%d %H:%M} · {mode} · "
            f"{len(rec.clusters)} clusters, {edited} edited</div></div>")

    prop_header = ("Cell Proportion"
                   + (f" ({'/'.join(rec.proportion_group_labels)})"
                      if rec.proportion_group_labels else ""))
    rows = []
    for c in rec.clusters:
        if c.user_override:
            type_cell = (f"<span style='font-weight:600;color:#7c3aed'>{_esc(c.final_cell_type)}</span>"
                         f" <s class='muted'>{_esc(c.original_cell_type)}</s>"
                         f"<span class='tag'>User edited</span>")
        else:
            type_cell = _esc(c.final_cell_type)
        detail_bits = []
        if c.key_markers:
            detail_bits.append(f"<div><b>Markers:</b> {_esc(', '.join(c.key_markers))}</div>")
        if c.alternative_cell_types:
            detail_bits.append(f"<div><b>Alternatives:</b> {_esc(', '.join(c.alternative_cell_types))}</div>")
        if c.low_confidence_reason:
            detail_bits.append(f"<div><b>Reason:</b> {_esc(c.low_confidence_reason)}</div>")
        detail = ("<details><summary class='muted'>detail</summary>"
                  f"<div style='margin-top:6px;font-size:12px'>{''.join(detail_bits)}</div></details>"
                  if detail_bits else "")
        if rec.proportion_group_labels:
            prop = " / ".join(f"{c.group_proportions.get(l, 0):.1f}%" for l in rec.proportion_group_labels)
        else:
            prop = "—"
        cell_count = (
            f"{c.n_cells} ({_esc(c.n_cells_source)})"
            if getattr(c, "n_cells_known", False)
            else "Unknown"
        )
        rows.append(f"<tr><td style='font-family:monospace'>{_esc(c.cluster_id)}</td>"
                    f"<td>{type_cell}{detail}</td><td>{_conf_badge(c.confidence)}</td>"
                    f"<td>{_esc(cell_count)}</td><td>{_esc(prop)}</td></tr>")
    table = (f"<table><thead><tr><th>Cluster</th><th>Cell Type</th>"
             f"<th>Confidence</th><th>Cell Count</th><th>{_esc(prop_header)}</th></tr></thead>"
             f"<tbody>{''.join(rows)}</tbody></table>")
    body = head + table
    validation_status = getattr(rec, "validation_status", None) or "not_run"
    validation_summary = getattr(rec, "validation_summary", None) or ""
    body += (
        "<h3>Annotation validation</h3>"
        f"<p><b>Outcome:</b> {_esc(validation_status)}</p>"
        + (f"<p>{_esc(validation_summary)}</p>" if validation_summary else "")
    )
    if rec.supplementary_requests:
        body += _supp_block(rec.supplementary_requests)
    return _doc("Cell Annotation Record", body)


def _direction_record_entry_pairs(state: AnalysisState) -> list[tuple[DirectionSelectionRecord, Dict[str, str]]]:
    pairs = []
    for i, rec in enumerate(state.direction_selection_records):
        if rec.event_type == "open":
            continue
        fname = f"direction_record_{i:02d}_round{rec.round_number}.html"
        pairs.append((rec, {"filename": fname, "label": _direction_record_label(rec)}))
    return pairs


def direction_record_entries(state: AnalysisState) -> list:
    """Single source of the direction-record filename + label formula.

    Returns [{"filename": ..., "label": ...}, ...] in record order, skipping
    'open' events. Both the file writer and the downloads renderer consume this.
    """
    return [entry for _, entry in _direction_record_entry_pairs(state)]


def write_interaction_record_files(
    state: AnalysisState,
    reports_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from ...config import get_session_reports

    result: Dict[str, Any] = {"annotation_record": None, "direction_records": []}
    record_entry_pairs = _direction_record_entry_pairs(state)
    if state.annotation_record is None and not record_entry_pairs:
        return result

    reports_dir = Path(reports_dir) if reports_dir is not None else get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if state.annotation_record is not None:
        fname = "annotation_record.html"
        (reports_dir / fname).write_text(
            render_annotation_record_html(state.annotation_record), encoding="utf-8")
        result["annotation_record"] = fname

    for rec, entry in record_entry_pairs:
        (reports_dir / entry["filename"]).write_text(
            render_direction_record_html(rec), encoding="utf-8")
        result["direction_records"].append(entry)
    return result
