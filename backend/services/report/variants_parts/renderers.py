"""Shared HTML fragment renderers for direction evidence and score badges.

Moved verbatim out of backend/services/report/variants.py.
"""
import html as html_module
from typing import Dict, Any

from backend.utils.formatting import as_list

logger = __import__('logging').getLogger(__name__)


_SPECIES_TISSUE_LABELS = {
    "matched_species_tissue_available": "Matched species/tissue evidence available",
    "no_matched_species_tissue_evidence": "No matched species/tissue evidence",
    "cross_tissue_only": "Cross-tissue evidence only",
    "cross_species_only": "Cross-species evidence only",
    "": "",
}


def _render_score_badge(text, fallback_class="badge-na"):
    t = str(text).lower() if text else ""
    if t.startswith("high"):
        return '<span class="badge-high">High</span>'
    if t.startswith("medium"):
        return '<span class="badge-medium">Medium</span>'
    if t.startswith("low"):
        return '<span class="badge-low">Low</span>'
    return f'<span class="{fallback_class}">{html_module.escape(str(text)[:20])}</span>'

def _render_direction_evidence_section_from_lists(
    candidates: list[dict],
    evidence: list[dict],
) -> str:
    if not candidates:
        return ""

    evidence_lookup: Dict[str, Dict[str, Any]] = {}
    for ev in evidence:
        eid = ev.get("direction_id", "")
        if eid:
            evidence_lookup[eid] = ev

    def _status_badge(status, timed_out=False, total_evidence=0):
        s = status.lower()
        if s == "completed":
            return '<span class="badge-high">Completed</span>'
        if s == "partial" and timed_out and total_evidence == 0:
            return '<span class="badge-na">Timed out</span>'
        if s == "partial":
            return '<span class="badge-medium">Partial</span>'
        if s in ("failed", "skipped"):
            return '<span class="badge-low">Failed/Skipped</span>'
        return f'<span class="badge-medium">{html_module.escape(status[:20])}</span>'

    def _render_evidence_list(items, label):
        if not items:
            return ""
        parts = [f"<li><strong>{label}:</strong></li>"]
        for item in items:
            pmid = html_module.escape(str(item.get("pmid", "")))
            title = html_module.escape(str(item.get("title", ""))[:120])
            relevance = html_module.escape(str(item.get("relevance", "")))
            why = html_module.escape(str(item.get("why_it_matters", "")))
            parts.append(f'<li class="ev-pmid">PMID: {pmid} &mdash; {title} <em>({relevance})</em><br/><span class="ev-why">{why}</span></li>')
        return "\n".join(parts)

    summary_rows = ""
    detail_sections = ""

    for i, c in enumerate(candidates, 1):
        did = c.get("direction_id", "")
        direction = html_module.escape(str(c.get("direction", "")))
        overall_priority = c.get("overall_priority", "N/A")
        data_reliability = c.get("data_reliability", "N/A")
        literature_novelty = c.get("literature_novelty", "N/A")
        ev = evidence_lookup.get(c.get("evidence_ref") or did, {})
        status = ev.get("status", "no_evidence") if ev else "no_evidence"
        is_new_policy = bool(ev.get("policy_version")) if ev else False

        direct_count = ev.get("direct_evidence_count", 0) if ev else 0
        analogous_count = ev.get("analogous_evidence_count", 0) if ev else 0
        contradictory_count = ev.get("contradictory_evidence_count", 0) if ev else 0
        method_precedent_count = ev.get("methodological_precedent_count", 0) if ev else 0
        total_evidence = direct_count + analogous_count + contradictory_count + method_precedent_count
        if is_new_policy:
            total_evidence = len(ev.get("article_inventory", []) or [])
        timed_out = bool(ev.get("timed_out", False)) if ev else False

        row_class = ""
        if status in ("failed", "skipped"):
            row_class = ' class="evidence-failed-row"'
        elif status == "partial":
            row_class = ' class="evidence-partial-row"'

        summary_rows += f"""
        <tr{row_class}>
            <td class="dir-number">{i}</td>
            <td><div class="dir-title">{direction}</div></td>
            <td class="score-cell">{_render_score_badge(overall_priority, fallback_class="badge-medium")}</td>
            <td class="score-cell">{_render_score_badge(data_reliability, fallback_class="badge-medium")}</td>
            <td class="score-cell">{_render_score_badge(literature_novelty, fallback_class="badge-medium")}</td>
            <td class="score-cell">{_status_badge(status, timed_out=timed_out, total_evidence=total_evidence)}</td>
        </tr>"""

        data_support = html_module.escape(str(c.get("data_support_summary", "")))
        bio_expected = html_module.escape(str(c.get("biology_expectedness", "N/A")))
        data_novelty = html_module.escape(str(c.get("data_novelty", "N/A")))
        mech_importance = html_module.escape(str(c.get("mechanistic_importance", "N/A")))
        func_impact = html_module.escape(str(c.get("functional_impact", "N/A")))
        data_suff = html_module.escape(str(c.get("data_sufficiency", "N/A")))
        method_gap = html_module.escape(str(c.get("methodological_gap", "N/A")))

        ev_summary = html_module.escape(ev.get("summary", "")) if ev else ""
        gap_assessment = html_module.escape(ev.get("gap_assessment", "")) if ev else ""
        negative_val = html_module.escape(str(ev.get("negative_validation_confidence", "N/A"))) if ev else "N/A"

        species_tissue = ev.get("species_tissue_qualification", "") if ev else ""
        species_label = _SPECIES_TISSUE_LABELS.get(species_tissue, "")
        species_display = f"{html_module.escape(species_tissue)} ({html_module.escape(species_label)})" if species_label else (html_module.escape(species_tissue) if species_tissue else "")

        prereqs = as_list(c.get("prerequisite_dependencies"))
        prereq_html = ""
        if prereqs:
            prereq_html = "<li><strong>Prerequisite Dependencies:</strong> " + ", ".join(html_module.escape(str(p)) for p in prereqs) + "</li>"

        risk_flags = as_list(c.get("risk_flags"))
        risk_html = ""
        if risk_flags:
            risk_html = "<li><strong>Risk Flags:</strong> " + ", ".join(html_module.escape(str(r)) for r in risk_flags) + "</li>"

        method_pmids = []
        for mp in (ev.get("methodological_precedent", []) if ev else []):
            pmid = mp.get("pmid", "")
            if pmid:
                method_pmids.append(html_module.escape(str(pmid)))

        direct_ev_html = _render_evidence_list(ev.get("direct_evidence", []) if ev else [], "Direct Evidence")
        analogous_ev_html = _render_evidence_list(ev.get("analogous_evidence", []) if ev else [], "Analogous Evidence")
        contradictory_ev_html = _render_evidence_list(ev.get("contradictory_evidence", []) if ev else [], "Contradictory Evidence")
        method_prec_html = _render_evidence_list(ev.get("methodological_precedent", []) if ev else [], "Methodological Precedent")

        all_key_pmids = []
        if ev:
            for etype in ["direct_evidence", "analogous_evidence", "contradictory_evidence", "methodological_precedent"]:
                for item in ev.get(etype, []):
                    if item.get("pmid"):
                        all_key_pmids.append(item)
        key_pmids_html = _render_evidence_list(all_key_pmids, "Key PMIDs")

        queries_html = ""
        for q in (ev.get("queries", []) if ev else [])[:4]:
            cat = html_module.escape(str(q.get("category", "")))
            query_text = html_module.escape(str(q.get("query", "")))
            rc = q.get("result_count", 0)
            rlc = q.get("relevant_count", 0)
            queries_html += f"<li><strong>{cat}:</strong> {query_text} (results: {rc}, relevant: {rlc})</li>"
        if queries_html:
            queries_html = "<li><strong>Queries:</strong></li>\n" + queries_html

        limitations_html = ""
        limitations = list(ev.get("limitations", []) if ev else [])
        if is_new_policy:
            limitations.extend(ev.get("assessment_limitations", []) or [])
            limitations = list(dict.fromkeys(str(item) for item in limitations if item))
        for lim in limitations:
            limitations_html += f"<li>{html_module.escape(str(lim))}</li>"
        if limitations_html:
            limitations_html = "<li><strong>Limitations:</strong></li>\n" + limitations_html

        new_policy_html = ""
        if is_new_policy:
            inventory_html = "".join(
                "<li class=\"ev-pmid\"><strong>Retained PMID:</strong> "
                + html_module.escape(str(item.get("pmid") or ""))
                + (" &mdash; " + html_module.escape(str(item.get("title") or "")[:160]) if item.get("title") else "")
                + "</li>"
                for item in (ev.get("article_inventory") or [])
                if isinstance(item, dict) and item.get("pmid")
            )
            extraction_html = "".join(
                "<li><strong>Source passage (PMID "
                + html_module.escape(str(item.get("pmid") or ""))
                + "):</strong> "
                + html_module.escape(str(item.get("source_passage") or item.get("finding") or ""))
                + "</li>"
                for item in (ev.get("extractions") or [])
                if isinstance(item, dict) and (item.get("source_passage") or item.get("finding"))
            )
            execution_html = "".join(
                "<li><strong>Task "
                + html_module.escape(str(item.get("task_id") or ""))
                + ":</strong> "
                + html_module.escape(str(item.get("status") or "unknown"))
                + "</li>"
                for item in (ev.get("task_executions") or [])
                if isinstance(item, dict)
            )
            uncertainty = "; ".join(
                html_module.escape(str(item))
                for item in (ev.get("assessment_uncertainty") or [])
            )
            new_policy_html = "".join((
                "<li><strong>PI Assessment:</strong> "
                + html_module.escape(str(ev.get("assessment_status") or "unassessed"))
                + "</li>",
                ("<li><strong>Assessment Uncertainty:</strong> " + uncertainty + "</li>") if uncertainty else "",
                inventory_html,
                extraction_html,
                execution_html,
            ))

        warning_html = ""
        if status in ("failed", "skipped"):
            warning_html = '<div class="ev-warning ev-failed">&#9888; Literature evidence search failed or was skipped.</div>'
        elif status == "partial" and timed_out and total_evidence == 0:
            warning_html = '<div class="ev-warning ev-partial">&#9888; Literature search timed out before references could be retained.</div>'
        elif status == "partial":
            warning_html = '<div class="ev-warning ev-partial">&#9888; Literature evidence search only partially completed; listed references are partial.</div>'

        detail_items = f"""
                <ul class="ev-list">
                    {("<li><strong>Data Support Summary:</strong> " + data_support + "</li>") if data_support else ""}
                    <li><strong>Data Reliability:</strong> {_render_score_badge(data_reliability, fallback_class="badge-medium")}</li>
                    <li><strong>Literature Novelty:</strong> {_render_score_badge(literature_novelty, fallback_class="badge-medium")}</li>
                    <li><strong>Overall Priority:</strong> {_render_score_badge(overall_priority, fallback_class="badge-medium")}</li>
                    <li><strong>Biology Expectedness:</strong> {bio_expected}</li>
                    <li><strong>Data Novelty:</strong> {data_novelty}</li>
                    <li><strong>Mechanistic Importance:</strong> {mech_importance}</li>
                    <li><strong>Functional Impact:</strong> {func_impact}</li>
                    <li><strong>Data Sufficiency:</strong> {data_suff}</li>
                    <li><strong>Methodological Gap:</strong> {method_gap}</li>
                    {("<li><strong>Evidence Summary:</strong> " + ev_summary + "</li>") if ev_summary else ""}
                    {("<li><strong>Gap Assessment:</strong> " + gap_assessment + "</li>") if gap_assessment else ""}
                    <li><strong>Negative Validation Confidence:</strong> {negative_val}</li>
                    {("<li><strong>Species/Tissue Qualification:</strong> " + species_display + "</li>") if species_display else ""}
                    {("<li><strong>Methodological Precedent Count:</strong> " + str(method_precedent_count) + (" (PMIDs: " + ", ".join(method_pmids) + ")") if method_pmids else "") + "</li>" if method_precedent_count > 0 or method_pmids else ""}
                    {("<li><strong>Direct Evidence Count:</strong> " + str(direct_count) + "</li>") if direct_count else ""}
                    {("<li><strong>Analogous Evidence Count:</strong> " + str(analogous_count) + "</li>") if analogous_count else ""}
                    {("<li><strong>Contradictory Evidence Count:</strong> " + str(contradictory_count) + "</li>") if contradictory_count else ""}
                    {queries_html}
                    {new_policy_html}
                    {key_pmids_html}
                    {limitations_html}
                    {prereq_html}
                    {risk_html}
                </ul>
        """

        detail_sections += f"""
        <details class="ev-details">
            <summary>Evidence Details &mdash; {direction}</summary>
            <div class="ev-details-content">
                {warning_html}
                {detail_items}
                {direct_ev_html if not key_pmids_html else ""}
                {analogous_ev_html if not key_pmids_html else ""}
                {contradictory_ev_html if not key_pmids_html else ""}
                {method_prec_html if not key_pmids_html else ""}
            </div>
        </details>"""

    return f"""
    <div class="section">
        <div class="section-header">
            <h2>7. Literature-Gated Direction Evaluation</h2>
        </div>
        <div class="section-content">
            <p style="margin-bottom:12px;color:var(--text-secondary);font-size:0.9rem;">
                Detailed literature evidence assessment for each research direction candidate,
                including evidence status, mechanistic validation, and species/tissue context.
            </p>
            <table class="direction-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Direction</th>
                        <th>Priority</th>
                        <th>Data Reliability</th>
                        <th>Lit. Novelty</th>
                        <th>Evidence Status</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_rows}
                </tbody>
            </table>
            <div style="margin-top:24px;">
                {detail_sections}
            </div>
        </div>
    </div>"""

def _render_direction_evidence_section(exploration_report) -> str:
    return _render_direction_evidence_section_from_lists(
        exploration_report.research_direction_candidates or [],
        exploration_report.direction_literature_evidence or [],
    )

def _render_direction_candidates_rows(candidates) -> str:
    """Render the main Research Direction Candidates table rows.

    Reads the final scored fields (overall_priority / literature_novelty /
    data_reliability) produced by score_directions_with_evidence(), falling back
    to the legacy expected_innovation / expected_novelty fields for old sessions.
    """
    def score_detail(text):
        text = str(text)
        for sep in [" — ", " - ", ": ", " "]:
            parts = text.split(sep, 1)
            if len(parts) > 1 and parts[0].lower() in ("high", "medium", "low"):
                return html_module.escape(parts[1])
        return html_module.escape(text)

    rows = ""
    for i, c in enumerate(candidates or [], 1):
        if not isinstance(c, dict):
            continue
        direction = html_module.escape(str(c.get("direction", "")))
        rationale = html_module.escape(str(c.get("rationale", "")))
        priority = c.get("overall_priority") or c.get("expected_innovation", "N/A")
        novelty = c.get("literature_novelty") or c.get("expected_novelty", "N/A")
        reliability = c.get("data_reliability", "N/A")
        rows += f"""
        <tr>
            <td class="dir-number">{i}</td>
            <td>
                <div class="dir-title">{direction}</div>
                <div class="dir-rationale">{rationale}</div>
            </td>
            <td class="score-cell">{_render_score_badge(priority)}<div class="score-detail">{score_detail(priority)}</div></td>
            <td class="score-cell">{_render_score_badge(novelty)}<div class="score-detail">{score_detail(novelty)}</div></td>
            <td class="score-cell">{_render_score_badge(reliability)}<div class="score-detail">{score_detail(reliability)}</div></td>
        </tr>
        """
    return rows
