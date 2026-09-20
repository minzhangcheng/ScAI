"""Cross-direction integration report HTML.

Moved verbatim out of backend/services/report/variants.py.
"""
import os
import sys
import html as html_module
from datetime import datetime
from typing import Dict, Any, List

from backend.utils.report_html import markdown_to_html
from backend.config import get_session_dir, get_session_reports
from backend.utils.blob_store import inline_blobs
from backend.services.report.score_section import build_score_definitions_section_html
from backend.models.schemas import AnalysisState

logger = __import__('logging').getLogger(__name__)

from .renderers import _render_score_badge
from .. import variant_sections as _variant_sections_owner


def _inline_session_text(value: Any, state: AnalysisState) -> str:
    if value is None:
        return ""
    try:
        inlined = inline_blobs(value, get_session_dir(state.id) / "blobs")
    except ValueError as exc:
        logger.warning("Invalid blob reference while rendering integration report for %s: %s", state.id, exc)
        inlined = value
    if inlined is None:
        return ""
    return str(inlined)

def _render_integration_section(index: int, title: str, content_html: str) -> str:
    return f"""
        <div class="section">
            <div class="section-header">
                <h2>{index}. {html_module.escape(title)}</h2>
            </div>
            <div class="section-content">
                {content_html}
            </div>
        </div>"""

def _render_novelty_score(score):
    if score is None:
        return '<span class="badge-na">N/A</span>'
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return '<span class="badge-na">N/A</span>'
    formatted = f"{score_f:.1f}/10"
    if score_f >= 7:
        color = "#059669"
    elif score_f >= 4:
        color = "#d97706"
    else:
        color = "#dc2626"
    return f'<span style="color:{color};font-weight:600">{formatted}</span>'

def _render_evidence_strength(text):
    t = str(text).lower() if text else ""
    if t == "strong":
        return "High"
    if t == "moderate":
        return "Medium"
    if t == "weak":
        return "Low"
    return str(text)

def _render_integration_candidates_rows(candidates):
    rows = ""
    for i, c in enumerate(candidates or [], 1):
        direction = html_module.escape(str(c.get("direction", "")))
        rationale = html_module.escape(str(c.get("rationale", "")))
        rank = c.get("priority_rank", "N/A")
        novelty = _render_novelty_score(c.get("novelty_score"))
        evidence = _render_evidence_strength(c.get("evidence_strength"))
        confidence = _render_score_badge(c.get("confidence", ""))
        rows += f"""
        <tr>
            <td class="dir-number">{i}</td>
            <td>
                <div class="dir-title">{direction}</div>
                <div class="dir-rationale">{rationale}</div>
            </td>
            <td class="score-cell">Rank {rank}</td>
            <td class="score-cell">{novelty}</td>
            <td class="score-cell">{evidence}</td>
            <td class="score-cell">{confidence}</td>
        </tr>"""
    return rows

def _render_integration_evidence_section(candidates, literature_evidence):
    if not candidates or not literature_evidence:
        return ""

    evidence_lookup: Dict[str, Dict[str, Any]] = {}
    for ev in literature_evidence:
        direction_id = str(ev.get("direction_id") or "")
        direction = str(ev.get("direction") or "")
        if direction_id:
            evidence_lookup[direction_id] = ev
        if direction:
            evidence_lookup[direction] = ev

    rows = ""
    for i, candidate in enumerate(candidates, 1):
        direction = str(candidate.get("direction") or "")
        direction_id = str(candidate.get("direction_id") or "")
        evidence = evidence_lookup.get(direction_id) or evidence_lookup.get(direction) or {}
        if not evidence:
            continue
        status = html_module.escape(str(evidence.get("status") or "N/A"))
        direct = html_module.escape(str(evidence.get("direct_evidence_count", "N/A")))
        analogous = html_module.escape(str(evidence.get("analogous_evidence_count", "N/A")))
        contradictory = html_module.escape(str(evidence.get("contradictory_evidence_count", "N/A")))
        summary = html_module.escape(str(evidence.get("summary") or ""))
        detail = ""
        if evidence.get("policy_version"):
            assessment = str(evidence.get("assessment_status") or "unassessed")
            limitations = "; ".join(str(item) for item in (evidence.get("assessment_limitations") or evidence.get("limitations") or [])[:3])
            source_id_values = list(dict.fromkeys(
                str(item.get("pmid"))
                for collection in (
                    evidence.get("article_inventory") or [],
                    evidence.get("extractions") or [],
                )
                for item in collection
                if isinstance(item, dict) and item.get("pmid")
            ))[:5]
            source_ids = ", ".join(source_id_values)
            task_statuses = ", ".join(
                f"{item.get('task_id') or 'task'}={item.get('status') or 'unknown'}"
                for item in (evidence.get("task_executions") or [])
                if isinstance(item, dict)
            )
            uncertainty = "; ".join(
                str(item) for item in (evidence.get("assessment_uncertainty") or [])
            )
            detail = " | ".join(item for item in (
                f"PI assessment: {assessment}",
                f"Source PMIDs: {source_ids}" if source_ids else "",
                f"Task status: {task_statuses}" if task_statuses else "",
                f"PI uncertainty: {uncertainty}" if uncertainty else "",
                f"Limitations: {limitations}" if limitations else "",
            ) if item)
        summary_cell = (
            f"{summary}<br>{html_module.escape(detail)}"
            if evidence.get("policy_version")
            else summary
        )
        strength = _render_evidence_strength(candidate.get("evidence_strength"))
        rows += f"""
            <tr>
                <td class="dir-number">{i}</td>
                <td>{html_module.escape(direction)}</td>
                <td>{status}</td>
                <td>{direct}</td>
                <td>{analogous}</td>
                <td>{contradictory}</td>
                <td>{html_module.escape(strength)}</td>
                <td>{summary_cell}</td>
            </tr>"""

    if not rows:
        return ""

    return f"""
        <table class="direction-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Direction</th>
                    <th>Literature status</th>
                    <th>Direct</th>
                    <th>Analogous</th>
                    <th>Contradictory</th>
                    <th>Evidence strength</th>
                    <th>Summary</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""

_INTEGRATION_STYLE = "    :root {\n        --primary-color: #4f46e5;\n        --primary-light: #eef2ff;\n        --text-color: #1e293b;\n        --text-secondary: #64748b;\n        --border-color: #e2e8f0;\n        --bg-color: #f8fafc;\n        --white: #ffffff;\n        --success: #059669;\n        --warning: #d97706;\n        --danger: #dc2626;\n    }\n\n    * {\n        margin: 0;\n        padding: 0;\n        box-sizing: border-box;\n    }\n\n    body {\n        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;\n        line-height: 1.6;\n        color: var(--text-color);\n        background: var(--bg-color);\n        padding: 20px;\n    }\n\n    .container {\n        max-width: 1100px;\n        margin: 0 auto;\n        background: var(--white);\n        border-radius: 16px;\n        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);\n        overflow: hidden;\n    }\n\n    .header {\n        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);\n        color: white;\n        padding: 40px;\n        text-align: center;\n    }\n\n    .header h1 {\n        font-size: 2rem;\n        margin-bottom: 10px;\n    }\n\n    .header .meta {\n        opacity: 0.9;\n        font-size: 0.9rem;\n    }\n\n    .content {\n        padding: 40px;\n    }\n\n    .section {\n        margin-bottom: 40px;\n        border: 1px solid var(--border-color);\n        border-radius: 12px;\n        overflow: hidden;\n    }\n\n    .section-header {\n        background: var(--primary-light);\n        padding: 16px 24px;\n        border-bottom: 1px solid var(--border-color);\n    }\n\n    .section-header h2 {\n        font-size: 1.25rem;\n        color: var(--primary-color);\n        display: flex;\n        align-items: center;\n        gap: 10px;\n    }\n\n    .section-content {\n        padding: 24px;\n    }\n\n    .section-content p {\n        margin-bottom: 16px;\n        text-align: justify;\n    }\n\n    .section-content ul {\n        list-style: none;\n        padding: 0;\n    }\n\n    .section-content ul li {\n        padding: 8px 0 8px 20px;\n        position: relative;\n        border-bottom: 1px solid #f1f5f9;\n    }\n\n    .section-content ul li:last-child {\n        border-bottom: none;\n    }\n\n    .section-content ul li::before {\n        content: '▸';\n        position: absolute;\n        left: 0;\n        color: var(--primary-color);\n        font-weight: bold;\n    }\n\n    /* Direction evaluation table */\n    .direction-table {\n        width: 100%;\n        max-width: 100%;\n        border-collapse: collapse;\n        margin-top: 16px;\n        display: block;\n        overflow-x: auto;\n    }\n\n    .direction-table th {\n        background: var(--primary-light);\n        padding: 12px 16px;\n        text-align: left;\n        font-size: 0.85rem;\n        color: var(--primary-color);\n        border-bottom: 2px solid var(--border-color);\n        white-space: nowrap;\n    }\n\n    .direction-table td {\n        padding: 16px;\n        border-bottom: 1px solid var(--border-color);\n        vertical-align: top;\n    }\n\n    .direction-table tr:last-child td {\n        border-bottom: none;\n    }\n\n    .dir-number {\n        font-weight: 700;\n        color: var(--primary-color);\n        font-size: 1.1rem;\n        width: 40px;\n        text-align: center;\n    }\n\n    .dir-title {\n        font-weight: 600;\n        color: var(--text-color);\n        margin-bottom: 6px;\n    }\n\n    .dir-rationale {\n        font-size: 0.88rem;\n        color: var(--text-secondary);\n        line-height: 1.5;\n    }\n\n    .score-cell {\n        text-align: center;\n        min-width: 120px;\n    }\n\n    .badge-high {\n        display: inline-block;\n        padding: 2px 10px;\n        border-radius: 9999px;\n        font-size: 0.75rem;\n        font-weight: 600;\n        background: #dcfce7;\n        color: #166534;\n    }\n\n    .badge-medium {\n        display: inline-block;\n        padding: 2px 10px;\n        border-radius: 9999px;\n        font-size: 0.75rem;\n        font-weight: 600;\n        background: #fef3c7;\n        color: #92400e;\n    }\n\n    .badge-low {\n        display: inline-block;\n        padding: 2px 10px;\n        border-radius: 9999px;\n        font-size: 0.75rem;\n        font-weight: 600;\n        background: #fee2e2;\n        color: #991b1b;\n    }\n\n    .badge-na {\n        display: inline-block;\n        padding: 2px 10px;\n        border-radius: 9999px;\n        font-size: 0.75rem;\n        font-weight: 600;\n        background: #e2e8f0;\n        color: #475569;\n    }\n\n    .findings-grid {\n        display: grid;\n        grid-template-columns: 1fr 1fr;\n        gap: 20px;\n    }\n\n    .findings-card {\n        border: 1px solid var(--border-color);\n        border-radius: 8px;\n        overflow: hidden;\n    }\n\n    .findings-card-header {\n        padding: 12px 16px;\n        font-weight: 600;\n        font-size: 0.9rem;\n    }\n\n    .findings-card-header.bio {\n        background: #e0f2fe;\n        color: #0369a1;\n    }\n\n    .findings-card-header.path {\n        background: #fef3c7;\n        color: #92400e;\n    }\n\n    .findings-card-header.biochem {\n        background: #f3e8ff;\n        color: #7c3aed;\n    }\n\n    .findings-card-header.lit {\n        background: #f0fdf4;\n        color: #15803d;\n    }\n\n    .findings-card-content {\n        padding: 16px;\n    }\n\n    .findings-card-content ul {\n        list-style: none;\n        padding: 0;\n    }\n\n    .findings-card-content ul li {\n        padding: 6px 0;\n        border-bottom: 1px solid #f1f5f9;\n        font-size: 0.88rem;\n    }\n\n    .findings-card-content ul li:last-child {\n        border-bottom: none;\n    }\n\n    .pi-assessment {\n        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);\n        padding: 24px;\n        border-radius: 12px;\n        border-left: 4px solid #0284c7;\n    }\n\n    .pi-assessment h3 {\n        color: #0369a1;\n        margin-bottom: 12px;\n    }\n\n    .footer {\n        text-align: center;\n        padding: 20px;\n        color: var(--text-secondary);\n        font-size: 0.85rem;\n        border-top: 1px solid var(--border-color);\n    }\n\n    @media (max-width: 768px) {\n        .content { padding: 20px; }\n        .findings-grid { grid-template-columns: 1fr; }\n        .direction-table { font-size: 0.85rem; }\n    }\n\n    .evidence-failed-row { background: #fef2f2; }\n    .evidence-partial-row { background: #fffbeb; }\n    .ev-warning { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 600; }\n    .ev-failed { background: #fee2e2; color: #991b1b; }\n    .ev-partial { background: #fef3c7; color: #92400e; }\n    .ev-details { margin-bottom: 16px; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; }\n    .ev-details summary { padding: 12px 16px; cursor: pointer; font-weight: 600; color: var(--primary-color); background: var(--primary-light); user-select: none; }\n    .ev-details summary:hover { background: #dbeafe; }\n    .ev-details-content { padding: 16px; }\n    .ev-list { list-style: none; padding: 0; }\n    .ev-list li { padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }\n    .ev-list li:last-child { border-bottom: none; }\n    .ev-pmid { padding: 4px 0 4px 16px !important; font-size: 0.88rem; }\n    .ev-why { color: var(--text-secondary); font-size: 0.85rem; display: block; padding-left: 24px; }\n\n    @media print {\n        body { background: white; padding: 0; }\n        .container { box-shadow: none; border-radius: 0; }\n    }"


def generate_integration_report_html(state, candidates, literature_evidence):
    return _variant_sections_owner.generate_integration_report_html(
        state,
        candidates,
        literature_evidence,
        bindings=sys.modules[__name__],
    )
