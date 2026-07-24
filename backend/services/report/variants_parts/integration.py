"""Cross-direction integration report HTML.

Moved verbatim out of backend/services/report/variants.py.
"""
import os
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
                <td>{summary}</td>
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

def generate_integration_report_html(state, candidates, literature_evidence):
    project_name = html_module.escape(state.name or "Untitled Analysis")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections: List[tuple[str, str]] = []

    correlation_text = _inline_session_text(state.multi_direction_correlation, state)
    if correlation_text:
        sections.append(("Cross-Direction Correlation", markdown_to_html(correlation_text)))

    innovation_rounds = [r for r in (state.round_results or []) if r.phase == "innovation"]
    if innovation_rounds:
        round_cards = ""
        for r in innovation_rounds:
            direction = html_module.escape(r.direction or "N/A")
            pi_summary = markdown_to_html(_inline_session_text(r.pi_summary or "N/A", state))
            bio_items = "".join(f"<li>{html_module.escape(str(f))}</li>" for f in r.bioinformatics_findings[:5])
            path_items = "".join(f"<li>{html_module.escape(str(f))}</li>" for f in r.pathology_findings[:5])
            biochem_items = "".join(f"<li>{html_module.escape(str(f))}</li>" for f in r.biochemistry_findings[:5])
            lit_items = "".join(f"<li>{html_module.escape(str(f))}</li>" for f in r.literature_findings[:5])
            round_cards += f"""
            <div class="findings-card">
                <div class="findings-card-header" style="background:var(--primary-light);color:var(--primary-color);">
                    Round {r.round_number}: {direction}
                </div>
                <div class="findings-card-content">
                    <div class="pi-assessment" style="margin-bottom:16px;">
                        <h3 style="margin-top:0;">PI Summary</h3>
                        {pi_summary}
                    </div>
                    <div class="findings-grid">
                        <div class="findings-card">
                            <div class="findings-card-header bio">Bioinformatics</div>
                            <div class="findings-card-content"><ul>{bio_items or "<li>No findings</li>"}</ul></div>
                        </div>
                        <div class="findings-card">
                            <div class="findings-card-header path">Pathology</div>
                            <div class="findings-card-content"><ul>{path_items or "<li>No findings</li>"}</ul></div>
                        </div>
                        <div class="findings-card">
                            <div class="findings-card-header biochem">Biochemistry</div>
                            <div class="findings-card-content"><ul>{biochem_items or "<li>No findings</li>"}</ul></div>
                        </div>
                        <div class="findings-card">
                            <div class="findings-card-header lit">Literature</div>
                            <div class="findings-card-content"><ul>{lit_items or "<li>No findings</li>"}</ul></div>
                        </div>
                    </div>
                </div>
            </div>"""
        sections.append(("Innovation Round Summaries", round_cards))

    candidates_rows = _render_integration_candidates_rows(candidates)
    if candidates_rows:
        candidate_table = f"""
            <table class="direction-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Direction &amp; Rationale</th>
                        <th>Rank</th>
                        <th>Novelty</th>
                        <th>Evidence</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>{candidates_rows}</tbody>
            </table>"""
        sections.append(("Research Direction Candidates", candidate_table))

    evidence_html = _render_integration_evidence_section(candidates, literature_evidence)
    if evidence_html:
        sections.append(("Literature-Gated Direction Evaluation", evidence_html))

    numbered_sections_html = "\n".join(
        _render_integration_section(index, title, content_html)
        for index, (title, content_html) in enumerate(sections, 1)
    )

    score_section = build_score_definitions_section_html(state.score_definitions)

    style_block = """    :root {
        --primary-color: #4f46e5;
        --primary-light: #eef2ff;
        --text-color: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --bg-color: #f8fafc;
        --white: #ffffff;
        --success: #059669;
        --warning: #d97706;
        --danger: #dc2626;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background: var(--bg-color);
        padding: 20px;
    }

    .container {
        max-width: 1100px;
        margin: 0 auto;
        background: var(--white);
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }

    .header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 40px;
        text-align: center;
    }

    .header h1 {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .header .meta {
        opacity: 0.9;
        font-size: 0.9rem;
    }

    .content {
        padding: 40px;
    }

    .section {
        margin-bottom: 40px;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
    }

    .section-header {
        background: var(--primary-light);
        padding: 16px 24px;
        border-bottom: 1px solid var(--border-color);
    }

    .section-header h2 {
        font-size: 1.25rem;
        color: var(--primary-color);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-content {
        padding: 24px;
    }

    .section-content p {
        margin-bottom: 16px;
        text-align: justify;
    }

    .section-content ul {
        list-style: none;
        padding: 0;
    }

    .section-content ul li {
        padding: 8px 0 8px 20px;
        position: relative;
        border-bottom: 1px solid #f1f5f9;
    }

    .section-content ul li:last-child {
        border-bottom: none;
    }

    .section-content ul li::before {
        content: '\u25b8';
        position: absolute;
        left: 0;
        color: var(--primary-color);
        font-weight: bold;
    }

    /* Direction evaluation table */
    .direction-table {
        width: 100%;
        max-width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        display: block;
        overflow-x: auto;
    }

    .direction-table th {
        background: var(--primary-light);
        padding: 12px 16px;
        text-align: left;
        font-size: 0.85rem;
        color: var(--primary-color);
        border-bottom: 2px solid var(--border-color);
        white-space: nowrap;
    }

    .direction-table td {
        padding: 16px;
        border-bottom: 1px solid var(--border-color);
        vertical-align: top;
    }

    .direction-table tr:last-child td {
        border-bottom: none;
    }

    .dir-number {
        font-weight: 700;
        color: var(--primary-color);
        font-size: 1.1rem;
        width: 40px;
        text-align: center;
    }

    .dir-title {
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 6px;
    }

    .dir-rationale {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    .score-cell {
        text-align: center;
        min-width: 120px;
    }

    .badge-high {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #dcfce7;
        color: #166534;
    }

    .badge-medium {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #fef3c7;
        color: #92400e;
    }

    .badge-low {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #fee2e2;
        color: #991b1b;
    }

    .badge-na {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #e2e8f0;
        color: #475569;
    }

    .findings-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }

    .findings-card {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }

    .findings-card-header {
        padding: 12px 16px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .findings-card-header.bio {
        background: #e0f2fe;
        color: #0369a1;
    }

    .findings-card-header.path {
        background: #fef3c7;
        color: #92400e;
    }

    .findings-card-header.biochem {
        background: #f3e8ff;
        color: #7c3aed;
    }

    .findings-card-header.lit {
        background: #f0fdf4;
        color: #15803d;
    }

    .findings-card-content {
        padding: 16px;
    }

    .findings-card-content ul {
        list-style: none;
        padding: 0;
    }

    .findings-card-content ul li {
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.88rem;
    }

    .findings-card-content ul li:last-child {
        border-bottom: none;
    }

    .pi-assessment {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #0284c7;
    }

    .pi-assessment h3 {
        color: #0369a1;
        margin-bottom: 12px;
    }

    .footer {
        text-align: center;
        padding: 20px;
        color: var(--text-secondary);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-color);
    }

    @media (max-width: 768px) {
        .content { padding: 20px; }
        .findings-grid { grid-template-columns: 1fr; }
        .direction-table { font-size: 0.85rem; }
    }

    .evidence-failed-row { background: #fef2f2; }
    .evidence-partial-row { background: #fffbeb; }
    .ev-warning { padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 600; }
    .ev-failed { background: #fee2e2; color: #991b1b; }
    .ev-partial { background: #fef3c7; color: #92400e; }
    .ev-details { margin-bottom: 16px; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; }
    .ev-details summary { padding: 12px 16px; cursor: pointer; font-weight: 600; color: var(--primary-color); background: var(--primary-light); user-select: none; }
    .ev-details summary:hover { background: #dbeafe; }
    .ev-details-content { padding: 16px; }
    .ev-list { list-style: none; padding: 0; }
    .ev-list li { padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }
    .ev-list li:last-child { border-bottom: none; }
    .ev-pmid { padding: 4px 0 4px 16px !important; font-size: 0.88rem; }
    .ev-why { color: var(--text-secondary); font-size: 0.85rem; display: block; padding-left: 24px; }

    @media print {
        body { background: white; padding: 0; }
        .container { box-shadow: none; border-radius: 0; }
    }"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Integration Report - {project_name}</title>
<style>
{style_block}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Integration Report</h1>
        <div class="meta">{project_name} | {timestamp}</div>
    </div>
    <div class="content">
        {numbered_sections_html}
        {score_section}
    </div>
    <div class="footer">
        Generated by ScAI Multi-Agent Analysis System | {timestamp}
    </div>
</div>
</body>
</html>"""

    reports_dir = get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "integration_report.html"
    tmp_path = html_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(str(tmp_path), str(html_path))

    logger.info(f"Integration report HTML saved to: {html_path}")
    return str(html_path)
