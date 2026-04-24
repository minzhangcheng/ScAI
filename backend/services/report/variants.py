"""Report variant generators - HTML report builders for different analysis phases."""
import os
import html as html_module
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.utils.report_html import (
    markdown_to_html,
    _make_figure_block,
    _build_figure_description_map,
    render_table_html,
)
from backend.utils.image_utils import compress_image_to_base64, categorize_figures
from backend.models.schemas import AnalysisState

logger = __import__('logging').getLogger(__name__)


def _build_score_definitions_html_section(state: AnalysisState, section_number: str = "") -> str:
    """Build an HTML section for score definitions. Returns empty string if no scores."""
    if not state.score_definitions:
        return ""
    rows = []
    for sd in state.score_definitions:
        genes = html_module.escape(", ".join(sd.gene_list[:10]))
        if len(sd.gene_list) > 10:
            genes += f" ... ({len(sd.gene_list)} total)"
        params_str = html_module.escape(
            ", ".join(f"{k}={v}" for k, v in sd.parameters.items()) if sd.parameters else "default"
        )
        rows.append([
            html_module.escape(sd.score_name),
            f"<code>{html_module.escape(sd.method)}</code>",
            html_module.escape(sd.description),
            params_str,
            genes,
            html_module.escape(sd.gene_list_source or "N/A"),
        ])
    table_html = render_table_html(
        headers=["Score Name", "Method", "Description", "Parameters", "Gene List", "Source"],
        rows=rows,
        style="font-size:0.9em",
    )
    prefix = f"{section_number}. " if section_number else ""
    return f"""
        <div class="section" id="score-definitions">
            <div class="section-header">
                <h2>{prefix}Score Definitions</h2>
            </div>
            <div class="section-content">
                <p>The following scores were computed during the analysis. Each score's methodology is documented for transparency and reproducibility.</p>
                {table_html}
            </div>
        </div>"""


def generate_exploration_report_html(
    state: AnalysisState,
    exploration_report,
    round_result
) -> str:
    """
    Generate a static HTML report for the exploration phase (Report A).
    No LLM calls — data is already structured.

    Args:
        state: AnalysisState with session info
        exploration_report: ExplorationReport model
        round_result: RoundResult from Round 1

    Returns:
        Path to the saved HTML file
    """
    from backend.config import get_session_reports, get_session_workspace

    # Prepare data
    project_name = html_module.escape(state.name or "Untitled Analysis")
    exp_design = html_module.escape(state.build_experimental_design_summary())
    cluster_overview = html_module.escape(exploration_report.cluster_overview or "N/A")
    pi_assessment = markdown_to_html(exploration_report.pi_direction_assessment or "N/A")

    # Data quality warnings
    dq_items = ""
    for w in (exploration_report.data_quality_warnings or []):
        dq_items += f"<li>{html_module.escape(w)}</li>\n"
    if not dq_items:
        dq_items = "<li>No data quality warnings</li>"

    # Key findings sections
    def build_list(items, empty_msg="None identified"):
        if not items:
            return f"<li>{empty_msg}</li>"
        return "\n".join(f"<li>{html_module.escape(str(i))}</li>" for i in items)

    heterogeneity_items = build_list(exploration_report.strongest_heterogeneity)
    pathway_items = build_list(exploration_report.key_pathway_changes)
    abnormal_items = build_list(exploration_report.abnormal_markers)
    low_conf_items = build_list(exploration_report.low_confidence_annotations)

    # Round 1 findings
    bio_findings = build_list(round_result.bioinformatics_findings[:10], "No bioinformatics findings")
    path_findings = build_list(round_result.pathology_findings[:5], "No pathology findings")
    biochem_findings = build_list(round_result.biochemistry_findings[:5], "No biochemistry findings")
    lit_findings = build_list(round_result.literature_findings[:5], "No literature findings")

    # Key analysis figures from Round 1
    key_figures_html = ""
    try:
        workspace_dir = get_session_workspace(state.id) if state.id else None
        if workspace_dir and workspace_dir.exists():
            fig_desc_map = _build_figure_description_map(state.analysis_log)
            figure_blocks = []
            seen_stems: set = set()
            for scan_dir in [workspace_dir, workspace_dir / "figures"]:
                if not scan_dir.exists():
                    continue
                for img_file in sorted(scan_dir.glob("*.jpg")) + sorted(scan_dir.glob("*.png")):
                    if img_file.stem in seen_stems:
                        continue
                    seen_stems.add(img_file.stem)
                    try:
                        b64 = compress_image_to_base64(str(img_file), max_width=1100, quality=85)
                        caption = html_module.escape(
                            fig_desc_map.get(img_file.stem, img_file.stem.replace("_", " ").title())
                        )
                        figure_blocks.append(_make_figure_block(b64, caption))
                    except Exception:
                        logger.warning("Failed to embed figure in exploration report: %s", img_file, exc_info=True)
            if figure_blocks:
                key_figures_html = f"""
        <div class="section">
            <div class="section-header">
                <h2>4b. Key Analysis Figures</h2>
            </div>
            <div class="section-content">
                {"".join(figure_blocks)}
            </div>
        </div>"""
    except Exception:
        logger.warning("Could not build figures section for exploration report", exc_info=True)

    # Direction candidates table
    candidates = exploration_report.research_direction_candidates or []
    direction_rows = ""
    for i, c in enumerate(candidates, 1):
        direction = html_module.escape(str(c.get("direction", "")))
        rationale = html_module.escape(str(c.get("rationale", "")))
        innovation = html_module.escape(str(c.get("expected_innovation", "N/A")))
        novelty = html_module.escape(str(c.get("expected_novelty", "N/A")))
        reliability = html_module.escape(str(c.get("data_reliability", "N/A")))

        def score_badge(text):
            t = text.lower()
            if t.startswith("high"):
                return '<span class="badge-high">High</span>'
            elif t.startswith("medium"):
                return '<span class="badge-medium">Medium</span>'
            elif t.startswith("low"):
                return '<span class="badge-low">Low</span>'
            return f'<span class="badge-medium">{html_module.escape(text[:20])}</span>'

        def score_detail(text):
            # Extract detail after "High/Medium/Low — " or "High/Medium/Low: "
            for sep in [" — ", " - ", ": ", " "]:
                parts = text.split(sep, 1)
                if len(parts) > 1 and parts[0].lower() in ("high", "medium", "low"):
                    return html_module.escape(parts[1])
            return html_module.escape(text)

        direction_rows += f"""
        <tr>
            <td class="dir-number">{i}</td>
            <td>
                <div class="dir-title">{direction}</div>
                <div class="dir-rationale">{rationale}</div>
            </td>
            <td class="score-cell">
                {score_badge(innovation)}
                <div class="score-detail">{score_detail(innovation)}</div>
            </td>
            <td class="score-cell">
                {score_badge(novelty)}
                <div class="score-detail">{score_detail(novelty)}</div>
            </td>
            <td class="score-cell">
                {score_badge(reliability)}
                <div class="score-detail">{score_detail(reliability)}</div>
            </td>
        </tr>
        """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exploration Report - {project_name}</title>
<style>
    :root {{
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
    }}

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background: var(--bg-color);
        padding: 20px;
    }}

    .container {{
        max-width: 1100px;
        margin: 0 auto;
        background: var(--white);
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }}

    .header {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 40px;
        text-align: center;
    }}

    .header h1 {{
        font-size: 2rem;
        margin-bottom: 10px;
    }}

    .header .meta {{
        opacity: 0.9;
        font-size: 0.9rem;
    }}

    .content {{
        padding: 40px;
    }}

    .section {{
        margin-bottom: 40px;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
    }}

    .section-header {{
        background: var(--primary-light);
        padding: 16px 24px;
        border-bottom: 1px solid var(--border-color);
    }}

    .section-header h2 {{
        font-size: 1.25rem;
        color: var(--primary-color);
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .section-content {{
        padding: 24px;
    }}

    .section-content p {{
        margin-bottom: 16px;
        text-align: justify;
    }}

    .section-content ul {{
        list-style: none;
        padding: 0;
    }}

    .section-content ul li {{
        padding: 8px 0 8px 20px;
        position: relative;
        border-bottom: 1px solid #f1f5f9;
    }}

    .section-content ul li:last-child {{
        border-bottom: none;
    }}

    .section-content ul li::before {{
        content: '▸';
        position: absolute;
        left: 0;
        color: var(--primary-color);
        font-weight: bold;
    }}

    .warning-list li::before {{
        content: '⚠' !important;
        color: var(--warning) !important;
    }}

    /* Direction evaluation table */
    .direction-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
    }}

    .direction-table th {{
        background: var(--primary-light);
        padding: 12px 16px;
        text-align: left;
        font-size: 0.85rem;
        color: var(--primary-color);
        border-bottom: 2px solid var(--border-color);
        white-space: nowrap;
    }}

    .direction-table td {{
        padding: 16px;
        border-bottom: 1px solid var(--border-color);
        vertical-align: top;
    }}

    .direction-table tr:last-child td {{
        border-bottom: none;
    }}

    .dir-number {{
        font-weight: 700;
        color: var(--primary-color);
        font-size: 1.1rem;
        width: 40px;
        text-align: center;
    }}

    .dir-title {{
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 6px;
    }}

    .dir-rationale {{
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}

    .score-cell {{
        text-align: center;
        min-width: 120px;
    }}

    .score-detail {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 6px;
        text-align: left;
    }}

    .badge-high {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #dcfce7;
        color: #166534;
    }}

    .badge-medium {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #fef3c7;
        color: #92400e;
    }}

    .badge-low {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #fee2e2;
        color: #991b1b;
    }}

    .findings-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}

    .findings-card {{
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }}

    .findings-card-header {{
        padding: 12px 16px;
        font-weight: 600;
        font-size: 0.9rem;
    }}

    .findings-card-header.bio {{
        background: #e0f2fe;
        color: #0369a1;
    }}

    .findings-card-header.path {{
        background: #fef3c7;
        color: #92400e;
    }}

    .findings-card-header.biochem {{
        background: #f3e8ff;
        color: #7c3aed;
    }}

    .findings-card-header.lit {{
        background: #f0fdf4;
        color: #15803d;
    }}

    .findings-card-content {{
        padding: 16px;
    }}

    .findings-card-content ul {{
        list-style: none;
        padding: 0;
    }}

    .findings-card-content ul li {{
        padding: 6px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.88rem;
    }}

    .findings-card-content ul li:last-child {{
        border-bottom: none;
    }}

    .pi-assessment {{
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #0284c7;
    }}

    .pi-assessment h3 {{
        color: #0369a1;
        margin-bottom: 12px;
    }}

    .step-figure {{
        margin: 14px 0;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }}

    .step-figure img {{
        width: 100%;
        height: auto;
        display: block;
    }}

    .step-figure-caption {{
        padding: 12px 16px;
        font-size: 0.9rem;
        color: var(--text-secondary);
        background: #f8fafc;
        line-height: 1.5;
        border-top: 1px solid var(--border-color);
    }}

    .footer {{
        text-align: center;
        padding: 20px;
        color: var(--text-secondary);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-color);
    }}

    @media (max-width: 768px) {{
        .content {{ padding: 20px; }}
        .findings-grid {{ grid-template-columns: 1fr; }}
        .direction-table {{ font-size: 0.85rem; }}
    }}

    @media print {{
        body {{ background: white; padding: 0; }}
        .container {{ box-shadow: none; border-radius: 0; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Exploration Report</h1>
        <div class="meta">{project_name} | {timestamp}</div>
    </div>
    <div class="content">
        <!-- Experimental Design -->
        <div class="section">
            <div class="section-header">
                <h2>1. Experimental Design</h2>
            </div>
            <div class="section-content">
                <pre style="background:#f1f5f9;padding:16px;border-radius:8px;font-size:0.9rem;white-space:pre-wrap;">{exp_design}</pre>
            </div>
        </div>

        <!-- Cluster Overview -->
        <div class="section">
            <div class="section-header">
                <h2>2. Cluster Overview</h2>
            </div>
            <div class="section-content">
                <p>{cluster_overview}</p>
            </div>
        </div>

        <!-- Data Quality Warnings -->
        <div class="section">
            <div class="section-header">
                <h2>3. Data Quality Assessment</h2>
            </div>
            <div class="section-content">
                <ul class="warning-list">{dq_items}</ul>
            </div>
        </div>

        <!-- Round 1 Findings -->
        <div class="section">
            <div class="section-header">
                <h2>4. Round 1 Findings</h2>
            </div>
            <div class="section-content">
                <div class="findings-grid">
                    <div class="findings-card">
                        <div class="findings-card-header bio">🧬 Bioinformatics</div>
                        <div class="findings-card-content"><ul>{bio_findings}</ul></div>
                    </div>
                    <div class="findings-card">
                        <div class="findings-card-header path">🔬 Pathology</div>
                        <div class="findings-card-content"><ul>{path_findings}</ul></div>
                    </div>
                    <div class="findings-card">
                        <div class="findings-card-header biochem">🧪 Biochemistry</div>
                        <div class="findings-card-content"><ul>{biochem_findings}</ul></div>
                    </div>
                    <div class="findings-card">
                        <div class="findings-card-header lit">📚 Literature</div>
                        <div class="findings-card-content"><ul>{lit_findings}</ul></div>
                    </div>
                </div>
            </div>
        </div>

        {key_figures_html}

        <!-- Key Findings -->
        <div class="section">
            <div class="section-header">
                <h2>5. Key Findings</h2>
            </div>
            <div class="section-content">
                <h3 style="margin-bottom: 8px; color: var(--primary-color);">Strongest Heterogeneity</h3>
                <ul>{heterogeneity_items}</ul>

                <h3 style="margin: 16px 0 8px; color: var(--primary-color);">Key Pathway Changes</h3>
                <ul>{pathway_items}</ul>

                <h3 style="margin: 16px 0 8px; color: var(--primary-color);">Abnormal Markers</h3>
                <ul>{abnormal_items}</ul>

                <h3 style="margin: 16px 0 8px; color: var(--primary-color);">Low Confidence Annotations</h3>
                <ul>{low_conf_items}</ul>
            </div>
        </div>

        <!-- Direction Candidates -->
        <div class="section">
            <div class="section-header">
                <h2>6. Research Direction Candidates</h2>
            </div>
            <div class="section-content">
                <table class="direction-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Direction & Rationale</th>
                            <th>Innovation</th>
                            <th>Novelty</th>
                            <th>Reliability</th>
                        </tr>
                    </thead>
                    <tbody>
                        {direction_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Score Definitions -->
        {_build_score_definitions_html_section(state, "7")}

        <!-- PI Assessment -->
        <div class="section">
            <div class="section-header">
                <h2>{"8" if state.score_definitions else "7"}. PI Overall Assessment</h2>
            </div>
            <div class="section-content">
                <div class="pi-assessment">
                    <h3>🎯 PI Direction Assessment</h3>
                    {pi_assessment}
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        Generated by ScAI Multi-Agent Analysis System | {timestamp}
    </div>
</div>
</body>
</html>"""

    # Save HTML file
    reports_dir = get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "exploration_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Exploration report HTML saved to: {html_path}")
    return str(html_path)


def generate_detailed_report_segment_html(
    state: "AnalysisState",
    phase: str,
    title: str,
    graph_state: Optional[Dict[str, Any]] = None,
    round_result=None,
) -> str:
    """Generate one segment of the incremental detailed report as HTML.

    Args:
        state: Current AnalysisState
        phase: "pre_annotation" or "exploration_complete" or "supplementary_*"
        title: Human-readable segment title
        graph_state: Current LangGraph WorkflowState (for in-progress results)
        round_result: Completed RoundResult (for post-round segments)

    Returns:
        HTML string for this segment
    """
    parts = []
    parts.append(f'<div class="report-segment" id="{phase}">')
    parts.append(f'<h2 class="segment-title">{html_module.escape(title)}</h2>')
    parts.append(f'<p class="segment-timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')

    try:
        exp_summary = state.build_experimental_design_summary()
        if exp_summary:
            parts.append('<div class="segment-section">')
            parts.append('<h3>Experimental Design</h3>')
            parts.append(f'<div class="content">{markdown_to_html(exp_summary)}</div>')
            parts.append('</div>')
    except Exception as e:
        logger.debug(f"Experimental design section failed: {e}")
        pass

    # --- Section: Agent Findings (from graph_state if available) ---
    if graph_state:
        for agent_key, agent_label in [
            ("biochemistry_result", "Biochemistry Findings"),
            ("pathology_result", "Pathology Findings"),
            ("bioinformatics_result", "Bioinformatics Findings"),
            ("literature_result", "Literature Research"),
        ]:
            result = graph_state.get(agent_key)
            if result and isinstance(result, dict):
                findings = result.get("findings", [])
                detail = result.get("analysis_detail") or result.get("summary", "")
                if findings or detail:
                    parts.append(f'<div class="segment-section"><h3>{agent_label}</h3>')
                    if findings:
                        parts.append('<ul>')
                        for f in findings:
                            parts.append(f'<li>{html_module.escape(str(f))}</li>')
                        parts.append('</ul>')
                    if detail:
                        parts.append(f'<div class="detail">{markdown_to_html(str(detail)[:3000])}</div>')
                    parts.append('</div>')

    # --- Section: Analysis Log Summary ---
    if state.analysis_log:
        recent_steps = state.analysis_log[-50:]
        parts.append('<div class="segment-section"><h3>Analysis Steps Executed</h3>')
        parts.append('<table class="steps-table"><thead><tr><th>#</th><th>Step</th><th>Status</th><th>Details</th></tr></thead><tbody>')
        for i, step in enumerate(recent_steps, 1):
            # Derive status from code_error field (AnalysisStep has no status field)
            has_error = bool(step.code_error)
            status_label = "error" if has_error else "completed"
            status_class = "error" if has_error else "success"
            detail_text = ""
            if step.step_type == "search_literature" and step.literature_topic:
                detail_text = html_module.escape(step.literature_topic[:120])
            elif step.code:
                detail_text = html_module.escape(step.code[:100])
            parts.append(f'<tr><td>{i}</td><td>{html_module.escape(step.step_type)}</td>'
                         f'<td class="{status_class}">{html_module.escape(status_label)}</td>'
                         f'<td>{detail_text}</td></tr>')
        parts.append('</tbody></table></div>')

    try:
        from backend.config import get_session_workspace
        workspace_dir = get_session_workspace(state.id)
        if workspace_dir and workspace_dir.exists():
            figures_subdir = workspace_dir / "figures"
            categories = categorize_figures(figures_subdir if figures_subdir.exists() else workspace_dir)
            fig_desc_map = _build_figure_description_map(state.analysis_log)
            has_figures = any(figs for figs in categories.values())
            if has_figures:
                parts.append('<div class="segment-section"><h3>Generated Figures</h3>')
                parts.append('<div class="figure-gallery">')
                count = 0
                for cat, figs in categories.items():
                    for fig_info in figs[:5]:
                        if count >= 20:
                            break
                        try:
                            fig_path = fig_info.get("png_path") or fig_info.get("path", "")
                            if not fig_path:
                                continue
                            b64 = compress_image_to_base64(fig_path, max_width=1000, quality=82)
                            fig_name = fig_info.get("name", "")
                            caption = fig_desc_map.get(fig_name, fig_name.replace("_", " ").title())
                            parts.append(f'<div class="figure-item"><img src="{b64}" />'
                                         f'<p class="caption">{html_module.escape(str(caption))}</p></div>')
                            count += 1
                        except Exception as e:
                            logger.debug(f"Figure embedding failed for {fig_info.get('name', '?')}: {e}")
                parts.append('</div></div>')
    except Exception as e:
        logger.warning(f"Figure section generation failed: {e}")

    # --- Section: Round Result (if provided) ---
    if round_result:
        parts.append('<div class="segment-section"><h3>PI Synthesis</h3>')
        if hasattr(round_result, 'pi_summary') and round_result.pi_summary:
            parts.append(f'<div class="content">{markdown_to_html(round_result.pi_summary)}</div>')
        if hasattr(round_result, 'conflicts') and round_result.conflicts:
            parts.append('<h4>Identified Conflicts</h4><ul>')
            for c in round_result.conflicts:
                parts.append(f'<li>{html_module.escape(c.get("description", str(c)))}</li>')
            parts.append('</ul>')
        if hasattr(round_result, 'next_steps') and round_result.next_steps:
            parts.append('<h4>Recommended Next Steps</h4><ul>')
            for s in round_result.next_steps:
                parts.append(f'<li>{html_module.escape(str(s))}</li>')
            parts.append('</ul>')
        parts.append('</div>')

    # --- Section: Exploration Report (if available) ---
    if phase == "exploration_complete" and state.exploration_report:
        er = state.exploration_report
        parts.append('<div class="segment-section"><h3>Exploration Report Summary</h3>')
        if er.cluster_overview:
            parts.append(f'<p><strong>Cluster Overview:</strong> {html_module.escape(er.cluster_overview)}</p>')
        if er.data_quality_warnings:
            parts.append('<h4>Data Quality Warnings</h4><ul>')
            for w in er.data_quality_warnings:
                parts.append(f'<li class="warning">{html_module.escape(w)}</li>')
            parts.append('</ul>')
        if er.pi_direction_assessment:
            parts.append(f'<h4>PI Direction Assessment</h4>'
                         f'<div class="content">{markdown_to_html(er.pi_direction_assessment)}</div>')
        parts.append('</div>')

    parts.append('</div>')  # Close report-segment
    return "\n".join(parts)


def generate_combined_detailed_report_html(state: "AnalysisState") -> str:
    """Generate combined HTML from all detailed report segments.

    Returns:
        Path to the saved HTML file
    """
    from backend.config import get_session_reports
    reports_dir = Path(get_session_reports(state.id))
    reports_dir.mkdir(parents=True, exist_ok=True)

    segments_html = "\n<hr class='segment-divider'>\n".join(
        seg.content_html for seg in state.detailed_report_segments if seg.content_html
    )

    toc_items = "".join(
        f'<li><a href="#{seg.phase}">{html_module.escape(seg.title)}'
        f' (Round {seg.round_number})</a></li>'
        for seg in state.detailed_report_segments
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Detailed Analysis Report - {html_module.escape(state.name)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8fafc; color: #1e293b; }}
.report-header {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; }}
.report-header h1 {{ margin: 0 0 0.5rem; font-size: 1.5rem; }}
.toc {{ background: white; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.toc h2 {{ margin-top: 0; font-size: 1.1rem; }}
.toc ul {{ padding-left: 1.5rem; }}
.toc li {{ margin: 0.3rem 0; }}
.toc a {{ color: #6366f1; text-decoration: none; }}
.report-segment {{ background: white; padding: 2rem; border-radius: 0.75rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.segment-title {{ color: #6366f1; margin-top: 0; }}
.segment-timestamp {{ color: #94a3b8; font-size: 0.85rem; }}
.segment-section {{ margin: 1.5rem 0; padding: 1rem; background: #f1f5f9; border-radius: 0.5rem; }}
.segment-section h3 {{ color: #334155; margin-top: 0; font-size: 1rem; }}
.segment-section h4 {{ color: #475569; font-size: 0.9rem; }}
.steps-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
.steps-table th, .steps-table td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
.steps-table th {{ background: #e2e8f0; }}
.success {{ color: #16a34a; }}
.error {{ color: #dc2626; }}
.warning {{ color: #d97706; }}
.figure-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr)); gap: 28px; }}
.figure-item {{ border: 1px solid #e2e8f0; border-radius: 0.5rem; overflow: hidden; }}
.figure-item img {{ width: 100%; height: auto; display: block; }}
.caption {{ padding: 0.75rem 1rem; font-size: 0.88rem; color: #334155; line-height: 1.5; background: #f8fafc; }}
.segment-divider {{ border: none; border-top: 2px dashed #cbd5e1; margin: 2rem 0; }}
.detail {{ font-size: 0.9rem; color: #475569; }}
.content {{ line-height: 1.6; }}
</style>
</head>
<body>
<div class="report-header">
  <h1>Detailed Analysis Report</h1>
  <p>{html_module.escape(state.name)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
<div class="toc">
  <h2>Table of Contents</h2>
  <ul>{toc_items}</ul>
</div>
{segments_html}
</body>
</html>"""

    output_path = reports_dir / "detailed_report.html"
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(str(tmp_path), str(output_path))

    state.detailed_report_html_path = str(output_path)
    logger.info(f"Combined detailed report saved to: {output_path}")
    return str(output_path)


def generate_annotation_context_report_html(
    state: AnalysisState,
    graph_state: Optional[Dict[str, Any]],
    cluster_annotations: list
) -> str:
    """
    Generate a self-contained HTML report providing full analysis context
    for the cell annotation confirmation step. No LLM calls.

    Args:
        state: AnalysisState with session info
        graph_state: Current workflow graph state dict
        cluster_annotations: List of ClusterAnnotation objects

    Returns:
        Path to the saved HTML file
    """
    from backend.config import get_session_reports, get_session_workspace

    project_name = html_module.escape(state.name or "Untitled Analysis")
    exp_design = html_module.escape(state.build_experimental_design_summary())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Biochemistry & Pathology findings ---
    biochem_html = ""
    patho_html = ""
    if graph_state:
        biochem = graph_state.get("biochemistry_result")
        if biochem and biochem.get("findings"):
            items = "\n".join(f"<li>{html_module.escape(str(f))}</li>" for f in biochem["findings"])
            biochem_html = f"""
        <div class="section">
            <div class="section-header"><h2>2. Biochemistry Findings</h2></div>
            <div class="section-content"><ul>{items}</ul></div>
        </div>"""

        patho = graph_state.get("pathology_result")
        if patho and patho.get("findings"):
            items = "\n".join(f"<li>{html_module.escape(str(f))}</li>" for f in patho["findings"])
            patho_html = f"""
        <div class="section">
            <div class="section-header"><h2>3. Pathology Findings</h2></div>
            <div class="section-content"><ul>{items}</ul></div>
        </div>"""

    # --- Build figure description map ---
    figure_desc_map = _build_figure_description_map(state.analysis_log)
    workspace_dir = get_session_workspace(state.id) if state.id else None

    def _resolve_fig_path(fname: str) -> Optional[Path]:
        """Find figure file in workspace or figures subdirectory."""
        if not workspace_dir:
            return None
        for candidate in [workspace_dir / fname, workspace_dir / "figures" / fname]:
            if candidate.exists():
                return candidate
        return None

    # --- Integrated bioinformatics steps + figures ---
    bio_steps_html = ""
    bio_steps = [s for s in state.analysis_log if s.step_type == "execute_code"]
    linked_stems: set = set()

    if bio_steps:
        step_items = []
        for s in bio_steps:
            desc = html_module.escape(s.code_description or f"Step {s.step_number}")

            output_html = ""
            if s.code_output:
                output_preview = html_module.escape(s.code_output[:400])
                output_html = (
                    f'<details class="step-output-details">'
                    f'<summary>代码输出</summary>'
                    f'<pre class="step-output">{output_preview}</pre>'
                    f'</details>'
                )

            explanation_html = ""
            if s.agent_reasoning:
                explanation_html = (
                    f'<div class="step-explanation">'
                    f'{markdown_to_html(s.agent_reasoning[:800])}'
                    f'</div>'
                )

            step_fig_blocks = []
            for fname in (s.saved_figure_filenames or []):
                stem = Path(fname).stem
                linked_stems.add(stem)
                fpath = _resolve_fig_path(fname)
                if fpath:
                    try:
                        b64 = compress_image_to_base64(str(fpath), max_width=1100, quality=85)
                        caption = html_module.escape(
                            figure_desc_map.get(stem, s.code_description or fname)
                        )
                        step_fig_blocks.append(_make_figure_block(b64, caption))
                    except Exception:
                        logger.warning("Failed to embed step figure: %s", fpath, exc_info=True)

            figures_block = "".join(step_fig_blocks)
            step_items.append(
                f'<div class="analysis-step">'
                f'<div class="step-header"><span class="step-marker">&#9656;</span>'
                f'<strong class="step-desc">{desc}</strong></div>'
                f'{explanation_html}'
                f'{output_html}'
                f'{figures_block}'
                f'</div>'
            )

        bio_steps_html = f"""
        <div class="section">
            <div class="section-header"><h2>4. Bioinformatics Analysis Steps</h2></div>
            <div class="section-content">{"".join(step_items)}</div>
        </div>"""

    # --- Orphan figures (workspace figures not linked to any step) ---
    figures_html = ""
    orphan_blocks = []
    seen_stems: set = set(linked_stems)

    if workspace_dir and workspace_dir.exists():
        scan_dirs = [workspace_dir]
        figures_subdir = workspace_dir / "figures"
        if figures_subdir.exists():
            scan_dirs.append(figures_subdir)

        for scan_dir in scan_dirs:
            for img_file in sorted(scan_dir.glob("*.jpg")) + sorted(scan_dir.glob("*.png")):
                if img_file.stem in seen_stems:
                    continue
                seen_stems.add(img_file.stem)
                try:
                    b64 = compress_image_to_base64(str(img_file), max_width=1100, quality=85)
                    caption = html_module.escape(
                        figure_desc_map.get(img_file.stem, img_file.stem.replace("_", " ").title())
                    )
                    orphan_blocks.append(_make_figure_block(b64, caption))
                except Exception:
                    logger.warning("Failed to embed figure in HTML: %s", img_file, exc_info=True)

    # From publication_figures
    for pf in (state.publication_figures or []):
        pf_name = getattr(pf, 'name', '') or 'Figure'
        b64_data = getattr(pf, 'png_base64', None)
        if b64_data and pf_name not in seen_stems:
            seen_stems.add(pf_name)
            title = html_module.escape(pf_name)
            fmt = "png" if b64_data[:5] == "iVBOR" else "jpeg"
            orphan_blocks.append(_make_figure_block(f"data:image/{fmt};base64,{b64_data}", title))

    if orphan_blocks:
        section_num = "5" if bio_steps_html else "4"
        figures_html = f"""
        <div class="section">
            <div class="section-header"><h2>{section_num}. Additional Figures</h2></div>
            <div class="section-content">{"".join(orphan_blocks)}</div>
        </div>"""

    # --- Literature research ---
    lit_html = ""
    lit_steps = [s for s in state.analysis_log if s.step_type == "search_literature"]
    if lit_steps:
        lit_items = []
        for s in lit_steps:
            topic = html_module.escape(s.literature_topic or "Unknown topic")
            result = html_module.escape((s.literature_result or "No findings")[:500])
            lit_items.append(f"<li><strong>{topic}</strong><br/><span class=\"lit-result\">{result}</span></li>")
        lit_html = f"""
        <div class="section">
            <div class="section-header"><h2>6. Literature Research</h2></div>
            <div class="section-content"><ul>{"".join(lit_items)}</ul></div>
        </div>"""

    # --- Cluster annotation table ---
    ann_rows = ""
    for ca in cluster_annotations:
        cid = html_module.escape(str(ca.cluster_id))
        ct = html_module.escape(ca.cell_type)
        subtypes = html_module.escape(", ".join(ca.subtypes)) if ca.subtypes else ""
        markers = html_module.escape(", ".join(ca.key_markers[:10]))
        conf = ca.confidence
        evidence = html_module.escape(ca.evidence[:300]) if ca.evidence else ""

        conf_class = "badge-high" if conf == "high" else ("badge-medium" if conf == "medium" else "badge-low")
        conf_badge = f'<span class="{conf_class}">{html_module.escape(conf)}</span>'

        alt_types = ""
        if ca.alternative_cell_types:
            alt_list = ", ".join(html_module.escape(t) for t in ca.alternative_cell_types)
            alt_types = f'<div class="alt-types">Alternatives: {alt_list}</div>'

        reason = ""
        if ca.low_confidence_reason:
            reason = f'<div class="low-conf-reason">{html_module.escape(ca.low_confidence_reason)}</div>'

        ann_rows += f"""
            <tr class="{'ann-low' if conf == 'low' else ('ann-medium' if conf == 'medium' else '')}">
                <td class="cluster-id">{cid}</td>
                <td>{ct}{f'<div class="subtypes">{subtypes}</div>' if subtypes else ''}{alt_types}{reason}</td>
                <td class="text-center">{conf_badge}</td>
                <td class="markers">{markers}</td>
                <td class="evidence">{evidence}</td>
            </tr>"""

    ann_html = f"""
        <div class="section">
            <div class="section-header"><h2>7. Cluster Annotation Details</h2></div>
            <div class="section-content">
                <table class="ann-table">
                    <thead>
                        <tr>
                            <th>Cluster</th>
                            <th>Cell Type</th>
                            <th>Confidence</th>
                            <th>Key Markers</th>
                            <th>Evidence</th>
                        </tr>
                    </thead>
                    <tbody>{ann_rows}</tbody>
                </table>
            </div>
        </div>"""

    # --- Full HTML ---
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Annotation Context Report - {project_name}</title>
<style>
    :root {{
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
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6; color: var(--text-color); background: var(--bg-color); padding: 20px;
    }}
    .container {{
        max-width: 1200px; margin: 0 auto; background: var(--white);
        border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); overflow: hidden;
    }}
    .header {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; padding: 40px; text-align: center;
    }}
    .header h1 {{ font-size: 2rem; margin-bottom: 10px; }}
    .header .meta {{ opacity: 0.9; font-size: 0.9rem; }}
    .content {{ padding: 40px; }}
    .section {{ margin-bottom: 40px; border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }}
    .section-header {{
        background: var(--primary-light); padding: 16px 24px; border-bottom: 1px solid var(--border-color);
    }}
    .section-header h2 {{ font-size: 1.25rem; color: var(--primary-color); }}
    .section-content {{ padding: 24px; }}
    .section-content ul {{ list-style: none; padding: 0; }}
    .section-content ul li {{
        padding: 8px 0 8px 20px; position: relative; border-bottom: 1px solid #f1f5f9;
    }}
    .section-content ul li:last-child {{ border-bottom: none; }}
    .section-content ul li::before {{
        content: '\\25B8'; position: absolute; left: 0; color: var(--primary-color); font-weight: bold;
    }}
    .step-output {{
        background: #f1f5f9; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem;
        margin-top: 4px; white-space: pre-wrap; max-height: 150px; overflow-y: auto;
    }}
    .analysis-step {{
        margin-bottom: 28px; padding-bottom: 28px;
        border-bottom: 1px solid #f1f5f9;
    }}
    .analysis-step:last-child {{ border-bottom: none; margin-bottom: 0; }}
    .step-header {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }}
    .step-marker {{ color: var(--primary-color); font-weight: bold; flex-shrink: 0; margin-top: 2px; }}
    .step-desc {{ font-size: 1rem; color: var(--text-color); line-height: 1.5; }}
    .step-figure {{
        margin: 14px 0; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .step-figure img {{ width: 100%; height: auto; display: block; }}
    .step-figure-caption {{
        padding: 12px 16px; font-size: 0.9rem; color: var(--text-secondary);
        background: #f8fafc; line-height: 1.5; border-top: 1px solid var(--border-color);
    }}
    .step-explanation {{
        margin-top: 10px; padding: 12px 16px;
        background: #f0f9ff; border-left: 3px solid #0284c7;
        border-radius: 0 6px 6px 0; font-size: 0.88rem; color: #334155; line-height: 1.6;
    }}
    .step-output-details {{ margin-top: 6px; }}
    .step-output-details summary {{
        cursor: pointer; color: var(--text-secondary); font-size: 0.85rem; user-select: none;
    }}
    .lit-result {{ font-size: 0.88rem; color: var(--text-secondary); }}
    .ann-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    .ann-table th {{
        background: var(--primary-light); padding: 12px 16px; text-align: left;
        font-size: 0.85rem; color: var(--primary-color); border-bottom: 2px solid var(--border-color);
    }}
    .ann-table td {{ padding: 12px 16px; border-bottom: 1px solid var(--border-color); vertical-align: top; }}
    .ann-table tr:last-child td {{ border-bottom: none; }}
    .ann-low {{ background: #fef2f2; }}
    .ann-medium {{ background: #fffbeb; }}
    .cluster-id {{ font-family: monospace; font-weight: 600; color: var(--primary-color); white-space: nowrap; }}
    .markers {{ font-size: 0.85rem; color: var(--text-secondary); }}
    .evidence {{ font-size: 0.85rem; color: var(--text-secondary); max-width: 300px; }}
    .subtypes {{ font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px; }}
    .alt-types {{
        margin-top: 6px; padding: 4px 8px; background: #fef3c7; border-radius: 4px;
        font-size: 0.8rem; color: #92400e;
    }}
    .low-conf-reason {{
        margin-top: 4px; padding: 4px 8px; background: #fee2e2; border-radius: 4px;
        font-size: 0.8rem; color: #991b1b;
    }}
    .text-center {{ text-align: center; }}
    .badge-high {{
        display: inline-block; padding: 2px 10px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #dcfce7; color: #166534;
    }}
    .badge-medium {{
        display: inline-block; padding: 2px 10px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #fef3c7; color: #92400e;
    }}
    .badge-low {{
        display: inline-block; padding: 2px 10px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #fee2e2; color: #991b1b;
    }}
    .footer {{
        text-align: center; padding: 20px; color: var(--text-secondary);
        font-size: 0.85rem; border-top: 1px solid var(--border-color);
    }}
    @media (max-width: 768px) {{
        .content {{ padding: 20px; }}
        .figures-grid {{ grid-template-columns: 1fr; }}
        .ann-table {{ font-size: 0.8rem; }}
    }}
    @media print {{
        body {{ background: white; padding: 0; }}
        .container {{ box-shadow: none; border-radius: 0; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Annotation Context Report</h1>
        <div class="meta">{project_name} | {timestamp}</div>
    </div>
    <div class="content">
        <div class="section">
            <div class="section-header"><h2>1. Experimental Design</h2></div>
            <div class="section-content">
                <pre style="background:#f1f5f9;padding:12px;border-radius:8px;font-size:0.85rem;white-space:pre-wrap;">{exp_design}</pre>
            </div>
        </div>
        {biochem_html}
        {patho_html}
        {bio_steps_html}
        {figures_html}
        {lit_html}
        {ann_html}
        {_build_score_definitions_html_section(state)}
    </div>
    <div class="footer">
        Generated by ScAI Multi-Agent Analysis System | {timestamp}
    </div>
</div>
</body>
</html>"""

    # Save HTML file
    reports_dir = get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "annotation_context_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Annotation context report HTML saved to: {html_path}")
    return str(html_path)