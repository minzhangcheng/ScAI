"""Exploration-phase report HTML.

Moved verbatim out of backend/services/report/variants.py.
"""
import html as html_module
from datetime import datetime
from pathlib import Path

from backend.utils.report_html import (
    markdown_to_html,
    _make_figure_block,
    _build_figure_description_map,
)
from backend.utils.image_utils import compress_image_to_base64
from backend.services.report.score_section import build_score_definitions_section_html
from backend.services.report.figure_discovery import (
    discover_figures_in_workspace,
    ensure_legacy_artifact_inventory,
    prepare_discovered_figures,
)
from backend.models.schemas import AnalysisState
from .figure_records import resolved_record_figure_candidates

logger = __import__('logging').getLogger(__name__)

from .renderers import _render_direction_candidates_rows, _render_direction_evidence_section


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
            ensure_legacy_artifact_inventory(state, workspace_dir)
            fig_desc_map = _build_figure_description_map(state.analysis_log)
            figure_blocks = []
            seen_stems: set = set()
            # Resolve registered artifacts first. Disk discovery alone cannot
            # recover a reviewed figure after its original png_path moved or was
            # overwritten, while the record-owned preview still can.
            discovered = resolved_record_figure_candidates(
                state.publication_figures,
                workspace_dir,
                report_only=True,
            )
            discovered.extend(prepare_discovered_figures(
                discover_figures_in_workspace(workspace_dir),
                state.publication_figures,
                invalidated_stems=state.invalidated_artifact_stems,
                report_only=True,
                allow_unregistered_diagnostics=False,
                registry_version=state.artifact_registry_version,
                legacy_artifacts=state.legacy_artifacts,
            ))
            for candidate in discovered:
                img_file = Path(candidate["path"])
                if img_file.stem in seen_stems:
                    continue
                seen_stems.add(img_file.stem)
                try:
                    b64 = compress_image_to_base64(str(img_file), max_width=1100, quality=85)
                    caption = html_module.escape(
                        candidate.get("description")
                        or fig_desc_map.get(
                            img_file.stem, img_file.stem.replace("_", " ").title()
                        )
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
    direction_rows = _render_direction_candidates_rows(candidates)

    evidence_section_html = _render_direction_evidence_section(exploration_report)

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
        max-width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        /* Wide tables scroll horizontally instead of overflowing the page. */
        display: block;
        overflow-x: auto;
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

    .badge-na {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #e2e8f0;
        color: #475569;
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

    .evidence-failed-row {{ background: #fef2f2; }}
    .evidence-partial-row {{ background: #fffbeb; }}
    .ev-warning {{ padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 600; }}
    .ev-failed {{ background: #fee2e2; color: #991b1b; }}
    .ev-partial {{ background: #fef3c7; color: #92400e; }}
    .ev-details {{ margin-bottom: 16px; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden; }}
    .ev-details summary {{ padding: 12px 16px; cursor: pointer; font-weight: 600; color: var(--primary-color); background: var(--primary-light); user-select: none; }}
    .ev-details summary:hover {{ background: #dbeafe; }}
    .ev-details-content {{ padding: 16px; }}
    .ev-list {{ list-style: none; padding: 0; }}
    .ev-list li {{ padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }}
    .ev-list li:last-child {{ border-bottom: none; }}
    .ev-pmid {{ padding: 4px 0 4px 16px !important; font-size: 0.88rem; }}
    .ev-why {{ color: var(--text-secondary); font-size: 0.85rem; display: block; padding-left: 24px; }}

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
                            <th>Priority</th>
                            <th>Literature Novelty</th>
                            <th>Data Reliability</th>
                        </tr>
                    </thead>
                    <tbody>
                        {direction_rows}
                    </tbody>
                </table>
            </div>
        </div>

        {evidence_section_html}

        <!-- Score Definitions -->
        {build_score_definitions_section_html(state.score_definitions)}

        <!-- PI Assessment -->
        <div class="section">
            <div class="section-header">
                <h2>{"9" if state.score_definitions else "8"}. PI Overall Assessment</h2>
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
