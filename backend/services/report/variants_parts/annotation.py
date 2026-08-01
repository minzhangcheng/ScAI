"""Annotation-context report HTML.

Moved verbatim out of backend/services/report/variants.py.
"""
import html as html_module
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from backend.utils.report_html import (
    markdown_to_html,
    _make_figure_block,
    _build_figure_description_map,
)
from backend.utils.image_utils import compress_image_to_base64
from backend.services.report.score_section import build_score_definitions_section_html
from backend.services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
)
from backend.models.schemas import AnalysisState

logger = __import__('logging').getLogger(__name__)


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
    eligible_figures = eligible_artifacts(state.publication_figures)
    excluded_figure_stems = excluded_artifact_stems(
        state.publication_figures,
        eligible_figures,
        invalidated_stems=state.invalidated_artifact_stems,
    )
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
                if stem in excluded_figure_stems:
                    continue
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
                if img_file.stem in excluded_figure_stems:
                    continue
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
    for pf in eligible_figures:
        pf_name = getattr(pf, 'name', '') or 'Figure'
        b64_data = getattr(pf, 'png_base64', None)
        pf_stems = artifact_stems([pf])
        if b64_data and not (pf_stems & seen_stems):
            seen_stems.update(pf_stems)
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
    .ann-table {{ width: 100%; max-width: 100%; border-collapse: collapse; font-size: 0.9rem; display: block; overflow-x: auto; }}
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
        {build_score_definitions_section_html(state.score_definitions)}
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
