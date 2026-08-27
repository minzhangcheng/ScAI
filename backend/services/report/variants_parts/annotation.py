"""Annotation-context report HTML.

Moved verbatim out of backend/services/report/variants.py.
"""
import html as html_module
import sys
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
    resolve_archival_render,
)
from backend.services.report.figure_discovery import (
    discover_figures_in_workspace,
    ensure_legacy_artifact_inventory,
    prepare_discovered_figures,
)
from backend.models.schemas import AnalysisState

logger = __import__('logging').getLogger(__name__)

from .. import variant_sections as _variant_sections_owner


_ANNOTATION_STYLE = """    :root {
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
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6; color: var(--text-color); background: var(--bg-color); padding: 20px;
    }
    .container {
        max-width: 1200px; margin: 0 auto; background: var(--white);
        border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); overflow: hidden;
    }
    .header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; padding: 40px; text-align: center;
    }
    .header h1 { font-size: 2rem; margin-bottom: 10px; }
    .header .meta { opacity: 0.9; font-size: 0.9rem; }
    .content { padding: 40px; }
    .section { margin-bottom: 40px; border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }
    .section-header {
        background: var(--primary-light); padding: 16px 24px; border-bottom: 1px solid var(--border-color);
    }
    .section-header h2 { font-size: 1.25rem; color: var(--primary-color); }
    .section-content { padding: 24px; }
    .section-content ul { list-style: none; padding: 0; }
    .section-content ul li {
        padding: 8px 0 8px 20px; position: relative; border-bottom: 1px solid #f1f5f9;
    }
    .section-content ul li:last-child { border-bottom: none; }
    .section-content ul li::before {
        content: '\\25B8'; position: absolute; left: 0; color: var(--primary-color); font-weight: bold;
    }
    .step-output {
        background: #f1f5f9; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem;
        margin-top: 4px; white-space: pre-wrap; max-height: 150px; overflow-y: auto;
    }
    .analysis-step {
        margin-bottom: 28px; padding-bottom: 28px;
        border-bottom: 1px solid #f1f5f9;
    }
    .analysis-step:last-child { border-bottom: none; margin-bottom: 0; }
    .step-header { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
    .step-marker { color: var(--primary-color); font-weight: bold; flex-shrink: 0; margin-top: 2px; }
    .step-desc { font-size: 1rem; color: var(--text-color); line-height: 1.5; }
    .step-figure {
        margin: 14px 0; border: 1px solid var(--border-color); border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .step-figure img { width: 100%; height: auto; display: block; }
    .step-figure-caption {
        padding: 12px 16px; font-size: 0.9rem; color: var(--text-secondary);
        background: #f8fafc; line-height: 1.5; border-top: 1px solid var(--border-color);
    }
    .step-explanation {
        margin-top: 10px; padding: 12px 16px;
        background: #f0f9ff; border-left: 3px solid #0284c7;
        border-radius: 0 6px 6px 0; font-size: 0.88rem; color: #334155; line-height: 1.6;
    }
    .step-output-details { margin-top: 6px; }
    .step-output-details summary {
        cursor: pointer; color: var(--text-secondary); font-size: 0.85rem; user-select: none;
    }
    .lit-result { font-size: 0.88rem; color: var(--text-secondary); }
    .ann-table { width: 100%; max-width: 100%; border-collapse: collapse; font-size: 0.9rem; display: block; overflow-x: auto; }
    .ann-table th {
        background: var(--primary-light); padding: 12px 16px; text-align: left;
        font-size: 0.85rem; color: var(--primary-color); border-bottom: 2px solid var(--border-color);
    }
    .ann-table td { padding: 12px 16px; border-bottom: 1px solid var(--border-color); vertical-align: top; }
    .ann-table tr:last-child td { border-bottom: none; }
    .ann-low { background: #fef2f2; }
    .ann-medium { background: #fffbeb; }
    .cluster-id { font-family: monospace; font-weight: 600; color: var(--primary-color); white-space: nowrap; }
    .markers { font-size: 0.85rem; color: var(--text-secondary); }
    .evidence { font-size: 0.85rem; color: var(--text-secondary); max-width: 300px; }
    .subtypes { font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px; }
    .alt-types {
        margin-top: 6px; padding: 4px 8px; background: #fef3c7; border-radius: 4px;
        font-size: 0.8rem; color: #92400e;
    }
    .low-conf-reason {
        margin-top: 4px; padding: 4px 8px; background: #fee2e2; border-radius: 4px;
        font-size: 0.8rem; color: #991b1b;
    }
    .text-center { text-align: center; }
    .badge-high {
        display: inline-block; padding: 2px 10px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #dcfce7; color: #166534;
    }
    .badge-medium {
        display: inline-block; padding: 2px 10px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #fef3c7; color: #92400e;
    }
    .badge-low {
        display: inline-block; padding: 2px 10px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #fee2e2; color: #991b1b;
    }
    .footer {
        text-align: center; padding: 20px; color: var(--text-secondary);
        font-size: 0.85rem; border-top: 1px solid var(--border-color);
    }
    @media (max-width: 768px) {
        .content { padding: 20px; }
        .figures-grid { grid-template-columns: 1fr; }
        .ann-table { font-size: 0.8rem; }
    }
    @media print {
        body { background: white; padding: 0; }
        .container { box-shadow: none; border-radius: 0; }
    }"""


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
    return _variant_sections_owner.generate_annotation_context_report_html(
        state,
        graph_state,
        cluster_annotations,
        bindings=sys.modules[__name__],
    )
