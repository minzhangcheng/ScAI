"""Exploration-phase report HTML.

Moved verbatim out of backend/services/report/variants.py.
"""
import html as html_module
import sys
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
from .. import variant_sections as _variant_sections_owner


_EXPLORATION_STYLE = r"""    :root {
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
        content: '▸';
        position: absolute;
        left: 0;
        color: var(--primary-color);
        font-weight: bold;
    }

    .warning-list li::before {
        content: '⚠' !important;
        color: var(--warning) !important;
    }

    /* Direction evaluation table */
    .direction-table {
        width: 100%;
        max-width: 100%;
        border-collapse: collapse;
        margin-top: 16px;
        /* Wide tables scroll horizontally instead of overflowing the page. */
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

    .score-detail {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 6px;
        text-align: left;
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

    .step-figure {
        margin: 14px 0;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    .step-figure img {
        width: 100%;
        height: auto;
        display: block;
    }

    .step-figure-caption {
        padding: 12px 16px;
        font-size: 0.9rem;
        color: var(--text-secondary);
        background: #f8fafc;
        line-height: 1.5;
        border-top: 1px solid var(--border-color);
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
    return _variant_sections_owner.generate_exploration_report_html(
        state,
        exploration_report,
        round_result,
        bindings=sys.modules[__name__],
    )
