"""Shared 'Related Reports & Downloads' HTML renderer for detailed reports."""
from __future__ import annotations

import html as html_module
from typing import Any, Dict

from .literature_files import render_literature_download_li

esc = html_module.escape


def render_related_reports_downloads_html(
    related: Dict[str, Any],
    *,
    notebook_filename: str = "",
    literature_report: str = "",
) -> str:
    """Render the appendix 'Related Reports & Downloads' block from a normalized
    related-reports dict ({annotation_record, direction_records:[{filename,label}]}).
    Returns '' when there is nothing to list.
    """
    related = related or {}
    if not (related or notebook_filename or literature_report):
        return ""
    items = []
    if related.get("annotation_record"):
        items.append(f'<li><a href="{esc(related["annotation_record"])}" download style="color:#4f46e5;">'
                     f'Cell Annotation Record</a></li>')
    for dr in related.get("direction_records", []):
        items.append(f'<li><a href="{esc(dr["filename"])}" download style="color:#4f46e5;">'
                     f'Direction Selection Record — {esc(dr["label"])}</a></li>')
    if literature_report:
        items.append(render_literature_download_li(literature_report))
    if notebook_filename:
        items.append(f'<li><a href="{esc(notebook_filename)}" download style="color:#4f46e5;">'
                     f'Analysis Notebook (.ipynb)</a></li>')
    if not items:
        return ""
    return '<h3>Related Reports &amp; Downloads</h3><ul>' + "".join(items) + '</ul>'
