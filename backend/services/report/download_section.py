"""Shared 'Related Reports & Downloads' HTML renderer for detailed reports."""
from __future__ import annotations

import html as html_module
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from .literature_files import render_literature_download_li

esc = html_module.escape
logger = logging.getLogger(__name__)


def build_related_reports_metadata(
    state: Any,
    record_files: Dict[str, Any] | None,
    *,
    reports_dir: Path | None = None,
) -> Dict[str, Any]:
    """Build related-report metadata and stage linked report files in reports_dir."""
    record_files = record_files or {}
    related: Dict[str, Any] = {
        "annotation_record": record_files.get("annotation_record"),
        "direction_records": record_files.get("direction_records", []),
    }

    source = getattr(state, "integration_report_html_path", None)
    if source:
        source_path = Path(source)
        if source_path.exists():
            if reports_dir is None:
                related["integration_report"] = source_path.name
            else:
                target_path = Path(reports_dir) / "integration_report.html"
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if source_path.resolve() != target_path.resolve():
                        shutil.copy2(source_path, target_path)
                    related["integration_report"] = target_path.name
                except OSError:
                    logger.exception("Failed to stage integration report for detailed report appendix")
        else:
            logger.warning("Integration report path is set but missing: %s", source_path)

    return related


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
    if related.get("integration_report"):
        items.append(f'<li><a href="{esc(related["integration_report"])}" download style="color:#4f46e5;">'
                     f'Integration Report</a></li>')
    if literature_report:
        items.append(render_literature_download_li(literature_report))
    if notebook_filename:
        items.append(f'<li><a href="{esc(notebook_filename)}" download style="color:#4f46e5;">'
                     f'Analysis Notebook (.ipynb)</a></li>')
    if not items:
        return ""
    return '<h3>Related Reports &amp; Downloads</h3><ul>' + "".join(items) + '</ul>'
