"""Shared 'Related Reports & Downloads' HTML renderer for detailed reports."""
from __future__ import annotations

import html as html_module
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from .history_exports import iter_report_history_members
from .literature_files import render_literature_download_li

esc = html_module.escape
logger = logging.getLogger(__name__)


def _stage_report_history(
    state: Any,
    reports_dir: Path,
    *,
    lineage_id: str,
) -> Dict[str, list[Dict[str, str]]]:
    """Stage retained stage/round reports beside the final report."""
    history_index: Dict[str, Any] = {}
    try:
        for relative_path, content in iter_report_history_members(state, lineage_id):
            if relative_path.is_absolute() or ".." in relative_path.parts:
                logger.warning("Skipped unsafe retained report path: %s", relative_path)
                continue
            target_path = Path(reports_dir) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(content)
            if relative_path.as_posix() == "history/index.json":
                history_index = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        logger.exception("Failed to stage retained stage/round reports")
        return {"stage_reports": [], "round_reports": []}

    stage_reports = []
    round_reports = []
    for entry in history_index.get("reports", []):
        filename = entry.get("path")
        if not filename:
            continue
        if entry.get("kind") == "stage":
            stage = str(entry.get("stage", "stage")).replace("stage", "Stage ").strip()
            stage_reports.append({
                "filename": filename,
                "label": (
                    f"Round {entry.get('round', 1)} {stage} Report — "
                    f"Revision {entry.get('revision', '?')} ({entry.get('status', 'available')})"
                ),
            })
        elif entry.get("kind") == "round":
            title = str(entry.get("title") or "").strip()
            phase = str(entry.get("phase") or "").replace("_", " ").title()
            suffix = title or phase
            round_reports.append({
                "filename": filename,
                "label": (
                    f"Round {entry.get('round', '?')} Report"
                    + (f" — {suffix}" if suffix else "")
                ),
            })
    return {"stage_reports": stage_reports, "round_reports": round_reports}


def build_related_reports_metadata(
    state: Any,
    record_files: Dict[str, Any] | None,
    *,
    reports_dir: Path | None = None,
    lineage_id: str | None = None,
) -> Dict[str, Any]:
    """Build related-report metadata and stage linked report files in reports_dir."""
    record_files = record_files or {}
    related: Dict[str, Any] = {
        "annotation_record": record_files.get("annotation_record"),
        "direction_records": record_files.get("direction_records", []),
        "stage_reports": [],
        "round_reports": [],
    }

    if reports_dir is not None:
        related.update(_stage_report_history(
            state,
            Path(reports_dir),
            lineage_id=(
                lineage_id
                or str(getattr(state, "restore_branch_id", None) or "main")
            ),
        ))

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
    for stage_report in related.get("stage_reports", []):
        items.append(
            f'<li><a href="{esc(stage_report.get("filename", ""))}" download '
            f'style="color:#4f46e5;">{esc(stage_report.get("label", "Stage Report"))}</a></li>'
        )
    for round_report in related.get("round_reports", []):
        items.append(
            f'<li><a href="{esc(round_report.get("filename", ""))}" download '
            f'style="color:#4f46e5;">{esc(round_report.get("label", "Round Report"))}</a></li>'
        )
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
