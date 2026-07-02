"""Discover candidate workspace figures for the brief Results gallery.

This module only enumerates and normalizes candidate images. It does not decide
which figures are shown; PI curation happens in
``PIAgent.select_brief_figures``. Keep it generic: no dataset-, disease-,
tissue-, species-, or project-specific file names.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.config import get_session_workspace
from backend.utils.files import validate_path_within, validate_session_id

logger = logging.getLogger(__name__)

_IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg")

# Source priority for deduplication and ordering (lower wins).
_PRIORITY_FIGURES = 0
_PRIORITY_ROOT = 1
_PRIORITY_NESTED = 2


def infer_figure_type(stem: str) -> str:
    """Infer a coarse figure type from the file stem. Generic rules only."""
    s = stem.lower()
    if "umap" in s:
        return "umap"
    if "volcano" in s:
        return "volcano"
    if "violin" in s:
        return "violin"
    if "dotplot" in s or "dot_plot" in s:
        return "dotplot"
    if "heatmap" in s:
        return "heatmap"
    if "barplot" in s or "bar_plot" in s or "composition" in s or "proportion" in s:
        return "barplot"
    return "other"


def discover_workspace_figures(session_id: str) -> List[Dict[str, Any]]:
    """Return normalized, deduplicated, ordered figure candidates.

    Each candidate is a dict with ``name``, ``type``, ``path``, ``pdf_path``,
    ``mtime``, and ``source``. Only files inside the session workspace are
    returned. Callers are responsible for any candidate cap before prompting.
    """
    validate_session_id(session_id)
    workspace = get_session_workspace(session_id)
    if not workspace.exists():
        return []

    scan_dirs: List[Tuple[Path, str, int]] = []
    figures_dir = workspace / "figures"
    if figures_dir.is_dir():
        scan_dirs.append((figures_dir, "figures", _PRIORITY_FIGURES))
    scan_dirs.append((workspace, "workspace_root", _PRIORITY_ROOT))
    # `*_outputs/figures` covers round*_outputs and stage*_outputs.
    for out_fig in sorted(workspace.glob("*_outputs/figures")):
        if out_fig.is_dir():
            rel = out_fig.relative_to(workspace).as_posix()
            scan_dirs.append((out_fig, rel, _PRIORITY_NESTED))

    raw: List[Dict[str, Any]] = []
    for scan_dir, source, priority in scan_dirs:
        for pattern in _IMAGE_GLOBS:
            for img in scan_dir.glob(pattern):
                try:
                    resolved = validate_path_within(img, workspace)
                    stat = img.stat()
                except Exception:
                    continue
                pdf = img.with_suffix(".pdf")
                pdf_path = None
                if pdf.exists():
                    try:
                        pdf_path = str(validate_path_within(pdf, workspace))
                    except Exception:
                        pdf_path = None
                raw.append({
                    "name": img.stem,
                    "type": infer_figure_type(img.stem),
                    "path": str(resolved),
                    "pdf_path": pdf_path,
                    "mtime": stat.st_mtime,
                    "source": source,
                    "_priority": priority,
                })

    # Deduplicate by stem: keep best (lowest priority, then newest mtime).
    best: Dict[str, Dict[str, Any]] = {}
    for cand in raw:
        current = best.get(cand["name"])
        if current is None or (cand["_priority"], -cand["mtime"]) < (
            current["_priority"], -current["mtime"]
        ):
            best[cand["name"]] = cand

    # Order: source priority, then mtime ascending, then name.
    ordered = sorted(
        best.values(),
        key=lambda c: (c["_priority"], c["mtime"], c["name"]),
    )
    for cand in ordered:
        cand.pop("_priority", None)
    return ordered
