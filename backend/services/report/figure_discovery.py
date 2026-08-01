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
_PRIORITY_DEEP = 3
_PRIORITY_DIAGNOSTIC = 4

# Directory names the agent does not create as analysis output. ``reports`` holds
# copies the report generators already wrote; re-discovering them would offer the
# same image twice under a path that is not the analysis's own.
_SKIP_DIRS = frozenset({"__pycache__", ".ipynb_checkpoints", "reports", "tables"})

# Platform-written validation output. It is real and worth reaching, but it is a
# diagnostic rather than a result, so it sorts last and is tagged for the caller.
_DIAGNOSTIC_DIRS = frozenset({"annotation_validation"})

# How many directory levels below the workspace a candidate may sit. Across 27
# real sessions the deepest image was at depth 3, so the bound admits exactly
# that tier; it exists so an unforeseen tool that writes a deep tree cannot turn
# discovery into a full-disk walk. It is counted over the directories alone --
# ``relative_to(workspace).parts`` also carries the filename, and measuring
# against that admitted only two levels, excluding the very tier the corpus said
# was needed.
_MAX_SCAN_DEPTH = 3


def infer_figure_type(stem: str) -> str:
    """Infer a coarse figure type from the file stem. Generic rules only."""
    s = stem.lower()
    if "umap" in s:
        return "umap"
    if "volcano" in s:
        return "volcano"
    if "qc" in s or "quality_control" in s:
        return "qc"
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
    ``mtime``, ``source``, ``rel_path``, and ``diagnostic``. Only files inside the
    session workspace are returned. Callers are responsible for any candidate cap
    before prompting.

    The three named scan roots below are a priority ordering, not the reach of the
    search: nothing constrains where the agent saves a figure, and a run that put
    its work in a directory of its own invention had every one of those figures
    excluded from the Results gallery while the detailed report — which resolves
    recursively — showed them. The recursive tier closes that gap.
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

    def _add(img: Path, source: str, priority: int) -> None:
        try:
            resolved = validate_path_within(img, workspace)
            stat = img.stat()
        except Exception:
            return
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
            "rel_path": img.relative_to(workspace).as_posix(),
            "diagnostic": priority == _PRIORITY_DIAGNOSTIC,
            "_priority": priority,
        })

    for scan_dir, source, priority in scan_dirs:
        for pattern in _IMAGE_GLOBS:
            for img in scan_dir.glob(pattern):
                _add(img, source, priority)

    # Recursive tier: anywhere else the agent chose to write. Deduplication below
    # keeps the named roots winning any stem they also hold.
    for pattern in _IMAGE_GLOBS:
        for img in workspace.rglob(pattern):
            try:
                parts = img.relative_to(workspace).parts
            except ValueError:
                continue
            dirs = parts[:-1]
            if len(dirs) > _MAX_SCAN_DEPTH:
                continue
            if any(d.startswith(".") or d in _SKIP_DIRS for d in dirs):
                continue
            if not dirs:
                continue  # already covered by the workspace-root tier
            priority = (
                _PRIORITY_DIAGNOSTIC
                if any(d in _DIAGNOSTIC_DIRS for d in dirs)
                else _PRIORITY_DEEP
            )
            _add(img, "/".join(dirs), priority)

    # Deduplicate by stem: keep best (lowest priority, then newest mtime).
    best: Dict[str, Dict[str, Any]] = {}
    for cand in raw:
        current = best.get(cand["name"])
        if current is None or (cand["_priority"], -cand["mtime"]) < (
            current["_priority"], -current["mtime"]
        ):
            best[cand["name"]] = cand

    # Order: source priority, then mtime DESCENDING, then name. Callers cap the
    # list by head-slicing it, so ascending order made the cap discard the newest
    # figures — the ones a late round just produced — and keep the oldest.
    ordered = sorted(
        best.values(),
        key=lambda c: (c["_priority"], -c["mtime"], c["name"]),
    )
    for cand in ordered:
        cand.pop("_priority", None)
    return ordered
