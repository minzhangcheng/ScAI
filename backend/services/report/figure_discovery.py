"""Discover candidate workspace figures for the brief Results gallery.

This module only enumerates and normalizes candidate images. It does not decide
which figures are shown; PI curation happens in
``PIAgent.select_brief_figures``. Keep it generic: no dataset-, disease-,
tissue-, species-, or project-specific file names.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from backend.config import get_session_workspace
from backend.services.report.artifacts import (
    archival_path_matches_record,
    archival_render_available,
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
    gallery_artifacts,
    image_file_sha256,
    MIN_ARCHIVAL_PREVIEW_HEIGHT,
    MIN_ARCHIVAL_PREVIEW_WIDTH,
)
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


def discover_figures_in_workspace(workspace: Path) -> List[Dict[str, Any]]:
    """Return normalized, deduplicated figure candidates below ``workspace``.

    Each candidate is a dict with ``name``, ``type``, ``path``, ``pdf_path``,
    ``mtime``, ``source``, ``rel_path``, and ``diagnostic``. Only files inside
    ``workspace`` are returned. Callers are responsible for applying artifact
    eligibility and any candidate cap before prompting or rendering.

    The three named scan roots below are a priority ordering, not the reach of the
    search: nothing constrains where the agent saves a figure, and a run that put
    its work in a directory of its own invention had every one of those figures
    excluded from the Results gallery while the detailed report — which resolves
    recursively — showed them. The recursive tier closes that gap.
    """
    workspace = Path(workspace)
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
            # Automatic post-execution repair is computation-only. Historical
            # versions could leave unreviewed images in a supplementary-named
            # tree; keep them visible to browsing surfaces if desired, but never
            # promote them into an archival report without a PublicationFigure
            # record.
            "restricted_supplementary": any(
                "supplement" in part.lower()
                for part in img.relative_to(workspace).parts
            ),
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

    # Deduplicate by stem: an automatic restricted-supplementary render must
    # never hide a normal analysis output with the same name.  Those restricted
    # executors are computation-only; choosing their file here can make the
    # later report filter discard the stem altogether even though a legitimate
    # non-restricted render exists.  Within the same restriction class, retain
    # the established source-priority/newest ordering.
    best: Dict[str, Dict[str, Any]] = {}
    for cand in raw:
        current = best.get(cand["name"])
        candidate_key = (
            bool(cand.get("restricted_supplementary")),
            cand["_priority"],
            -cand["mtime"],
        )
        current_key = (
            bool(current.get("restricted_supplementary")),
            current["_priority"],
            -current["mtime"],
        ) if current is not None else None
        if current is None or candidate_key < current_key:
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


def discover_workspace_figures(session_id: str) -> List[Dict[str, Any]]:
    """Discover figures for one validated session workspace.

    This public session-id wrapper preserves the Results-gallery API while
    report generators that already hold a workspace path reuse the same bounded,
    recursive implementation through :func:`discover_figures_in_workspace`.
    """
    validate_session_id(session_id)
    return discover_figures_in_workspace(get_session_workspace(session_id))


def publication_figure_metadata(
    publication_figures: Iterable[Any] | None,
    *,
    report_only: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Index authoritative figure metadata by every known filename stem.

    Registered metadata outranks filename inference. For archival reports only
    report-eligible records are indexed; browsing callers may request the weaker
    gallery predicate explicitly.
    """
    figures = list(publication_figures or [])
    kept = eligible_artifacts(figures) if report_only else gallery_artifacts(figures)
    indexed: Dict[str, Dict[str, Any]] = {}
    for figure in kept:
        status = getattr(figure, "review_status", None)
        metadata = {
            "publication_name": getattr(figure, "name", "") or "",
            "type": getattr(figure, "figure_type", "") or "other",
            "description": getattr(figure, "description", "") or "",
            "interpretation": getattr(figure, "interpretation", "") or "",
            "diagnostic": bool(getattr(figure, "diagnostic", False)),
            "round": int(getattr(figure, "round_number", 0) or 0) or None,
            "review_status": str(getattr(status, "value", status) or ""),
            "plan_step_id": getattr(figure, "plan_step_id", None),
            "plan_id": getattr(figure, "plan_id", None),
            "purpose": getattr(figure, "purpose", "result") or "result",
            "is_publication": True,
        }
        for stem in artifact_stems([figure]):
            indexed[stem] = metadata
    return indexed


def prepare_discovered_figures(
    candidates: Iterable[Dict[str, Any]],
    publication_figures: Iterable[Any] | None,
    *,
    invalidated_stems: Iterable[str] | None = None,
    report_only: bool = True,
    allow_unregistered_diagnostics: bool = False,
) -> List[Dict[str, Any]]:
    """Apply eligibility and authoritative metadata to discovered files.

    A registered but stale/rejected figure can never re-enter through a disk
    scan. Legacy unregistered result files remain compatible with existing
    reports. Unregistered validation diagnostics are admitted only when a caller
    explicitly identifies itself as an annotation-context surface; archival
    reports require such a diagnostic to have an eligible publication record.
    """
    figures = list(publication_figures or [])
    eligible = eligible_artifacts(figures) if report_only else gallery_artifacts(figures)
    kept = (
        [figure for figure in eligible if archival_render_available(figure)]
        if report_only else eligible
    )
    excluded = excluded_artifact_stems(
        figures, kept, invalidated_stems=invalidated_stems,
    )
    metadata_by_stem = publication_figure_metadata(
        figures, report_only=report_only,
    )
    kept_stems = artifact_stems(kept)
    metadata_by_stem = {
        stem: metadata for stem, metadata in metadata_by_stem.items()
        if stem in kept_stems
    }
    records_by_stem: Dict[str, Any] = {}
    for figure in kept:
        for stem in artifact_stems([figure]):
            records_by_stem.setdefault(stem, figure)

    prepared: List[Dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate.get("name") or "")
        if not name or name in excluded:
            continue
        metadata = metadata_by_stem.get(name)
        record = records_by_stem.get(name)
        if (
            report_only
            and record is None
            and image_file_sha256(
                candidate.get("path"),
                min_bytes=1,
                min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
                min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
            ) is None
        ):
            # Filename extensions and PNG/JPEG magic bytes are not physical
            # evidence that an image can be embedded. Unregistered legacy
            # workspace outputs have no record-owned preview to fall back to,
            # so require a complete pixel decode before an archival/context
            # report may admit them.
            continue
        if (
            report_only
            and record is not None
            and not archival_path_matches_record(record, candidate.get("path"))
        ):
            # Same stem is not provenance: require the candidate bytes to match
            # the reviewed record (or, for legacy records, at least remain a
            # sufficiently large decodable render).
            continue
        if report_only and candidate.get("restricted_supplementary") and metadata is None:
            continue
        if (
            candidate.get("diagnostic")
            and metadata is None
            and not allow_unregistered_diagnostics
        ):
            continue
        merged = dict(candidate)
        if metadata is not None:
            merged.update(metadata)
            # Directory provenance remains meaningful even for an older record
            # that predates the diagnostic field.
            merged["diagnostic"] = bool(candidate.get("diagnostic")) or bool(
                metadata.get("diagnostic")
            )
        else:
            merged.setdefault("description", "")
            merged.setdefault("interpretation", "")
            merged.setdefault("is_publication", False)
        prepared.append(merged)
    return prepared
