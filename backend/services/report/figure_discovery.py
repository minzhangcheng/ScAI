"""Discover candidate workspace figures for the brief Results gallery.

This module only enumerates and normalizes candidate images. It does not decide
which figures are shown; PI curation happens in
``PIAgent.select_brief_figures``. Keep it generic: no dataset-, disease-,
tissue-, species-, or project-specific file names.
"""
import logging
import uuid
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
    pdf_file_sha256,
    resolve_archival_pdf,
    resolve_archival_render,
    resolve_legacy_artifact_pdf,
    resolve_legacy_artifact_render,
    MIN_ARCHIVAL_PREVIEW_HEIGHT,
    MIN_ARCHIVAL_PREVIEW_WIDTH,
    ARTIFACT_REGISTRY_VERSION,
)
from backend.models.schemas import LegacyArtifactRecord
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
_SKIP_DIRS = frozenset({
    "__pycache__",
    ".ipynb_checkpoints",
    ".attempts",
    ".revisions",
    "reports",
    "tables",
})

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

    Registered metadata outranks filename inference. Archival and browsing
    callers now share the same publication predicate and published-head
    resolver; ``report_only`` remains for API compatibility.
    """
    figures = list(publication_figures or [])
    kept = eligible_artifacts(figures) if report_only else gallery_artifacts(figures)
    indexed: Dict[str, Dict[str, Any]] = {}
    for figure in kept:
        status = getattr(figure, "review_status", None)
        metadata = {
            "id": getattr(figure, "id", "") or "",
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
            "logical_figure_id": getattr(figure, "logical_figure_id", None),
            "render_attempt_id": getattr(figure, "render_attempt_id", None),
            "producer_status": getattr(figure, "producer_status", "published"),
            "revision_of_id": getattr(figure, "revision_of_id", None),
            "supersedes_id": getattr(figure, "supersedes_id", None),
            "is_published_head": bool(
                getattr(figure, "is_published_head", False)
            ),
            "is_publication": True,
            "render_sha256": getattr(figure, "render_sha256", None),
            "preview_sha256": getattr(figure, "preview_sha256", None),
            "pdf_sha256": getattr(figure, "pdf_sha256", None),
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
    registry_version: int | None = None,
    legacy_artifacts: Iterable[Any] | None = None,
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

    legacy_records = list(legacy_artifacts or [])
    strict_registry = bool(
        registry_version is not None
        and int(registry_version) >= ARTIFACT_REGISTRY_VERSION
    )
    prepared: List[Dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate.get("name") or "")
        if not name or name in excluded:
            continue
        metadata = metadata_by_stem.get(name)
        record = records_by_stem.get(name)
        candidate_digest = None
        if record is None and (report_only or strict_registry or legacy_records):
            candidate_digest = image_file_sha256(
                candidate.get("path"),
                min_bytes=1,
                min_width=MIN_ARCHIVAL_PREVIEW_WIDTH,
                min_height=MIN_ARCHIVAL_PREVIEW_HEIGHT,
            )
        if record is None and (report_only or strict_registry) and candidate_digest is None:
            # Filename extensions and PNG/JPEG magic bytes are not physical
            # evidence that an image can be embedded. Unregistered legacy
            # workspace outputs have no record-owned preview to fall back to,
            # so require a complete pixel decode before an archival/context
            # report may admit them.
            continue
        legacy_record = None
        if record is None and candidate_digest is not None:
            relative_path = str(candidate.get("rel_path") or "")
            for item in legacy_records:
                if (
                    str(getattr(item, "render_sha256", "") or "") == candidate_digest
                    and str(getattr(item, "relative_path", "") or "")
                    == relative_path
                ):
                    legacy_record = item
                    break
            if legacy_record is None and strict_registry:
                # New sessions are record-only on every report/gallery surface.
                continue
        if (
            record is not None
            and (report_only or strict_registry)
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
        if legacy_record is not None:
            # The image identity and its optional PDF are one frozen legacy
            # record. A same-stem PDF that appeared or changed after migration
            # must not borrow the retained JPG identity on gallery, report, or
            # download surfaces.
            frozen_pdf_path = None
            frozen_pdf_digest = str(
                getattr(legacy_record, "pdf_sha256", "") or ""
            ).strip().lower()
            frozen_pdf_relative = str(
                getattr(legacy_record, "pdf_relative_path", "") or ""
            ).strip()
            candidate_pdf_path = candidate.get("pdf_path")
            candidate_pdf_relative = (
                Path(str(candidate.get("rel_path") or ""))
                .with_suffix(".pdf")
                .as_posix()
                if candidate_pdf_path else ""
            )
            if (
                frozen_pdf_digest
                and frozen_pdf_relative
                and candidate_pdf_relative == frozen_pdf_relative
                and pdf_file_sha256(candidate_pdf_path) == frozen_pdf_digest
            ):
                frozen_pdf_path = str(candidate_pdf_path)
            merged["pdf_path"] = frozen_pdf_path
            merged["pdf_sha256"] = (
                frozen_pdf_digest if frozen_pdf_path else None
            )
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
            if report_only or strict_registry:
                relative_path = str(candidate.get("rel_path") or "")
                legacy_id = (
                    str(getattr(legacy_record, "id", "") or "")
                    if legacy_record is not None
                    else str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"scai-legacy:{relative_path}:{candidate_digest}",
                    ))
                )
                merged.update({
                    "id": legacy_id,
                    "is_legacy": True,
                    "legacy_artifact_id": legacy_id,
                    "provenance_kind": "legacy_unregistered",
                    "render_sha256": candidate_digest,
                })
        prepared.append(merged)
    return prepared


def ensure_legacy_artifact_inventory(state: Any, workspace: Path) -> list[Any]:
    """Inventory valid unregistered images for an explicitly legacy session.

    The absence of ``artifact_registry_version`` is the compatibility signal.
    File times are deliberately irrelevant.  The resulting records are labelled
    unreviewed legacy provenance and can never satisfy PublicationFigure review.
    """
    if getattr(state, "artifact_registry_version", None) is not None:
        return list(getattr(state, "legacy_artifacts", None) or [])
    existing = list(getattr(state, "legacy_artifacts", None) or [])
    prepared = prepare_discovered_figures(
        discover_figures_in_workspace(Path(workspace)),
        getattr(state, "publication_figures", None) or [],
        invalidated_stems=getattr(state, "invalidated_artifact_stems", None),
        report_only=True,
        allow_unregistered_diagnostics=False,
        registry_version=None,
        legacy_artifacts=existing,
    )
    known_ids = {str(getattr(item, "id", "") or "") for item in existing}
    for candidate in prepared:
        if candidate.get("is_publication") or not candidate.get("is_legacy"):
            continue
        record_id = str(candidate.get("legacy_artifact_id") or "")
        if not record_id or record_id in known_ids:
            continue
        existing.append(LegacyArtifactRecord(
            id=record_id,
            name=str(candidate.get("name") or "legacy_figure"),
            relative_path=str(candidate.get("rel_path") or ""),
            render_sha256=str(candidate.get("render_sha256") or ""),
            pdf_relative_path=(
                Path(str(candidate.get("rel_path") or "")).with_suffix(".pdf").as_posix()
                if candidate.get("pdf_path") else None
            ),
            pdf_sha256=(
                pdf_file_sha256(candidate.get("pdf_path"))
                if candidate.get("pdf_path") else None
            ),
            source=str(candidate.get("source") or "legacy_workspace"),
            lineage_id="main",
        ))
        known_ids.add(record_id)
    state.legacy_artifacts = existing
    # The first scan is a migration boundary, not a perpetual compatibility
    # mode.  Persisting the registry version freezes this exact inventory so a
    # later file cannot acquire report/download identity merely by appearing on
    # disk.
    state.artifact_registry_version = ARTIFACT_REGISTRY_VERSION
    return existing


def resolve_state_figure_candidates(
    state: Any,
    workspace: Path,
    *,
    report_only: bool,
    lineage_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Resolve every report/gallery candidate through retained state identity."""
    workspace = Path(workspace).resolve()
    ensure_legacy_artifact_inventory(state, workspace)
    target_lineage = str(
        lineage_id or getattr(state, "restore_branch_id", None) or "main"
    )
    records = [
        item for item in (getattr(state, "publication_figures", None) or [])
        if str(getattr(item, "lineage_id", None) or "main") == target_lineage
    ]
    prepared = prepare_discovered_figures(
        discover_figures_in_workspace(workspace),
        records,
        invalidated_stems=getattr(state, "invalidated_artifact_stems", None),
        report_only=report_only,
        allow_unregistered_diagnostics=False,
        registry_version=getattr(state, "artifact_registry_version", None),
        legacy_artifacts=getattr(state, "legacy_artifacts", None),
    )

    represented_ids = {
        str(candidate.get("id") or candidate.get("legacy_artifact_id") or "")
        for candidate in prepared
    }
    for legacy in (getattr(state, "legacy_artifacts", None) or []):
        if str(getattr(legacy, "lineage_id", None) or "main") != target_lineage:
            continue
        legacy_id = str(getattr(legacy, "id", "") or "")
        if not legacy_id or legacy_id in represented_ids:
            continue
        image_path = resolve_legacy_artifact_render(legacy, workspace)
        if image_path is None:
            continue
        pdf_path = resolve_legacy_artifact_pdf(legacy, workspace)
        name = str(getattr(legacy, "name", "") or image_path.stem)
        prepared.append({
            "id": legacy_id,
            "legacy_artifact_id": legacy_id,
            "name": name,
            "type": infer_figure_type(name),
            "path": str(image_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "mtime": image_path.stat().st_mtime,
            "source": str(getattr(legacy, "source", "") or "legacy_inventory"),
            "rel_path": str(getattr(legacy, "relative_path", "") or ""),
            "diagnostic": False,
            "description": "",
            "interpretation": "",
            "is_publication": False,
            "is_legacy": True,
            "provenance_kind": "legacy_unregistered",
            "render_sha256": getattr(legacy, "render_sha256", None),
            "pdf_sha256": getattr(legacy, "pdf_sha256", None),
        })
        represented_ids.add(legacy_id)

    kept = eligible_artifacts(records) if report_only else gallery_artifacts(records)
    for record in kept:
        record_id = str(getattr(record, "id", "") or "")
        if not record_id or record_id in represented_ids:
            continue
        image_path = resolve_archival_render(record, workspace)
        if image_path is None:
            continue
        pdf_path = resolve_archival_pdf(record, workspace)
        status = getattr(record, "review_status", None)
        prepared.append({
            "id": record_id,
            "name": str(getattr(record, "name", "") or image_path.stem),
            "type": str(getattr(record, "figure_type", "") or "other"),
            "path": str(image_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "mtime": image_path.stat().st_mtime,
            "source": "publication_registry",
            "rel_path": (
                image_path.relative_to(workspace).as_posix()
                if image_path.is_relative_to(workspace) else image_path.name
            ),
            "diagnostic": bool(getattr(record, "diagnostic", False)),
            "description": str(getattr(record, "description", "") or ""),
            "interpretation": str(getattr(record, "interpretation", "") or ""),
            "is_publication": True,
            "review_status": str(getattr(status, "value", status) or ""),
            "round": int(getattr(record, "round_number", 0) or 0) or None,
            "purpose": str(getattr(record, "purpose", "result") or "result"),
            "render_sha256": getattr(record, "render_sha256", None),
            "preview_sha256": getattr(record, "preview_sha256", None),
            "pdf_sha256": getattr(record, "pdf_sha256", None),
        })
        represented_ids.add(record_id)
    return prepared
