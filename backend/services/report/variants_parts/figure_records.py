"""Record-owned figure recovery shared by static report variants."""

from pathlib import Path
from typing import Any, Iterable, Optional

from backend.services.report.artifacts import (
    eligible_artifacts,
    gallery_artifacts,
    resolve_archival_render,
)


def resolved_record_figure_candidates(
    publication_figures: Iterable[Any] | None,
    workspace_path: Any,
    *,
    report_only: bool = True,
    round_numbers: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Return normalized candidates resolved through artifact-owned bytes.

    Disk discovery starts from paths, so it cannot recover a reviewed record
    whose original path moved or was overwritten. Static report variants use
    this record-first tier before their legacy workspace scan.
    """
    records = (
        eligible_artifacts(publication_figures)
        if report_only
        else gallery_artifacts(publication_figures)
    )
    candidates: list[dict[str, Any]] = []
    seen_stems: set[str] = set()
    for record in records:
        record_round = int(getattr(record, "round_number", 0) or 0)
        if round_numbers is not None and record_round not in round_numbers:
            continue
        resolved = resolve_archival_render(record, workspace_path)
        if resolved is None:
            continue
        declared = Path(str(getattr(record, "png_path", "") or ""))
        canonical_stem = declared.stem or str(getattr(record, "name", "") or "")
        canonical_stem = canonical_stem or resolved.stem
        if canonical_stem in seen_stems:
            continue
        seen_stems.add(canonical_stem)
        try:
            modified_at = resolved.stat().st_mtime
        except OSError:
            modified_at = 0.0
        candidates.append({
            # ``name`` follows discovery semantics and identifies the actual
            # file, while ``record_stem`` keeps round/step attribution stable
            # when recovery had to add a ``_record_preview`` suffix.
            "name": resolved.stem,
            "record_stem": canonical_stem,
            "publication_name": getattr(record, "name", "") or "",
            "type": getattr(record, "figure_type", "") or "other",
            "path": str(resolved),
            "pdf_path": None,
            "mtime": modified_at,
            "source": "publication_record",
            "rel_path": resolved.name,
            "diagnostic": bool(getattr(record, "diagnostic", False)),
            "description": getattr(record, "description", "") or "",
            "interpretation": getattr(record, "interpretation", "") or "",
            "round": record_round or None,
            "is_publication": True,
        })
    return candidates
