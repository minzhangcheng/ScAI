"""Resolve reviewed Chat figure attachments through their frozen records."""

from pathlib import Path
from typing import Any

from backend.services.report.artifacts import (
    resolve_verified_archival_artifact,
    verified_artifact_bytes,
)


def publication_figure_attachment(figure: Any, workspace: Path) -> dict[str, Any] | None:
    image_path = figure.png_path
    if verified_artifact_bytes(image_path, figure, kind="image") is None:
        image_path = None
        resolved = resolve_verified_archival_artifact(figure, workspace, kind="image")
        # Materialized inline previews must meet the same delivery pixel limit
        # as existing files, in addition to matching the reviewed record.
        if resolved is not None and verified_artifact_bytes(
            resolved[1], figure, kind="image",
        ) is not None:
            image_path = str(resolved[1])
    pdf = resolve_verified_archival_artifact(figure, workspace, kind="pdf")
    if image_path is None and pdf is None:
        return None
    entry = {
        "name": figure.name,
        "artifact_id": figure.id,
        "figure_role": "publication",
        "review_status": figure.review_status.value,
        "review_unavailable": figure.review_unavailable,
    }
    if image_path is not None:
        entry["path"] = image_path
    if pdf is not None:
        entry["pdfPath"] = str(pdf[1])
    return entry
