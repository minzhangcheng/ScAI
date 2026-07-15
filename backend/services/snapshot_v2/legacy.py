"""V1 snapshot compatibility: inventory / export / delete only; restore cut over."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from backend.constants import VERSION
from backend.models.schemas import AnalysisState, Snapshot


class LegacyInventory(BaseModel):
    """Read-only inventory of non-restorable V1 snapshots."""

    snapshots: List[dict] = Field(default_factory=list)
    snapshot_count: int = 0
    warning_version: str = VERSION
    restorable: bool = False
    message: str = ""


def should_reject_v1_restore(
    cutover: bool = True,
    deprecation_version: Optional[str] = None,
) -> Optional[str]:
    """Return a rejection message for V1 restore, or None if restore were allowed.

    After the Batch 4 cutover, V1 restore is always rejected regardless of
    ``cutover`` (the parameter is retained for call-site compatibility only).
    """
    ver = deprecation_version or VERSION
    # Unconditional cutover: no config combination re-enables V1 restore.
    _ = cutover
    return (
        f"V1 snapshot restore is no longer supported (cutover since {ver}). "
        f"Legacy snapshots remain available for metadata, export, and delete only. "
        f"Use Snapshot V2 restore endpoints for restorable snapshots."
    )


def normalize_snapshot_format_version(value: Any) -> Optional[int]:
    """Normalize historical integer/string format-version representations."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_v1_snapshot(snap: Snapshot) -> bool:
    fv = normalize_snapshot_format_version(snap.format_version)
    return fv is None or fv < 2


def build_legacy_inventory(
    state: AnalysisState,
    warning_version: Optional[str] = None,
) -> LegacyInventory:
    """Build a non-restorable inventory of V1 snapshots from AnalysisState."""
    ver = warning_version or VERSION
    items: List[dict] = []
    for snap in state.snapshots or []:
        if not _is_v1_snapshot(snap):
            continue
        payload = snap.model_dump(mode="json")
        payload["restorable"] = False
        payload["unsupported"] = True
        payload["format_version"] = snap.format_version if snap.format_version is not None else 1
        items.append(payload)
    return LegacyInventory(
        snapshots=items,
        snapshot_count=len(items),
        warning_version=ver,
        restorable=False,
        message=(
            f"Legacy V1 snapshots are unsupported for restore since {ver}. "
            f"Metadata, export, and delete remain available."
        ),
    )


def filter_v1_from_v2_list(records: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Exclude V1 / unknown format records from a V2 list projection."""
    out: List[dict] = []
    for rec in records:
        manifest = rec.get("manifest") if isinstance(rec, Mapping) else None
        if not isinstance(manifest, Mapping):
            continue
        fv = manifest.get("format_version")
        if fv == 2 or fv == "2":
            out.append(dict(rec))
    return out


def v1_restore_gone_detail(version: Optional[str] = None) -> str:
    """Canonical HTTP 410 detail string for the V1 restore endpoint."""
    return should_reject_v1_restore(deprecation_version=version or VERSION) or (
        f"V1 snapshot restore is no longer supported since {version or VERSION}."
    )
