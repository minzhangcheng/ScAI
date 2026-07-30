"""Store-level status computation from committed manifests."""

from pydantic import BaseModel

from backend.config import (
    SNAPSHOT_QUOTA_WARNING_RATIO,
    default_config,
)
from backend.services.snapshot_v2.object_store import ManifestStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths


class SnapshotStoreStatus(BaseModel):
    """Store usage as the listing endpoint reports it.

    ``quota_bytes`` and ``quota_warning`` are resolved here rather than in the
    client so the threshold a user is advised at cannot drift from the ceiling
    that will actually refuse the capture — they are the same number, read from
    the same config.
    """
    logical_bytes: int = 0
    physical_bytes: int = 0
    dedup_savings: int = 0
    object_count: int = 0
    manifest_count: int = 0
    #: Per-session ceiling this store is measured against, in bytes.
    quota_bytes: int = 0
    #: ``physical_bytes / quota_bytes``, clamped at 0 when no quota is known.
    #: Not capped at 1.0: a store may legitimately sit above the ceiling after
    #: an operator lowers the quota, or via the pre_restore grace allowance, and
    #: reporting 1.0 there would hide how far over it is.
    quota_usage_ratio: float = 0.0
    #: True once usage reaches the advisory fraction of the quota. The physical
    #: figure is the one compared, because that is what `_check_quota` enforces
    #: — logical bytes count every reference and would warn far too early on a
    #: lineage that dedups well.
    quota_warning: bool = False


def compute_store_status(
    paths: SnapshotV2Paths,
    manifest_store: ManifestStore,
    *,
    quota_bytes: int | None = None,
) -> SnapshotStoreStatus:
    """Compute store stats for UI listing without fail-closed verification.

    Uses diagnostics enumeration so a single corrupt committed JSON does not
    turn the whole list route into 422. Safety paths (GC/delete/export) keep
    calling ``verify_all_committed`` separately.

    *quota_bytes* defaults to the configured per-session quota; callers that own
    a resolved ceiling (``SnapshotService``, which accepts a per-instance
    override) pass theirs so the reported figure matches the one that would
    refuse their next capture.
    """
    entries = manifest_store.list_committed_with_diagnostics()
    manifests = [e.manifest for e in entries if e.status == "ok" and e.manifest is not None]
    # Count every committed file (ok + corrupt) so UI capacity matches list size.
    manifest_count = len(entries)

    seen_objects: dict[str, int] = {}
    logical_bytes = 0

    for manifest in manifests:
        for obj_hash, obj_ref in manifest.objects.items():
            logical_bytes += obj_ref.size
            if obj_hash not in seen_objects:
                seen_objects[obj_hash] = obj_ref.size

    object_count = len(seen_objects)
    physical_bytes = 0
    for obj_hash, obj_size in seen_objects.items():
        obj_path = paths.objects / obj_hash
        if obj_path.exists():
            physical_bytes += obj_path.stat().st_size
        else:
            physical_bytes += obj_size

    dedup_savings = logical_bytes - physical_bytes

    effective_quota = (
        quota_bytes
        if quota_bytes is not None
        else default_config.snapshot_v2_session_quota_bytes
    )
    if effective_quota > 0:
        usage_ratio = physical_bytes / effective_quota
        quota_warning = usage_ratio >= SNAPSHOT_QUOTA_WARNING_RATIO
    else:
        # A non-positive quota is not a 100%-full store; it means no ceiling is
        # known here, and warning on it would be advice the user cannot act on.
        usage_ratio = 0.0
        quota_warning = False

    return SnapshotStoreStatus(
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
        dedup_savings=dedup_savings,
        object_count=object_count,
        manifest_count=manifest_count,
        quota_bytes=max(effective_quota, 0),
        quota_usage_ratio=usage_ratio,
        quota_warning=quota_warning,
    )
