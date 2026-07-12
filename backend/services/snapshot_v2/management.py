"""Store-level status computation from committed manifests."""

from pydantic import BaseModel

from backend.services.snapshot_v2.object_store import ManifestStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths


class SnapshotStoreStatus(BaseModel):
    logical_bytes: int = 0
    physical_bytes: int = 0
    dedup_savings: int = 0
    object_count: int = 0
    manifest_count: int = 0


def compute_store_status(
    paths: SnapshotV2Paths,
    manifest_store: ManifestStore,
) -> SnapshotStoreStatus:
    """Compute store stats for UI listing without fail-closed verification.

    Uses diagnostics enumeration so a single corrupt committed JSON does not
    turn the whole list route into 422. Safety paths (GC/delete/export) keep
    calling ``verify_all_committed`` separately.
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

    return SnapshotStoreStatus(
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
        dedup_savings=dedup_savings,
        object_count=object_count,
        manifest_count=manifest_count,
    )
