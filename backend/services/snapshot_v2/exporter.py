"""V2 snapshot export with explicit runtime policy.

Produces a tar.gz archive of verified snapshot objects.  Base/workspace
objects always export; runtime objects only with ``include_runtime=True``.
"""

import hashlib
import io
import json
import logging
import os
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotV2Error
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.models import (
    SnapshotManifestV2,
)

logger = logging.getLogger(__name__)


class ExportIntegrityError(SnapshotV2Error):
    """Raised when an object referenced by a manifest cannot be verified."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_export_id() -> str:
    return f"exp-{uuid.uuid4().hex[:16]}"


# ── Export protection journals ──────────────────────────────────────────


def _write_export_protection(
    paths,
    export_id: str,
    session_id: str,
    protected_ids: list[str],
) -> None:
    """Write a signed export_protection journal so deletion honours the lock."""
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope

    tx_id = f"export-{export_id}"
    env = make_envelope(
        kind="export_protection",
        transaction_id=tx_id,
        session_id=session_id,
        status="ACTIVE",
        payload={
            "export_id": export_id,
            "protected_snapshot_ids": list(protected_ids),
            "created_at": _now_iso(),
        },
    )
    JournalStore(paths).write(env)


def _remove_export_protection(paths, export_id: str) -> None:
    from backend.services.snapshot_v2.journal_store import JournalStore

    JournalStore(paths).remove(f"export-{export_id}")


def load_active_export_protected_ids(paths) -> set[str]:
    """Return set of snapshot IDs protected by any active V2 export."""
    protected: set[str] = set()
    journals_dir = Path(str(paths.journals))
    if not journals_dir.exists():
        return protected
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.errors import SnapshotIntegrityError
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    store = JournalStore(paths)
    for entry in journals_dir.iterdir():
        if not (
            entry.is_file()
            and entry.name.startswith("export-")
            and entry.suffix == ".json"
        ):
            continue
        try:
            env = store.read(entry.stem)
        except (SnapshotIntegrityError, RecoveryFailure, Exception):
            continue
        if env.kind != "export_protection":
            continue
        protected.update((env.payload or {}).get("protected_snapshot_ids") or [])
    return protected


def load_active_export_protected_by_reason(paths) -> set[str]:
    """Alias used by protection aggregation (export reason)."""
    return load_active_export_protected_ids(paths)


# ── Object classification ───────────────────────────────────────────────


def _classify_manifest_and_collect(
    manifest: SnapshotManifestV2,
) -> tuple[set[str], set[str]]:
    """Return (base_hashes, runtime_only_hashes) for one manifest.

    Base hashes are those referenced by workspace entries, analysis state,
    notebook, and checkpoint refs. Runtime-only hashes are those referenced
    exclusively by ``runtime`` (adata refs, residual ref).
    """
    workspace_hashes: set[str] = set()
    for we in manifest.workspace:
        workspace_hashes.add(we.object_hash)
    workspace_hashes.add(manifest.analysis_state_ref.sha256)
    workspace_hashes.add(manifest.notebook_ref.sha256)
    if manifest.checkpoint_ref is not None:
        workspace_hashes.add(manifest.checkpoint_ref.sha256)

    runtime_hashes: set[str] = set()
    rt = manifest.runtime
    for ref in (rt.adata_refs or {}).values():
        runtime_hashes.add(ref.sha256)
    if rt.residual_ref is not None:
        runtime_hashes.add(rt.residual_ref.sha256)

    base_hashes: set[str] = set()
    runtime_only: set[str] = set()
    for h in manifest.objects:
        if h in runtime_hashes and h not in workspace_hashes:
            runtime_only.add(h)
        else:
            base_hashes.add(h)

    return base_hashes, runtime_only


# ── SnapshotExporter ────────────────────────────────────────────────────


class SnapshotExporter:
    """Export verified snapshot V2 objects into a tar.gz archive.

    Usage::

        exporter = SnapshotExporter(session_id, paths)
        path = exporter.export(session_id, include_runtime=False)
    """

    def __init__(self, session_id: str, paths) -> None:
        self.session_id = session_id
        self.paths = paths
        self.object_store = ObjectStore(paths)
        self.manifest_store = ManifestStore(paths)

    def export(
        self,
        session_id: str,
        *,
        include_runtime: bool = True,
        snapshot_ids: list[str] | None = None,
    ) -> Path:
        """Build a tar.gz archive of verified snapshot objects.

        Returns the path to the archive on disk.  The caller is responsible
        for cleaning up the returned file.
        """
        # 1. Freeze the verified manifest set (raises on corrupt)
        manifests = self.manifest_store.verify_all_committed()
        if snapshot_ids is not None:
            snapshot_set = set(snapshot_ids)
            manifests = [m for m in manifests if m.snapshot_id in snapshot_set]

        if not manifests:
            raise ExportIntegrityError("NO_MANIFESTS", "No snapshots to export")

        # 2. Classify objects across all exported manifests
        base_hashes: set[str] = set()
        runtime_only_hashes: set[str] = set()
        all_snapshot_ids: list[str] = []

        for m in manifests:
            all_snapshot_ids.append(m.snapshot_id)
            bh, ro = _classify_manifest_and_collect(m)
            base_hashes.update(bh)
            runtime_only_hashes.update(ro)

        # 3. Decide final inclusion sets
        included_hashes: set[str] = set(base_hashes)
        omitted_runtime_hashes: list[str] = sorted(runtime_only_hashes)

        if include_runtime:
            included_hashes.update(runtime_only_hashes)

        # Hashes that are in runtime_only but NOT included
        omitted_runtime = sorted(runtime_only_hashes) if not include_runtime else []

        # 4. Write protection record
        export_id = _new_export_id()
        _write_export_protection(self.paths, export_id, session_id, all_snapshot_ids)

        try:
            # 5. Verify all included objects
            for m in manifests:
                all_included = {h for h in m.objects if h in included_hashes}
                for h in all_included:
                    ref = m.objects[h]
                    try:
                        self.object_store.verify(ref)
                    except SnapshotV2Error as exc:
                        raise ExportIntegrityError(
                            exc.code,
                            f"Object {h} verification failed for "
                            f"snapshot {m.snapshot_id}: {exc}",
                        ) from exc

            # 6. Build tar.gz archive
            fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz", prefix=f"v2-export-{export_id}-")
            os.close(fd)

            with tarfile.open(tmp_path, mode="w:gz") as tf:
                # 6a. Add object files (by hash)
                for h in sorted(included_hashes):
                    try:
                        data = self.object_store.get_bytes(h)
                    except SnapshotV2Error as exc:
                        raise ExportIntegrityError(
                            "OBJECT_NOT_FOUND",
                            f"Object {h} not found in store: {exc}",
                        ) from exc

                    info = tarfile.TarInfo(name=f"objects/{h}")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))

                # 6b. Add manifest files
                for manifest in manifests:
                    m_bytes = json.dumps(
                        manifest.model_dump(mode="json"),
                        sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                    m_info = tarfile.TarInfo(name=f"manifests/{manifest.snapshot_id}.json")
                    m_info.size = len(m_bytes)
                    tf.addfile(m_info, io.BytesIO(m_bytes))

                # 6c. Write metadata
                total_size = sum(
                    info.size for info in tf.getmembers()
                    if info.name.startswith("objects/")
                )
                metadata = {
                    "format_version": 2,
                    "session_id": session_id,
                    "export_id": export_id,
                    "exported_at": _now_iso(),
                    "runtime_included": include_runtime,
                    "included_snapshot_ids": all_snapshot_ids,
                    "included_hashes": sorted(included_hashes),
                    "omitted_runtime_hashes": omitted_runtime,
                    "object_count": len(included_hashes),
                    "manifest_count": len(manifests),
                    "total_size_bytes": total_size,
                }
                meta_bytes = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode()
                meta_info = tarfile.TarInfo(name="snapshot-v2-export.json")
                meta_info.size = len(meta_bytes)
                tf.addfile(meta_info, io.BytesIO(meta_bytes))

            return Path(tmp_path)

        finally:
            _remove_export_protection(self.paths, export_id)
