"""V2 snapshot import with verified extraction, rebinding, and trust downgrade.

Produces committed manifests and content-addressed objects in the destination
session's V2 store, with all IDs rebinding through a deterministic mapping
and a fresh HMAC computed for the destination session.

Object-store mutation is gated by ``build_import_plan`` → ``SanitizedImportPlan``:
only ``plan.object_hashes`` may reach ``put_bytes``.
"""

import hashlib
import json
import logging
import os
import re
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from backend.services.snapshot_v2.errors import (
    SnapshotIntegrityError,
    SnapshotStorageError,
    SnapshotV2Error,
)
from backend.services.snapshot_v2.models import (
    RuntimeManifest,
    SanitizedImportPlan,
    SnapshotManifestV2,
)
from backend.services.snapshot_v2.object_store import (
    ManifestStore,
    ObjectStore,
    canonical_json,
)
from backend.services.snapshot_v2.paths import SnapshotV2Paths

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rebind_id(source_id: str | None, dest_session_id: str) -> str | None:
    """Deterministic one-way ID mapping for snapshot/lineage IDs.

    The mapping is ``sha256(source_id | dest_session)[:16]`` so that
    the same source ID always maps to the same destination ID for a
    given target session, but different target sessions produce
    different results.
    """
    if source_id is None:
        return None
    material = f"{source_id}|{dest_session_id}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write a JSON dict atomically (tmp + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False))
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _collect_runtime_hashes(manifest: SnapshotManifestV2) -> set[str]:
    """Hashes referenced as runtime payload refs (adata / residual)."""
    rt = manifest.runtime
    hashes: set[str] = set()
    for ref in (rt.adata_refs or {}).values():
        hashes.add(ref.sha256)
    if rt.residual_ref is not None:
        hashes.add(rt.residual_ref.sha256)
    return hashes


def _nullify_runtime_inplace(manifest: SnapshotManifestV2) -> set[str]:
    """Clear runtime payload refs; return orphaned runtime hashes removed from objects."""
    rt = manifest.runtime
    orphan_hashes = _collect_runtime_hashes(manifest)
    for h in orphan_hashes:
        manifest.objects.pop(h, None)
    manifest.runtime = RuntimeManifest(
        status="missing",
        dependencies=rt.dependencies,
        residual_ref=None,
        captured_residual_names=[],
        adata_refs={},
    )
    return orphan_hashes


def _rebind_manifest_inplace(
    manifest: SnapshotManifestV2,
    dest_session_id: str,
) -> None:
    """Rebind session, snapshot, and lineage IDs to the destination."""
    if manifest.session_id == dest_session_id:
        return

    source_branch_label = manifest.branch_label
    source_branch_id = manifest.branch_id

    manifest.session_id = dest_session_id
    manifest.snapshot_id = _rebind_id(manifest.snapshot_id, dest_session_id)  # type: ignore[assignment]
    if manifest.parent_id is not None:
        manifest.parent_id = _rebind_id(manifest.parent_id, dest_session_id)
    if source_branch_id and source_branch_id != "main":
        manifest.branch_id = _rebind_id(source_branch_id, dest_session_id)  # type: ignore[assignment]
    if source_branch_label and source_branch_label != "main":
        # Rebind non-default labels so source-session lineage identity does not leak
        if source_branch_label == source_branch_id:
            manifest.branch_label = manifest.branch_id
        else:
            rebound = _rebind_id(source_branch_label, dest_session_id)
            if rebound is not None:
                manifest.branch_label = rebound
    if manifest.source_restore_transaction_id is not None:
        manifest.source_restore_transaction_id = _rebind_id(
            manifest.source_restore_transaction_id, dest_session_id,
        )


def build_import_plan(
    manifests: list[SnapshotManifestV2],
    archive_inventory: dict[str, bytes] | set[str],
    include_runtime: bool,
    trust_runtime: bool,
    *,
    dest_session_id: str,
) -> SanitizedImportPlan:
    """Build a sanitized write plan before any object-store mutation.

    Filters runtime hashes when ``trust_runtime`` is false (or runtime was not
    included in the archive). Rebinds all lineage IDs for the destination session.
    ``SanitizedImportPlan.object_hashes`` is the only set allowed to reach ``put_bytes``.
    """
    if isinstance(archive_inventory, dict):
        inventory_hashes = set(archive_inventory.keys())
    else:
        inventory_hashes = set(archive_inventory)

    omitted_runtime_src_ids: set[str] = set()
    skipped_runtime_hashes: set[str] = set()
    planned: list[SnapshotManifestV2] = []

    for original in manifests:
        manifest = original.model_copy(deep=True)
        runtime_hashes = _collect_runtime_hashes(manifest)
        has_runtime = (
            bool(runtime_hashes)
            or bool(manifest.runtime.captured_residual_names)
            or bool(manifest.runtime.dependencies.required)
            or manifest.runtime.residual_ref is not None
            or bool(manifest.runtime.adata_refs)
        )
        if has_runtime and not (include_runtime and trust_runtime):
            omitted_runtime_src_ids.add(original.snapshot_id)
            skipped = _nullify_runtime_inplace(manifest)
            skipped_runtime_hashes.update(skipped)

        _rebind_manifest_inplace(manifest, dest_session_id)
        planned.append(manifest)

    # Surviving ref closure: only hashes still referenced by planned manifests
    surviving: set[str] = set()
    for manifest in planned:
        surviving.update(manifest.objects.keys())
        for ref in (
            manifest.analysis_state_ref,
            manifest.notebook_ref,
            manifest.checkpoint_ref,
        ):
            if ref is not None:
                surviving.add(ref.sha256)
        if manifest.runtime.residual_ref is not None:
            surviving.add(manifest.runtime.residual_ref.sha256)
        for ref in (manifest.runtime.adata_refs or {}).values():
            surviving.add(ref.sha256)
        for entry in manifest.workspace or []:
            surviving.add(entry.object_hash)
        cont = manifest.continuation
        if hasattr(cont, "workflow_state_ref") and cont.workflow_state_ref is not None:
            surviving.add(cont.workflow_state_ref.sha256)
        if hasattr(cont, "payload_ref") and cont.payload_ref is not None:
            surviving.add(cont.payload_ref.sha256)

    # Only hashes present in the archive inventory and not skipped runtime
    object_hashes = (surviving & inventory_hashes) - skipped_runtime_hashes

    return SanitizedImportPlan(
        object_hashes=object_hashes,
        manifests=planned,
        omitted_runtime_src_ids=omitted_runtime_src_ids,
        skipped_runtime_hashes=skipped_runtime_hashes,
    )


class ImportResult(BaseModel):
    """Result of a V2 snapshot import operation."""

    success: bool = False
    dest_session_id: str = ""
    imported_snapshot_ids: list[str] = Field(default_factory=list)
    committed_count: int = 0
    skipped_runtime_snapshot_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def merge_error(self, code: str, message: str) -> None:
        self.errors.append(f"{code}: {message}")


def _write_import_journal(
    paths,
    *,
    transaction_id: str,
    session_id: str,
    status: str,
    payload: dict | None = None,
) -> None:
    from backend.services.snapshot_v2.journal_store import JournalStore, make_envelope

    env = make_envelope(
        kind="import",
        transaction_id=transaction_id,
        session_id=session_id,
        status=status,
        payload=payload or {},
    )
    JournalStore(paths).write(env)


def _remove_import_journal(paths, transaction_id: str) -> None:
    from backend.services.snapshot_v2.journal_store import JournalStore

    JournalStore(paths).remove(transaction_id)


def load_active_import_protected_ids(paths, session_id: str | None = None) -> set[str]:
    """Return snapshot IDs protected by any active (non-terminal) import journal."""
    protected: set[str] = set()
    journals_dir = Path(str(paths.journals))
    if not journals_dir.exists():
        return protected
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.recovery import RecoveryFailure

    store = JournalStore(paths)
    for entry in journals_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        if not entry.name.startswith("import-"):
            continue
        try:
            env = store.read(entry.stem)
        except (SnapshotIntegrityError, RecoveryFailure, Exception):
            continue
        if env.kind != "import":
            continue
        if session_id and env.session_id not in (None, session_id):
            continue
        status = str(env.status or "").upper()
        if status in ("COMMITTED", "FAILED"):
            continue
        payload = env.payload or {}
        for sid in payload.get("protected_snapshot_ids") or []:
            if sid:
                protected.add(str(sid))
        for sid in payload.get("committed_snapshot_ids") or []:
            if sid:
                protected.add(str(sid))
    return protected


class SnapshotImporter:
    """Import a V2 export tar.gz archive into a destination session store.

    Usage::

        importer = SnapshotImporter(dest_session_id, paths)
        result = importer.import_bundle("/path/to/export.tar.gz", trust_runtime=False)
    """

    def __init__(
        self,
        dest_session_id: str,
        paths: SnapshotV2Paths,
        object_store: Optional[ObjectStore] = None,
        manifest_store: Optional[ManifestStore] = None,
    ) -> None:
        self.dest_session_id = dest_session_id
        self.paths = paths
        self.object_store = object_store or ObjectStore(paths)
        self.manifest_store = manifest_store or ManifestStore(paths)

    def import_bundle(
        self,
        archive_path: str | Path,
        *,
        trust_runtime: bool = False,
    ) -> ImportResult:
        """Import a V2 export archive.

        Args:
            archive_path: Path to the ``.tar.gz`` archive produced by
                ``SnapshotExporter.export()``.
            trust_runtime: When ``True``, restore runtime objects (dill
                pickles, adata h5ad).  When ``False`` (default), mark
                runtime as unavailable and skip writing dill objects into
                the destination store.

        Returns:
            An ``ImportResult`` with status, committed count, errors/warnings.
        """
        archive_path = Path(archive_path)
        result = ImportResult(dest_session_id=self.dest_session_id)

        txn_id = f"import-{uuid.uuid4().hex[:12]}"

        try:
            # Write STAGING journal before any mutation
            _write_import_journal(
                self.paths,
                transaction_id=txn_id,
                session_id=self.dest_session_id,
                status="STAGING",
                payload={"created_at": _now_iso()},
            )

            with tarfile.open(str(archive_path), "r:gz") as tf:
                members = tf.getmembers()

                # Step 1: Verify archive safety (no symlinks, devices)
                self._verify_archive_safety(members)

                # Step 2: Read metadata inventory
                meta = self._read_metadata(tf, members)
                runtime_included = meta.get("runtime_included", True)

                # Step 3: Read objects from archive, verifying hashes
                archive_objects = self._read_archive_objects(tf, members)

                # Step 4: Read and validate manifests from archive
                manifests = self._read_manifests(tf, members)

                # Step 5: Verify all manifest-referenced objects exist in archive
                # (before trust filtering — archive must be self-consistent)
                for manifest in manifests:
                    self._verify_manifest_objects(manifest, archive_objects)

                # Step 6: Build sanitized plan BEFORE any put_bytes
                # Filters untrusted runtime hashes; rebinds lineage IDs.
                plan = build_import_plan(
                    manifests,
                    archive_objects,
                    include_runtime=bool(runtime_included),
                    trust_runtime=trust_runtime,
                    dest_session_id=self.dest_session_id,
                )

                # Step 7: Write ONLY plan.object_hashes to destination store
                for h in plan.object_hashes:
                    data = archive_objects.get(h)
                    if data is None:
                        raise SnapshotIntegrityError(
                            "MISSING_OBJECT",
                            f"Planned object {h} missing from archive inventory",
                        )
                    self.object_store.put_bytes(data, "application/octet-stream")

                # Update journal to STAGED (approved objects written)
                _write_import_journal(
                    self.paths,
                    transaction_id=txn_id,
                    session_id=self.dest_session_id,
                    status="STAGED",
                    payload={
                        "created_at": _now_iso(),
                        "object_hashes": sorted(plan.object_hashes),
                        "protected_snapshot_ids": [
                            m.snapshot_id for m in plan.manifests
                        ],
                    },
                )

                # Step 8: Stage, verify, and publish each rebound manifest
                # (re-signs with host integrity secret via ManifestStore)
                for manifest in plan.manifests:
                    staging_path = self.manifest_store.write_staging(
                        manifest, prefix=f"{txn_id}_",
                    )
                    self._verify_destination_objects(manifest)
                    self.manifest_store.publish_staging(staging_path)

                    result.imported_snapshot_ids.append(manifest.snapshot_id)
                    result.committed_count += 1

            # Step 9: Reconcile staging directory (fail-closed on corrupt)
            reconcile_result = self.manifest_store.reconcile()
            if reconcile_result.corrupt_ids:
                logger.error(
                    "Import [%s]: reconcile found corrupt manifests: %s",
                    txn_id, reconcile_result.corrupt_ids,
                )
                raise SnapshotIntegrityError(
                    "IMPORT_RECONCILE_CORRUPT",
                    f"Post-import reconcile found corrupt manifests: "
                    f"{reconcile_result.corrupt_ids}",
                )

            # Step 10: Run safe GC
            self._run_safe_gc()

            # Write COMMITTED journal as durable proof; removed below after
            # full reconciliation succeeds.
            _write_import_journal(
                self.paths,
                transaction_id=txn_id,
                session_id=self.dest_session_id,
                status="COMMITTED",
                payload={
                    "created_at": _now_iso(),
                    "committed_snapshot_ids": result.imported_snapshot_ids,
                },
            )

            # Remove COMMITTED journal — import fully succeeded.
            # Failed journals remain visible; STAGING/STAGED recovery still
            # works for crashes mid-import.
            _remove_import_journal(self.paths, txn_id)

            result.success = True
            result.skipped_runtime_snapshot_ids = [
                _rebind_id(sid, self.dest_session_id)  # type: ignore[misc]
                for sid in sorted(plan.omitted_runtime_src_ids)
            ]

        except SnapshotV2Error as exc:
            result.merge_error(exc.code, str(exc))
            logger.error("Import [%s] failed: %s — %s", txn_id, exc.code, exc)
            try:
                _write_import_journal(
                    self.paths,
                    transaction_id=txn_id,
                    session_id=self.dest_session_id,
                    status="FAILED",
                    payload={"created_at": _now_iso(), "error": str(exc)},
                )
            except Exception:
                pass
        except Exception as exc:
            msg = f"Unexpected import error: {exc}"
            result.merge_error("UNEXPECTED", msg)
            logger.exception("Import [%s] failed unexpectedly", txn_id)
            try:
                _write_import_journal(
                    self.paths,
                    transaction_id=txn_id,
                    session_id=self.dest_session_id,
                    status="FAILED",
                    payload={"created_at": _now_iso(), "error": msg},
                )
            except Exception:
                pass

        return result

    # ── Internal helpers ────────────────────────────────────────────────

    def _verify_archive_safety(self, members: list) -> None:
        """Reject archives with symlinks, hardlinks, device files, or FIFOs."""
        bad: list[str] = []
        for member in members:
            if member.issym():
                bad.append(f"symlink: {member.name}")
            if member.islnk():
                bad.append(f"hardlink: {member.name}")
            if member.isdev() or member.ischr() or member.isfifo():
                bad.append(f"special: {member.name} ({member.type})")
        if bad:
            raise SnapshotStorageError(
                "ARCHIVE_UNSAFE",
                f"Archive contains unsafe entries: {'; '.join(bad)}",
            )

    def _read_metadata(self, tf: tarfile.TarFile, members: list) -> dict:
        """Read and return the snapshot-v2-export.json metadata."""
        for m in members:
            if m.name == "snapshot-v2-export.json":
                f = tf.extractfile(m)
                if f is None:
                    break
                try:
                    return json.loads(f.read().decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise SnapshotIntegrityError(
                        "METADATA_PARSE_ERROR",
                        f"Failed to parse metadata: {exc}",
                    )
        raise SnapshotIntegrityError(
            "MISSING_METADATA",
            "snapshot-v2-export.json not found in archive",
        )

    def _read_manifests(
        self, tf: tarfile.TarFile, members: list,
    ) -> list[SnapshotManifestV2]:
        """Read and validate ``manifests/<snapshot_id>.json`` entries."""
        manifests: list[SnapshotManifestV2] = []
        seen_ids: set[str] = set()

        for m in members:
            if not m.isfile():
                continue
            if not m.name.startswith("manifests/") or not m.name.endswith(".json"):
                continue
            # Reject path traversal
            if ".." in m.name or m.name.startswith("/"):
                raise SnapshotStorageError(
                    "SUSPICIOUS_PATH",
                    f"Manifest path contains traversal: {m.name}",
                )

            f = tf.extractfile(m)
            if f is None:
                continue

            manifest_bytes = f.read()
            try:
                manifest = SnapshotManifestV2.model_validate_json(manifest_bytes)
            except Exception as exc:
                raise SnapshotIntegrityError(
                    "MANIFEST_PARSE_ERROR",
                    f"Failed to parse manifest {m.name}: {exc}",
                )

            if manifest.snapshot_id in seen_ids:
                raise SnapshotStorageError(
                    "DUPLICATE_MANIFEST_ID",
                    f"Duplicate snapshot ID in archive: {manifest.snapshot_id}",
                )
            seen_ids.add(manifest.snapshot_id)

            # Verify root_hash is correct
            raw = canonical_json(manifest)
            expected_root = hashlib.sha256(raw).hexdigest()
            if manifest.root_hash != expected_root:
                raise SnapshotIntegrityError(
                    "MANIFEST_ROOT_HASH_MISMATCH",
                    f"Root hash mismatch for {manifest.snapshot_id}: "
                    f"expected {expected_root}, got {manifest.root_hash}",
                )

            manifests.append(manifest)

        if not manifests:
            raise SnapshotIntegrityError(
                "NO_MANIFESTS",
                "No valid manifests found in archive",
            )

        return manifests

    def _read_archive_objects(
        self, tf: tarfile.TarFile, members: list,
    ) -> dict[str, bytes]:
        """Read ``objects/<sha256>`` entries, verifying content hash."""
        objects: dict[str, bytes] = {}
        for m in members:
            if not m.isfile():
                continue
            if not m.name.startswith("objects/") or m.name.count("/") != 1:
                continue

            obj_hash = m.name.split("/")[1]
            if not _HASH_RE.match(obj_hash):
                raise SnapshotStorageError(
                    "INVALID_OBJECT_HASH",
                    f"Invalid object hash filename: {m.name}",
                )

            f = tf.extractfile(m)
            if f is None:
                continue

            data = f.read()
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != obj_hash:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Object {obj_hash} content hash mismatch "
                    f"(actual: {actual_hash})",
                )

            if obj_hash not in objects:
                objects[obj_hash] = data

        return objects

    def _verify_manifest_objects(
        self,
        manifest: SnapshotManifestV2,
        archive_objects: dict[str, bytes],
    ) -> None:
        """Every object referenced by *manifest* must exist in the archive."""
        for h, ref in manifest.objects.items():
            data = archive_objects.get(h)
            if data is None:
                raise SnapshotIntegrityError(
                    "MISSING_OBJECT",
                    f"Object {h} referenced by manifest "
                    f"{manifest.snapshot_id} not found in archive",
                )
            if len(data) != ref.size:
                raise SnapshotIntegrityError(
                    "OBJECT_SIZE_MISMATCH",
                    f"Object {h} size {len(data)} != expected {ref.size}",
                )

    def _rebind_manifest(self, manifest: SnapshotManifestV2) -> None:
        """Rebind session, snapshot, and lineage IDs to the destination."""
        _rebind_manifest_inplace(manifest, self.dest_session_id)

    def _nullify_runtime(self, manifest: SnapshotManifestV2) -> None:
        """Mark runtime objects as unavailable by clearing their refs."""
        _nullify_runtime_inplace(manifest)

    def _verify_destination_objects(
        self, manifest: SnapshotManifestV2,
    ) -> None:
        """Every object referenced by *manifest* must exist in dest store."""
        for h, ref in manifest.objects.items():
            try:
                self.object_store.verify(ref)
            except SnapshotV2Error as exc:
                raise SnapshotIntegrityError(
                    "DESTINATION_OBJECT_MISSING",
                    f"Object {h} missing in destination store: {exc}",
                )

    def _run_safe_gc(self) -> None:
        """Run garbage collection to clean up orphaned objects."""
        try:
            from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

            gc = SnapshotGarbageCollector(
                self.dest_session_id,
                self.paths,
                manifest_store=self.manifest_store,
                object_store=self.object_store,
            )
            gcr = gc.run()
            if gcr.removed_count > 0:
                logger.info(
                    "Import GC: removed %d objects (%d bytes)",
                    gcr.removed_count, gcr.removed_bytes,
                )
        except Exception as exc:
            logger.warning("Import GC failed, continuing: %s", exc)
