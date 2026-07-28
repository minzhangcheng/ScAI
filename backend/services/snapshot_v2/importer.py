"""V2 snapshot import with verified extraction, rebinding, and trust downgrade.

Produces committed manifests and content-addressed objects in the destination
session's V2 store, with all IDs rebinding through a deterministic mapping
and a fresh HMAC computed for the destination session.

Object-store mutation is gated by ``build_import_plan`` → ``SanitizedImportPlan``:
only ``plan.object_hashes`` may reach ``put_stream``.

Two-pass streaming:
1. First pass: hash and record metadata only (ArchiveObjectMeta)
2. Build sanitized plan from metadata inventory
3. Second pass: stream approved members through ObjectStore
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

from backend.services.snapshot_v2.batch_store import ImportBatchRecord, ImportBatchStore
from backend.services.snapshot_v2.errors import (
    SnapshotIntegrityError,
    SnapshotStorageError,
    SnapshotV2Error,
)
from backend.services.snapshot_v2.models import (
    RuntimeManifest,
    SanitizedImportPlan,
    SnapshotManifestV2,
    ObjectRef,
)
from backend.services.snapshot_v2.object_store import (
    ArchiveObjectMeta,
    ManifestStore,
    ObjectStore,
    SnapshotImportLimits,
    canonical_json,
)
from backend.services.snapshot_v2.paths import SnapshotV2Paths

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1 * 1024 * 1024


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rebind_id(source_id: str | None, dest_session_id: str) -> str | None:
    if source_id is None:
        return None
    material = f"{source_id}|{dest_session_id}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _collect_runtime_hashes(manifest: SnapshotManifestV2) -> set[str]:
    rt = manifest.runtime
    hashes: set[str] = set()
    for ref in (rt.adata_refs or {}).values():
        hashes.add(ref.sha256)
    if rt.residual_ref is not None:
        hashes.add(rt.residual_ref.sha256)
    return hashes


def _nullify_runtime_inplace(manifest: SnapshotManifestV2) -> set[str]:
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
    if manifest.session_id == dest_session_id:
        return

    source_branch_label = manifest.branch_label
    source_branch_id = manifest.branch_id

    manifest.session_id = dest_session_id
    manifest.snapshot_id = _rebind_id(manifest.snapshot_id, dest_session_id)
    if manifest.parent_id is not None:
        manifest.parent_id = _rebind_id(manifest.parent_id, dest_session_id)
    if source_branch_id and source_branch_id != "main":
        manifest.branch_id = _rebind_id(source_branch_id, dest_session_id)
    if source_branch_label and source_branch_label != "main":
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
    archive_inventory: dict[str, ArchiveObjectMeta] | set[str],
    include_runtime: bool,
    trust_runtime: bool,
    *,
    dest_session_id: str,
) -> SanitizedImportPlan:
    """Build a sanitized write plan before any object-store mutation.

    ``archive_inventory`` maps hash → ArchiveObjectMeta (from first pass).
    ``SanitizedImportPlan.object_hashes`` is the only set allowed to reach
    ``put_stream`` in the second pass.
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

    object_hashes = (surviving & inventory_hashes) - skipped_runtime_hashes

    return SanitizedImportPlan(
        object_hashes=object_hashes,
        manifests=planned,
        omitted_runtime_src_ids=omitted_runtime_src_ids,
        skipped_runtime_hashes=skipped_runtime_hashes,
    )


class ImportResult(BaseModel):
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
    from backend.services.snapshot_v2.journal_store import (
        JOURNAL_OWNER_ID,
        JournalStore,
        make_envelope,
    )

    # Stamp the writing process. An import runs in a worker thread
    # (SnapshotService.import_snapshots -> asyncio.to_thread) while the event
    # loop keeps serving requests for the same session, and every request runs
    # recovery (ensure_session_loaded -> recover_snapshot_transactions). Without
    # this tag recovery cannot tell a running import from crash debris: it drops
    # the STAGING journal and post-recovery GC then sweeps the objects Pass 2 has
    # already streamed, so the manifests this import goes on to publish reference
    # objects that no longer exist.
    body = dict(payload or {})
    body["owner"] = JOURNAL_OWNER_ID

    env = make_envelope(
        kind="import",
        transaction_id=transaction_id,
        session_id=session_id,
        status=status,
        payload=body,
    )
    JournalStore(paths).write(env)


def _remove_import_journal(paths, transaction_id: str) -> None:
    from backend.services.snapshot_v2.journal_store import JournalStore
    JournalStore(paths).remove(transaction_id)


def load_active_import_protected_ids(paths, session_id: str | None = None) -> set[str]:
    protected: set[str] = set()
    journals_dir = Path(str(paths.journals))
    if not journals_dir.exists():
        return protected
    from backend.services.snapshot_v2.journal_store import JournalStore
    store = JournalStore(paths)
    for entry in journals_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        if not entry.name.startswith("import-"):
            continue
        try:
            env = store.read(entry.stem)
        except Exception:
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
    """Two-pass streaming import for V2 export tar.gz archives.

    Pass 1: scan metadata only (hash + size), validate, build plan.
    Pass 2: re-open archive, stream approved members through ObjectStore.

    Limits are enforced via ``SnapshotImportLimits``.
    """

    def __init__(
        self,
        dest_session_id: str,
        paths: SnapshotV2Paths,
        object_store: Optional[ObjectStore] = None,
        manifest_store: Optional[ManifestStore] = None,
        limits: Optional[SnapshotImportLimits] = None,
    ) -> None:
        self.dest_session_id = dest_session_id
        self.paths = paths
        self.object_store = object_store or ObjectStore(paths)
        self.manifest_store = manifest_store or ManifestStore(paths)
        self.limits = limits or SnapshotImportLimits()

    def import_bundle(
        self,
        archive_path: str | Path,
        *,
        trust_runtime: bool = False,
    ) -> ImportResult:
        """Import a V2 export archive with batch-atomic visibility.

        Two-pass streaming:
        - Pass 1: scan archive members, record ArchiveObjectMeta,
          validate all manifests, build SanitizedImportPlan.
        - Pass 2: re-open archive, stream only plan.object_hashes.
        """
        archive_path = Path(archive_path)
        result = ImportResult(dest_session_id=self.dest_session_id)

        txn_id = f"import-{uuid.uuid4().hex[:12]}"

        try:
            _write_import_journal(
                self.paths,
                transaction_id=txn_id,
                session_id=self.dest_session_id,
                status="STAGING",
                payload={"created_at": _now_iso()},
            )

            # ── Pass 1: scan metadata ──────────────────────────────
            archive_size = archive_path.stat().st_size
            if archive_size > self.limits.max_compressed_bytes:
                raise SnapshotStorageError(
                    "ARCHIVE_TOO_LARGE",
                    f"Archive size {archive_size} exceeds limit "
                    f"{self.limits.max_compressed_bytes}",
                )

            with tarfile.open(str(archive_path), "r:gz") as tf:
                members = tf.getmembers()

                self._verify_archive_safety(members)
                self._check_member_limits(members)

                meta = self._read_metadata(tf, members)
                runtime_included = meta.get("runtime_included", True)

                # First pass: scan objects (metadata only)
                archive_inventory = self._scan_archive_objects(tf, members)

                # Read and validate manifests (small, full content in memory)
                manifests = self._read_manifests(tf, members)

                # Verify all manifest-referenced objects exist in archive
                for manifest in manifests:
                    self._verify_manifest_objects_meta(manifest, archive_inventory)

            # Build sanitized plan from metadata inventory
            plan = build_import_plan(
                manifests,
                archive_inventory,
                include_runtime=bool(runtime_included),
                trust_runtime=trust_runtime,
                dest_session_id=self.dest_session_id,
            )

            # Destination identity is immutable. Re-importing the exact same
            # target root is idempotent; the same ID with different content
            # must never replace the committed manifest.
            idempotent_ids: set[str] = set()
            for manifest in plan.manifests:
                target_root = hashlib.sha256(canonical_json(manifest)).hexdigest()
                manifest.root_hash = target_root
                committed_path = (
                    Path(str(self.paths.committed))
                    / f"{manifest.snapshot_id}.json"
                )
                if not committed_path.exists():
                    continue
                existing = self.manifest_store.load_verified_including_pending(
                    manifest.snapshot_id,
                )
                if existing.root_hash != target_root:
                    raise SnapshotIntegrityError(
                        "IMPORT_SNAPSHOT_ID_COLLISION",
                        f"Snapshot ID {manifest.snapshot_id} already exists "
                        "with different content",
                    )
                idempotent_ids.add(manifest.snapshot_id)

            # ── Pass 2: stream approved objects through ObjectStore ─
            write_proofs: dict[str, ObjectRef] = {}
            with tarfile.open(str(archive_path), "r:gz") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if not m.name.startswith("objects/") or m.name.count("/") != 1:
                        continue
                    obj_hash = m.name.split("/")[1]
                    if obj_hash not in plan.object_hashes:
                        continue

                    f = tf.extractfile(m)
                    if f is None:
                        continue

                    meta = archive_inventory[obj_hash]
                    put_result = self.object_store.put_stream(
                        f,
                        expected_sha256=obj_hash,
                        expected_size=meta.size,
                        media_type="application/octet-stream",
                    )
                    write_proofs[obj_hash] = put_result.ref

            # Update journal to STAGED
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

            # Mark ALL target manifests pending BEFORE publishing any of them,
            # so a crash mid-loop leaves every published manifest invisible
            # until the batch record commits (design §6.2 batch atomicity).
            planned_ids = [
                m.snapshot_id
                for m in plan.manifests
                if m.snapshot_id not in idempotent_ids
            ]
            ManifestStore.mark_pending_imports(self.paths.committed, planned_ids)

            # Stage and publish each rebound manifest (write proofs only)
            manifest_roots: dict[str, str] = {}
            for manifest in plan.manifests:
                if manifest.snapshot_id in idempotent_ids:
                    manifest_roots[manifest.snapshot_id] = manifest.root_hash
                    result.imported_snapshot_ids.append(manifest.snapshot_id)
                    result.warnings.append(
                        f"Snapshot {manifest.snapshot_id} already imported; "
                        "reused identical committed root"
                    )
                    continue
                staging_path = self.manifest_store.write_staging(
                    manifest, prefix=f"{txn_id}_",
                )
                self._assert_manifest_write_proofs(manifest, write_proofs)
                self.manifest_store.publish_staging(staging_path)
                manifest_roots[manifest.snapshot_id] = manifest.root_hash
                result.imported_snapshot_ids.append(manifest.snapshot_id)
                result.committed_count += 1


            # Reconcile staging directory
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

            # Publish batch record
            batch_store = ImportBatchStore(self.paths.root)
            batch_record = ImportBatchRecord(
                batch_id=txn_id,
                session_id=self.dest_session_id,
                manifest_roots=manifest_roots,
            )
            batch_store.publish(batch_record)

            ManifestStore.clear_pending_imports(
                self.paths.committed,
                list(manifest_roots.keys()),
            )

            # Run safe GC
            self._run_safe_gc()

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

            _remove_import_journal(self.paths, txn_id)

            result.success = True
            result.skipped_runtime_snapshot_ids = [
                _rebind_id(sid, self.dest_session_id)
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

    # ── Limit checks ──────────────────────────────────────────────────

    def _check_member_limits(self, members: list) -> None:
        """Enforce member count and size limits from import bounds."""
        obj_count = 0
        manifest_count = 0
        snapshot_ids: set[str] = set()
        total_uncompressed = 0

        for m in members:
            if not m.isfile():
                continue
            if m.size > self.limits.max_single_member_bytes:
                raise SnapshotStorageError(
                    "MEMBER_TOO_LARGE",
                    f"Tar member {m.name} size {m.size} exceeds "
                    f"single-member limit {self.limits.max_single_member_bytes}",
                )
            if m.name.startswith("objects/") and m.name.count("/") == 1:
                obj_count += 1
                total_uncompressed += m.size
                if obj_count > self.limits.max_member_count:
                    raise SnapshotStorageError(
                        "TOO_MANY_OBJECTS",
                        f"Object count {obj_count} exceeds limit "
                        f"{self.limits.max_member_count}",
                    )
            if m.name.startswith("manifests/") and m.name.endswith(".json"):
                manifest_count += 1
                sid = m.name[len("manifests/"):-len(".json")]
                if sid:
                    snapshot_ids.add(sid)
                if manifest_count > self.limits.max_manifest_count:
                    raise SnapshotStorageError(
                        "TOO_MANY_MANIFESTS",
                        f"Manifest count {manifest_count} exceeds limit "
                        f"{self.limits.max_manifest_count}",
                    )

        if total_uncompressed > self.limits.max_uncompressed_bytes:
            raise SnapshotStorageError(
                "TOTAL_UNCOMPRESSED_TOO_LARGE",
                f"Total uncompressed object data {total_uncompressed} exceeds "
                f"limit {self.limits.max_uncompressed_bytes}",
            )
        if len(snapshot_ids) > self.limits.max_snapshot_count:
            raise SnapshotStorageError(
                "TOO_MANY_SNAPSHOTS",
                f"Snapshot count {len(snapshot_ids)} exceeds limit "
                f"{self.limits.max_snapshot_count}",
            )

    # ── Pass 1 helpers ────────────────────────────────────────────────

    def _verify_archive_safety(self, members: list) -> None:
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
        manifests: list[SnapshotManifestV2] = []
        seen_ids: set[str] = set()

        for m in members:
            if not m.isfile():
                continue
            if not m.name.startswith("manifests/") or not m.name.endswith(".json"):
                continue
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

    def _scan_archive_objects(
        self, tf: tarfile.TarFile, members: list,
    ) -> dict[str, ArchiveObjectMeta]:
        """Pass 1: hash and record metadata only (no full content in memory)."""
        inventory: dict[str, ArchiveObjectMeta] = {}
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

            hasher = hashlib.sha256()
            size = 0
            while True:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)

            actual_hash = hasher.hexdigest()
            if actual_hash != obj_hash:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Object {obj_hash} content hash mismatch "
                    f"(actual: {actual_hash})",
                )

            if obj_hash not in inventory:
                inventory[obj_hash] = ArchiveObjectMeta(sha256=obj_hash, size=size)

        return inventory

    def _verify_manifest_objects_meta(
        self,
        manifest: SnapshotManifestV2,
        archive_inventory: dict[str, ArchiveObjectMeta],
    ) -> None:
        """Every object referenced by *manifest* must exist in archive inventory."""
        for h, ref in manifest.objects.items():
            meta = archive_inventory.get(h)
            if meta is None:
                raise SnapshotIntegrityError(
                    "MISSING_OBJECT",
                    f"Object {h} referenced by manifest "
                    f"{manifest.snapshot_id} not found in archive",
                )
            if meta.size != ref.size:
                raise SnapshotIntegrityError(
                    "OBJECT_SIZE_MISMATCH",
                    f"Object {h} size {meta.size} != expected {ref.size}",
                )

    # ── Post-plan helpers ─────────────────────────────────────────────

    def _assert_manifest_write_proofs(
        self,
        manifest: SnapshotManifestV2,
        write_proofs: dict[str, ObjectRef],
    ) -> None:
        """Require each manifest object in write_proofs or verified pre-existing evidence.

        Never rereads destination objects written in this import transaction.
        """
        for h, ref in manifest.objects.items():
            proof = write_proofs.get(h)
            if proof is not None:
                if proof.size != ref.size:
                    raise SnapshotIntegrityError(
                        "OBJECT_SIZE_MISMATCH",
                        f"Object {h} write proof size {proof.size} != expected {ref.size}",
                    )
                continue
            # Pre-existing object: full verify once (not a same-transaction write)
            try:
                self.object_store.verify(ref)
            except SnapshotV2Error as exc:
                raise SnapshotIntegrityError(
                    "DESTINATION_OBJECT_MISSING",
                    f"Object {h} missing in destination store: {exc}",
                )

    def _run_safe_gc(self) -> None:
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
