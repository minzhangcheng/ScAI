import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from backend.services.snapshot_v2.errors import SnapshotIntegrityError, SnapshotStorageError
from backend.services.snapshot_v2.models import ObjectRef, SnapshotManifestV2
from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256, verify_hmac_sha256
from backend.utils.files import atomic_write_bytes, fsync_directory


CHUNK_SIZE = 1 * 1024 * 1024
_EVIDENCE_CACHE_MAX = 4096

#: How old a ``staging/.{sha}.*`` temp must be before a sweep may reclaim it.
#: Comfortably longer than any single put_* of a multi-GB h5ad, so a live
#: transfer is never mistaken for debris. Public because recovery's
#: orphan-staging sweep applies the same rule and the two must not drift.
STAGING_TEMP_TTL_SECONDS = 24 * 3600


def reclaim_stale_staging_temp(entry: Path) -> bool:
    """Unlink *entry* if it is a ``put_*`` temp older than the TTL.

    Returns whether it was removed. Shared by ``ManifestStore.reconcile`` and
    recovery's orphan-staging sweep so the two cannot drift.
    """
    try:
        if (entry.is_file()
                and time.time() - entry.stat().st_mtime
                > STAGING_TEMP_TTL_SECONDS):
            entry.unlink()
            return True
    except OSError:
        pass
    return False


def physical_object_store_bytes(objects_dir: Path) -> int:
    """Bytes currently occupied by the object store at *objects_dir*.

    The one measurement both capacity gates — capture quota and import
    admission — read, so the two can never answer differently.
    """
    objects_dir = Path(str(objects_dir))
    if not objects_dir.is_dir():
        return 0
    total = 0
    for path in objects_dir.iterdir():
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def free_bytes_at(root: Path) -> int:
    """Free bytes of the filesystem holding *root*'s nearest existing ancestor.

    The v2 root is not created until a capture runs, so the probe walks up —
    free space is identical across the same filesystem. ``OSError`` propagates:
    each capacity gate maps it to its own error code.
    """
    probe = Path(str(root))
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


@dataclass(frozen=True)
class ObjectPutResult:
    """Result of a successful put operation."""
    ref: ObjectRef
    created: bool


@dataclass(frozen=True)
class VerifiedObjectEvidence:
    """Bounded cached evidence that an object at *sha256* was verified.

    *ref* is the proven ObjectRef. *stat_fingerprint* captures the
    (device, inode, mtime_ns, ctime_ns, size) tuple so we can detect file
    replacement (GC deletion followed by new put) without rehashing.
    """
    ref: ObjectRef
    stat_fingerprint: str

    @classmethod
    def from_path(cls, path: Path, sha256: str) -> "VerifiedObjectEvidence":
        st = path.stat()
        fp = f"{st.st_dev}:{st.st_ino}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_size}"
        return cls(
            ref=ObjectRef(sha256=sha256, size=st.st_size, media_type="application/octet-stream"),
            stat_fingerprint=fp,
        )

    def matches_path(self, path: Path) -> bool:
        try:
            st = path.stat()
            return (
                f"{st.st_dev}:{st.st_ino}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_size}"
                == self.stat_fingerprint
            )
        except OSError:
            return False


class ArchiveObjectMeta(BaseModel):
    """Metadata for an object in an import archive (hash + size, no content)."""
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)


class SnapshotImportLimits(BaseModel):
    """Bounded import limits, typically derived from ``SystemConfig``.

    :param max_compressed_bytes: Maximum compressed archive size (20 GB default).
    :param max_uncompressed_bytes: Maximum total uncompressed object data (100 GB).
    :param max_single_member_bytes: Maximum single tar member size (80 GB).
    :param max_member_count: Maximum number of tar members (200,000).
    :param max_manifest_count: Maximum number of manifest JSON files (10,000).
    :param max_snapshot_count: Maximum number of unique snapshot IDs (10,000).
    :param session_quota_bytes: Session-level store quota (from SystemConfig).
    :param free_space_reserve_bytes: Free-space reserve (from SystemConfig).

    The two capacity defaults below mirror ``SystemConfig``; this module takes a
    duck-typed config rather than importing ``backend.config``, so they cannot
    reference it directly. ``test_capacity_defaults_mirror_system_config`` fails
    if the mirror drifts.
    """
    max_compressed_bytes: int = 20 * 1024 ** 3
    max_uncompressed_bytes: int = 100 * 1024 ** 3
    max_single_member_bytes: int = 80 * 1024 ** 3
    max_member_count: int = 200000
    max_manifest_count: int = 10000
    max_snapshot_count: int = 10000
    session_quota_bytes: int = 500 * 1024 ** 3
    free_space_reserve_bytes: int = 50 * 1024 ** 3
    #: Caps for the two member kinds the importer reads *whole* into memory.
    #: Without them both were bounded only by max_single_member_bytes (80 GB),
    #: and since JSON compresses at ~1000:1 a few MB of upload could force a
    #: multi-GB allocation on an endpoint any session owner can reach.
    max_metadata_bytes: int = 8 * 1024 ** 2
    max_manifest_bytes: int = 16 * 1024 ** 2

    @classmethod
    def from_config(cls, config: Any) -> "SnapshotImportLimits":
        """Build limits from a ``SystemConfig`` or duck-typed config object."""
        return cls(
            max_compressed_bytes=getattr(
                config, "snapshot_v2_import_compressed_bytes", 20 * 1024 ** 3,
            ),
            max_uncompressed_bytes=getattr(
                config, "snapshot_v2_import_uncompressed_bytes", 100 * 1024 ** 3,
            ),
            max_single_member_bytes=getattr(
                config, "snapshot_v2_import_single_member_bytes", 80 * 1024 ** 3,
            ),
            max_member_count=getattr(
                config, "snapshot_v2_import_member_count", 200000,
            ),
            max_manifest_count=getattr(
                config, "snapshot_v2_import_manifest_count", 10000,
            ),
            max_snapshot_count=getattr(
                config, "snapshot_v2_import_snapshot_count", 10000,
            ),
            session_quota_bytes=getattr(
                config,
                "snapshot_v2_session_quota_bytes",
                cls.model_fields["session_quota_bytes"].default,
            ),
            free_space_reserve_bytes=getattr(
                config,
                "snapshot_v2_free_space_reserve_bytes",
                cls.model_fields["free_space_reserve_bytes"].default,
            ),
        )


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_json(value: BaseModel | dict) -> bytes:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = dict(value)
    payload.pop("root_hash", None)
    return _canonical_json_bytes(payload)


def canonical_manifest_body_bytes(manifest: SnapshotManifestV2) -> bytes:
    """Canonical body bytes excluding root_hash (used for signing)."""
    payload = manifest.model_dump(mode="json")
    payload.pop("root_hash", None)
    return _canonical_json_bytes(payload)


def canonical_manifest_document_bytes(manifest: SnapshotManifestV2) -> bytes:
    """Canonical full document bytes including root_hash."""
    return _canonical_json_bytes(manifest.model_dump(mode="json"))


def compute_manifest_root_hash(manifest: SnapshotManifestV2) -> str:
    """The root_hash a manifest body derives to.

    The importer refuses a manifest whose root_hash does not re-derive
    (MANIFEST_ROOT_HASH_MISMATCH), so every producer must use this one rule.
    """
    return hashlib.sha256(canonical_manifest_body_bytes(manifest)).hexdigest()


def _hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    offset = 0
    while offset < len(data):
        h.update(data[offset : offset + CHUNK_SIZE])
        offset += CHUNK_SIZE
    return h.hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ObjectStore:
    def __init__(self, paths):
        self._paths = paths
        self._objects_dir = paths.objects
        # Bounded evidence cache: sha256 -> VerifiedObjectEvidence
        self._evidence: dict[str, "VerifiedObjectEvidence"] = {}
        self._evidence_max = _EVIDENCE_CACHE_MAX

    def _ensure_store_dirs(self) -> None:
        """Create staging/objects dirs on demand for first-write paths.

        Fresh sessions only create workspace/states/uploads/reports/logs.
        Snapshot V2 dirs are created lazily so put_* never fails with
        FileNotFoundError on missing staging/objects parents.
        """
        self._paths.staging.mkdir(parents=True, exist_ok=True)
        self._objects_dir.mkdir(parents=True, exist_ok=True)

    def _fsync_objects_dir(self) -> None:
        """Make an object's new directory entry durable after ``os.replace``.

        Fsyncing the temp file only makes its *contents* durable; the rename
        that publishes it into ``objects/`` is a directory-metadata operation
        and needs its own fsync. Without it the store can come back from a
        power loss with a committed, HMAC-verified manifest — ``publish_staging``
        does fsync ``committed/`` — whose objects were never linked. Nothing
        detects that: ``verify_all_committed`` checks signatures, not object
        existence, so the snapshot lists as healthy and only fails at restore,
        with no repair path.
        """
        try:
            fsync_directory(self._objects_dir)
        except OSError:
            # A directory fsync can legitimately fail on exotic filesystems.
            # The object is already installed; degrading durability is better
            # than failing a capture that otherwise succeeded.
            pass

    def _register_evidence(self, sha256: str, evidence: "VerifiedObjectEvidence") -> None:
        if len(self._evidence) >= self._evidence_max:
            # Evict oldest entries (simple FIFO-style)
            for key in list(self._evidence.keys())[:self._evidence_max // 4]:
                self._evidence.pop(key, None)
        self._evidence[sha256] = evidence

    def _get_evidence(self, sha256: str) -> "VerifiedObjectEvidence | None":
        ev = self._evidence.get(sha256)
        if ev is None:
            return None
        obj_path = self._objects_dir / sha256
        if not obj_path.exists() or not ev.matches_path(obj_path):
            self._evidence.pop(sha256, None)
            return None
        return ev

    def invalidate_evidence(self, sha256: str) -> None:
        """Drop cached proof after a transaction removes an unpublished object."""
        self._evidence.pop(sha256, None)

    def put_bytes(self, data: bytes, media_type: str) -> ObjectPutResult:
        sha = _hash_bytes(data)
        dest = self._objects_dir / sha

        # Fast path: cached evidence
        cached = self._get_evidence(sha)
        if cached is not None:
            return ObjectPutResult(
                ref=ObjectRef(sha256=sha, size=cached.ref.size, media_type=media_type),
                created=False,
            )

        if dest.exists():
            actual_size = dest.stat().st_size
            actual_hash = _hash_file(dest)
            if actual_size != len(data) or actual_hash != sha:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Existing object {sha} has mismatched content",
                )
            ev = VerifiedObjectEvidence.from_path(dest, sha)
            self._register_evidence(sha, ev)
            return ObjectPutResult(
                ref=ObjectRef(sha256=sha, size=len(data), media_type=media_type),
                created=False,
            )

        self._ensure_store_dirs()
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{sha}.",
                suffix=".tmp",
                dir=self._paths.staging,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(str(temp_path), str(dest))
            self._fsync_objects_dir()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        ev = VerifiedObjectEvidence.from_path(dest, sha)
        self._register_evidence(sha, ev)
        return ObjectPutResult(
            ref=ObjectRef(sha256=sha, size=len(data), media_type=media_type),
            created=True,
        )

    def put_file(self, path: Path, media_type: str) -> ObjectPutResult:
        resolved = path.resolve()
        if not resolved.exists():
            raise SnapshotStorageError("FILE_NOT_FOUND", f"File not found: {path}")
        if not resolved.is_file():
            raise SnapshotStorageError(
                "NOT_A_REGULAR_FILE",
                f"Path is not a regular file: {path}",
            )
        if resolved.is_symlink() or path.is_symlink():
            raise SnapshotStorageError(
                "SYMLINK_REJECTED",
                f"Symlinks are not allowed: {path}",
            )
        size = resolved.stat().st_size
        sha = _hash_file(resolved)
        dest = self._objects_dir / sha

        # Fast path: cached evidence
        cached = self._get_evidence(sha)
        if cached is not None:
            return ObjectPutResult(
                ref=ObjectRef(sha256=sha, size=cached.ref.size, media_type=media_type),
                created=False,
            )

        if dest.exists():
            actual_size = dest.stat().st_size
            actual_hash = _hash_file(dest)
            if actual_size != size or actual_hash != sha:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Existing object {sha} has mismatched content",
                )
            ev = VerifiedObjectEvidence.from_path(dest, sha)
            self._register_evidence(sha, ev)
            return ObjectPutResult(
                ref=ObjectRef(sha256=sha, size=size, media_type=media_type),
                created=False,
            )

        self._ensure_store_dirs()
        # ``delete=False`` means an error anywhere below leaves the partial temp
        # behind. Quota accounting only sums ``objects/``, and the staging sweeps
        # reclaim a dot-prefixed name only once it is a day old, so an ENOSPC
        # here (the likeliest failure, on the largest objects in the system)
        # would otherwise leak a near-complete copy on every HITL Retry and only
        # get it back a day later.
        tmp_path: Path | None = None
        copied = 0
        copied_hash = hashlib.sha256()
        try:
            with tempfile.NamedTemporaryFile(dir=str(self._paths.staging), prefix=f".{sha}.", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                with open(resolved, "rb") as src:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        copied += len(chunk)
                        copied_hash.update(chunk)
                        tmp.write(chunk)
                tmp.flush()
                os.fsync(tmp.fileno())
            # ``sha`` came from a *separate* pass over the source. A writer the
            # capture does not exclude can land between the two passes — the
            # chat-delegation sandbox holds its own exec lock and writes into
            # the same workspace the capture walks — and installing the second
            # pass's bytes under the first pass's name yields an object that
            # does not hash to its own name. The manifest naming it then commits
            # and HMAC-signs clean, verify_all_committed never reads object
            # content, and only the restore rehash notices, by which point the
            # snapshot is permanently unrestorable. Prove the copy instead, as
            # put_bytes and put_stream already do. Hashing the chunks we are
            # already reading costs no extra I/O — a second read of a multi-GB
            # h5ad would.
            if copied != size or copied_hash.hexdigest() != sha:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Source file changed while being stored: {path}",
                )
            os.replace(str(tmp_path), str(dest))
            self._fsync_objects_dir()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        ev = VerifiedObjectEvidence.from_path(dest, sha)
        self._register_evidence(sha, ev)
        return ObjectPutResult(
            ref=ObjectRef(sha256=sha, size=size, media_type=media_type),
            created=True,
        )

    def put_stream(
        self,
        reader,
        expected_sha256: str,
        expected_size: int,
        media_type: str,
    ) -> ObjectPutResult:
        """Stream *reader* into the object store without buffering full content.

        Writes chunks to a staging file while hashing and counting, checks
        expected hash/size, fsyncs, atomically installs, and returns evidence
        from the final path.
        """
        if not expected_sha256 or len(expected_sha256) != 64:
            raise SnapshotIntegrityError(
                "OBJECT_HASH_MISMATCH",
                f"Invalid expected sha256: {expected_sha256!r}",
            )
        if expected_size < 0:
            raise SnapshotIntegrityError(
                "OBJECT_SIZE_MISMATCH",
                f"Invalid expected size: {expected_size}",
            )

        dest = self._objects_dir / expected_sha256

        # Fast path: cached evidence with matching size
        cached = self._get_evidence(expected_sha256)
        if cached is not None:
            if cached.ref.size != expected_size:
                raise SnapshotIntegrityError(
                    "OBJECT_SIZE_MISMATCH",
                    f"Object {expected_sha256} size {cached.ref.size} != expected {expected_size}",
                )
            return ObjectPutResult(
                ref=ObjectRef(
                    sha256=expected_sha256,
                    size=expected_size,
                    media_type=media_type,
                ),
                created=False,
            )

        if dest.exists():
            actual_size = dest.stat().st_size
            actual_hash = _hash_file(dest)
            if actual_size != expected_size or actual_hash != expected_sha256:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Existing object {expected_sha256} has mismatched content",
                )
            # Drain reader so callers can finish the stream without keeping data
            while True:
                chunk = reader.read(CHUNK_SIZE)
                if not chunk:
                    break
            ev = VerifiedObjectEvidence.from_path(dest, expected_sha256)
            self._register_evidence(expected_sha256, ev)
            return ObjectPutResult(
                ref=ObjectRef(
                    sha256=expected_sha256,
                    size=expected_size,
                    media_type=media_type,
                ),
                created=False,
            )

        hasher = hashlib.sha256()
        total = 0
        self._ensure_store_dirs()
        # The three ad-hoc unlinks this replaces covered the write loop and the
        # two integrity branches but left ``os.replace`` outside all of them —
        # the last of the three put_* writers whose install could leak. An
        # ENOSPC or EIO there stranded a fully written, up-to-single-member-limit
        # temp in staging/ that only the day-old TTL sweep in recovery and
        # ManifestStore.reconcile() would ever reclaim -- import recovery matches
        # only ``{txn}``-prefixed entries, and quota sums only ``objects/``.
        # One finally spans the whole thing instead; after a
        # successful rename the unlink is a no-op via ``missing_ok``.
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(self._paths.staging),
                prefix=f".{expected_sha256}.",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                while True:
                    chunk = reader.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    total += len(chunk)
                    tmp.write(chunk)
                tmp.flush()
                os.fsync(tmp.fileno())

            actual_hash = hasher.hexdigest()
            if total != expected_size:
                raise SnapshotIntegrityError(
                    "OBJECT_SIZE_MISMATCH",
                    f"Object {expected_sha256} size {total} != expected {expected_size}",
                )
            if actual_hash != expected_sha256:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Object {expected_sha256} hash mismatch (actual: {actual_hash})",
                )

            os.replace(str(tmp_path), str(dest))
            self._fsync_objects_dir()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        ev = VerifiedObjectEvidence.from_path(dest, expected_sha256)
        self._register_evidence(expected_sha256, ev)
        return ObjectPutResult(
            ref=ObjectRef(
                sha256=expected_sha256,
                size=expected_size,
                media_type=media_type,
            ),
            created=True,
        )

    def register_existing_ref(self, ref: ObjectRef) -> bool:
        """Register an existing ObjectRef as verified evidence (fast path).

        Returns True if evidence was registered, False if the object does not
        exist or its hash does not match.

        This is the cross-transaction fast path: instead of re-verifying
        an object that was already proven in a prior transaction, callers
        can register it here so subsequent ``_get_evidence`` hits the cache.
        """
        if not ref.sha256 or len(ref.sha256) != 64:
            return False
        obj_path = self._objects_dir / ref.sha256
        if not obj_path.exists() or not obj_path.is_file():
            return False
        try:
            st = obj_path.stat()
        except OSError:
            return False
        # Validate size matches
        if st.st_size != ref.size:
            return False
        # Validate content hash (streaming verify for cold cache)
        cached = self._get_evidence(ref.sha256)
        if cached is not None:
            return True
        actual = _hash_file(obj_path)
        if actual != ref.sha256:
            return False
        ev = VerifiedObjectEvidence(ref=ref, stat_fingerprint=f"{st.st_dev}:{st.st_ino}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_size}")
        self._register_evidence(ref.sha256, ev)
        return True

    def get_bytes(self, sha256: str) -> bytes:
        obj_path = self._objects_dir / sha256
        if not obj_path.exists():
            raise SnapshotIntegrityError(
                "OBJECT_NOT_FOUND",
                f"Object {sha256} not found at {obj_path}",
            )
        return obj_path.read_bytes()

    def contains_verified(self, sha256: str) -> bool:
        """Return True if *sha256* exists on disk and its content hash matches.

        Used to compute physical_added_bytes (only count hashes absent before capture).
        """
        if not sha256 or len(sha256) != 64:
            return False
        # Fast path: cached evidence
        cached = self._get_evidence(sha256)
        if cached is not None:
            return True
        obj_path = self._objects_dir / sha256
        if not obj_path.exists() or not obj_path.is_file():
            return False
        try:
            actual = _hash_file(obj_path)
        except OSError:
            return False
        if actual == sha256:
            ev = VerifiedObjectEvidence.from_path(obj_path, sha256)
            self._register_evidence(sha256, ev)
            return True
        return False

    def verify(self, ref: ObjectRef) -> None:
        # Fast path: cached evidence
        cached = self._get_evidence(ref.sha256)
        if cached is not None:
            if cached.ref.size != ref.size:
                raise SnapshotIntegrityError(
                    "OBJECT_SIZE_MISMATCH",
                    f"Object {ref.sha256} size {cached.ref.size} != expected {ref.size}",
                )
            return
        obj_path = self._objects_dir / ref.sha256
        if not obj_path.exists():
            raise SnapshotIntegrityError(
                "OBJECT_NOT_FOUND",
                f"Object {ref.sha256} not found at {obj_path}",
            )
        actual_size = obj_path.stat().st_size
        if actual_size != ref.size:
            raise SnapshotIntegrityError(
                "OBJECT_SIZE_MISMATCH",
                f"Object {ref.sha256} size {actual_size} != expected {ref.size}",
            )
        actual_hash = _hash_file(obj_path)
        if actual_hash != ref.sha256:
            raise SnapshotIntegrityError(
                "OBJECT_HASH_MISMATCH",
                f"Object {ref.sha256} hash mismatch (actual: {actual_hash})",
            )
        ev = VerifiedObjectEvidence(ref=ref, stat_fingerprint=f"{obj_path.stat().st_dev}:{obj_path.stat().st_ino}:{obj_path.stat().st_mtime_ns}:{obj_path.stat().st_ctime_ns}:{obj_path.stat().st_size}")
        self._register_evidence(ref.sha256, ev)


@dataclass(frozen=True)
class ManifestListEntry:
    """One committed manifest path: either verified or a typed integrity diagnostic."""

    snapshot_id: str
    status: str  # "ok" | "corrupt"
    manifest: SnapshotManifestV2 | None = None
    error_code: str | None = None
    error_message: str | None = None

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest.model_dump(mode=mode)
        else:
            payload["manifest"] = None
        return payload


@dataclass(frozen=True)
class ReconcileResult:
    """Typed result of ManifestStore.reconcile()."""

    verified_ids: list[str]
    corrupt_ids: list[str]
    removed_staging_ids: list[str]


class ManifestStore:
    def __init__(self, paths):
        self._paths = paths

    def _sign(self, data: bytes) -> str:
        return sign_hmac_sha256(data)

    def _verify_signature(self, data: bytes, sig: str) -> bool:
        return verify_hmac_sha256(data, sig)

    def write_staging(self, manifest: SnapshotManifestV2, prefix: str = "") -> Path:
        manifest.root_hash = compute_manifest_root_hash(manifest)
        document = canonical_manifest_document_bytes(manifest)
        staging_path = self._paths.staging / f"{prefix}{manifest.snapshot_id}.json"
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.write_bytes(document)
        staging_fd = os.open(staging_path, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        return staging_path

    def publish_staging(self, staging_path: Path) -> SnapshotManifestV2:
        manifest_bytes = staging_path.read_bytes()
        manifest = SnapshotManifestV2.model_validate_json(manifest_bytes)
        committed_path = self._paths.committed / f"{manifest.snapshot_id}.json"
        sig_path = self._paths.integrity / f"{manifest.snapshot_id}.sig"
        # Write signature first (atomic), then commit manifest:
        # if we crash after sig but before commit, the missing manifest file
        # causes load() to raise MANIFEST_NOT_FOUND — the sig is harmless.
        atomic_write_bytes(sig_path, self._sign(manifest_bytes).encode())
        committed_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staging_path), str(committed_path))
        fsync_directory(committed_path.parent)
        # Verify integrity without pending-import visibility gate so pre-publish
        # markers (import batch atomicity) do not block the publisher itself.
        return self._load_verified(manifest.snapshot_id)

    def load(self, snapshot_id: str) -> SnapshotManifestV2:
        if snapshot_id in ManifestStore.list_pending_import_ids(self._paths.committed):
            raise SnapshotIntegrityError(
                "MANIFEST_NOT_FOUND",
                f"Committed manifest {snapshot_id} not found",
            )
        return self._load_verified(snapshot_id)

    def load_verified_including_pending(
        self, snapshot_id: str,
    ) -> SnapshotManifestV2:
        """Integrity-check a committed manifest without applying visibility."""
        return self._load_verified(snapshot_id)

    def load_pending_verified(self, snapshot_id: str) -> SnapshotManifestV2:
        """Integrity-check a manifest that is explicitly pending import."""
        pending = ManifestStore.list_pending_import_ids(self._paths.committed)
        if snapshot_id not in pending:
            raise SnapshotIntegrityError(
                "MANIFEST_NOT_PENDING",
                f"Manifest {snapshot_id} is not pending import",
            )
        return self._load_verified(snapshot_id)

    def _load_verified(self, snapshot_id: str) -> SnapshotManifestV2:
        """Load and integrity-check a committed manifest (ignores pending markers)."""
        committed_path = self._paths.committed / f"{snapshot_id}.json"
        sig_path = self._paths.integrity / f"{snapshot_id}.sig"
        if not committed_path.exists():
            raise SnapshotIntegrityError(
                "MANIFEST_NOT_FOUND",
                f"Committed manifest {snapshot_id} not found",
            )
        manifest_bytes = committed_path.read_bytes()
        sig = sig_path.read_text().strip() if sig_path.exists() else ""
        try:
            manifest = SnapshotManifestV2.model_validate_json(manifest_bytes)
        except PydanticValidationError:
            raise SnapshotIntegrityError(
                "MANIFEST_PARSE_ERROR",
                f"Committed manifest {snapshot_id} failed validation",
            )
        if not self._verify_signature(manifest_bytes, sig):
            raise SnapshotIntegrityError(
                "MANIFEST_HMAC_MISMATCH",
                f"HMAC signature mismatch for snapshot {snapshot_id}",
            )
        expected_root = hashlib.sha256(canonical_manifest_body_bytes(manifest)).hexdigest()
        if manifest.root_hash != expected_root:
            raise SnapshotIntegrityError(
                "MANIFEST_ROOT_HASH_MISMATCH",
                f"Root hash mismatch for snapshot {snapshot_id}: "
                f"expected {expected_root}, got {manifest.root_hash}",
            )
        return manifest

    @staticmethod
    def _pending_marker_path(committed_dir: Path, snapshot_id: str) -> Path:
        return committed_dir / f".pending-import-{snapshot_id}"

    @classmethod
    def mark_pending_imports(cls, committed_dir: Path, snapshot_ids: list[str]) -> None:
        """Mark manifests as pending import (invisible until batch is published)."""
        committed_dir.mkdir(parents=True, exist_ok=True)
        for sid in snapshot_ids:
            cls._pending_marker_path(committed_dir, sid).touch()

    @classmethod
    def clear_pending_imports(cls, committed_dir: Path, snapshot_ids: list[str]) -> None:
        """Remove pending-import markers, making manifests visible."""
        for sid in snapshot_ids:
            cls._pending_marker_path(committed_dir, sid).unlink(missing_ok=True)

    @classmethod
    def list_pending_import_ids(cls, committed_dir: Path) -> set[str]:
        """Return snapshot IDs that have pending-import markers."""
        if not committed_dir.exists():
            return set()
        pending: set[str] = set()
        for entry in committed_dir.iterdir():
            name = entry.name
            if name.startswith(".pending-import-"):
                pending.add(name[len(".pending-import-"):])
        return pending

    def _iter_committed_json_stems(self) -> list[str]:
        """Sorted non-audit committed manifest stems, excluding pending imports."""
        committed_dir = self._paths.committed
        if not committed_dir.exists():
            return []
        pending = self.list_pending_import_ids(committed_dir)
        stems: list[str] = []
        for entry in sorted(committed_dir.iterdir()):
            if entry.suffix != ".json" or not entry.is_file():
                continue
            if entry.stem.startswith("audit-"):
                continue
            if entry.stem in pending:
                continue
            stems.append(entry.stem)
        return stems

    def verify_all_committed(self) -> list[SnapshotManifestV2]:
        """Load and verify every committed (non-audit) manifest.

        Raises :class:`SnapshotIntegrityError` on the first corrupt record.
        Used by GC, deletion, export, and idempotency — never by UI listing.
        """
        verified: list[SnapshotManifestV2] = []
        for stem in self._iter_committed_json_stems():
            verified.append(self.load(stem))
        return verified

    def list_committed(self) -> list[SnapshotManifestV2]:
        """Return only verified committed manifests (raises on corruption).

        Alias of :meth:`verify_all_committed` for storage-safety callers.
        UI listing must use :meth:`list_committed_with_diagnostics` instead.
        """
        return self.verify_all_committed()

    def list_committed_with_diagnostics(self) -> list[ManifestListEntry]:
        """List committed manifests; corrupt entries become typed diagnostics.

        Presentation-only. Must not be used by GC, deletion, export, or
        idempotency paths.
        """
        result: list[ManifestListEntry] = []
        for stem in self._iter_committed_json_stems():
            try:
                manifest = self.load(stem)
                result.append(
                    ManifestListEntry(
                        snapshot_id=stem,
                        status="ok",
                        manifest=manifest,
                    )
                )
            except SnapshotIntegrityError as exc:
                result.append(
                    ManifestListEntry(
                        snapshot_id=stem,
                        status="corrupt",
                        error_code=exc.code,
                        error_message=str(exc),
                    )
                )
        return result

    def reconcile(self) -> "ReconcileResult":
        """Remove orphan staging and report verified vs corrupt committed IDs.

        Never silently discards integrity failures: corrupt IDs are returned
        in :attr:`ReconcileResult.corrupt_ids` for callers to surface.
        """
        removed_staging_ids: list[str] = []
        staging_dir = self._paths.staging
        if staging_dir.exists():
            for entry in list(staging_dir.iterdir()):
                # ``staging/`` is shared: ManifestStore writes
                # ``{tx}_{snapshot_id}.json`` here, but every ObjectStore.put_*
                # also writes its in-flight temp here as ``.{sha}.…``. Today no
                # caller can interleave the two — every reconcile() caller runs
                # under the per-session lock that capture also holds — so this
                # exemption is a contract, not a bug fix: it makes reconcile
                # safe on its own terms rather than on its callers', and matches
                # the filter recovery's orphan-staging sweep already applies.
                # write_staging only ever produces ``{prefix}{snapshot_id}.json``,
                # so nothing reconcile is meant to reclaim starts with a dot.
                if entry.name.startswith("."):
                    # An in-flight temp is never this old; anything older was
                    # stranded by a hard kill (OOM, container stop, power loss),
                    # where the put_* ``finally`` never ran. Nothing else walks
                    # ``staging/``, so an unreclaimed temp is invisible to GC and
                    # to quota accounting and counts against the free-space
                    # reserve forever. Its stem is a content hash, not a snapshot
                    # id, so it must not enter removed_staging_ids.
                    reclaim_stale_staging_temp(entry)
                    continue
                if entry.is_file() or entry.is_symlink():
                    removed_staging_ids.append(entry.stem)
                    entry.unlink()

        verified_ids: list[str] = []
        corrupt_ids: list[str] = []
        committed_dir = self._paths.committed
        pending = self.list_pending_import_ids(committed_dir) if committed_dir.exists() else set()
        if committed_dir.exists():
            for entry in sorted(committed_dir.iterdir()):
                if entry.suffix != ".json" or not entry.is_file():
                    continue
                if entry.stem.startswith("audit-"):
                    continue
                # Pending-import manifests are intentionally invisible until the
                # batch record commits; verify integrity without visibility gate.
                if entry.stem in pending:
                    try:
                        self._load_verified(entry.stem)
                    except SnapshotIntegrityError:
                        corrupt_ids.append(entry.stem)
                    continue
                try:
                    self.load(entry.stem)
                    verified_ids.append(entry.stem)
                except SnapshotIntegrityError:
                    corrupt_ids.append(entry.stem)


        return ReconcileResult(
            verified_ids=verified_ids,
            corrupt_ids=corrupt_ids,
            removed_staging_ids=removed_staging_ids,
        )
