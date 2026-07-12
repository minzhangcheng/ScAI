import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from backend.services.snapshot_v2.errors import SnapshotIntegrityError, SnapshotStorageError
from backend.services.snapshot_v2.models import ObjectRef, SnapshotManifestV2
from backend.utils.crypto import get_integrity_secret


CHUNK_SIZE = 1 * 1024 * 1024


def canonical_json(value: BaseModel | dict) -> bytes:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = dict(value)
    payload.pop("root_hash", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


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

    def put_bytes(self, data: bytes, media_type: str) -> ObjectRef:
        sha = _hash_bytes(data)
        dest = self._objects_dir / sha
        if dest.exists():
            actual_size = dest.stat().st_size
            actual_hash = _hash_file(dest)
            if actual_size != len(data) or actual_hash != sha:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Existing object {sha} has mismatched content",
                )
            return ObjectRef(sha256=sha, size=len(data), media_type=media_type)

        temp = self._paths.staging / f".{sha}.tmp"
        temp.write_bytes(data)
        temp_fd = os.open(temp, os.O_RDONLY)
        try:
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        os.replace(str(temp), str(dest))
        return ObjectRef(sha256=sha, size=len(data), media_type=media_type)

    def put_file(self, path: Path, media_type: str) -> ObjectRef:
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
        sha = _hash_file(resolved)
        dest = self._objects_dir / sha
        if dest.exists():
            actual_size = dest.stat().st_size
            if actual_size != resolved.stat().st_size:
                raise SnapshotIntegrityError(
                    "OBJECT_HASH_MISMATCH",
                    f"Existing object {sha} has mismatched size",
                )
            return ObjectRef(sha256=sha, size=resolved.stat().st_size, media_type=media_type)
        size = resolved.stat().st_size
        with tempfile.NamedTemporaryFile(dir=str(self._paths.staging), prefix=f".{sha}.", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with open(resolved, "rb") as src:
                while True:
                    chunk = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(str(tmp_path), str(dest))
        return ObjectRef(sha256=sha, size=size, media_type=media_type)

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
        obj_path = self._objects_dir / sha256
        if not obj_path.exists() or not obj_path.is_file():
            return False
        try:
            actual = _hash_file(obj_path)
        except OSError:
            return False
        return actual == sha256

    def verify(self, ref: ObjectRef) -> None:
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
        """HMAC-SHA256 using the host integrity secret (not session-id-derived)."""
        secret = get_integrity_secret()
        key = secret if isinstance(secret, bytes) else secret.encode()
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def _verify_signature(self, data: bytes, sig: str) -> bool:
        expected = self._sign(data)
        return hmac.compare_digest(expected, sig)

    def write_staging(self, manifest: SnapshotManifestV2, prefix: str = "") -> Path:
        raw = canonical_json(manifest)
        root_hash = hashlib.sha256(raw).hexdigest()
        manifest.root_hash = root_hash
        full = manifest.model_dump(mode="json")
        manifest_bytes = json.dumps(full, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        staging_path = self._paths.staging / f"{prefix}{manifest.snapshot_id}.json"
        staging_path.write_bytes(manifest_bytes)
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
        os.replace(str(staging_path), str(committed_path))
        sig = self._sign(manifest_bytes)
        sig_path = self._paths.integrity / f"{manifest.snapshot_id}.sig"
        sig_path.write_text(sig)
        return manifest

    def load(self, snapshot_id: str) -> SnapshotManifestV2:
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
        expected_root = hashlib.sha256(canonical_json(manifest)).hexdigest()
        if manifest.root_hash != expected_root:
            raise SnapshotIntegrityError(
                "MANIFEST_ROOT_HASH_MISMATCH",
                f"Root hash mismatch for snapshot {snapshot_id}: "
                f"expected {expected_root}, got {manifest.root_hash}",
            )
        return manifest

    def _iter_committed_json_stems(self) -> list[str]:
        """Sorted non-audit committed manifest stems."""
        committed_dir = self._paths.committed
        if not committed_dir.exists():
            return []
        stems: list[str] = []
        for entry in sorted(committed_dir.iterdir()):
            if entry.suffix != ".json" or not entry.is_file():
                continue
            if entry.stem.startswith("audit-"):
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
                if entry.is_file() or entry.is_symlink():
                    removed_staging_ids.append(entry.stem)
                    entry.unlink()

        verified_ids: list[str] = []
        corrupt_ids: list[str] = []
        committed_dir = self._paths.committed
        if committed_dir.exists():
            for entry in sorted(committed_dir.iterdir()):
                if entry.suffix != ".json" or not entry.is_file():
                    continue
                if entry.stem.startswith("audit-"):
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
