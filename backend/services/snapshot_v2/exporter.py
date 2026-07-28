"""V2 snapshot export with explicit runtime policy.

Produces a tar.gz archive of verified snapshot objects.  Base/workspace
objects always export; runtime objects only with ``include_runtime=True``.

Stream tar emission (no full buffer). Uses ``open_verified`` context
manager for memory-efficient object streaming.
"""

import contextlib
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
from typing import Any

from backend.services.snapshot_v2.errors import SnapshotV2Error
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.models import (
    SnapshotManifestV2,
)

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 * 1024 * 1024


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
    protected: set[str] = set()
    journals_dir = Path(str(paths.journals))
    if not journals_dir.exists():
        return protected
    from backend.services.snapshot_v2.journal_store import JournalStore

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
        except Exception:
            # Skip any journal that fails to read/verify (incomplete fixtures
            # or integrity failures) — protection is best-effort here.
            continue
        if env.kind != "export_protection":
            continue
        protected.update((env.payload or {}).get("protected_snapshot_ids") or [])
    return protected


def load_active_export_protected_by_reason(paths) -> set[str]:
    return load_active_export_protected_ids(paths)


# ── Streaming verified reader ────────────────────────────────────────────


@contextlib.contextmanager
def open_verified(path: Path, sha256: str):
    """Context manager: yield a file-like reader that validates SHA-256 at EOF.

    Usage::

        with open_verified(obj_path, expected_hash) as reader:
            while True:
                chunk = reader.read(CHUNK_SIZE)
                if not chunk:
                    break
                # process chunk

    Raises ``SnapshotIntegrityError`` with code ``OBJECT_HASH_MISMATCH``
    if the hash does not match *sha256* upon exhaustion.
    """
    if not path.exists() or not path.is_file():
        raise ExportIntegrityError(
            "OBJECT_NOT_FOUND",
            f"Object {sha256} not found at {path}",
        )

    class _HashingReader(io.RawIOBase):
        def __init__(self, filepath: Path):
            self._fp = open(filepath, "rb")
            self._hasher = hashlib.sha256()
            self._exhausted = False

        def readable(self) -> bool:
            return True

        def readinto(self, b: bytearray) -> int | None:
            n = self._fp.readinto(b)
            if n is None or n == 0:
                if not self._exhausted:
                    self._exhausted = True
                    actual = self._hasher.hexdigest()
                    if actual != sha256:
                        raise ExportIntegrityError(
                            "OBJECT_HASH_MISMATCH",
                            f"Object {sha256} hash mismatch (actual: {actual})",
                        )
                return n
            self._hasher.update(b[:n])
            return n

        def close(self) -> None:
            if not self._exhausted and not self._fp.closed:
                remaining = self._fp.read()
                if remaining:
                    self._hasher.update(remaining)
                self._exhausted = True
                actual = self._hasher.hexdigest()
                if actual != sha256:
                    raise ExportIntegrityError(
                        "OBJECT_HASH_MISMATCH",
                        f"Object {sha256} hash mismatch on close (actual: {actual})",
                    )
            self._fp.close()

    reader = _HashingReader(path)
    try:
        yield reader
    finally:
        if not reader.closed:
            reader.close()


# ── Object classification ───────────────────────────────────────────────


def _classify_manifest_and_collect(
    manifest: SnapshotManifestV2,
) -> tuple[set[str], set[str]]:
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
    """Export verified snapshot V2 objects into a tar.gz archive (streaming).

    Usage::

        exporter = SnapshotExporter(session_id, paths)
        path = exporter.export(session_id, include_runtime=False)

    Objects are streamed into the archive via ``open_verified`` context
    manager — no full object data is buffered in memory.
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
        """Build a tar.gz archive of verified snapshot objects (streaming).

        Returns the path to the archive on disk.  The caller is responsible
        for cleaning up the returned file.
        """
        manifests = self.manifest_store.verify_all_committed()
        if snapshot_ids is not None:
            snapshot_set = set(snapshot_ids)
            manifests = [m for m in manifests if m.snapshot_id in snapshot_set]

        if not manifests:
            raise ExportIntegrityError("NO_MANIFESTS", "No snapshots to export")

        if snapshot_ids is not None:
            found_ids = {m.snapshot_id for m in manifests}
            missing = [sid for sid in snapshot_ids if sid not in found_ids]
            if missing:
                raise ExportIntegrityError(
                    "UNKNOWN_SNAPSHOT_IDS",
                    f"Snapshot IDs not found: {', '.join(missing)}",
                )

        base_hashes: set[str] = set()
        runtime_only_hashes: set[str] = set()
        all_snapshot_ids: list[str] = []

        for m in manifests:
            all_snapshot_ids.append(m.snapshot_id)
            bh, ro = _classify_manifest_and_collect(m)
            base_hashes.update(bh)
            runtime_only_hashes.update(ro)

        included_hashes: set[str] = set(base_hashes)
        omitted_runtime_hashes: list[str] = sorted(runtime_only_hashes)

        if include_runtime:
            included_hashes.update(runtime_only_hashes)

        omitted_runtime = sorted(runtime_only_hashes) if not include_runtime else []

        export_id = _new_export_id()
        _write_export_protection(self.paths, export_id, session_id, all_snapshot_ids)

        tmp_path = None
        exported = False
        try:
            # Verify all included objects (uses evidence cache when available)
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

            # Build tar.gz archive with streaming emission
            fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz", prefix=f"v2-export-{export_id}-")
            os.close(fd)

            total_object_size = 0
            object_count = 0

            with tarfile.open(tmp_path, mode="w:gz") as tf:
                # 6a. Stream object files (by hash) using open_verified
                for h in sorted(included_hashes):
                    obj_path = self.paths.objects / h
                    if not obj_path.exists():
                        raise ExportIntegrityError(
                            "OBJECT_NOT_FOUND",
                            f"Object {h} not found in store",
                        )

                    with open_verified(obj_path, h) as reader:
                        info = tarfile.TarInfo(name=f"objects/{h}")
                        info.size = obj_path.stat().st_size
                        tf.addfile(info, reader)
                    total_object_size += info.size
                    object_count += 1

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
                metadata = {
                    "format_version": 2,
                    "session_id": session_id,
                    "export_id": export_id,
                    "exported_at": _now_iso(),
                    "runtime_included": include_runtime,
                    "included_snapshot_ids": all_snapshot_ids,
                    "included_hashes": sorted(included_hashes),
                    "omitted_runtime_hashes": omitted_runtime,
                    "object_count": object_count,
                    "manifest_count": len(manifests),
                    "total_size_bytes": total_object_size,
                }
                meta_bytes = json.dumps(
                    metadata, sort_keys=True, ensure_ascii=False,
                ).encode()
                meta_info = tarfile.TarInfo(name="snapshot-v2-export.json")
                meta_info.size = len(meta_bytes)
                tf.addfile(meta_info, io.BytesIO(meta_bytes))

            exported = True
            return Path(tmp_path)

        finally:
            _remove_export_protection(self.paths, export_id)
            # The archive is a multi-GB temp file in the system temp dir. Any
            # failure after mkstemp -- a missing object (ExportIntegrityError),
            # a hash mismatch, ENOSPC mid-write -- propagates to the route,
            # which turns it into an HTTP error and never sees tmp_path. Only
            # the success path moves the file somewhere owned, so without this
            # every failed export (and every user retry of it) leaves another
            # partial archive behind until the host fills up -- which then also
            # breaks session-metadata writes, since those use mkstemp too.
            if not exported and tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning(
                        "Could not remove partial export archive %s", tmp_path,
                    )
