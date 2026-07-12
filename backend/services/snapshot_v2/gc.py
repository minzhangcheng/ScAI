"""Mark-and-sweep garbage collection for Snapshot V2.

Collects all object hashes referenced by surviving committed manifests
and active transaction journals (mark), then removes hash-named files
in the ``objects/`` directory that are not in the marked set (sweep).

Safety: if any surviving manifest fails verification, sweep is aborted
with ``SnapshotIntegrityError`` — never guess.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GCResult:
    removed_count: int
    removed_bytes: int
    surviving_count: int


class SnapshotGarbageCollector:
    """Mark-and-sweep garbage collector for snapshot object storage."""

    def __init__(
        self,
        session_id: str,
        paths,
        manifest_store: ManifestStore | None = None,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.session_id = session_id
        self.paths = paths
        self.manifest_store = manifest_store or ManifestStore(paths)
        self.object_store = object_store or ObjectStore(paths)

    def mark(self) -> set[str]:
        """Collect all object hashes referenced by surviving manifests.

        Returns the set of all hash strings referenced by every
        verifiable committed manifest (excluding ``audit-*`` manifests)
        plus any object references found in active transaction journals.

        Restore journals are HMAC-verified via
        ``read_verified_restore_journal``. Tampered or unreadable restore
        journals fail closed with ``SnapshotIntegrityError`` (no sweep).
        """
        from backend.services.snapshot_v2.journal_store import JournalStore
        from backend.services.snapshot_v2.recovery import RecoveryFailure

        marked: set[str] = set()

        # 1. Collect from every verified committed manifest (raises on corrupt)
        for manifest in self.manifest_store.verify_all_committed():
            marked.update(manifest.objects.keys())

        # 2. Collect from active transaction journals (verified envelopes)
        journals_dir = Path(str(self.paths.journals))
        store = JournalStore(self.paths)
        if journals_dir.exists():
            for jp in journals_dir.iterdir():
                if not jp.is_file() or jp.suffix != ".json":
                    continue
                tx_id = jp.stem
                try:
                    env = store.read(tx_id)
                except RecoveryFailure as exc:
                    raise SnapshotIntegrityError(
                        "JOURNAL_HMAC",
                        f"Active transaction journal unreadable/unsigned: "
                        f"{jp.name}: {exc}",
                        details={"path": str(jp), "transaction_id": tx_id},
                    ) from exc
                except SnapshotIntegrityError:
                    raise
                payload = env.payload or {}
                # object_refs may live in payload (restore_apply) or top-level legacy
                obj_refs = payload.get("object_refs") or {}
                if isinstance(obj_refs, dict):
                    for sha256 in obj_refs.keys():
                        if _HASH_RE.match(str(sha256)):
                            marked.add(str(sha256))
                # Import STAGED object_hashes
                for h in payload.get("object_hashes") or []:
                    if _HASH_RE.match(str(h)):
                        marked.add(str(h))

        return marked

    def _verify_surviving_manifests(self) -> None:
        """Verify every committed manifest. Raises on first failure."""
        self.manifest_store.verify_all_committed()

    def sweep(self, marked: set[str]) -> GCResult:
        """Remove hash-named files in ``objects/`` not in *marked*.

        Safety check: every committed manifest is verified first via
        ``ManifestStore.load()`` (HMAC + root hash). If any fails,
        sweep is aborted with ``SnapshotIntegrityError`` and no files
        are removed.

        Only removes regular files whose names are 64-char hex strings.
        """
        self._verify_surviving_manifests()

        # Find all hash-named files in objects dir
        objects_dir = Path(str(self.paths.objects))
        if not objects_dir.exists():
            return GCResult(removed_count=0, removed_bytes=0, surviving_count=0)

        removed_count = 0
        removed_bytes = 0
        surviving_count = 0

        for entry in list(objects_dir.iterdir()):
            if entry.is_file() and _HASH_RE.match(entry.name):
                if entry.name not in marked:
                    removed_bytes += entry.stat().st_size
                    entry.unlink()
                    removed_count += 1
                else:
                    surviving_count += 1

        return GCResult(
            removed_count=removed_count,
            removed_bytes=removed_bytes,
            surviving_count=surviving_count,
        )

    def run(self) -> GCResult:
        """Mark then sweep — one operation."""
        marked = self.mark()
        return self.sweep(marked)
