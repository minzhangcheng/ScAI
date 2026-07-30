# Read-only snapshot listing, preview, store status, and manifest rebuild.
# Moved verbatim out of backend/services/snapshot_v2/service.py; composed into SnapshotService.
"""Snapshot V2 capture service — atomic capture transaction.

Lock order (fixed):
  session_lock → safe-point acknowledgement → await_pending_saves → sandbox._exec_lock

Never acquire session_lock while holding _exec_lock.
"""

from __future__ import annotations

import asyncio
import copy
import contextlib
import hashlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config._config import default_config, get_session_states
from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.continuation import ContinuationRegistry
from backend.services.snapshot_v2.errors import (
    RequiredSnapshotBlocked,
    SnapshotCaptureError,
    SnapshotDeleteConflict,
    SnapshotStorageError,
    SnapshotV2Error,
)
from backend.services.snapshot_v2.models import (
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    RuntimeManifest,
    SnapshotCaptureRequest,
    SnapshotFailureRecord,
    SnapshotManifestV2,
    SnapshotRecordV2,
    SnapshotRestoreRequest,
    SnapshotTransactionRecord,
    WorkspaceEntry,
    empty_complete_runtime,
    runtime_manifest_from_generation,
    validate_system_commit,
)
from backend.services.snapshot_v2.object_store import (
    ObjectPutResult,
    ObjectStore,
    ManifestStore,
    canonical_json,
)
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.runtime import RuntimeCapture
from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker
from backend.services.snapshot_v2.safe_point import (
    SafePointManager,
    get_snapshot_safe_point_manager,
)
from backend.services.snapshot_v2.workspace import WorkspaceCapture
from backend.state import get_session_lock

logger = logging.getLogger(__name__)

from .shared import (  # noqa: F401
    asyncio,
    copy,
    contextlib,
    hashlib,
    json,
    logging,
    shutil,
    uuid,
    datetime,
    timezone,
    Path,
    Any,
    default_config,
    get_session_states,
    checkpoint_sqlite_path,
    ContinuationRegistry,
    RequiredSnapshotBlocked,
    SnapshotCaptureError,
    SnapshotDeleteConflict,
    SnapshotStorageError,
    SnapshotV2Error,
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    RuntimeManifest,
    SnapshotCaptureRequest,
    SnapshotFailureRecord,
    SnapshotManifestV2,
    SnapshotRecordV2,
    SnapshotRestoreRequest,
    SnapshotTransactionRecord,
    WorkspaceEntry,
    empty_complete_runtime,
    runtime_manifest_from_generation,
    validate_system_commit,
    ObjectPutResult,
    ObjectStore,
    ManifestStore,
    canonical_json,
    SnapshotV2Paths,
    RuntimeCapture,
    RuntimeDependencyTracker,
    SafePointManager,
    get_snapshot_safe_point_manager,
    WorkspaceCapture,
    get_session_lock,
    _now_iso,
    _new_snapshot_id,
    _as_name_set,
    _null_async_cm,
    _workflow_state_for_snapshot,
    _has_trusted_workflow_snapshot_context,
)


def _listing_sort_key(snap) -> datetime:
    """Order key that tolerates a listing mixing naive V1 and aware V2 records."""
    ts = snap.timestamp
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


class QueriesMixin:
    """Read-only snapshot listing, preview, store status, and manifest rebuild."""

    def compute_protected_snapshot_ids(self) -> dict[str, set[str]]:
        """Return snapshot IDs protected from deletion, keyed by reason.

        Reasons:
          live_head, continuation_source, restore_target, pre_restore,
          import, export

        Sources: committed session state (lineage head / last restored),
        verified create/restore/delete/import/export journals, and the
        in-process restore job registry.
        """
        from pathlib import Path as _Path

        protected: dict[str, set[str]] = {
            "live_head": set(),
            "continuation_source": set(),
            "restore_target": set(),
            "pre_restore": set(),
            "import": set(),
            "export": set(),
        }

        # Live lineage from in-memory session state
        try:
            from backend.state import sessions
            state = sessions.get(self.session_id)
        except Exception:
            state = None
        if state is not None:
            head = getattr(state, "last_snapshot_id", None)
            if head:
                protected["live_head"].add(str(head))
            from backend.workflow.provenance import restore_provenance_id
            cont = restore_provenance_id(state)
            if cont:
                protected["continuation_source"].add(str(cont))

        # In-process restore job registry (active non-terminal jobs)
        try:
            from backend.state import snapshot_restore_jobs
            terminal = {"committed", "failed", "cancelled", "timed_out"}
            for handle in list(snapshot_restore_jobs.values()):
                record = getattr(handle, "record", None)
                if record is None:
                    continue
                if getattr(record, "session_id", None) != self.session_id:
                    continue
                status = getattr(record, "status", None)
                if status in terminal:
                    continue
                sid = getattr(record, "snapshot_id", None)
                if sid:
                    protected["restore_target"].add(str(sid))
        except Exception:
            pass

        # Journals: restore / import / export / delete (signed envelopes)
        journals_dir = _Path(str(self.paths.journals))
        if journals_dir.exists():
            from backend.services.snapshot_v2.journal_store import JournalStore

            store = JournalStore(self.paths)
            for entry in journals_dir.iterdir():
                if not entry.is_file() or entry.suffix != ".json":
                    continue
                try:
                    env = store.read(entry.stem)
                except Exception:
                    # Fall back to raw parse for incomplete fixtures only
                    try:
                        data = json.loads(entry.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(data, dict):
                        continue
                    # Minimal legacy fallback for unsigned fixtures in tests
                    kind = data.get("kind")
                    payload = data if kind else {}
                    status = str(data.get("status") or "").upper()
                    if kind == "export_protection":
                        for sid in data.get("protected_snapshot_ids") or []:
                            if sid:
                                protected["export"].add(str(sid))
                    elif kind == "import" and status not in ("COMMITTED", "FAILED"):
                        for sid in data.get("protected_snapshot_ids") or []:
                            if sid:
                                protected["import"].add(str(sid))
                    elif data.get("pre_restore_ref") or data.get("target_snapshot_id"):
                        if data.get("pre_restore_ref"):
                            protected["pre_restore"].add(str(data["pre_restore_ref"]))
                        t = data.get("target_snapshot_id")
                        if t:
                            protected["restore_target"].add(str(t))
                    continue

                if env.session_id and env.session_id != self.session_id:
                    continue
                payload = env.payload or {}
                status = str(env.status or "").upper()
                st_lower = str(env.status or "").lower()

                if env.kind == "export_protection":
                    for sid in payload.get("protected_snapshot_ids") or []:
                        if sid:
                            protected["export"].add(str(sid))
                    continue

                if env.kind == "import":
                    if status in ("COMMITTED", "FAILED"):
                        continue
                    for sid in payload.get("protected_snapshot_ids") or []:
                        if sid:
                            protected["import"].add(str(sid))
                    for sid in payload.get("committed_snapshot_ids") or []:
                        if sid:
                            protected["import"].add(str(sid))
                    continue

                if env.kind == "restore_apply":
                    if st_lower in ("cancelled", "timed_out"):
                        continue
                    if st_lower == "committed" and not payload.get("pre_restore_ref"):
                        continue
                    target = payload.get("target_snapshot_id")
                    if target:
                        protected["restore_target"].add(str(target))
                    pre = payload.get("pre_restore_ref")
                    if pre:
                        protected["pre_restore"].add(str(pre))
                    continue

                if env.kind == "restore_job":
                    if st_lower in ("committed", "failed", "cancelled", "timed_out"):
                        continue
                    sid = payload.get("snapshot_id")
                    if sid:
                        protected["restore_target"].add(str(sid))
                    continue

                if env.kind == "delete":
                    # Active delete does not protect delete_set from this path.
                    continue

        return protected

    def delete_snapshot(self, snapshot_id: str, *, branch: bool = False) -> Any:
        """Delete a snapshot with live-lineage and active-transaction protection.

        Raises ``SnapshotDeleteConflict`` with typed ``details["reason"]`` for
        HTTP 409 consumers.
        """
        from backend.services.snapshot_v2.deletion import DeletionService

        by_reason = self.compute_protected_snapshot_ids()
        for reason, ids in by_reason.items():
            if snapshot_id in ids:
                raise SnapshotDeleteConflict(
                    "PROTECTED",
                    f"Snapshot {snapshot_id} is protected ({reason}) and cannot be deleted",
                    details={
                        "reason": reason,
                        "snapshot_id": snapshot_id,
                        "protection": reason,
                    },
                )

        protected_flat: set[str] = set()
        for ids in by_reason.values():
            protected_flat |= ids

        deletion = DeletionService(
            self.session_id,
            self.paths,
            manifest_store=self.manifest_store,
            object_store=self.object_store,
        )
        if branch:
            all_manifests = self.manifest_store.verify_all_committed()
            children: dict[str, list[str]] = {}
            for m in all_manifests:
                pid = m.parent_id
                if pid:
                    children.setdefault(pid, []).append(m.snapshot_id)
            descendant_ids: list[str] = []
            stack = list(children.get(snapshot_id, []))
            while stack:
                sid = stack.pop()
                descendant_ids.append(sid)
                stack.extend(children.get(sid, []))
            for sid in descendant_ids:
                for reason, ids in by_reason.items():
                    if sid in ids:
                        raise SnapshotDeleteConflict(
                            "PROTECTED",
                            f"Snapshot {sid} is protected ({reason}) and cannot be deleted",
                            details={
                                "reason": reason,
                                "snapshot_id": sid,
                                "protection": reason,
                            },
                        )
            return deletion.delete_branch(
                snapshot_id, descendant_ids, protected_ids=protected_flat,
            )
        return deletion.delete_leaf(snapshot_id, protected_ids=protected_flat)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List committed snapshots with typed integrity diagnostics.

        Never silently drops a corrupt committed manifest: each entry is either
        status ``ok`` (with full manifest) or ``corrupt`` (with error code).
        """
        entries = self.manifest_store.list_committed_with_diagnostics()
        out: list[dict[str, Any]] = []
        for e in entries:
            if e.status == "ok" and e.manifest is not None:
                flat = e.manifest.model_dump(mode="json")
                flat["status"] = "ok"
                flat["integrity_status"] = "ok"
                out.append(flat)
            else:
                out.append({
                    "snapshot_id": e.snapshot_id,
                    "status": "corrupt",
                    "integrity_status": "corrupt",
                    "error_code": e.error_code,
                    "error_message": e.error_message,
                    "manifest": None,
                })
        return out

    def preview_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Load a single verified V2 snapshot manifest for detail views."""
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError

        try:
            manifest = self.manifest_store.load(snapshot_id)
        except SnapshotIntegrityError:
            raise
        return {"manifest": manifest.model_dump(mode="json")}

    def store_status(self) -> Any:
        """Compute store-level usage statistics from verified manifests."""
        from backend.services.snapshot_v2.management import compute_store_status

        return compute_store_status(self.paths, self.manifest_store)

    def rebuild_snapshots_from_manifests(self, existing: list | None = None) -> list:
        """Rebuild AnalysisState.snapshots listing from verified committed manifests.

        Authoritative for V2 projections only. Legacy V1 records (``format_version``
        null/1) predate the store and have no committed manifest by construction,
        so a listing rebuilt from manifests alone silently deletes them — and with
        them the only handle ``legacy.build_legacy_inventory`` and the V1 delete
        routes have on sidecars under ``sessions/<id>/snapshots/`` that V2 GC never
        sweeps. They are carried over from *existing*; V2 entries never are, so a
        projection whose manifest is gone still disappears.

        *existing* defaults to the cached session listing because the capture and
        delete call sites rebuild in place on the cached state and pass nothing —
        without the fallback each of them would re-drop what the load path kept.
        """
        from backend.models.schemas import Snapshot
        from datetime import datetime as _dt

        from backend.services.snapshot_v2.legacy import _is_v1_snapshot

        if existing is None:
            from backend.state import sessions

            cached = sessions.get(self.session_id)
            existing = getattr(cached, "snapshots", None)
            if not isinstance(existing, list):
                existing = None

        rebuilt = [snap for snap in (existing or []) if _is_v1_snapshot(snap)]
        for manifest in self.manifest_store.verify_all_committed():
            try:
                ts = _dt.fromisoformat(
                    manifest.created_at.replace("Z", "+00:00")
                )
            except Exception:
                ts = _dt.now()
            cont_kind = None
            cont = manifest.continuation
            if cont is not None:
                cont_kind = getattr(cont, "kind", None)
            rebuilt.append(
                Snapshot(
                    id=manifest.snapshot_id,
                    name=manifest.name or manifest.snapshot_id,
                    description=manifest.description or "",
                    phase=manifest.phase,
                    round_number=manifest.round_number,
                    timestamp=ts,
                    parent_id=manifest.parent_id,
                    branch_label=manifest.branch_label,
                    branch_id=manifest.branch_id,
                    size_bytes=manifest.logical_size_bytes,
                    format_version=2,
                    runtime_status=manifest.runtime.status if manifest.runtime else None,
                    logical_size_bytes=manifest.logical_size_bytes,
                    physical_added_bytes=manifest.physical_added_bytes,
                    source_restore_transaction_id=manifest.source_restore_transaction_id,
                    continuation_kind=cont_kind,
                    runtime_state_included=bool(
                        manifest.runtime
                        and (
                            manifest.runtime.adata_refs
                            or manifest.runtime.residual_ref is not None
                        )
                    ),
                )
            )
        # Sort by created_at ascending for stable lineage listing. V2 projections
        # carry tz-aware timestamps (manifest ISO strings end in Z) while V1
        # records were persisted naive; Python refuses to order the two, so a
        # mixed list would raise here and take the whole reconcile down.
        rebuilt.sort(key=_listing_sort_key)
        return rebuilt

    def _resolve_previous_manifest(
        self, analysis_state: Any,
    ) -> SnapshotManifestV2 | None:
        """Resolve and verify the current V2 lineage head, when declared."""
        if analysis_state is None:
            return None
        snapshot_id = str(
            getattr(analysis_state, "last_snapshot_id", None) or ""
        ).strip()
        if not snapshot_id:
            return None
        return self.manifest_store.load(snapshot_id)

    def _find_committed_by_idempotency(
        self, request: SnapshotCaptureRequest,
    ) -> SnapshotManifestV2 | None:
        key = getattr(request, "idempotency_key", None)
        if not key:
            return None
        phase = request.phase or request.reason or "manual"
        for manifest in self.manifest_store.verify_all_committed():
            if (
                manifest.session_id == request.session_id
                and (manifest.phase or "") == phase
                and getattr(manifest, "idempotency_key", None) == key
            ):
                return manifest
        return None
