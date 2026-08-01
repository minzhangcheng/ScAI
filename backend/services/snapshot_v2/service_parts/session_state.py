# Safe-point ownership, session state load/save, and websocket state sync.
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


class SessionStateMixin:
    """Safe-point ownership, session state load/save, and websocket state sync."""

    async def _create_safe_point(self, workflow_state: dict[str, Any] | None = None) -> None:
        # Capture lifecycle may store graph-position metadata, but must NOT
        # authorize empty-complete from untrusted capture args.
        # Token is process-local for this service instance's lifecycle hold.
        owner_token = f"capture:{self.session_id}:{id(self)}"
        self._safe_point_owner_token = owner_token
        acquired = await self.safe_point.acquire(
            self.session_id,
            workflow_state,
            eligibility_trusted=False,
            owner_token=owner_token,
        )
        if not acquired:
            self._safe_point_owner_token = None
            raise SnapshotCaptureError(
                "SAFE_POINT_BUSY",
                "Another snapshot or restore operation owns the session safe point",
            )

    async def _release_safe_point(self) -> None:
        owner_token = getattr(self, "_safe_point_owner_token", None) or (
            f"capture:{self.session_id}:{id(self)}"
        )
        await self.safe_point.release(self.session_id, owner_token)
        self._safe_point_owner_token = None

    def _guard_active_safe_point(self) -> None:
        """Raise SnapshotTransactionError if a safe point is active.

        Called by mutation methods (import, delete) before acquiring the
        session lock to prevent concurrent snapshot/restore operations
        from corrupting the session state.
        """
        from backend.services.snapshot_v2.errors import SnapshotTransactionError

        if self.safe_point.is_paused(self.session_id):
            raise SnapshotTransactionError(
                "SAFE_POINT_ACTIVE",
                f"A snapshot/restore operation is in progress for "
                f"session {self.session_id}; rejecting mutation",
            )

    def _has_active_workflow_ownership(self) -> bool:
        """True when this session owns a live workflow in the process map."""
        try:
            from backend.state import workflows
            return self.session_id in workflows and workflows.get(self.session_id) is not None
        except Exception:
            return False

    async def _save_session_state(self, session_id: str) -> None:
        from backend.services.session_store import async_save_session
        state = await self._get_session_state(session_id)
        if state is not None:
            await async_save_session(session_id, state)

    async def _get_session_state(self, session_id: str) -> Any:
        from backend.state import sessions
        return sessions.get(session_id)

    async def _reconcile_analysis_state_snapshots(self) -> None:
        """Rebuild AnalysisState.snapshots from verified manifests and persist."""
        from backend.services.session_store import async_save_session, update_session_cache
        from backend.state import sessions

        state = sessions.get(self.session_id)
        if state is None:
            return
        try:
            state.snapshots = self.rebuild_snapshots_from_manifests()
        except Exception as exc:
            logger.warning(
                "[%s] rebuild snapshots during reconcile failed: %s",
                self.session_id, exc,
            )
            return
        try:
            await update_session_cache(self.session_id, state)
            await async_save_session(self.session_id, state)
        except Exception as exc:
            logger.warning(
                "[%s] persist after snapshot reconcile failed: %s",
                self.session_id, exc,
            )

    def _to_failure_record(
        self,
        snapshot_id: str,
        session_id: str,
        error: SnapshotV2Error,
    ) -> SnapshotRecordV2:
        stub_ref = ObjectRef(
            sha256="0" * 64, size=0, media_type="application/octet-stream",
        )
        stub_manifest = SnapshotManifestV2(
            session_id=session_id,
            snapshot_id=snapshot_id,
            name=snapshot_id,
            description="capture-failure-stub",
            phase="manual",
            created_at=_now_iso(),
            analysis_state_ref=stub_ref,
            notebook_ref=stub_ref,
            runtime=empty_complete_runtime(),
            continuation=IdleCapsule(graph_version="2.0"),
            logical_size_bytes=0,
            physical_added_bytes=0,
        )
        return SnapshotRecordV2(
            snapshot_id=snapshot_id,
            manifest=stub_manifest,
            failure=SnapshotFailureRecord(
                snapshot_id=snapshot_id,
                session_id=session_id,
                failed_at=_now_iso(),
                code=error.code,
                message=str(error),
                details=error.details,
                # allowed_actions is deliberately not passed: the record derives
                # it from the code (see SnapshotFailureRecord). Hardcoding the
                # base set here made a memory-preflight refusal indistinguishable
                # from an ordinary one, and pre_restore hands this record
                # straight to the failure gate.
            ),
            committed=False,
            session_reconciliation_pending=False,
        )

    async def _send_state_sync(self, payload: dict[str, Any]) -> None:
        """Emit a state_sync message for the snapshot failure gate."""
        from backend.state import websockets
        ws = websockets.get(self.session_id)
        if ws is not None:
            try:
                await ws.send_json({
                    "type": "state_sync",
                    "change_type": payload.get("change_type", "snapshot_failure"),
                    **payload,
                })
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to send snapshot failure state sync: %s",
                    self.session_id, exc,
                )
