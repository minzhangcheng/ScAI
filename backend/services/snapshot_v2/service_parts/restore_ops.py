# Restore entry point, post-apply persistence, and partial-apply failure handling.
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


class RestoreOpsMixin:
    """Restore entry point, post-apply persistence, and partial-apply failure handling."""

    async def _persist_restored_state(self, restored_state: Any) -> None:
        """Write restored analysis state to the live session store."""
        from backend.state import sessions

        sessions[self.session_id] = restored_state
        if self.session_persistence is not None:
            await self.session_persistence(self.session_id, restored_state)
        else:
            from backend.services.session_store import async_save_session
            await async_save_session(self.session_id, restored_state)

    def _block_session_after_partial_apply(self, failure: dict[str, Any]) -> None:
        """Guard the in-process session after a post-persist restore failure.

        Once ``_persist_restored_state`` has run, ``sessions[session_id]`` holds
        the target state while the restore journal is still ``APPLYING`` and the
        transaction failed. Mark the session recovery-blocked so forward
        mutation is refused until startup recovery rolls back (or, for
        ``forward_only``, rolls forward) on the next session load — the memory
        state must not be silently trusted as a successful restore.
        """
        try:
            from backend.state import sessions
            from backend.services.session_store import mark_recovery_blocked

            mark_recovery_blocked(
                self.session_id, sessions.get(self.session_id), failure,
            )
        except Exception:
            logger.exception(
                "[%s] failed to mark recovery-blocked after partial restore apply",
                self.session_id,
            )

    def _partial_apply_failure(self, code: str, message: str) -> dict[str, Any]:
        """Build a recovery-required failure and block live session mutation."""
        failure = {
            "code": str(code),
            "message": str(message),
            "recovery_required": True,
        }
        self._block_session_after_partial_apply(failure)
        return failure

    async def _verify_live_restore(
        self,
        *,
        target_manifest: Any,
        restored_state: Any,
        staged_checkpoint: Any,
        workspace_root: Path,
        session_dir: Path,
        states_root: Path | None = None,
        runtime_choice: str | None = None,
    ) -> dict[str, Any]:
        """Verify live restore side effects while the journal is still APPLYING.

        Raises ``SnapshotTransactionError("LIVE_VERIFY_FAILED", ...)`` on any
        mismatch. Returns a verification proof dict for the COMMITTED journal.

        Uses the shared contract in ``live_verify.verify_live_restore_components``:
        production residual is ``{session_id}_session.pkl`` + integrity signature
        (not session_state.pkl / residual.pkl); session.json is presence-checked
        only (post-activate state is continuation-mutated).
        """
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.services.snapshot_v2.live_verify import (
            verify_live_restore_components,
        )

        states = Path(states_root) if states_root is not None else session_dir / "states"
        try:
            return verify_live_restore_components(
                session_id=self.session_id,
                target_manifest=target_manifest,
                session_dir=session_dir,
                workspace_root=workspace_root,
                states_root=states,
                runtime_choice=runtime_choice,
                restored_state=restored_state,
                staged_checkpoint=staged_checkpoint,
                journal=None,
            )
        except ValueError as exc:
            raise SnapshotTransactionError(
                "LIVE_VERIFY_FAILED",
                str(exc) or "live side effects not verified",
            ) from exc
        except SnapshotTransactionError:
            raise
        except Exception as exc:
            raise SnapshotTransactionError(
                "LIVE_VERIFY_FAILED",
                str(exc) or "live side effects not verified",
            ) from exc

    async def _capture_locked_for_restore(
        self,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
        workflow_state: dict[str, Any] | None = None,
        analysis_state: Any = None,
        snapshot_id_override: str | None = None,
    ) -> SnapshotRecordV2:
        """Capture for restore (caller already holds session lock + safe point).

        Always skips cooperative safe-point create/release so the restore
        transaction's already-held safe point is retained for the full apply.
        Does not persist session state — restore owns final session persistence
        after apply (avoids re-entrant save while the session lock is held).
        """
        return await self._capture_locked(
            request,
            sandbox=sandbox,
            workspace_root=workspace_root,
            workflow_state=workflow_state,
            analysis_state=analysis_state,
            snapshot_id_override=snapshot_id_override,
            skip_safe_point_lifecycle=True,
            persist_session=False,
        )
