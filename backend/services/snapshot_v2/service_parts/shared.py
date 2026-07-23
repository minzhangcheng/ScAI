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


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_snapshot_id() -> str:
    return f"snap-{uuid.uuid4().hex[:16]}"


def _as_name_set(raw: Any) -> set[str]:
    """Normalize list/set/tuple/dict keys of dependency names into a set."""
    if not raw:
        return set()
    if isinstance(raw, dict):
        return {str(n) for n in raw if n}
    if isinstance(raw, (list, set, tuple)):
        return {str(n) for n in raw if n}
    return {str(raw)}


@contextlib.asynccontextmanager
async def _null_async_cm():
    yield


def _workflow_state_for_snapshot(workflow: Any) -> dict[str, Any]:
    """Read a detached workflow state, preferring the explicit contract."""
    if workflow is None:
        return {}
    accessor = getattr(workflow, "get_snapshot_workflow_state", None)
    if callable(accessor):
        raw = accessor()
        if isinstance(raw, dict):
            return copy.deepcopy(raw)
    # Compatibility fallback for test doubles and older in-process workflows.
    for attr in ("_current_graph_state", "_last_workflow_state"):
        raw = getattr(workflow, attr, None)
        if isinstance(raw, dict):
            return copy.deepcopy(raw)
    return {}


def _has_trusted_workflow_snapshot_context(
    workflow: Any,
    workflow_state: dict[str, Any],
    live_safe_point: dict[str, Any],
) -> bool:
    """Require observable state before freezing an active workflow."""
    return workflow is None or bool(workflow_state) or bool(live_safe_point)


