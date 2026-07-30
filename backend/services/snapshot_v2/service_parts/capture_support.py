# Capture orchestration helpers: locking, validation, quota, hashing, cleanup.
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


class CaptureSupportMixin:
    """Capture orchestration helpers: locking, validation, quota, hashing, cleanup."""

    async def _capture_locked(
        self,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
        transaction: str | None = None,
        workflow_state: dict[str, Any] | None = None,
        snapshot_id_override: str | None = None,
        analysis_state: Any = None,
        safe_point_owner_token: str | None = None,
        skip_safe_point_lifecycle: bool = False,
        persist_session: bool = True,
    ) -> SnapshotRecordV2:
        """Capture transaction (caller already holds the session lock).

        ``physical_added_bytes`` prefers ``ObjectPutResult.created`` for
        refs produced by ``put_bytes``/``put_file``. Bare ``ObjectRef``
        values (workspace / runtime / continuation) are counted only when
        their hash was absent from the object store before this capture
        (pre-scan of ``paths.objects``). Never use
        ``register_existing_ref()`` return value for quota decisions.
        """
        snapshot_id = snapshot_id_override or _new_snapshot_id()
        tx = transaction or snapshot_id
        try:
            if safe_point_owner_token:
                if not self.safe_point.is_owned_by(
                    request.session_id, safe_point_owner_token,
                ):
                    raise SnapshotCaptureError(
                        "SAFE_POINT_OWNERSHIP_LOST",
                        "Capture no longer owns the supplied session safe point",
                    )
                skip_safe_point_lifecycle = True
            manifest, pending = await self._capture_impl(
                request, snapshot_id, tx,
                sandbox=sandbox,
                workspace_root=workspace_root,
                workflow_state=workflow_state,
                analysis_state=analysis_state,
                skip_safe_point_lifecycle=skip_safe_point_lifecycle,
                persist_session=persist_session,
            )
            return SnapshotRecordV2(
                snapshot_id=manifest.snapshot_id,
                manifest=manifest,
                failure=None,
                committed=True,
                session_reconciliation_pending=pending,
            )
        except SnapshotV2Error as exc:
            logger.warning(
                "[snapshot %s] capture failed: %s — %s",
                snapshot_id, exc.code, exc,
            )
            return self._to_failure_record(snapshot_id, request.session_id, exc)
        except Exception as exc:
            logger.exception(
                "[snapshot %s] unexpected capture error: %s", snapshot_id, exc,
            )
            return self._to_failure_record(
                snapshot_id, request.session_id,
                SnapshotCaptureError("UNEXPECTED", str(exc)),
            )

    async def _capture_and_apply(
        self,
        state: Any,
        request: SnapshotCaptureRequest,
        *,
        sandbox: Any = None,
        workspace_root: str | None = None,
    ) -> SnapshotRecordV2:
        """Capture and apply result to the AnalysisState.

        Unlike ``capture_required_snapshot``, this does not raise
        RequiredSnapshotBlocked (used by unit tests that inspect failure
        fields in-place). Production producers must use
        ``capture_required_snapshot``.
        """
        from backend.services.snapshot_v2.errors import SNAPSHOT_FAILURE_ACTIONS

        if workspace_root is None:
            try:
                from backend.config._config import get_session_workspace
                workspace_root = str(get_session_workspace(request.session_id))
            except Exception:
                workspace_root = None

        record = await self.capture(
            request,
            sandbox=sandbox,
            workspace_root=workspace_root,
            analysis_state=state,
        )
        if record.failure is not None:
            failure_payload = record.failure.model_dump(mode="json")
            failure_payload["allowed_actions"] = list(SNAPSHOT_FAILURE_ACTIONS)
            state.snapshot_failure = failure_payload
            state.snapshot_transaction = {
                "snapshot_id": record.snapshot_id,
                "status": "failed",
                "failed_at": _now_iso(),
            }
            state.wait_input_reason = "snapshot_failure"
            await self._send_state_sync({
                "change_type": "snapshot_failure",
                "version": getattr(state, "state_version", 0),
                "status": getattr(state, "status", None),
                "snapshot_failure": failure_payload,
                "snapshot_transaction": state.snapshot_transaction,
            })
        else:
            state.snapshot_failure = None
            state.snapshot_transaction = {
                "snapshot_id": record.snapshot_id,
                "status": "completed",
                "completed_at": _now_iso(),
            }
            if state.wait_input_reason == "snapshot_failure":
                state.wait_input_reason = None
            try:
                state.snapshots = self.rebuild_snapshots_from_manifests()
                state.last_snapshot_id = record.snapshot_id
            except Exception as exc:
                logger.warning(
                    "[%s] rebuild snapshots after capture failed: %s",
                    self.session_id, exc,
                )
        return record

    def _clear_failure(self, state: Any) -> None:
        """Clear the failure gate state for a retry."""
        state.snapshot_failure = None
        state.snapshot_transaction = None
        if state.wait_input_reason == "snapshot_failure":
            state.wait_input_reason = None

    def _validate_state(self, request: SnapshotCaptureRequest) -> None:
        if not request.session_id:
            raise SnapshotCaptureError(
                "MISSING_SESSION_ID", "session_id is required",
            )
        if request.session_id != self.session_id:
            raise SnapshotCaptureError(
                "SESSION_ID_MISMATCH",
                f"Request session_id '{request.session_id}' "
                f"does not match service session_id '{self.session_id}'",
            )

    async def _compute_root_hash(self, manifest: SnapshotManifestV2) -> None:
        raw = canonical_json(manifest)
        manifest.root_hash = hashlib.sha256(raw).hexdigest()

    async def _check_quota(
        self,
        manifest: SnapshotManifestV2,
        *,
        quota_baseline_bytes: int | None = None,
        allow_over_quota_baseline: bool = False,
    ) -> None:
        # Session quota counts physical additions for this capture.
        added = manifest.physical_added_bytes
        addition_ceiling = self.session_quota_bytes
        if (
            allow_over_quota_baseline
            and quota_baseline_bytes is not None
            and quota_baseline_bytes > self.session_quota_bytes
        ):
            addition_ceiling = self.pre_restore_quota_grace_bytes
        if added > addition_ceiling:
            raise SnapshotStorageError(
                "snapshot_quota_exceeded",
                f"Snapshot physical addition {added} exceeds quota "
                f"{addition_ceiling}",
            )
        cumulative = self._physical_object_store_bytes()
        cumulative_ceiling = self.session_quota_bytes
        if (
            allow_over_quota_baseline
            and quota_baseline_bytes is not None
            and quota_baseline_bytes > self.session_quota_bytes
        ):
            cumulative_ceiling = (
                quota_baseline_bytes + self.pre_restore_quota_grace_bytes
            )
        if cumulative > cumulative_ceiling:
            raise SnapshotStorageError(
                "snapshot_quota_exceeded",
                f"Snapshot object store uses {cumulative} bytes, exceeding "
                f"capture ceiling {cumulative_ceiling} bytes "
                f"(session quota {self.session_quota_bytes})",
            )
        # Also enforce cumulative store size soft bound via logical objects.
        # The v2 root dir is not created during a normal capture (objects/
        # staging/committed are siblings), so probe the nearest existing
        # ancestor — free space is identical across the same filesystem.
        probe = self.paths.root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            # This runs at capture step 8 — *after* every object has already been
            # written into objects/ — so `usage.free` already excludes `added`.
            # Subtracting it again demanded roughly twice the headroom the
            # capture actually needed, and failed deterministically on exactly
            # the workloads this system exists for (multi-GB AnnData captures):
            # the bytes were safely on disk, the capture was rejected anyway, the
            # objects were cleaned up, and Retry reproduced the same arithmetic
            # forever. The reserve is the only meaningful post-write guard.
            if usage.free < self.free_space_reserve_bytes:
                raise SnapshotStorageError(
                    "snapshot_quota_exceeded",
                    f"Insufficient disk space: {usage.free} free after writing "
                    f"{added}, below {self.free_space_reserve_bytes} reserve",
                )
        except OSError as exc:
            raise SnapshotStorageError(
                "DISK_USAGE_CHECK_FAILED",
                f"Failed to check disk usage: {exc}",
            )

    def _cleanup_unpublished_capture_objects(
        self, existing_hashes: set[str],
    ) -> None:
        """Remove only transaction-created objects proven unreachable."""
        objects_dir = Path(str(self.paths.objects))
        if not objects_dir.is_dir():
            return
        candidates = {
            path.name
            for path in objects_dir.iterdir()
            if path.is_file() and path.name not in existing_hashes
        }
        if not candidates:
            return
        try:
            from backend.services.snapshot_v2.gc import SnapshotGarbageCollector

            protected = SnapshotGarbageCollector(
                self.session_id, self.paths,
            ).mark()
        except Exception:
            logger.exception(
                "[%s] cannot prove unpublished object reachability; "
                "leaving capture objects for recovery GC",
                self.session_id,
            )
            return
        for sha256 in candidates - protected:
            path = objects_dir / sha256
            try:
                path.unlink(missing_ok=True)
                self.object_store.invalidate_evidence(sha256)
            except OSError:
                logger.exception(
                    "[%s] failed to remove unpublished capture object %s",
                    self.session_id, sha256,
                )

    def _physical_object_store_bytes(self) -> int:
        objects_dir = Path(str(self.paths.objects))
        total = 0
        if not objects_dir.is_dir():
            return total
        for path in objects_dir.iterdir():
            if not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    async def _estimate_capture_addition(
        self,
        request: SnapshotCaptureRequest,
        workspace_root: str | None,
        analysis_state: Any = None,
        *,
        sandbox: Any = None,
    ) -> int:
        """Estimate new physical bytes after content-addressed deduplication.

        Includes a runtime component. Without one the estimate counted only the
        state, notebook, checkpoint and workspace — a handful of MB against the
        tens of GB a runtime capture writes — so it was not conservative but a
        systematic under-count, and the pre-write ceiling it feeds could
        essentially never fire for a runtime-bearing capture. Every over-quota
        capture was therefore caught only *after* the kernel had been driven
        through a full multi-minute serialization and the bytes were on disk,
        and each Retry repeated the whole write-then-delete cycle with no
        possibility of a different outcome.

        The runtime figure is taken from the lineage head's manifest: it is the
        best available projection of what this capture will write, and it costs
        no kernel round trip. Unlike the state/notebook/workspace components it
        is deliberately not deduped against the store — see
        `_estimate_runtime_addition`.
        """
        estimate = 0
        known_hashes: set[str] = set()
        objects_dir = Path(str(self.paths.objects))
        if objects_dir.is_dir():
            known_hashes = {
                path.name for path in objects_dir.iterdir() if path.is_file()
            }

        def add_bytes(raw: bytes) -> None:
            nonlocal estimate
            sha = hashlib.sha256(raw).hexdigest()
            if sha not in known_hashes:
                estimate += len(raw)
                known_hashes.add(sha)

        def add_file(path: Path) -> None:
            nonlocal estimate
            try:
                size = 0
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
            except FileNotFoundError:
                return
            sha = digest.hexdigest()
            if sha not in known_hashes:
                estimate += size
                known_hashes.add(sha)

        if request.analysis_state_ref is None:
            state = analysis_state or await self._get_session_state(request.session_id)
            if state is not None:
                add_bytes(self._serialize_analysis_state(state))

        states_dir = get_session_states(request.session_id)
        if request.notebook_ref is None:
            from backend.utils.live_paths import live_notebook_path

            notebook = live_notebook_path(states_dir, request.session_id)
            if notebook.is_file():
                add_file(notebook)
            else:
                add_bytes(json.dumps(
                    {
                        "nbformat": 4,
                        "nbformat_minor": 5,
                        "metadata": {},
                        "cells": [],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"))

        if request.checkpoint_ref is None:
            checkpoint = checkpoint_sqlite_path(states_dir)
            if checkpoint.is_file():
                add_file(checkpoint)

        if workspace_root:
            from backend.services.snapshot_v2.workspace import WorkspaceScope

            root = Path(workspace_root)
            for relative in WorkspaceScope().list_live_scoped_paths(root):
                path = root / relative
                add_file(path)

        estimate += self._estimate_runtime_addition(
            request, analysis_state, known_hashes, sandbox=sandbox,
        )
        return estimate

    def _estimate_runtime_addition(
        self,
        request: SnapshotCaptureRequest,
        analysis_state: Any,
        known_hashes: set[str],
        *,
        sandbox: Any = None,
    ) -> int:
        """Projected new bytes for the runtime half of *request*.

        Zero when no kernel will be captured. Otherwise the lineage head's
        runtime object sizes — the same data `_can_reuse_runtime` already
        consults, so it needs no kernel round trip and cannot itself fail a
        capture.
        """
        if request.runtime is not None:
            return 0
        if sandbox is None or getattr(sandbox, "km", None) is None:
            return 0
        try:
            previous = self._resolve_previous_manifest(analysis_state)
        except Exception:
            # A missing or unverifiable lineage head is not a capture failure;
            # it only means we have nothing to project from.
            return 0
        if previous is None or previous.runtime is None:
            return 0
        projected = 0
        # Deduped only against itself (an adata and the residual can carry the
        # same hash), never against the caller's store-derived *known_hashes*.
        # The head's refs model the bytes this capture is about to write; they
        # are not those bytes. GC marks every object of a committed lineage head
        # reachable, so the head's own runtime objects are present in objects/
        # by construction — testing membership there zeroed the projection in
        # exactly the steady state the pre-I/O ceiling exists for, leaving the
        # over-quota capture to be caught only after the kernel had been driven
        # through a full serialization. The genuine-reuse case is short-circuited
        # upstream instead: `_can_reuse_runtime` copies the previous runtime
        # block onto the request, and `request.runtime is not None` returns 0 at
        # the top of this function.
        seen: set[str] = set()
        refs = list((previous.runtime.adata_refs or {}).values())
        if previous.runtime.residual_ref is not None:
            refs.append(previous.runtime.residual_ref)
        for ref in refs:
            if ref.sha256 in seen:
                continue
            projected += ref.size
            seen.add(ref.sha256)
            # Still recorded so the caller cannot count the same hash twice.
            known_hashes.add(ref.sha256)
        return projected

    def _mark_runtime_committed(
        self, sandbox: Any, runtime_revision: int | None = None,
    ) -> None:
        self.runtime_capture.mark_runtime_committed(
            sandbox, runtime_revision=runtime_revision,
        )
