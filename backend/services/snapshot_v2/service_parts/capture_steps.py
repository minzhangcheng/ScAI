# Per-producer capture steps: analysis state, notebook, checkpoint, workspace, runtime eligibility.
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


class CaptureStepsMixin:
    """Per-producer capture steps: analysis state, notebook, checkpoint, workspace, runtime eligibility."""

    async def _capture_analysis_state(
        self,
        request: SnapshotCaptureRequest,
        analysis_state: Any = None,
    ) -> ObjectPutResult:
        """Canonicalize + externalize AnalysisState into the object store.

        Returns ObjectPutResult with ``created`` flag for physical_added accounting.
        """
        if request.analysis_state_ref is not None:
            self.object_store.verify(request.analysis_state_ref)
            return ObjectPutResult(
                ref=request.analysis_state_ref,
                created=False,
            )

        state = analysis_state
        if state is None:
            state = await self._get_session_state(request.session_id)
        if state is None:
            raise SnapshotCaptureError(
                "MISSING_ANALYSIS_STATE",
                "No AnalysisState available to capture",
            )

        raw = self._serialize_analysis_state(state)
        return self.object_store.put_bytes(raw, "application/json")

    @staticmethod
    def _serialize_analysis_state(state: Any) -> bytes:
        """Serialize the canonical business-state payload used by capture."""
        from backend.utils.secret_redaction import strip_analysis_state_secrets

        # model_dump excluding derived listing fields that are rebuilt
        state_dict = strip_analysis_state_secrets(state.model_dump(mode="json"))
        state_dict.pop("snapshots", None)
        state_dict.pop("realtime_messages", None)
        for seg in state_dict.get("detailed_report_segments", []) or []:
            if isinstance(seg, dict):
                seg.pop("content_html", None)

        # Externalize large blobs into object store via inline base64 reduction
        # by writing canonical JSON bytes directly (blob refs stay as tokens).
        raw = json.dumps(
            state_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return raw

    async def _capture_notebook(
        self,
        request: SnapshotCaptureRequest,
    ) -> ObjectPutResult:
        if request.notebook_ref is not None:
            self.object_store.verify(request.notebook_ref)
            return ObjectPutResult(
                ref=request.notebook_ref,
                created=False,
            )

        states_dir = get_session_states(request.session_id)
        from backend.utils.live_paths import live_notebook_path

        nb_path = live_notebook_path(states_dir, request.session_id)
        if not nb_path.exists():
            # Empty notebook is valid when no kernel work has been done.
            empty_nb = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [],
            }
            raw = json.dumps(
                empty_nb, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            return self.object_store.put_bytes(raw, "application/x-ipynb+json")
        return self.object_store.put_file(nb_path, "application/x-ipynb+json")

    async def _capture_checkpoint(
        self,
        request: SnapshotCaptureRequest,
    ) -> ObjectPutResult | None:
        if request.checkpoint_ref is not None:
            self.object_store.verify(request.checkpoint_ref)
            return ObjectPutResult(
                ref=request.checkpoint_ref,
                created=False,
            )
        # Optional: LangGraph checkpoint sqlite if present
        states_dir = get_session_states(request.session_id)
        ckpt = checkpoint_sqlite_path(states_dir)
        if not ckpt.exists() or not ckpt.is_file():
            return None
        return self.object_store.put_file(ckpt, "application/x-sqlite3")

    async def _capture_workspace(
        self, workspace_root: str,
    ) -> tuple[list[WorkspaceEntry], list[str]]:
        warnings: list[str] = []
        entries, ws_warnings = self.workspace_capture.capture_scope(
            self.object_store, workspace_root,
        )
        warnings.extend(ws_warnings)
        return entries, warnings

    def _collect_checkpoint_dependencies(
        self,
        workflow_state: dict[str, Any] | None,
        sandbox_state: dict[str, Any],
        analysis_state: Any = None,
    ) -> set[str]:
        deps: set[str] = set()
        # The safe-point descriptor is a first-class source here, exactly as it
        # is for `_resolve_kernel_present`: it is the channel a producer at a
        # graph boundary actually has, and omitting it was half of why the
        # fail-closed tier had no production writer at all.
        sources: list[Any] = []
        desc = self.safe_point.get_descriptor(self.session_id)
        if desc is not None:
            sources.append(desc)
        sources.extend((workflow_state or {}, sandbox_state))
        for source in sources:
            if isinstance(source, dict):
                deps |= _as_name_set(source.get("checkpoint_dependencies"))
        if analysis_state is not None:
            ss = getattr(analysis_state, "sandbox_state", None) or {}
            if isinstance(ss, dict):
                deps |= _as_name_set(ss.get("checkpoint_dependencies"))
        return deps

    def _rehydrate_tracker(
        self,
        sandbox: Any,
        sandbox_state: dict[str, Any],
    ) -> RuntimeDependencyTracker | None:
        tracker = getattr(sandbox, "runtime_dependency_tracker", None)
        if tracker is not None:
            return tracker
        data = sandbox_state.get("runtime_dependency_tracker_v2")
        if not data:
            return None
        tracker = RuntimeDependencyTracker.from_state_dict(data)
        try:
            sandbox.runtime_dependency_tracker = tracker
        except Exception:
            pass
        return tracker

    def _resolve_kernel_present(
        self,
        sandbox: Any,
        workflow_state: dict[str, Any] | None,
    ) -> bool:
        sources: list[Any] = []
        desc = self.safe_point.get_descriptor(self.session_id)
        if desc is not None:
            sources.append(desc)
        if workflow_state is not None:
            sources.append(workflow_state)

        for source in sources:
            if not isinstance(source, dict):
                continue
            if "kernel_present" in source:
                return bool(source["kernel_present"])
            nested = source.get("safe_point_descriptor")
            if isinstance(nested, dict) and "kernel_present" in nested:
                return bool(nested["kernel_present"])

        return getattr(sandbox, "km", None) is not None

    def _resolve_empty_runtime_eligibility(
        self,
        phase: str | None,
        sandbox: Any,
        safe_point_descriptor: dict[str, Any] | None,
    ) -> bool:
        """Whether empty-complete runtime may be committed without a sandbox.

        Eligibility requires:
        - no live sandbox object
        - an eligibility-trusted descriptor (ack / explicit producer acquire)
        - ``kernel_present is False`` on that descriptor
        - no active workflow ownership (except inactive post_report/manual paths
          that already proved no kernel)
        - terminal or explicitly inactive semantics for dynamic phases

        Raw ``workflow_state`` passed into capture is never sufficient alone.
        ``pre_restore`` with an active workflow never accepts empty-complete,
        even if a contradictory no-kernel descriptor is present. Absence of a
        trusted descriptor is never proof of inactivity.
        """
        if sandbox is not None:
            return False

        phase_s = str(phase or "manual")
        desc = dict(safe_point_descriptor or {}) if safe_point_descriptor else {}

        # No trusted descriptor → unproven; cannot empty-complete.
        if not desc:
            return False

        if "kernel_present" not in desc or desc.get("kernel_present") is not False:
            return False

        active_workflow = self._has_active_workflow_ownership()

        # Active pre_restore always requires the live sandbox.
        if phase_s == "pre_restore" and active_workflow:
            return False

        inactive = desc.get("inactive") is True
        terminal = (
            desc.get("status") == "completed"
            or desc.get("continuation_kind") == "terminal"
            or desc.get("kind") == "terminal"
        )

        if active_workflow:
            # Active pre_restore / post_chat always require the live sandbox.
            if phase_s in ("post_chat", "pre_restore"):
                return False
            if phase_s == "post_report":
                return True
            return inactive or terminal

        # No active workflow ownership.
        # post_report and idle pre_restore may empty-complete with proven
        # no-kernel descriptor. post_chat/manual still need inactive/terminal.
        if phase_s in ("post_report", "pre_restore"):
            return True
        if phase_s in ("post_chat", "manual"):
            return inactive or terminal

        # Unknown / other phases: only with explicit inactive or terminal.
        return inactive or terminal
