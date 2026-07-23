# Staging-area file primitives and validation of staged objects.
# Moved verbatim out of backend/services/snapshot_v2/restore.py.
"""Crash-recoverable multi-artifact restore coordinator for Snapshot V2.

Provides ``RestoreCoordinator`` with four-phase lifecycle::

    stage(snapshot_id, ...) -> StagedRestore
    apply(staged, paths, ...) -> AppliedRestore
    commit(transaction_id, paths) -> None
    cleanup(transaction_id, paths) -> None

Every transition is journaled under ``journals/<transaction_id>.json`` with
HMAC integrity, so ``recover_snapshot_transactions`` in ``recovery.py`` can
complete or roll back an interrupted transaction.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.models import ObjectRef
from backend.services.snapshot_v2.object_store import ObjectStore

logger = logging.getLogger(__name__)


# -- Result types ------------------------------------------------------------


def _stream_copy(src, dst) -> None:
    """Streaming file copy with fsync (no full read_bytes)."""
    dst = Path(dst) if isinstance(dst, str) else dst
    src = Path(src) if isinstance(src, str) else src
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.tmp")
    with open(src, "rb") as rf, open(tmp, "wb") as wf:
        shutil.copyfileobj(rf, wf, length=1024 * 1024)
        wf.flush()
        os.fsync(wf.fileno())
    os.replace(str(tmp), str(dst))
    dfd = os.open(str(dst.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)

def _atomic_copy(src, dst) -> None:
    """Copy file using shared atomic primitive."""
    from backend.utils.files import atomic_copy_file
    atomic_copy_file(Path(src), Path(dst))

def _atomic_replace(src, dst) -> None:
    """Atomically replace dst with src using shared primitive."""
    from backend.utils.files import atomic_write_bytes
    data = Path(src).read_bytes()
    atomic_write_bytes(Path(dst), data)

def _require_declared_object(manifest, ref: ObjectRef, *, field: str) -> None:
    if ref is None:
        return
    sha = getattr(ref, "sha256", None)
    if not sha:
        raise SnapshotIntegrityError(
            "UNDECLARED_OBJECT",
            f"ObjectRef for {field} has empty sha256",
            details={"field": field},
        )
    objects = getattr(manifest, "objects", None) or {}
    if sha not in objects:
        raise SnapshotIntegrityError(
            "UNDECLARED_OBJECT",
            f"{field} references undeclared object {sha}",
            details={"field": field, "object_hash": sha},
        )

def _stage_object_ref(object_store: ObjectStore, paths, ref, dst: Path) -> str:
    object_store.verify(ref)
    src = Path(str(paths.objects)) / ref.sha256
    _atomic_copy(src, dst)
    return str(dst)

def _validate_staged_analysis_state(path: Path) -> None:
    from backend.models.schemas import AnalysisState

    try:
        raw = path.read_bytes()
        AnalysisState.model_validate_json(raw)
    except Exception as exc:
        raise SnapshotIntegrityError(
            "STAGED_ANALYSIS_STATE_INVALID",
            f"Staged AnalysisState failed Pydantic validation: {exc}",
            details={"path": str(path)},
        ) from exc

def _validate_staged_notebook(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(
            "STAGED_NOTEBOOK_INVALID",
            f"Staged notebook is not valid JSON: {exc}",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise SnapshotIntegrityError(
            "STAGED_NOTEBOOK_INVALID",
            "Staged notebook JSON must be an object",
            details={"path": str(path)},
        )
    if "cells" not in payload and "nbformat" not in payload:
        raise SnapshotIntegrityError(
            "STAGED_NOTEBOOK_INVALID",
            "Staged notebook missing nbformat/cells structure",
            details={"path": str(path)},
        )

def _validate_staged_checkpoint(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise SnapshotIntegrityError(
            "STAGED_CHECKPOINT_INVALID",
            f"Staged checkpoint missing: {path}",
        )
    if path.stat().st_size <= 0:
        raise SnapshotIntegrityError(
            "STAGED_CHECKPOINT_INVALID",
            f"Staged checkpoint is empty: {path}",
        )

def _validate_checkpoint_capsule_compatibility(manifest) -> None:
    cont = getattr(manifest, "continuation", None)
    if cont is None:
        return
    kind = getattr(cont, "kind", None)

    if kind == "graph_node":
        next_node = getattr(cont, "next_node", None)
        if not next_node or not isinstance(next_node, str):
            raise SnapshotIntegrityError(
                "CHECKPOINT_CAPSULE_INCOMPATIBLE",
                "graph_node capsule requires a non-empty next_node",
                details={"kind": kind},
            )
        planning_stage = getattr(cont, "planning_stage", None)
        if planning_stage not in ("stage1", "stage2", "full"):
            raise SnapshotIntegrityError(
                "CHECKPOINT_CAPSULE_INCOMPATIBLE",
                f"graph_node capsule has invalid planning_stage: {planning_stage!r}",
                details={"kind": kind, "planning_stage": planning_stage},
            )

    elif kind == "semantic_gate":
        gate = getattr(cont, "gate", None)
        if not gate or not isinstance(gate, str):
            raise SnapshotIntegrityError(
                "CHECKPOINT_CAPSULE_INCOMPATIBLE",
                "semantic_gate capsule requires a non-empty gate",
                details={"kind": kind},
            )

    elif kind == "terminal":
        status = getattr(cont, "status", None)
        if status not in ("completed", "stopped"):
            raise SnapshotIntegrityError(
                "CHECKPOINT_CAPSULE_INCOMPATIBLE",
                f"terminal capsule has invalid status: {status!r}",
                details={"kind": kind, "status": status},
            )

    elif kind == "idle":
        pass

    elif kind is not None:
        raise SnapshotIntegrityError(
            "CHECKPOINT_CAPSULE_INCOMPATIBLE",
            f"Unknown continuation kind: {kind!r}",
            details={"kind": kind},
        )

def _iter_manifest_object_refs(manifest):
    for field_name in ("analysis_state_ref", "notebook_ref", "checkpoint_ref"):
        ref = getattr(manifest, field_name, None)
        if ref is not None:
            yield field_name, ref

    rt = getattr(manifest, "runtime", None)
    if rt is not None:
        if getattr(rt, "residual_ref", None) is not None:
            yield "runtime.residual_ref", rt.residual_ref
        for var_name, ref in (getattr(rt, "adata_refs", None) or {}).items():
            if ref is not None:
                yield f"runtime.adata_refs.{var_name}", ref

    cont = getattr(manifest, "continuation", None)
    if cont is not None:
        kind = getattr(cont, "kind", None)
        if kind == "graph_node":
            ref = getattr(cont, "workflow_state_ref", None)
            if ref is not None:
                yield "continuation.workflow_state_ref", ref
        elif kind == "semantic_gate":
            ref = getattr(cont, "payload_ref", None)
            if ref is not None:
                yield "continuation.payload_ref", ref

    for entry in getattr(manifest, "workspace", None) or []:
        yield f"workspace:{entry.path}", ObjectRef(
            sha256=entry.object_hash,
            size=entry.size,
            media_type="application/octet-stream",
        )

def _runtime_status_from_manifest(manifest_dict: dict) -> str:
    rt = (manifest_dict or {}).get("runtime") or {}
    if isinstance(rt, dict):
        return str(rt.get("status") or "")
    return str(getattr(rt, "status", "") or "")
