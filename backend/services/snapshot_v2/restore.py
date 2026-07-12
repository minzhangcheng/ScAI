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

import hashlib
import hmac as _hmac
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.models import ObjectRef
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.runtime import validate_runtime_dependency_proof
from backend.services.snapshot_v2.workspace import WorkspaceScope

logger = logging.getLogger(__name__)


# -- Result types ------------------------------------------------------------


@dataclass
class StagedRestore:
    transaction_id: str
    session_id: str
    snapshot_id: str
    recovery_mode: str
    pre_restore_ref: str
    branch_id: str
    continuation_kind: str
    manifest: dict
    staged_workspace: str
    staged_runtime_pkl: Optional[str] = None
    staged_runtime_h5ads: dict = field(default_factory=dict)
    staged_notebook: Optional[str] = None
    staged_analysis_state: Optional[str] = None
    staged_checkpoint: Optional[str] = None


@dataclass
class AppliedRestore:
    transaction_id: str
    steps_applied: list
    workspace_files_count: int = 0
    runtime_applied: bool = False
    notebook_applied: bool = False
    state_applied: bool = False
    checkpoint_applied: bool = False
    runtime_restore_status: str = "complete"
    runtime_warning: Optional[str] = None


@dataclass
class RestoreJournal:
    """Validated restore journal (HMAC stripped after verification).

    ``restore_journal_version`` is distinct from the generic envelope
    ``schema_version`` introduced later. Absence of the field means legacy
    restore journal version 0.
    """

    transaction_id: str
    target_snapshot_id: str
    session_id: str
    recovery_mode: str
    status: str
    pre_restore_ref: str = ""
    branch_id: str = "main"
    continuation_kind: str = "idle"
    target_root_hash: str = ""
    applied_step_ledger: list = field(default_factory=list)
    restore_journal_version: Literal[0, 1] = 0
    verified_live_root_hash: str = ""
    component_hashes: dict = field(default_factory=dict)
    verified_at: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """Return a plain dict suitable for re-signing (no hmac)."""
        d = {
            "transaction_id": self.transaction_id,
            "target_snapshot_id": self.target_snapshot_id,
            "session_id": self.session_id,
            "recovery_mode": self.recovery_mode,
            "status": self.status,
            "pre_restore_ref": self.pre_restore_ref,
            "branch_id": self.branch_id,
            "continuation_kind": self.continuation_kind,
            "applied_step_ledger": list(self.applied_step_ledger),
            "restore_journal_version": int(self.restore_journal_version),
        }
        if self.target_root_hash:
            d["target_root_hash"] = self.target_root_hash
        if self.verified_live_root_hash:
            d["verified_live_root_hash"] = self.verified_live_root_hash
        if self.component_hashes:
            d["component_hashes"] = dict(self.component_hashes)
        if self.verified_at:
            d["verified_at"] = self.verified_at
        for k, v in self.raw.items():
            if k not in d and k != "hmac":
                d[k] = v
        return d


# -- Journal helpers ---------------------------------------------------------


def _new_transaction_id() -> str:
    return f"restore-{uuid.uuid4().hex[:16]}"


def _journal_path(paths, transaction_id: str):
    return Path(str(paths.journals)) / f"{transaction_id}.json"


def _journal_hmac(data: bytes) -> str:
    """Legacy flat-body HMAC (Task 1 restore shape)."""
    from backend.utils.crypto import get_integrity_secret
    return _hmac.new(get_integrity_secret().encode(), data, hashlib.sha256).hexdigest()


def _canonicalize_journal_payload(payload: dict) -> bytes:
    """Canonical JSON of flat legacy journal body with hmac removed."""
    body = {k: v for k, v in payload.items() if k != "hmac"}
    return json.dumps(body, sort_keys=True, ensure_ascii=False).encode()


def _restore_dict_from_envelope_payload(
    transaction_id: str,
    session_id: str,
    status: str,
    payload: dict,
) -> dict:
    """Merge envelope identity + payload into flat restore journal dict."""
    body = dict(payload or {})
    body["transaction_id"] = transaction_id
    body["session_id"] = session_id
    body["status"] = status
    return body


def _restore_journal_from_body(body: dict, transaction_id: str) -> RestoreJournal:
    """Parse a flat restore body (legacy or envelope-flattened) into RestoreJournal."""
    required = (
        "transaction_id",
        "target_snapshot_id",
        "session_id",
        "recovery_mode",
        "status",
    )
    missing = [k for k in required if k not in body]
    if missing:
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal missing fields {missing}: {transaction_id}",
            details={"transaction_id": transaction_id, "missing": missing},
        )

    if "kind" in body and body.get("kind") not in (
        None,
        "",
        "restore_apply",
        "restore",
    ):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal has conflicting kind {body.get('kind')!r}: "
            f"{transaction_id}",
            details={"transaction_id": transaction_id, "kind": body.get("kind")},
        )

    raw_version = body.get("restore_journal_version")
    if raw_version is None:
        version: Literal[0, 1] = 0
    else:
        try:
            version_int = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Restore journal invalid restore_journal_version: {transaction_id}",
                details={"transaction_id": transaction_id},
            ) from exc
        if version_int not in (0, 1):
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Restore journal unsupported restore_journal_version={version_int}: "
                f"{transaction_id}",
                details={"transaction_id": transaction_id},
            )
        version = version_int  # type: ignore[assignment]

    return RestoreJournal(
        transaction_id=str(body["transaction_id"]),
        target_snapshot_id=str(body["target_snapshot_id"]),
        session_id=str(body["session_id"]),
        recovery_mode=str(body["recovery_mode"]),
        status=str(body["status"]),
        pre_restore_ref=str(body.get("pre_restore_ref") or ""),
        branch_id=str(body.get("branch_id") or "main"),
        continuation_kind=str(body.get("continuation_kind") or "idle"),
        target_root_hash=str(body.get("target_root_hash") or ""),
        applied_step_ledger=list(body.get("applied_step_ledger") or []),
        restore_journal_version=version,
        verified_live_root_hash=str(body.get("verified_live_root_hash") or ""),
        component_hashes=dict(body.get("component_hashes") or {}),
        verified_at=str(body.get("verified_at") or ""),
        raw=body,
    )


def _verify_flat_legacy_restore_hmac(payload: dict, transaction_id: str) -> dict:
    """Verify Task-1 flat restore journal HMAC; return body without hmac."""
    stored_hmac = payload.get("hmac")
    if not stored_hmac or not isinstance(stored_hmac, str):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal missing hmac: {transaction_id}",
            details={"transaction_id": transaction_id},
        )
    canonical = _canonicalize_journal_payload(payload)
    expected = _journal_hmac(canonical)
    if not _hmac.compare_digest(expected, stored_hmac):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal HMAC mismatch: {transaction_id}",
            details={"transaction_id": transaction_id},
        )
    return {k: v for k, v in payload.items() if k != "hmac"}


def read_verified_restore_journal(paths, transaction_id: str) -> RestoreJournal:
    """Load a restore journal and verify its HMAC integrity.

    Accepts schema-v1 ``restore_apply`` envelopes (via JournalStore) and
    exact Task-1 flat HMAC restore journals (versions 0 and 1).

    Raises ``SnapshotIntegrityError`` with code ``JOURNAL_HMAC`` on missing
    signature, parse failure, or signature mismatch. Callers must treat this
    as a recovery-required, mutation-blocking condition.
    """
    jp = _journal_path(paths, transaction_id)
    if not jp.exists():
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal missing: {transaction_id}",
            details={"transaction_id": transaction_id, "path": str(jp)},
        )
    try:
        raw_text = jp.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal unreadable/corrupt: {transaction_id}: {exc}",
            details={"transaction_id": transaction_id, "path": str(jp)},
        ) from exc

    if not isinstance(payload, dict):
        raise SnapshotIntegrityError(
            "JOURNAL_HMAC",
            f"Restore journal is not an object: {transaction_id}",
            details={"transaction_id": transaction_id},
        )

    # Schema-v1 envelope (kind restore_apply)
    if payload.get("schema_version") == 1 and payload.get("kind") == "restore_apply":
        from backend.services.snapshot_v2.journal_store import JournalStore

        env = JournalStore(paths).read(transaction_id)
        if env.kind != "restore_apply":
            raise SnapshotIntegrityError(
                "JOURNAL_HMAC",
                f"Expected restore_apply envelope, got {env.kind!r}: {transaction_id}",
                details={"transaction_id": transaction_id, "kind": env.kind},
            )
        body = _restore_dict_from_envelope_payload(
            env.transaction_id, env.session_id, env.status, env.payload
        )
        return _restore_journal_from_body(body, transaction_id)

    # Flat legacy restore (Task 1)
    body = _verify_flat_legacy_restore_hmac(payload, transaction_id)
    return _restore_journal_from_body(body, transaction_id)


def _read_journal(paths, transaction_id: str) -> Optional[dict]:
    """Verified journal read. Returns None if absent; raises on tamper."""
    jp = _journal_path(paths, transaction_id)
    if not jp.exists():
        return None
    journal = read_verified_restore_journal(paths, transaction_id)
    return journal.to_dict()


def _write_journal_hmac_signed(paths, transaction_id: str, journal: dict) -> None:
    """Atomically write a restore_apply envelope (or flat v0 for test helper).

    Production writers always use schema-v1 ``restore_apply`` envelopes via
    JournalStore. The flat HMAC form is retained only for
    ``write_committed_restore_v0_for_test`` (explicit Task-1 migration fixtures).

    On-disk JSON mirrors restore fields at both top-level (compat with tests
    and readers that inspect flat keys) and inside ``payload`` (canonical
    envelope contract). HMAC is over the full envelope body excluding hmac.
    """
    # If caller requests explicit flat legacy (no envelope), keep Task-1 shape.
    if journal.get("_force_flat_legacy_v0"):
        body = {k: v for k, v in journal.items() if k not in ("hmac", "_force_flat_legacy_v0")}
        jp = _journal_path(paths, transaction_id)
        jp.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        hmac_val = _journal_hmac(data)
        signed = json.loads(data)
        signed["hmac"] = hmac_val
        tmp = jp.with_suffix(jp.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(signed, sort_keys=True, ensure_ascii=False))
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp), str(jp))
        dfd = os.open(str(jp.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return

    from backend.services.snapshot_v2.journal_store import (
        JournalStore,
        make_envelope,
        sign_envelope_body,
        _atomic_write_bytes,
    )

    body = {k: v for k, v in journal.items() if k != "hmac"}
    status = str(body.get("status") or "")
    session_id = str(body.get("session_id") or "")
    payload = {
        k: v
        for k, v in body.items()
        if k
        not in (
            "transaction_id",
            "session_id",
            "status",
            "schema_version",
            "kind",
            "payload",
            "hmac",
        )
    }
    env = make_envelope(
        kind="restore_apply",
        transaction_id=transaction_id,
        session_id=session_id,
        status=status,
        payload=payload,
    )
    # Write envelope with restore fields also mirrored at top-level for
    # backward-compatible journal inspection (status, restore_journal_version, …).
    # Signing includes mirrored fields so tamper detection covers them.
    disk: dict = env.model_dump(mode="json")
    disk.pop("hmac", None)
    for k, v in payload.items():
        if k not in disk:
            disk[k] = v
    # Ensure identity fields present
    disk["transaction_id"] = transaction_id
    disk["session_id"] = session_id
    disk["status"] = status
    disk["kind"] = "restore_apply"
    disk["schema_version"] = 1
    disk["payload"] = payload
    sig = sign_envelope_body(disk)
    disk["hmac"] = sig
    text = json.dumps(disk, sort_keys=True, ensure_ascii=False)
    path = _journal_path(paths, transaction_id)
    _atomic_write_bytes(path, text.encode())


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
    _stream_copy(src, dst)


def _atomic_replace(src, dst) -> None:
    dst = Path(dst) if isinstance(dst, str) else dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dst))
    dfd = os.open(str(dst.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


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


# -- RestoreCoordinator ------------------------------------------------------


class RestoreCoordinator:
    """Four-phase restore lifecycle with journaled crash recovery."""

    def __init__(self, fault_barrier=None):
        self._fault_barrier = fault_barrier
        self._last_journal = None

    def _check_fault(self, point: str) -> None:
        if self._fault_barrier is not None:
            self._fault_barrier.check(point)

    async def stage(
        self,
        snapshot_id: str,
        session_id: str,
        paths,
        *,
        recovery_mode: str = "rollback",
        pre_restore_ref: str = "",
        branch_id: str = "main",
        continuation_kind: str = "idle",
        transaction_id: str | None = None,
    ) -> StagedRestore:
        object_store = ObjectStore(paths)
        manifest_store = ManifestStore(paths)

        manifest = manifest_store.load(snapshot_id)
        manifest_dict = manifest.model_dump(mode="json")

        _validate_checkpoint_capsule_compatibility(manifest)

        for field_name, ref in _iter_manifest_object_refs(manifest):
            if field_name.startswith("workspace:"):
                path = field_name.split(":", 1)[1]
                if ref.sha256 not in (manifest.objects or {}):
                    raise SnapshotIntegrityError(
                        "UNDECLARED_OBJECT",
                        f"Workspace entry {path} references unknown object "
                        f"{ref.sha256}",
                        details={"path": path, "object_hash": ref.sha256},
                    )
                declared = manifest.objects[ref.sha256]
                object_store.verify(declared)
            else:
                _require_declared_object(manifest, ref, field=field_name)
                object_store.verify(ref)

        tx_id = transaction_id or _new_transaction_id()
        staged_root = Path(str(paths.staging)) / tx_id

        workspace_dir = staged_root / "workspace"
        runtime_dir = staged_root / "runtime"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir.mkdir(parents=True, exist_ok=True)

        staged_runtime_pkl = None
        staged_runtime_h5ads: dict = {}
        staged_notebook = None
        staged_analysis_state = None
        staged_checkpoint = None

        for entry in manifest.workspace:
            src = Path(str(paths.objects)) / entry.object_hash
            dst = workspace_dir / entry.path
            _atomic_copy(src, dst)

        rt = getattr(manifest, "runtime", None)
        if rt is not None:
            validate_runtime_dependency_proof(rt)
            for var_name, ref in (rt.adata_refs or {}).items():
                dst = runtime_dir / f"{var_name}.h5ad"
                staged_runtime_h5ads[var_name] = _stage_object_ref(
                    object_store, paths, ref, dst,
                )

            if rt.residual_ref is not None:
                ref = rt.residual_ref
                dst = runtime_dir / "residual.pkl"
                staged_runtime_pkl = _stage_object_ref(object_store, paths, ref, dst)

        if getattr(manifest, "analysis_state_ref", None) is not None:
            sref = manifest.analysis_state_ref
            dst = staged_root / "analysis_state.json"
            staged_analysis_state = _stage_object_ref(object_store, paths, sref, dst)
            _validate_staged_analysis_state(Path(staged_analysis_state))

        if getattr(manifest, "notebook_ref", None) is not None:
            nref = manifest.notebook_ref
            dst = staged_root / "notebook.ipynb"
            staged_notebook = _stage_object_ref(object_store, paths, nref, dst)
            _validate_staged_notebook(Path(staged_notebook))

        if getattr(manifest, "checkpoint_ref", None) is not None and manifest.checkpoint_ref is not None:
            cref = manifest.checkpoint_ref
            dst = staged_root / "checkpoint.sqlite"
            staged_checkpoint = _stage_object_ref(object_store, paths, cref, dst)
            _validate_staged_checkpoint(Path(staged_checkpoint))

        journal = {
            "transaction_id": tx_id,
            "target_snapshot_id": snapshot_id,
            "target_root_hash": manifest.root_hash,
            "session_id": session_id,
            "pre_restore_ref": pre_restore_ref,
            "recovery_mode": recovery_mode,
            "branch_id": branch_id,
            "continuation_kind": continuation_kind,
            "status": "STAGED",
            "applied_step_ledger": [],
        }
        _write_journal_hmac_signed(paths, tx_id, journal)
        self._last_journal = journal

        self._check_fault("target_staged")

        return StagedRestore(
            transaction_id=tx_id,
            session_id=session_id,
            snapshot_id=snapshot_id,
            recovery_mode=recovery_mode,
            pre_restore_ref=pre_restore_ref,
            branch_id=branch_id,
            continuation_kind=continuation_kind,
            manifest=manifest_dict,
            staged_workspace=str(workspace_dir),
            staged_runtime_pkl=staged_runtime_pkl,
            staged_runtime_h5ads=staged_runtime_h5ads,
            staged_notebook=staged_notebook,
            staged_analysis_state=staged_analysis_state,
            staged_checkpoint=staged_checkpoint,
        )

    async def apply(
        self,
        staged: StagedRestore,
        paths,
        *,
        session_dir: str,
        workspace_root: str,
        states_root: str,
        runtime_choice: Optional[str] = None,
    ) -> AppliedRestore:
        """Replace live content with staged versions, journaling each step.

        Runtime policy:
        - Complete target: install staged pkl+sig+h5ad; ignore runtime_choice.
        - Unavailable + reset: quarantine live, set unavailable.
        - Unavailable + keep: no live runtime mutate, intentionally_kept.
        """
        from backend.utils.runtime_apply import (
            atomic_replace_pkl, atomic_replace_h5ad, quarantine_live_runtime,
            resolve_runtime_choice, RuntimeChoiceRequired,
        )

        tx_id = staged.transaction_id
        session_id = staged.session_id
        workspace_root_p = Path(workspace_root)
        states_root_p = Path(states_root)
        session_dir_p = Path(session_dir)

        existing = _read_journal(paths, tx_id)
        journal = existing or {
            "transaction_id": tx_id,
            "target_snapshot_id": staged.snapshot_id,
            "session_id": session_id,
            "pre_restore_ref": staged.pre_restore_ref,
            "recovery_mode": staged.recovery_mode,
            "status": "APPLYING",
            "applied_step_ledger": [],
        }
        journal["status"] = "APPLYING"
        _write_journal_hmac_signed(paths, tx_id, journal)

        ledger = journal.get("applied_step_ledger", [])
        completed_steps = {s["step"] for s in ledger if s.get("completed")}

        steps_applied = list(completed_steps)
        ws_count = 0
        runtime_applied = False
        nb_applied = False
        state_applied = False
        ckpt_applied = False
        runtime_restore_status = "complete"
        runtime_warning = None

        if "workspace" not in completed_steps:
            ws_count = self._apply_workspace(
                staged, workspace_root_p, paths, tx_id, ledger,
            )
            steps_applied.append("workspace")
        else:
            ws_count = sum(1 for s in ledger if s["step"].startswith("ws:"))

        self._check_fault("workspace_applied")

        rt_status = _runtime_status_from_manifest(staged.manifest)
        has_complete_runtime = (
            rt_status == "complete"
            and (bool(staged.staged_runtime_pkl) or bool(staged.staged_runtime_h5ads))
        )

        if has_complete_runtime:
            # Ignore supplied runtime_choice for complete targets.
            runtime_restore_status = "complete"
            if "runtime_pkl" not in completed_steps and staged.staged_runtime_pkl:
                atomic_replace_pkl(session_id, staged.staged_runtime_pkl)
                ledger.append({"step": "runtime_pkl", "completed": True})
                self._persist_ledger(paths, tx_id, ledger)
                steps_applied.append("runtime_pkl")

            self._check_fault("runtime_pkl_replaced")

            if "runtime_sig" not in completed_steps and staged.staged_runtime_h5ads:
                for var_name, h5ad_path in staged.staged_runtime_h5ads.items():
                    atomic_replace_h5ad(session_id, h5ad_path, var_name=var_name)
                ledger.append({"step": "runtime_sig", "completed": True})
                self._persist_ledger(paths, tx_id, ledger)
                steps_applied.append("runtime_sig")

            self._check_fault("runtime_sig_replaced")
            runtime_applied = True
        else:
            status_for_choice = rt_status if rt_status in (
                "missing", "state_only", "corrupt", "complete",
            ) else "missing"
            try:
                action, runtime_restore_status = resolve_runtime_choice(
                    status_for_choice,  # type: ignore[arg-type]
                    runtime_choice,
                )
            except RuntimeChoiceRequired:
                raise

            if action == "reset":
                if "runtime_reset" not in completed_steps:
                    quarantine_live_runtime(session_id)
                    ledger.append({"step": "runtime_reset", "completed": True})
                    self._persist_ledger(paths, tx_id, ledger)
                    steps_applied.append("runtime_reset")
                runtime_applied = False
                runtime_restore_status = "unavailable"
            elif action == "keep":
                if "runtime_keep" not in completed_steps:
                    ledger.append({"step": "runtime_keep", "completed": True})
                    self._persist_ledger(paths, tx_id, ledger)
                    steps_applied.append("runtime_keep")
                runtime_applied = False
                runtime_restore_status = "intentionally_kept"
                runtime_warning = (
                    "Live runtime intentionally kept; target runtime unavailable. "
                    "Resume proceeds under persistent risk warning."
                )
            elif action == "apply":
                if "runtime_pkl" not in completed_steps and staged.staged_runtime_pkl:
                    atomic_replace_pkl(session_id, staged.staged_runtime_pkl)
                    ledger.append({"step": "runtime_pkl", "completed": True})
                    self._persist_ledger(paths, tx_id, ledger)
                    steps_applied.append("runtime_pkl")
                self._check_fault("runtime_pkl_replaced")
                if "runtime_sig" not in completed_steps and staged.staged_runtime_h5ads:
                    for var_name, h5ad_path in staged.staged_runtime_h5ads.items():
                        atomic_replace_h5ad(session_id, h5ad_path, var_name=var_name)
                    ledger.append({"step": "runtime_sig", "completed": True})
                    self._persist_ledger(paths, tx_id, ledger)
                    steps_applied.append("runtime_sig")
                self._check_fault("runtime_sig_replaced")
                runtime_applied = bool(staged.staged_runtime_pkl) or bool(
                    staged.staged_runtime_h5ads
                )

            if "runtime_pkl" not in steps_applied:
                self._check_fault("runtime_pkl_replaced")
            if "runtime_sig" not in steps_applied:
                self._check_fault("runtime_sig_replaced")

        if "notebook" not in completed_steps and staged.staged_notebook:
            nb_dst = states_root_p / "notebook.ipynb"
            _atomic_replace(staged.staged_notebook, nb_dst)
            ledger.append({"step": "notebook", "completed": True})
            self._persist_ledger(paths, tx_id, ledger)
            steps_applied.append("notebook")
            nb_applied = True

        self._check_fault("notebook_applied")

        if "state" not in completed_steps and staged.staged_analysis_state:
            state_dst = session_dir_p / "session.json"
            _atomic_replace(staged.staged_analysis_state, state_dst)
            ledger.append({"step": "state", "completed": True})
            self._persist_ledger(paths, tx_id, ledger)
            steps_applied.append("state")
            state_applied = True

        self._check_fault("state_replaced")

        if "checkpoint" not in completed_steps and staged.staged_checkpoint:
            ckpt_dst = states_root_p / "checkpoint.sqlite"
            _atomic_replace(staged.staged_checkpoint, ckpt_dst)
            ledger.append({"step": "checkpoint", "completed": True})
            self._persist_ledger(paths, tx_id, ledger)
            steps_applied.append("checkpoint")
            ckpt_applied = True

        self._check_fault("checkpoint_seeded")

        journal = _read_journal(paths, tx_id) or journal
        journal["runtime_restore_status"] = runtime_restore_status
        if runtime_warning:
            journal["runtime_warning"] = runtime_warning
        journal["applied_step_ledger"] = ledger
        journal["status"] = "APPLYING"
        _write_journal_hmac_signed(paths, tx_id, journal)

        return AppliedRestore(
            transaction_id=tx_id,
            steps_applied=steps_applied,
            workspace_files_count=ws_count,
            runtime_applied=runtime_applied,
            notebook_applied=nb_applied,
            state_applied=state_applied,
            checkpoint_applied=ckpt_applied,
            runtime_restore_status=runtime_restore_status,
            runtime_warning=runtime_warning,
        )

    def _apply_workspace(self, staged, workspace_root_p, paths, tx_id, ledger) -> int:
        scope = WorkspaceScope()
        staged_ws = Path(staged.staged_workspace)
        staged_paths: set[str] = set()
        ws_count = 0

        if staged_ws.exists():
            for fpath in staged_ws.rglob("*"):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(staged_ws).as_posix()
                if not scope.is_in_scope(rel):
                    continue
                staged_paths.add(rel)
                dst = workspace_root_p / rel
                _stream_copy(fpath, dst)
                ledger.append({"step": f"ws:{rel}", "completed": True})
                ws_count += 1

        scope.remove_added_files(workspace_root_p, staged_paths)

        ledger.append({"step": "workspace", "completed": True})
        self._persist_ledger(paths, tx_id, ledger)
        return ws_count

    def _persist_ledger(self, paths, tx_id: str, ledger: list) -> None:
        journal = _read_journal(paths, tx_id) or {}
        journal["applied_step_ledger"] = ledger
        journal["status"] = "APPLYING"
        _write_journal_hmac_signed(paths, tx_id, journal)

    async def commit(
        self,
        transaction_id: str,
        paths,
        *,
        verification_proof: dict | None = None,
    ) -> None:
        """Transition a verified APPLYING journal to COMMITTED.

        Production commits **require** ``verification_proof`` with
        ``verified_live_root_hash``, component hashes and ``verified_at``.
        Only this method may write status=COMMITTED for restore apply journals.

        Tests that need a legacy restore-journal-v0 COMMITTED record (recovery
        migration) must use ``write_committed_restore_v0_for_test`` — not
        ``commit(None)``.
        """
        from backend.services.snapshot_v2.errors import SnapshotTransactionError

        journal = _read_journal(paths, transaction_id)
        if journal is None:
            logger.warning("No journal found for transaction %s", transaction_id)
            return
        if not verification_proof or not verification_proof.get("verified_live_root_hash"):
            raise SnapshotTransactionError(
                "COMMIT_WITHOUT_VERIFICATION",
                f"transaction {transaction_id} cannot commit without live "
                "verification proof (use write_committed_restore_v0_for_test "
                "for explicit v0 test journals)",
            )
        journal["status"] = "COMMITTED"
        journal["restore_journal_version"] = 1
        journal["verified_live_root_hash"] = str(
            verification_proof.get("verified_live_root_hash") or ""
        )
        journal["component_hashes"] = dict(
            verification_proof.get("component_hashes") or {}
        )
        journal["verified_at"] = str(verification_proof.get("verified_at") or "")
        if verification_proof.get("restore_journal_version") is not None:
            journal["restore_journal_version"] = int(
                verification_proof["restore_journal_version"]
            )
        _write_journal_hmac_signed(paths, transaction_id, journal)

        self._check_fault("journal_committed")

    def write_committed_restore_v0_for_test(
        self,
        transaction_id: str,
        paths,
        *,
        journal_body: dict | None = None,
    ) -> None:
        """Test-only helper: write an HMAC-valid restore-journal-v0 COMMITTED.

        Production ``commit`` always requires verification proof. Recovery
        migration tests use this lower-level path to simulate pre-Task-1
        COMMITTED journals that lack proof. Writes the exact flat Task-1
        shape (not a schema-v1 envelope).
        """
        journal = dict(journal_body or (_read_journal(paths, transaction_id) or {}))
        journal["transaction_id"] = transaction_id
        journal["status"] = "COMMITTED"
        journal.pop("restore_journal_version", None)
        journal.pop("verified_live_root_hash", None)
        journal.pop("component_hashes", None)
        journal.pop("verified_at", None)
        journal.pop("schema_version", None)
        journal.pop("kind", None)
        journal.pop("payload", None)
        journal["_force_flat_legacy_v0"] = True
        _write_journal_hmac_signed(paths, transaction_id, journal)

    async def cleanup(self, transaction_id: str, paths) -> None:
        staged_root = Path(str(paths.staging)) / transaction_id
        if staged_root.exists():
            shutil.rmtree(staged_root)

        self._check_fault("cleanup")

        jp = _journal_path(paths, transaction_id)
        if jp.exists():
            jp.unlink()


def migrate_verified_restore_v0(
    paths,
    journal: RestoreJournal | dict,
    proof: dict,
) -> RestoreJournal:
    """Atomically rewrite a verified v0 COMMITTED journal as signed v1 envelope.

    Never accepts unsigned input — *journal* must already be HMAC-verified.
    Writes a schema-v1 ``restore_apply`` envelope with restore_journal_version=1.
    """
    if isinstance(journal, RestoreJournal):
        body = journal.to_dict()
        tx_id = journal.transaction_id
    else:
        body = dict(journal)
        tx_id = str(body["transaction_id"])
    body["status"] = "COMMITTED"
    body["restore_journal_version"] = 1
    body["verified_live_root_hash"] = str(proof.get("verified_live_root_hash") or "")
    body["component_hashes"] = dict(proof.get("component_hashes") or {})
    body["verified_at"] = str(proof.get("verified_at") or "")
    _write_journal_hmac_signed(paths, tx_id, body)
    return read_verified_restore_journal(paths, tx_id)
