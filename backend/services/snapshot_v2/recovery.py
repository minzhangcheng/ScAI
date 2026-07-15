"""Journal-based crash recovery for Snapshot V2 restore transactions.

Scans the session's journal directory, detects incomplete transactions, and
applies the appropriate recovery action:

- ``rollback`` mode: materialize the pre_restore snapshot to undo partial apply
- ``forward_only`` mode: replay the target to complete the apply
- ``COMMITTED`` journal: finalize cleanup only

The entry point ``recover_snapshot_transactions()`` is called before any
session load, workflow construction, or kernel start.

Every journal is verified via ``read_verified_restore_journal`` before any
mutation. Tampered journals and missing/corrupt pre_restore snapshots are
typed recovery failures that leave the journal in place and set
``mutations_blocked=True``.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.runtime_names import runtime_h5ad_path
from backend.services.snapshot_v2.workspace import WorkspaceScope

if TYPE_CHECKING:
    from backend.services.snapshot_v2.gc import GCResult

logger = logging.getLogger(__name__)


class RecoveryFailure(Exception):
    """Typed recovery failure — do not treat as successful rollback/forward."""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.code and self.code not in base:
            return f"{self.code}: {base}"
        return base


@dataclass
class RecoveryResult:
    recovered: bool
    recovered_transactions: list = field(default_factory=list)
    failed_transactions: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    mutations_blocked: bool = False
    gc_result: Optional["GCResult"] = None


# -- Journal helpers ----------------------------------------------------------


def _journal_path(paths, transaction_id: str):
    return paths.journals / f"{transaction_id}.json"


def _stream_copy(src: Path, dst: Path) -> None:
    """Copy *src* → *dst* using streaming I/O (no full read_bytes)."""
    dst = Path(str(dst))
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


def _clear_path(path) -> None:
    path = Path(str(path))
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


# -- Scoped restore coordinator for recovery ----------------------------------


class _RecoveryRestorer:
    """Minimal restore logic used during recovery (no journal writes)."""

    def __init__(self, paths, session_dir, workspace_root, states_root):
        self.paths = paths
        self.session_dir = Path(str(session_dir))
        self.workspace_root = Path(str(workspace_root))
        self.states_root = Path(str(states_root))
        self._scope = WorkspaceScope()

    def _apply_scoped_workspace_from_manifest(self, manifest) -> None:
        """Restore workspace files from *manifest* using WorkspaceScope only.

        Allowed path set is derived exclusively from the verified manifest
        workspace entries (in-scope only). Later-added scoped files are removed;
        out-of-scope directories are never touched.
        """
        object_store = ObjectStore(self.paths)
        allowed: set[str] = set()
        failures: list[str] = []

        for entry in manifest.workspace:
            if not self._scope.is_in_scope(entry.path):
                continue
            try:
                obj_ref = manifest.objects.get(entry.object_hash)
                if obj_ref is None:
                    failures.append(f"undeclared object for {entry.path}")
                    continue
                object_store.verify(obj_ref)
                src = Path(str(self.paths.objects)) / entry.object_hash
                if not src.exists():
                    failures.append(f"missing object {entry.object_hash} for {entry.path}")
                    continue
                dst = self.workspace_root / entry.path
                _stream_copy(src, dst)
                allowed.add(entry.path)
            except Exception as exc:
                failures.append(f"{entry.path}: {exc}")

        if failures:
            raise RecoveryFailure(
                "PRE_RESTORE_APPLY_FAILED",
                f"Scoped workspace restore failed: {'; '.join(failures[:5])}",
                details={"failures": failures},
            )

        self._scope.remove_added_files(self.workspace_root, allowed)

    def _apply_scoped_workspace_from_staged(self, staged_workspace: Path) -> None:
        """Copy staged workspace into live paths using WorkspaceScope only."""
        allowed: set[str] = set()
        if staged_workspace.exists():
            for fpath in staged_workspace.rglob("*"):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(staged_workspace).as_posix()
                if not self._scope.is_in_scope(rel):
                    continue
                dst = self.workspace_root / rel
                _stream_copy(fpath, dst)
                allowed.add(rel)
        self._scope.remove_added_files(self.workspace_root, allowed)

    def rollback(self, journal: dict) -> bool:
        """Revert a partial apply by materializing the pre_restore snapshot.

        Returns True only when pre_restore is fully applied. On any failure
        (absent ref, missing/corrupt manifest, missing objects, apply error)
        raises RecoveryFailure — never returns True for partial success.
        """
        pre_ref = (journal.get("pre_restore_ref") or "").strip()
        if not pre_ref:
            raise RecoveryFailure(
                "PRE_RESTORE_MISSING",
                "pre_restore_ref absent; cannot roll back safely",
                details={"transaction_id": journal.get("transaction_id")},
            )

        try:
            manifest_store = ManifestStore(self.paths)
            pre_manifest = manifest_store.load(pre_ref)
        except Exception as exc:
            raise RecoveryFailure(
                "PRE_RESTORE_CORRUPT",
                f"pre_restore manifest {pre_ref} not available: {exc}",
                details={"pre_restore_ref": pre_ref},
            ) from exc

        # Scoped workspace restore from verified pre_restore manifest only.
        self._apply_scoped_workspace_from_manifest(pre_manifest)

        object_store = ObjectStore(self.paths)
        rsc = pre_manifest.runtime
        if rsc is not None:
            for var_name, ref in (rsc.adata_refs or {}).items():
                try:
                    object_store.verify(ref)
                    src = Path(str(self.paths.objects)) / ref.sha256
                    if not src.exists():
                        raise RecoveryFailure(
                            "PRE_RESTORE_OBJECT_MISSING",
                            f"h5ad object missing for {var_name}: {ref.sha256}",
                            details={"var": var_name, "sha256": ref.sha256},
                        )
                    dst = runtime_h5ad_path(self.states_root, var_name)
                    _stream_copy(src, dst)
                except RecoveryFailure:
                    raise
                except Exception as exc:
                    raise RecoveryFailure(
                        "PRE_RESTORE_APPLY_FAILED",
                        f"failed to restore h5ad {var_name}: {exc}",
                        details={"var": var_name},
                    ) from exc

            if rsc.residual_ref is not None:
                ref = rsc.residual_ref
                try:
                    object_store.verify(ref)
                    from backend.utils.runtime_apply import atomic_replace_pkl
                    src = Path(str(self.paths.objects)) / ref.sha256
                    if not src.exists():
                        raise RecoveryFailure(
                            "PRE_RESTORE_OBJECT_MISSING",
                            f"residual pkl object missing: {ref.sha256}",
                            details={"sha256": ref.sha256},
                        )
                    session_id = journal.get("session_id", "")
                    tmp = self.states_root / f".rollback-{ref.sha256[:12]}.pkl"
                    _stream_copy(src, tmp)
                    atomic_replace_pkl(session_id, str(tmp))
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                except RecoveryFailure:
                    raise
                except Exception as exc:
                    raise RecoveryFailure(
                        "PRE_RESTORE_APPLY_FAILED",
                        f"failed to restore residual pkl: {exc}",
                    ) from exc

        # Analysis state / notebook if declared on pre_restore
        if getattr(pre_manifest, "analysis_state_ref", None) is not None:
            try:
                ref = pre_manifest.analysis_state_ref
                object_store.verify(ref)
                src = Path(str(self.paths.objects)) / ref.sha256
                dst = self.session_dir / "session.json"
                _stream_copy(src, dst)
            except Exception as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to restore analysis state: {exc}",
                ) from exc

        if getattr(pre_manifest, "notebook_ref", None) is not None:
            try:
                ref = pre_manifest.notebook_ref
                object_store.verify(ref)
                src = Path(str(self.paths.objects)) / ref.sha256
                dst = self.states_root / "notebook.ipynb"
                _stream_copy(src, dst)
            except Exception as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to restore notebook: {exc}",
                ) from exc

        logger.info(
            "Rollback: restored from pre_restore snapshot %s",
            pre_ref,
        )
        return True

    def forward_complete(self, journal: dict) -> bool:
        """Replay the target apply by reading staged artifacts (streaming).

        Returns False when required target steps remain incomplete and staging
        is missing (cannot finish the apply). Callers must still re-verify live
        state before treating recovery as success.
        """
        from backend.utils.runtime_apply import (
            atomic_replace_pkl,
            install_live_runtime_h5ad_set,
        )

        tx_id = journal["transaction_id"]
        staged_root = Path(str(self.paths.staging)) / tx_id
        workspace_dir = staged_root / "workspace"
        runtime_dir = staged_root / "runtime"
        staging_present = staged_root.exists()
        session_id = journal.get("session_id", "")

        ledger = journal.get("applied_step_ledger", [])
        completed = {s["step"] for s in ledger if s.get("completed")}

        if "workspace" not in completed and workspace_dir.exists():
            self._apply_scoped_workspace_from_staged(workspace_dir)
            ledger.append({"step": "workspace", "completed": True})
            completed.add("workspace")

        if "runtime_pkl" not in completed:
            pkl_src = runtime_dir / "residual.pkl" if runtime_dir.exists() else None
            if pkl_src and pkl_src.exists():
                atomic_replace_pkl(session_id, str(pkl_src))
                ledger.append({"step": "runtime_pkl", "completed": True})
                completed.add("runtime_pkl")

        if "runtime_sig" not in completed and staging_present:
            # Pkl-only / residual-only complete targets may omit runtime/; still
            # rewrite the live h5ad set (empty → clear stale sidecars).
            staged_h5ads = {}
            if runtime_dir.exists():
                staged_h5ads = {
                    h5ad_file.stem: str(h5ad_file)
                    for h5ad_file in runtime_dir.glob("*.h5ad")
                }
            install_live_runtime_h5ad_set(session_id, staged_h5ads)
            ledger.append({"step": "runtime_sig", "completed": True})
            completed.add("runtime_sig")

        if "notebook" not in completed:
            nb_src = staged_root / "notebook.ipynb"
            if nb_src.exists():
                dst = self.states_root / "notebook.ipynb"
                _stream_copy(nb_src, dst)
                ledger.append({"step": "notebook", "completed": True})
                completed.add("notebook")

        if "state" not in completed:
            state_src = staged_root / "analysis_state.json"
            if not state_src.exists():
                state_src = staged_root / "session.json"
            if state_src.exists():
                dst = self.session_dir / "session.json"
                _stream_copy(state_src, dst)
                ledger.append({"step": "state", "completed": True})
                completed.add("state")

        if "checkpoint" not in completed:
            ckpt_src = checkpoint_sqlite_path(staged_root)
            if ckpt_src.exists():
                dst = checkpoint_sqlite_path(self.states_root)
                _stream_copy(ckpt_src, dst)
                ledger.append({"step": "checkpoint", "completed": True})
                completed.add("checkpoint")

        # Required apply steps that live verify still re-checks.
        required_steps = {"workspace", "state"}
        incomplete = required_steps - completed
        if incomplete and not staging_present:
            return False
        return True

    def remove_staging(self, journal: dict) -> None:
        tx_id = journal["transaction_id"]
        staged_root = Path(str(self.paths.staging)) / tx_id
        _clear_path(staged_root)

    def verify_target_complete(self, journal: dict) -> dict:
        """Forward-verify that live paths match the target snapshot.

        Aligns with ``SnapshotService._verify_live_restore`` via the shared
        ``verify_live_restore_components`` contract:

        - session.json presence only (NOT sha256 == analysis_state_ref after
          activate/persist — live session.json is continuation-mutated)
        - production residual path ``{session_id}_session.pkl`` + integrity sig
        - scoped workspace / notebook / checkpoint / branch / continuation

        Returns a proof dict suitable for v0→v1 migration / COMMITTED journals.
        Raises ``RecoveryFailure`` when verification fails.
        """
        from backend.services.snapshot_v2.live_verify import (
            verify_live_restore_components,
        )

        target_id = (journal.get("target_snapshot_id") or "").strip()
        if not target_id:
            raise RecoveryFailure(
                "TARGET_MISSING",
                "target_snapshot_id absent; cannot verify committed restore",
                details={"transaction_id": journal.get("transaction_id")},
            )

        try:
            manifest_store = ManifestStore(self.paths)
            manifest = manifest_store.load(target_id)
        except Exception as exc:
            raise RecoveryFailure(
                "TARGET_CORRUPT",
                f"target manifest {target_id} not available: {exc}",
                details={"target_snapshot_id": target_id},
            ) from exc

        session_id = str(journal.get("session_id") or "")
        try:
            return verify_live_restore_components(
                session_id=session_id,
                target_manifest=manifest,
                session_dir=self.session_dir,
                workspace_root=self.workspace_root,
                states_root=self.states_root,
                runtime_choice=journal.get("runtime_choice"),
                restored_state=None,
                staged_checkpoint=None,
                journal=journal,
                require_analysis_state=True,
            )
        except ValueError as exc:
            failures = [p.strip() for p in str(exc).split(";") if p.strip()]
            raise RecoveryFailure(
                "TARGET_VERIFY_FAILED",
                f"target verification failed for {target_id}: "
                f"{'; '.join(failures[:8])}",
                details={"target_snapshot_id": target_id, "failures": failures},
            ) from exc
        except RecoveryFailure:
            raise
        except Exception as exc:
            raise RecoveryFailure(
                "TARGET_VERIFY_FAILED",
                f"target verification failed for {target_id}: {exc}",
                details={"target_snapshot_id": target_id},
            ) from exc


# -- Kind-dispatched recovery handlers ----------------------------------------


def _recover_restore_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    restorer: "_RecoveryRestorer",
    recovered_txns: list,
) -> None:
    """Recover a restore_apply envelope (Task 1 state machine preserved)."""
    from backend.services.snapshot_v2.journal_store import (
        JournalStore,
        envelope_to_restore_dict,
    )
    from backend.services.snapshot_v2.restore import (
        RestoreJournal,
        migrate_verified_restore_v0,
        read_verified_restore_journal,
    )

    journal = envelope_to_restore_dict(envelope)
    tx_id = envelope.transaction_id
    status = journal.get("status", "")
    recovery_mode = journal.get("recovery_mode", "rollback")
    restore_journal_version = int(
        journal.get("restore_journal_version")
        if journal.get("restore_journal_version") is not None
        else 0
    )
    jp = _journal_path(paths, tx_id)

    if status == "COMMITTED":
        verified = read_verified_restore_journal(paths, tx_id)
        if restore_journal_version == 0:
            proof = restorer.verify_target_complete(journal)
            migrate_verified_restore_v0(paths, verified, proof)
            verified = read_verified_restore_journal(paths, tx_id)
            journal = verified.to_dict()
            restore_journal_version = int(verified.restore_journal_version)
        elif not (
            journal.get("verified_live_root_hash")
            or verified.verified_live_root_hash
        ):
            raise RecoveryFailure(
                "COMMITTED_WITHOUT_VERIFICATION",
                f"transaction {tx_id} lacks live verification proof",
            )
        restorer.verify_target_complete(journal)
        restorer.remove_staging(journal)
        JournalStore(paths).remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: cleanup for committed restore_apply %s",
            session_id, tx_id,
        )
        return

    if status in ("APPLYING", "STAGED"):
        if recovery_mode == "forward_only":
            ok = restorer.forward_complete(journal)
            if not ok:
                raise RecoveryFailure(
                    "FORWARD_COMPLETE_FAILED",
                    f"forward_complete returned failure for {tx_id}",
                )
            # Live contract gate: never drop APPLYING journal without verify.
            restorer.verify_target_complete(journal)
            restorer.remove_staging(journal)
            JournalStore(paths).remove(tx_id)
            recovered_txns.append(tx_id)
            logger.info(
                "Recovery [%s]: forward-completed restore_apply %s",
                session_id, tx_id,
            )
        else:
            ok = restorer.rollback(journal)
            if not ok:
                raise RecoveryFailure(
                    "ROLLBACK_FAILED",
                    f"rollback returned failure for {tx_id}",
                )
            restorer.remove_staging(journal)
            JournalStore(paths).remove(tx_id)
            recovered_txns.append(tx_id)
            logger.info(
                "Recovery [%s]: rolled back restore_apply %s",
                session_id, tx_id,
            )
        return

    if status == "STAGING":
        restorer.remove_staging(journal)
        JournalStore(paths).remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: removed partial staging for restore_apply %s",
            session_id, tx_id,
        )
        return

    logger.warning(
        "Recovery [%s]: restore_apply %s unknown status %r — left in place",
        session_id, tx_id, status,
    )


def _recover_restore_job_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    """Non-mutating job status recovery: drop terminal job journals only."""
    from backend.services.snapshot_v2.journal_store import JournalStore

    st = str(envelope.status or "").lower()
    terminal = {"committed", "failed", "cancelled", "timed_out"}
    if st in terminal:
        JournalStore(paths).remove(envelope.transaction_id)
        recovered_txns.append(envelope.transaction_id)
        logger.info(
            "Recovery [%s]: cleaned terminal restore_job %s (%s)",
            session_id, envelope.transaction_id, st,
        )
    else:
        # Active job journal — leave for status polls / operator
        logger.info(
            "Recovery [%s]: restore_job %s status=%s left in place",
            session_id, envelope.transaction_id, st,
        )


def _recover_delete_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    from backend.services.snapshot_v2.deletion import recover_delete_transaction

    journal = dict(envelope.payload or {})
    journal["transaction_id"] = envelope.transaction_id
    journal["session_id"] = envelope.session_id
    journal["status"] = envelope.status
    recover_delete_transaction(journal, paths=paths, session_id=session_id)
    recovered_txns.append(envelope.transaction_id)
    logger.info(
        "Recovery [%s]: recovered delete transaction %s",
        session_id, envelope.transaction_id,
    )


def _recover_import_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    """Import recovery: STAGING cleans import staging; STAGED reconciles;
    COMMITTED cleans up; FAILED remains visible (no restore masquerade).
    """
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.object_store import ManifestStore

    tx_id = envelope.transaction_id
    status = str(envelope.status or "").upper()
    store = JournalStore(paths)
    staging_root = Path(str(paths.staging))

    if status == "STAGING":
        # Remove only import staging dirs prefixed with this txn
        if staging_root.exists():
            for entry in list(staging_root.iterdir()):
                if entry.name.startswith(f"{tx_id}") or entry.name.startswith(
                    f"{tx_id}_"
                ):
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        try:
                            entry.unlink()
                        except OSError:
                            pass
        store.remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: cleaned STAGING import journal %s",
            session_id, tx_id,
        )
        return

    if status == "STAGED":
        # Collect snapshot IDs from the envelope payload so we can clean
        # up any pending-import markers left over from the interrupted
        # batch.  The batch was never committed, so these manifests are
        # orphaned and must be removed.
        payload = envelope.payload or {}
        pending_ids = set(
            str(sid) for sid in payload.get("protected_snapshot_ids") or []
            if sid
        )

        # Remove the manifests directly if they have pending-import
        # markers (the batch was never published).
        if pending_ids:
            from backend.services.snapshot_v2.deletion import DeletionService
            from backend.services.snapshot_v2.object_store import ObjectStore

            del_paths = paths
            del_service = DeletionService(
                session_id, del_paths,
                manifest_store=ManifestStore(del_paths),
                object_store=ObjectStore(del_paths),
            )
            try:
                del_service.delete_pending_import_manifests(list(pending_ids))
            except Exception as exc:
                logger.warning(
                    "Recovery [%s]: cleanup of pending-import manifests "
                    "failed for %s: %s",
                    session_id, tx_id, exc,
                )

        # Also remove any partial batch record
        try:
            from backend.services.snapshot_v2.batch_store import ImportBatchStore
            batch_store = ImportBatchStore(paths.root)
            batch_store.remove(tx_id)
        except Exception as exc:
            logger.warning(
                "Recovery [%s]: cleanup of partial batch record "
                "failed for %s: %s",
                session_id, tx_id, exc,
            )

        try:
            reconcile_result = ManifestStore(paths).reconcile()
        except Exception as exc:
            logger.error(
                "Recovery [%s]: import STAGED reconcile failed for %s: %s",
                session_id, tx_id, exc,
            )
            raise RecoveryFailure(
                "IMPORT_RECONCILE_FAILED",
                f"Import STAGED reconcile failed for {tx_id}: {exc}",
                details={"transaction_id": tx_id, "session_id": session_id},
            ) from exc
        if reconcile_result.corrupt_ids:
            logger.error(
                "Recovery [%s]: import STAGED reconcile found corrupt "
                "manifests for %s: %s",
                session_id, tx_id, reconcile_result.corrupt_ids,
            )
            raise RecoveryFailure(
                "IMPORT_RECONCILE_CORRUPT",
                f"Import STAGED reconcile found corrupt manifests: "
                f"{reconcile_result.corrupt_ids}",
                details={
                    "transaction_id": tx_id,
                    "session_id": session_id,
                    "corrupt_ids": list(reconcile_result.corrupt_ids),
                },
            )
        try:
            from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
            SnapshotGarbageCollector(session_id, paths).run()
        except Exception as exc:
            logger.warning(
                "Recovery [%s]: import STAGED GC failed for %s: %s",
                session_id, tx_id, exc,
            )
        store.remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: reconciled STAGED import journal %s",
            session_id, tx_id,
        )
        return

    if status == "COMMITTED":
        store.remove(tx_id)
        recovered_txns.append(tx_id)
        logger.info(
            "Recovery [%s]: cleaned COMMITTED import journal %s",
            session_id, tx_id,
        )
        return

    if status == "FAILED":
        logger.warning(
            "Recovery [%s]: FAILED import journal %s left in place",
            session_id, tx_id,
        )
        return

    logger.warning(
        "Recovery [%s]: import journal %s unknown status %r — left in place",
        session_id, tx_id, status,
    )


def _recover_export_protection_envelope(
    envelope,
    *,
    paths,
    session_id: str,
    recovered_txns: list,
) -> None:
    """Export protection is non-mutating; remove only if no active owner.

    Crash recovery cannot confirm an active export handle, so leave ACTIVE
    journals in place (fail closed for protection). Orphaned protections
    older than a process restart are still safe to leave — they only block
    deletes until an operator removes them.
    """
    logger.info(
        "Recovery [%s]: export_protection %s left in place (non-mutating)",
        session_id, envelope.transaction_id,
    )


# -- Public recovery entry point ----------------------------------------------


async def recover_snapshot_transactions(
    session_id: str,
    paths,
    session_dir: str,
    workspace_root: str,
    states_root: str,
) -> RecoveryResult:
    """Scan journals for the given session and recover incomplete transactions.

    Recovery is two-phase and fail-closed: phase 1 reads and verifies every
    journal via ``JournalStore``; phase 2 dispatches by ``envelope.kind``. If
    any journal is unverifiable (tampered / unsigned legacy / corrupt), phase 2
    runs only proven non-mutating cleanup (terminal ``restore_job`` and
    ``export_protection`` markers) and defers all mutating recovery, so live
    state is never changed after the session is proven untrustworthy. Failed
    rollback/forward likewise leaves the journal in place. Any failure sets
    ``mutations_blocked=True``.
    """
    from backend.services.snapshot_v2.journal_store import JournalStore

    result = RecoveryResult(recovered=False)
    journals_dir = paths.journals
    if not journals_dir.exists():
        return result

    restorer = _RecoveryRestorer(paths, session_dir, workspace_root, states_root)
    recovered_txns: list = []
    failed_txns: list = []
    errors: list = []
    store = JournalStore(paths)

    # ── Phase 1: read + verify every journal BEFORE any mutating recovery ──
    # An unverifiable journal (unsigned/tampered/corrupt) proves the session
    # state is untrustworthy. Fail-closed requires that we never mutate live
    # state (restore_apply/delete/import) after such proof, so we detect it
    # across all journals first and only then dispatch. See P1 review finding.
    verified: list[tuple[str, Any]] = []
    any_unverifiable = False
    for jp in sorted(Path(str(journals_dir)).iterdir()):
        if not jp.is_file() or jp.suffix != ".json":
            continue

        tx_id = jp.stem

        try:
            envelope = store.read(tx_id)
        except RecoveryFailure as exc:
            # Record integrity/read failures (including LEGACY_UNSIGNED_JOURNAL).
            errors.append(str(exc))
            failed_txns.append(tx_id)
            any_unverifiable = True
            logger.warning(
                "Recovery [%s]: recovery failure reading %s: %s",
                session_id, jp.name, exc,
            )
            continue
        except SnapshotIntegrityError as exc:
            errors.append(f"Journal integrity failure {jp.name}: {exc}")
            failed_txns.append(tx_id)
            any_unverifiable = True
            logger.warning(
                "Recovery [%s]: journal HMAC/parse failure for %s: %s",
                session_id, jp.name, exc,
            )
            continue
        except Exception as exc:
            errors.append(f"Corrupt journal: {jp.name}: {exc}")
            failed_txns.append(tx_id)
            any_unverifiable = True
            logger.exception(
                "Recovery [%s]: unexpected read failure for %s",
                session_id, tx_id,
            )
            continue

        if envelope.session_id not in (None, "", session_id):
            continue

        verified.append((tx_id, envelope))

    # ── Phase 2: dispatch recovery by kind ─────────────────────────────────
    # If any journal was unverifiable, run ONLY proven non-mutating cleanup
    # (terminal restore_job / export_protection markers) and defer every
    # mutating recovery. Deferred transactions keep their journals and recover
    # on a later load once the bad journal is resolved; mutations stay blocked.
    _NON_MUTATING_KINDS = {"restore_job", "export_protection"}
    for tx_id, envelope in verified:
        if any_unverifiable and envelope.kind not in _NON_MUTATING_KINDS:
            logger.warning(
                "Recovery [%s]: deferring %s recovery for %s — an unverifiable "
                "journal blocks mutating recovery (fail-closed)",
                session_id, envelope.kind, tx_id,
            )
            continue

        try:
            if envelope.kind == "restore_apply":
                _recover_restore_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    restorer=restorer,
                    recovered_txns=recovered_txns,
                )
            elif envelope.kind == "restore_job":
                _recover_restore_job_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                )
            elif envelope.kind == "delete":
                _recover_delete_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                )
            elif envelope.kind == "import":
                _recover_import_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                )
            elif envelope.kind == "export_protection":
                _recover_export_protection_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                )
            else:
                errors.append(f"Unknown journal kind {envelope.kind!r} for {tx_id}")
                failed_txns.append(tx_id)

        except RecoveryFailure as exc:
            logger.error(
                "Recovery [%s] typed failure for transaction %s: %s",
                session_id, tx_id, exc,
            )
            failed_txns.append(tx_id)
            errors.append(str(exc))

        except Exception as exc:
            logger.exception(
                "Recovery [%s] failed for transaction %s: %s",
                session_id, tx_id, exc,
            )
            failed_txns.append(tx_id)
            errors.append(str(exc))

    if recovered_txns:
        result.recovered = True

    result.recovered_transactions = recovered_txns
    result.failed_transactions = failed_txns
    result.errors = errors

    # ── Post-recovery: orphan staging cleanup ─────────────────────
    staging_dir = Path(str(paths.staging))
    if staging_dir.exists():
        known_failed = set(failed_txns)
        for entry in list(staging_dir.iterdir()):
            if entry.is_dir():
                if entry.name in known_failed:
                    continue
                if entry.name not in recovered_txns:
                    jp_check = _journal_path(paths, entry.name)
                    if not jp_check.exists():
                        shutil.rmtree(entry)
                        logger.info(
                            "Recovery [%s]: removed orphan staging dir %s",
                            session_id, entry.name,
                        )
            elif entry.is_file() and not entry.name.startswith("."):
                entry.unlink()
                logger.info(
                    "Recovery [%s]: removed orphan staging file %s",
                    session_id, entry.name,
                )

    # ── Post-recovery: safe GC ────────────────────────────────────
    if not failed_txns:
        try:
            from backend.services.snapshot_v2.gc import SnapshotGarbageCollector
            gc = SnapshotGarbageCollector(session_id, paths)
            result.gc_result = gc.run()
            if result.gc_result and result.gc_result.removed_count > 0:
                logger.info(
                    "Recovery [%s]: GC removed %d objects (%d bytes)",
                    session_id,
                    result.gc_result.removed_count,
                    result.gc_result.removed_bytes,
                )
        except Exception as exc:
            logger.warning(
                "Recovery [%s]: GC failed, continuing: %s",
                session_id, exc,
            )

    if failed_txns:
        result.mutations_blocked = True
        logger.warning(
            "Recovery [%s]: %d failed transactions — mutations blocked",
            session_id, len(failed_txns),
        )

    return result


async def recover_for_all_sessions(paths_factory=None) -> RecoveryResult:
    """Scan all known session directories for incomplete restore journals."""
    from backend.config._config import SESSION_DIR

    combined = RecoveryResult(recovered=False)
    session_root = SESSION_DIR

    if not session_root.exists():
        return combined

    for sess_dir in sorted(session_root.iterdir()):
        if not sess_dir.is_dir():
            continue
        sid = sess_dir.name
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        paths = SnapshotV2Paths.for_session(sid)

        workspace_root = str(sess_dir / "workspace")
        states_root = str(sess_dir / "states")
        session_dir = str(sess_dir)

        try:
            result = await recover_snapshot_transactions(
                session_id=sid,
                paths=paths,
                session_dir=session_dir,
                workspace_root=workspace_root,
                states_root=states_root,
            )
            if result.recovered:
                combined.recovered = True
            # Always merge failure/error state even when recovered=False
            # (e.g. HMAC failures, blocked rollbacks with no successful recovery).
            combined.recovered_transactions.extend(result.recovered_transactions)
            combined.failed_transactions.extend(result.failed_transactions)
            combined.errors.extend(result.errors)
            if result.mutations_blocked:
                combined.mutations_blocked = True
        except Exception as exc:
            logger.exception("Recovery for session %s failed: %s", sid, exc)
            combined.errors.append(f"{sid}: {exc}")

    return combined
