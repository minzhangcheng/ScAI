# RestoreCoordinator: stages, applies, and rolls back a restore transaction.
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

import logging
import shutil
from pathlib import Path
from typing import Optional

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.object_store import ManifestStore, ObjectStore
from backend.services.snapshot_v2.runtime import validate_runtime_dependency_proof
from backend.services.snapshot_v2.runtime_names import runtime_h5ad_path
from backend.services.snapshot_v2.workspace import WorkspaceScope

logger = logging.getLogger(__name__)


# -- Result types ------------------------------------------------------------


from .journal import AppliedRestore, StagedRestore, _journal_path, _new_transaction_id, _read_journal, _write_journal_hmac_signed
from .staging import _atomic_copy, _atomic_replace, _iter_manifest_object_refs, _require_declared_object, _runtime_status_from_manifest, _stage_object_ref, _stream_copy, _validate_checkpoint_capsule_compatibility, _validate_staged_analysis_state, _validate_staged_checkpoint, _validate_staged_notebook


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
                dst = runtime_h5ad_path(runtime_dir, var_name)
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
            dst = checkpoint_sqlite_path(staged_root)
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
            "target_runtime_status": getattr(rt, "status", None),
            "target_runtime_has_residual": bool(
                getattr(rt, "residual_ref", None)
            ),
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
            atomic_replace_pkl, quarantine_live_runtime,
            resolve_runtime_choice, RuntimeChoiceRequired,
            install_live_runtime_h5ad_set,
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
            "target_runtime_status": _runtime_status_from_manifest(staged.manifest),
            "target_runtime_has_residual": bool(
                ((staged.manifest or {}).get("runtime") or {}).get("residual_ref")
            ),
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
        has_complete_runtime = rt_status == "complete"

        if has_complete_runtime:
            # Ignore supplied runtime_choice for complete targets.
            runtime_restore_status = "complete"
            if "runtime_pkl" not in completed_steps:
                if staged.staged_runtime_pkl:
                    atomic_replace_pkl(session_id, staged.staged_runtime_pkl)
                else:
                    # A complete runtime with no residual is an exact empty
                    # residual target, not an unavailable runtime policy.
                    quarantine_live_runtime(session_id)
                ledger.append({"step": "runtime_pkl", "completed": True})
                self._persist_ledger(paths, tx_id, ledger)
                steps_applied.append("runtime_pkl")

            self._check_fault("runtime_pkl_replaced")

            if "runtime_sig" not in completed_steps:
                # Always rewrite live h5ad set to the target — including empty
                # (residual/pkl-only complete) so stale sidecars cannot rehydrate.
                install_live_runtime_h5ad_set(
                    session_id, staged.staged_runtime_h5ads or {},
                )
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
                if "runtime_sig" not in completed_steps:
                    install_live_runtime_h5ad_set(
                        session_id, staged.staged_runtime_h5ads or {},
                    )
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
            from backend.utils.live_paths import live_notebook_path

            nb_dst = live_notebook_path(states_root_p, session_id)
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
            ckpt_dst = checkpoint_sqlite_path(states_root_p)
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
