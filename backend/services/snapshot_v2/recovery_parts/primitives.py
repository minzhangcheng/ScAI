"""Recovery data types, filesystem primitives, and restorer."""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from backend.services.snapshot_v2.checkpoint_paths import (
    SQLITE_COMPANION_SUFFIXES,
    checkpoint_sqlite_path,
    sqlite_companion_paths,
)
from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.object_store import (
    ManifestStore,
    ObjectStore,
    reclaim_stale_staging_temp,
)
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
    #: True when recovery rewrote the live ``session.json`` (rollback or
    #: forward-complete). The in-memory ``sessions`` cache is *not* touched by
    #: recovery, so on this path it still holds the state the interrupted
    #: restore installed — the caller must evict it, or the very next
    #: ``ensure_session_loaded`` writes that stale object straight back over the
    #: file recovery just restored and silently half-undoes the rollback.
    session_state_rewritten: bool = False


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


def _discard_sqlite_db(db_path: Path) -> None:
    """Remove a SQLite database together with its -wal/-shm companions.

    A WAL-mode database is three files, so unlinking only the main one leaves
    companions that the next connection replays into a half-existing database.
    Callers use this when the whole DB should be gone — never when the main file
    is being *replaced*, where the surviving WAL still belongs to it.
    """
    db_path = Path(str(db_path))
    for path in (db_path, *sqlite_companion_paths(db_path)):
        path.unlink(missing_ok=True)


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
        #: True once the current rollback/forward pass has streamed a new live
        #: ``session.json``. Both writes land several failable steps before the
        #: pass returns, so completion is the wrong moment to report the rewrite
        #: — the caller's cache eviction has to key off the durable write, not
        #: off success. Reset per transaction by rollback/forward_complete.
        self.session_json_rewritten = False

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

    def _restore_stashed_checkpoint_companions(
        self, journal: dict, ckpt_dst: Path,
    ) -> None:
        """Put back the WAL companions ``apply()`` parked in the staging tree.

        Absent for a transaction whose live DB had no companions when apply ran
        (and for journals written before apply started stashing them), which is
        indistinguishable here and equally fine: nothing to replay, and whatever
        is live keeps the pre-existing treatment of belonging to the main file
        being put back.

        When a stash does exist, apply took the originals away, so any companion
        live *now* was created after the target image was installed and would be
        replayed onto the pre_restore database — drop it before reinstating.
        """
        from backend.services.snapshot_v2.restore_parts.coordinator import (
            prev_checkpoint_companion_path,
        )

        tx_id = journal.get("transaction_id") or ""
        if not tx_id:
            return
        staged_root = Path(str(self.paths.staging)) / tx_id
        companions = SQLITE_COMPANION_SUFFIXES
        stashed = {
            companion: prev_checkpoint_companion_path(staged_root, companion)
            for companion in companions
        }
        if not any(path.exists() for path in stashed.values()):
            return
        for companion in companions:
            live = ckpt_dst.with_name(ckpt_dst.name + companion)
            live.unlink(missing_ok=True)
            if stashed[companion].exists():
                _stream_copy(stashed[companion], live)

    def rollback(self, journal: dict) -> bool:
        """Revert a partial apply by materializing the pre_restore snapshot.

        Returns True only when pre_restore is fully applied. On any failure
        (absent ref, missing/corrupt manifest, missing objects, apply error)
        raises RecoveryFailure — never returns True for partial success.
        """
        self.session_json_rewritten = False
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
        # Object hashes are the h5ad file contents, so they are exactly what the
        # live sidecar manifest must record (see the rewrite below).
        restored_h5ad_hashes: dict[str, str] = {}
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
                    restored_h5ad_hashes[var_name] = ref.sha256
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

        # Mirror of the forward path's else-branch (RestoreCoordinator.apply
        # quarantines the live runtime when a complete target has no staged
        # residual). Without it a rollback to a pre_restore snapshot that had no
        # residual — the normal shape for a session with no live kernel, which is
        # exactly what users restore from — left the TARGET's pkl installed by
        # apply() in place, with a valid signature. The next kernel start then
        # rehydrated the target's globals into a session whose files, notebook
        # and checkpoint had all been reverted, and the session ended up with a
        # residual it never owned before the restore.
        if rsc is None or rsc.residual_ref is None:
            try:
                from backend.utils.runtime_apply import quarantine_live_runtime_pkl

                pkl_session_id = (
                    journal.get("session_id")
                    or getattr(pre_manifest, "session_id", "")
                    or ""
                )
                if pkl_session_id:
                    quarantine_live_runtime_pkl(pkl_session_id)
            except Exception as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to quarantine the target residual pkl: {exc}",
                ) from exc

        # Rewrite the live h5ad sidecar manifest to match what we just put back.
        # The forward path does this via install_live_runtime_h5ad_set(), but
        # rollback copied the sidecars with a bare _stream_copy and left the
        # manifest describing the TARGET snapshot. The kernel checks each h5ad
        # against those hashes on restore and refuses to rehydrate on mismatch,
        # so the rollback meant to rescue the session was what stopped its kernel
        # from starting. Written unconditionally — when pre_restore had no h5ad
        # runtime the correct manifest is an empty one, not a stale one.
        rollback_session_id = (
            journal.get("session_id")
            or getattr(pre_manifest, "session_id", "")
            or ""
        )
        if rollback_session_id:
            try:
                from backend.utils.runtime_apply import (
                    finalize_live_runtime_h5ad_set,
                )

                # Also quarantines sidecars the TARGET installed that pre_restore
                # does not name — rewriting the manifest alone would leave those
                # files live in the states dir, so the rollback would not fully
                # undo the restore.
                finalize_live_runtime_h5ad_set(
                    rollback_session_id, restored_h5ad_hashes,
                )
            except Exception as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to rewrite h5ad manifest during rollback: {exc}",
                    details={"vars": sorted(restored_h5ad_hashes)},
                ) from exc

        # Analysis state / notebook if declared on pre_restore
        if getattr(pre_manifest, "analysis_state_ref", None) is not None:
            try:
                ref = pre_manifest.analysis_state_ref
                object_store.verify(ref)
                src = Path(str(self.paths.objects)) / ref.sha256
                dst = self.session_dir / "session.json"
                _stream_copy(src, dst)
                # Durable from here on, with the notebook and checkpoint steps
                # still to come — record it now so a failure below still tells
                # the caller its cached state is stale.
                self.session_json_rewritten = True
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
                from backend.utils.live_paths import live_notebook_path

                dst = live_notebook_path(self.states_root, pre_manifest.session_id)
                _stream_copy(src, dst)
            except Exception as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to restore notebook: {exc}",
                ) from exc

        # The LangGraph checkpoint decides which graph node the session resumes
        # at. apply() replaces it with the target snapshot's copy
        # (restore_parts/coordinator.py), so a rollback that skips it leaves the
        # session resuming from the TARGET position against a pre_restore
        # workspace and runtime — the node's expected values simply are not
        # there. forward_complete() already restores the checkpoint on its half
        # of recovery; this is the missing mirror image. Restored last, so a
        # crash mid-rollback leaves the journal and the next boot re-runs this
        # (idempotent) rollback.
        ckpt_dst = checkpoint_sqlite_path(self.states_root)
        ckpt_ref = getattr(pre_manifest, "checkpoint_ref", None)
        if ckpt_ref is not None:
            try:
                object_store.verify(ckpt_ref)
                src = Path(str(self.paths.objects)) / ckpt_ref.sha256
                if not src.exists():
                    raise RecoveryFailure(
                        "PRE_RESTORE_OBJECT_MISSING",
                        f"checkpoint object missing: {ckpt_ref.sha256}",
                        details={"sha256": ckpt_ref.sha256},
                    )
                # LangGraph runs this DB in WAL mode and capture stored the MAIN
                # FILE only, so commits still sitting in the WAL were never in
                # the snapshot; the WAL belongs to the very main file being put
                # back, and replaying it restores the graph progress the
                # snapshot could not capture. Measured: keeping it yields
                # [PRE_A, PRE_IN_WAL] with integrity_check ok; losing it
                # silently drops PRE_IN_WAL.
                #
                # apply() no longer leaves those companions live — SQLite would
                # replay them over the TARGET image — so it stashes them in the
                # transaction's staging tree, and this is where they come back.
                _stream_copy(src, ckpt_dst)
                self._restore_stashed_checkpoint_companions(journal, ckpt_dst)
            except RecoveryFailure:
                raise
            except Exception as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to restore checkpoint: {exc}",
                ) from exc
        else:
            # pre_restore had no checkpoint DB, so the session had none either:
            # whatever is live now was seeded by apply() from the target, and any
            # companion can only belong to that. Remove the whole triple.
            try:
                _discard_sqlite_db(ckpt_dst)
            except OSError as exc:
                raise RecoveryFailure(
                    "PRE_RESTORE_APPLY_FAILED",
                    f"failed to discard target checkpoint during rollback: {exc}",
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
            quarantine_live_runtime,
        )

        self.session_json_rewritten = False
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
            elif (
                journal.get("target_runtime_status") == "complete"
                and not journal.get("target_runtime_has_residual")
            ):
                quarantine_live_runtime(session_id)
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
                from backend.utils.live_paths import live_notebook_path

                dst = live_notebook_path(self.states_root, session_id)
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
                self.session_json_rewritten = True
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
                objects_root=self.paths.objects,
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

RecoveryFailure.__module__ = "backend.services.snapshot_v2.recovery"
RecoveryResult.__module__ = "backend.services.snapshot_v2.recovery"

__all__ = [name for name in globals() if not name.startswith("__")]
