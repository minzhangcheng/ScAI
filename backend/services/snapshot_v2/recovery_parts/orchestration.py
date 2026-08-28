"""Fail-closed recovery iteration and public orchestration."""
from __future__ import annotations

import logging
import os
import shutil
import sys
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


def _facade_bindings():
    return sys.modules["backend.services.snapshot_v2.recovery"]

from .primitives import *
from .envelopes import *

async def _recover_snapshot_transactions_unlocked(
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
    live_txns: list = []
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
                if _facade_bindings()._recover_restore_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    restorer=restorer,
                    recovered_txns=recovered_txns,
                ):
                    result.session_state_rewritten = True
            elif envelope.kind == "restore_job":
                _facade_bindings()._recover_restore_job_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                )
            elif envelope.kind == "delete":
                _facade_bindings()._recover_delete_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                )
            elif envelope.kind == "import":
                _facade_bindings()._recover_import_envelope(
                    envelope,
                    paths=paths,
                    session_id=session_id,
                    recovered_txns=recovered_txns,
                    live_txns=live_txns,
                )
            elif envelope.kind == "export_protection":
                _facade_bindings()._recover_export_protection_envelope(
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
            _report_partial_session_state_rewrite(restorer, result)

        except Exception as exc:
            logger.exception(
                "Recovery [%s] failed for transaction %s: %s",
                session_id, tx_id, exc,
            )
            failed_txns.append(tx_id)
            errors.append(str(exc))
            _report_partial_session_state_rewrite(restorer, result)

    if recovered_txns:
        result.recovered = True

    result.recovered_transactions = recovered_txns
    result.failed_transactions = failed_txns
    result.errors = errors

    # ── Post-recovery: orphan staging cleanup ─────────────────────
    staging_dir = Path(str(paths.staging))
    if staging_dir.exists():
        known_failed = set(failed_txns)
        # Staging of a deferred live transaction ("<tx>" dirs, "<tx>_*.json"
        # staged manifests) is being written right now - it is not orphaned.
        live_prefixes = tuple(live_txns)
        for entry in list(staging_dir.iterdir()):
            if live_prefixes and entry.name.startswith(live_prefixes):
                continue
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
            elif entry.is_file():
                if entry.name.startswith("."):
                    # An ObjectStore.put_* temp. Its ``finally`` covers every
                    # in-process path, so one still here after a full day was
                    # stranded by a hard kill. Nothing else walks staging/, and
                    # the bytes count against the free-space reserve, so this is
                    # one of the two places they are ever reclaimed.
                    if reclaim_stale_staging_temp(entry):
                        logger.info(
                            "Recovery [%s]: removed stranded object temp %s",
                            session_id, entry.name,
                        )
                    continue
                entry.unlink()
                logger.info(
                    "Recovery [%s]: removed orphan staging file %s",
                    session_id, entry.name,
                )

    # ── Post-recovery: safe GC ────────────────────────────────────
    # Never sweep while a safe point is held. This entry point runs on every
    # ensure_session_loaded(), i.e. on every authenticated request for the
    # session, and takes no session lock - so it interleaves with a capture
    # parked in an await (runtime serialisation). A capture writes no journal,
    # so mark() has nothing that protects the objects it has already put, and
    # sweeping them lets that capture publish a committed manifest whose
    # analysis_state/notebook/checkpoint refs point at deleted files. Deletion
    # and import already refuse to touch the object store under an active safe
    # point (_guard_active_safe_point); GC is the third object-store mutator
    # and needs the same guard. Skipping only defers reclamation - the next
    # load with no safe point held collects the same garbage.
    # A live import is also a reason to skip: mark() protects objects only via
    # committed manifests, pending-import markers and journal payloads, and an
    # import between its first put_stream and its STAGED journal has none of
    # those for the objects already on disk. GC is opportunistic - the import
    # runs its own sweep at commit, and the next request retries this one.
    if (
        not failed_txns
        and not live_txns
        and not _gc_blocked_by_active_safe_point(session_id)
    ):
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


async def recover_snapshot_transactions(
    session_id: str,
    paths,
    session_dir: str,
    workspace_root: str,
    states_root: str,
) -> RecoveryResult:
    """Recover snapshot journals under the shared lifecycle transaction lock."""
    from backend.services.session_lifecycle import (
        assert_session_generation_active,
        lifecycle_file_lock,
    )

    # The recovery implementation deliberately has no suspension points. This
    # lets one flock cover the complete journal decision and every live-file
    # mutation without blocking unrelated event-loop work mid-transaction.
    with lifecycle_file_lock(session_id):
        assert_session_generation_active(session_id)
        result = await _recover_snapshot_transactions_unlocked(
            session_id=session_id,
            paths=paths,
            session_dir=session_dir,
            workspace_root=workspace_root,
            states_root=states_root,
        )
        assert_session_generation_active(session_id)
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
            result = await _facade_bindings().recover_snapshot_transactions(
                session_id=sid,
                paths=paths,
                session_dir=session_dir,
                workspace_root=workspace_root,
                states_root=states_root,
            )
            if result.recovered:
                combined.recovered = True
            if result.session_state_rewritten:
                combined.session_state_rewritten = True
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

__all__ = [name for name in globals() if not name.startswith("__")]
