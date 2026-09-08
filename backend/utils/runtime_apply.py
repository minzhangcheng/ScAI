"""Runtime-state safety: coherence, forward-resume policy, integrity checks,
and recovery of a runtime apply a previous build left half-written.

Nothing here installs a runtime any more -- the snapshot restore that did was
removed, and the ordinary save installs through
``sandbox_parts.regular_save._commit_pkl_transaction``. What survives is the
part that reads: whether the live pkl is coherent with the state that
references it, whether forward resume is safe, and how to retire or roll back
what is on disk.

Intentionally free of workflow/route imports so it can be unit tested and
reused by sandbox startup recovery.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..config import get_session_states

logger = logging.getLogger(__name__)

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

RuntimeStatus = Literal["complete", "missing", "state_only", "corrupt"]
RuntimeRestoreStatus = Literal["complete", "unavailable", "intentionally_kept"]


class RuntimeApplyError(Exception):
    """Journaled runtime apply failed or the installed pkl failed integrity."""


class RuntimeChoiceRequired(Exception):
    """Runtime is unavailable and the caller supplied no runtime_choice."""

    def __init__(self, status: str):
        self.status = status
        super().__init__(f"runtime status '{status}' requires an explicit runtime_choice")


def _journal_hmac(data: bytes) -> str:
    from ..utils.crypto import sign_hmac_sha256
    return sign_hmac_sha256(data)


def _live_pkl(session_id: str) -> Path:
    return get_session_states(session_id) / f"{session_id}_session.pkl"


def _live_sig(session_id: str) -> Path:
    from .sandbox import _sig_path, _integrity_dir
    return _sig_path(_live_pkl(session_id), _integrity_dir(session_id))



def _journal_path(session_id: str) -> Path:
    from .sandbox import _integrity_dir
    return _integrity_dir(session_id) / "runtime_apply.journal.json"




def _quarantine_path(path: Path) -> Path | None:
    """Move *path* aside without overwriting earlier quarantine evidence."""
    if not path.exists():
        return None
    base = path.with_suffix(path.suffix + ".quarantine")
    candidate = base
    index = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}.{index}")
        index += 1
    path.replace(candidate)
    from backend.utils.files import fsync_directory
    fsync_directory(path.parent)
    return candidate




def recover_runtime_apply_journal(session_id: str) -> None:
    # Kept although nothing writes this journal any more: the producer was the
    # snapshot restore, and a session whose apply was interrupted by the crash
    # that preceded this upgrade still has one on disk. Deleting the recovery
    # with the producer would leave that pair half-replaced forever.
    """Complete or roll back an interrupted runtime-pair apply. Idempotent."""
    jp = _journal_path(session_id)
    if not jp.exists():
        return
    try:
        journal = json.loads(jp.read_text())
    except (OSError, json.JSONDecodeError):
        quarantine = jp.with_suffix(".json.corrupt")
        os.replace(jp, quarantine)
        return

    # Verify HMAC (present on journals written after Stage 1 hardening).
    from backend.utils.crypto import verify_hmac_sha256
    stored_hmac = journal.pop("hmac", None)
    if stored_hmac is not None:
        pre_hmac = json.dumps(journal, sort_keys=True).encode()
        if not verify_hmac_sha256(pre_hmac, stored_hmac):
            quarantine = jp.with_suffix(".json.corrupt")
            logger.warning("[%s] journal HMAC mismatch; quarantining", session_id)
            os.replace(jp, quarantine)
            return

    live = _live_pkl(session_id)
    sig = _live_sig(session_id)
    from .sandbox import _integrity_dir
    idir = _integrity_dir(session_id)
    bak_pkl = live.with_suffix(live.suffix + ".bak")
    bak_sig = sig.with_suffix(sig.suffix + ".bak")

    if journal.get("state") == "APPLYING":
        had_live = journal.get("had_live", False)
        if had_live and bak_pkl.exists():
            # The apply backed up a prior live pair — roll back to it.
            os.replace(bak_pkl, live)
            if bak_sig.exists():
                os.replace(bak_sig, sig)
            else:
                sig.unlink(missing_ok=True)
        else:
            # had_live=False: no prior pair was backed up. Remove the
            # half-applied pkl unless it currently verifies (could be a
            # legitimate later save, not apply garbage).
            from .sandbox import _verify_pkl_integrity
            keep_live = live.exists() and _verify_pkl_integrity(live, idir)
            if not keep_live:
                live.unlink(missing_ok=True)
                sig.unlink(missing_ok=True)
        logger.warning("[%s] rolled back interrupted runtime apply", session_id)

    for p in (bak_pkl, bak_sig):
        p.unlink(missing_ok=True)
    jp.unlink(missing_ok=True)


def state_runtime_coherence(state) -> str:
    """Whether this session's newest generation belongs to its persisted state.

    Computed on read rather than latched at write time, so it cannot be
    consulted long after the record behind it changed.

    A state that predates these fields reports "current": absence of a record is
    not evidence of a mismatch, and blocking on it would strand every session
    that existed before this contract.
    """
    from backend.utils.runtime_manifest import runtime_coherence

    persistence = getattr(state, "runtime_persistence", None) or {}
    if not isinstance(persistence, dict):
        return "current"
    has_generation = bool(persistence.get("committed_source")) and bool(
        persistence.get("committed_digest")
    )
    verdict = runtime_coherence(
        committed_revision=persistence.get("committed_revision"),
        last_attempt_revision=int(persistence.get("last_attempt_revision") or 0),
        last_attempt_status=persistence.get("last_attempt_status"),
        has_generation=has_generation,
        current_revision=max(
            int(persistence.get("current_revision") or 0),
            int(persistence.get("degraded_from_revision") or 0),
        ),
    )
    if verdict != "current":
        return verdict
    pairing = persistence.get("committed_pairing")
    sid = getattr(state, "id", None)
    if sid:
        from backend.utils.runtime_manifest import runtime_manifest_path, read_runtime_manifest, RuntimeManifestError
        manifest_path = runtime_manifest_path(sid)
        if manifest_path.exists():
            try:
                manifest = read_runtime_manifest(manifest_path)
            except RuntimeManifestError:
                return "inconsistent"
            if manifest["pkl_sha256"] != persistence.get("committed_digest"):
                return "inconsistent"
            pairing = manifest.get("pairing")
    if pairing is not None:
        version = int(getattr(state, "state_version", 0) or 0)
        if version < int(pairing["state_version"]):
            # Regression, not progress: this record cannot belong to the
            # session in front of us.
            return "inconsistent"
        # Forward drift is NOT staleness. `state_version` advances on every
        # state sync but the persistence commit, and the checkpoint hash
        # advances whenever a graph node returns -- neither requires the kernel
        # to execute anything. Treating either as stale meant that a save at a
        # round boundary, a click on Continue, `round_start`, and a crash during
        # planning left a correct dump that resume refused and that POST /run
        # then retired. `runtime_coherence` above already answered the question
        # this function is asked here, using the kernel's own revision counter.
        #
        # The matched-pair property the pairing proves is still enforced where
        # it belongs: `resumable_export` compares the manifest's pairing with a
        # freshly captured one before it will build a resumable archive, so a
        # dump and a checkpoint that disagree still cannot be shipped together.
    return verdict


def is_forward_resume_blocked(
    runtime_restore_status: str,
    *,
    coherence: str = "current",
    recompute_pending: bool = False,
) -> bool:
    """Whether the run may continue straight into the restored namespace.

    Two independent reasons to stop, and neither is a failure of the session:

    * ``runtime_restore_status == "unavailable"`` -- there is no namespace to
      continue from. "intentionally_kept" is a user override that proceeds under
      a persistent warning; "complete" is normal.
    * a runtime that is not coherent with the persisted workflow state -- stale
      (a closing save failed, so the state is newer than the newest committed
      generation) or inconsistent (a generation committed against a state that
      was never persisted). Continuing would bind an older namespace to a newer
      graph position without anyone being told.

    Both route into the same existing affordance: the user is offered a re-run
    from their data rather than being handed a silent mismatch. Viewing results
    and exporting them are unaffected -- this predicate governs resumption only.

    The keyword defaults to ``"current"`` so a caller that has only a status
    behaves exactly as before.
    """
    if recompute_pending:
        # The user already answered this block by choosing to re-run from their
        # data. Asking again is how the offer became a loop: the lifecycle route
        # cleared runtime_restore_status, the graph re-checked the same
        # condition through coherence, and failed the session it was supposed to
        # restart.
        return False
    if runtime_restore_status == "unavailable":
        return True
    # "unavailable" coherence is deliberately not a block. It means no
    # generation has been recorded, which is true of a fresh session, of every
    # session that predates this contract, and of one restored by a path that
    # installs a dump without writing a manifest. Whether a session that has
    # *worked* has a usable runtime is a different question, and
    # ``runtime_restore_status`` already owns it -- import marks it
    # "unavailable" for exactly that case. Answering it here as well produced
    # false blocks wherever the persistence record simply had not been
    # populated.
    return coherence in {"stale", "inconsistent"}


def verify_live_pkl_or_raise(session_id: str) -> None:
    """Hard gate for sandbox startup: a present live pkl with a missing/invalid
    HMAC is an error, never a silent skip. No pkl at all is fine (fresh kernel).
    """
    from .sandbox import _integrity_dir, _verify_pkl_integrity

    live = _live_pkl(session_id)
    if not live.exists():
        return
    if not _verify_pkl_integrity(live, _integrity_dir(session_id)):
        raise RuntimeApplyError(
            f"[{session_id}] live pkl present but HMAC missing/invalid; refusing silent skip"
        )



def _states_dir(session_id: str) -> Path:
    return get_session_states(session_id)


RUNTIME_H5AD_MANIFEST_FILENAME = "runtime_h5ad_manifest.json"


def runtime_h5ad_manifest_path(
    session_id: str,
    *,
    states_dir: str | Path | None = None,
) -> Path:
    """Return the live h5ad sidecar manifest path for a session."""
    root = Path(states_dir) if states_dir is not None else _states_dir(session_id)
    return root / RUNTIME_H5AD_MANIFEST_FILENAME




def restore_generation_pair(session_id: str, *, revision: int) -> bool:
    """Put a committed generation's captured positions back beside its dump.

    The generation itself is healthy -- it passed its HMAC, its manifest
    verifies, its digest matches. What made it unusable is that the state, the
    graph position and the executor position all moved on before the crash was
    noticed, so nothing on disk agreed with it any more. B2-1 captured those
    three at commit time; this puts them back, and the run continues from the
    step the dump was taken at rather than from the input files.

    Returns False when the pair is absent or incomplete, which leaves
    retirement as the fallback exactly as before. Half a restore point is not
    a restore point: a state without its graph position would resume a
    position that no longer exists.

    Copies are staged beside their targets and published by rename, so a
    failure while reading the pair leaves the live files untouched.
    """
    from backend.utils.checkpoint_paths import checkpoint_sqlite_path
    from backend.utils.files import fsync_directory
    from backend.utils.sandbox_parts.regular_save import (
        GENERATION_PAIR_DIRNAME,
        _copy_durable,
    )

    states_dir = _states_dir(session_id)
    pair = states_dir / GENERATION_PAIR_DIRNAME / str(int(revision))
    paired_state = pair / "state.json"
    paired_checkpoint = pair / "checkpoints.sqlite"
    if not (paired_state.is_file() and paired_checkpoint.is_file()):
        return False

    live_state = states_dir / f"{session_id}_state.json"
    live_checkpoint = checkpoint_sqlite_path(states_dir)
    live_executor = states_dir / f"{session_id}_bioinfo_checkpoint.json"
    paired_executor = pair / "executor_checkpoint.json"

    # Ordered so the *declaration* lands last. There is no journal here, and
    # adding one would duplicate the commit transaction for three small files;
    # the ordering does the same job. `<sid>_state.json` is what says the
    # restored generation is the current one, so publishing it last means a
    # crash between publishes leaves a record that is still newer than the
    # generation -- coherence still reads `stale`, the next start attempts the
    # restore again, and this function is idempotent.
    #
    # The other order is the dangerous one: a record claiming the older
    # generation is current, beside a graph position from the newer work, is a
    # mismatch nothing downstream can detect.
    sources = [(paired_checkpoint, live_checkpoint)]
    if paired_executor.is_file():
        sources.append((paired_executor, live_executor))
    sources.append((paired_state, live_state))

    # What this restore costs, read from the record it is about to replace.
    # Spec B2-3: the restored record reads `current` again -- the pair holds a
    # generation whose current_revision equals its committed_revision -- so
    # without carrying this across, the disclosure would go silent at the exact
    # moment the user most needs it.
    discarded_from = 0
    discarded_since = None
    try:
        previous = (
            json.loads(live_state.read_text()).get("runtime_persistence") or {}
        )
        discarded_from = int(previous.get("current_revision") or 0)
        discarded_since = previous.get("last_saved_at")
    except (OSError, ValueError, TypeError, AttributeError):
        logger.warning(
            "[%s] Could not read the record being replaced by a pair restore",
            session_id,
        )

    incoming: list[tuple[Path, Path]] = []
    staged_state = None
    try:
        for source, target in sources:
            staged = target.with_name(target.name + ".incoming")
            _copy_durable(source, staged)
            if target == live_state:
                staged_state = staged
            incoming.append((staged, target))
    except BaseException:
        for staged, _target in incoming:
            staged.unlink(missing_ok=True)
        raise

    if staged_state is not None:
        try:
            payload = json.loads(staged_state.read_text())
            persistence = payload.setdefault("runtime_persistence", {})
            persistence["pair_restore"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "restored_revision": int(revision),
                "discarded_revisions": max(0, discarded_from - int(revision)),
                "discarded_since": discarded_since,
            }
            staged_state.write_text(json.dumps(payload, indent=2))
        except (OSError, ValueError, TypeError):
            # The restore itself is sound; only its disclosure is not. Failing
            # here would strand a session whose runtime is fine.
            logger.warning(
                "[%s] Could not annotate the restored record with what the "
                "pair restore discarded",
                session_id,
                exc_info=True,
            )

    for staged, target in incoming:
        os.replace(staged, target)
    fsync_directory(states_dir)

    # The write-ahead log and shared-memory file belong to the database that
    # was just replaced. SQLite would replay them over the restored one, which
    # would put the graph back where the crash left it.
    for companion in (
        live_checkpoint.with_name(live_checkpoint.name + "-wal"),
        live_checkpoint.with_name(live_checkpoint.name + "-shm"),
    ):
        companion.unlink(missing_ok=True)

    if not paired_executor.is_file():
        # This generation has no executor position: a round boundary reached
        # after the executor finished clears its checkpoint. A live one is then
        # newer than the namespace being restored -- the §3.11 mismatch in a
        # new place, and the one place a resume would not catch it, since
        # `validate_checkpoint` would see a checkpoint *ahead* of the committed
        # revision and treat it as ordinary.
        _quarantine_path(live_executor)

    logger.info(
        "[%s] Restored the generation pair for revision %d", session_id, revision,
    )
    return True


def retire_live_runtime(session_id: str, *, include_checkpoint: bool = False) -> bool:
    """Move the live runtime aside so the next kernel starts empty.

    "Re-run from your data" promised an empty namespace and the agent prompt
    says so, but clearing ``runtime_restore_status`` only removed the *guard*:
    the dump was still on disk, so the new kernel rehydrated the old variables
    and the run proceeded over state the user had asked to discard.

    Quarantined rather than deleted -- the bytes are the only copy of that
    generation, and a user who re-runs by mistake should not lose it silently.
    """
    from backend.utils.runtime_manifest import (
        runtime_manifest_path,
        runtime_manifest_sig_path,
    )

    live = _live_pkl(session_id)
    retired = _quarantine_path(live) is not None
    _quarantine_path(_live_sig(session_id))
    manifest = runtime_manifest_path(session_id)
    _quarantine_path(manifest)
    _quarantine_path(runtime_manifest_sig_path(manifest))

    # Legacy h5ad sidecars, written by a snapshot restore this build no longer
    # has. Nothing reads them any more, but they name variables the retired
    # generation held, and a session starting fresh beside them would be told
    # it depends on them. Retire them with the rest of the generation.
    states_dir = _live_pkl(session_id).parent
    h5ad_manifest = states_dir / RUNTIME_H5AD_MANIFEST_FILENAME
    if h5ad_manifest.exists():
        _quarantine_path(h5ad_manifest)
        for sidecar in sorted(states_dir.glob("*.h5ad")):
            _quarantine_path(sidecar)
        retired = True

    if include_checkpoint:
        # A checkpoint names the graph node to continue from and assumes the
        # values that node produced are still in memory. Kept beside a retired
        # runtime it is worse than absent: resume() prefers it over the explicit
        # start intent and continues an old position on an empty kernel.
        from backend.utils.checkpoint_paths import (
            checkpoint_sqlite_path,
        )

        checkpoint = checkpoint_sqlite_path(states_dir)
        for path in (
            checkpoint,
            checkpoint.with_name(checkpoint.name + "-wal"),
            checkpoint.with_name(checkpoint.name + "-shm"),
        ):
            if _quarantine_path(path) is not None:
                retired = True

        # The executor's own position, and its account of what it has already
        # done -- `step_number`, the message history, `code_history`,
        # `analysis_log`. Left behind it is worse than the graph checkpoint
        # above, because nothing catches it: `validate_checkpoint` compares
        # only `round_number` and `planning_stage`, and retirement preserves
        # the round, so the next run restored all of it over a namespace that
        # no longer holds anything it refers to -- while the agent was being
        # told the kernel was empty.
        #
        # The report generator's checkpoint (`_report_checkpoint.json`) is
        # deliberately not included: it resumes from session data, not from
        # kernel state, so a retired runtime does not invalidate it.
        executor_checkpoint = states_dir / f"{session_id}_bioinfo_checkpoint.json"
        if _quarantine_path(executor_checkpoint) is not None:
            retired = True
    return retired
