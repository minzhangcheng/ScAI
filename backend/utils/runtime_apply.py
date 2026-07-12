"""Stage 1 runtime-state safety: authoritative status, choice resolution, and a
crash-recoverable journaled apply of a snapshot's kernel pkl to the live path.

This module is intentionally free of workflow/route imports so it can be unit
tested and reused by both the restore route and sandbox startup recovery.
"""
import hashlib
import hmac as _hmac
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Literal, Optional

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


def _snapshot_zst(session_id: str, snapshot) -> Path:
    from .snapshot import SnapshotManager
    return SnapshotManager(session_id).snapshots_dir / f"{snapshot.id}_session.pkl.zst"


def compute_runtime_status(snapshot, session_id: str) -> RuntimeStatus:
    """Compute the effective runtime status from files, not the stored flag alone.

    - state_only: snapshot was created without runtime state;
    - missing:    metadata claims runtime but the .zst is absent;
    - corrupt:    .zst present but not a valid zstd frame;
    - complete:   .zst present and a valid zstd frame.

    Stage 1 does a cheap zstd-magic check only. A file whose magic is intact but
    whose body is truncated/corrupt reads as `complete` here; that residual
    corruption is still caught fail-closed at materialization time by
    `apply_runtime_pkl` (decompression error) and `verify_live_pkl_or_raise`
    (HMAC), so it never rehydrates a bad kernel. Full decompress/HMAC-level
    corruption classification at status time is Stage 3 (preflight).
    """
    if not getattr(snapshot, "runtime_state_included", False):
        return "state_only"
    zst = _snapshot_zst(session_id, snapshot)
    if not zst.exists():
        return "missing"
    try:
        with open(zst, "rb") as f:
            head = f.read(4)
    except OSError:
        return "corrupt"
    if head != ZSTD_MAGIC:
        return "corrupt"
    return "complete"


def resolve_runtime_choice(
    status: RuntimeStatus, runtime_choice: Optional[str]
) -> tuple[str, RuntimeRestoreStatus]:
    """Map (status, choice) to (action, persisted runtime_restore_status).

    Fail closed: an unavailable runtime with no choice raises RuntimeChoiceRequired,
    which the route surfaces as HTTP 409. There is no silent default.

    Actions: "apply" (materialize snapshot runtime), "reset" (quarantine live
    runtime, block resume), "keep" (retain live runtime, mark risk accepted).
    """
    if status == "complete":
        return ("apply", "complete")
    if runtime_choice is None:
        raise RuntimeChoiceRequired(status)
    if runtime_choice == "reset":
        return ("reset", "unavailable")
    if runtime_choice == "keep":
        return ("keep", "intentionally_kept")
    raise ValueError(f"unknown runtime_choice: {runtime_choice!r}")


def _journal_hmac(data: bytes) -> str:
    from ..utils.crypto import get_integrity_secret
    return _hmac.new(get_integrity_secret().encode(), data, hashlib.sha256).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def preflight_runtime_zst(session_id: str, snapshot, runtime_action: str) -> None:
    if runtime_action != "apply":
        return
    zst = _snapshot_zst(session_id, snapshot)
    if not zst.exists():
        raise RuntimeApplyError(f"[{session_id}] runtime_action='apply' but zst absent")
    import zstandard as zstd
    try:
        dctx = zstd.ZstdDecompressor()
        with open(zst, "rb") as src, open(os.devnull, "wb") as dst:
            dctx.copy_stream(src, dst)
    except Exception as exc:
        raise RuntimeApplyError(f"[{session_id}] zst preflight failed: {exc}") from exc


def _live_pkl(session_id: str) -> Path:
    return get_session_states(session_id) / f"{session_id}_session.pkl"


def _live_sig(session_id: str) -> Path:
    from .sandbox import _sig_path, _integrity_dir
    return _sig_path(_live_pkl(session_id), _integrity_dir(session_id))


def _journal_path(session_id: str) -> Path:
    from .sandbox import _integrity_dir
    return _integrity_dir(session_id) / "runtime_apply.journal.json"


def apply_runtime_pkl(session_id: str, snapshot_zst: Path) -> None:
    """Decompress a snapshot's pkl to the live path and re-sign its HMAC, as one
    journaled runtime pair. Recoverable across a crash between the two replaces.
    """
    import zstandard as zstd
    from .sandbox import _pkl_hmac_stream, _integrity_dir, _verify_pkl_integrity

    live = _live_pkl(session_id)
    sig = _live_sig(session_id)
    idir = _integrity_dir(session_id)
    idir.mkdir(parents=True, exist_ok=True)
    live.parent.mkdir(parents=True, exist_ok=True)

    staged = live.with_suffix(live.suffix + ".staged")
    dctx = zstd.ZstdDecompressor()
    with open(snapshot_zst, "rb") as src, open(staged, "wb") as dst:
        dctx.copy_stream(src, dst)
    staged_sig_value = _pkl_hmac_stream(staged)

    had_live = live.exists()
    bak_pkl = live.with_suffix(live.suffix + ".bak")
    bak_sig = sig.with_suffix(sig.suffix + ".bak")
    if had_live:
        shutil.copy2(live, bak_pkl)
    had_sig = sig.exists()
    if had_sig:
        shutil.copy2(sig, bak_sig)

    journal = {"state": "APPLYING", "had_live": had_live}
    pre_hmac = json.dumps(journal, sort_keys=True).encode()
    journal["hmac"] = _journal_hmac(pre_hmac)
    _atomic_write(_journal_path(session_id), json.dumps(journal).encode())

    try:
        os.replace(staged, live)
        _atomic_write(sig, staged_sig_value.encode())
        if not _verify_pkl_integrity(live, idir):
            raise RuntimeApplyError(f"[{session_id}] restored pkl failed integrity after apply")
    except Exception:
        recover_runtime_apply_journal(session_id)  # roll back to the backed-up pair
        raise

    journal["state"] = "COMMITTED"
    journal["hmac"] = _journal_hmac(json.dumps({"state": "COMMITTED", "had_live": had_live}, sort_keys=True).encode())
    _atomic_write(_journal_path(session_id), json.dumps(journal).encode())
    for p in (bak_pkl, bak_sig):
        p.unlink(missing_ok=True)
    _journal_path(session_id).unlink(missing_ok=True)


def quarantine_live_runtime(session_id: str) -> None:
    """reset choice: move the live pkl + signature aside so no stale runtime loads."""
    live = _live_pkl(session_id)
    sig = _live_sig(session_id)
    for p in (live, sig):
        if p.exists():
            p.replace(p.with_suffix(p.suffix + ".quarantine"))


def recover_runtime_apply_journal(session_id: str) -> None:
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
    stored_hmac = journal.pop("hmac", None)
    if stored_hmac is not None:
        pre_hmac = json.dumps(journal, sort_keys=True).encode()
        if not _hmac.compare_digest(_journal_hmac(pre_hmac), stored_hmac):
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


def is_forward_resume_blocked(runtime_restore_status: str) -> bool:
    """Only "unavailable" blocks forward resume. "intentionally_kept" is a user
    override that proceeds under a persistent warning; "complete" is normal."""
    return runtime_restore_status == "unavailable"


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


def clear_runtime_restore_status(session_id: str) -> None:
    from ..state import sessions
    state = sessions.get(session_id)
    if state is not None and getattr(state, "runtime_restore_status", "complete") != "complete":
        state.runtime_restore_status = "complete"


# -- Stage 3 restore helpers (atomic pkl/h5ad replacement) -------------------


def _states_dir(session_id: str) -> Path:
    return get_session_states(session_id)


def atomic_replace_pkl(session_id: str, src_pkl_path: str) -> str:
    """Atomically replace the live session.pkl with a staged copy.

    Re-signs the HMAC after replacement.  Returns the SHA-256 of the
    written pkl content.
    """
    from .sandbox import _pkl_hmac_stream, _integrity_dir, _verify_pkl_integrity

    src = Path(src_pkl_path)
    if not src.exists():
        raise RuntimeApplyError(
            f"[{session_id}] staged pkl not found: {src_pkl_path}"
        )

    live = _live_pkl(session_id)
    sig = _live_sig(session_id)
    idir = _integrity_dir(session_id)
    idir.mkdir(parents=True, exist_ok=True)
    live.parent.mkdir(parents=True, exist_ok=True)

    # Atomic copy to live pkl
    tmp = live.with_suffix(live.suffix + f".{os.getpid()}.staged")
    shutil.copy2(src, tmp)
    tmp_fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(tmp_fd)
    finally:
        os.close(tmp_fd)
    os.replace(str(tmp), str(live))
    dir_fd = os.open(str(live.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    # Re-sign HMAC
    sig_value = _pkl_hmac_stream(live)
    _atomic_write(sig, sig_value.encode())

    if not _verify_pkl_integrity(live, idir):
        raise RuntimeApplyError(
            f"[{session_id}] replaced pkl failed integrity check"
        )

    import hashlib
    return hashlib.sha256(live.read_bytes()).hexdigest()


def atomic_replace_h5ad(session_id: str, src_h5ad_path: str, var_name: str = "adata") -> str:
    """Atomically place a staged h5ad into the session's states directory.

    The file is written as ``{states_dir}/{var_name}.h5ad``.
    Returns the SHA-256 of the written h5ad content.
    """
    src = Path(src_h5ad_path)
    if not src.exists():
        raise RuntimeApplyError(
            f"[{session_id}] staged h5ad not found: {src_h5ad_path}"
        )

    states = _states_dir(session_id)
    states.mkdir(parents=True, exist_ok=True)
    dst = states / f"{var_name}.h5ad"

    # Atomic copy
    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.staged")
    shutil.copy2(src, tmp)
    tmp_fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(tmp_fd)
    finally:
        os.close(tmp_fd)
    os.replace(str(tmp), str(dst))
    dir_fd = os.open(str(states), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    import hashlib
    return hashlib.sha256(dst.read_bytes()).hexdigest()
