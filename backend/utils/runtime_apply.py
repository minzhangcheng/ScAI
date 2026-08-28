"""Stage 1 runtime-state safety: authoritative status, choice resolution, and a
crash-recoverable journaled apply of a snapshot's kernel pkl to the live path.

This module is intentionally free of workflow/route imports so it can be unit
tested and reused by both the restore route and sandbox startup recovery.
"""
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Literal, Optional

from ..config import get_session_states
from backend.services.snapshot_v2.runtime_names import (
    runtime_h5ad_filename,
    runtime_h5ad_path,
    validate_runtime_global_name,
)

logger = logging.getLogger(__name__)

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

# Staging temp of atomic_replace_pkl / atomic_replace_h5ad: the pid is what
# tells an abandoned partial apart from a copy that is still in flight.
_STAGED_TEMP_SUFFIX = re.compile(r"\.(\d+)\.staged$")

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
    from ..utils.crypto import sign_hmac_sha256
    return sign_hmac_sha256(data)


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


def commit_runtime_persistence_metadata(
    session_id: str,
    *,
    source: str = "snapshot_apply",
) -> dict:
    """Stamp explicit committed-generation metadata after a verified apply."""
    from .sandbox import NotebookState

    states = get_session_states(session_id)
    session_json = states.parent / "session.json"
    state_payload = {}
    if session_json.exists():
        try:
            state_payload = json.loads(session_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state_payload = {}
    sandbox_state = state_payload.get("sandbox_state") or {}
    revision = int(
        sandbox_state.get("runtime_revision")
        or sandbox_state.get("saved_runtime_revision")
        or 0
    )
    from backend.services.session_lifecycle import read_lifecycle_record

    lifecycle = read_lifecycle_record(session_id) or {}
    notebook = NotebookState(
        session_id,
        str(states),
        lifecycle_generation=int(lifecycle.get("generation") or 0),
    )
    live_pkl = notebook.session_path
    from .sandbox import _sha256_file

    runtime_digest = _sha256_file(live_pkl) if live_pkl.is_file() else None
    persistence = notebook.update_runtime_persistence(
        "current",
        current_revision=revision,
        saved_revision=revision,
        commit_source=source,
        runtime_digest=runtime_digest,
    )
    if state_payload:
        state_payload["runtime_persistence"] = persistence
        sandbox_state["runtime_revision"] = revision
        sandbox_state["saved_runtime_revision"] = revision
        sandbox_state["runtime_persistence"] = persistence
        state_payload["sandbox_state"] = sandbox_state
        from backend.utils.files import atomic_write_bytes

        atomic_write_bytes(
            session_json,
            json.dumps(state_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return persistence


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

    from backend.utils.files import atomic_write_bytes
    journal = {"state": "APPLYING", "had_live": had_live}
    pre_hmac = json.dumps(journal, sort_keys=True).encode()
    journal["hmac"] = _journal_hmac(pre_hmac)
    atomic_write_bytes(_journal_path(session_id), json.dumps(journal).encode())

    try:
        os.replace(staged, live)
        atomic_write_bytes(sig, staged_sig_value.encode())
        if not _verify_pkl_integrity(live, idir):
            raise RuntimeApplyError(f"[{session_id}] restored pkl failed integrity after apply")
    except Exception:
        recover_runtime_apply_journal(session_id)  # roll back to the backed-up pair
        raise

    journal["state"] = "COMMITTED"
    journal["hmac"] = _journal_hmac(json.dumps({"state": "COMMITTED", "had_live": had_live}, sort_keys=True).encode())
    atomic_write_bytes(_journal_path(session_id), json.dumps(journal).encode())
    for p in (bak_pkl, bak_sig):
        p.unlink(missing_ok=True)
    _journal_path(session_id).unlink(missing_ok=True)


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
    return candidate


def quarantine_live_runtime_pkl(session_id: str) -> None:
    """Move the live dill pkl and its signature aside, leaving h5ads alone.

    The pkl-only half of :func:`quarantine_live_runtime`, for callers that own
    the h5ad sidecar set separately (rollback drives it through
    :func:`finalize_live_runtime_h5ad_set`).
    """
    for p in (_live_pkl(session_id), _live_sig(session_id)):
        _quarantine_path(p)


def quarantine_live_runtime(session_id: str) -> None:
    """reset choice: quarantine live pkl, signature, h5ad sidecars, and manifest.

    After Task 3 the sandbox rehydrates from ``runtime_h5ad_manifest.json`` even
    without a pkl.  Leaving those sidecars in place after ``reset`` would break
    the reset contract by rehydrating stale AnnData into a fresh kernel.
    """
    live = _live_pkl(session_id)
    sig = _live_sig(session_id)
    for p in (live, sig):
        _quarantine_path(p)

    states = _states_dir(session_id)
    _quarantine_path(runtime_h5ad_manifest_path(session_id))

    # Quarantine every live *.h5ad confined to the states directory (safer than
    # trusting only manifest-referenced names when the manifest is corrupt).
    if states.is_dir():
        try:
            states_resolved = states.resolve()
        except OSError:
            states_resolved = states
        for p in states.glob("*.h5ad"):
            if not p.is_file():
                continue
            try:
                if p.resolve().parent != states_resolved:
                    continue
            except OSError:
                continue
            _quarantine_path(p)


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


RUNTIME_H5AD_MANIFEST_FILENAME = "runtime_h5ad_manifest.json"


def runtime_h5ad_manifest_path(
    session_id: str,
    *,
    states_dir: str | Path | None = None,
) -> Path:
    """Return the live h5ad sidecar manifest path for a session."""
    root = Path(states_dir) if states_dir is not None else _states_dir(session_id)
    return root / RUNTIME_H5AD_MANIFEST_FILENAME


def write_runtime_h5ad_manifest(
    session_id: str,
    entries: dict[str, str],
    *,
    authoritative_pkl_names: set[str] | list[str] | None = None,
    states_dir: str | Path | None = None,
) -> Path:
    """Atomically write the live h5ad sidecar manifest.

    entries maps validated runtime variable name to the SHA-256 of the live
    h5ad file written as `{name}.h5ad`.
    """
    from backend.utils.files import atomic_write_bytes

    manifest_entries = {}
    for name, sha256 in sorted(entries.items()):
        filename = runtime_h5ad_filename(name)
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise RuntimeApplyError(
                f"[{session_id}] invalid h5ad sha256 for {name!r}: {sha256!r}"
            )
        manifest_entries[name] = {
            "path": filename,
            "sha256": sha256,
        }
    payload = {
        "version": 2 if authoritative_pkl_names is not None else 1,
        "entries": manifest_entries,
    }
    if authoritative_pkl_names is not None:
        validated_names = []
        for name in sorted(set(authoritative_pkl_names)):
            try:
                validated_names.append(validate_runtime_global_name(name))
            except ValueError as exc:
                raise RuntimeApplyError(
                    f"[{session_id}] invalid authoritative pkl name: {name!r}"
                ) from exc
        payload["authoritative_pkl_names"] = validated_names
    path = runtime_h5ad_manifest_path(session_id, states_dir=states_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
    )
    return path


def _staged_copy_replace(src: Path, dst: Path) -> None:
    """Copy *src* onto *dst* through a ``<dst>.<pid>.staged`` temp, then fsync.

    The temp name is one no sweeper in the system can match: both states-dir
    sweeps glob ``*.h5ad``, GC walks ``objects/`` only, and the snapshot staging
    sweeps operate on ``<session>/staging``. Quota accounting sums ``objects/``
    too, so a dataset-sized partial left here by a failed copy — ENOSPC is the
    realistic case, since restore holds the object-store copy, the staged copy
    and the live copy at once — is invisible, unreclaimable, and packed verbatim
    into every v1 export archive. Unlink it while we still know its name.
    """
    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.staged")
    try:
        shutil.copy2(src, tmp)
        tmp_fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(tmp_fd)
        finally:
            os.close(tmp_fd)
        os.replace(str(tmp), str(dst))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dir_fd = os.open(str(dst.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


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

    _staged_copy_replace(src, live)

    # Re-sign HMAC
    from backend.utils.files import atomic_write_bytes
    sig_value = _pkl_hmac_stream(live)
    atomic_write_bytes(sig, sig_value.encode())

    if not _verify_pkl_integrity(live, idir):
        raise RuntimeApplyError(
            f"[{session_id}] replaced pkl failed integrity check"
        )

    from backend.services.snapshot_v2.object_store import _hash_file
    return _hash_file(live)


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

    try:
        safe_var_name = validate_runtime_global_name(var_name)
    except ValueError as exc:
        raise RuntimeApplyError(
            f"[{session_id}] invalid runtime variable name: {var_name!r}"
        ) from exc

    states = _states_dir(session_id)
    states.mkdir(parents=True, exist_ok=True)
    dst = runtime_h5ad_path(states, safe_var_name)

    _staged_copy_replace(src, dst)

    from backend.services.snapshot_v2.object_store import _hash_file
    return _hash_file(dst)


def _reclaim_orphan_staged_temps(states: Path, states_resolved: Path) -> None:
    """Unlink ``*.<pid>.staged`` partials whose writing process is gone.

    ``atomic_replace_pkl``/``atomic_replace_h5ad`` unlink their own temp when the
    copy fails, but a process killed mid-copy cannot, and nothing else ever
    will — the `*.h5ad` sweeps here and in ``quarantine_live_runtime`` cannot
    match the suffix. A live pid means an in-flight copy, ours or a concurrent
    process's, and unlinking that would sabotage a running restore, so liveness
    is what decides rather than the name alone.
    """
    for p in states.glob("*.staged"):
        if not p.is_file():
            continue
        match = _STAGED_TEMP_SUFFIX.search(p.name)
        if match is None:
            continue
        try:
            if p.resolve().parent != states_resolved:
                continue
        except OSError:
            continue
        try:
            os.kill(int(match.group(1)), 0)
        except ProcessLookupError:
            pass  # writer is gone: the partial can never be completed
        except OSError:
            continue  # alive but not signalable by us
        else:
            continue  # still running
        try:
            p.unlink()
        except OSError:
            logger.warning("Could not reclaim orphan staged temp: %s", p)


def _quarantine_unreferenced_live_h5ads(
    session_id: str,
    keep_names: set[str],
    *,
    states_dir: str | Path | None = None,
) -> None:
    """Quarantine live ``*.h5ad`` files not in *keep_names* (confined to states)."""
    states = Path(states_dir) if states_dir is not None else _states_dir(session_id)
    if not states.is_dir():
        return
    try:
        states_resolved = states.resolve()
    except OSError:
        states_resolved = states

    _reclaim_orphan_staged_temps(states, states_resolved)

    keep_filenames = set()
    for name in keep_names:
        try:
            keep_filenames.add(runtime_h5ad_filename(name))
        except ValueError:
            continue

    for p in states.glob("*.h5ad"):
        if not p.is_file():
            continue
        try:
            if p.resolve().parent != states_resolved:
                continue
        except OSError:
            continue
        if p.name in keep_filenames:
            continue
        _quarantine_path(p)


def quarantine_unreferenced_runtime_h5ads(
    session_id: str,
    keep_names: set[str],
    *,
    states_dir: str | Path | None = None,
) -> None:
    """Public bounded cleanup after an authoritative manifest is committed."""
    _quarantine_unreferenced_live_h5ads(
        session_id,
        keep_names,
        states_dir=states_dir,
    )


def finalize_live_runtime_h5ad_set(
    session_id: str,
    h5ad_hashes: dict[str, str],
) -> None:
    """Make the live sidecar set and its manifest describe exactly *h5ad_hashes*.

    Sandbox restore refuses to rehydrate an h5ad whose sha256 disagrees with its
    manifest entry, so the files and ``runtime_h5ad_manifest.json`` must never be
    updated apart — and a sidecar nobody names must not be left lying in the
    states dir, where the `*.h5ad` sweep would still find it.

    Split out for callers that already placed the bytes themselves: recovery
    rollback streams them straight out of the object store, so completing the set
    must not cost a second full-size copy.
    """
    _quarantine_unreferenced_live_h5ads(session_id, set(h5ad_hashes))
    write_runtime_h5ad_manifest(session_id, h5ad_hashes)


def install_live_runtime_h5ad_set(
    session_id: str,
    staged_h5ads: dict[str, str],
) -> dict[str, str]:
    """Install the target live h5ad set and rewrite the manifest to match.

    *staged_h5ads* maps runtime variable name → staged h5ad path.  An empty
    mapping is valid (pkl-only / residual-only complete runtime): previous live
    ``*.h5ad`` sidecars are quarantined and the manifest is rewritten with
    ``entries: {}`` so sandbox restore cannot rehydrate stale AnnData.
    """
    h5ad_hashes: dict[str, str] = {}
    for var_name, h5ad_path in staged_h5ads.items():
        h5ad_hashes[var_name] = atomic_replace_h5ad(
            session_id, h5ad_path, var_name=var_name,
        )
    finalize_live_runtime_h5ad_set(session_id, h5ad_hashes)
    return h5ad_hashes
