"""Authenticated regular-save transaction primitives."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256
from .workspace_generation import publish_workspace_generation

logger = logging.getLogger("backend.utils.sandbox_parts.shared")


def _phase_begin(phase_recorder):
    """Start one optional telemetry interval, or return None."""
    if phase_recorder is None:
        return None
    return time.monotonic()


def _phase_record(phase_recorder, name: str, started) -> None:
    """Close one exclusive telemetry interval; never raises into the commit."""
    if phase_recorder is None or started is None:
        return
    try:
        phase_recorder.add(name, time.monotonic() - started)
    except Exception:
        logger.debug("runtime save phase telemetry %s failed", name, exc_info=True)

def _sig_path(pkl: Path, integrity_dir: Path) -> Path:
    return integrity_dir / (pkl.name + ".sig")


def _pkl_hmac(data: bytes) -> str:
    return sign_hmac_sha256(data)


def _pkl_hmac_stream(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hmac.new(get_integrity_secret().encode(), b"", hashlib.sha256)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_pkl_signature(pkl: Path, integrity_dir: Path) -> None:
    integrity_dir.mkdir(parents=True, exist_ok=True)
    _sig_path(pkl, integrity_dir).write_text(_pkl_hmac_stream(pkl))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


#: The three files a committed generation has to be paired with to be
#: continuable, as {source key: (component kind, name inside the pair)}. Spec
#: B2-1. The names are fixed so recovery can rebuild each path from the kind
#: and the journal's revision rather than trusting a path out of the journal.
#: Directory under a session's states/ that holds one generation's pair.
GENERATION_PAIR_DIRNAME = "generation"

_PAIRED_COMPONENTS = {
    "state": ("paired_state", "state.json"),
    "checkpoint": ("paired_checkpoint", "checkpoints.sqlite"),
    "executor_checkpoint": ("paired_executor_checkpoint", "executor_checkpoint.json"),
}


def paired_generation_sources(sandbox) -> Dict[str, Path]:
    """The live files a generation has to be captured beside. Spec B2-1.

    Three, not two: §3.11 -- the executor's own step checkpoint holds its
    position and its account of what it has already done, and is bound to the
    generation by nothing else. It is absent at a round boundary reached after
    the executor finished, and absent is not an error.

    Here rather than in `runtime_save` because it names the same three things
    `_PAIRED_COMPONENTS` above does, and the two drifting apart is the whole
    failure mode this pairing exists to prevent.
    """
    from backend.utils.checkpoint_paths import checkpoint_sqlite_path

    states = Path(sandbox.notebook_state.state_dir)
    try:
        from backend.agents.bioinformatics.executor_checkpoint import (
            is_executor_checkpoint_name,
        )

        active_name = str(
            getattr(sandbox, "active_executor_checkpoint_name", "") or ""
        )
        if active_name and Path(active_name).name == active_name and (
            is_executor_checkpoint_name(
                active_name, session_id=str(sandbox.session_id),
            )
        ):
            executor_checkpoint = states / active_name
        else:
            # No active owner means there is no safe basis for choosing one of
            # several scoped transcripts.  Retain only the narrow legacy path.
            executor_checkpoint = (
                states / f"{sandbox.session_id}_bioinfo_checkpoint.json"
            )
    except Exception:
        executor_checkpoint = states / f"{sandbox.session_id}_bioinfo_checkpoint.json"
    return {
        "state": Path(sandbox.notebook_state.state_path),
        "checkpoint": checkpoint_sqlite_path(states),
        "executor_checkpoint": executor_checkpoint,
    }


def _copy_sqlite_durable(src: Path, dst: Path) -> None:
    """Copy a live SQLite database through its own backup API.

    Not a byte copy: LangGraph writes the checkpoint database in WAL mode, so
    the main file on its own is not a consistent database and its `-wal`
    companion is not covered by this transaction. `checkpoint_identity` reads
    the same file through a read-only URI connection for the same reason.
    """
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(dst)) as target:
            source.backup(target)
    with open(dst, "rb") as handle:
        os.fsync(handle.fileno())


#: The workspace inventory a generation is paired with, inside the pair
#: directory so it is pruned with it. Not a fourth journalled component: it is
#: written after the transaction commits, so a crash in between leaves a pair
#: with no inventory, and a rollback with no inventory simply cleans nothing --
#: which is what every pair written before this existed does. Failing the save
#: over a hygiene record would be the wrong trade.
WORKSPACE_INVENTORY_NAME = "workspace.json"

#: Bumped when the shape changes. A reader that does not recognise the version
#: cleans nothing rather than guessing, because the consequence of guessing
#: here is deleting a file.
WORKSPACE_INVENTORY_SCHEMA = 1


def _prune_older_generations(generation_root: Path, *, keep: int) -> None:
    """Retention is one (B2-1), and the newest is durable before this runs.

    A generation is not three files any more: it also holds the workspace copy
    the rollback restores from, which is a directory. `unlink` raises on one,
    the exception left the commit through this call, and the save that was
    otherwise complete was recorded as failed -- seen in production before a
    test caught it, because a single save never prunes a generation that has
    one.
    """
    if not generation_root.is_dir():
        return
    for entry in generation_root.iterdir():
        if not entry.is_dir() or entry.name == str(keep):
            continue
        for child in entry.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    _remove_durable(child)
            except OSError:
                logger.warning(
                    "Could not remove %s from a superseded generation", child,
                    exc_info=True,
                )
        try:
            entry.rmdir()
        except OSError:
            logger.warning("Could not remove a superseded generation pair: %s", entry)


def _regular_save_journal_path(live_pkl: Path) -> Path:
    return live_pkl.with_name(f"{live_pkl.name}.regular-save.json")


def _regular_save_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.regular-save.bak")


def _journal_body(payload: Dict[str, Any]) -> bytes:
    unsigned = {k: v for k, v in payload.items() if k != "journal_hmac"}
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_regular_save_journal(path: Path, payload: Dict[str, Any]) -> None:
    from backend.utils.files import atomic_write_json

    signed = {k: v for k, v in payload.items() if k != "journal_hmac"}
    signed["journal_hmac"] = sign_hmac_sha256(_journal_body(signed))
    atomic_write_json(path, signed, canonical=True)


def _read_regular_save_journal(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("regular runtime save journal is not an object")
    expected = str(payload.get("journal_hmac") or "")
    if not expected or not hmac.compare_digest(
        expected, sign_hmac_sha256(_journal_body(payload))
    ):
        raise RuntimeError("regular runtime save journal HMAC is invalid")
    return payload


def _copy_durable(src: Path, dst: Path) -> None:
    """Copy one transaction component through a bounded sibling temporary."""
    from backend.utils.files import fsync_directory
    from . import shared as shared_facade

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{dst.name}.",
        suffix=".tmp",
        dir=str(dst.parent),
    )
    temporary = Path(temporary_name)
    owned_fd: int | None = fd
    try:
        with open(src, "rb") as source:
            target = os.fdopen(fd, "wb")
            owned_fd = None
            with target:
                while True:
                    chunk = source.read(
                        shared_facade._REGULAR_SAVE_COPY_CHUNK_BYTES
                    )
                    if not chunk:
                        break
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        os.replace(temporary, dst)
        fsync_directory(dst.parent)
    except BaseException:
        if owned_fd is not None:
            os.close(owned_fd)
        temporary.unlink(missing_ok=True)
        raise


def _remove_durable(path: Path) -> None:
    from backend.utils.files import fsync_directory

    if path.exists():
        path.unlink()
        fsync_directory(path.parent)


def _regular_component(
    kind: str,
    path: Path,
    *,
    previous_exists: bool,
    authoritative: bool = False,
) -> Dict[str, Any]:
    """Describe one file in the regular-save atomicity boundary.

    Paths are represented by a kind plus a basename rather than by an absolute
    path.  Recovery reconstructs the canonical path from its trusted arguments;
    a signed journal therefore cannot be used as an arbitrary-file restore
    primitive even if an old installation leaked its integrity secret.
    """
    return {
        "kind": kind,
        "name": path.name,
        "previous_exists": bool(previous_exists),
        "previous_sha256": (
            _sha256_file(path) if previous_exists and path.is_file() else None
        ),
        "authoritative": bool(authoritative),
        "target_exists": None,
        "target_sha256": None,
    }


def _regular_component_paths(
    payload: Dict[str, Any],
    live_pkl: Path,
    live_sig: Path,
    metadata_path: Path | None,
    manifest_path: Path | None,
    runtime_manifest_path: Path | None = None,
    generation_root: Path | None = None,
) -> List[Tuple[Dict[str, Any], Path]]:
    """Resolve signed component descriptors to canonical trusted paths."""
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        # Compatibility with version-1 journals written by 0.38.9 prereleases.
        legacy_paths = [live_pkl, live_sig]
        if metadata_path is not None:
            legacy_paths.append(metadata_path)
        if manifest_path is not None:
            legacy_paths.append(manifest_path)
        result = []
        for index, path in enumerate(legacy_paths):
            result.append(({
                "kind": ("pkl", "sig", "metadata", "manifest")[index],
                "name": path.name,
                "previous_exists": bool(payload.get(f"had_component_{index}")),
                "previous_sha256": None,
                "authoritative": True,
                "target_exists": True,
                "target_sha256": (
                    payload.get("target_digest") if index == 0 else None
                ),
            }, path))
        return result

    from backend.utils.runtime_manifest import runtime_manifest_sig_path

    known = {
        "pkl": live_pkl,
        "sig": live_sig,
        "metadata": metadata_path,
        "manifest": manifest_path,
        # The generation manifest and its detached signature roll back together
        # with the dump they describe: a manifest surviving a rollback would
        # declare names the restored dump does not contain.
        "runtime_manifest": runtime_manifest_path,
        "runtime_manifest_sig": (
            runtime_manifest_sig_path(runtime_manifest_path)
            if runtime_manifest_path is not None
            else None
        ),
    }
    # Rebuilt from the journal's own revision plus a trusted root, never from a
    # path inside the journal -- the same rule as every kind above.
    pair_dir = (
        Path(generation_root) / str(int(payload.get("target_revision") or 0))
        if generation_root is not None
        else None
    )
    for _key, (kind, filename) in _PAIRED_COMPONENTS.items():
        known[kind] = pair_dir / filename if pair_dir is not None else None
    resolved: List[Tuple[Dict[str, Any], Path]] = []
    seen: set[Tuple[str, str]] = set()
    for raw in raw_components:
        if not isinstance(raw, dict):
            raise RuntimeError("regular runtime save component is not an object")
        component = dict(raw)
        kind = str(component.get("kind") or "")
        name = str(component.get("name") or "")
        key = (kind, name)
        if key in seen:
            raise RuntimeError(f"duplicate regular runtime component: {kind}:{name}")
        seen.add(key)
        if kind not in known:
            # A journal written before the h5ad sidecar was removed can still
            # list components of a kind this build no longer installs. Skipping
            # one leaves the sidecar where it is; failing the resolution would
            # abandon the whole recovery, including the pkl it is really about.
            logger.warning(
                "Ignoring unknown component kind %r in a runtime save journal",
                kind,
            )
            continue
        path = known.get(kind)
        if path is None or name != path.name:
            raise RuntimeError(
                f"regular runtime component does not match canonical {kind} path"
            )
        resolved.append((component, path))
    return resolved



def _preflight_regular_rollback(
    components: List[Tuple[Dict[str, Any], Path]],
) -> None:
    """Authenticate every previous-generation backup before mutating targets."""
    for component, path in components:
        if not bool(component.get("previous_exists")):
            continue
        backup = _regular_save_backup_path(path)
        if not backup.is_file():
            raise RuntimeError(f"regular runtime save backup missing: {backup.name}")
        expected = component.get("previous_sha256")
        if expected and _sha256_file(backup) != expected:
            raise RuntimeError(
                f"regular runtime save backup digest mismatch: {backup.name}"
            )


def _restore_regular_components(
    components: List[Tuple[Dict[str, Any], Path]],
    *,
    integrity_dir: Path,
) -> None:
    _preflight_regular_rollback(components)
    for component, path in components:
        if bool(component.get("previous_exists")):
            _copy_durable(_regular_save_backup_path(path), path)
        else:
            _remove_durable(path)
    for component, path in components:
        expected = component.get("previous_sha256")
        if bool(component.get("previous_exists")) and expected:
            if not path.is_file() or _sha256_file(path) != expected:
                raise RuntimeError(
                    f"restored regular runtime component failed verification: {path.name}"
                )
    pkl_was_present = any(
        component.get("kind") == "pkl" and component.get("previous_exists")
        for component, _path in components
    )
    if pkl_was_present and not _verify_pkl_integrity(
        next(path for component, path in components if component.get("kind") == "pkl"),
        integrity_dir,
    ):
        raise RuntimeError("restored regular runtime pkl/signature pair is invalid")




def _regular_metadata_matches(
    metadata_path: Path | None,
    *,
    revision: int,
    digest: str,
) -> bool:
    if metadata_path is None or not metadata_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        persistence = payload.get("runtime_persistence") or {}
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return bool(
        persistence.get("status") == "current"
        and persistence.get("commit_source") == "regular_save"
        and int(persistence.get("current_revision") or 0) == revision
        and int(persistence.get("saved_revision") or 0) == revision
        and persistence.get("runtime_digest") == digest
    )


def _cleanup_regular_save_transaction(
    journal_path: Path,
    components: List[Path],
    fault_hook=None,
) -> None:
    for index, component in enumerate(components):
        _remove_durable(_regular_save_backup_path(component))
        if fault_hook is not None:
            fault_hook(f"cleanup_backup_{index}")
    if len(components) >= 2:
        _remove_durable(
            components[0].with_name(f"{components[0].name}.tmp")
        )
        _remove_durable(
            components[1].with_name(
                f"{components[1].name}.regular-save.tmp"
            )
        )
    _remove_durable(journal_path)
    if fault_hook is not None:
        fault_hook("cleanup_complete")


def recover_regular_save_transaction(
    live_pkl: Path,
    integrity_dir: Path,
    *,
    metadata_path: Path | None = None,
    manifest_path: Path | None = None,
    runtime_manifest_path: Path | None = None,
    generation_root: Path | None = None,
    session_id: str = "",
) -> str | None:
    """Recover a process crash in regular runtime persistence.

    Before the durable COMMITTED marker the old generation wins. Afterwards
    the verified new generation wins and cleanup is completed idempotently.
    """
    journal_path = _regular_save_journal_path(live_pkl)
    if not journal_path.exists():
        return None
    payload = _read_regular_save_journal(journal_path)
    live_sig = _sig_path(live_pkl, integrity_dir)
    resolved_components = _regular_component_paths(
        payload, live_pkl, live_sig, metadata_path, manifest_path,
        runtime_manifest_path, generation_root,
    )
    component_paths = [path for _component, path in resolved_components]

    status = str(payload.get("status") or "")
    target_digest = str(payload.get("target_digest") or "")
    target_revision = int(payload.get("target_revision") or 0)
    if status == "COMMITTED":
        try:
            for component, path in resolved_components:
                if not bool(component.get("authoritative", True)):
                    continue
                expected_exists = component.get("target_exists")
                if expected_exists is not True or not path.is_file():
                    raise RuntimeError(
                        f"committed regular runtime component is missing: {path.name}"
                    )
                expected_digest = component.get("target_sha256")
                if expected_digest and _sha256_file(path) != expected_digest:
                    raise RuntimeError(
                        f"committed regular runtime component is corrupt: {path.name}"
                    )
            if not (
                live_pkl.is_file()
                and _sha256_file(live_pkl) == target_digest
                and _verify_pkl_integrity(live_pkl, integrity_dir)
            ):
                raise RuntimeError("committed runtime pkl/signature pair is corrupt")
            if metadata_path is not None and not _regular_metadata_matches(
                metadata_path,
                revision=target_revision,
                digest=target_digest,
            ):
                raise RuntimeError("committed runtime metadata is inconsistent")
            target_valid = True
        except RuntimeError:
            logger.exception(
                "Committed regular runtime generation failed verification"
            )
            target_valid = False
        if not target_valid:
            # The commit marker is durable only after all target components are
            # verified. If they no longer verify and backups remain, prefer the
            # last good generation; otherwise fail closed for operator repair.
            try:
                _preflight_regular_rollback(resolved_components)
            except RuntimeError as exc:
                raise RuntimeError(
                    "committed regular runtime generation is corrupt and its "
                    "previous-generation backups are incomplete"
                ) from exc
            status = "APPLYING"
        else:
            _cleanup_regular_save_transaction(journal_path, component_paths)
            return "forward"

    # PREPARED/APPLYING, or a committed target that failed verification.
    component_paths = [path for _component, path in resolved_components]
    _restore_regular_components(
        resolved_components,
        integrity_dir=integrity_dir,
    )
    _cleanup_regular_save_transaction(journal_path, component_paths)
    return "rollback"


def _commit_pkl_transaction(
    staged_pkl: Path,
    live_pkl: Path,
    integrity_dir: Path,
    commit_metadata,
    *,
    metadata_path: Path | None = None,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: Dict[str, Any] | None = None,
    target_revision: int = 0,
    authoritative_pkl_names: set[str] | None = None,
    fault_hook=None,
    generation_root: Path | None = None,
    paired_sources: Dict[str, Path | None] | None = None,
    workspace_root: Path | None = None,
    phase_recorder=None,
) -> None:
    """Install a runtime generation with crash-recoverable commit semantics.

    ``runtime_manifest`` is the generation's own declaration -- what it saved,
    what it skipped, and which workflow state it belongs to. It is installed by
    this transaction rather than beside it, because a manifest that can survive
    a rollback of the dump it describes is worse than no manifest at all: it
    would declare names the restored dump does not contain, and integrity would
    then report corruption on a generation that is intact.
    """
    from . import shared as shared_facade

    copy_durable = shared_facade._copy_durable
    integrity_dir.mkdir(parents=True, exist_ok=True)
    live_pkl.parent.mkdir(parents=True, exist_ok=True)
    live_sig = _sig_path(live_pkl, integrity_dir)
    staged_sig = live_sig.with_name(f"{live_sig.name}.regular-save.tmp")
    _descriptors_started = _phase_begin(phase_recorder)
    component_records: List[Tuple[Dict[str, Any], Path]] = [
        (_regular_component(
            "pkl", live_pkl,
            previous_exists=live_pkl.is_file(), authoritative=True,
        ), live_pkl),
        (_regular_component(
            "sig", live_sig,
            previous_exists=live_sig.is_file(), authoritative=True,
        ), live_sig),
    ]
    if metadata_path is not None:
        component_records.append((
            _regular_component(
                "metadata", metadata_path,
                previous_exists=metadata_path.is_file(), authoritative=True,
            ),
            metadata_path,
        ))
    if runtime_manifest_path is not None:
        from backend.utils.runtime_manifest import runtime_manifest_sig_path

        for kind, target in (
            ("runtime_manifest", runtime_manifest_path),
            ("runtime_manifest_sig", runtime_manifest_sig_path(runtime_manifest_path)),
        ):
            component_records.append((
                _regular_component(
                    kind, target,
                    previous_exists=target.is_file(), authoritative=True,
                ),
                target,
            ))

    # Registered before the journal, like every other component, so a crash
    # rolls the pair back with the generation it describes. A pair that
    # outlived its rollback would name a dump that is no longer on disk.
    pair_dir = (
        Path(generation_root) / str(int(target_revision))
        if generation_root is not None and paired_sources
        else None
    )
    pair_targets: List[Tuple[str, Path, Path]] = []
    pair_removals: List[Tuple[str, Path]] = []
    if pair_dir is not None:
        for key, (kind, filename) in _PAIRED_COMPONENTS.items():
            source = (paired_sources or {}).get(key)
            target = pair_dir / filename
            if source is None or not Path(source).is_file():
                # Absent is absent, not a failed copy. A round boundary reached
                # after the executor finished has no step checkpoint, and
                # demanding one would fail every save taken at one.
                #
                # But the directory is named by revision and is therefore
                # reused: a save at an unchanged revision -- which happens
                # whenever the graph advances without the kernel executing --
                # can find a *previous* save's file here. It describes a
                # position this generation no longer holds, a restore would
                # copy it back, and it would pass the exact-revision rule,
                # because the revision it records really is this generation's.
                # Removed inside the transaction, so a rollback puts it back.
                if target.is_file():
                    pair_removals.append((kind, target))
                    component_records.append((
                        _regular_component(kind, target, previous_exists=True),
                        target,
                    ))
                continue
            pair_targets.append((kind, Path(source), target))
            component_records.append((
                _regular_component(
                    kind, target, previous_exists=target.is_file(),
                ),
                target,
            ))

    _phase_record(
        phase_recorder, "previous_generation_backup", _descriptors_started,
    )
    journal_path = _regular_save_journal_path(live_pkl)
    _digest_started = _phase_begin(phase_recorder)
    target_digest = _sha256_file(staged_pkl)
    _phase_record(phase_recorder, "staged_digest_hmac", _digest_started)

    staged_sig_owned = True
    prejournal_backups: List[Path] = []
    journal = None
    try:
        _hmac_started = _phase_begin(phase_recorder)
        staged_sig.write_text(_pkl_hmac_stream(staged_pkl), encoding="utf-8")
        with open(staged_sig, "r+", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _phase_record(phase_recorder, "staged_digest_hmac", _hmac_started)
        # Backups are complete and durable before the journal exposes a transaction.
        _backup_started = _phase_begin(phase_recorder)
        for descriptor, component in component_records:
            backup = _regular_save_backup_path(component)
            backup.unlink(missing_ok=True)
            if descriptor["previous_exists"]:
                # Register before the copy: a copy implementation may publish
                # the destination and then fail while flushing its directory.
                prejournal_backups.append(backup)
                copy_durable(component, backup)
                if _sha256_file(backup) != descriptor["previous_sha256"]:
                    raise RuntimeError(
                        f"regular runtime save backup failed verification: {backup.name}"
                    )

        _phase_record(phase_recorder, "previous_generation_backup", _backup_started)

        journal = {
            "version": 2,
            "status": "APPLYING",
            "target_digest": target_digest,
            "target_revision": int(target_revision),
            "components": [descriptor for descriptor, _path in component_records],
        }
        _write_regular_save_journal(journal_path, journal)
    except BaseException:
        # No signed journal owns these products yet. Remove only the products
        # established by this attempt; after publication recovery owns cleanup.
        if not journal_path.exists():
            staged_pkl.unlink(missing_ok=True)
            if staged_sig_owned:
                staged_sig.unlink(missing_ok=True)
            for backup in prejournal_backups:
                backup.unlink(missing_ok=True)
        raise
    if fault_hook is not None:
        fault_hook("journal_applying")

    committed = False
    try:
        os.replace(staged_pkl, live_pkl)
        from backend.utils.files import fsync_directory

        fsync_directory(live_pkl.parent)
        if fault_hook is not None:
            fault_hook("pkl_replaced")
        os.replace(staged_sig, live_sig)
        fsync_directory(live_sig.parent)
        if fault_hook is not None:
            fault_hook("sig_replaced")
        _target_hmac_started = _phase_begin(phase_recorder)
        target_integrity_ok = _verify_pkl_integrity(live_pkl, integrity_dir)
        _phase_record(
            phase_recorder, "target_verification", _target_hmac_started,
        )
        if not target_integrity_ok:
            raise RuntimeError("new runtime pkl failed HMAC verification")
        _metadata_started = _phase_begin(phase_recorder)
        commit_metadata()
        if fault_hook is not None:
            fault_hook("metadata_replaced")
        if runtime_manifest_path is not None and runtime_manifest is not None:
            from backend.utils.runtime_manifest import (
                build_runtime_manifest,
                write_runtime_manifest,
            )

            # Subscripted rather than defaulted: a caller that omitted a field
            # must fail the commit, not silently publish a manifest declaring
            # less than the generation actually holds.
            write_runtime_manifest(
                runtime_manifest_path,
                build_runtime_manifest(
                    revision=int(target_revision),
                    state_revision=int(runtime_manifest["state_revision"]),
                    pkl_digest=target_digest,
                    saved_names=runtime_manifest["saved_names"],
                    skipped_names=runtime_manifest["skipped_names"],
                    serializer_version=str(runtime_manifest["serializer_version"]),
                    pairing=runtime_manifest.get("pairing"),
                    # `.get`, like `pairing`: a producer that predates §5.5's
                    # narrowing emits no such record and is not incomplete.
                    carried_definitions=runtime_manifest.get("carried_definitions"),
                ),
            )
            if fault_hook is not None:
                fault_hook("runtime_manifest_replaced")
        _phase_record(phase_recorder, "metadata_publication", _metadata_started)

        # After `commit_metadata`, so the captured state is the one this
        # generation belongs to rather than the one it replaced.
        _capture_started = _phase_begin(phase_recorder)
        if pair_targets or pair_removals:
            pair_dir.mkdir(parents=True, exist_ok=True)
            for kind, source, target in pair_targets:
                if kind == "paired_checkpoint":
                    _copy_sqlite_durable(source, target)
                else:
                    copy_durable(source, target)
            for _kind, target in pair_removals:
                _remove_durable(target)
            from backend.utils.files import fsync_directory as _fsync_dir

            _fsync_dir(pair_dir)
            if fault_hook is not None:
                fault_hook("pair_captured")
        _phase_record(phase_recorder, "graph_workspace_capture", _capture_started)

        _verify_started = _phase_begin(phase_recorder)
        if metadata_path is not None and not _regular_metadata_matches(
            metadata_path,
            revision=int(target_revision),
            digest=target_digest,
        ):
            raise RuntimeError("new runtime metadata failed generation verification")

        # Components written inside the committed region are authenticated
        # against what they now hold, not against the digest recorded when the
        # transaction registered their previous copy.
        from backend.utils.runtime_manifest import runtime_manifest_sig_path

        target_paths = {
            "pkl": live_pkl,
            "sig": live_sig,
            "metadata": metadata_path,
            "runtime_manifest": runtime_manifest_path,
            "runtime_manifest_sig": (
                runtime_manifest_sig_path(runtime_manifest_path)
                if runtime_manifest_path is not None
                else None
            ),
        }
        for kind, _source, target in pair_targets:
            target_paths[kind] = target
        for descriptor, path in component_records:
            kind = descriptor.get("kind")
            if kind in target_paths and target_paths[kind] is not None:
                descriptor["authoritative"] = True
                descriptor["target_exists"] = True
                descriptor["target_sha256"] = _sha256_file(path)
            if descriptor.get("authoritative"):
                if not path.is_file():
                    raise RuntimeError(
                        f"new runtime component is missing: {path.name}"
                    )
                expected = descriptor.get("target_sha256")
                if not expected or _sha256_file(path) != expected:
                    raise RuntimeError(
                        f"new runtime component failed digest verification: {path.name}"
                    )
        _phase_record(phase_recorder, "target_verification", _verify_started)
        journal["components"] = [
            descriptor for descriptor, _path in component_records
        ]
        # Persist the complete authenticated target set while status is still
        # APPLYING. A crash here rolls back; only the next durable marker permits
        # forward recovery.
        _write_regular_save_journal(journal_path, journal)
        journal["status"] = "COMMITTED"
        _write_regular_save_journal(journal_path, journal)
        committed = True
        if fault_hook is not None:
            fault_hook("commit_marker")
    except Exception:
        _restore_regular_components(
            component_records,
            integrity_dir=integrity_dir,
        )
        _cleanup_regular_save_transaction(
            journal_path,
            [path for _descriptor, path in component_records],
        )
        raise
    finally:
        # A process kill/BaseException intentionally leaves journal + backups;
        # startup recovery decides rollback vs forward from the commit marker.
        if not journal_path.exists() or not committed:
            staged_pkl.unlink(missing_ok=True)
            staged_sig.unlink(missing_ok=True)

    _cleanup_started = _phase_begin(phase_recorder)
    _cleanup_regular_save_transaction(
        journal_path,
        [path for _descriptor, path in component_records],
        fault_hook=fault_hook,
    )
    _phase_record(phase_recorder, "cleanup", _cleanup_started)
    if pair_dir is not None and (pair_targets or pair_removals):
        if workspace_root is not None:
            _workspace_started = _phase_begin(phase_recorder)
            publish_workspace_generation(
                Path(workspace_root), pair_dir, Path(generation_root),
                int(target_revision),
            )
            _phase_record(phase_recorder, "graph_workspace_capture", _workspace_started)
        _prune_started = _phase_begin(phase_recorder)
        _prune_older_generations(Path(generation_root), keep=int(target_revision))
        _phase_record(phase_recorder, "cleanup", _prune_started)


def _verify_pkl_integrity(pkl: Path, integrity_dir: Path) -> bool:
    sig = _sig_path(pkl, integrity_dir)
    if not pkl.exists() or not sig.exists():
        return False
    return hmac.compare_digest(
        sig.read_text().strip(), _pkl_hmac_stream(pkl)
    )
