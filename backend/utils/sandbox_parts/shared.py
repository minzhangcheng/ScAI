"""
Local Jupyter Kernel Executor with State Persistence

Uses dill.dump_session/load_session to save and restore the complete Python environment.
Supports pause/resume by saving and restoring the full session state.
"""

import os
import re
import json
import base64
import asyncio
import copy
import time
import hmac
import hashlib
import logging
import tempfile
from queue import Empty as QueueEmpty
from typing import Optional, Dict, Any, List, Tuple, Final
from datetime import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
from jupyter_client import KernelManager

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256
from backend.config._config import DATA_BASE_DIR

# Cap on a single iopub poll so the wall-clock deadline is re-checked at least
# this often even while the kernel keeps streaming output (tqdm/scanpy verbose).
IOPUB_POLL_CAP = 30.0
_REGULAR_SAVE_COPY_CHUNK_BYTES: Final[int] = 8 * 1024 * 1024

_NOTEBOOK_INLINE_IMAGE_LIMIT_BYTES: Final[int] = 64 * 1024
_NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS: Final[int] = 256 * 1024
_NOTEBOOK_HTML_OUTPUT_LIMIT_CHARS: Final[int] = 256 * 1024
_NOTEBOOK_TRACEBACK_LIMIT_LINES: Final[int] = 200
_NOTEBOOK_OUTPUT_OMISSION = "\n...[output omitted]...\n"
_NOTEBOOK_HTML_OMISSION = "[HTML output omitted: exceeds limit]"
_NOTEBOOK_TRACEBACK_OMISSION = "[... traceback omitted ...]"


# ---------------------------------------------------------------------------
# Session pkl HMAC integrity helpers (host-side only)
# ---------------------------------------------------------------------------

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
                    chunk = source.read(_REGULAR_SAVE_COPY_CHUNK_BYTES)
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

    known = {
        "pkl": live_pkl,
        "sig": live_sig,
        "metadata": metadata_path,
        "manifest": manifest_path,
    }
    resolved: List[Tuple[Dict[str, Any], Path]] = []
    seen: set[Tuple[str, str]] = set()
    states_root = live_pkl.parent.resolve()
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
        if kind == "h5ad":
            from backend.services.snapshot_v2.runtime_names import (
                resolve_confined_runtime_h5ad_path,
            )

            if not name.endswith(".h5ad"):
                raise RuntimeError(f"invalid runtime h5ad component name: {name!r}")
            variable = name[:-5]
            try:
                path = resolve_confined_runtime_h5ad_path(
                    states_root, variable, name,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid runtime h5ad component: {name!r}"
                ) from exc
        else:
            path = known.get(kind)
            if path is None or name != path.name:
                raise RuntimeError(
                    f"regular runtime component does not match canonical {kind} path"
                )
        resolved.append((component, path))
    return resolved


def _validate_runtime_h5ad_manifest(
    manifest_path: Path,
) -> Dict[str, Tuple[Path, str]]:
    """Validate manifest schema and return canonical referenced sidecars."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime h5ad manifest is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("version") not in (1, 2):
        raise RuntimeError("runtime h5ad manifest has an unsupported schema")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError("runtime h5ad manifest entries are not an object")
    if payload.get("version") == 2:
        pkl_names = payload.get("authoritative_pkl_names")
        if not isinstance(pkl_names, list):
            raise RuntimeError(
                "runtime h5ad manifest v2 lacks authoritative_pkl_names"
            )
        from backend.services.snapshot_v2.runtime_names import (
            validate_runtime_global_name,
        )

        try:
            validated = [validate_runtime_global_name(name) for name in pkl_names]
        except ValueError as exc:
            raise RuntimeError(
                "runtime h5ad manifest has an invalid authoritative pkl name"
            ) from exc
        if len(validated) != len(set(validated)):
            raise RuntimeError(
                "runtime h5ad manifest repeats an authoritative pkl name"
            )

    from backend.services.snapshot_v2.runtime_names import (
        resolve_confined_runtime_h5ad_path,
        validate_runtime_global_name,
    )

    referenced: Dict[str, Tuple[Path, str]] = {}
    for raw_name, raw_entry in entries.items():
        try:
            name = validate_runtime_global_name(raw_name)
        except ValueError as exc:
            raise RuntimeError("runtime h5ad manifest has an invalid name") from exc
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"runtime h5ad manifest entry {name!r} is invalid")
        digest = raw_entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            raise RuntimeError(
                f"runtime h5ad manifest entry {name!r} has an invalid digest"
            )
        try:
            sidecar = resolve_confined_runtime_h5ad_path(
                manifest_path.parent,
                name,
                raw_entry.get("path"),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"runtime h5ad manifest entry {name!r} has an unsafe path"
            ) from exc
        if not sidecar.is_file() or _sha256_file(sidecar) != digest.lower():
            raise RuntimeError(
                f"runtime h5ad sidecar {sidecar.name!r} failed digest verification"
            )
        referenced[name] = (sidecar, digest.lower())
    return referenced


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


def _include_new_runtime_h5ads_in_rollback(
    components: List[Tuple[Dict[str, Any], Path]],
    states_dir: Path,
) -> None:
    """Add sidecars created after the APPLYING journal to the rollback set.

    A manifest callback can create a brand-new sidecar before it returns.  A
    process kill at that exact boundary leaves the initial signed journal
    unaware of the filename.  Every sidecar that existed before APPLYING was
    already recorded, so an unrecorded canonical ``*.h5ad`` is necessarily a
    transaction product and the old-generation rollback must remove it.
    """
    known_names = {
        path.name
        for component, path in components
        if component.get("kind") == "h5ad"
    }
    for path in sorted(states_dir.glob("*.h5ad")):
        if not path.is_file() or path.name in known_names:
            continue
        components.append((
            _regular_component(
                "h5ad",
                path,
                previous_exists=False,
                authoritative=False,
            ),
            path,
        ))
        known_names.add(path.name)


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
            if manifest_path is not None:
                referenced = _validate_runtime_h5ad_manifest(manifest_path)
                target_h5ads = {
                    path.name: component.get("target_sha256")
                    for component, path in resolved_components
                    if component.get("kind") == "h5ad"
                    and component.get("authoritative")
                }
                if target_h5ads != {
                    path.name: digest for path, digest in referenced.values()
                }:
                    raise RuntimeError(
                        "committed runtime manifest sidecars do not match the journal"
                    )
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
            if session_id:
                from backend.utils.runtime_apply import (
                    quarantine_unreferenced_runtime_h5ads,
                )

                quarantine_unreferenced_runtime_h5ads(
                    session_id,
                    set(payload.get("keep_h5ad_names") or []),
                    states_dir=live_pkl.parent,
                )
            _cleanup_regular_save_transaction(journal_path, component_paths)
            return "forward"

    # PREPARED/APPLYING, or a committed target that failed verification. A
    # manifest callback may have created a new sidecar before the journal could
    # be extended with its descriptor; include that bounded states-dir product
    # so rollback cannot leave an orphan from the rejected generation.
    _include_new_runtime_h5ads_in_rollback(
        resolved_components,
        live_pkl.parent,
    )
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
    manifest_path: Path | None = None,
    commit_manifest=None,
    retire_sidecars=None,
    target_revision: int = 0,
    authoritative_pkl_names: set[str] | None = None,
    fault_hook=None,
) -> None:
    """Install a runtime generation with crash-recoverable commit semantics."""
    integrity_dir.mkdir(parents=True, exist_ok=True)
    live_pkl.parent.mkdir(parents=True, exist_ok=True)
    live_sig = _sig_path(live_pkl, integrity_dir)
    staged_sig = live_sig.with_name(f"{live_sig.name}.regular-save.tmp")
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
    if manifest_path is not None:
        component_records.append((
            _regular_component(
                "manifest", manifest_path,
                previous_exists=manifest_path.is_file(), authoritative=True,
            ),
            manifest_path,
        ))

    # The old generation may reference sidecars that the new generation
    # retires. Back every live sidecar before publishing the APPLYING journal so
    # a rollback can restore the complete prior generation. New referenced
    # sidecars are discovered and added below before COMMITTED is published.
    for h5ad_path in sorted(live_pkl.parent.glob("*.h5ad")):
        if not h5ad_path.is_file():
            continue
        component_records.append((
            _regular_component(
                "h5ad", h5ad_path,
                previous_exists=True, authoritative=False,
            ),
            h5ad_path,
        ))
    journal_path = _regular_save_journal_path(live_pkl)
    target_digest = _sha256_file(staged_pkl)

    staged_sig_owned = True
    prejournal_backups: List[Path] = []
    journal = None
    try:
        staged_sig.write_text(_pkl_hmac_stream(staged_pkl), encoding="utf-8")
        with open(staged_sig, "r+", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        # Backups are complete and durable before the journal exposes a transaction.
        for descriptor, component in component_records:
            backup = _regular_save_backup_path(component)
            backup.unlink(missing_ok=True)
            if descriptor["previous_exists"]:
                # Register before the copy: a copy implementation may publish
                # the destination and then fail while flushing its directory.
                prejournal_backups.append(backup)
                _copy_durable(component, backup)
                if _sha256_file(backup) != descriptor["previous_sha256"]:
                    raise RuntimeError(
                        f"regular runtime save backup failed verification: {backup.name}"
                    )

        journal = {
            "version": 2,
            "status": "APPLYING",
            "target_digest": target_digest,
            "target_revision": int(target_revision),
            "keep_h5ad_names": [],
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
        if not _verify_pkl_integrity(live_pkl, integrity_dir):
            raise RuntimeError("new runtime pkl failed HMAC verification")
        if commit_manifest is not None:
            commit_manifest()
            if fault_hook is not None:
                fault_hook("manifest_replaced")
        commit_metadata()
        if fault_hook is not None:
            fault_hook("metadata_replaced")
        if metadata_path is not None and not _regular_metadata_matches(
            metadata_path,
            revision=int(target_revision),
            digest=target_digest,
        ):
            raise RuntimeError("new runtime metadata failed generation verification")

        # Authenticate every target component inside the same signed journal
        # boundary. The manifest is schema-checked before its referenced h5ad
        # files are admitted as authoritative transaction components.
        referenced_h5ads: Dict[str, Tuple[Path, str]] = {}
        if manifest_path is not None:
            referenced_h5ads = _validate_runtime_h5ad_manifest(manifest_path)
        known_h5ads = {
            path.name: (descriptor, path)
            for descriptor, path in component_records
            if descriptor.get("kind") == "h5ad"
        }
        for variable, (h5ad_path, expected_digest) in referenced_h5ads.items():
            record = known_h5ads.get(h5ad_path.name)
            if record is None:
                # A callback may have created a new sidecar. It did not exist at
                # the transaction boundary, so rollback removes it.
                descriptor = _regular_component(
                    "h5ad", h5ad_path,
                    previous_exists=False, authoritative=True,
                )
                component_records.append((descriptor, h5ad_path))
                known_h5ads[h5ad_path.name] = (descriptor, h5ad_path)
            else:
                descriptor, _path = record
                descriptor["authoritative"] = True
            descriptor["target_exists"] = True
            descriptor["target_sha256"] = expected_digest

        target_paths = {
            "pkl": live_pkl,
            "sig": live_sig,
            "metadata": metadata_path,
            "manifest": manifest_path,
        }
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
        journal["keep_h5ad_names"] = sorted(referenced_h5ads)
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
        _include_new_runtime_h5ads_in_rollback(
            component_records,
            live_pkl.parent,
        )
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

    if retire_sidecars is not None:
        try:
            retire_sidecars()
            if fault_hook is not None:
                fault_hook("sidecars_retired")
        except Exception:
            # COMMITTED is already durable, so this is maintenance rather than
            # a reason to roll back the new pkl. Keep the signed journal and
            # backups: startup recovery uses its keep_h5ad_names instruction to
            # retry the bounded quarantine, then retires the transaction.
            logger.exception("Committed runtime sidecar cleanup deferred to recovery")
            return
    _cleanup_regular_save_transaction(
        journal_path,
        [path for _descriptor, path in component_records],
        fault_hook=fault_hook,
    )


def _verify_pkl_integrity(pkl: Path, integrity_dir: Path) -> bool:
    sig = _sig_path(pkl, integrity_dir)
    if not pkl.exists() or not sig.exists():
        return False
    return hmac.compare_digest(
        sig.read_text().strip(), _pkl_hmac_stream(pkl)
    )


def _integrity_dir(sid: str) -> Path:
    return DATA_BASE_DIR / "integrity" / sid


def _path_is_within(path: str, root: str) -> bool:
    resolved = str(Path(path).resolve())
    root_resolved = str(Path(root).resolve()).rstrip(os.sep)
    return resolved == root_resolved or resolved.startswith(root_resolved + os.sep)


def compute_remaining_timeout(deadline: float, now: float) -> float:
    """Seconds left until the wall-clock deadline; <= 0 means expired."""
    return deadline - now


async def _to_thread_drained(func, /, *args, **kwargs):
    """Run a blocking resource call without orphaning its worker on cancel.

    The owned executor keeps both cancellation and loop shutdown from orphaning
    a worker that still touches the same socket, client, or process manager.
    """
    from backend.utils.asyncio_tasks import run_in_owned_thread

    return await run_in_owned_thread(func, *args, **kwargs)


logger = logging.getLogger(__name__)


def _is_kernel_protocol_error(exc: Exception) -> bool:
    """True for ZMQ/jupyter session-key corruption that makes the kernel
    permanently unusable until restarted (vs. a normal user code error)."""
    if not isinstance(exc, (ValueError, KeyError)):
        return False
    s = str(exc).lower()
    return (
        "invalid signature" in s
        or "<ids|msg>" in s
        or "is not in list" in s
    )


# Dangerous code patterns that should not be executed in the sandbox.
# Enforced at the sandbox level so ALL callers (BioinformaticsAgent,
# report_generator, chat sandbox, etc.) are protected.
# NOTE: regex is defense-in-depth only; runtime _sandboxed_open + write
# allowlist are the real boundary.
DANGEROUS_PATTERNS = [
    # System command execution
    (r'\bos\.system\b', 'os.system calls'),
    (r'\bsubprocess\b', 'subprocess module'),
    (r'\bos\.popen\b', 'os.popen calls'),
    (r'\bos\.spawn\w*\b', 'os.spawn* calls'),
    (r'\bos\.posix_spawnp?\b', 'os.posix_spawn* calls'),
    (r'\bos\.exec\w*\b', 'os.exec* calls'),
    (r'\bos\.fork\b', 'os.fork calls'),
    (r'\bpty\b', 'pty module'),

    # Code execution
    (r'\beval\s*\(', 'eval() function'),
    (r'\bexec\s*\(', 'exec() function'),
    (r'\b__import__\s*\(', '__import__ function'),
    # Block the dangerous built-in, but allow ordinary attribute calls such as
    # ``re.compile(...)``.  The previous word-boundary expression matched both.
    (r'(?<![\w.])compile\s*\(', 'compile() function'),
    (r'\bimportlib\b', 'importlib module'),
    (r'\bscai_kernel_trust\b', 'reserved kernel trust module'),

    # Dangerous attribute access
    (r'\b__builtins__\b', '__builtins__ access'),
    (r'\b__class__\s*\.\s*__bases__\b', '__class__.__bases__ access'),
    (r'\b__class__\s*\.\s*__mro__\b', '__class__.__mro__ access'),
    (r'\b__subclasses__\s*\(', '__subclasses__ access'),
    # globals() and getattr-with-dunder are commonly used by scanpy/anndata
    # and pose no additional risk in an already-isolated Jupyter kernel.
    # locals() is read-only (same as globals()) and commonly used in data-science code.
    (r'\bsetattr\s*\([^,]+,\s*["\']__', 'setattr with dunder'),
    (r'\bdelattr\s*\([^,]+,\s*["\']__', 'delattr with dunder'),
    (r'\bvars\s*\(\s*\)', 'vars() function'),

    # Reflection-based sandbox bypasses
    # getattr on dangerous modules (os, sys, subprocess, etc.) can retrieve dangerous funcs dynamically
    (r'\bgetattr\s*\(\s*(?:os|sys|subprocess|shutil|builtins|__builtins__)\b', 'getattr on dangerous module'),
    # __dict__ subscript access can bypass attribute-level checks
    (r'\b__dict__\s*\[', '__dict__ subscript access'),
    # __globals__ gives access to the global namespace of any function
    (r'\b__globals__\b', '__globals__ access'),
    # __class__.__init__ chain is used to reach __globals__ via method objects
    (r'\b__class__\s*\.\s*__init__\b', '__class__.__init__ chain'),
    # type.__subclasses__() enumerates all subclasses to find exploitable ones
    (r'\btype\s*\.\s*__subclasses__', 'type.__subclasses__ access'),

    # File system operations (destructive)
    # Note: open() writes are checked contextually in validate_code_safety()
    # rather than blanket-blocked, because the runtime I/O sandbox (_sandboxed_open)
    # already enforces path safety.  The static check only flags writes that do NOT
    # reference known safe path variables.
    # Note: This static check can be bypassed via variable indirection (e.g. mode='w'; open(f, mode)).
    # The runtime I/O sandbox (_sandboxed_open) is the primary defense; this is a best-effort early check.
    (
        r'\bopen\s*\([^)]*,\s*["\'][^"\']*[wax+]',
        'file write operations outside safe directories',
    ),
    (
        r'\.open\s*\(\s*["\'][^"\']*[wax+]',
        'file write operations outside safe directories',
    ),
    (r'\brm\s+-rf\b', 'rm -rf command'),
    (r'\bshutil\.rmtree\b', 'shutil.rmtree'),
    (r'\bos\.remove\b', 'os.remove'),
    (r'\bos\.unlink\b', 'os.unlink'),
    (r'\bos\.rmdir\b', 'os.rmdir'),
    # os.makedirs is allowed — directory creation is non-destructive (idempotent with exist_ok=True)
    # and LLM-generated bioinformatics code commonly uses it for cache/output directories.
    (r'\bos\.rename\b', 'os.rename'),
    (r'\bos\.replace\b', 'os.replace'),
    (r'\bshutil\.move\b', 'shutil.move'),
    (r'\bshutil\.copy\b', 'shutil.copy (use explicit paths)'),

    # Network access
    (r'\bsocket\b', 'socket module'),
    (r'\brequests\.(?:get|post|put|delete|patch|head|options)\b', 'HTTP requests'),
    (r'\burllib\b', 'urllib module'),
    (r'\bhttpx\b', 'httpx module'),
    (r'\baiohttp\b', 'aiohttp module'),

    # Low-level / dangerous modules
    (r'\bctypes\b', 'ctypes module'),
    (r'\bmultiprocessing\b', 'multiprocessing module'),
    (r'\bpickle\.loads?\b', 'pickle.load/loads (arbitrary code execution)'),
    (r'\bcPickle\b', 'cPickle module'),
    (r'\bmarshal\b', 'marshal module'),
    (r'\bshelve\b', 'shelve module'),

    # Environment manipulation
    (r'\bos\.chdir\b', 'os.chdir (changing working directory)'),
    (r'\bos\.chroot\b', 'os.chroot'),
    (r'\bsys\.path\s*\.(?:append|insert|extend)\b', 'sys.path modification'),
]


def _is_safe_os_system(code: str) -> bool:
    """No os.system calls are allowed in notebook execution."""
    return False


def _is_safe_subprocess(code: str) -> bool:
    """Block the subprocess surface, including import-only cross-cell aliases."""
    return False


def _is_safe_importlib(code: str) -> bool:
    """Allow only the small, non-executing importlib surface analyses need."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    allowed_modules = {"importlib", "importlib.util", "importlib.metadata"}
    allowed_from = {
        "importlib": {"import_module", "reload"},
        "importlib.util": {"find_spec"},
        "importlib.metadata": {
            "version", "PackageNotFoundError", "packages_distributions",
        },
    }
    allowed_calls = {
        "importlib.import_module", "importlib.reload",
        "importlib.util.find_spec", "importlib.metadata.version",
        "importlib.metadata.packages_distributions",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("importlib") and alias.name not in allowed_modules
                   for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("importlib"):
            if node.module not in allowed_from:
                return False
            if any(alias.name not in allowed_from[node.module] for alias in node.names):
                return False
        elif isinstance(node, ast.Call):
            parts = []
            expr = node.func
            while isinstance(expr, ast.Attribute):
                parts.append(expr.attr)
                expr = expr.value
            if isinstance(expr, ast.Name) and expr.id == "importlib":
                dotted = ".".join([expr.id, *reversed(parts)])
                if dotted not in allowed_calls:
                    return False
    return True


def _is_safe_file_write(code: str) -> bool:
    """Reject static file-open calls in write modes on unsafe literal paths.

    Aliased or computed paths are allowed here and re-enforced at runtime by
    the kernel's ``_sandboxed_open`` write allowlist (``_ALLOWED_WRITE_ROOTS``),
    which is the authoritative boundary for anything too dynamic to prove
    statically. Keeping this check a cheap early-reject for obvious-bad
    literals avoids a parallel path-provenance prover that would have to grow
    with every new idiom (BinOp/JoinedStr/Call chains, fixpoint alias
    propagation) to track a boundary the runtime already enforces.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    def _literal_is_safe(value: str) -> bool:
        normalized = value.replace("\\", "/").lower()
        return (
            normalized.startswith("/tmp/scai-")
            or "/workspace/" in normalized
            or "/figures/" in normalized
            or "/tables/" in normalized
            or "/annotation_validation/" in normalized
            or normalized.startswith((
                "figures/", "tables/", "annotation_validation/",
                "./figures/", "./tables/", "./annotation_validation/",
            ))
        )

    def _literal_path(expr):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, (ast.Name, ast.Attribute))
            and getattr(expr.func, "id", getattr(expr.func, "attr", None)) == "Path"
            and expr.args
            and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)
        ):
            return expr.args[0].value
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        path_arg = None
        positional_mode_index = None
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            if node.args:
                path_arg = node.args[0]
                positional_mode_index = 1
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            receiver = node.func.value
            if (
                isinstance(receiver, ast.Name)
                and receiver.id in {"builtins", "io"}
                and node.args
            ):
                path_arg = node.args[0]
                positional_mode_index = 1
            else:
                path_arg = receiver
                positional_mode_index = 0
        if path_arg is None:
            continue

        mode = None
        if (
            positional_mode_index is not None
            and len(node.args) > positional_mode_index
            and isinstance(node.args[positional_mode_index], ast.Constant)
        ):
            mode = node.args[positional_mode_index].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        if not (isinstance(mode, str) and any(flag in mode for flag in ("w", "a", "x", "+"))):
            continue
        # Only literal paths can be ruled on statically; everything else is
        # resolved and gated by the process-level boundary at execution time.
        literal_path = _literal_path(path_arg)
        if literal_path is not None and not _literal_is_safe(literal_path):
            return False
    return True


# Maps pattern description keywords to their safety check functions
_EXCEPTION_CHECKS = {
    'os.system':   _is_safe_os_system,
    'subprocess':  _is_safe_subprocess,
    'importlib':   _is_safe_importlib,
    'file write':  _is_safe_file_write,
}


_DANGEROUS_AST_FUNCS = frozenset({
    'system', 'popen', 'exec', 'eval', 'execfile', 'spawn', 'fork',
    'remove', 'unlink', 'rmdir', 'rmtree', 'rename', 'replace',
})

_DANGEROUS_AST_ATTRS = frozenset({
    '__globals__', '__init__', '__subclasses__', '__bases__', '__mro__',
    '__builtins__',
})

_DANGEROUS_AST_MODULES = frozenset({
    'os', 'sys', 'subprocess', 'shutil', 'builtins', 'scai_kernel_trust',
})

_SUBPROCESS_EXEC_FUNCS = frozenset({
    "run",
    "call",
    "check_call",
    "check_output",
    "Popen",
})

_OS_PROCESS_EXEC_FUNCS = frozenset({
    "system",
    "popen",
    "fork",
    "forkpty",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "posix_spawn",
    "posix_spawnp",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
})


def _validate_code_safety_ast(code: str) -> Tuple[bool, str]:
    """AST-based validation layer that catches reflection-based bypasses
    which regex patterns may miss (e.g. string concatenation, variable indirection).

    Returns (True, "") if safe, (False, reason) if dangerous pattern detected.
    Falls back to (True, "") on parse errors so this is purely additive.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # Cannot parse — let regex-only check decide

    module_aliases: dict[str, str] = {}
    blocked_direct_calls: dict[str, str] = {}
    dangerous_string_aliases: dict[str, str] = {}
    namespace_aliases: set[str] = set()

    def _constant_string(expr) -> Optional[str]:
        """Resolve only statically known strings; ordinary string data is safe."""
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Name):
            return dangerous_string_aliases.get(expr.id)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left = _constant_string(expr.left)
            right = _constant_string(expr.right)
            if left is not None and right is not None:
                return left + right
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"subprocess", "os"}:
                    module_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                for alias in node.names:
                    if alias.name in _SUBPROCESS_EXEC_FUNCS:
                        return False, (
                            "Import of blocked subprocess function "
                            f"'subprocess.{alias.name}'"
                        )
            elif node.module == "os":
                for alias in node.names:
                    if alias.name in _OS_PROCESS_EXEC_FUNCS:
                        return False, (
                            "Import of blocked process function "
                            f"'os.{alias.name}'"
                        )

    def _module_alias_from_expr(expr) -> Optional[str]:
        if isinstance(expr, ast.Name):
            return module_aliases.get(expr.id)
        return None

    def _blocked_call_from_expr(expr) -> Optional[str]:
        if isinstance(expr, ast.Name):
            if expr.id in {"eval", "exec", "__import__", "compile"}:
                return expr.id
            return blocked_direct_calls.get(expr.id)
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            module_name = module_aliases.get(expr.value.id)
            if module_name == "subprocess" and expr.attr in _SUBPROCESS_EXEC_FUNCS:
                return f"subprocess.{expr.attr}"
            if module_name == "os" and expr.attr in _OS_PROCESS_EXEC_FUNCS:
                return f"os.{expr.attr}"
        return None

    def _is_namespace_expr(expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in namespace_aliases
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            if expr.func.id in {"globals", "locals"} and not expr.args:
                return True
            if expr.func.id == "vars" and len(expr.args) == 1:
                target = expr.args[0]
                if isinstance(target, ast.Name):
                    target_name = module_aliases.get(target.id, target.id)
                    return target_name in {"builtins", "__builtins__"}
        return False

    def _subscript_key(expr) -> Optional[str]:
        if not isinstance(expr, ast.Subscript):
            return None
        return _constant_string(expr.slice)

    def _dangerous_lookup_from_expr(expr) -> Optional[str]:
        if isinstance(expr, ast.Subscript) and _is_namespace_expr(expr.value):
            key = _subscript_key(expr)
            if key in {"eval", "exec", "__import__", "compile"}:
                return key
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "getattr"
            and len(expr.args) >= 2
        ):
            target = expr.args[0]
            attr = _constant_string(expr.args[1])
            if isinstance(target, ast.Name):
                target_name = module_aliases.get(target.id, target.id)
                if (
                    target_name in {"builtins", "__builtins__"}
                    and attr in {"eval", "exec", "__import__", "compile"}
                ):
                    return attr
        return None

    def _propagate_alias_to_target(target, value) -> bool:
        changed_alias = False
        if isinstance(target, ast.Name):
            module_name = _module_alias_from_expr(value)
            if module_name and module_aliases.get(target.id) != module_name:
                module_aliases[target.id] = module_name
                changed_alias = True

            blocked_reason = _blocked_call_from_expr(value)
            blocked_reason = blocked_reason or _dangerous_lookup_from_expr(value)
            if blocked_reason and blocked_direct_calls.get(target.id) != blocked_reason:
                blocked_direct_calls[target.id] = blocked_reason
                changed_alias = True

            string_value = _constant_string(value)
            if (
                string_value in {"eval", "exec", "__import__", "compile"}
                and dangerous_string_aliases.get(target.id) != string_value
            ):
                dangerous_string_aliases[target.id] = string_value
                changed_alias = True

            if _is_namespace_expr(value) and target.id not in namespace_aliases:
                namespace_aliases.add(target.id)
                changed_alias = True
            return changed_alias

        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            for target_item, value_item in zip(target.elts, value.elts):
                changed_alias = _propagate_alias_to_target(target_item, value_item) or changed_alias
        return changed_alias

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    changed = _propagate_alias_to_target(target, node.value) or changed
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                changed = _propagate_alias_to_target(node.target, node.value) or changed

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("_scai_"):
            return False, "Access to reserved ScAI kernel namespace"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_direct_calls:
                return False, f"Call to blocked dependency/system function '{blocked_direct_calls[node.func.id]}'"

            dangerous_lookup = _dangerous_lookup_from_expr(node.func)
            if dangerous_lookup:
                return False, f"Call to dynamically retrieved dangerous builtin '{dangerous_lookup}'"

            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name = module_aliases.get(node.func.value.id)
                if module_name == "subprocess" and node.func.attr in _SUBPROCESS_EXEC_FUNCS:
                    return False, f"Call to blocked subprocess function 'subprocess.{node.func.attr}'"
                if module_name == "os" and node.func.attr in _OS_PROCESS_EXEC_FUNCS:
                    return False, f"Call to blocked os.{node.func.attr}"

        # Detect getattr(os, 'system') / getattr(os, dangerous_string)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr':
            if len(node.args) >= 2:
                target = node.args[0]
                attr_arg = node.args[1]
                # Check if target is a dangerous module
                if isinstance(target, ast.Name):
                    target_module = module_aliases.get(target.id, target.id)
                    if target_module in _DANGEROUS_AST_MODULES:
                        return False, f"getattr() on dangerous module '{target_module}'"
                # A dangerous-looking string is data unless this call retrieves
                # it from a dangerous module/namespace.  This avoids rejecting
                # normal column names, metric labels and parameter dictionaries.

        # Detect attribute chains containing __globals__, __init__.__globals__, etc.
        if isinstance(node, ast.Attribute):
            if node.attr in ('__globals__', '__builtins__'):
                return False, f"Access to '{node.attr}' attribute"
            # Detect __class__.__init__ chain (used to reach __globals__)
            if node.attr == '__init__' and isinstance(node.value, ast.Attribute) and node.value.attr == '__class__':
                return False, "Access to __class__.__init__ chain"

        # Detect __dict__[...] subscript access
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == '__dict__':
                return False, "__dict__ subscript access"

        if isinstance(node, ast.Subscript) and _is_namespace_expr(node.value):
            key = _subscript_key(node)
            if isinstance(key, str) and key.startswith("_scai_"):
                return False, "Access to reserved ScAI kernel namespace"

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "setdefault"}
            and _is_namespace_expr(node.func.value)
        ):
            candidates = list(node.args)
            candidates.extend(keyword.value for keyword in node.keywords)
            if node.func.attr == "setdefault" and node.args:
                key = _constant_string(node.args[0])
                if isinstance(key, str) and key.startswith("_scai_"):
                    return False, "Access to reserved ScAI kernel namespace"
            for candidate in candidates:
                if not isinstance(candidate, ast.Dict):
                    continue
                for key_node in candidate.keys:
                    key = _constant_string(key_node)
                    if isinstance(key, str) and key.startswith("_scai_"):
                        return False, "Access to reserved ScAI kernel namespace"

    return True, ""


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Validate code for security before execution.

    Uses a two-layer approach:
    1. Regex pattern matching against DANGEROUS_PATTERNS (fast, broad)
    2. AST-based analysis to catch reflection/indirection bypasses

    Args:
        code: The Python code to validate

    Returns:
        Tuple of (is_safe, error_message). is_safe is True if the code
        passes all security checks.
    """
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            # Check if the matched pattern has a known safe-usage exception
            if any(key in description and check(code) for key, check in _EXCEPTION_CHECKS.items()):
                continue
            logger.warning(
                "Security: blocked dangerous code pattern [%s] — snippet: %.200s",
                description, code.replace('\n', ' '),
            )
            return False, f"Dangerous pattern detected: {description}"

    # Second layer: AST-based validation for reflection/indirection bypasses
    ast_safe, ast_reason = _validate_code_safety_ast(code)
    if not ast_safe:
        logger.warning(
            "Security: AST analysis blocked code [%s] — snippet: %.200s",
            ast_reason, code.replace('\n', ' '),
        )
        return False, f"Dangerous pattern detected (AST): {ast_reason}"

    return True, ""


def extract_code_from_markdown(text: str, language: str = "python") -> str:
    """
    Extract code from a markdown-formatted LLM response.

    Uses a state-machine approach to correctly handle:
    - Fenced code blocks with language tag (```python ... ```)
    - Fenced code blocks without language tag (``` ... ```)
    - Multiple code blocks (returns the first matching one)
    - Nested backticks inside code blocks

    Args:
        text: The LLM response text that may contain markdown code blocks
        language: The language tag to look for (default "python")

    Returns:
        The extracted code string, or the original text stripped if no
        code block is found.
    """
    if not text:
        return ""

    fence_with_lang = f"```{language}"
    fence_plain = "```"

    # Try language-tagged fence first, then plain fence
    for fence_start in (fence_with_lang, fence_plain):
        pos = text.find(fence_start)
        if pos == -1:
            continue

        # Move past the opening fence line
        code_start = text.find("\n", pos)
        if code_start == -1:
            continue
        code_start += 1  # skip the newline

        # Find closing fence: scan for ``` at the start of a line
        # (handles nested backticks by requiring ``` on its own line)
        search_pos = code_start
        while search_pos < len(text):
            line_end = text.find("\n", search_pos)
            if line_end == -1:
                line_end = len(text)
            line = text[search_pos:line_end].strip()
            if line == fence_plain:
                return text[code_start:search_pos].strip()
            search_pos = line_end + 1

        # No closing fence found — take everything after the opening fence
        return text[code_start:].strip()

    # No code block found — return text as-is
    return text.strip()


class CodeExecutionResult:
    """Result of code execution"""
    def __init__(
        self,
        success: bool,
        output: str = "",
        error: Optional[str] = None,
        figures: Optional[List[str]] = None,  # Base64 encoded images
        execution_count: int = 0,
        timed_out: bool = False,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.figures = figures or []
        self.execution_count = execution_count
        self.timed_out = bool(timed_out)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "figures": self.figures,
            "execution_count": self.execution_count
        }


class NotebookState:
    """Represents the state of a Jupyter notebook session"""
    def __init__(self, session_id: str, state_dir: str = "states"):
        self.session_id = session_id
        self.state_dir = Path(state_dir).absolute()  # Use absolute path
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir = self.state_dir.parent / "workspace"
        
        self.cells: List[Dict[str, Any]] = []  # List of {code, outputs, execution_count}
        self.execution_count: int = 0
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()
        self.runtime_persistence: Dict[str, Any] = {
            "status": "stale",
            "current_revision": 0,
            "saved_revision": 0,
            "dirty_since": None,
            "last_saved_at": None,
            "last_restored_at": None,
            "last_failed_at": None,
            "failed_variables": [],
            "error_summary": None,
            "requires_recompute": False,
            "recompute_variables": [],
            "recommended_action": None,
            "commit_source": None,
            "runtime_digest": None,
            # The committed generation remains independently addressable when
            # a later save/restore attempt fails. Attempt state must never make
            # the last verified pkl inaccessible.
            "committed_revision": 0,
            "committed_source": None,
            "committed_digest": None,
            "last_attempt_status": None,
            "last_attempt_at": None,
            "last_attempt_revision": 0,
            "last_attempt_error": None,
            "last_attempt_failed_variables": [],
            "degraded_recovery": False,
            "degraded_from_revision": None,
            "degraded_to_revision": None,
            "budget_violations": [],
            "top_offenders": [],
        }
        self._load_runtime_persistence_metadata()
    
    @property
    def notebook_path(self) -> Path:
        from backend.utils.live_paths import live_notebook_path

        return live_notebook_path(self.state_dir, self.session_id)
    
    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.session_id}_state.json"
    
    @property
    def session_path(self) -> Path:
        """Path for dill session dump"""
        return self.state_dir / f"{self.session_id}_session.pkl"

    @staticmethod
    def _bound_notebook_text(value: Any, limit: int) -> Any:
        if not isinstance(value, str) or len(value) <= limit:
            return value
        marker = _NOTEBOOK_OUTPUT_OMISSION
        if limit <= len(marker):
            return marker[:limit]
        remaining = limit - len(marker)
        head = (remaining + 1) // 2
        tail = remaining - head
        return value[:head] + marker + (value[-tail:] if tail else "")

    def _compact_outputs_for_state(self, outputs: List[Dict]) -> List[Dict]:
        """Compact outputs with this state's pinned workspace context."""
        compacted = copy.deepcopy(outputs or [])
        from backend.utils.notebook_media import (
            EXTERNAL_IMAGE_PLACEHOLDER,
            NOTEBOOK_EXTERNAL_IMAGE_MIME,
            externalize_notebook_image,
        )

        def estimated_size(encoded: Any) -> int:
            if not isinstance(encoded, str):
                return 0
            padding = len(encoded) - len(encoded.rstrip("="))
            return max(0, (len(encoded) * 3) // 4 - padding)

        for output in compacted:
            if not isinstance(output, dict):
                continue
            for field in ("text", "evalue", "error"):
                if isinstance(output.get(field), str):
                    output[field] = self._bound_notebook_text(
                        output[field], _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS
                    )
            traceback = output.get("traceback")
            if isinstance(traceback, list):
                lines = [
                    self._bound_notebook_text(line, _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS)
                    for line in traceback
                ]
                output["traceback"] = (
                    lines[:100] + [_NOTEBOOK_TRACEBACK_OMISSION] + lines[-100:]
                    if len(lines) > _NOTEBOOK_TRACEBACK_LIMIT_LINES
                    else lines
                )
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            existing_external = data.pop(NOTEBOOK_EXTERNAL_IMAGE_MIME, None)
            external_references: List[Any] = []
            if isinstance(existing_external, list):
                external_references.extend(existing_external)
            elif existing_external is not None:
                external_references.append(existing_external)
            oversized_html = False
            for key, value in list(data.items()):
                if key == "text/plain":
                    data[key] = self._bound_notebook_text(
                        value, _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS
                    )
                elif key == "text/html" and isinstance(value, str):
                    if len(value) > _NOTEBOOK_HTML_OUTPUT_LIMIT_CHARS:
                        data.pop(key, None)
                        oversized_html = True
                elif key in {"image/png", "image/jpeg"}:
                    if estimated_size(value) <= _NOTEBOOK_INLINE_IMAGE_LIMIT_BYTES:
                        continue
                    reference = externalize_notebook_image(
                        self.working_dir,
                        self.notebook_path,
                        value,
                        media_type=key,
                    )
                    data.pop(key, None)
                    if reference is None:
                        data.setdefault("text/plain", EXTERNAL_IMAGE_PLACEHOLDER)
                    else:
                        external_references.append(reference)
                        data.setdefault("text/plain", f"[external image: {reference['path']}]")
            if external_references:
                data[NOTEBOOK_EXTERNAL_IMAGE_MIME] = (
                    external_references[0]
                    if len(external_references) == 1
                    else external_references
                )
            if oversized_html:
                plain = data.get("text/plain")
                if isinstance(plain, str) and plain:
                    plain = f"{plain}\n{_NOTEBOOK_HTML_OMISSION}"
                else:
                    plain = _NOTEBOOK_HTML_OMISSION
                data["text/plain"] = self._bound_notebook_text(
                    plain,
                    _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS,
                )
        return compacted

    def add_cell(self, code: str, outputs: List[Dict], execution_count: int):
        """Add an executed cell to the history"""
        self.cells.append({
            "code": code,
            "outputs": self._compact_outputs_for_state(outputs),
            "execution_count": execution_count,
            "executed_at": datetime.now().isoformat()
        })
        self.execution_count = execution_count
        self.updated_at = datetime.now().isoformat()
    
    def save_notebook(self):
        """Save as Jupyter notebook format"""
        nb = new_notebook()
        for index, cell in enumerate(self.cells):
            code_cell = new_code_cell(source=cell["code"])
            # nbformat generates a random cell id when one is omitted.  A
            # repeated save must be byte-stable so exports and checkpoints do
            # not look changed merely because notebook metadata was rebuilt.
            code_cell["id"] = hashlib.sha256(
                f"{index}:{cell['code']}".encode("utf-8")
            ).hexdigest()[:8]
            code_cell.execution_count = cell["execution_count"]
            
            # Convert outputs to notebook format
            cell_outputs = []
            for out in self._compact_outputs_for_state(cell["outputs"]):
                if out["type"] == "stream":
                    cell_outputs.append(new_output(
                        output_type="stream",
                        name=out.get("name", "stdout"),
                        text=out["text"]
                    ))
                elif out["type"] == "execute_result":
                    cell_outputs.append(new_output(
                        output_type="execute_result",
                        data=out.get("data", {}),
                        execution_count=cell["execution_count"]
                    ))
                elif out["type"] == "display_data":
                    cell_outputs.append(new_output(
                        output_type="display_data",
                        data=out.get("data", {})
                    ))
                elif out["type"] == "error":
                    cell_outputs.append(new_output(
                        output_type="error",
                        ename=out.get("ename", "Error"),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", [])
                    ))
            
            code_cell.outputs = cell_outputs
            nb.cells.append(code_cell)
        
        # Write notebook file atomically. The live notebook is rewritten in
        # place on every save and is packed into every session export, so a
        # truncate-and-rewrite window let the exporter capture a half-written
        # file -- an archive that imports "successfully" with a corrupt notebook.
        tmp_path = self.notebook_path.with_name(
            f"{self.notebook_path.name}.{os.getpid()}.tmp"
        )
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        try:
            with lifecycle_file_lock(self.session_id):
                assert_session_generation_active(self.session_id)
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                    f.flush()
                    os.fsync(f.fileno())
                assert_session_generation_active(self.session_id)
                os.replace(tmp_path, self.notebook_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        
        logger.info(f"Saved notebook: {self.notebook_path}")
    
    def _load_runtime_persistence_metadata(self) -> None:
        """Load only persistence metadata, even when no notebook is present."""
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            persisted = payload.get("runtime_persistence")
            if isinstance(persisted, dict):
                self.runtime_persistence.update(persisted)
                self._migrate_committed_runtime_metadata()
        except Exception:
            logger.warning(
                "[%s] Could not read runtime persistence metadata",
                self.session_id,
                exc_info=True,
            )

    def _migrate_committed_runtime_metadata(self) -> None:
        """Derive explicit committed fields from pre-separation metadata."""
        persistence = self.runtime_persistence
        source = persistence.get("committed_source")
        digest = persistence.get("committed_digest")
        revision = int(persistence.get("committed_revision") or 0)
        if (
            not source
            and persistence.get("commit_source") in {
                "regular_save", "snapshot_apply"
            }
            and isinstance(persistence.get("runtime_digest"), str)
            and len(str(persistence.get("runtime_digest"))) == 64
        ):
            source = persistence.get("commit_source")
            digest = persistence.get("runtime_digest")
            revision = int(persistence.get("saved_revision") or 0)
        persistence["committed_source"] = source
        persistence["committed_digest"] = digest
        persistence["committed_revision"] = max(0, revision)

    def update_runtime_persistence(
        self,
        status: str,
        *,
        current_revision: int,
        saved_revision: int,
        failed_variables: Optional[List[str]] = None,
        error_summary: Optional[str] = None,
        requires_recompute: bool = False,
        recompute_variables: Optional[List[str]] = None,
        commit_source: Optional[str] = None,
        runtime_digest: Optional[str] = None,
        budget_violations: Optional[List[Dict[str, Any]]] = None,
        top_offenders: Optional[List[Dict[str, Any]]] = None,
        recommended_action: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Persist the live runtime lifecycle without exposing raw tracebacks."""
        now = datetime.now().isoformat()
        previous = dict(self.runtime_persistence)
        from backend.utils.error_parsing import bounded_diagnostic, compact_traceback

        def _safe_runtime_text(value: Any, limit: int) -> str:
            # Redaction must precede slicing: a credential or private path near
            # the retained edge must never be persisted merely because a caller
            # supplied an already-long diagnostic string.
            rendered = compact_traceback(
                str(value or ""), max_chars=max(512, limit * 4),
            )
            rendered = rendered.replace(
                "[... traceback frames omitted ...]", "",
            )
            return " ".join(rendered.split())[:limit]

        safe_error = None
        if error_summary:
            safe_error = bounded_diagnostic(error_summary, 500)
        safe_variables = sorted({
            _safe_runtime_text(name, 128)
            for name in (failed_variables or [])
            if _safe_runtime_text(name, 128)
        })[:100]
        safe_budget_violations = []
        for item in list(budget_violations or [])[:20]:
            if not isinstance(item, dict):
                continue
            resource = _safe_runtime_text(item.get("resource"), 128)
            if not resource:
                continue
            safe_budget_violations.append({
                "resource": resource,
                "observed": item.get("observed")
                if isinstance(item.get("observed"), (int, float))
                and not isinstance(item.get("observed"), bool)
                else None,
                "limit": item.get("limit")
                if isinstance(item.get("limit"), (int, float))
                and not isinstance(item.get("limit"), bool)
                else None,
            })
        safe_top_offenders = []
        for item in list(top_offenders or [])[:20]:
            if not isinstance(item, dict):
                continue
            name = _safe_runtime_text(item.get("name"), 128)
            if not name:
                continue
            offender = {
                "name": name,
                "type": _safe_runtime_text(
                    item.get("type") or item.get("type_name"), 128,
                ),
            }
            raw_size = (
                item.get("bytes")
                if item.get("bytes") is not None
                else item.get("estimated_bytes")
            )
            if isinstance(raw_size, (int, float)) and not isinstance(raw_size, bool):
                offender["bytes"] = max(0, raw_size)
            safe_top_offenders.append(offender)
        self.runtime_persistence.update({
            "status": status,
            "current_revision": max(0, int(current_revision)),
            "saved_revision": max(0, int(saved_revision)),
            "failed_variables": safe_variables,
            "error_summary": safe_error,
            "requires_recompute": bool(requires_recompute),
            "recompute_variables": sorted({
                _safe_runtime_text(name, 128)
                for name in (recompute_variables or [])
                if _safe_runtime_text(name, 128)
            })[:100],
            "recommended_action": (
                _safe_runtime_text(recommended_action, 128)
                or (
                    "recompute_required_dependencies"
                    if requires_recompute else None
                )
            ),
            "commit_source": (
                str(commit_source)
                if commit_source is not None
                else previous.get("commit_source")
            ),
            "runtime_digest": (
                str(runtime_digest)
                if runtime_digest is not None
                else previous.get("runtime_digest")
            ),
            "budget_violations": safe_budget_violations,
            "top_offenders": safe_top_offenders,
        })
        if status == "live_dirty" and not previous.get("dirty_since"):
            self.runtime_persistence["dirty_since"] = now
        elif status == "current":
            self.runtime_persistence.update({
                "dirty_since": None,
                "last_saved_at": now,
                "failed_variables": [],
                "error_summary": None,
                "requires_recompute": False,
                "recompute_variables": [],
                "recommended_action": None,
                "budget_violations": [],
                "top_offenders": [],
                "committed_revision": max(0, int(saved_revision)),
                "committed_source": (
                    str(commit_source)
                    if commit_source is not None
                    else previous.get("committed_source")
                ),
                "committed_digest": (
                    str(runtime_digest)
                    if runtime_digest is not None
                    else previous.get("committed_digest")
                ),
                "last_attempt_status": "succeeded",
                "last_attempt_at": now,
                "last_attempt_revision": max(0, int(current_revision)),
                "last_attempt_error": None,
                "last_attempt_failed_variables": [],
                "degraded_recovery": False,
                "degraded_from_revision": None,
                "degraded_to_revision": None,
            })
        elif status == "failed":
            self.runtime_persistence.update({
                "last_failed_at": now,
                "last_attempt_status": "failed",
                "last_attempt_at": now,
                "last_attempt_revision": max(0, int(current_revision)),
                "last_attempt_error": safe_error,
                "last_attempt_failed_variables": safe_variables,
            })
        if persist:
            self.save_metadata()
        return dict(self.runtime_persistence)

    def record_runtime_restored(
        self,
        *,
        current_revision: int,
        saved_revision: int,
        commit_source: Optional[str],
        degraded_recovery: bool = False,
        attempted_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record rehydrate without fabricating a new persistence event."""
        restored_revision = max(0, int(saved_revision))
        attempted = max(
            restored_revision,
            int(
                attempted_revision
                if attempted_revision is not None
                else self.runtime_persistence.get("last_attempt_revision") or 0
            ),
        )
        self.runtime_persistence.update({
            "status": "degraded" if degraded_recovery else "current",
            "current_revision": restored_revision,
            "saved_revision": restored_revision,
            "commit_source": (
                str(commit_source)
                if commit_source is not None
                else self.runtime_persistence.get("commit_source")
            ),
            "last_restored_at": datetime.now().isoformat(),
            "degraded_recovery": bool(degraded_recovery),
            "degraded_from_revision": attempted if degraded_recovery else None,
            "degraded_to_revision": restored_revision if degraded_recovery else None,
            "requires_recompute": (
                True
                if degraded_recovery
                else bool(self.runtime_persistence.get("requires_recompute"))
            ),
            "recommended_action": (
                "recompute_or_reduce_runtime"
                if degraded_recovery
                else self.runtime_persistence.get("recommended_action")
            ),
        })
        self._migrate_committed_runtime_metadata()
        self.save_metadata()
        return dict(self.runtime_persistence)

    def _metadata_payload(self) -> Dict[str, Any]:
        persistence = dict(self.runtime_persistence)
        self._migrate_committed_runtime_metadata()
        persistence = dict(self.runtime_persistence)
        runtime_digest = persistence.get("committed_digest")
        has_current_session = bool(
            persistence.get("committed_source") in {
                "regular_save", "snapshot_apply"
            }
            and self.session_path.exists()
            and _verify_pkl_integrity(self.session_path, _integrity_dir(self.session_id))
            and isinstance(runtime_digest, str)
            and len(runtime_digest) == 64
            and _sha256_file(self.session_path) == runtime_digest
        )
        return {
            "session_id": self.session_id,
            "execution_count": self.execution_count,
            "cell_count": len(self.cells),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # Compatibility field, now derived from committed generation state.
            "has_session": has_current_session,
            "runtime_persistence": persistence,
        }

    def save_metadata(self):
        """Save state metadata atomically."""
        state_data = self._metadata_payload()
        tmp_path = self.state_path.with_name(
            f"{self.state_path.name}.{os.getpid()}.tmp"
        )
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        try:
            with lifecycle_file_lock(self.session_id):
                assert_session_generation_active(self.session_id)
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                assert_session_generation_active(self.session_id)
                os.replace(tmp_path, self.state_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    
    def save(self):
        """Save notebook and metadata"""
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        with lifecycle_file_lock(self.session_id):
            assert_session_generation_active(self.session_id)
            self.save_notebook()
            self.save_metadata()
    
    def load(self) -> bool:
        """Load notebook state from disk"""
        if not self.notebook_path.exists():
            return False
        
        try:
            # Load notebook
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            self.cells = []
            for cell in nb.cells:
                if cell.cell_type == "code":
                    outputs = []
                    for out in cell.outputs:
                        if out.output_type == "stream":
                            outputs.append({
                                "type": "stream",
                                "name": out.get("name", "stdout"),
                                "text": out.text
                            })
                        elif out.output_type == "execute_result":
                            outputs.append({
                                "type": "execute_result",
                                "data": dict(out.data)
                            })
                        elif out.output_type == "display_data":
                            outputs.append({
                                "type": "display_data",
                                "data": dict(out.data)
                            })
                        elif out.output_type == "error":
                            outputs.append({
                                "type": "error",
                                "ename": out.ename,
                                "evalue": out.evalue,
                                "traceback": out.traceback
                            })
                    
                    self.cells.append({
                        "code": cell.source,
                        "outputs": outputs,
                        "execution_count": cell.execution_count or 0,
                        "executed_at": None
                    })
            
            # Load state metadata
            if self.state_path.exists():
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                self.execution_count = state_data.get("execution_count", 0)
                self.created_at = state_data.get("created_at", self.created_at)
                self.updated_at = state_data.get("updated_at", self.updated_at)
                persisted = state_data.get("runtime_persistence")
                if isinstance(persisted, dict):
                    self.runtime_persistence.update(persisted)
                    self._migrate_committed_runtime_metadata()
            
            logger.info(f"Loaded notebook state: {len(self.cells)} cells")
            return True
        except Exception as e:
            logger.exception("Failed to load notebook state")
            return False
    
    def has_session_dump(self) -> bool:
        """Check for a committed, integrity-valid dump of the live generation."""
        self._migrate_committed_runtime_metadata()
        persistence = self.runtime_persistence
        runtime_digest = persistence.get("committed_digest")
        return bool(
            persistence.get("committed_source") in {
                "regular_save", "snapshot_apply"
            }
            and self.session_path.exists()
            and _verify_pkl_integrity(self.session_path, _integrity_dir(self.session_id))
            and isinstance(runtime_digest, str)
            and len(runtime_digest) == 64
            and _sha256_file(self.session_path) == runtime_digest
        )


# Template for unwinding any stale SCAI no-op stubs from plt.close/plt.show
# and installing fresh ones.  Embedded into f-string code templates via
# {_TPL_NOOP_INSTALL}.  The string must NOT contain { or } to be f-string safe.
_TPL_NOOP_INSTALL = (
    "# Temporarily disable plt.close/plt.show to keep figures alive.\n"
    "# Unwind any leftover SCAI no-op stubs from prior runs before installing fresh ones.\n"
    "_unwrap_close_guard = 0\n"
    "while getattr(plt.close, '_scai_noop', False) and _unwrap_close_guard < 64:\n"
    "    plt.close = getattr(plt.close, '_scai_prev', plt.close)\n"
    "    _unwrap_close_guard += 1\n"
    "_unwrap_show_guard = 0\n"
    "while getattr(plt.show, '_scai_noop', False) and _unwrap_show_guard < 64:\n"
    "    plt.show = getattr(plt.show, '_scai_prev', plt.show)\n"
    "    _unwrap_show_guard += 1\n"
    "# Defensive baseline restore for legacy sessions with unmarked stubs.\n"
    "try:\n"
    "    if '_original_plt_close' in globals() and plt.close is not _original_plt_close \\\n"
    "            and not getattr(plt.close, '_scai_noop', False):\n"
    "        plt.close = _original_plt_close\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    if '_capturing_show' in globals() and plt.show is not _capturing_show \\\n"
    "            and not getattr(plt.show, '_scai_noop', False):\n"
    "        plt.show = _capturing_show\n"
    "except Exception:\n"
    "    pass\n"
    "_orig_close = plt.close\n"
    "_orig_show = plt.show\n"
    "def _noop_close(*a, **kw): return None\n"
    "_noop_close._scai_noop = True\n"
    "_noop_close._scai_prev = _orig_close\n"
    "def _noop_show(*a, **kw): return None\n"
    "_noop_show._scai_noop = True\n"
    "_noop_show._scai_prev = _orig_show\n"
    "plt.close = _noop_close\n"
    "plt.show = _noop_show\n"
)
