"""Authenticated regular-save transaction primitives."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backend.utils.crypto import get_integrity_secret, sign_hmac_sha256

logger = logging.getLogger("backend.utils.sandbox_parts.shared")

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
    runtime_manifest_path: Path | None = None,
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
        runtime_manifest_path,
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
    runtime_manifest_path: Path | None = None,
    runtime_manifest: Dict[str, Any] | None = None,
    commit_manifest=None,
    retire_sidecars=None,
    target_revision: int = 0,
    authoritative_pkl_names: set[str] | None = None,
    fault_hook=None,
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
                copy_durable(component, backup)
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
                ),
            )
            if fault_hook is not None:
                fault_hook("runtime_manifest_replaced")
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

        # Components written inside the committed region are authenticated
        # against what they now hold, not against the digest recorded when the
        # transaction registered their previous copy.
        from backend.utils.runtime_manifest import runtime_manifest_sig_path

        target_paths = {
            "pkl": live_pkl,
            "sig": live_sig,
            "metadata": metadata_path,
            "manifest": manifest_path,
            "runtime_manifest": runtime_manifest_path,
            "runtime_manifest_sig": (
                runtime_manifest_sig_path(runtime_manifest_path)
                if runtime_manifest_path is not None
                else None
            ),
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
