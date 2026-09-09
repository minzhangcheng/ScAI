"""The signed contract for one decision snapshot. Part B spec 4.4.

Two properties make a bundle worth reading. The manifest is signed with this
installation's integrity key, because per-file digests prove only what the
manifest claims and say nothing about the manifest itself; and it is validated
strictly on read, by reconstructing it through the same builder that wrote it,
because a manifest that merely parses could declare an empty bundle and would
then restore a session holding nothing.

It is verified before any other file in the bundle is opened, and before any
dill frame is loaded. Re-signing anything in a restored session happens only
after that verification passes -- re-signing first would authenticate whatever
had been substituted.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from backend.utils.crypto import sign_hmac_sha256, verify_hmac_sha256
from backend.utils.runtime_manifest import CARRIED_KINDS, SKIP_REASONS

from .gates import (
    DECISION_GATES,
    next_node_for,
    restored_status_for,
)
from .layout import (
    CONTINUATION_NAME,
    MANIFEST_NAME,
    MANIFEST_SIG_NAME,
    PAIR_CHECKPOINT_RELPATH,
    PAIR_EXECUTOR_RELPATH,
    PAIR_STATE_RELPATH,
    RUNTIME_MANIFEST_RELPATH,
    RUNTIME_PKL_RELPATH,
    SESSION_STATE_NAME,
    parse_snapshot_id,
)

SNAPSHOT_MANIFEST_SCHEMA_VERSION = 1
CONTINUATION_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: Files every bundle must declare. A manifest that omits one of these
#: describes something a restore cannot use, and saying so here is cheaper than
#: discovering it halfway through materialising a session.
_REQUIRED_FILES: tuple[str, ...] = (
    CONTINUATION_NAME,
    SESSION_STATE_NAME,
    RUNTIME_PKL_RELPATH,
    RUNTIME_MANIFEST_RELPATH,
    PAIR_STATE_RELPATH,
    PAIR_CHECKPOINT_RELPATH,
)

#: Never declared as bundle content: the manifest cannot contain its own
#: digest, and its signature is not part of what it signs.
_FORBIDDEN_FILES: frozenset[str] = frozenset({MANIFEST_NAME, MANIFEST_SIG_NAME})


class SnapshotManifestError(RuntimeError):
    """The manifest is absent, unsigned, tampered with, or malformed."""


def manifest_sig_path(path: str | Path) -> Path:
    """The detached signature beside a manifest."""
    path = Path(path)
    return path.with_name(path.name + ".sig")


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The exact bytes that are signed, and therefore the only ones written."""
    return json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _relpath(value: Any) -> str:
    """A bundle-relative path that cannot escape the bundle.

    The manifest drives what a restore reads and where it writes, so a key like
    ``../../etc/passwd`` has to be refused where it is declared rather than
    where it is used. Requiring the canonical spelling as well means two
    manifests describing the same file cannot differ in their bytes and so
    cannot differ in their signature.
    """
    text = str(value)
    if not text or text != text.strip() or "\x00" in text or "\\" in text:
        raise ValueError(f"not a usable bundle path: {value!r}")
    pure = PurePosixPath(text)
    if not pure.parts or pure.is_absolute():
        raise ValueError(f"bundle path names no file: {value!r}")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"bundle path escapes the bundle: {value!r}")
    if pure.as_posix() != text:
        raise ValueError(f"bundle path is not canonical: {value!r}")
    return text


def _digest(value: Any, *, field: str) -> str:
    text = str(value)
    if not _SHA256.match(text):
        raise ValueError(f"{field} must be a lowercase sha256 hex string")
    return text


def _size(value: Any, *, field: str) -> int:
    size = int(value)
    if size < 0:
        raise ValueError(f"{field} must not be negative")
    return size


def _file_table(entries: Mapping[str, Mapping[str, Any]], *, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, Mapping):
        raise ValueError(f"{field} is not an object")
    table: dict[str, dict[str, Any]] = {}
    for key, entry in entries.items():
        path = _relpath(key)
        if path in _FORBIDDEN_FILES:
            raise ValueError(f"{field} declares its own manifest: {path}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"{field}[{path}] is not an object")
        table[path] = {
            "sha256": _digest(entry.get("sha256"), field=f"{field}[{path}].sha256"),
            "size": _size(entry.get("size"), field=f"{field}[{path}].size"),
        }
    return table


def _aware_timestamp(value: Any, *, field: str) -> str:
    """An ISO timestamp that carries its offset.

    Naive timestamps were a real defect on the runtime record (repair spec
    B1-4): the browser reads them as local time and the reader's zone decides
    what they mean. A snapshot's listing shows this value to a user who is
    usually not in the container's timezone, so the offset is required rather
    than assumed.
    """
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must carry a UTC offset: {value!r}")
    return text


def _coverage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What the generation held and what it could not hold.

    Copied verbatim from the runtime manifest the save signed, and reusing that
    module's vocabularies rather than restating them: a reason or a kind that
    the save can emit and the snapshot cannot record would make a bundle
    unreadable for a reason that has nothing to do with the bundle.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("coverage is not an object")
    saved = int(payload.get("saved_names") or 0)
    if saved < 0:
        raise ValueError("coverage.saved_names must not be negative")

    skipped: list[dict[str, str]] = []
    for entry in payload.get("skipped_names") or []:
        if not isinstance(entry, Mapping):
            raise ValueError(f"coverage.skipped_names entry is not an object: {entry!r}")
        name = str(entry.get("name", ""))
        reason = str(entry.get("reason", ""))
        if not name.isidentifier():
            raise ValueError(f"coverage.skipped_names entry is not an identifier: {name!r}")
        if reason not in SKIP_REASONS:
            raise ValueError(f"unknown coverage.skipped_names reason: {reason!r}")
        skipped.append({"name": name, "type": str(entry.get("type", "")), "reason": reason})

    carried: list[dict[str, str]] = []
    for entry in payload.get("carried_definitions") or []:
        if not isinstance(entry, Mapping):
            raise ValueError(f"coverage.carried_definitions entry is not an object: {entry!r}")
        name = str(entry.get("name", ""))
        kind = str(entry.get("kind", ""))
        if not name.isidentifier():
            raise ValueError(
                f"coverage.carried_definitions entry is not an identifier: {name!r}"
            )
        if kind not in CARRIED_KINDS:
            raise ValueError(f"unknown coverage.carried_definitions kind: {kind!r}")
        item = {"name": name, "kind": kind}
        note = str(entry.get("note", ""))
        if note:
            item["note"] = note
        carried.append(item)

    return {
        "saved_names": saved,
        "skipped_names": sorted(skipped, key=lambda item: item["name"]),
        "carried_definitions": sorted(carried, key=lambda item: item["name"]),
    }


def _skipped_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Workspace entries the capture refused to copy, and why.

    Refusing a symlink, a FIFO or a path that resolves outside the workspace is
    correct, but a bundle that is quietly missing a file the analysis wrote is
    the coverage problem again in a new place. Recorded so a restore can say
    what is absent.
    """
    result: list[dict[str, str]] = []
    for entry in entries or []:
        if not isinstance(entry, Mapping):
            raise ValueError(f"skipped_entries entry is not an object: {entry!r}")
        path = str(entry.get("path", ""))
        reason = str(entry.get("reason", ""))
        if not path or not reason:
            raise ValueError(f"skipped_entries entry needs a path and a reason: {entry!r}")
        result.append({"path": path, "reason": reason})
    return sorted(result, key=lambda item: item["path"])


def build_snapshot_manifest(
    *,
    snapshot_id: str,
    source_session_id: str,
    source_session_name: str,
    source_lifecycle_generation: int,
    scai_version: str,
    created_at: str,
    round_number: int,
    gate: str,
    retention_rank: int,
    runtime_revision: int,
    pkl_digest: str,
    files: Mapping[str, Mapping[str, Any]],
    uploads: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Any],
    executor_checkpoint_present: bool,
    bytes_total: int,
    capture_seconds: float,
    skipped_entries: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble one bundle's declaration, rejecting anything unusable.

    Validation lives here rather than at the call sites so that the same rules
    apply to a manifest being written and to one being read back: reading
    reconstructs through this function, which is what makes a signed but
    nonsensical manifest a hard error instead of a silent one.
    """
    parsed_round, parsed_gate, parsed_revision = parse_snapshot_id(snapshot_id)
    if str(gate) not in DECISION_GATES:
        raise ValueError(f"not a decision gate: {gate!r}")
    if parsed_gate != str(gate):
        raise ValueError(f"snapshot id names {parsed_gate!r}, record says {gate!r}")
    if parsed_round != int(round_number):
        raise ValueError(
            f"snapshot id names round {parsed_round}, record says {int(round_number)}"
        )
    if parsed_revision != int(runtime_revision):
        raise ValueError(
            f"snapshot id names revision {parsed_revision}, record says "
            f"{int(runtime_revision)}"
        )
    if int(retention_rank) < 1 or int(retention_rank) > 4:
        raise ValueError(f"retention_rank out of range: {retention_rank!r}")

    file_table = _file_table(files, field="files")
    upload_table = _file_table(uploads, field="uploads")
    missing = [name for name in _REQUIRED_FILES if name not in file_table]
    if missing:
        raise ValueError("bundle does not declare: " + ", ".join(missing))

    # The dump's digest appears twice on purpose: once as the generation's
    # identity, once as the bundle file's content. A bundle whose two copies
    # disagree is describing a dump it does not carry.
    if file_table[RUNTIME_PKL_RELPATH]["sha256"] != _digest(pkl_digest, field="pkl_sha256"):
        raise ValueError("pkl_sha256 does not match the declared runtime dump")

    present = bool(executor_checkpoint_present)
    if present != (PAIR_EXECUTOR_RELPATH in file_table):
        raise ValueError(
            "executor_checkpoint_present disagrees with the declared files"
        )

    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "snapshot_id": str(snapshot_id),
        "source_session_id": str(source_session_id),
        "source_session_name": str(source_session_name),
        "source_lifecycle_generation": max(0, int(source_lifecycle_generation)),
        "scai_version": str(scai_version),
        "created_at": _aware_timestamp(created_at, field="created_at"),
        "round": int(round_number),
        "gate": str(gate),
        "retention_rank": int(retention_rank),
        "runtime_revision": int(runtime_revision),
        "pkl_sha256": str(pkl_digest),
        "files": file_table,
        "uploads": upload_table,
        "coverage": _coverage(coverage),
        "executor_checkpoint_present": present,
        "skipped_entries": _skipped_entries(skipped_entries),
        "bytes_total": _size(bytes_total, field="bytes_total"),
        "capture_seconds": round(float(capture_seconds), 3),
    }


def write_snapshot_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Install a manifest and its detached signature, both durable.

    The signature is written second: a manifest without one is refused on read,
    so a crash between the two leaves a bundle that fails closed rather than
    one that presents unsigned content as trustworthy.
    """
    from backend.utils.files import atomic_write_bytes

    path = Path(path)
    payload = canonical_manifest_bytes(manifest)
    atomic_write_bytes(path, payload)
    atomic_write_bytes(
        manifest_sig_path(path), sign_hmac_sha256(payload).encode("utf-8"),
    )


def read_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    """Return a verified manifest, or raise.

    There is no degraded mode. Unlike the runtime manifest, which has to
    tolerate generations written before manifests existed, every snapshot ever
    written carries one; a bundle without a readable manifest is not an old
    bundle, it is a broken one.
    """
    path = Path(path)
    try:
        payload = path.read_bytes()
        signature = manifest_sig_path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SnapshotManifestError(
            f"snapshot manifest or its signature is unreadable: {exc}"
        ) from exc

    if not verify_hmac_sha256(payload, signature):
        raise SnapshotManifestError("snapshot manifest signature does not verify")

    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotManifestError(f"snapshot manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SnapshotManifestError("snapshot manifest is not an object")
    if manifest.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise SnapshotManifestError(
            "unsupported snapshot manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    try:
        return build_snapshot_manifest(
            snapshot_id=manifest["snapshot_id"],
            source_session_id=manifest["source_session_id"],
            source_session_name=manifest["source_session_name"],
            source_lifecycle_generation=manifest["source_lifecycle_generation"],
            scai_version=manifest["scai_version"],
            created_at=manifest["created_at"],
            round_number=manifest["round"],
            gate=manifest["gate"],
            retention_rank=manifest["retention_rank"],
            runtime_revision=manifest["runtime_revision"],
            pkl_digest=manifest["pkl_sha256"],
            files=manifest["files"],
            uploads=manifest["uploads"],
            coverage=manifest["coverage"],
            executor_checkpoint_present=manifest["executor_checkpoint_present"],
            bytes_total=manifest["bytes_total"],
            capture_seconds=manifest["capture_seconds"],
            skipped_entries=manifest.get("skipped_entries") or [],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotManifestError(
            f"snapshot manifest failed validation: {exc}"
        ) from exc


def build_continuation(
    *,
    gate: str,
    analysis_phase: str,
    current_round: int,
    state_version: int,
    workflow_state_keys: Iterable[str],
) -> dict[str, Any]:
    """What a restore arms the new session with.

    ``next_node`` and ``restored_status`` are derived rather than supplied:
    both follow from the gate, and a capsule that could disagree with its own
    gate would put a restored session either on the wrong dialog or into the
    wrong part of the graph. The capsule lives inside ``files``, so the
    manifest's signature covers it.

    The workflow-state values themselves are deliberately not here. They are
    read back at restore from the paired graph checkpoint, which already holds
    them; only their key set is recorded, so a restore can refuse a checkpoint
    that has lost a channel the capture saw.
    """
    keys = sorted({str(key) for key in workflow_state_keys})
    return {
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "gate": str(gate),
        "next_node": next_node_for(gate),
        "restored_status": restored_status_for(gate),
        "analysis_phase": str(analysis_phase),
        "current_round": int(current_round),
        "state_version": max(0, int(state_version)),
        "workflow_state_keys": keys,
    }


def validate_continuation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the capsule, or raise ``SnapshotManifestError``."""
    if not isinstance(payload, Mapping):
        raise SnapshotManifestError("continuation is not an object")
    if payload.get("schema_version") != CONTINUATION_SCHEMA_VERSION:
        raise SnapshotManifestError(
            f"unsupported continuation schema_version: {payload.get('schema_version')!r}"
        )
    gate = str(payload.get("gate", ""))
    if gate not in DECISION_GATES:
        raise SnapshotManifestError(f"continuation names no decision gate: {gate!r}")
    try:
        rebuilt = build_continuation(
            gate=gate,
            analysis_phase=payload["analysis_phase"],
            current_round=payload["current_round"],
            state_version=payload["state_version"],
            workflow_state_keys=payload["workflow_state_keys"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotManifestError(f"continuation failed validation: {exc}") from exc
    if payload.get("next_node") != rebuilt["next_node"]:
        raise SnapshotManifestError(
            f"continuation next_node {payload.get('next_node')!r} does not follow "
            f"from gate {gate!r}"
        )
    if payload.get("restored_status") != rebuilt["restored_status"]:
        raise SnapshotManifestError(
            f"continuation restored_status {payload.get('restored_status')!r} does "
            f"not follow from gate {gate!r}"
        )
    return rebuilt
