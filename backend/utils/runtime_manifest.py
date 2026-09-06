"""The committed, signed contract for one runtime generation.

Restore binds whatever dill frames a dump contains and checks them against
nothing, so until now the only thing standing between a partial dump and a
resumed analysis was the save aborting. This manifest is what a generation
declares it persisted: integrity checks the dump against it, coverage records
what was left out of it, and coherence asks which workflow state it belongs to.

Two properties make it worth trusting. It is signed with the installation key,
because the dump's own HMAC covers bytes rather than claims -- nothing in it
protects ``saved_names`` or ``state_revision``. And it is validated strictly on
read, because a manifest that merely parses as an object could declare nothing
at all and would then report a complete generation.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from backend.utils.crypto import sign_hmac_sha256, verify_hmac_sha256

RUNTIME_MANIFEST_SCHEMA_VERSION = 1
RUNTIME_MANIFEST_SUFFIX = "_session.runtime.json"

#: Why a live name is absent from the dump. The three are different facts about
#: the same generation: a value dill refused, a value never offered because of
#: its kind, and a value excluded as a volatile handle. Collapsing them loses
#: the only signal that says whether a name is worth recomputing.
SKIP_REASONS = frozenset(
    {"serialization_failed", "excluded_by_kind", "excluded_volatile"}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


COHERENCE_CURRENT = "current"
COHERENCE_STALE = "stale"
COHERENCE_INCONSISTENT = "inconsistent"
COHERENCE_UNAVAILABLE = "unavailable"


def runtime_coherence(
    *, saved_revision: int | None, current_revision: int, has_generation: bool = True
) -> str:
    """Whether the newest committed runtime still describes the live namespace.

    Both numbers belong to the persistence layer and advance with it:
    ``current_revision`` rises whenever an execution could have changed user
    globals, ``saved_revision`` only when a generation commits. Comparing them
    is therefore a statement about this session's own history.

    An earlier draft compared the session *lifecycle generation* instead. That
    counter fences incarnations -- deletion, same-id re-import -- and is
    normally 1 for the whole life of a session, so the comparison could never
    be anything but equal and the verdict was permanently "current".

    * ``stale`` -- the live namespace moved past the newest committed
      generation, which is what a failed closing save leaves behind. Results
      stay viewable and exportable; exact continuation is not promised.
    * ``inconsistent`` -- a generation claims a revision the live runtime never
      reached. Not the benign converse of stale: it means a commit recorded a
      revision that was rolled back, and reporting "current" would hide it.
    * ``unavailable`` -- nothing committed yet.

    This is a report. The only thing it gates is silently binding an older
    namespace to a newer graph position (see ``is_forward_resume_blocked``).
    """
    if not has_generation or saved_revision is None:
        return COHERENCE_UNAVAILABLE
    saved = int(saved_revision or 0)
    current = int(current_revision or 0)
    if saved < current:
        return COHERENCE_STALE
    if saved > current:
        return COHERENCE_INCONSISTENT
    return COHERENCE_CURRENT


class RuntimeManifestError(RuntimeError):
    """The manifest is absent, unsigned, tampered with, or malformed."""


def runtime_manifest_path(
    session_id: str, *, states_dir: str | Path | None = None
) -> Path:
    """Where a session's generation manifest lives, beside its dump."""
    if states_dir is None:
        from backend.config import get_session_states

        states_dir = get_session_states(session_id)
    return Path(states_dir) / f"{session_id}{RUNTIME_MANIFEST_SUFFIX}"


def runtime_manifest_sig_path(manifest_path: str | Path) -> Path:
    """The detached signature beside the manifest."""
    path = Path(manifest_path)
    return path.with_name(path.name + ".sig")


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """The exact bytes that are signed, and therefore the only ones written.

    Sorted keys and no whitespace, so a manifest that round-trips through JSON
    produces the same bytes and the same signature.
    """
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_runtime_manifest(
    *,
    revision: int,
    state_revision: int,
    pkl_digest: str,
    saved_names: Iterable[str],
    skipped_names: Iterable[dict[str, Any]],
    serializer_version: str,
) -> dict[str, Any]:
    """Assemble a generation's declaration, rejecting anything unusable.

    Validation happens here rather than at the call sites so that the same
    rules apply to a manifest being written and to one being read back: reading
    reconstructs through this function, which is what makes a signed but
    nonsensical manifest a hard error instead of a silent one.
    """
    if not _SHA256.match(str(pkl_digest or "")):
        raise ValueError("pkl_digest must be a lowercase sha256 hex string")

    saved = [str(name) for name in saved_names]
    if len(saved) != len(set(saved)):
        raise ValueError("saved_names contains a duplicate")
    for name in saved:
        if not name.isidentifier():
            raise ValueError(f"saved_names entry is not a Python identifier: {name!r}")

    skipped: list[dict[str, str]] = []
    for entry in skipped_names:
        if not isinstance(entry, dict):
            raise ValueError(f"skipped_names entry is not an object: {entry!r}")
        name = str(entry.get("name", ""))
        reason = str(entry.get("reason", ""))
        if not name.isidentifier():
            raise ValueError(
                f"skipped_names entry is not a Python identifier: {name!r}"
            )
        if reason not in SKIP_REASONS:
            raise ValueError(f"unknown skipped_names reason: {reason!r}")
        skipped.append(
            {"name": name, "type": str(entry.get("type", "")), "reason": reason}
        )

    overlap = sorted(set(saved) & {entry["name"] for entry in skipped})
    if overlap:
        raise ValueError("names appear both saved and skipped: " + ", ".join(overlap))

    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "revision": int(revision),
        "state_revision": int(state_revision),
        "pkl_sha256": str(pkl_digest),
        "saved_names": saved,
        "skipped_names": sorted(skipped, key=lambda item: item["name"]),
        "serializer_version": str(serializer_version),
    }


def _write_durably(target: Path, data: bytes) -> None:
    """Write, replace, and fsync the directory entry.

    Without the directory fsync the rename can be lost to a power cut while the
    transaction has already published its COMMITTED journal -- a generation
    reported as installed whose manifest is not there, which recovery then
    rolls back or refuses.
    """
    from backend.utils.files import fsync_directory

    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        fsync_directory(target.parent)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_runtime_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Install a manifest and its detached signature, both fsynced.

    The signature is written after the manifest: a manifest without a signature
    is refused on read, so the intermediate state fails closed rather than
    presenting unsigned content as trustworthy.
    """
    path = Path(path)
    payload = canonical_manifest_bytes(manifest)
    signature = sign_hmac_sha256(payload)
    _write_durably(path, payload)
    _write_durably(runtime_manifest_sig_path(path), signature.encode("utf-8"))


def read_runtime_manifest(path: str | Path) -> dict[str, Any]:
    """Return a verified manifest, or raise.

    There is no degraded mode here on purpose. A caller that wants to tolerate
    a generation predating manifests checks for absence itself; everything that
    is present must verify.
    """
    path = Path(path)
    try:
        payload = path.read_bytes()
        signature = runtime_manifest_sig_path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeManifestError(
            f"runtime manifest or its signature is unreadable: {exc}"
        ) from exc

    if not verify_hmac_sha256(payload, signature):
        raise RuntimeManifestError("runtime manifest signature does not verify")

    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(
            f"runtime manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeManifestError("runtime manifest is not an object")
    if manifest.get("schema_version") != RUNTIME_MANIFEST_SCHEMA_VERSION:
        raise RuntimeManifestError(
            "unsupported runtime manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    try:
        return build_runtime_manifest(
            revision=manifest["revision"],
            state_revision=manifest["state_revision"],
            pkl_digest=manifest["pkl_sha256"],
            saved_names=manifest["saved_names"],
            skipped_names=manifest["skipped_names"],
            serializer_version=manifest["serializer_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeManifestError(
            f"runtime manifest failed validation: {exc}"
        ) from exc
