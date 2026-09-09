"""Where a decision snapshot lives, and what its identifier says.

Part B spec 4.3. Every path below is derived from the configured snapshot root
plus a validated session id and a parsed snapshot id, never from a string that
arrived in a request or in a file. A manifest therefore cannot be used to name
somewhere else on disk, which is the same rule the regular-save journal follows
(``_regular_component_paths`` rebuilds each component path from a trusted root
rather than trusting the path it was handed).
"""
from __future__ import annotations

import re
from pathlib import Path

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
)

#: The decision a snapshot precedes, as the one word that appears in its id.
#: Spec 1 lists exactly these four and excludes every error or incident gate --
#: a snapshot exists so a user can revisit a decision, and an LLM failure is not
#: a decision.
GATE_WORDS: dict[str, str] = {
    GATE_ANNOTATION_CONFIRMATION: "annotation",
    GATE_DIRECTION_SELECTION: "direction",
    GATE_INTEGRATION_SELECTION: "integration",
    GATE_ROUND_WAIT: "round",
}

#: The inverse, for reading an id back.
WORD_GATES: dict[str, str] = {word: gate for gate, word in GATE_WORDS.items()}

#: ``r003-round-g400``: the round it precedes, the decision, and the runtime
#: revision the commit installed. Round and gate alone are not unique -- a
#: re-run keeps the round number and reaches the same decision again, and a
#: supplementary analysis re-presents the annotation gate -- so the revision is
#: part of the identity rather than a detail.
_SNAPSHOT_ID = re.compile(r"^r(\d{3,})-([a-z]+)-g(\d+)$")

MANIFEST_NAME = "manifest.json"
MANIFEST_SIG_NAME = "manifest.json.sig"
CONTINUATION_NAME = "continuation.json"
SESSION_STATE_NAME = "session.json"
SNAPSHOTS_DIRNAME = "snapshots"
UPLOADS_DIRNAME = "uploads"
STAGING_DIRNAME = ".staging"
BLOBS_DIRNAME = "blobs"
NOTEBOOKS_DIRNAME = "notebooks"
RUNTIME_DIRNAME = "runtime"
WORKSPACE_DIRNAME = "workspace"
FAILED_SUFFIX = ".failed.json"

#: Paths inside a bundle that the manifest declares, spelled once so the writer
#: and the reader cannot disagree about them.
RUNTIME_PKL_RELPATH = f"{RUNTIME_DIRNAME}/session.pkl"
RUNTIME_MANIFEST_RELPATH = f"{RUNTIME_DIRNAME}/session.runtime.json"
RUNTIME_MANIFEST_SIG_RELPATH = f"{RUNTIME_DIRNAME}/session.runtime.json.sig"
PAIR_STATE_RELPATH = f"{RUNTIME_DIRNAME}/pair/state.json"
PAIR_CHECKPOINT_RELPATH = f"{RUNTIME_DIRNAME}/pair/checkpoints.sqlite"
PAIR_EXECUTOR_RELPATH = f"{RUNTIME_DIRNAME}/pair/executor_checkpoint.json"


class SnapshotsDisabled(RuntimeError):
    """No snapshot root is configured, so no snapshot path exists.

    Raised rather than returning ``None`` because every caller has already
    asked ``snapshots_enabled()``; reaching here means a caller skipped that
    question, and a silent ``None`` would turn into a path built from it.
    """


def snapshot_storage_root() -> Path:
    """The configured snapshot root, or raise.

    Read through the accessor rather than imported by value: the config
    package re-exports its module with ``import *``, so a constant read at
    import is a copy nothing can change.
    """
    from backend.config import snapshot_dir

    root = snapshot_dir()
    if root is None:
        raise SnapshotsDisabled(
            "SCAI_SNAPSHOT_DIR is not set, so snapshots are disabled"
        )
    return root


def snapshot_id(*, round_number: int, gate: str, revision: int) -> str:
    """Assemble the identifier for one decision snapshot."""
    word = GATE_WORDS.get(str(gate))
    if word is None:
        raise ValueError(f"not a decision gate: {gate!r}")
    round_number = int(round_number)
    revision = int(revision)
    if round_number < 0 or revision < 0:
        raise ValueError(
            f"round and revision must not be negative: {round_number}, {revision}"
        )
    return f"r{round_number:03d}-{word}-g{revision}"


def parse_snapshot_id(value: str) -> tuple[int, str, int]:
    """Return ``(round, gate, revision)``, or raise ``ValueError``.

    Parsing is also the only validation a snapshot id gets before it becomes a
    path component, so it is deliberately total: anything this does not match
    never reaches the filesystem.
    """
    match = _SNAPSHOT_ID.match(str(value))
    if match is None:
        raise ValueError(f"not a snapshot id: {value!r}")
    round_text, word, revision_text = match.groups()
    gate = WORD_GATES.get(word)
    if gate is None:
        raise ValueError(f"not a decision gate word: {word!r}")
    return int(round_text), gate, int(revision_text)


def _validated_session_id(session_id: str) -> str:
    from backend.utils.files import validate_session_id

    validate_session_id(session_id)
    return str(session_id)


def session_root(session_id: str) -> Path:
    """Everything one source session's snapshots occupy."""
    return snapshot_storage_root() / _validated_session_id(session_id)


def snapshots_root(session_id: str) -> Path:
    return session_root(session_id) / SNAPSHOTS_DIRNAME


def uploads_root(session_id: str) -> Path:
    """The analysis inputs, stored once per session rather than per snapshot.

    They do not change after upload and they can be tens of gigabytes, so every
    snapshot of a session references the same copy and lists the digests it
    depends on in its own manifest.
    """
    return session_root(session_id) / UPLOADS_DIRNAME


def staging_root(session_id: str) -> Path:
    return session_root(session_id) / STAGING_DIRNAME


def snapshot_dir(session_id: str, snapshot: str) -> Path:
    parse_snapshot_id(snapshot)
    return snapshots_root(session_id) / snapshot


def manifest_path(session_id: str, snapshot: str) -> Path:
    return snapshot_dir(session_id, snapshot) / MANIFEST_NAME


def remove_snapshot(session_id: str, snapshot: str) -> None:
    """Give up one bundle, and the shared copy it was the last to need.

    Renamed aside before removal, so a caller never sees a directory being
    emptied under it. A dump shared with a neighbouring snapshot survives: the
    filesystem reference counts it, which is the whole reason sharing is a
    hardlink rather than a record.

    The inputs are the exception, because they are shared by a record rather
    than by the filesystem: they are stored once per session and named by every
    bundle's manifest, so once the last bundle is gone nothing refers to them.
    Leaving them behind kept the largest part of a session's snapshot storage
    alive with no restore point left to spend it on -- and a reclaim under
    space pressure would have walked away from a session it had just emptied
    while its inputs still occupied the volume.

    Two things are deliberately kept. Failed-capture records survive: they are
    small, and they are the only account of a decision that has no restore
    point. And nothing is reclaimed while a capture is staging, because that
    capture is about to publish a manifest naming these inputs.
    """
    import shutil

    bundle = snapshot_dir(session_id, snapshot)
    if bundle.is_dir():
        doomed = bundle.with_name(f".deleting-{bundle.name}")
        shutil.rmtree(doomed, ignore_errors=True)
        bundle.rename(doomed)
        shutil.rmtree(doomed, ignore_errors=True)

    snapshots = snapshots_root(session_id)
    if not snapshots.is_dir():
        return
    if any(
        entry.is_dir() and not entry.name.startswith(".")
        for entry in snapshots.iterdir()
    ):
        return
    staging = staging_root(session_id)
    if staging.is_dir() and any(staging.iterdir()):
        return

    shutil.rmtree(uploads_root(session_id), ignore_errors=True)
    if not any(snapshots.iterdir()):
        shutil.rmtree(session_root(session_id), ignore_errors=True)


def failed_record_path(session_id: str, snapshot: str) -> Path:
    """Where an un-snapshotted decision is recorded.

    Beside the bundles rather than inside one, and named after the id the
    bundle would have had, so the listing can show which decision has no
    restore point without inventing a directory that a restore might try to
    read.
    """
    parse_snapshot_id(snapshot)
    return snapshots_root(session_id) / f"{snapshot}{FAILED_SUFFIX}"
