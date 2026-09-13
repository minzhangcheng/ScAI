"""Restore one decision snapshot into a new session. Part B spec 4.6.

Restoring forks; it never rewinds. The source session is opened read-only, is
not locked, may be running, and is not modified -- what comes out is a new
session parked on the same decision, holding what the source held at that
instant.

This is an import from a local bundle, and it says so by using the import
path's primitives rather than copies of them: the two-phase lifecycle claim
that makes a half-built session invisible and reclaimable, the anchored path
remap that rewrites a data root without touching URLs or free text, and the
re-signing that makes a foreign dump this installation's own. The differences
from an archive import are small and deliberate: the manifest is this
installation's own and is verified rather than re-keyed on trust, the
continuation is a parked gate rather than a fresh start, and the reports and
logs are not carried at all -- a rolled-back session has not produced them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .layout import (
    BLOBS_DIRNAME,
    CONTINUATION_NAME,
    MANIFEST_NAME,
    NOTEBOOKS_DIRNAME,
    PAIR_CHECKPOINT_RELPATH,
    PAIR_EXECUTOR_RELPATH,
    PAIR_STATE_RELPATH,
    RUNTIME_MANIFEST_RELPATH,
    RUNTIME_PKL_RELPATH,
    SESSION_STATE_NAME,
    WORKSPACE_DIRNAME,
    SnapshotsDisabled,
    parse_snapshot_id,
    snapshot_dir,
    uploads_root,
)
from .manifest import (
    SnapshotManifestError,
    read_snapshot_manifest,
    validate_continuation,
)

logger = logging.getLogger(__name__)

_CHUNK = 4 * 1024 * 1024

#: Channels the capsule never carries. The analysis state is restored from
#: ``session.json`` and the message list is rebuilt by the graph.
_UNCARRIED_CHANNELS = frozenset({"analysis_state", "messages"})

#: Transient graph channels that have a reviewed, non-carried duplicate in the
#: canonical AnalysisState. ``round_result`` is the one such channel today:
#: AnalysisState.round_results is authoritative, and the post-action targets do
#: not need the transient copy. Any other non-JSON channel is a refusal rather
#: than a silent drop.
_KNOWN_REDUNDANT_GRAPH_CHANNELS = frozenset({"round_result"})

#: The complete set of report URL fields the workflow writes into persisted
#: messages or top-level API shapes. Kept in one place so a new producer has to
#: update the census test rather than silently bypassing restore sanitation.
_REPORT_URL_FIELDS = frozenset({
    "html_url",
    "report_html_url",
    "detailed_report_url",
    "exploration_report_url",
    "annotation_report_url",
    "integration_report_url",
})

#: Relative local report endpoints whose backing report/log was deliberately
#: not copied into the restored child. The session id segment is intentionally
#: not restricted to the source or child: whatever a snapshot happens to embed,
#: the new session cannot serve it until it generates that report itself.
#:
#: Only relative paths are local. An absolute URL with the same path shape
#: belongs to some external origin and must survive sanitization; production
#: emits these paths relatively, so accepting every host would delete a
#: legitimate citation.
_LOCAL_REPORT_URL = re.compile(
    r"^/api/session/[^/?#\s]+/"
    r"(?:report|exploration-report|annotation-report|integration-report|"
    r"detailed-report)(?:/|$|[?#])"
)


class SnapshotRestoreError(RuntimeError):
    """The bundle cannot be turned into a session, and nothing was left behind."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def _refuse(reason: str, detail: str) -> SnapshotRestoreError:
    return SnapshotRestoreError(reason, detail)


# --------------------------------------------------------------- materialising


def _verified_copy(source: Path, target: Path, expected: str, progress) -> int:
    """Copy one declared file, refusing bytes the manifest does not describe.

    Verified while copying rather than afterwards: a second pass over a
    multi-gigabyte dump doubles the read, and a file checked after it has been
    published is a file that was briefly usable while wrong.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with open(source, "rb") as reader, open(target, "wb") as writer:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != expected:
        raise _refuse(
            "bundle_corrupt",
            f"{source.name} does not match the digest its manifest declares",
        )
    if progress is not None:
        progress(size)
    return size


def _verify_in_place(source: Path, expected: str, progress) -> None:
    digest = hashlib.sha256()
    size = 0
    with open(source, "rb") as reader:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != expected:
        raise _refuse(
            "bundle_corrupt",
            f"{source.name} does not match the digest its manifest declares",
        )
    if progress is not None:
        progress(size)


def _destination(
    relpath: str, session_id: str, source_session_id: str = "",
) -> Path | None:
    """Where one declared bundle file lands in the new session, or nowhere.

    The graph checkpoint is read once and discarded: it is keyed by the source
    session's thread id and cannot be renamed into place, so a new session that
    installed it would resume a thread that does not exist here.
    """
    from backend.config import (
        get_session_dir,
        get_session_states,
    )

    states = get_session_states(session_id)
    if relpath == SESSION_STATE_NAME:
        return get_session_dir(session_id) / "session.json"
    if relpath == CONTINUATION_NAME:
        return None
    if relpath.startswith(f"{BLOBS_DIRNAME}/"):
        return get_session_dir(session_id) / relpath
    if relpath.startswith(f"{NOTEBOOKS_DIRNAME}/"):
        name = relpath.split("/", 1)[1]
        # The bundle stores the notebook under the name it had, which is the
        # source session's id. A session reads its history from the file named
        # after itself, so laying it down unrenamed gives the fork an empty
        # notebook and leaves its real history in a file naming a session that
        # does not exist here -- which then travels into the fork's own
        # exports. Anything else in the directory keeps the name it had.
        if source_session_id and name == f"{source_session_id}.ipynb":
            name = f"{session_id}.ipynb"
        return states / name
    if relpath.startswith(f"{WORKSPACE_DIRNAME}/"):
        return get_session_dir(session_id) / relpath
    if relpath == RUNTIME_PKL_RELPATH:
        return states / f"{session_id}_session.pkl"
    if relpath == PAIR_STATE_RELPATH:
        return states / f"{session_id}_state.json"
    if relpath == PAIR_EXECUTOR_RELPATH:
        return states / f"{session_id}_bioinfo_checkpoint.json"
    # The generation manifest is rebuilt and re-signed under this
    # installation's key rather than copied, and the checkpoint is read in
    # place. Both are handled outside this table.
    return None


def _materialise(bundle: Path, manifest: dict, session_id: str, progress) -> None:
    """Lay the bundle down inside the claimed session directory.

    Not a second staging area: the directory is invisible until the lifecycle
    record says ``active``, and ``cleanup_failed_import_generation`` removes it
    whole on failure, so staging it twice would buy nothing and cost a copy.
    """
    from backend.config import ensure_session_dirs

    ensure_session_dirs(session_id)
    for relpath, entry in sorted(manifest["files"].items()):
        source = bundle / relpath
        if not source.is_file():
            raise _refuse("bundle_incomplete", f"{relpath} is missing from the bundle")
        target = _destination(relpath, session_id, manifest["source_session_id"])
        if target is None:
            # Read in place rather than installed -- the graph checkpoint and
            # the generation manifest are both handled elsewhere -- but still
            # verified here, or a bundle that had lost one would reach its
            # reader unchecked and fail with a message about SQLite.
            _verify_in_place(source, entry["sha256"], progress)
            continue
        _verified_copy(source, target, entry["sha256"], progress)


def _materialise_uploads(source_session_id: str, manifest: dict, session_id: str, progress) -> None:
    """The inputs, from the copy every snapshot of the source session shares.

    A digest that no longer matches is refused rather than warned about: the
    inputs are what the restored analysis re-reads, and a session whose data
    silently changed under it is worse than one that would not open.
    """
    from backend.config import get_session_uploads

    shared = uploads_root(source_session_id)
    destination = get_session_uploads(session_id)
    for relpath, entry in sorted(manifest["uploads"].items()):
        source = shared / relpath
        if not source.is_file():
            raise _refuse("bundle_incomplete", f"input {relpath} is missing")
        _verified_copy(source, destination / relpath, entry["sha256"], progress)


# --------------------------------------------------------------- the state


def _strip_report_paths(payload: Any, roots: tuple[str, ...]) -> Any:
    """Remove references to reports and logs the restored session has not made.

    A rolled-back session has not produced the later report, and a path left
    dangling would render as a broken link rather than as an absence. Done on
    the payload rather than by naming fields, because the field list is exactly
    the sort of thing that grows without this code hearing about it.
    """
    if isinstance(payload, str):
        return None if any(root in payload for root in roots) else payload
    if isinstance(payload, list):
        return [
            item for item in (_strip_report_paths(value, roots) for value in payload)
            if item is not None
        ]
    if isinstance(payload, dict):
        return {key: _strip_report_paths(value, roots) for key, value in payload.items()}
    return payload


def _is_unavailable_local_report_url(value: Any) -> bool:
    """Whether a field value is a local report endpoint the child cannot serve."""
    return bool(
        isinstance(value, str)
        and value
        and _LOCAL_REPORT_URL.match(value.strip())
    )


def _strip_unavailable_report_urls(payload: Any) -> Any:
    """Null report URL fields that point at an uncopied local report.

    Applied only at the field boundary: the message, list, decision content,
    external citations, and arbitrary prose that merely mentions a session id
    all survive. A top-level report path is handled by ``_strip_report_paths``;
    this covers URLs embedded inside retained messages, which path removal
    cannot see.
    """
    if isinstance(payload, dict):
        return {
            key: (
                None
                if key in _REPORT_URL_FIELDS
                and _is_unavailable_local_report_url(value)
                else _strip_unavailable_report_urls(value)
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_strip_unavailable_report_urls(item) for item in payload]
    return payload


def _rebound_state(source_session_id: str, session_id: str, user_id: str):
    """The source's state, rewritten to belong to the new session."""
    from backend.config import (
        DATA_BASE_DIR,
        get_session_dir,
        get_session_logs,
        get_session_reports,
    )
    from backend.models.schemas import AnalysisState
    from backend.routes.export_import_parts.import_archive_policy import (
        _rebind_session_id_values,
        _remap_paths_in_text,
    )
    from backend.utils.blob_store import inline_blobs

    payload = json.loads(
        (get_session_dir(session_id) / "session.json").read_text(encoding="utf-8")
    )
    payload = inline_blobs(payload, get_session_dir(session_id) / BLOBS_DIRNAME)
    text = json.dumps(payload, ensure_ascii=False)
    text = _remap_paths_in_text(
        text, str(DATA_BASE_DIR), str(DATA_BASE_DIR), source_session_id, session_id,
    )
    payload = _rebind_session_id_values(
        json.loads(text), source_session_id, session_id,
    )
    payload = _strip_report_paths(
        payload,
        (str(get_session_reports(session_id)), str(get_session_logs(session_id))),
    )
    # After path remapping and identity rebinding (so top-level source ids have
    # already become the child's) and before validation: no source/child report
    # endpoint may survive inside a retained message.
    payload = _strip_unavailable_report_urls(payload)
    try:
        state = AnalysisState.model_validate(payload)
    except Exception as exc:
        raise _refuse("state_unusable", f"the captured session state is unreadable: {exc}") from exc
    state.id = session_id
    state.user_id = user_id
    state.created_at = datetime.now(timezone.utc)
    return state


def restored_session_name(source_name: str, round_number: int, gate: str, *, ambiguous: bool) -> str:
    """What the new session is called. Decided by the project owner.

    ``rollback <source> round N``, with the decision appended only where a
    round holds more than one -- round 1 has both the annotation and the
    direction gate, and two sessions named the same thing would be worse than a
    longer name.
    """
    from .layout import GATE_WORDS

    base = f"rollback {source_name or 'session'} round {int(round_number)}"
    return f"{base} {GATE_WORDS[gate]}" if ambiguous else base


# --------------------------------------------------------------- continuation


def _jsonable(value: Any) -> Any | None:
    """The value if it survives a JSON round trip, else None.

    The capsule is stored on ``AnalysisState`` and written into
    ``session.json``, so a channel holding a live model object could not be
    persisted at all. The analysis state is the authority either way -- it is
    restored from the captured ``session.json`` -- and these channels are
    routing context, so dropping one costs a hint rather than a fact.
    """
    try:
        # Python's default encoder accepts NaN/Infinity even though they are
        # not valid JSON and would be rejected by strict consumers later.
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return value


def _continuation_channels(
    checkpoint: Path, source_session_id: str, capsule: dict,
) -> dict[str, Any]:
    """The graph channels the source run held, read from its paired checkpoint.

    Read here and then discarded, rather than installed: the database is keyed
    by the source's thread id (spec 3.6) and cannot be renamed into place, so a
    new session that kept it would resume a thread that does not exist.
    """
    from backend.utils.checkpoint import DurableJsonPlusSerializer

    if not checkpoint.is_file():
        raise _refuse("continuation_unreadable", "the paired graph checkpoint is missing")
    try:
        with closing(
            sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            row = connection.execute(
                "SELECT type, checkpoint FROM checkpoints WHERE thread_id=? "
                "AND checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1",
                (source_session_id,),
            ).fetchone()
        if row is None:
            raise ValueError("the paired checkpoint holds no root position")
        serde = DurableJsonPlusSerializer(allowed_json_modules=[
            ("backend.models.schemas", name)
            for name in ("AgentType", "AnalysisState", "AnalysisPhase",
                         "RoundResult", "TaskStatus", "MessageRole")
        ])
        values = serde.loads_typed(row)["channel_values"]
    except Exception as exc:
        raise _refuse(
            "continuation_unreadable", f"the paired graph checkpoint is unusable: {exc}",
        ) from exc

    held = values.get("analysis_state")
    if getattr(held, "id", None) not in (None, source_session_id):
        raise _refuse(
            "continuation_unreadable",
            "the paired checkpoint belongs to a different session",
        )

    carried: dict[str, Any] = {}
    intentionally_skipped: set[str] = set()
    redundant: list[str] = []
    for key, value in values.items():
        name = str(key)
        if name in _UNCARRIED_CHANNELS or name.startswith("_"):
            # The analysis state is restored from session.json, messages are
            # rebuilt by the graph, and underscore channels are one-shot
            # routing internals that must never enter a persisted capsule.
            intentionally_skipped.add(name)
            continue
        if name in _KNOWN_REDUNDANT_GRAPH_CHANNELS:
            # Reviewed transient duplicate: the canonical value lives in
            # AnalysisState (round_results), and the post-action target does
            # not need the live model object that cannot survive JSON.
            intentionally_skipped.add(name)
            if value is not None:
                redundant.append(name)
            continue
        safe = _jsonable(value)
        if safe is None and value is not None:
            # A future channel holding a live object is not a hint to drop.
            # Refuse before the child is published so the lifecycle tombstone
            # path leaves nothing behind.
            raise _refuse(
                "continuation_unreadable",
                f"graph channel {name!r} holds a value that cannot be "
                "persisted; restore cannot continue",
            )
        carried[name] = safe
    if redundant:
        logger.info(
            "Snapshot restore omitted known redundant graph channel(s) that "
            "cannot be persisted: %s", ", ".join(sorted(redundant)),
        )
    missing = sorted(
        set(capsule["workflow_state_keys"]) - set(carried) - intentionally_skipped
    )
    if missing:
        raise _refuse(
            "continuation_unreadable",
            "the paired checkpoint has lost channels the capture recorded: "
            + ", ".join(missing),
        )
    return carried


def _arm(state, capsule: dict, source_session_id: str, snapshot: str, channels: dict) -> None:
    """Park the new session on the decision the snapshot precedes.

    Whatever the source's skip time was, a restored session waits: there is no
    workflow running a timeout here, and the person who restored it is the
    reason it exists.
    """
    from backend.config import get_session_workspace
    from backend.models.schemas import TaskStatus
    from backend.services.report.lineage import clone_publication_figures_for_branch
    from backend.workflow.direction_coverage import clone_round_results_for_branch
    from backend.workflow.restore_targets import (
        stable_restore_branch_id,
        stable_restore_branch_label,
    )

    state.restore_resume_kind = "semantic_gate"
    state.restore_resume_gate = capsule["gate"]
    state.restore_resume_node = None
    state.restore_resume_workflow_state = channels
    state.restore_start_intent = None
    source_lineage_id = state.restore_branch_id or "main"
    if state.restore_branch_id and not any(
        (getattr(record, "lineage_id", None) or "main") == source_lineage_id
        for records in (
            state.round_results,
            state.publication_figures,
            state.publication_tables,
            state.code_history,
        )
        for record in (records or [])
    ):
        source_lineage_id = (
            state._legacy_restore_source_lineage() or source_lineage_id
        )
    state.restore_source_lineage_id = source_lineage_id
    branch_seed = f"{source_session_id[:8]}-{snapshot}"
    branch_id = stable_restore_branch_id(branch_seed, None)
    branch_label = stable_restore_branch_label(branch_seed, None)
    origin = f"{source_session_id}/{snapshot}"
    state.round_results = clone_round_results_for_branch(
        state.round_results,
        branch_id=branch_id,
        branch_label=branch_label,
        source_snapshot_id=origin,
        source_lineage_id=source_lineage_id,
    )
    state.publication_figures = clone_publication_figures_for_branch(
        state.publication_figures,
        branch_id=branch_id,
        source_lineage_id=source_lineage_id,
        workspace_path=get_session_workspace(state.id),
    )
    state.restore_branch_id = branch_id
    state.restore_branch_label = branch_label
    state.last_restored_snapshot_id = origin
    state.restore_provenance_snapshot_id = origin
    state.status = TaskStatus(capsule["restored_status"])
    state.wait_input_reason = None
    state.waiting_input_started_at = datetime.now(timezone.utc)
    state.runtime_restore_status = "complete"
    state.runtime_restore_warning = None
    state.runtime_recompute_pending = False


# --------------------------------------------------------------- the runtime


def _rekey_runtime(state, bundle: Path, manifest: dict, session_id: str) -> None:
    """Make the restored generation this installation's and this session's.

    The dump arrives verified against a manifest this installation signed, so
    unlike an archive import there is no question of trusting a foreign key.
    What still has to change is identity: the HMAC is keyed to the file, the
    generation manifest names a state revision, and the notebook record names a
    session. The pairing's checkpoint is cleared because the graph position is
    deliberately not carried, which is the same thing ``validate_imported_pairing``
    does for a reallocated import.
    """
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )
    from backend.config import get_session_states
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    states = get_session_states(session_id)
    pkl = states / f"{session_id}_session.pkl"
    _write_pkl_signature(pkl, _integrity_dir(session_id))

    committed = read_snapshot_runtime_manifest(bundle)
    write_runtime_manifest(
        runtime_manifest_path(session_id, states_dir=states),
        build_runtime_manifest(
            revision=committed["revision"],
            state_revision=int(getattr(state, "state_version", 0) or 0),
            pkl_digest=manifest["pkl_sha256"],
            saved_names=committed["saved_names"],
            skipped_names=committed["skipped_names"],
            serializer_version=committed["serializer_version"],
            pairing={
                "state_version": int(getattr(state, "state_version", 0) or 0),
                "checkpoint": None,
            },
            carried_definitions=committed.get("carried_definitions"),
        ),
    )

    record_path = states / f"{session_id}_state.json"
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _refuse(
                "runtime_unverified", f"the captured notebook record is unreadable: {exc}",
            ) from exc
        record["session_id"] = session_id
        record["lifecycle_generation"] = int(getattr(state, "lifecycle_generation", 0) or 0)
        persistence = record.get("runtime_persistence")
        if isinstance(persistence, dict):
            persistence["lifecycle_generation"] = record["lifecycle_generation"]
            persistence["pair_restore"] = None
            pairing = persistence.get("committed_pairing")
            if isinstance(pairing, dict):
                pairing["checkpoint"] = None
                pairing["state_version"] = int(getattr(state, "state_version", 0) or 0)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    executor = states / f"{session_id}_bioinfo_checkpoint.json"
    if executor.is_file():
        try:
            payload = json.loads(executor.read_text(encoding="utf-8"))
            payload["session_id"] = session_id
            executor.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise _refuse(
                "runtime_unverified",
                f"the captured executor position is unreadable: {exc}",
            ) from exc

    from backend.services.session_store import refresh_runtime_persistence_projection

    refresh_runtime_persistence_projection(session_id, state)


def read_snapshot_runtime_manifest(bundle: Path) -> dict:
    """The generation manifest the source save signed, read out of the bundle.

    Verified against the *source* installation's key, which is this one -- a
    snapshot never travels, so unlike an archive there is no foreign signature
    to re-key on trust.
    """
    from backend.utils.runtime_manifest import RuntimeManifestError, read_runtime_manifest

    try:
        return read_runtime_manifest(bundle / RUNTIME_MANIFEST_RELPATH)
    except RuntimeManifestError as exc:
        raise _refuse(
            "runtime_unverified", f"the captured generation manifest is unusable: {exc}",
        ) from exc


def _verify_restored_runtime(state, session_id: str, capsule: dict) -> None:
    """Refuse to publish a session whose runtime would not come back.

    Three checks, all of them things a later start would discover the hard way:
    the dump loads against its own manifest, the record is coherent with the
    state, and -- for the annotation decision -- the executor position belongs
    to the generation beside it.
    """
    from backend.config import get_session_states
    from backend.constants import GATE_ANNOTATION_CONFIRMATION
    from backend.utils.runtime_apply import state_runtime_coherence
    from backend.utils.runtime_manifest import runtime_manifest_path
    from backend.utils.sandbox_parts.runtime_restore import RuntimeIntegrityError, plan_restore

    states = get_session_states(session_id)
    try:
        plan_restore(
            runtime_manifest_path(session_id, states_dir=states),
            states / f"{session_id}_session.pkl",
        )
    except RuntimeIntegrityError as exc:
        raise _refuse("runtime_unverified", str(exc)) from exc

    coherence = state_runtime_coherence(state)
    if coherence != "current":
        raise _refuse(
            "runtime_unverified",
            f"the restored runtime reads {coherence!r} against its own state",
        )

    executor = states / f"{session_id}_bioinfo_checkpoint.json"
    if capsule["gate"] == GATE_ANNOTATION_CONFIRMATION and executor.is_file():
        from backend.agents.bioinformatics.executor_checkpoint import validate_checkpoint

        payload = json.loads(executor.read_text(encoding="utf-8"))
        persistence = dict(getattr(state, "runtime_persistence", None) or {})
        errors = validate_checkpoint(
            payload,
            current_round=int(getattr(state, "current_round", 0) or 0),
            planning_stage=str(payload.get("planning_stage") or ""),
            committed_revision=persistence.get("committed_revision"),
        )
        if errors:
            raise _refuse(
                "runtime_unverified",
                "the captured executor position does not belong to this "
                "generation: " + "; ".join(errors),
            )


# --------------------------------------------------------------- the job


async def restore_snapshot_into_new_session(
    *,
    source_session_id: str,
    snapshot: str,
    user_id: str,
    progress: Callable[[int], None] | None = None,
) -> str:
    """Create a new session from one decision snapshot. Returns its id.

    On any failure before publication the claimed session is tombstoned and
    removed, so a caller sees either a session or nothing -- never a directory
    that half exists.
    """
    import asyncio
    import uuid

    from backend.config import get_session_dir
    from backend.services.session_lifecycle import (
        bind_import_generation,
        claim_import_generation,
        commit_import_generation,
    )
    from backend.services.session_store import async_save_session, update_session_cache

    try:
        parse_snapshot_id(snapshot)
        bundle = snapshot_dir(source_session_id, snapshot)
    except (ValueError, SnapshotsDisabled) as exc:
        raise _refuse("unknown_snapshot", str(exc)) from exc
    # Verified before anything else in the bundle is opened, and before any
    # dill frame is loaded: per-file digests prove only what the manifest
    # claims, and re-signing anything first would authenticate a substitution.
    try:
        manifest = read_snapshot_manifest(bundle / MANIFEST_NAME)
    except SnapshotManifestError as exc:
        # The detail travels to the client, so it says what is wrong rather
        # than where: the underlying message names the manifest's absolute
        # path, which describes the server's storage layout to a caller who
        # can do nothing with it.
        logger.warning("Snapshot %s/%s failed verification: %s", source_session_id, snapshot, exc)
        raise _refuse(
            "snapshot_unverified",
            "The snapshot's manifest is missing, unsigned, or does not verify.",
        ) from exc
    if manifest["source_session_id"] != source_session_id:
        raise _refuse(
            "snapshot_unverified", "the bundle belongs to a different session",
        )
    capsule = _read_capsule(bundle, manifest)

    session_id = str(uuid.uuid4())
    while claim_import_generation(session_id, user_id, get_session_dir(session_id)) is None:
        session_id = str(uuid.uuid4())  # pragma: no cover - uuid4 collision

    try:
        await asyncio.to_thread(_materialise, bundle, manifest, session_id, progress)
        await asyncio.to_thread(
            _materialise_uploads, source_session_id, manifest, session_id, progress,
        )
        state = _rebound_state(source_session_id, session_id, user_id)
        # ``scai_version`` means the application version that created *this*
        # session. The signed snapshot version is provenance only and must not
        # select compatibility behavior; reports render it separately.
        from backend.constants import VERSION

        state.scai_version = VERSION
        state.restore_source_scai_version = manifest["scai_version"]
        state.name = restored_session_name(
            manifest["source_session_name"], manifest["round"], manifest["gate"],
            ambiguous=_round_holds_more_than_one_decision(source_session_id, manifest),
        )
        channels = _continuation_channels(
            bundle / PAIR_CHECKPOINT_RELPATH, source_session_id, capsule,
        )
        _arm(state, capsule, source_session_id, snapshot, channels)
        bind_import_generation(state)
        _rekey_runtime(state, bundle, manifest, session_id)
        _verify_restored_runtime(state, session_id, capsule)

        await async_save_session(session_id, state, force=True)
        commit_import_generation(state)
        await update_session_cache(session_id, state)
    except BaseException:
        _discard(session_id)
        raise
    logger.info(
        "Restored %s/%s into a new session %s", source_session_id, snapshot, session_id,
    )
    return session_id


def _read_capsule(bundle: Path, manifest: dict) -> dict:
    payload = bundle / CONTINUATION_NAME
    entry = manifest["files"].get(CONTINUATION_NAME)
    try:
        raw = payload.read_bytes()
    except OSError as exc:
        raise _refuse("bundle_incomplete", f"the continuation is unreadable: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise _refuse("bundle_corrupt", "the continuation does not match its digest")
    try:
        return validate_continuation(json.loads(raw.decode("utf-8")))
    except (ValueError, SnapshotManifestError) as exc:
        raise _refuse("bundle_corrupt", f"the continuation is unusable: {exc}") from exc


def _round_holds_more_than_one_decision(source_session_id: str, manifest: dict) -> bool:
    """Whether the source's round has snapshots at more than one gate.

    Round 1 normally does -- the annotation confirmation and then the direction
    selection -- and two restored sessions with the same name would be worse
    than a longer one.
    """
    from .listing import read_session_listing

    try:
        rows = read_session_listing(source_session_id)["snapshots"]
    except SnapshotsDisabled:
        return False
    gates = {
        row["gate"] for row in rows if int(row["round"]) == int(manifest["round"])
    }
    return len(gates) > 1


def _discard(session_id: str) -> None:
    """Remove a session that never reached publication.

    Uses the failed-import cleanup rather than an ``rmtree`` of its own: that
    path takes the cross-process lease, writes the tombstone before removing
    anything, and clears the integrity directory the re-signing created.
    """
    from backend.services.session_deletion import cleanup_failed_import_generation

    try:
        cleanup_failed_import_generation(session_id)
    except Exception:
        # Logged and left: the tombstone sweeper reclaims it on its next pass,
        # and raising here would replace the failure the caller came for.
        logger.exception("Could not discard the partial restore %s", session_id)
