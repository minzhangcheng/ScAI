"""Boundary-based, immutable AnnData checkpoint capture and replay."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.models.schema_parts.adata_checkpoints import (
    AdataCheckpointRecord,
    canonical_id_set_digest,
    normalize_id_set,
)

from .errors import AdataCaptureError
from .layout import blob_relative_path
from .storage import publish_or_reuse

logger = logging.getLogger(__name__)

_EXECUTION_SAMPLE_LIMIT = 200


@dataclass
class CaptureOutcome:
    record: Optional[AdataCheckpointRecord] = None
    published: bool = False
    reused: bool = False
    disclosure: Optional[str] = None
    error_summary: Optional[str] = None
    registry_name: Optional[str] = None
    anchor_name: Optional[str] = None
    namespace_changed: bool = False

    @property
    def available(self) -> bool:
        return self.record is not None and self.record.status == "available"


def _clip(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit]


def _session_lineage(state: Any) -> str:
    return str(getattr(state, "restore_branch_id", None) or "main")


def _event_identity_payload(
    *,
    boundary: str,
    session_lineage_id: str,
    round_number: int,
    planning_stage: str,
    phase: str,
    source_preparation_ids: list[str],
    active_source_preparation_id: Optional[str],
    source_scope_id_at_capture: Optional[str],
    executor_settlement_scope_id_at_capture: Optional[str],
    supplementary_request_id: Optional[str],
    round_result_identity_key: Optional[str],
    direction: str,
    execution_ids: list[str],
    attempt_execution_ids: list[str],
    outcome: str,
    content_hash: str = "",
) -> str:
    payload = {
        "boundary": str(boundary),
        "session_lineage_id": str(session_lineage_id),
        "round_number": int(round_number),
        "planning_stage": str(planning_stage),
        "phase": str(phase),
        "source_preparation_ids_digest": canonical_id_set_digest(source_preparation_ids),
        "source_preparation_ids_count": len(source_preparation_ids),
        "active_source_preparation_id": active_source_preparation_id or "",
        "source_scope_id_at_capture": source_scope_id_at_capture or "",
        "executor_settlement_scope_id_at_capture": (
            executor_settlement_scope_id_at_capture or ""
        ),
        "supplementary_request_id": supplementary_request_id or "",
        "round_result_identity_key": round_result_identity_key or "",
        "direction": str(direction or ""),
        "execution_ids_digest": canonical_id_set_digest(execution_ids),
        "execution_ids_count": len(execution_ids),
        "attempt_execution_ids_digest": canonical_id_set_digest(attempt_execution_ids),
        "attempt_execution_ids_count": len(attempt_execution_ids),
        "outcome": str(outcome or ""),
        "content_hash": str(content_hash or ""),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_event(state: Any, boundary_event_id: str) -> Optional[AdataCheckpointRecord]:
    for record in getattr(state, "adata_checkpoints", None) or []:
        if str(getattr(record, "boundary_event_id", "")) == boundary_event_id:
            return record
    return None


def _next_revision(
    state: Any,
    *,
    boundary: str,
    session_lineage_id: str,
    round_number: int,
    round_result_identity_key: Optional[str],
) -> int:
    highest = 0
    for record in getattr(state, "adata_checkpoints", None) or []:
        if str(getattr(record, "boundary", "")) != boundary:
            continue
        if str(getattr(record, "session_lineage_id", "main")) != session_lineage_id:
            continue
        if int(getattr(record, "round_number", 0) or 0) != int(round_number):
            continue
        if (
            round_result_identity_key
            and str(getattr(record, "round_result_identity_key", "") or "")
            not in ("", str(round_result_identity_key))
        ):
            continue
        highest = max(highest, int(getattr(record, "boundary_revision", 0) or 0))
    return highest + 1


def _round_head(state: Any, *, session_lineage_id: str, round_number: int) -> Optional[str]:
    latest = None
    latest_seq = -1
    seq = int(getattr(state, "adata_checkpoint_sequence", 0) or 0)
    for index, record in enumerate(getattr(state, "adata_checkpoints", None) or []):
        if str(getattr(record, "session_lineage_id", "main")) != session_lineage_id:
            continue
        if int(getattr(record, "round_number", 0) or 0) != int(round_number):
            continue
        # Records are appended in sequence; later list entries supersede earlier
        # ones without relying on a separate sequence field.
        if index >= latest_seq:
            latest_seq = index
            latest = str(getattr(record, "checkpoint_id", "") or "") or None
    return latest


def _append_record(state: Any, record: AdataCheckpointRecord) -> None:
    records = getattr(state, "adata_checkpoints", None)
    if records is None:
        records = []
        state.adata_checkpoints = records
    records.append(record)


def _registry_names(state: Any, sandbox: Any) -> tuple[str, str, bool]:
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    registry_name = str(
        sandbox_state.get("adata_binding_registry_name")
        or getattr(sandbox, "adata_binding_registry_name", "")
        or ""
    ).strip()
    generated_registry = False
    if not registry_name:
        registry_name = f"scai_adata_registry_{uuid.uuid4().hex[:12]}"
        generated_registry = True
    anchor_name = f"scai_adata_anchor_{uuid.uuid4().hex[:12]}"
    return registry_name, anchor_name, generated_registry


async def capture_boundary_locked(
    sandbox: Any,
    state: Any,
    *,
    boundary: str,
    event_scope: str = "",
    round_number: int = 0,
    planning_stage: str = "full",
    phase: str = "",
    source_plan_id: Optional[str] = None,
    round_result: Any = None,
    direction: str = "",
    research_label: str = "",
    summary_at_capture: str = "",
    outcome_at_capture: str = "",
    source_preparation_ids_at_capture: Iterable[str] = (),
    active_source_preparation_id: Optional[str] = None,
    source_scope_id_at_capture: Optional[str] = None,
    executor_settlement_scope_id_at_capture: Optional[str] = None,
    supplementary_request_id: Optional[str] = None,
    execution_ids_at_capture: Iterable[str] = (),
    attempt_execution_ids_at_capture: Iterable[str] = (),
    recapture_nonce: Optional[str] = None,
    now: Optional[datetime] = None,
    content_hash: str = "",
) -> CaptureOutcome:
    """Capture one boundary while the caller already holds ``_exec_lock``."""
    if sandbox is None or getattr(sandbox, "km", None) is None:
        return CaptureOutcome(disclosure="no_live_kernel")

    source_prep_ids = normalize_id_set(source_preparation_ids_at_capture)
    execution_ids = normalize_id_set(execution_ids_at_capture)
    attempt_ids = normalize_id_set(attempt_execution_ids_at_capture)
    if recapture_nonce:
        event_scope = f"{event_scope}|recapture:{recapture_nonce}"

    boundary_event_id = _event_identity_payload(
        boundary=boundary,
        session_lineage_id=_session_lineage(state),
        round_number=int(round_number or 0),
        planning_stage=str(planning_stage or "full"),
        phase=str(phase or ""),
        source_preparation_ids=source_prep_ids,
        active_source_preparation_id=active_source_preparation_id,
        source_scope_id_at_capture=source_scope_id_at_capture,
        executor_settlement_scope_id_at_capture=executor_settlement_scope_id_at_capture,
        supplementary_request_id=supplementary_request_id,
        round_result_identity_key=getattr(round_result, "result_identity_key", None),
        direction=direction,
        execution_ids=execution_ids,
        attempt_execution_ids=attempt_ids,
        outcome=outcome_at_capture,
        content_hash=content_hash or event_scope,
    )
    existing = _find_event(state, boundary_event_id)
    if existing is not None:
        return CaptureOutcome(
            record=existing,
            published=False,
            reused=existing.status == "available",
            disclosure=None if existing.status != "failed" else "failed_capture_replayed",
            error_summary=existing.error_summary,
        )

    registry_state_name = str(
        (dict(getattr(state, "sandbox_state", None) or {})).get(
            "adata_binding_registry_name"
        )
        or ""
    ).strip()
    registry_facts = await sandbox.query_adata_registry_locked(
        registry_name=registry_state_name or None
    )
    canonical = dict(registry_facts.get("canonical") or {})
    if not canonical.get("adata_present"):
        return CaptureOutcome(disclosure="no_canonical_adata")
    existing_anchor_name = str(canonical.get("anchor_name") or "").strip()
    _remember = getattr(sandbox, "remember_host_managed_names", None)
    if callable(_remember):
        _remember([registry_state_name, existing_anchor_name])

    candidate = sandbox._checkpoint_candidate_path()
    serialized: dict[str, Any] = {}
    try:
        serialized = await sandbox.serialize_adata_candidate_locked(candidate)
        if not serialized.get("ok"):
            reason = str(serialized.get("reason") or "serialization_failed")
            record = AdataCheckpointRecord(
                status="failed",
                boundary=boundary,
                boundary_event_id=boundary_event_id,
                round_number=int(round_number or 0),
                planning_stage=planning_stage,
                boundary_revision=_next_revision(
                    state,
                    boundary=boundary,
                    session_lineage_id=_session_lineage(state),
                    round_number=int(round_number or 0),
                    round_result_identity_key=getattr(
                        round_result, "result_identity_key", None
                    ),
                ),
                phase=str(phase or ""),
                session_lineage_id=_session_lineage(state),
                adata_lineage_id=getattr(state, "active_adata_lineage_id", None),
                source_plan_id=source_plan_id,
                active_source_preparation_id=active_source_preparation_id,
                source_preparation_ids_at_capture=source_prep_ids,
                source_preparation_id_count_at_capture=len(source_prep_ids),
                source_preparation_id_set_digest_at_capture=canonical_id_set_digest(
                    source_prep_ids
                ),
                source_scope_id_at_capture=source_scope_id_at_capture,
                executor_settlement_scope_id_at_capture=(
                    executor_settlement_scope_id_at_capture
                ),
                supplementary_request_id=supplementary_request_id,
                round_result_identity_key=getattr(
                    round_result, "result_identity_key", None
                ),
                direction=str(direction or ""),
                research_label=_clip(research_label, 240),
                summary_at_capture=_clip(summary_at_capture, 600),
                execution_ids_at_capture=execution_ids[:_EXECUTION_SAMPLE_LIMIT],
                attempt_execution_ids_at_capture=attempt_ids[:_EXECUTION_SAMPLE_LIMIT],
                execution_id_count_at_capture=len(execution_ids),
                execution_id_set_digest_at_capture=canonical_id_set_digest(
                    execution_ids
                ),
                attempt_execution_id_count_at_capture=len(attempt_ids),
                attempt_execution_id_set_digest_at_capture=canonical_id_set_digest(
                    attempt_ids
                ),
                outcome_at_capture=_clip(outcome_at_capture, 200),
                error_summary=f"serialization/unsupported capture: {reason}"[:1000],
                created_at=now or datetime.now(),
            )
            _append_record(state, record)
            return CaptureOutcome(
                record=record,
                disclosure="failed_capture",
                error_summary=record.error_summary,
            )

        try:
            _budget = getattr(
                sandbox, "adata_checkpoint_candidate_byte_budget", None
            )
            from backend.utils.asyncio_tasks import run_in_owned_bulk_thread

            storage = await run_in_owned_bulk_thread(
                publish_or_reuse,
                str(state.id),
                candidate,
                byte_budget=_budget,
                allowed_root=Path(getattr(sandbox, "_session_tmp")),
            )
        except Exception as exc:  # storage converts its own errors; keep capture bounded
            record = AdataCheckpointRecord(
                status="failed",
                boundary=boundary,
                boundary_event_id=boundary_event_id,
                round_number=int(round_number or 0),
                planning_stage=planning_stage,
                boundary_revision=_next_revision(
                    state,
                    boundary=boundary,
                    session_lineage_id=_session_lineage(state),
                    round_number=int(round_number or 0),
                    round_result_identity_key=getattr(
                        round_result, "result_identity_key", None
                    ),
                ),
                phase=str(phase or ""),
                session_lineage_id=_session_lineage(state),
                adata_lineage_id=getattr(state, "active_adata_lineage_id", None),
                source_plan_id=source_plan_id,
                active_source_preparation_id=active_source_preparation_id,
                source_preparation_ids_at_capture=source_prep_ids,
                source_preparation_id_count_at_capture=len(source_prep_ids),
                source_preparation_id_set_digest_at_capture=canonical_id_set_digest(
                    source_prep_ids
                ),
                source_scope_id_at_capture=source_scope_id_at_capture,
                executor_settlement_scope_id_at_capture=(
                    executor_settlement_scope_id_at_capture
                ),
                supplementary_request_id=supplementary_request_id,
                round_result_identity_key=getattr(
                    round_result, "result_identity_key", None
                ),
                direction=str(direction or ""),
                research_label=_clip(research_label, 240),
                summary_at_capture=_clip(summary_at_capture, 600),
                execution_ids_at_capture=execution_ids[:_EXECUTION_SAMPLE_LIMIT],
                attempt_execution_ids_at_capture=attempt_ids[:_EXECUTION_SAMPLE_LIMIT],
                execution_id_count_at_capture=len(execution_ids),
                execution_id_set_digest_at_capture=canonical_id_set_digest(execution_ids),
                attempt_execution_id_count_at_capture=len(attempt_ids),
                attempt_execution_id_set_digest_at_capture=canonical_id_set_digest(
                    attempt_ids
                ),
                outcome_at_capture=_clip(outcome_at_capture, 200),
                error_summary=f"blob publication failed: {type(exc).__name__}"[:1000],
                created_at=now or datetime.now(),
            )
            _append_record(state, record)
            return CaptureOutcome(
                record=record,
                disclosure="failed_capture",
                error_summary=record.error_summary,
            )

        same_as_parent = False
        parent_checkpoint_id: Optional[str] = None
        adata_lineage_id = getattr(state, "active_adata_lineage_id", None)
        exact_canonical = bool(
            canonical.get("same_as_anchor")
            and canonical.get("binding_identity_status") == "exact"
        )
        if exact_canonical and adata_lineage_id:
            parent_checkpoint_id = getattr(state, "active_adata_checkpoint_id", None)
            parent = next(
                (
                    r for r in (getattr(state, "adata_checkpoints", None) or [])
                    if r.checkpoint_id == parent_checkpoint_id
                ),
                None,
            )
            if parent is not None and parent.blob_sha256 == storage.identity.sha256:
                same_as_parent = True
        else:
            adata_lineage_id = uuid.uuid4().hex
            parent_checkpoint_id = None

        record = AdataCheckpointRecord(
            status="available",
            boundary=boundary,
            boundary_event_id=boundary_event_id,
            round_number=int(round_number or 0),
            planning_stage=planning_stage,
            boundary_revision=_next_revision(
                state,
                boundary=boundary,
                session_lineage_id=_session_lineage(state),
                round_number=int(round_number or 0),
                round_result_identity_key=getattr(
                    round_result, "result_identity_key", None
                ),
            ),
            phase=str(phase or ""),
            session_lineage_id=_session_lineage(state),
            adata_lineage_id=str(adata_lineage_id),
            parent_checkpoint_id=parent_checkpoint_id,
            supersedes_checkpoint_id=_round_head(
                state,
                session_lineage_id=_session_lineage(state),
                round_number=int(round_number or 0),
            ),
            source_plan_id=source_plan_id,
            active_source_preparation_id=active_source_preparation_id,
            source_preparation_ids_at_capture=source_prep_ids,
            source_preparation_id_count_at_capture=len(source_prep_ids),
            source_preparation_id_set_digest_at_capture=canonical_id_set_digest(
                source_prep_ids
            ),
            source_scope_id_at_capture=source_scope_id_at_capture,
            executor_settlement_scope_id_at_capture=(
                executor_settlement_scope_id_at_capture
            ),
            supplementary_request_id=supplementary_request_id,
            round_result_identity_key=getattr(round_result, "result_identity_key", None),
            direction=str(direction or ""),
            research_label=_clip(research_label, 240),
            summary_at_capture=_clip(summary_at_capture, 600),
            execution_ids_at_capture=execution_ids[:_EXECUTION_SAMPLE_LIMIT],
            attempt_execution_ids_at_capture=attempt_ids[:_EXECUTION_SAMPLE_LIMIT],
            execution_id_count_at_capture=len(execution_ids),
            execution_id_set_digest_at_capture=canonical_id_set_digest(execution_ids),
            attempt_execution_id_count_at_capture=len(attempt_ids),
            attempt_execution_id_set_digest_at_capture=canonical_id_set_digest(
                attempt_ids
            ),
            outcome_at_capture=_clip(outcome_at_capture, 200),
            captured_runtime_instance_id=getattr(sandbox, "runtime_instance_id", None),
            captured_runtime_revision=int(
                getattr(sandbox, "runtime_revision", 0) or 0
            ),
            blob_sha256=storage.identity.sha256,
            blob_relative_path=blob_relative_path(storage.identity.sha256),
            size_bytes=storage.identity.size,
            n_obs=int(serialized.get("n_obs") or 0),
            n_vars=int(serialized.get("n_vars") or 0),
            same_as_parent=bool(same_as_parent),
            created_at=now or datetime.now(),
        )
        registry_name, anchor_name, generated_registry = _registry_names(state, sandbox)
        if registry_state_name and registry_name != registry_state_name:
            # A persisted configured name must not be replaced by a generated
            # one; generation only happens when no durable name exists.
            registry_name = registry_state_name
        if exact_canonical and existing_anchor_name:
            anchor_name = existing_anchor_name
        try:
            registration = await sandbox.register_canonical_anchor_locked(
                registry_name=registry_name,
                anchor_name=anchor_name,
                adata_lineage_id=str(record.adata_lineage_id),
                ancestor_checkpoint_id=record.checkpoint_id,
            )
        except Exception as exc:
            registration = {
                "ok": False,
                "reason": f"{type(exc).__name__}",
            }
        namespace_changed = bool(registration.get("ok"))
        if not namespace_changed:
            failed_payload = record.model_dump(mode="python")
            failed_payload.update({
                "status": "failed",
                "adata_lineage_id": None,
                "parent_checkpoint_id": None,
                "supersedes_checkpoint_id": None,
                "blob_sha256": None,
                "blob_relative_path": None,
                "size_bytes": None,
                "n_obs": None,
                "n_vars": None,
                "same_as_parent": False,
                "error_summary": (
                    "canonical anchor registration failed: "
                    f"{registration.get('reason')}"
                )[:1000],
            })
            failed_record = AdataCheckpointRecord.model_validate(failed_payload)
            _append_record(state, failed_record)
            return CaptureOutcome(
                record=failed_record,
                disclosure="failed_capture",
                error_summary=failed_record.error_summary,
            )

        _append_record(state, record)
        state.active_adata_lineage_id = record.adata_lineage_id
        state.active_adata_checkpoint_id = record.checkpoint_id
        state.adata_checkpoint_sequence = int(
            getattr(state, "adata_checkpoint_sequence", 0) or 0
        ) + 1
        sandbox.adata_binding_registry_name = registry_name
        sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
        sandbox_state["adata_binding_registry_name"] = registry_name
        state.sandbox_state = sandbox_state
        try:
            sandbox._note_execution_effect(
                mutates_namespace=True,
                reached_kernel=True,
            )
        except Exception:
            logger.warning("checkpoint capture could not advance runtime revision")
        record.captured_runtime_revision = int(
            getattr(sandbox, "runtime_revision", 0) or 0
        )
        try:
            _telemetry = list(
                (getattr(state, "sandbox_state", None) or {}).get(
                    "adata_checkpoint_telemetry"
                )
                or []
            )
            _telemetry.append({
                "checkpoint_id": record.checkpoint_id,
                "boundary": boundary,
                "published": not storage.reused,
                "reused": storage.reused,
                "size_bytes": storage.identity.size,
                "staging_bytes": int(serialized.get("size_bytes") or 0),
                "runtime_instance_id": record.captured_runtime_instance_id,
            })
            _sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
            _sandbox_state["adata_checkpoint_telemetry"] = _telemetry[-50:]
            state.sandbox_state = _sandbox_state
        except Exception:
            logger.debug("could not append checkpoint capture telemetry")

        return CaptureOutcome(
            record=record,
            published=not storage.reused,
            reused=storage.reused,
            registry_name=registry_name,
            anchor_name=anchor_name,
            namespace_changed=True,
        )
    finally:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


async def capture_boundary(sandbox: Any, state: Any, **kwargs: Any) -> CaptureOutcome:
    """Lock-owning capture entry point for callers not already holding cli lock."""
    if sandbox is None:
        return CaptureOutcome(disclosure="no_live_kernel")
    lock = getattr(sandbox, "_exec_lock", None)
    if lock is None:
        return await capture_boundary_locked(sandbox, state, **kwargs)
    async with lock:
        return await capture_boundary_locked(sandbox, state, **kwargs)
