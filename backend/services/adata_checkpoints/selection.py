"""Plan-level AnnData source preparation and warm/cold reconciliation."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from backend.models.schema_parts.adata_checkpoints import (
    AdataLiveBindingRecord,
    AdataSourcePreparation,
    AdataSourceSelection,
)

from .errors import AdataSourcePreparationError
from .layout import resolve_session_relative, verify_blob

logger = logging.getLogger(__name__)

_TERMINAL_BINDING_ERRORS = {
    "binding_stale": "binding_stale",
    "binding_name_collision": "binding_name_collision",
    "anchor_collision": "anchor_collision",
    "shape_mismatch": "shape_mismatch",
    "backed_object_not_allowed": "checkpoint_unreadable",
    "insufficient_memory_headroom": "insufficient_memory_headroom",
    "unsupported_mode": "unsupported_source_mode",
}


def selection_signature(selection: AdataSourceSelection) -> str:
    payload = json.dumps(
        selection.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _session_lineage(state: Any) -> str:
    return str(getattr(state, "restore_branch_id", None) or "main")


def _find_preparation(state: Any, preparation_id: Optional[str]):
    if not preparation_id:
        return None
    for record in getattr(state, "adata_source_preparations", None) or []:
        if str(getattr(record, "preparation_id", "")) == str(preparation_id):
            return record
    return None


def _checkpoint_record(state: Any, checkpoint_id: str):
    for record in getattr(state, "adata_checkpoints", None) or []:
        if str(getattr(record, "checkpoint_id", "")) == str(checkpoint_id):
            return record
    return None


def _binding_record(state: Any, binding_id: str):
    for record in getattr(state, "adata_live_bindings", None) or []:
        if str(getattr(record, "binding_id", "")) == str(binding_id):
            return record
    return None


def _registry_name(state: Any, sandbox: Any) -> str:
    state_name = str(
        (dict(getattr(state, "sandbox_state", None) or {})).get(
            "adata_binding_registry_name"
        )
        or ""
    ).strip()
    if state_name:
        return state_name
    return str(getattr(sandbox, "adata_binding_registry_name", "") or "").strip() or (
        f"scai_adata_registry_{uuid.uuid4().hex[:12]}"
    )


def _remember_registry(state: Any, sandbox: Any, registry_name: str) -> None:
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    sandbox_state["adata_binding_registry_name"] = registry_name
    state.sandbox_state = sandbox_state
    try:
        sandbox.adata_binding_registry_name = registry_name
    except Exception:
        pass
    remember = getattr(sandbox, "remember_host_managed_names", None)
    if callable(remember):
        remember([registry_name])


def _error_from_payload(payload: Dict[str, Any], selection: AdataSourceSelection):
    reason = str(payload.get("reason") or "source_preparation_failed")
    return AdataSourcePreparationError(
        "AnnData source could not be prepared",
        reason=_TERMINAL_BINDING_ERRORS.get(reason, "source_preparation_failed"),
        context={
            "mode": selection.mode,
            "checkpoint_id": selection.checkpoint_id,
            "live_binding_id": selection.live_binding_id,
            "detail": reason,
        },
    )


async def ensure_adata_source_prepared(
    sandbox: Any,
    state: Any,
    selection: AdataSourceSelection,
    *,
    plan: Any = None,
    owner_kind: str = "plan",
    owner_id: Optional[str] = None,
    request_scope_id: Optional[str] = None,
    inherited_preparation_id: Optional[str] = None,
) -> Optional[AdataSourcePreparation]:
    """Apply or verify one requested source.

    Returns ``None`` for ``current``.  Raises ``AdataSourcePreparationError``
    only for the requested source operation.
    """
    if not isinstance(selection, AdataSourceSelection):
        selection = AdataSourceSelection.model_validate(selection)
    if selection.mode == "current":
        return None
    if sandbox is None or getattr(sandbox, "km", None) is None:
        raise AdataSourcePreparationError(
            "The live kernel is unavailable; cannot prepare the selected source",
            reason="no_live_kernel",
            context={"mode": selection.mode},
        )

    lock = getattr(sandbox, "_exec_lock", None)
    if lock is None:
        return await _ensure_locked(
            sandbox,
            state,
            selection,
            plan=plan,
            owner_kind=owner_kind,
            owner_id=owner_id,
            request_scope_id=request_scope_id,
            inherited_preparation_id=inherited_preparation_id,
        )
    async with lock:
        return await _ensure_locked(
            sandbox,
            state,
            selection,
            plan=plan,
            owner_kind=owner_kind,
            owner_id=owner_id,
            request_scope_id=request_scope_id,
            inherited_preparation_id=inherited_preparation_id,
        )


async def _ensure_locked(
    sandbox: Any,
    state: Any,
    selection: AdataSourceSelection,
    *,
    plan: Any = None,
    owner_kind: str,
    owner_id: Optional[str],
    request_scope_id: Optional[str],
    inherited_preparation_id: Optional[str] = None,
) -> AdataSourcePreparation:
    owner_id = str(
        owner_id or getattr(plan, "id", "") or request_scope_id or uuid.uuid4().hex
    )
    session_lineage = _session_lineage(state)
    registry_name = _registry_name(state, sandbox)

    # Same-source revisions inherit the preparation record and must not reload.
    inherited = _find_preparation(
        state,
        inherited_preparation_id
        or getattr(plan, "adata_source_preparation_id", None),
    )
    registry_facts = await sandbox.query_adata_registry_locked(
        registry_name=registry_name
    )
    if inherited is not None:
        if (
            inherited.selection.model_dump(mode="json")
            != selection.model_dump(mode="json")
        ):
            raise AdataSourcePreparationError(
                "The inherited preparation belongs to a different source selection",
                reason="preparation_selection_mismatch",
                context={"preparation_id": inherited.preparation_id},
            )
        registry_entry = (
            (registry_facts.get("preparations") or {}).get(
                inherited.preparation_id
            )
            or {}
        )
        token_ok = (
            str(registry_entry.get("token") or "") == inherited.application_token
        )
        scope_ok = str(registry_entry.get("scope_id") or "") == inherited.scope_id
        if (
            registry_entry
            and token_ok
            and scope_ok
            and str(registry_entry.get("selection_signature") or "")
            == inherited.selection_signature
        ):
            inherited.binding_identity_status = (
                "exact"
                if registry_entry.get("binding_identity_status") == "exact"
                else str(registry_entry.get("binding_identity_status") or "evolved")
            )
            _remember_registry(state, sandbox, registry_name)
            changed = False
            sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
            if owner_kind == "supplementary":
                consumers = list(
                    inherited.consumer_supplementary_request_ids or []
                )
                if owner_id not in consumers:
                    consumers.append(owner_id)
                    inherited.consumer_supplementary_request_ids = consumers
                    changed = True
                request_map = dict(
                    sandbox_state.get("supplementary_requests_v1") or {}
                )
                request_record = dict(request_map.get(owner_id) or {})
                if request_record:
                    if (
                        request_record.get("adata_source_preparation_id")
                        != inherited.preparation_id
                    ):
                        request_record["adata_source_preparation_id"] = (
                            inherited.preparation_id
                        )
                        changed = True
                    request_record["source_selection_signature"] = (
                        inherited.selection_signature
                    )
                    request_map[owner_id] = request_record
                    sandbox_state["supplementary_requests_v1"] = request_map
            elif plan is not None:
                consumers = list(inherited.consumer_plan_ids or [])
                if plan.id not in consumers:
                    consumers.append(plan.id)
                    inherited.consumer_plan_ids = consumers
                    changed = True
                if getattr(plan, "adata_source_preparation_id", None) != (
                    inherited.preparation_id
                ):
                    plan.adata_source_preparation_id = inherited.preparation_id
                    changed = True
            resolved_context = {
                "preparation_id": inherited.preparation_id,
                "active_source_preparation_id": inherited.preparation_id,
                "source_preparation_ids": [inherited.preparation_id],
                "scope_id": inherited.scope_id,
                "active_source_scope_id": inherited.scope_id,
                "mode": inherited.selection.mode,
                "source_selection_signature": inherited.selection_signature,
                "binding_id": inherited.binding_id,
                "binding_name": inherited.binding_name,
                "binding_identity_status": inherited.binding_identity_status,
                "adata_lineage_id": inherited.adata_lineage_id,
                "loaded_checkpoint_id": inherited.loaded_checkpoint_id,
                "ancestor_checkpoint_id": inherited.ancestor_checkpoint_id,
            }
            if sandbox_state.get("resolved_adata_source_context_v1") != resolved_context:
                sandbox_state["resolved_adata_source_context_v1"] = resolved_context
                changed = True
            state.sandbox_state = sandbox_state
            if changed:
                await state.increment_version()
                from backend.services.session_store import async_save_session

                await async_save_session(str(state.id), state, force=True)
            return inherited
        inherited.status = "unresolved"
        raise AdataSourcePreparationError(
            "The restored runtime does not contain the inherited source preparation",
            reason="source_unresolved",
            context={"preparation_id": inherited.preparation_id},
        )

    preparation = AdataSourcePreparation(
        preparation_id=f"prep_{uuid.uuid4().hex}",
        scope_id=f"prep_scope_{uuid.uuid4().hex}",
        owner_kind="supplementary" if owner_kind == "supplementary" else "plan",
        owner_id=owner_id,
        selection=selection,
        selection_signature=selection_signature(selection),
        application_token=uuid.uuid4().hex,
        session_lineage_id=session_lineage,
    )
    namespace_changed = False
    live_binding: Optional[AdataLiveBindingRecord] = None

    try:
        if selection.mode in {"checkpoint_copy", "checkpoint_branch"}:
            record = _checkpoint_record(state, selection.checkpoint_id or "")
            if record is None or record.status != "available":
                raise AdataSourcePreparationError(
                    "The selected checkpoint is unavailable",
                    reason="checkpoint_unavailable",
                    context={"checkpoint_id": selection.checkpoint_id},
                )
            blob_path = resolve_session_relative(state.id, record.blob_relative_path)
            verify_blob(
                blob_path,
                expected_sha256=record.blob_sha256 or "",
                expected_size=record.size_bytes,
            )
            anchor_name = f"scai_adata_anchor_{uuid.uuid4().hex[:12]}"
            if selection.mode == "checkpoint_copy":
                binding_id = f"lb_{uuid.uuid4().hex}"
                binding_name = f"scai_work_{uuid.uuid4().hex[:8]}"
                adata_lineage_id = uuid.uuid4().hex
                ancestor_checkpoint_id = record.checkpoint_id
                payload = await sandbox.apply_adata_source_locked(
                    mode="checkpoint_copy",
                    registry_name=registry_name,
                    preparation_id=preparation.preparation_id,
                    application_token=preparation.application_token,
                    scope_id=preparation.scope_id,
                    owner_kind=preparation.owner_kind,
                    owner_id=preparation.owner_id,
                    selection_signature=preparation.selection_signature,
                    session_lineage_id=session_lineage,
                    checkpoint_path=str(blob_path),
                    expected_n_obs=record.n_obs,
                    expected_n_vars=record.n_vars,
                    adata_lineage_id=adata_lineage_id,
                    ancestor_checkpoint_id=ancestor_checkpoint_id,
                    binding_id=binding_id,
                    binding_name=binding_name,
                    anchor_name=anchor_name,
                )
                if not payload.get("ok"):
                    raise _error_from_payload(payload, selection)
                namespace_changed = True
                live_binding = AdataLiveBindingRecord(
                    binding_id=binding_id,
                    name=str(payload.get("binding_name") or binding_name),
                    anchor_name=str(payload.get("anchor_name") or anchor_name),
                    mode_created="checkpoint_copy",
                    adata_lineage_id=str(
                        payload.get("adata_lineage_id") or adata_lineage_id
                    ),
                    created_from_checkpoint_id=record.checkpoint_id,
                    ancestor_checkpoint_id=record.checkpoint_id,
                    preparation_id=preparation.preparation_id,
                    session_lineage_id=session_lineage,
                    status="registered",
                )
                preparation.binding_id = live_binding.binding_id
                preparation.binding_name = live_binding.name
                preparation.anchor_name = live_binding.anchor_name
                preparation.adata_lineage_id = live_binding.adata_lineage_id
                preparation.loaded_checkpoint_id = record.checkpoint_id
                preparation.ancestor_checkpoint_id = record.checkpoint_id
                preparation.binding_identity_status = "exact"
            else:  # checkpoint_branch
                adata_lineage_id = uuid.uuid4().hex
                payload = await sandbox.apply_adata_source_locked(
                    mode="checkpoint_branch",
                    registry_name=registry_name,
                    preparation_id=preparation.preparation_id,
                    application_token=preparation.application_token,
                    scope_id=preparation.scope_id,
                    owner_kind=preparation.owner_kind,
                    owner_id=preparation.owner_id,
                    selection_signature=preparation.selection_signature,
                    session_lineage_id=session_lineage,
                    checkpoint_path=str(blob_path),
                    expected_n_obs=record.n_obs,
                    expected_n_vars=record.n_vars,
                    adata_lineage_id=adata_lineage_id,
                    ancestor_checkpoint_id=record.checkpoint_id,
                    anchor_name=anchor_name,
                )
                if not payload.get("ok"):
                    raise _error_from_payload(payload, selection)
                namespace_changed = True
                preparation.anchor_name = str(
                    payload.get("anchor_name") or anchor_name
                )
                preparation.adata_lineage_id = str(
                    payload.get("adata_lineage_id") or adata_lineage_id
                )
                preparation.loaded_checkpoint_id = record.checkpoint_id
                preparation.ancestor_checkpoint_id = record.checkpoint_id
                preparation.binding_identity_status = "exact"
                state.active_adata_lineage_id = preparation.adata_lineage_id
                state.active_adata_checkpoint_id = record.checkpoint_id
        elif selection.mode in {"live_binding", "promote_live"}:
            binding = _binding_record(state, selection.live_binding_id or "")
            if binding is None or binding.status not in {"registered", "pending_reconcile"}:
                raise AdataSourcePreparationError(
                    "The selected live binding is unavailable",
                    reason="binding_stale",
                    context={"live_binding_id": selection.live_binding_id},
                )
            registry_binding = (
                (registry_facts.get("bindings") or {}).get(binding.binding_id) or {}
            )
            if (
                str(registry_binding.get("anchor_name") or "")
                != binding.anchor_name
                or str(registry_binding.get("name") or "") != binding.name
            ):
                binding.status = "stale"
                raise AdataSourcePreparationError(
                    "The selected live binding no longer matches its runtime anchor",
                    reason="binding_stale",
                    context={"live_binding_id": binding.binding_id},
                )
            anchor_name = (
                f"scai_adata_anchor_{uuid.uuid4().hex[:12]}"
                if selection.mode == "promote_live"
                else binding.anchor_name
            )
            payload = await sandbox.apply_adata_source_locked(
                mode=selection.mode,
                registry_name=registry_name,
                preparation_id=preparation.preparation_id,
                application_token=preparation.application_token,
                scope_id=preparation.scope_id,
                owner_kind=preparation.owner_kind,
                owner_id=preparation.owner_id,
                selection_signature=preparation.selection_signature,
                session_lineage_id=session_lineage,
                adata_lineage_id=binding.adata_lineage_id,
                ancestor_checkpoint_id=binding.ancestor_checkpoint_id,
                binding_id=binding.binding_id,
                binding_name=binding.name,
                anchor_name=anchor_name,
            )
            if not payload.get("ok"):
                raise _error_from_payload(payload, selection)
            namespace_changed = True
            preparation.binding_id = binding.binding_id
            preparation.binding_name = binding.name
            preparation.anchor_name = str(payload.get("anchor_name") or anchor_name)
            preparation.adata_lineage_id = binding.adata_lineage_id
            preparation.ancestor_checkpoint_id = binding.ancestor_checkpoint_id
            preparation.binding_identity_status = "exact"
            if selection.mode == "promote_live":
                state.active_adata_lineage_id = binding.adata_lineage_id
                state.active_adata_checkpoint_id = binding.ancestor_checkpoint_id
                binding.mode_created = "promote_live"
        else:
            raise AdataSourcePreparationError(
                "Unsupported AnnData source mode",
                reason="unsupported_source_mode",
                context={"mode": selection.mode},
            )
    except AdataSourcePreparationError:
        preparation.status = "failed"
        raise

    preparation.status = "applied_warm"
    if namespace_changed:
        try:
            sandbox._note_execution_effect(
                mutates_namespace=True,
                reached_kernel=True,
            )
        except Exception:
            logger.warning("source preparation could not advance runtime revision")

    _remember_registry(state, sandbox, registry_name)
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    sandbox_state["resolved_adata_source_context_v1"] = {
        "preparation_id": preparation.preparation_id,
        "active_source_preparation_id": preparation.preparation_id,
        "source_preparation_ids": [preparation.preparation_id],
        "scope_id": preparation.scope_id,
        "active_source_scope_id": preparation.scope_id,
        "mode": preparation.selection.mode,
        "source_selection_signature": preparation.selection_signature,
        "binding_id": preparation.binding_id,
        "binding_name": preparation.binding_name,
        "binding_identity_status": preparation.binding_identity_status,
        "adata_lineage_id": preparation.adata_lineage_id,
        "loaded_checkpoint_id": preparation.loaded_checkpoint_id,
        "ancestor_checkpoint_id": preparation.ancestor_checkpoint_id,
    }
    state.sandbox_state = sandbox_state
    preparations = getattr(state, "adata_source_preparations", None)
    if preparations is None:
        preparations = []
        state.adata_source_preparations = preparations
    preparations.append(preparation)
    if live_binding is not None:
        bindings = getattr(state, "adata_live_bindings", None)
        if bindings is None:
            bindings = []
            state.adata_live_bindings = bindings
        bindings.append(live_binding)
    if plan is not None and owner_kind != "supplementary":
        plan.adata_source_preparation_id = preparation.preparation_id
        consumers = list(preparation.consumer_plan_ids or [])
        if plan.id not in consumers:
            consumers.append(plan.id)
        preparation.consumer_plan_ids = consumers
    elif owner_kind == "supplementary":
        consumers = list(preparation.consumer_supplementary_request_ids or [])
        if owner_id not in consumers:
            consumers.append(owner_id)
        preparation.consumer_supplementary_request_ids = consumers
        request_map = dict(sandbox_state.get("supplementary_requests_v1") or {})
        request_record = dict(request_map.get(owner_id) or {})
        if request_record:
            request_record["adata_source_preparation_id"] = preparation.preparation_id
            request_record["source_selection_signature"] = (
                preparation.selection_signature
            )
            request_map[owner_id] = request_record
            sandbox_state["supplementary_requests_v1"] = request_map
            state.sandbox_state = sandbox_state

    try:
        await state.increment_version()
        from backend.services.session_store import async_save_session

        await async_save_session(str(state.id), state, force=True)
        runtime_saved = False
        try:
            await sandbox._save_session_locked()
            sandbox.sync_runtime_persistence_state(state)
            runtime_saved = True
        except Exception:
            logger.warning(
                "source preparation runtime save failed; keeping warm state"
            )
        if runtime_saved:
            try:
                from backend.utils.runtime_manifest import (
                    read_runtime_manifest,
                    runtime_manifest_path,
                )

                manifest = read_runtime_manifest(
                    runtime_manifest_path(str(state.id))
                )
                saved_names = set(manifest.get("saved_names") or [])
                preparation.registry_saved_name_ok = registry_name in saved_names
                required_names = [preparation.binding_name] if (
                    preparation.binding_identity_status == "exact"
                    and preparation.binding_name
                ) else []
                if required_names:
                    preparation.binding_saved_names_ok = all(
                        name in saved_names for name in required_names
                    )
                if preparation.registry_saved_name_ok and (
                    not required_names
                    or preparation.binding_saved_names_ok is True
                ):
                    preparation.status = "durable"
                preparation.runtime_instance_id = getattr(
                    sandbox, "runtime_instance_id", None
                )
                preparation.runtime_revision = int(
                    getattr(sandbox, "runtime_revision", 0) or 0
                )
                preparation.lifecycle_generation = int(
                    getattr(state, "lifecycle_generation", 0) or 0
                )
                preparation.runtime_manifest_revision = int(
                    manifest.get("revision") or 0
                )
                preparation.runtime_pkl_sha256 = str(
                    manifest.get("pkl_sha256") or ""
                ) or None
                pairing = manifest.get("pairing") or {}
                preparation.runtime_pairing_state_version = int(
                    pairing.get("state_version") or 0
                )
                preparation.runtime_pairing_checkpoint = pairing.get("checkpoint")
                state.sandbox_state = dict(state.sandbox_state or {})
                state.sandbox_state["source_preparation_ack_v1"] = {
                    preparation.preparation_id: preparation.status,
                }
                await state.increment_version()
                await async_save_session(str(state.id), state, force=True)
            except Exception:
                logger.warning(
                    "source preparation runtime acknowledgment could not be read",
                    exc_info=True,
                )
    except Exception:
        logger.exception("source preparation state save failed")
    return preparation
