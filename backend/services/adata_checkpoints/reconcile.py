"""Truthful catalog availability reconciliation after load/import/snapshot."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .layout import resolve_session_relative, verify_blob


def reconcile_catalog_availability(
    state: Any,
    *,
    verify_digests: bool = True,
    runtime_available: bool = True,
) -> bool:
    """Mark missing/corrupt blobs unavailable; never delete audit metadata.

    Returns True when any record changed.  This function performs no load and
    makes no scientific judgement.
    """
    changed = False
    for record in getattr(state, "adata_checkpoints", None) or []:
        if str(getattr(record, "status", "")) != "available":
            continue
        relative = str(getattr(record, "blob_relative_path", "") or "")
        try:
            path = resolve_session_relative(state.id, relative)
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError("checkpoint blob is missing")
            if verify_digests:
                verify_blob(
                    path,
                    expected_sha256=str(getattr(record, "blob_sha256", "") or ""),
                    expected_size=getattr(record, "size_bytes", None),
                )
            else:
                expected_size = getattr(record, "size_bytes", None)
                if expected_size is not None and path.stat().st_size != int(
                    expected_size
                ):
                    raise ValueError("checkpoint blob size does not match")
        except Exception as exc:
            record.status = "unavailable"
            record.unavailable_reason = (
                "corrupt" if "digest" in str(exc).lower() else "not_materialized"
            )
            record.error_summary = (
                f"catalog reconciliation: {type(exc).__name__}"
            )[:1000]
            changed = True

    if not runtime_available:
        for binding in getattr(state, "adata_live_bindings", None) or []:
            if str(getattr(binding, "status", "")) in {
                "registered", "pending_reconcile",
            }:
                binding.status = "unresolved"
                changed = True
    return changed


def scrub_runtime_dependent_state(state_payload: dict) -> dict:
    """Return a deep-scrubbed copy for a results-only portable archive.

    Historical execution and checkpoint records remain provenance.  Only
    continuation state that cannot be truthfully resumed without a runtime is
    removed or downgraded.
    """
    import copy

    payload = copy.deepcopy(state_payload) if isinstance(state_payload, dict) else {}
    plan = payload.get("current_plan")
    if isinstance(plan, dict):
        plan["bioinformatics_run_scope_id"] = None
        plan["bioinformatics_settlement_scope_id"] = None
    payload["bioinformatics_settlement_attempt_ids"] = {}
    for preparation in payload.get("adata_source_preparations") or []:
        if not isinstance(preparation, dict):
            continue
        if str(preparation.get("status") or "") in {
            "durable", "applied_warm", "pending_reconcile",
        }:
            preparation["status"] = "unresolved"
        preparation["registry_saved_name_ok"] = None
        preparation["binding_saved_names_ok"] = None
        preparation["runtime_pairing_state_version"] = None
        preparation["runtime_pairing_checkpoint"] = None
    for binding in payload.get("adata_live_bindings") or []:
        if not isinstance(binding, dict):
            continue
        if str(binding.get("status") or "") in {
            "registered", "pending_reconcile",
        }:
            binding["status"] = "unresolved"
    sandbox_state = payload.get("sandbox_state")
    if isinstance(sandbox_state, dict):
        sandbox_state.pop("resolved_adata_source_context_v1", None)
        sandbox_state.pop("source_preparation_ack_v1", None)
        request_map = sandbox_state.get("supplementary_requests_v1")
        if isinstance(request_map, dict):
            for record in request_map.values():
                if isinstance(record, dict) and str(record.get("status") or "") in {
                    "queued", "planning", "running", "interrupted",
                }:
                    record["status"] = "failed"
                    record["error_summary"] = (
                        "Request was not carried into a results-only export."
                    )[:500]
    return payload


def mark_runtime_reconciliation_pending(state: Any, *, runtime_available: bool) -> bool:
    """Downgrade Host live-state assertions that require a kernel proof.

    A deserialized durable preparation or registered binding is not proof of the
    live namespace.  It becomes pending until the first kernel start verifies
    manifest token/scope/identity.
    """
    changed = False
    for preparation in getattr(state, "adata_source_preparations", None) or []:
        if str(getattr(preparation, "status", "")) in {
            "durable", "applied_warm",
        }:
            preparation.status = (
                "pending_reconcile" if runtime_available else "unresolved"
            )
            preparation.binding_identity_status = "unknown"
            changed = True
    for binding in getattr(state, "adata_live_bindings", None) or []:
        if str(getattr(binding, "status", "")) == "registered":
            binding.status = (
                "pending_reconcile" if runtime_available else "unresolved"
            )
            changed = True
    return changed


def rebase_inherited_source_state(
    state: Any,
    *,
    child_lineage_id: str,
    runtime_available: bool,
) -> bool:
    """Rebase inherited preparation/live-binding ownership to a child session.

    Ultimate historical origin is preserved separately from the immediate
    materialized parent.  File materialization never proves a live object, so
    status becomes pending (or unresolved without a runtime) until the child's
    first kernel start verifies the registry.
    """
    changed = False
    child = str(child_lineage_id or "main")
    for preparation in getattr(state, "adata_source_preparations", None) or []:
        parent = str(
            getattr(preparation, "session_lineage_id", "main") or "main"
        )
        origin = (
            getattr(preparation, "origin_session_lineage_id", None)
            or parent
        )
        preparation.origin_session_lineage_id = origin
        preparation.reconcile_from_session_lineage_id = parent
        preparation.session_lineage_id = child
        preparation.binding_identity_status = "unknown"
        preparation.status = (
            "pending_reconcile" if runtime_available else "unresolved"
        )
        preparation.registry_saved_name_ok = None
        preparation.binding_saved_names_ok = None
        preparation.runtime_instance_id = None
        preparation.runtime_revision = None
        preparation.lifecycle_generation = None
        preparation.runtime_manifest_revision = None
        preparation.runtime_pkl_sha256 = None
        preparation.runtime_pairing_state_version = None
        preparation.runtime_pairing_checkpoint = None
        changed = True
    for binding in getattr(state, "adata_live_bindings", None) or []:
        parent = str(
            getattr(binding, "session_lineage_id", "main") or "main"
        )
        origin = (
            getattr(binding, "origin_session_lineage_id", None)
            or parent
        )
        binding.origin_session_lineage_id = origin
        binding.reconcile_from_session_lineage_id = parent
        binding.session_lineage_id = child
        binding.status = "pending_reconcile" if runtime_available else "unresolved"
        binding.last_runtime_instance_id = None
        binding.last_runtime_revision = None
        changed = True
    return changed


async def reconcile_live_runtime_state(sandbox: Any, state: Any) -> bool:
    """Resolve pending source records from the signed live runtime and registry.

    This is the first-kernel counterpart to the no-kernel load reconciliation.
    It never chooses or reloads a source; it only upgrades live claims that the
    committed manifest and actual object identities prove.
    """
    pending_preparations = [
        item for item in (getattr(state, "adata_source_preparations", None) or [])
        if str(getattr(item, "status", "")) == "pending_reconcile"
    ]
    pending_bindings = [
        item for item in (getattr(state, "adata_live_bindings", None) or [])
        if str(getattr(item, "status", "")) == "pending_reconcile"
    ]
    if not pending_preparations and not pending_bindings:
        return False

    changed = False
    try:
        from backend.utils.runtime_manifest import (
            read_runtime_manifest,
            runtime_manifest_path,
        )

        manifest = read_runtime_manifest(runtime_manifest_path(str(state.id)))
        committed_digest = str(
            (getattr(state, "runtime_persistence", None) or {}).get(
                "committed_digest", ""
            )
            or ""
        )
        if committed_digest and manifest.get("pkl_sha256") != committed_digest:
            raise ValueError("runtime manifest digest does not match state")
        committed_pairing = (
            getattr(state, "runtime_persistence", None) or {}
        ).get("committed_pairing")
        if committed_pairing and manifest.get("pairing") != committed_pairing:
            raise ValueError("runtime manifest pairing does not match state")
        saved_names = set(manifest.get("saved_names") or [])
        registry_name = str(
            (getattr(state, "sandbox_state", None) or {}).get(
                "adata_binding_registry_name", ""
            )
            or getattr(sandbox, "adata_binding_registry_name", "")
            or ""
        )
        if not registry_name or registry_name not in saved_names:
            raise ValueError("runtime manifest does not retain the registry")
        lock = getattr(sandbox, "_exec_lock", None)
        if lock is None:
            reconciled = await sandbox._reconcile_adata_registry_locked(
                registry_name=registry_name
            )
            registry = await sandbox.query_adata_registry_locked(
                registry_name=registry_name
            )
        else:
            async with lock:
                reconciled = await sandbox._reconcile_adata_registry_locked(
                    registry_name=registry_name
                )
                registry = await sandbox.query_adata_registry_locked(
                    registry_name=registry_name
                )
        if not reconciled.get("ok"):
            raise ValueError("live registry reconciliation failed")
        if not registry.get("registry_present"):
            raise ValueError("live registry is absent")

        preparation_entries = registry.get("preparations") or {}
        preparation_statuses = reconciled.get("preparation_statuses") or {}
        for preparation in pending_preparations:
            entry = preparation_entries.get(preparation.preparation_id) or {}
            expected_lineages = {
                str(preparation.session_lineage_id or "main"),
                str(preparation.reconcile_from_session_lineage_id or ""),
            }
            identity_ok = bool(
                entry
                and str(entry.get("token") or "")
                == preparation.application_token
                and str(entry.get("scope_id") or "") == preparation.scope_id
                and str(entry.get("selection_signature") or "")
                == preparation.selection_signature
                and str(entry.get("session_lineage_id") or "main")
                in expected_lineages
            )
            required_names = {
                str(value)
                for value in (
                    entry.get("anchor_name"),
                    entry.get("binding_name"),
                )
                if str(value or "")
            }
            if not identity_ok or not required_names.issubset(saved_names):
                preparation.status = "unresolved"
                preparation.binding_identity_status = "unknown"
                changed = True
                continue
            status = str(
                preparation_statuses.get(preparation.preparation_id)
                or entry.get("binding_identity_status")
                or "unknown"
            )
            preparation.status = "durable"
            preparation.binding_identity_status = (
                status if status in {"exact", "evolved", "missing"} else "unknown"
            )
            preparation.runtime_instance_id = getattr(
                sandbox, "runtime_instance_id", None
            )
            preparation.runtime_revision = int(
                getattr(sandbox, "runtime_revision", 0) or 0
            )
            preparation.runtime_manifest_revision = int(
                manifest.get("revision") or 0
            )
            preparation.runtime_pkl_sha256 = str(
                manifest.get("pkl_sha256") or ""
            ) or None
            pairing = manifest.get("pairing") or {}
            preparation.runtime_pairing_state_version = pairing.get(
                "state_version"
            )
            preparation.runtime_pairing_checkpoint = pairing.get("checkpoint")
            preparation.registry_saved_name_ok = True
            preparation.binding_saved_names_ok = True
            changed = True

        binding_statuses = reconciled.get("binding_statuses") or {}
        registry_bindings = registry.get("bindings") or {}
        for binding in pending_bindings:
            entry = registry_bindings.get(binding.binding_id) or {}
            names_ok = {
                str(entry.get("anchor_name") or ""),
                str(entry.get("name") or ""),
            } - {""}
            status = str(binding_statuses.get(binding.binding_id) or "")
            if (
                entry
                and names_ok.issubset(saved_names)
                and status == "registered"
                and str(entry.get("session_lineage_id") or "main") in {
                    str(binding.session_lineage_id or "main"),
                    str(binding.reconcile_from_session_lineage_id or ""),
                }
            ):
                binding.status = "registered"
                binding.last_runtime_instance_id = getattr(
                    sandbox, "runtime_instance_id", None
                )
                binding.last_runtime_revision = int(
                    getattr(sandbox, "runtime_revision", 0) or 0
                )
            else:
                binding.status = "unresolved"
            changed = True

        preparation_lineages = {
            preparation.preparation_id: {
                "from": str(
                    preparation.reconcile_from_session_lineage_id or "main"
                ),
                "to": str(preparation.session_lineage_id or "main"),
            }
            for preparation in pending_preparations
            if preparation.status == "durable"
            and preparation.reconcile_from_session_lineage_id
        }
        binding_lineages = {
            binding.binding_id: {
                "from": str(binding.reconcile_from_session_lineage_id or "main"),
                "to": str(binding.session_lineage_id or "main"),
            }
            for binding in pending_bindings
            if binding.status == "registered"
            and binding.reconcile_from_session_lineage_id
        }
        if preparation_lineages or binding_lineages:
            lock = getattr(sandbox, "_exec_lock", None)
            if lock is None:
                rebased = await sandbox.rebase_adata_registry_lineage_locked(
                    registry_name=registry_name,
                    preparation_lineages=preparation_lineages,
                    binding_lineages=binding_lineages,
                )
            else:
                async with lock:
                    rebased = await sandbox.rebase_adata_registry_lineage_locked(
                        registry_name=registry_name,
                        preparation_lineages=preparation_lineages,
                        binding_lineages=binding_lineages,
                    )
            if not rebased.get("ok"):
                for preparation in pending_preparations:
                    if preparation.preparation_id in preparation_lineages:
                        preparation.status = "unresolved"
                        preparation.binding_identity_status = "unknown"
                for binding in pending_bindings:
                    if binding.binding_id in binding_lineages:
                        binding.status = "unresolved"
                changed = True
            else:
                pending_rebase = {
                    "preparation_ids": sorted(preparation_lineages),
                    "binding_ids": sorted(binding_lineages),
                }
                for preparation in pending_preparations:
                    if preparation.preparation_id in preparation_lineages:
                        preparation.status = "pending_reconcile"
                for binding in pending_bindings:
                    if binding.binding_id in binding_lineages:
                        binding.status = "pending_reconcile"
                setattr(sandbox, "_adata_lineage_rebase_pending", pending_rebase)
                try:
                    sandbox._note_execution_effect(
                        mutates_namespace=True,
                        reached_kernel=True,
                    )
                except Exception:
                    pass
                changed = True
    except Exception:
        for preparation in pending_preparations:
            preparation.status = "unresolved"
            preparation.binding_identity_status = "unknown"
            changed = True
        for binding in pending_bindings:
            binding.status = "unresolved"
            changed = True
    return changed
