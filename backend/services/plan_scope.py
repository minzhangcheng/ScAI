"""Durable foundation identity and Innovation plan-scope contracts.

This module records only Host-observed plan, execution, checkpoint, and cohort
identity. It deliberately does not infer scientific overlap from method text.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable, Optional

from backend.models.schema_parts.session import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPlan,
    FoundationReference,
)
from backend.services.cohort_identity import active_cohort_identity
from backend.services.execution_evidence import (
    PLAN_LINKED_RECEIPT_STATUSES,
    execution_is_current_receipt,
)
from backend.services.figure_contract import collect_plan_figure_requirements


SCOPE_CONTRACT_VERSION = 1
_MAX_INVENTORIES = 8
_MAX_STEP_RECEIPTS = 64
_MAX_FOUNDATION_REFERENCES = 16


class FoundationInventoryError(ValueError):
    """A Round 1 foundation cannot be identified from durable Host records."""


class PlanScopeValidationError(ValueError):
    """A plan's structured scope references violate the technical contract."""


def current_lineage_id(state: Any) -> str:
    return str(getattr(state, "restore_branch_id", None) or "main")


def is_innovation_scope(
    state: Any,
    plan: Any = None,
    *,
    phase: Optional[str] = None,
) -> bool:
    phase = str(
        phase
        or getattr(plan, "phase", None)
        or getattr(getattr(state, "analysis_phase", None), "value", "")
        or ""
    ).lower()
    return phase == "innovation" or int(
        getattr(state, "current_round", 0) or 0
    ) > 1


def _inventory_identity(lineage_id: str, plan_id: str, checkpoint_id: str) -> str:
    payload = "\0".join((lineage_id, plan_id, checkpoint_id)).encode("utf-8")
    return "foundation-v1-" + hashlib.sha256(payload).hexdigest()[:40]


def _valid_foundation_receipts(
    state: Any,
    *,
    round_number: int,
    plan_id: str,
    lineage_id: str,
) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for receipt in getattr(state, "code_history", None) or []:
        if getattr(receipt, "executor_run_kind", None) != "ordinary":
            continue
        if (
            getattr(receipt, "plan_binding_status", None)
            not in PLAN_LINKED_RECEIPT_STATUSES
        ):
            continue
        if not execution_is_current_receipt(
            receipt,
            round_number=round_number,
            plan_id=plan_id,
            lineage_id=lineage_id,
        ):
            continue
        for raw_step_id in getattr(receipt, "plan_step_ids", None) or []:
            try:
                step_id = int(raw_step_id)
            except (TypeError, ValueError):
                continue
            if step_id <= 0:
                continue
            result.setdefault(step_id, []).append(str(receipt.id))
    return {
        step_id: list(dict.fromkeys(ids))[:_MAX_STEP_RECEIPTS]
        for step_id, ids in result.items()
    }


def _figure_keys_by_step(plan: Any) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    try:
        requirements = collect_plan_figure_requirements(plan)
    except Exception:
        requirements = []
    for requirement in requirements:
        step_id = getattr(requirement, "plan_step_id", None)
        if step_id is None:
            continue
        result.setdefault(int(step_id), []).append(str(requirement.key))
    return {
        step_id: list(dict.fromkeys(keys))[:64]
        for step_id, keys in result.items()
    }


def capture_accepted_foundation_inventory(
    state: Any,
    plan: Any,
    checkpoint: Any,
    *,
    provenance: str = "captured",
) -> AcceptedFoundationInventory:
    """Create an idempotent Round 1 inventory from a successful checkpoint.

    No kernel query is performed. In particular, unavailable AnnData structure
    fields remain empty instead of being inferred from scientific code or prose.
    """
    if plan is None or int(getattr(plan, "round_number", 0) or 0) not in (0, 1):
        raise FoundationInventoryError("Round 1 plan identity is unavailable")
    checkpoint_status = str(getattr(checkpoint, "status", "") or "")
    if checkpoint is None or checkpoint_status != "available":
        raise FoundationInventoryError(
            "No available canonical AnnData checkpoint identifies the accepted foundation"
        )
    checkpoint_id = str(getattr(checkpoint, "checkpoint_id", "") or "")
    plan_id = str(getattr(plan, "id", "") or "")
    checkpoint_plan_id = str(getattr(checkpoint, "source_plan_id", "") or "")
    if not checkpoint_id or not plan_id or checkpoint_plan_id != plan_id:
        raise FoundationInventoryError(
            "The checkpoint does not identify the accepted Round 1 plan"
        )
    round_number = int(getattr(checkpoint, "round_number", 0) or 0)
    if round_number != 1:
        raise FoundationInventoryError("Foundation checkpoint is not from Round 1")

    lineage_id = str(
        getattr(checkpoint, "session_lineage_id", None)
        or current_lineage_id(state)
    )
    receipts_by_step = _valid_foundation_receipts(
        state,
        round_number=round_number,
        plan_id=plan_id,
        lineage_id=lineage_id,
    )
    figure_keys = _figure_keys_by_step(plan)
    foundation_steps = []
    for step in getattr(plan, "steps", None) or []:
        try:
            step_id = int(getattr(step, "id", 0) or 0)
        except (TypeError, ValueError):
            continue
        step_receipts = receipts_by_step.get(step_id, [])
        if (
            step_id <= 0
            or (
                str(getattr(step, "status", "")) != "completed"
                and not step_receipts
            )
        ):
            continue
        foundation_steps.append(AcceptedFoundationStep(
            plan_step_id=step_id,
            title=str(getattr(step, "title", "") or "")[:300],
            method=str(getattr(step, "method", "") or "")[:600],
            evidence_execution_ids=step_receipts,
            figure_requirement_keys=figure_keys.get(step_id, []),
        ))

    cohort = active_cohort_identity(state)
    cohort_digest = str(cohort.get("cohort_digest") or "") or None
    inventory_id = _inventory_identity(lineage_id, plan_id, checkpoint_id)
    inventory = AcceptedFoundationInventory(
        inventory_id=inventory_id,
        round_number=round_number,
        plan_id=plan_id,
        planning_stage=str(getattr(checkpoint, "planning_stage", "full") or "full"),
        lineage_id=lineage_id,
        provenance=provenance,
        adata_checkpoint_id=checkpoint_id,
        adata_source_preparation_ids=list(
            dict.fromkeys(
                str(value)
                for value in (
                    getattr(checkpoint, "source_preparation_ids_at_capture", None)
                    or []
                )
                if str(value or "").strip()
            )
        )[:32],
        cohort_digest=cohort_digest,
        steps=foundation_steps[:128],
        # Structure keys are not currently captured at this boundary. Do not
        # issue a second scientific/runtime probe just to populate this field.
        observed_adata_structure={},
        accepted_at=getattr(checkpoint, "created_at", None) or datetime.now(),
    )

    inventories = list(getattr(state, "accepted_foundation_inventories", None) or [])
    replaced = False
    for index, existing in enumerate(inventories):
        if str(getattr(existing, "inventory_id", "")) == inventory_id:
            inventories[index] = inventory
            replaced = True
            break
    if not replaced:
        inventories.append(inventory)
    state.accepted_foundation_inventories = inventories[-_MAX_INVENTORIES:]
    return inventory


def _find_round1_plan(state: Any, plan_id: str) -> Optional[AnalysisPlan]:
    current_plan = getattr(state, "current_plan", None)
    if current_plan is not None and str(getattr(current_plan, "id", "")) == plan_id:
        try:
            return AnalysisPlan.model_validate(current_plan.model_dump(mode="python"))
        except Exception:
            return None
    for entry in reversed(getattr(state, "plan_dialogue_history", None) or []):
        if not isinstance(entry, dict) or entry.get("pi_decision") != "approved":
            continue
        raw_plan = entry.get("plan")
        if not isinstance(raw_plan, dict) or str(raw_plan.get("id") or "") != plan_id:
            continue
        try:
            return AnalysisPlan.model_validate(raw_plan)
        except Exception:
            return None
    return None


def _reconstruct_inventory_from_checkpoint(
    state: Any,
    checkpoint: Any,
    receipts_by_step: dict[int, list[str]],
) -> AcceptedFoundationInventory:
    """Rebuild only identities retained by a successful legacy boundary.

    Older staged sessions did not retain the Stage 1 plan after Stage 2 planning.
    The available Stage 1 checkpoint still host-stamps the plan, lineage, and
    attempt set. Keep titles and methods empty instead of inferring semantics from
    code or filenames.
    """
    lineage_id = str(getattr(checkpoint, "session_lineage_id", None) or "main")
    plan_id = str(getattr(checkpoint, "source_plan_id", "") or "")
    checkpoint_id = str(getattr(checkpoint, "checkpoint_id", "") or "")
    receipt_map = {
        str(getattr(receipt, "id", "")): receipt
        for receipt in (getattr(state, "code_history", None) or [])
    }
    cohort_digests = {
        str(getattr(receipt_map.get(execution_id), "cohort_digest", "") or "")
        for execution_ids in receipts_by_step.values()
        for execution_id in execution_ids
        if str(getattr(receipt_map.get(execution_id), "cohort_digest", "") or "")
    }
    inventory = AcceptedFoundationInventory(
        inventory_id=_inventory_identity(lineage_id, plan_id, checkpoint_id),
        round_number=1,
        plan_id=plan_id,
        planning_stage=str(
            getattr(checkpoint, "planning_stage", "stage1") or "stage1"
        ),
        lineage_id=lineage_id,
        provenance="reconstructed",
        adata_checkpoint_id=checkpoint_id,
        adata_source_preparation_ids=list(dict.fromkeys(
            str(value)
            for value in (
                getattr(checkpoint, "source_preparation_ids_at_capture", None) or []
            )
            if str(value or "").strip()
        ))[:32],
        cohort_digest=(next(iter(cohort_digests)) if len(cohort_digests) == 1 else None),
        steps=[
            AcceptedFoundationStep(
                plan_step_id=step_id,
                evidence_execution_ids=execution_ids[:_MAX_STEP_RECEIPTS],
            )
            for step_id, execution_ids in sorted(receipts_by_step.items())
        ][:128],
        observed_adata_structure={},
        accepted_at=getattr(checkpoint, "created_at", None) or datetime.now(),
    )
    inventories = list(getattr(state, "accepted_foundation_inventories", None) or [])
    inventories = [
        existing for existing in inventories
        if str(getattr(existing, "inventory_id", "")) != inventory.inventory_id
    ]
    state.accepted_foundation_inventories = (inventories + [inventory])[-_MAX_INVENTORIES:]
    return inventory


def reconstruct_foundation_inventory(state: Any) -> Optional[AcceptedFoundationInventory]:
    """Rebuild legacy identity from a retained plan or successful Host boundary."""
    lineage_id = current_lineage_id(state)
    checkpoints = sorted(
        (
            record
            for record in getattr(state, "adata_checkpoints", None) or []
            if str(getattr(record, "status", "")) == "available"
            and int(getattr(record, "round_number", 0) or 0) == 1
            and str(getattr(record, "boundary", "")) in {
                "stage1_complete", "round_complete",
            }
            and (getattr(record, "session_lineage_id", None) or "main") == lineage_id
            and str(getattr(record, "source_plan_id", "") or "")
        ),
        key=lambda item: (
            str(getattr(item, "boundary", "")) == "stage1_complete",
            getattr(item, "created_at", datetime.min),
        ),
        reverse=True,
    )
    for checkpoint in checkpoints:
        plan_id = str(getattr(checkpoint, "source_plan_id", "") or "")
        plan = _find_round1_plan(state, plan_id)
        receipts_by_step = _valid_foundation_receipts(
            state,
            round_number=1,
            plan_id=plan_id,
            lineage_id=lineage_id,
        )
        captured_attempt_ids = {
            str(value)
            for value in (
                getattr(checkpoint, "attempt_execution_ids_at_capture", None) or []
            )
            if str(value or "")
        }
        if captured_attempt_ids:
            receipts_by_step = {
                step_id: [
                    execution_id for execution_id in execution_ids
                    if execution_id in captured_attempt_ids
                ]
                for step_id, execution_ids in receipts_by_step.items()
            }
            receipts_by_step = {
                step_id: execution_ids
                for step_id, execution_ids in receipts_by_step.items()
                if execution_ids
            }
        # Reconstruction requires retained execution linkage; a plan alone is
        # not enough to invent accepted work.
        if not receipts_by_step:
            continue
        if plan is None:
            return _reconstruct_inventory_from_checkpoint(
                state, checkpoint, receipts_by_step
            )
        for step in plan.steps:
            if int(getattr(step, "id", 0) or 0) in receipts_by_step:
                step.status = "completed"
        try:
            inventory = capture_accepted_foundation_inventory(
                state,
                plan,
                checkpoint,
                provenance="reconstructed",
            )
        except (FoundationInventoryError, ValueError):
            continue
        return inventory
    return None


def resolve_foundation_inventory(
    state: Any,
    *,
    reconstruct: bool = True,
) -> Optional[AcceptedFoundationInventory]:
    lineage_id = current_lineage_id(state)
    inventories = list(getattr(state, "accepted_foundation_inventories", None) or [])
    for inventory in reversed(inventories):
        if (getattr(inventory, "lineage_id", None) or "main") == lineage_id:
            return inventory
    if not reconstruct:
        return None
    return reconstruct_foundation_inventory(state)


def clone_foundation_inventories_for_branch(
    state: Any,
    *,
    source_lineage_id: str,
    branch_id: str,
) -> int:
    """Copy source-lineage inventory metadata into a restored child unchanged."""
    inventories = list(getattr(state, "accepted_foundation_inventories", None) or [])
    copies = []
    for inventory in inventories:
        if (getattr(inventory, "lineage_id", None) or "main") != source_lineage_id:
            continue
        existing = next(
            (
                candidate
                for candidate in inventories
                if str(getattr(candidate, "inventory_id", ""))
                == str(getattr(inventory, "inventory_id", ""))
                and str(getattr(candidate, "lineage_id", "") or "") == branch_id
            ),
            None,
        )
        if existing is not None:
            continue
        copies.append(inventory.model_copy(deep=True, update={
            "lineage_id": branch_id,
            "inherited_from_lineage_id": source_lineage_id,
            "provenance": "inherited",
        }))
    if copies:
        state.accepted_foundation_inventories = (inventories + copies)[-_MAX_INVENTORIES:]
    return len(copies)


def _bounded_inventory_projection(
    inventory: AcceptedFoundationInventory,
) -> dict[str, Any]:
    payload = inventory.model_dump(mode="json")
    payload["steps"] = [
        {
            "plan_step_id": step.plan_step_id,
            "title": step.title,
            "method": step.method,
            "status": step.status,
            "evidence_execution_ids": step.evidence_execution_ids[:12],
            "figure_requirement_keys": step.figure_requirement_keys[:12],
        }
        for step in inventory.steps[:64]
    ]
    payload["adata_source_preparation_ids"] = inventory.adata_source_preparation_ids[:16]
    payload["observed_adata_structure"] = {
        str(key): list(values)[:40]
        for key, values in list(inventory.observed_adata_structure.items())[:8]
    }
    return payload


def render_innovation_scope_context(
    state: Any,
    *,
    plan: Any = None,
    phase: Optional[str] = None,
) -> str:
    """Render the single bounded scope/inventory block shared by all planners."""
    if not is_innovation_scope(state, plan, phase=phase):
        return ""
    inventory = resolve_foundation_inventory(state, reconstruct=True)
    inventory_text = (
        json.dumps(
            _bounded_inventory_projection(inventory),
            ensure_ascii=False,
            sort_keys=True,
        )[:9_000]
        if inventory is not None
        else "No durable accepted Round 1 foundation inventory is currently available."
    )
    from .figure_followup import pending_figure_analysis_context

    pending_analysis_context = pending_figure_analysis_context(state)
    return f"""
## Innovation Scope Contract v{SCOPE_CONTRACT_VERSION}
Use the accepted Round 1 foundation below as prior input, not as default new work.
Accepted full-cohort loading, QC, doublet handling, normalization, dimensionality
reduction, standard clustering, marker calling, and annotation are not repeated by
default. Targeted subset work is legal; declare cohort_scope="subset" and state
the defined subset in the step. Sensitivity work should reuse existing measured
values. Declare relationship="revise" rather than silently replacing a prior
foundation decision, with requested_by="user" or "pi", a concrete reason, and
exact foundation references. New downstream work may declare relationship="new".
Current-round figure requirements describe current-round outputs only; do not
redeclare an earlier foundation figure merely to make it look current. Method
overlap alone is not a Host rejection criterion. Return scope_contract_version=1,
foundation_inventory_id, and analysis_scope for every step.

Accepted foundation inventory (Host-observed technical identity only):
{inventory_text}

Each analysis_scope must use relationship new|reuse|revise, cohort_scope full|subset,
foundation_refs (exact inventory_id, round_number, plan_id, plan_step_ids,
execution_ids, optional checkpoint_id), reason, and optional requested_by.
""".strip() + pending_analysis_context


def _normalize_reference(reference: Any) -> dict[str, Any]:
    if hasattr(reference, "model_dump"):
        return reference.model_dump(mode="python")
    if isinstance(reference, dict):
        return reference
    raise PlanScopeValidationError("foundation reference must be an object")


def _validate_reference(
    reference: Any,
    inventory: AcceptedFoundationInventory,
) -> None:
    data = _normalize_reference(reference)
    if str(data.get("inventory_id") or "") != inventory.inventory_id:
        raise PlanScopeValidationError("foundation reference names an unknown inventory_id")
    if int(data.get("round_number", 0) or 0) != inventory.round_number:
        raise PlanScopeValidationError("foundation reference round_number does not match inventory")
    if str(data.get("plan_id") or "") != inventory.plan_id:
        raise PlanScopeValidationError("foundation reference plan_id does not match inventory")

    inventory_steps = {step.plan_step_id: step for step in inventory.steps}
    step_ids = [int(value) for value in (data.get("plan_step_ids") or [])]
    execution_ids = [str(value) for value in (data.get("execution_ids") or [])]
    if len(step_ids) != len(set(step_ids)):
        raise PlanScopeValidationError("foundation reference has duplicate plan_step_ids")
    if len(execution_ids) != len(set(execution_ids)):
        raise PlanScopeValidationError("foundation reference has duplicate execution_ids")
    unknown_steps = sorted(set(step_ids) - set(inventory_steps))
    if unknown_steps:
        raise PlanScopeValidationError(
            f"foundation reference names unknown plan_step_ids: {unknown_steps}"
        )
    available_execution_ids = {
        value
        for step in inventory.steps
        for value in step.evidence_execution_ids
    }
    unknown_executions = sorted(set(execution_ids) - available_execution_ids)
    if unknown_executions:
        raise PlanScopeValidationError(
            "foundation reference names execution_ids not retained by the inventory"
        )
    if step_ids and execution_ids:
        selected_execution_ids = {
            value
            for step_id in step_ids
            for value in inventory_steps[step_id].evidence_execution_ids
        }
        if not set(execution_ids).issubset(selected_execution_ids):
            raise PlanScopeValidationError(
                "foundation execution_ids do not belong to the referenced plan_step_ids"
            )
    checkpoint_id = data.get("checkpoint_id")
    if checkpoint_id is not None and str(checkpoint_id) != str(
        inventory.adata_checkpoint_id or ""
    ):
        raise PlanScopeValidationError("foundation checkpoint_id does not match inventory")
    if not step_ids and not execution_ids and not checkpoint_id:
        raise PlanScopeValidationError(
            "foundation reference must identify a plan step, execution, or checkpoint"
        )


def validate_innovation_plan_scope(
    state: Any,
    plan: Any,
    *,
    allow_legacy_approved: bool = False,
) -> None:
    """Validate structured declarations and exact references, never semantics."""
    if not is_innovation_scope(state, plan):
        return
    if allow_legacy_approved and getattr(plan, "scope_contract_version", None) is None:
        return
    if getattr(plan, "scope_contract_version", None) != SCOPE_CONTRACT_VERSION:
        raise PlanScopeValidationError(
            "Innovation plan must use scope_contract_version=1 before approval"
        )
    inventory = resolve_foundation_inventory(state, reconstruct=True)
    if inventory is None:
        raise PlanScopeValidationError(
            "Accepted Round 1 foundation identity is unavailable; reconstruct it from "
            "an approved retained plan and successful receipts before approving this plan"
        )
    if str(getattr(plan, "foundation_inventory_id", "") or "") != inventory.inventory_id:
        raise PlanScopeValidationError(
            "Innovation plan foundation_inventory_id does not resolve in the current lineage"
        )
    if (inventory.lineage_id or "main") != current_lineage_id(state):
        raise PlanScopeValidationError(
            "Accepted foundation inventory belongs to another lineage"
        )

    for step in getattr(plan, "steps", None) or []:
        scope = getattr(step, "analysis_scope", None)
        if scope is None:
            raise PlanScopeValidationError(
                f"Innovation step {getattr(step, 'id', '?')} is missing analysis_scope"
            )
        refs = list(getattr(scope, "foundation_refs", None) or [])
        if len(refs) > _MAX_FOUNDATION_REFERENCES:
            raise PlanScopeValidationError("analysis_scope has too many foundation references")
        ref_keys = []
        for reference in refs:
            data = _normalize_reference(reference)
            key = (
                str(data.get("inventory_id") or ""),
                str(data.get("plan_id") or ""),
                tuple(data.get("plan_step_ids") or []),
                tuple(data.get("execution_ids") or []),
                str(data.get("checkpoint_id") or ""),
            )
            ref_keys.append(key)
            _validate_reference(reference, inventory)
        if len(ref_keys) != len(set(ref_keys)):
            raise PlanScopeValidationError("analysis_scope has duplicate foundation references")
        relationship = str(getattr(scope, "relationship", "") or "")
        if relationship in {"reuse", "revise"} and not refs:
            raise PlanScopeValidationError(
                f"Innovation step {getattr(step, 'id', '?')} declares {relationship} "
                "without an exact foundation reference"
            )
        if relationship == "revise":
            if getattr(scope, "requested_by", None) not in {"user", "pi"}:
                raise PlanScopeValidationError(
                    f"Innovation step {getattr(step, 'id', '?')} revises a foundation "
                    "without requested_by='user' or 'pi'"
                )
            if not str(getattr(scope, "reason", "") or "").strip():
                raise PlanScopeValidationError(
                    f"Innovation step {getattr(step, 'id', '?')} revises a foundation "
                    "without a reason"
                )


def innovation_inventory_id(state: Any, plan: Any = None) -> Optional[str]:
    inventory = resolve_foundation_inventory(state, reconstruct=True)
    return inventory.inventory_id if inventory is not None else None
