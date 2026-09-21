"""One deterministic, path-free AnnData checkpoint projection for agents."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 50


def _entry_from_checkpoint(
    record: Any,
    *,
    current_head: bool = False,
    selected: bool = False,
) -> Dict[str, Any]:
    return {
        "entry_kind": "current" if current_head else "checkpoint",
        "selected_by_current_plan": selected,
        "checkpoint_id": str(getattr(record, "checkpoint_id", "") or ""),
        "availability": str(getattr(record, "status", "") or "unavailable"),
        "boundary": str(getattr(record, "boundary", "") or ""),
        "round": int(getattr(record, "round_number", 0) or 0),
        "planning_stage": str(getattr(record, "planning_stage", "") or ""),
        "revision": int(getattr(record, "boundary_revision", 0) or 0),
        "phase": str(getattr(record, "phase", "") or ""),
        "direction": str(getattr(record, "direction", "") or ""),
        "parent_checkpoint_id": getattr(record, "parent_checkpoint_id", None),
        "supersedes_checkpoint_id": getattr(record, "supersedes_checkpoint_id", None),
        "n_obs": getattr(record, "n_obs", None),
        "n_vars": getattr(record, "n_vars", None),
        "same_as_parent": bool(getattr(record, "same_as_parent", False)),
        "research_label": str(getattr(record, "research_label", "") or "")[:240],
        "summary_at_capture": str(getattr(record, "summary_at_capture", "") or "")[:600],
        "outcome_at_capture": str(getattr(record, "outcome_at_capture", "") or ""),
        "execution_ids": list(getattr(record, "execution_ids_at_capture", None) or [])[:20],
        "source_preparation_ids": list(
            getattr(record, "source_preparation_ids_at_capture", None) or []
        )[:20],
        "supplementary_request_id": getattr(
            record, "supplementary_request_id", None
        ),
        "error_summary": str(getattr(record, "error_summary", "") or "")[:500] or None,
        "unavailable_reason": (
            str(getattr(record, "unavailable_reason", "") or "")[:300] or None
        ),
    }


def _entry_from_binding(record: Any, *, selected: bool = False) -> Dict[str, Any]:
    return {
        "entry_kind": "live_binding",
        "selected_by_current_plan": selected,
        "live_binding_id": str(getattr(record, "binding_id", "") or ""),
        "binding_name": str(getattr(record, "name", "") or ""),
        "availability": (
            "available"
            if str(getattr(record, "status", "")) == "registered"
            else str(getattr(record, "status", ""))
        ),
        "adata_lineage_id": str(getattr(record, "adata_lineage_id", "") or ""),
        "created_from_checkpoint_id": getattr(record, "created_from_checkpoint_id", None),
        "ancestor_checkpoint_id": getattr(record, "ancestor_checkpoint_id", None),
        "mode_created": str(getattr(record, "mode_created", "") or ""),
        "identity_status": str(getattr(record, "status", "") or ""),
    }


def _ordered_entries(
    state: Any,
    *,
    checkpoint_id: Optional[str] = None,
    live_binding_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    checkpoints = list(getattr(state, "adata_checkpoints", None) or [])
    by_id = {str(getattr(r, "checkpoint_id", "")): r for r in checkpoints}

    if checkpoint_id:
        record = by_id.get(str(checkpoint_id))
        if record is not None:
            entries.append(_entry_from_checkpoint(record))
    if live_binding_id:
        for binding in getattr(state, "adata_live_bindings", None) or []:
            if str(getattr(binding, "binding_id", "")) == str(live_binding_id):
                entries.append(_entry_from_binding(binding))
                break
    if checkpoint_id or live_binding_id:
        return entries

    # Current head is always first, even when no checkpoint exists yet.
    current_id = str(getattr(state, "active_adata_checkpoint_id", "") or "")
    current_record = by_id.get(current_id)
    if current_record is not None:
        entries.append(_entry_from_checkpoint(current_record, current_head=True))
    else:
        entries.append(
            {
                "entry_kind": "current",
                "checkpoint_id": None,
                "availability": "requires_runtime_verification",
                "research_label": "",
                "summary_at_capture": "No checkpoint ancestor recorded; live `adata` presence is unverified.",
            }
        )

    # Keep an explicit plan selection ahead of entries that may be trimmed.
    selection = getattr(getattr(state, "current_plan", None), "adata_source", None)
    selected_checkpoint_id = str(
        getattr(selection, "checkpoint_id", "") or ""
    )
    selected_live_binding_id = str(
        getattr(selection, "live_binding_id", "") or ""
    )
    if selected_checkpoint_id and selected_checkpoint_id in by_id:
        entries.append(
            _entry_from_checkpoint(
                by_id[selected_checkpoint_id],
                selected=True,
            )
        )
    if selected_live_binding_id:
        for binding in getattr(state, "adata_live_bindings", None) or []:
            if str(getattr(binding, "binding_id", "")) != selected_live_binding_id:
                continue
            if str(getattr(binding, "status", "")) in {
                "registered",
                "pending_reconcile",
            }:
                entries.append(_entry_from_binding(binding, selected=True))
            break

    # Pinned Stage 1 baseline, then the latest available record per round.
    stage1 = [
        record for record in checkpoints
        if str(getattr(record, "boundary", "")) == "stage1_complete"
        and str(getattr(record, "status", "")) == "available"
    ]
    if stage1:
        entries.append(_entry_from_checkpoint(stage1[0]))
    latest_per_round: Dict[int, Any] = {}
    for record in checkpoints:
        if str(getattr(record, "status", "")) != "available":
            continue
        if str(getattr(record, "boundary", "")) not in {
            "round_complete",
            "supplementary_complete",
        }:
            continue
        round_number = int(getattr(record, "round_number", 0) or 0)
        existing = latest_per_round.get(round_number)
        def _head_order(item):
            return (
                int(getattr(item, "boundary_revision", 0) or 0),
                str(getattr(item, "created_at", "") or ""),
                str(getattr(item, "checkpoint_id", "") or ""),
            )
        if existing is None or _head_order(record) >= _head_order(existing):
            latest_per_round[round_number] = record
    for round_number in sorted(latest_per_round, reverse=True):
        entries.append(_entry_from_checkpoint(latest_per_round[round_number]))

    for binding in getattr(state, "adata_live_bindings", None) or []:
        status = str(getattr(binding, "status", ""))
        if status == "registered":
            entries.append(_entry_from_binding(binding))
        elif status == "pending_reconcile":
            entry = _entry_from_binding(binding)
            entry["availability"] = "requires_runtime_verification"
            entries.append(entry)

    # De-duplicate by opaque identity while preserving order.
    seen = set()
    unique: List[Dict[str, Any]] = []
    for entry in entries:
        key = (
            entry.get("entry_kind"),
            entry.get("checkpoint_id") or entry.get("live_binding_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def build_checkpoint_projection(
    state: Any,
    *,
    page: int = 1,
    page_size: int = _DEFAULT_PAGE_SIZE,
    checkpoint_id: Optional[str] = None,
    live_binding_id: Optional[str] = None,
) -> Dict[str, Any]:
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or _DEFAULT_PAGE_SIZE), _MAX_PAGE_SIZE))
    entries = _ordered_entries(
        state,
        checkpoint_id=checkpoint_id,
        live_binding_id=live_binding_id,
    )
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = entries[start:end]
    return {
        "page": page,
        "page_size": page_size,
        "total": len(entries),
        "has_more": end < len(entries),
        "entries": page_entries,
        "omitted_count": max(0, len(entries) - end),
        "note": (
            "Use page/page_size or an exact checkpoint_id/live_binding_id to "
            "retrieve omitted choices."
            if end < len(entries) or start > 0
            else None
        ),
    }


def build_prompt_catalog_text(state: Any, *, max_entries: int = 12) -> str:
    projection = build_checkpoint_projection(state, page_size=max_entries)
    if not projection["entries"]:
        return ""
    lines = ["## Available AnnData states"]
    for entry in projection["entries"]:
        if entry.get("entry_kind") == "live_binding":
            lines.append(
                f"- {entry.get('live_binding_id')}: retained working object "
                f"`{entry.get('binding_name')}` ({entry.get('availability')})"
            )
        elif entry.get("entry_kind") == "current":
            ancestor = entry.get("checkpoint_id") or "none"
            lines.append(
                f"- current: `adata` binding requires runtime verification; "
                f"last known checkpoint ancestor {ancestor}, not reloaded"
            )
        else:
            shape = (
                f"{entry.get('n_obs')} x {entry.get('n_vars')}"
                if entry.get("n_obs") is not None else "shape unavailable"
            )
            lines.append(
                f"- {entry.get('checkpoint_id')}: round {entry.get('round')} "
                f"{entry.get('boundary')} rev {entry.get('revision')}, {shape}; "
                f"label: {entry.get('research_label') or '(none)'}"
            )
    if projection.get("omitted_count"):
        lines.append(
            f"- {projection['omitted_count']} further choice(s) omitted; use the "
            "list_adata_checkpoints lookup for exact IDs."
        )
    return "\n".join(lines)
