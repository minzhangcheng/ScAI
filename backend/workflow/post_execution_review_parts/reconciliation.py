"""Current execution-evidence linkage and legacy reconciliation helpers."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

from ...services.execution_evidence import execution_is_current_receipt
from ...services.report.artifacts import (
    eligible_artifacts,
    verified_artifact_bytes,
)

def _identity_context(analysis_state, plan) -> Dict[str, Any]:
    # Generation/cohort values remain observations on receipts and artifacts;
    # they are not active-value equality gates for current-plan progress.
    return {
        "round_number": int(getattr(analysis_state, "current_round", 0) or 0),
        "lineage_id": getattr(analysis_state, "restore_branch_id", None) or "main",
        "plan_id": str(getattr(plan, "id", "") or ""),
        "adata_source_preparation_id": (
            str(getattr(plan, "adata_source_preparation_id", "") or "") or None
        ),
    }


def _matches_current_identity(
    item: Any,
    identity: Dict[str, Any],
    *,
    require_plan: bool = True,
) -> bool:
    if int(getattr(item, "round_number", 0) or 0) != identity["round_number"]:
        return False
    if (getattr(item, "lineage_id", None) or "main") != identity["lineage_id"]:
        return False
    if require_plan and str(getattr(item, "plan_id", "") or "") != identity["plan_id"]:
        return False
    return True


def _collect_current_evidence(
    analysis_state,
    result: Dict[str, Any],
    identity: Dict[str, Any],
) -> Dict[str, list[Any]]:
    executions = [
        item for item in (getattr(analysis_state, "code_history", None) or [])
        if execution_is_current_receipt(
            item,
            round_number=identity["round_number"],
            plan_id=identity["plan_id"],
            lineage_id=identity["lineage_id"],
        )
        and (
            identity.get("adata_source_preparation_id") is None
            or str(
                getattr(item, "adata_source_preparation_id", "") or ""
            ) == identity["adata_source_preparation_id"]
        )
    ]
    figures = [
        item for item in eligible_artifacts(
            getattr(analysis_state, "publication_figures", None) or []
        )
        if _matches_current_identity(item, identity)
    ][-30:]
    execution_ids = {item.id for item in executions}
    tables = [
        item
        for item in (getattr(analysis_state, "publication_tables", None) or [])
        if _matches_current_identity(item, identity)
        and str(getattr(item, "producer_status", "published")) == "published"
        and bool(getattr(item, "is_current", True))
        and getattr(item, "invalidated_at", None) is None
        and str(getattr(item, "source_execution_id", "") or "") in execution_ids
        and verified_artifact_bytes(
            Path(str(getattr(item, "csv_path", "") or "")),
            item,
            kind="table",
        ) is not None
    ][-30:]

    finding_records = []
    for raw in (result.get("finding_records") or [])[-40:]:
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        record = raw
        record["id"] = str(record.get("id") or uuid.uuid4())
        if int(record.get("round_number", identity["round_number"]) or 0) != identity["round_number"]:
            continue
        if str(record.get("lineage_id") or identity["lineage_id"]) != identity["lineage_id"]:
            continue
        if str(record.get("plan_id") or identity["plan_id"]) != identity["plan_id"]:
            continue
        finding_records.append(record)
    return {
        "executions": executions,
        "figures": figures,
        "tables": tables,
        "finding_records": finding_records,
    }


def _build_catalog(
    analysis_state,
    plan,
    identity: Dict[str, Any],
    evidence: Dict[str, list[Any]],
) -> Dict[str, Any]:
    executions = evidence["executions"]
    finding_records = evidence["finding_records"]
    figures = evidence["figures"]
    tables = evidence["tables"]
    return {
        "session_id": str(getattr(analysis_state, "id", "") or ""),
        **identity,
        "plan_steps": [
            {
                "id": int(step.id),
                "title": str(step.title)[:500],
                "method": str(step.method)[:1500],
                "input": str(step.input)[:1000],
                "expected_output": str(step.expected_output)[:1500],
                "rationale": str(step.rationale)[:1500],
                "figure_requirements": [
                    requirement.model_dump(mode="json")
                    for requirement in (
                        getattr(step, "figure_requirements", None) or []
                    )
                ],
            }
            for step in plan.steps
        ],
        "executions": [
            {
                "id": item.id,
                "description": next((
                    str(getattr(step, "code_description", "") or "")
                    for step in reversed(
                        getattr(analysis_state, "analysis_log", None) or []
                    )
                    if getattr(step, "execution_id", None) == item.id
                ), "")[:1000],
                "code": str(getattr(item, "code", "") or "")[:2500],
                "output": str(getattr(item, "output", "") or "")[-2500:],
                "candidate_plan_step_ids": list(
                    getattr(item, "plan_step_ids", None) or []
                ),
                "execution_purpose": getattr(item, "execution_purpose", None),
                "persistent_outputs": list(
                    getattr(item, "realized_persistent_outputs", None) or []
                ),
            }
            for item in executions
        ],
        "findings": [
            {
                "id": record["id"],
                "text": str(record.get("text") or "")[:2000],
                "modality": record.get("modality"),
                "candidate_execution_ids": list(
                    record.get("candidate_evidence_execution_ids")
                    or record.get("evidence_execution_ids")
                    or []
                )[:40],
            }
            for record in finding_records
        ],
        "figures": [
            {
                "id": item.id,
                "name": str(item.name)[:300],
                "description": str(item.description)[:2000],
                "figure_type": str(item.figure_type)[:200],
                "source_execution_id": item.source_execution_id,
                "submitted_plan_step_id": item.submitted_plan_step_id,
                "plan_requirement_key": item.plan_requirement_key,
            }
            for item in figures
        ],
        "tables": [
            {
                "id": item.id,
                "name": str(item.name)[:300],
                "description": str(item.description)[:2000],
                "columns": list(item.columns)[:80],
                "source_execution_id": item.source_execution_id,
                "candidate_plan_step_ids": list(item.plan_step_ids),
            }
            for item in tables
        ],
    }


def _fallback_decision(plan, error: Exception) -> Dict[str, Any]:
    reason = f"Supervisor reconciliation unavailable: {error}"
    return {
        "links": [
            {
                "plan_step_id": int(step.id),
                "status": "unverified",
                "execution_ids": [],
                "finding_ids": [],
                "figure_ids": [],
                "table_ids": [],
                "reason": reason,
            }
            for step in plan.steps
        ],
        "overall_reason": reason,
        "parse_failed": True,
    }


def _normalize_links(
    decision: Dict[str, Any],
    plan,
    evidence: Dict[str, list[Any]],
) -> list[Dict[str, Any]]:
    valid_ids = {
        "execution_ids": {item.id for item in evidence["executions"]},
        "finding_ids": {item["id"] for item in evidence["finding_records"]},
        "figure_ids": {item.id for item in evidence["figures"]},
        "table_ids": {item.id for item in evidence["tables"]},
    }
    artifact_sources = {
        "figure_ids": {
            item.id: str(getattr(item, "source_execution_id", "") or "")
            for item in evidence["figures"]
        },
        "table_ids": {
            item.id: str(getattr(item, "source_execution_id", "") or "")
            for item in evidence["tables"]
        },
    }
    valid_steps = {int(step.id) for step in plan.steps}
    links = []
    for raw in decision.get("links") or []:
        if not isinstance(raw, dict):
            continue
        try:
            step_id = int(raw.get("plan_step_id"))
        except (TypeError, ValueError):
            continue
        if step_id not in valid_steps:
            continue
        status = str(raw.get("status") or "unverified")
        if status not in {"evidenced", "partial", "unverified"}:
            status = "unverified"
        link = {
            "plan_step_id": step_id,
            "status": status,
            "reason": str(raw.get("reason") or "")[:2000],
        }
        for field_name, allowed in valid_ids.items():
            link[field_name] = [
                str(value) for value in (raw.get(field_name) or [])
                if str(value) in allowed
            ][:40]
        linked_sources = [
            artifact_sources[field_name].get(artifact_id, "")
            for field_name in ("figure_ids", "table_ids")
            for artifact_id in link[field_name]
        ]
        link["execution_ids"] = list(dict.fromkeys(
            link["execution_ids"]
            + [
                source_id for source_id in linked_sources
                if source_id in valid_ids["execution_ids"]
            ]
        ))[:40]
        objective_ids = (
            link["execution_ids"] + link["figure_ids"] + link["table_ids"]
        )
        if status == "evidenced" and not link["execution_ids"]:
            link.update({
                "status": "unverified",
                "finding_ids": [],
                "reason": (
                    "No valid current execution receipt remained after host "
                    "validation; a finding or artifact without its producer "
                    "receipt cannot establish complete plan evidence."
                ),
            })
        elif status != "unverified" and not objective_ids:
            link.update({
                "status": "unverified",
                "finding_ids": [],
                "reason": (
                    "No valid current execution or registered artifact identity "
                    "remained after host validation; finding prose alone cannot "
                    "establish plan evidence."
                ),
            })
        links.append(link)
    linked_steps = {item["plan_step_id"] for item in links}
    for step_id in sorted(valid_steps - linked_steps):
        links.append({
            "plan_step_id": step_id,
            "status": "unverified",
            "execution_ids": [],
            "finding_ids": [],
            "figure_ids": [],
            "table_ids": [],
            "reason": "No Supervisor semantic decision was available.",
        })
    return links


def _apply_links(
    analysis_state,
    result: Dict[str, Any],
    evidence: Dict[str, list[Any]],
    links: list[Dict[str, Any]],
    plan_id: str,
) -> None:
    finding_records = evidence["finding_records"]
    finding_by_id = {item["id"]: item for item in finding_records}
    table_by_id = {item.id: item for item in evidence["tables"]}
    for record in finding_records:
        candidates = list(
            record.get("candidate_evidence_execution_ids")
            or record.get("evidence_execution_ids")
            or []
        )
        record.update({
            "candidate_evidence_execution_ids": candidates[:40],
            "evidence_execution_ids": [],
            "evidence_status": "unverified",
            "confirmed_plan_step_ids": [],
            "semantic_status": "unverified",
        })
    for link in links:
        for finding_id in link["finding_ids"]:
            record = finding_by_id.get(finding_id)
            if record is None:
                continue
            if link["status"] == "evidenced":
                record["confirmed_plan_step_ids"] = sorted(set(
                    list(record.get("confirmed_plan_step_ids") or [])
                    + [link["plan_step_id"]]
                ))
                record["semantic_status"] = "evidenced"
                if link["execution_ids"]:
                    record["evidence_execution_ids"] = sorted(set(
                        list(record.get("evidence_execution_ids") or [])
                        + list(link["execution_ids"])
                    ))[:40]
                    record["evidence_status"] = "verified"
            elif record.get("semantic_status") != "evidenced":
                record["semantic_status"] = "partial"
        for table_id in link["table_ids"]:
            table = table_by_id.get(table_id)
            if table is not None and link["status"] == "evidenced":
                table.plan_step_ids = sorted(set(
                    list(table.plan_step_ids) + [link["plan_step_id"]]
                ))
                table.plan_binding_status = "confirmed"
                table.plan_binding_reason = link["reason"]

    verified_ids = sorted({
        str(value)
        for record in finding_records
        if record.get("evidence_status") == "verified"
        and record.get("semantic_status") == "evidenced"
        for value in (record.get("evidence_execution_ids") or [])
        if str(value)
    })
    result["evidence_execution_ids"] = verified_ids
    result["candidate_evidence_execution_ids"] = sorted({
        str(value)
        for record in finding_records
        for value in (record.get("candidate_evidence_execution_ids") or [])
        if str(value)
    })
    _sync_analysis_log(analysis_state, finding_records)

    records_verified = bool(finding_records) and all(
        item.get("evidence_status") == "verified"
        and item.get("semantic_status") == "evidenced"
        for item in finding_records
    )
    result["evidence_verified"] = records_verified
    gates = [
        item for item in (result.get("evidence_gate_findings") or [])
        if not (
            isinstance(item, dict)
            and item.get("gate") == "supervisor-plan-evidence"
        )
    ]
    if finding_records and not records_verified:
        gates.append({
            "gate": "supervisor-plan-evidence",
            "rule_id": "scai-native/supervisor-plan-evidence",
            "detail": (
                "One or more findings could not be semantically reconciled "
                "to current execution evidence. They remain retained as "
                "unverified observations for PI and report disclosure."
            ),
            "subject": plan_id,
            "severity": "advise",
            "rejection_count": 0,
            "rejection_limit": 0,
            "forced_release": False,
        })
    result["evidence_gate_findings"] = gates
    # Advisory reconciliation observations stay visible to the PI without
    # changing workflow status. Preserve only an already-recorded operational
    # degradation (for example a Supervisor force-stop) or a forced release.
    result["evidence_gate_degraded"] = bool(
        result.get("evidence_gate_degraded")
        or any(
            item.get("forced_release")
            for item in gates
            if isinstance(item, dict)
        )
    )


def _apply_execution_links(
    analysis_state,
    result: Dict[str, Any],
    evidence: Dict[str, list[Any]],
    plan_id: str,
) -> None:
    """Verify finding-to-execution identity without making a scientific verdict."""
    finding_records = evidence["finding_records"]
    eligible_ids = {
        str(item.id) for item in evidence["executions"] if str(item.id)
    }
    source_records = [
        item for item in (result.get("finding_records") or [])[-40:]
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]

    for record in finding_records:
        nominated = {
            str(value)
            for value in (
                record.get("candidate_evidence_execution_ids")
                or record.get("evidence_execution_ids")
                or []
            )
            if str(value)
        }
        linked_ids = sorted(nominated & eligible_ids)[:40]
        record.update({
            "candidate_evidence_execution_ids": linked_ids,
            "evidence_execution_ids": linked_ids,
            "evidence_status": "verified" if linked_ids else "unverified",
            # Execution identity is objective. Whether the execution scientifically
            # completes a plan step remains a PI decision.
            "confirmed_plan_step_ids": [],
            "semantic_status": "unverified",
        })

    verified_ids = sorted({
        str(value)
        for record in finding_records
        if record.get("evidence_status") == "verified"
        for value in (record.get("evidence_execution_ids") or [])
        if str(value)
    })
    result["evidence_execution_ids"] = verified_ids
    result["candidate_evidence_execution_ids"] = sorted({
        str(value)
        for record in finding_records
        for value in (record.get("candidate_evidence_execution_ids") or [])
        if str(value)
    })
    _sync_analysis_log(analysis_state, finding_records)

    records_verified = None if not source_records else (
        len(finding_records) == len(source_records)
        and all(
            item.get("evidence_status") == "verified"
            and bool(item.get("evidence_execution_ids"))
            for item in finding_records
        )
    )
    result["evidence_verified"] = records_verified

    gates = [
        item for item in (result.get("evidence_gate_findings") or [])
        if not (
            isinstance(item, dict)
            and item.get("gate") in {
                "supervisor-plan-evidence",
                "execution-evidence-linkage",
            }
        )
    ]
    if source_records and records_verified is False:
        gates.append({
            "gate": "execution-evidence-linkage",
            "rule_id": "scai-native/execution-evidence-linkage",
            "detail": (
                "One or more findings lacked a valid current successful execution "
                "receipt. They remain retained as technically unlinked observations "
                "for PI review."
            ),
            "subject": plan_id,
            "severity": "advise",
            "rejection_count": 0,
            "rejection_limit": 0,
            "forced_release": False,
        })
    result["evidence_gate_findings"] = gates
    result["evidence_gate_degraded"] = bool(
        result.get("evidence_gate_degraded")
        or any(
            item.get("forced_release")
            for item in gates
            if isinstance(item, dict)
        )
    )


def _sync_analysis_log(analysis_state, finding_records: list[dict]) -> None:
    reconciled = {
        str(record.get("id") or ""): record for record in finding_records
    }
    for step in getattr(analysis_state, "analysis_log", None) or []:
        if getattr(step, "step_type", "") != "report_findings":
            continue
        updated_records = [
            reconciled.get(str(raw.get("id") or ""), raw)
            for raw in (getattr(step, "finding_records", None) or [])
            if isinstance(raw, dict)
        ]
        if not updated_records:
            continue
        step.finding_records = updated_records
        step.evidence_execution_ids = sorted({
            str(value)
            for record in updated_records
            if record.get("evidence_status") == "verified"
            for value in (record.get("evidence_execution_ids") or [])
            if str(value)
        })


async def reconcile_plan_evidence(
    workflow,
    analysis_state,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind findings to current successful receipts without an LLM verdict."""
    plan = getattr(analysis_state, "current_plan", None)
    if plan is None or not getattr(plan, "steps", None):
        return {}
    identity = _identity_context(analysis_state, plan)
    evidence = _collect_current_evidence(analysis_state, result, identity)
    _apply_execution_links(
        analysis_state,
        result,
        evidence,
        identity["plan_id"],
    )
    return {}
