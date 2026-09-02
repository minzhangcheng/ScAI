"""Supervisor-only semantic binding of proposed figures to plan requirements."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from ...constants import NO_OUTPUT_TOKEN_CAP
from ...services.figure_contract import (
    audit_figure_contract,
    collect_plan_figure_requirements,
)

logger = logging.getLogger(__name__)

_TOOL_NAME = "submit_figure_plan_binding"
_TOOL = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": "Submit one semantic figure-to-plan binding decision.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["accept", "rebind", "supplementary", "reject"],
                },
                "plan_requirement_key": {"type": ["string", "null"]},
                "reason": {"type": "string", "maxLength": 2000},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["decision", "plan_requirement_key", "reason"],
        },
    },
}

_PLAN_EVIDENCE_TOOL_NAME = "submit_plan_evidence_reconciliation"
_PLAN_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": _PLAN_EVIDENCE_TOOL_NAME,
        "description": "Submit semantic evidence links for the current approved plan.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "links": {
                    "type": "array",
                    "maxItems": 80,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "plan_step_id": {"type": "integer", "minimum": 1},
                            "status": {
                                "type": "string",
                                "enum": ["evidenced", "partial", "unverified"],
                            },
                            "execution_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                            "finding_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                            "figure_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                            "table_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                            "reason": {"type": "string", "maxLength": 2000},
                        },
                        "required": ["plan_step_id", "status", "reason"],
                    },
                },
                "overall_reason": {"type": "string", "maxLength": 2000},
            },
            "required": ["links", "overall_reason"],
        },
    },
}


def validate_figure_binding(value: Any) -> tuple[Dict[str, Any] | None, str]:
    if not isinstance(value, dict):
        return None, "decision must be an object"
    decision = str(value.get("decision") or "").strip().lower()
    if decision not in {"accept", "rebind", "supplementary", "reject"}:
        return None, "decision is outside the closed vocabulary"
    raw_key = value.get("plan_requirement_key")
    key = str(raw_key or "").strip() or None
    if decision in {"accept", "rebind"} and key is None:
        return None, f"{decision} requires plan_requirement_key"
    if decision in {"supplementary", "reject"} and key is not None:
        return None, f"{decision} must not claim plan_requirement_key"
    reason = str(value.get("reason") or "").strip()
    if not reason:
        return None, "reason is required"
    confidence = value.get("confidence")
    if confidence is not None:
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return None, "confidence must be numeric"
    return {
        "decision": decision,
        "plan_requirement_key": key,
        "reason": reason[:2000],
        "confidence": confidence,
    }, ""


def validate_plan_evidence(
    value: Any, catalog: Dict[str, Any]
) -> tuple[Dict[str, Any] | None, str]:
    """Validate only catalog identities; semantic judgment remains with the LLM."""
    if not isinstance(value, dict) or not isinstance(value.get("links"), list):
        return None, "links must be an array"
    offered = {
        "execution_ids": {str(item.get("id")) for item in catalog.get("executions", [])},
        "finding_ids": {str(item.get("id")) for item in catalog.get("findings", [])},
        "figure_ids": {str(item.get("id")) for item in catalog.get("figures", [])},
        "table_ids": {str(item.get("id")) for item in catalog.get("tables", [])},
    }
    step_ids = {int(item.get("id")) for item in catalog.get("plan_steps", [])}
    links = []
    seen_steps = set()
    for raw in value.get("links", [])[:80]:
        if not isinstance(raw, dict):
            return None, "each link must be an object"
        try:
            step_id = int(raw.get("plan_step_id"))
        except (TypeError, ValueError):
            return None, "plan_step_id must be an offered integer"
        if step_id not in step_ids or step_id in seen_steps:
            return None, "plan_step_id is unknown or duplicated"
        status = str(raw.get("status") or "").strip().lower()
        if status not in {"evidenced", "partial", "unverified"}:
            return None, "status is outside the closed vocabulary"
        reason = str(raw.get("reason") or "").strip()
        if not reason:
            return None, "reason is required"
        normalized = {
            "plan_step_id": step_id,
            "status": status,
            "reason": reason[:2000],
        }
        evidence_count = 0
        for field_name, valid_ids in offered.items():
            values = list(dict.fromkeys(
                str(item).strip() for item in (raw.get(field_name) or [])
                if str(item).strip()
            ))[:40]
            if any(item not in valid_ids for item in values):
                return None, f"{field_name} contains an identity outside the catalog"
            normalized[field_name] = values
            evidence_count += len(values)
        if status in {"evidenced", "partial"} and not evidence_count:
            return None, f"{status} requires at least one offered evidence record"
        if status == "unverified" and evidence_count:
            return None, "unverified must not claim evidence identities"
        seen_steps.add(step_id)
        links.append(normalized)
    for step_id in sorted(step_ids - seen_steps):
        links.append({
            "plan_step_id": step_id,
            "status": "unverified",
            "execution_ids": [],
            "finding_ids": [],
            "figure_ids": [],
            "table_ids": [],
            "reason": "Supervisor returned no semantic decision for this step.",
        })
    return {
        "links": links,
        "overall_reason": str(value.get("overall_reason") or "")[:2000],
    }, ""


class FigureBindingMixin:
    """LLM semantic preflight; objective identity is validated by the caller."""

    async def review_figure_plan_binding(
        self, state, proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        plan = getattr(state, "current_plan", None)
        requirements = collect_plan_figure_requirements(plan)
        audit = audit_figure_contract(
            state,
            plan=plan,
            planning_stage=str(getattr(plan, "stage", "full") or "full"),
            include_stage_baseline=False,
        )
        missing_keys = {item.requirement.key for item in audit.missing}
        materialized = [
            {
                "key": item.key,
                "plan_id": item.plan_id,
                "plan_step_id": item.plan_step_id,
                "figure_type": item.figure_type,
                "description": item.description,
                "minimum_count": item.minimum_count,
                "satisfied": item.key not in missing_keys,
            }
            for item in requirements
        ]
        payload = {
            "round_number": int(getattr(state, "current_round", 0) or 0),
            "analysis_phase": str(getattr(state, "analysis_phase", "") or ""),
            "plan_id": str(getattr(plan, "id", "") or "") or None,
            "requirements": materialized,
            "proposal": {
                "figure_name": str(proposal.get("figure_name") or "")[:240],
                "figure_type": str(proposal.get("figure_type") or "")[:120],
                "description": str(proposal.get("description") or "")[:2000],
                "purpose": str(proposal.get("purpose") or "result")[:80],
                "submitted_plan_step_id": proposal.get("plan_step_id"),
                "code_summary": str(proposal.get("code") or "")[:2500],
            },
        }
        system = (
            "You are the semantic figure-plan binding reviewer. Judge whether the "
            "proposed scientific figure actually satisfies one exact materialized "
            "visual requirement. Use scientific meaning, not filename, token, or "
            "keyword matching. Accept the submitted binding only when correct; "
            "rebind only when exactly one different unmet requirement is a clear "
            "match; mark useful out-of-plan work supplementary; reject ambiguous "
            "or unsupported claims. This is not visual-quality review."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = "no structured decision"
        for _ in range(2):
            self.refresh_runtime_config_if_needed()
            self._set_llm_context()
            response = await self._budgeted_text_call(
                self.llm_client.chat_with_tools,
                model=self.model,
                messages=messages,
                tools=[_TOOL],
                tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
                temperature=self.temperature,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
            )
            for call in response.tool_calls:
                if call.name != _TOOL_NAME or call.parse_error:
                    continue
                validated, error = validate_figure_binding(call.arguments)
                if validated is not None:
                    return validated
                last_error = error
            messages.append({
                "role": "user",
                "content": f"The decision was invalid: {last_error}. Submit one valid tool call.",
            })
        logger.warning("Supervisor figure binding failed: %s", last_error)
        return {
            "decision": "reject",
            "plan_requirement_key": None,
            "reason": f"Supervisor binding unavailable: {last_error}",
            "confidence": None,
            "parse_failed": True,
        }

    async def review_plan_evidence(
        self, state, catalog: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Semantically reconcile current evidence without blocking the workflow."""
        system = (
            "You are the Supervisor performing semantic plan-evidence reconciliation. "
            "For every approved plan step, decide whether the offered executions and "
            "registered outputs scientifically perform that step. Judge meaning and "
            "scientific function; do not use filename, token, substring, or keyword "
            "matching. Producer-nominated step IDs are hints, never authority. Mark a "
            "step evidenced only when the offered records substantively complete it, "
            "partial when they perform only part of it, and unverified when evidence is "
            "insufficient or ambiguous. When linking a finding, include the execution "
            "ID or registered artifact that scientifically supports it in the same "
            "step link; finding prose alone is never evidence. Never invent an ID. "
            "This review assigns plan credit only; it does not reject or delete work."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(catalog, ensure_ascii=False)},
        ]
        last_error = "no structured reconciliation"
        for _ in range(2):
            self.refresh_runtime_config_if_needed()
            self._set_llm_context()
            response = await self._budgeted_text_call(
                self.llm_client.chat_with_tools,
                model=self.model,
                messages=messages,
                tools=[_PLAN_EVIDENCE_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": _PLAN_EVIDENCE_TOOL_NAME},
                },
                temperature=self.temperature,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
            )
            for call in response.tool_calls:
                if call.name != _PLAN_EVIDENCE_TOOL_NAME or call.parse_error:
                    continue
                validated, error = validate_plan_evidence(call.arguments, catalog)
                if validated is not None:
                    return validated
                last_error = error
            messages.append({
                "role": "user",
                "content": (
                    f"The reconciliation was invalid: {last_error}. Submit one valid "
                    "tool call using only offered IDs."
                ),
            })
        logger.warning("Supervisor plan evidence reconciliation failed: %s", last_error)
        return {
            "links": [
                {
                    "plan_step_id": int(item.get("id")),
                    "status": "unverified",
                    "execution_ids": [],
                    "finding_ids": [],
                    "figure_ids": [],
                    "table_ids": [],
                    "reason": f"Supervisor reconciliation unavailable: {last_error}",
                }
                for item in catalog.get("plan_steps", [])
            ],
            "overall_reason": f"Supervisor reconciliation unavailable: {last_error}",
            "parse_failed": True,
        }
