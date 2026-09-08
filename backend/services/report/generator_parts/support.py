"""Pure report completion, coverage, and appendix helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models.schemas import AnalysisState
from backend.services.report.quality import build_report_partial_fields
from backend.services.report.report_outline import build_report_outline

logger = logging.getLogger(__name__)


def report_completion_fields(
    state: AnalysisState,
    unreported_figures: Optional[List[str]] = None,
    figureless_sections: Optional[List[str]] = None,
    quality_disclosures: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    """Status fields for the non-Supervisor report path.

    Partial-report approval is a property of the report being generated, not a
    transient workflow notification. Persist it in ``report.json`` so reloads
    and exports cannot silently relabel it as complete.
    """
    status, current_step, _primary, _reasons = build_report_partial_fields(
        unreported_figures=unreported_figures,
        figureless_sections=figureless_sections,
        approved_partial=bool(getattr(state, "allow_partial_report", False)),
        quality_disclosures=quality_disclosures,
    )
    return status, current_step


def report_partial_reason(
    state: AnalysisState,
    unreported_figures: Optional[List[str]] = None,
    figureless_sections: Optional[List[str]] = None,
    quality_disclosures: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return the stable machine-readable reason for a partial report."""
    _status, _current_step, primary, _reasons = build_report_partial_fields(
        unreported_figures=unreported_figures,
        figureless_sections=figureless_sections,
        approved_partial=bool(getattr(state, "allow_partial_report", False)),
        quality_disclosures=quality_disclosures,
    )
    return primary


def figureless_outline_sections(
    state: AnalysisState,
    sections: List[Dict[str, Any]],
) -> List[str]:
    """Return result sections that lack an embed-ready figure."""
    sections_by_id = {
        str(section.get("id") or ""): section for section in (sections or [])
    }
    missing: List[str] = []
    for outline_section in build_report_outline(state):
        if not outline_section.expects_figures:
            continue
        section = sections_by_id.get(outline_section.id)
        usable = any(
            str(figure.get("png_path") or "").strip()
            and Path(str(figure.get("png_path"))).is_file()
            for figure in ((section or {}).get("figures") or [])
        )
        if not usable:
            missing.append(outline_section.id)
    return missing


def ensure_report_plan_figure_coverage(
    report_plan: Dict[str, Any],
    existing_figures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bind every scanned figure to a deterministic plan section."""
    figures_by_name = {
        str(figure.get("filename")): figure
        for figure in existing_figures
        if figure.get("filename")
    }
    if not figures_by_name:
        return report_plan
    sections = [dict(section) for section in report_plan.get("sections", [])]
    assigned: set[str] = set()
    for section in sections:
        for reference in section.get("existing_figures", []) or []:
            name = (
                reference
                if isinstance(reference, str)
                else reference.get("filename", "")
            )
            if name in figures_by_name:
                assigned.add(name)
    missing = [name for name in figures_by_name if name not in assigned]
    if not missing:
        return report_plan
    existing_ids = {str(section.get("id") or "") for section in sections}
    recovery_id = "additional_results"
    suffix = 2
    while recovery_id in existing_ids:
        recovery_id = f"additional_results_{suffix}"
        suffix += 1
    sections.append(
        {
            "id": recovery_id,
            "title": "Additional Analysis Results",
            "description": "Validated analysis figures not assigned by the report plan",
            "analysis_type": "specific",
            "existing_figures": [dict(figures_by_name[name]) for name in missing],
            "existing_tables": [],
        }
    )
    recovered_plan = dict(report_plan)
    recovered_plan["sections"] = sections
    logger.warning(
        "Deterministically recovered %d unassigned figure(s) into additional_results",
        len(missing),
    )
    return recovered_plan


def unreported_figure_names(
    existing_figures: List[Dict[str, Any]],
    generated_figures: List[Dict[str, Any]],
) -> List[str]:
    """Return scanned figure filenames absent from the generated report."""
    generated_names = {
        os.path.basename(str(figure.get("png_path") or ""))
        for figure in generated_figures
        if figure.get("png_path")
    }
    return [
        str(figure.get("filename"))
        for figure in existing_figures
        if figure.get("filename")
        and str(figure.get("filename")) not in generated_names
    ]


def rebind_checkpoint_asset_paths(
    value: Any,
    *,
    figures_dir: Path,
    tables_dir: Path,
) -> Any:
    """Rebase checkpoint asset paths into the current private generation."""
    if isinstance(value, dict):
        rebound = {
            key: rebind_checkpoint_asset_paths(
                item, figures_dir=figures_dir, tables_dir=tables_dir
            )
            for key, item in value.items()
        }
        for key, directory in (
            ("png_path", figures_dir),
            ("pdf_path", figures_dir),
            ("csv_path", tables_dir),
        ):
            raw_path = value.get(key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = directory / Path(raw_path).name
            rebound[key] = str(candidate) if candidate.is_file() else None
        return rebound
    if isinstance(value, list):
        return [
            rebind_checkpoint_asset_paths(
                item, figures_dir=figures_dir, tables_dir=tables_dir
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            rebind_checkpoint_asset_paths(
                item, figures_dir=figures_dir, tables_dir=tables_dir
            )
            for item in value
        )
    return value


def build_agent_models_appendix(state) -> dict:
    """Return agent models, retaining the actually used literature model."""
    models = dict(state.custom_models) if state.custom_models else {}
    actual = getattr(state, "literature_actual_model", None)
    if actual:
        models["literature"] = actual
    return models


def build_cross_modal_appendix(state) -> list:
    """Return full per-round cross-modal synthesis for traceability."""
    out = []
    for result in state.round_results:
        synthesis = result.cross_modal_synthesis
        if not synthesis:
            continue
        out.append(
            {
                "round_number": result.round_number,
                "matrix_markdown": synthesis.matrix_markdown,
                "candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in synthesis.candidates
                ],
                "excluded_or_low_support_themes": (
                    synthesis.excluded_or_low_support_themes
                ),
            }
        )
    return out


def build_report_followups(
    state: AnalysisState,
    *,
    lineage_id: str = "main",
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    """Flatten bounded PI follow-up records for the product-facing report.

    The direction projection owns lineage matching and follow-up bounds.  This
    helper only selects readable fields for ``report.json`` and keeps the stable
    follow-up reference required by the Chat handoff.
    """
    from backend.utils.direction_payload import collect_direction_projections

    target = str(lineage_id or "main").strip() or "main"
    try:
        cap = max(0, int(max_items))
    except (TypeError, ValueError):
        cap = 12
    if cap <= 0:
        return []

    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for projection in collect_direction_projections(
        state, lineage_id=target, max_directions=12,
    ):
        for item in projection.get("followups") or []:
            if not isinstance(item, dict):
                continue
            followup_id = str(item.get("followup_id") or "").strip()
            question = str(item.get("question") or "").strip()
            if not followup_id or not question or followup_id in seen:
                continue
            record: Dict[str, Any] = {
                "followup_id": followup_id[:160],
                "question": question[:1000],
                "why_it_matters": str(item.get("why_it_matters") or "")[:1200],
                "disposition": str(item.get("disposition") or "")[:80],
                "reason": str(item.get("reason") or "")[:1200],
                "suggested_instruction": str(item.get("suggested_instruction") or "")[:1400],
                "required_input": str(item.get("required_input") or "")[:1200],
                "evidence_references": [
                    str(value)[:160]
                    for value in (item.get("evidence_references") or [])
                    if str(value).strip()
                ][:12],
                "source_round": int(item.get("source_round", 0) or 0),
                "direction": str(item.get("direction") or projection.get("direction") or "")[:800],
                "branch_id": target,
            }
            for field_name in ("execution_outcome", "execution_reference"):
                value = item.get(field_name)
                if value not in (None, "", [], {}):
                    record[field_name] = str(value)[:1200]
            output.append(record)
            seen.add(followup_id)
            if len(output) >= cap:
                return output
    return output
