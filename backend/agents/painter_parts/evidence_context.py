"""Bounded evidence context shared by the graphic abstract and its text fallback.

The painter prompt previously extracted findings from two modalities only, so
its instruction to depict "only what the results support" was enforced by the
model's own discretion over an input that could not show a partial round, a
contradicting citation, or a modality that was never measured.

Bounded on purpose: this goes into an image-generation prompt with a small token
budget, so every list is capped and the caller can drop it whole if needed.
"""
from __future__ import annotations

from typing import Any, List


def _items(values, limit: int, prefix: str = "") -> List[str]:
    out = []
    for value in (values or [])[:limit]:
        text = str(value).strip()
        if text:
            out.append(f"{prefix}{text}")
    return out


def build_graphic_evidence_context(state: Any, max_chars: int = 4000) -> str:
    """Summarise what the run actually established, and how firmly."""
    lines: List[str] = []

    modalities = []
    for label, attr in (
        ("single-cell sequencing", "has_scrnaseq"),
        ("bulk sequencing", "has_bulk_rnaseq"),
    ):
        try:
            if getattr(state, attr)():
                modalities.append(label)
        except (AttributeError, TypeError):
            continue
    data_files = getattr(state, "data_files", None) or []
    sections = {str(getattr(item, "section", "")).strip().lower() for item in data_files}
    rounds = (getattr(state, "round_results", None) or [])[-4:]
    if "pathology" in sections or any(
        getattr(result, "pathology_findings", None) for result in rounds
    ):
        modalities.append("pathology")
    if (
        "biochemistry" in sections
        or getattr(state, "biochemistry_indicators", None)
        or any(getattr(result, "biochemistry_findings", None) for result in rounds)
    ):
        modalities.append("biochemistry")
    if modalities:
        lines.append("Modalities analysed: " + ", ".join(sorted(set(modalities))))

    for round_result in rounds:
        header = f"Round {getattr(round_result, 'round_number', '?')}"
        status = getattr(round_result, "round_status", None)
        if status:
            header += f" [status: {status}]"
        verified = getattr(round_result, "bioinformatics_evidence_verified", None)
        linkage = {
            True: "all structured findings linked to current successful receipts",
            False: "one or more findings lack a current successful receipt",
            None: "execution linkage not recorded",
        }[verified]
        header += f" [execution linkage: {linkage}]"
        lines.append(header)
        for label, attr, limit in (
            ("bioinformatics", "bioinformatics_findings", 3),
            ("pathology", "pathology_findings", 2),
            ("biochemistry", "biochemistry_findings", 2),
            ("literature", "literature_findings", 2),
        ):
            lines.extend(_items(getattr(round_result, attr, None), limit, f"  - {label}: "))
        lines.extend(_items(getattr(round_result, "evidence_notes", None), 3, "  ! caveat: "))

    report = getattr(state, "exploration_report", None)
    if report is not None:
        lines.extend(
            _items(getattr(report, "data_quality_warnings", None), 3, "! data quality: ")
        )

    if not lines:
        return "No structured evidence context is available for this figure."
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [evidence context truncated]"
    return text
