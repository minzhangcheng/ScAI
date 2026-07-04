"""Shared prompt-fragment builders used across PI and Supervisor agents."""
from __future__ import annotations

from typing import Literal

from ..models.schemas import AnalysisState

_INTEGRATION_TRUNCATE = 2000

IntegrationBlockStyle = Literal["final", "integrated", "detailed"]


def build_integration_block(state: AnalysisState, *, style: IntegrationBlockStyle) -> str:
    """Render the cross-direction integration analyses fragment.

    style:
      - "final":      section-headed block for the final report prompt.
      - "integrated": bold-headed block for the integrated report prompt.
      - "detailed":   bare cycle list (the section template supplies its heading).
    """
    reports = getattr(state, "integration_reports", None)
    if not reports:
        return ""
    parts = [
        f"### Integration Cycle {i}\n{report[:_INTEGRATION_TRUNCATE]}"
        for i, report in enumerate(reports, 1)
    ]
    body = "\n\n".join(parts)
    if style == "final":
        return (
            "\n## Cross-Direction Integration Analyses\n"
            "The following cross-direction integration analyses were performed "
            "between direction batches. Incorporate their insights into the report.\n\n"
            + body
        )
    if style == "integrated":
        return (
            "**Cross-Direction Integration Analyses:**\n"
            "The following cross-direction integration analyses were performed "
            "between direction batches. Incorporate their insights.\n\n"
            + body
        )
    if style == "detailed":
        return body
    raise ValueError(f"unknown integration block style: {style!r}")
