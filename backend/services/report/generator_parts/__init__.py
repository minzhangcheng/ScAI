"""Implementation owners for the detailed-report generator facade."""

from .support import (
    build_agent_models_appendix,
    build_cross_modal_appendix,
    build_report_followups,
    ensure_report_plan_figure_coverage,
    figureless_outline_sections,
    rebind_checkpoint_asset_paths,
    report_completion_fields,
    report_partial_reason,
    unreported_figure_names,
)

__all__ = [
    "build_agent_models_appendix",
    "build_cross_modal_appendix",
    "build_report_followups",
    "ensure_report_plan_figure_coverage",
    "figureless_outline_sections",
    "rebind_checkpoint_asset_paths",
    "report_completion_fields",
    "report_partial_reason",
    "unreported_figure_names",
]
