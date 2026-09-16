"""Live dependency capture for the publication-figure workflow."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import MappingProxyType, ModuleType
from typing import Any, Mapping


_DEPENDENCY_NAMES = (
    "_MAX_ROLLBACK_SNAPSHOT_BYTES",
    "_build_publication_figure_wrapper",
    "_discard_render_attempt",
    "_layout_unverified_marker",
    "_new_render_attempt_paths",
    "_publication_output_error",
    "_publish_render_attempt",
    "_violation_must_fix_items",
    "acquire_figure_publish_locks",
    "apply_publish_record_paths",
    "archival_render_available",
    "begin_figure_publish_transaction",
    "blocking_violations",
    "build_expression_runtime_code",
    "commit_figure_publish_transaction",
    "compact_traceback",
    "eligible_artifacts",
    "extract_error_type",
    "format_violations_for_prompt",
    "has_direct_expression_contract_call",
    "image_file_sha256",
    "instrument_expression_contract_calls",
    "lint_ran",
    "logger",
    "mark_figure_record_recovery_durable",
    "mark_publish_record_applied",
    "mutates_expression_representation",
    "normalize_text_list",
    "parse_expression_contracts",
    "parse_layout_violations",
    "pdf_file_sha256",
    "publication_pdf_is_valid",
    "record_executed_representation_change",
    "record_expression_contract",
    "refreeze_pending_figure_attempt",
    "register_pending_figure_attempt",
    "rollback_figure_publish_transaction",
    "strip_expression_contract_markers",
    "summarize_violations",
    "workspace_artifact_versions",
)


@dataclass(frozen=True)
class FigureDependencies:
    """Patch-visible values captured once at the historical facade boundary."""

    values: Mapping[str, Any]
    figures_facade: ModuleType
    agent_facade: ModuleType

    @classmethod
    def from_facade(cls) -> "FigureDependencies":
        figures = importlib.import_module("backend.agents.bioinformatics.figures")
        agent = importlib.import_module("backend.agents.bioinformatics_agent")
        values: dict[str, Any] = {}
        for name in _DEPENDENCY_NAMES:
            if hasattr(figures, name):
                values[name] = getattr(figures, name)
            elif hasattr(agent, name):
                values[name] = getattr(agent, name)
            else:  # pragma: no cover - import contract failure
                raise AttributeError(f"missing publication-figure dependency: {name}")
        return cls(MappingProxyType(values), figures, agent)

    def get(self, name: str) -> Any:
        return self.values[name]
