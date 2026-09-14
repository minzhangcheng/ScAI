"""Ephemeral context records for one autonomous executor run."""
from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any


@dataclass
class ExecutorBudget:
    per_attempt_max_steps: int
    max_steps: int
    step_count: int = 0
    attempt_start_step: int = 0


@dataclass
class ExecutorBindings:
    facade: Any

    @classmethod
    def from_facade(cls) -> "ExecutorBindings":
        return cls(sys.modules["backend.agents.bioinformatics_agent"])


@dataclass
class ExecutorRunState:
    agent: Any
    analysis_state: Any
    messages: list
    findings: list
    figures: list
    analysis_log: list
    checkpoint: dict | None
    checkpoint_valid: bool
    budget: ExecutorBudget


@dataclass
class ExecutorLoopContext:
    agent: Any
    analysis_state: Any
    messages: list
    findings: list
    figures: list
    analysis_log: list
    mechanism_programs: list
    checkpoint_manager: Any
    save_checkpoint: Any
    review_callback: Any
    supervisor_callback: Any
    planning_stage: str
    max_steps: int
    attempt_start_step: int
    step_count: int
    completed: bool = False
    annotation_completed: bool = False
    annotation_retry_required: bool = False
    annotation_retry_attempts: int = 0
    mechanism_programs_registered: bool = False
    evidence_gate_degraded: bool = False
    evidence_gate_findings: list = None
    reported_evidence_ids: list = None
    accepted_finding_records: list = None
    llm_error: str | None = None
    text_response_count: int = 0
    empty_response_count: int = 0
    figure_iteration_count: int = 0
    register_score_iteration_count: int = 0
    figures_rendered: int = 0
    consecutive_errors: int = 0
    total_errors: int = 0
    plan_revision_needed: bool = False
    plan_revision_context: dict = None
    force_break: bool = False
    max_text_responses: int = 2
    max_empty_responses: int = 3
    max_free_figure_iterations: int = 5
    max_free_register_score_iterations: int = 5
    executor_run_kind: str = "ordinary"
    executor_run_scope_id: str | None = None
    executor_settlement_scope_id: str | None = None
    supplementary_request_id: str | None = None
    attempt_execution_ids: list = None
    pending_dispatch: dict | None = None
    resolved_adata_source_context: dict | None = None
    adata_source: Any = None

    def __post_init__(self) -> None:
        if self.evidence_gate_findings is None:
            self.evidence_gate_findings = []
        if self.reported_evidence_ids is None:
            self.reported_evidence_ids = []
        if self.accepted_finding_records is None:
            self.accepted_finding_records = []
        if self.plan_revision_context is None:
            self.plan_revision_context = {}
        if self.attempt_execution_ids is None:
            self.attempt_execution_ids = []
        if self.pending_dispatch is None:
            self.pending_dispatch = {}
        if self.resolved_adata_source_context is None:
            self.resolved_adata_source_context = {}
