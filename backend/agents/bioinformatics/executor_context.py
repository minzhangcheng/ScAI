"""Ephemeral context records for one autonomous executor run."""
from __future__ import annotations

from dataclasses import dataclass
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
