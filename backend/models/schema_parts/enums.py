# Enumerations and literal type aliases shared by every model.
# Moved verbatim out of backend/models/schemas.py.
"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from typing import Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    PI = "pi"
    PATHOLOGY = "pathology"
    BIOINFORMATICS = "bioinformatics"
    BIOCHEMISTRY = "biochemistry"
    LITERATURE = "literature"
    PAINTER = "painter"
    SUPERVISOR = "supervisor"
    TECHNIQUE = "technique"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_INPUT = "waiting_input"
    WAITING_ANNOTATION = "waiting_annotation"  # Paused for user annotation confirmation
    WAITING_PLAN_REVIEW = "waiting_plan_review"  # Paused for user plan review/approval
    WAITING_SPECIES_INPUT = "waiting_species_input"  # Paused for user species input
    PAUSED = "paused"
    STOPPED = "stopped"
    GENERATING_REPORT = "generating_report"

class AnalysisPhase(str, Enum):
    """Multi-phase workflow: exploration (Round 1), innovation (Round 2+), integration (cross-direction round)"""
    EXPLORATION = "exploration"
    INNOVATION = "innovation"
    INTEGRATION = "integration"

SnapshotPhase = Literal[
    "pre_annotation", "post_annotation", "pre_direction", "post_direction",
    "pre_round_wait",
    "pre_execution_stage1", "pre_execution_stage2", "pre_execution_full",
    "pre_major_revision", "post_report",
    "manual",
    "pre_integration_gate",
    "post_chat",
    "pre_restore",
]

AnalysisPhaseType = Literal["exploration", "innovation", "integration"]

PlanStage = Literal["full", "stage1", "stage2"]

PlanStatus = Literal["draft", "pi_reviewing", "user_reviewing", "approved", "rejected", "executing", "revised"]

class FigureReviewStatus(str, Enum):
    """Status of figure review by PI Agent"""
    PENDING = "pending"
    APPROVED = "approved"
    ACCEPTED_WITH_RESERVATIONS = "accepted_with_reservations"
    REVISION_REQUESTED = "revision_requested"
