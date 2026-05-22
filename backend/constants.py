"""Shared constants for the ScAI backend.

Single source of truth for node-to-agent mappings, action enums,
and LLM failure intervention constants.
"""
from enum import Enum


# ── Node → Agent mapping ───────────────────────────────────────────
NODE_TO_AGENT: dict[str, str] = {
    "pi_planning": "pi",
    "planning_dialogue": "pi",
    "pathology_analysis": "pathology",
    "bioinformatics_analysis": "bioinformatics",
    "biochemistry_analysis": "biochemistry",
    "literature_research": "literature",
    "pi_synthesis": "pi",
    "user_review_gate": "pi",
}

# ── Node → WorkflowState result key mapping ─────────────────────────
NODE_TO_RESULT_KEY: dict[str, str] = {
    "pi_planning": "pi_instructions",
    "planning_dialogue": "planning_result",
    "pathology_analysis": "pathology_result",
    "bioinformatics_analysis": "bioinformatics_result",
    "biochemistry_analysis": "biochemistry_result",
    "literature_research": "literature_result",
}


# ── LLM failure intervention constants ─────────────────────────────

class LlmFailureAction(str, Enum):
    RETRY = "retry"
    SKIP = "skip"
    TERMINATE = "terminate"


class WaitInputReason(str, Enum):
    LLM_FAILURE = "llm_failure"
    AGENT_DISABLED = "agent_disabled"


class StateSyncChangeType(str, Enum):
    LLM_FAILURE = "llm_failure"
    LLM_RETRY = "llm_retry"
    AGENT_DISABLED = "agent_disabled"
    AGENT_RESUMED = "agent_resumed"


# Core agents whose absence blocks the workflow. Disabling any of these
# must surface a user-intervention dialog rather than silently skipping.
CORE_AGENTS = frozenset({"pi", "bioinformatics", "literature"})


# ── PI-agent signalling constants ──────────────────────────────────
# Sentinel tag injected into PI responses to signal plan revision is needed
PLAN_REVISION_NEEDED_TAG = "[__PLAN_REVISION_NEEDED__]"
PI_REVIEW_FALLBACK_DECISION = "revise"

# TODO(brittleness): Substring matching is inherently fragile — PI natural-language
# responses like "I cannot execute this step because X" will trigger false positives
# on "cannot execute". Consider replacing with structured signals (e.g. a dedicated
# tool call or a prefixed tag like [PLAN_REVISION]) so that intent is unambiguous.
# HITL signal sentinel values used in analysis_state.user_chosen_direction and
# pending_human_input fields. Centralized so a typo in any consumer is caught
# at import time rather than silently breaking comparison semantics.
CONTINUE_SIGNAL = "[CONTINUE]"
PI_DECIDE_SIGNAL = "[PI_DECIDE]"


PLAN_REVISION_KEYWORDS = frozenset([
    "plan needs revision", "plan revision", "replan",
    "cannot execute", "not applicable", "different approach",
    "rethink", "redesign", "fundamental change",
    "major change", "hypothesis change",
])
