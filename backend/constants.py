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


class WaitInputReason(str, Enum):
    LLM_FAILURE = "llm_failure"


class StateSyncChangeType(str, Enum):
    LLM_FAILURE = "llm_failure"
    LLM_RETRY = "llm_retry"


# ── PI-agent signalling constants ──────────────────────────────────
# Sentinel tag injected into PI responses to signal plan revision is needed
PLAN_REVISION_NEEDED_TAG = "[__PLAN_REVISION_NEEDED__]"
