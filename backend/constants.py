"""Shared constants for the ScAI backend.

Single source of truth for version string, node-to-agent mappings,
action enums, and LLM failure intervention constants.
"""
from enum import Enum

VERSION = "0.36.11"


# ── Authoritative graph node registry — single source of truth ─────
WORKFLOW_GRAPH_NODES: frozenset[str] = frozenset({
    "restore_entry", "initialize_round", "pi_planning", "planning_dialogue",
    "user_review_gate", "pathology_analysis", "bioinformatics_analysis",
    "biochemistry_analysis", "literature_research", "pi_synthesis",
    "wait_for_user_input", "check_convergence", "generate_final_report",
    "generate_graphic_abstract", "generate_detailed_report", "integration_synthesis",
})

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
    "integration_synthesis": "pi",
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

# Nodes that participate in the agent registry but intentionally have no
# independent WorkflowState result key. Update this set when adding such a
# node; otherwise the import-time assertion below will fail and force the
# author to decide whether the new node needs a result key.
RESULT_KEYLESS_NODES: frozenset[str] = frozenset({
    "pi_synthesis",
    "user_review_gate",
    "integration_synthesis",
})

# Drift guard — fires at import time if the two maps go out of sync.
assert set(NODE_TO_RESULT_KEY).issubset(set(NODE_TO_AGENT)), (
    "NODE_TO_RESULT_KEY contains nodes missing from NODE_TO_AGENT: "
    f"{set(NODE_TO_RESULT_KEY) - set(NODE_TO_AGENT)}"
)
undocumented = set(NODE_TO_AGENT) - set(NODE_TO_RESULT_KEY) - RESULT_KEYLESS_NODES
stale = RESULT_KEYLESS_NODES - (set(NODE_TO_AGENT) - set(NODE_TO_RESULT_KEY))
assert not undocumented and not stale, (
    "NODE_TO_AGENT / NODE_TO_RESULT_KEY drift: "
    f"undocumented nodes (in agent map but no result key and not in RESULT_KEYLESS_NODES): {undocumented or 'none'}, "
    f"stale entries (in RESULT_KEYLESS_NODES but no longer keyless): {stale or 'none'}"
)


# ── LLM failure intervention constants ─────────────────────────────

class LlmFailureAction(str, Enum):
    RETRY = "retry"
    SKIP = "skip"
    TERMINATE = "terminate"


class WaitInputReason(str, Enum):
    LLM_FAILURE = "llm_failure"
    AGENT_DISABLED = "agent_disabled"
    MARKER_DB_MISSING = "marker_db_missing"
    SPECIES_INPUT = "species_input"
    GENE_SPECIES_MISMATCH = "gene_species_mismatch"
    SNAPSHOT_FAILURE = "snapshot_failure"


class StateSyncChangeType(str, Enum):
    LLM_FAILURE = "llm_failure"
    LLM_RETRY = "llm_retry"
    AGENT_DISABLED = "agent_disabled"
    AGENT_RESUMED = "agent_resumed"
    SPECIES_INPUT = "species_input"
    GENE_SPECIES_MISMATCH = "gene_species_mismatch"
    SPECIES_RESOLVED = "species_resolved"
    SPECIES_AUTO_SKIPPED = "species_auto_skipped"
    SNAPSHOT_FAILURE = "snapshot_failure"


# Core agents whose absence blocks the workflow. Disabling any of these
# must surface a user-intervention dialog rather than silently skipping.
CORE_AGENTS = frozenset({"pi", "bioinformatics", "literature"})


# ── PI-agent signalling constants ──────────────────────────────────
# Sentinel tag injected into PI responses to signal plan revision is needed
PLAN_REVISION_NEEDED_TAG = "[__PLAN_REVISION_NEEDED__]"
PI_REVIEW_FALLBACK_DECISION = "revise"

# HITL signal sentinel values used in analysis_state.user_chosen_direction and
# pending_human_input fields. Centralized so a typo in any consumer is caught
# at import time rather than silently breaking comparison semantics.
CONTINUE_SIGNAL = "[CONTINUE]"
PI_DECIDE_SIGNAL = "[PI_DECIDE]"

# Semantic gate identifiers used by restore targets, wait helpers, and resume routing.
GATE_ROUND_WAIT = "round_wait"
GATE_DIRECTION_SELECTION = "direction_selection"
GATE_INTEGRATION_SELECTION = "integration_selection"
GATE_ANNOTATION_CONFIRMATION = "annotation_confirmation"

# Only gates that can resume through the normal wait_for_user_input path.
SEMANTIC_WAIT_GATES: frozenset[str] = frozenset({
    GATE_ROUND_WAIT,
    GATE_INTEGRATION_SELECTION,
})

# ── PI human-input stop keywords (single source of truth) ─────────
# Used by pi_agent.process_human_guidance and pi_agent.process_user_input
# fallback parsing when the LLM does not return a structured decision.
STOP_KEYWORDS: frozenset[str] = frozenset({
    "stop", "terminate", "end", "finish", "conclude",
    "summarize", "report",
    "停止", "终止", "结束", "总结", "报告",
})

# ── Reasoning model output-token floor ──────────────────────────────
# OpenAI-style reasoning models (o1, o3, o4) may produce empty content
# when max_tokens is too low because the reasoning budget consumes the
# entire allocation.  This floor ensures enough headroom for the actual
# content.  Call sites should use this constant instead of hardcoding 4096.
REASONING_MODEL_MIN_TOKENS: int = 4096
