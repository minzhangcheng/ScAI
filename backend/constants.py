"""Shared constants for the ScAI backend.

Single source of truth for version string, node-to-agent mappings,
action enums, and LLM failure intervention constants.
"""
from enum import Enum

VERSION = "0.28.8.1"


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

# Nodes that participate in the agent registry but intentionally have no
# independent WorkflowState result key. Update this set when adding such a
# node; otherwise the import-time assertion below will fail and force the
# author to decide whether the new node needs a result key.
RESULT_KEYLESS_NODES: frozenset[str] = frozenset({
    "pi_synthesis",       # writes into round_result, not its own key
    "user_review_gate",   # gating node, not an analysis step
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
    SPECIES_INPUT = "species_input"
    GENE_SPECIES_MISMATCH = "gene_species_mismatch"


class StateSyncChangeType(str, Enum):
    LLM_FAILURE = "llm_failure"
    LLM_RETRY = "llm_retry"
    AGENT_DISABLED = "agent_disabled"
    AGENT_RESUMED = "agent_resumed"
    SPECIES_INPUT = "species_input"
    GENE_SPECIES_MISMATCH = "gene_species_mismatch"
    SPECIES_RESOLVED = "species_resolved"
    SPECIES_AUTO_SKIPPED = "species_auto_skipped"


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

# ── PI human-input stop keywords (single source of truth) ─────────
# Used by pi_agent.process_human_guidance and pi_agent.process_user_input
# fallback parsing when the LLM does not return a structured decision.
STOP_KEYWORDS: frozenset[str] = frozenset({
    "stop", "terminate", "end", "finish", "conclude",
    "summarize", "report",
    "停止", "终止", "结束", "总结", "报告",
})
