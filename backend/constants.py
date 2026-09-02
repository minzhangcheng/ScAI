"""Shared constants for the ScAI backend.

Single source of truth for version string, node-to-agent mappings,
action enums, and LLM failure intervention constants.
"""
from enum import Enum

VERSION = "0.51.11"


# ── Authoritative graph node registry — single source of truth ─────
WORKFLOW_GRAPH_NODES: frozenset[str] = frozenset({
    "restore_entry", "initialize_round", "data_profiling", "pi_planning",
    "planning_dialogue",
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
    RUNTIME_PERSISTENCE_FAILURE = "runtime_persistence_failure"


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
    RUNTIME_PERSISTENCE_FAILURE = "runtime_persistence_failure"
    RUNTIME_PERSISTENCE_COMMITTED = "runtime_persistence_committed"


# Core agents whose absence blocks the workflow. Disabling any of these
# must surface a user-intervention dialog rather than silently skipping.
CORE_AGENTS = frozenset({"pi", "bioinformatics", "literature"})


# ── PI-agent signalling constants ──────────────────────────────────
# Sentinel tag injected into PI responses to signal plan revision is needed
PLAN_REVISION_NEEDED_TAG = "[__PLAN_REVISION_NEEDED__]"

# sandbox_state key holding the kernel variable names the pre-planning
# profiling pass left behind. Written by the data_profiling node, read by the
# plan prompt in backend/agents/bioinformatics/planning.py -- two different
# layers, so the name lives here rather than as a literal in both.
PROFILED_OBJECTS_KEY = "profiled_objects_v1"
PI_REVIEW_FALLBACK_DECISION = "revise"

# HITL signal sentinel values used in analysis_state.user_chosen_direction and
# pending_human_input fields. Centralized so a typo in any consumer is caught
# at import time rather than silently breaking comparison semantics.
CONTINUE_SIGNAL = "[CONTINUE]"
PI_DECIDE_SIGNAL = "[PI_DECIDE]"

# Substituted for a figure message's base64 payload before the message is
# persisted: keeping 70+ inline images would bloat session.json for no gain,
# since the figures themselves are on disk. The frontend must recognise this
# exact value -- rendered as an image source it produces a broken <img>, which
# is what every reloaded session showed. Mirrored in
# frontend/src/constants/storage.ts.
STRIPPED_FIGURE_PLACEHOLDER = "[image]"

# Semantic gate identifiers used by restore targets, wait helpers, and resume routing.
GATE_ROUND_WAIT = "round_wait"
GATE_DIRECTION_SELECTION = "direction_selection"
GATE_INTEGRATION_SELECTION = "integration_selection"
GATE_ANNOTATION_CONFIRMATION = "annotation_confirmation"
GATE_PLAN_REVISION = "plan_revision"

# Every frozen semantic gate with a supported user action. The action endpoint
# converts the capsule to its exact post-action graph node before reconnect.
SEMANTIC_WAIT_GATES: frozenset[str] = frozenset({
    GATE_ROUND_WAIT,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ANNOTATION_CONFIRMATION,
    GATE_PLAN_REVISION,
})

SEMANTIC_GATE_ACTION_NEXT_NODES: dict[str, str] = {
    GATE_ROUND_WAIT: "check_convergence",
    GATE_DIRECTION_SELECTION: "check_convergence",
    GATE_INTEGRATION_SELECTION: "check_convergence",
    GATE_ANNOTATION_CONFIRMATION: "bioinformatics_analysis",
    GATE_PLAN_REVISION: "planning_dialogue",
}

# ── PI human-input stop keywords (single source of truth) ─────────
# Used by pi_agent.process_human_guidance and pi_agent.process_user_input
# fallback parsing when the LLM does not return a structured decision.
STOP_KEYWORDS: frozenset[str] = frozenset({
    "stop", "terminate", "end", "finish", "conclude",
    "summarize", "report",
    "停止", "终止", "结束", "总结", "报告",
})

# ── Output-token cap ────────────────────────────────────────────────
# There is none, deliberately.  A thinking model spends its output budget on
# private reasoning first, so any cap we set is a cap on *thinking*, not on the
# answer: when the budget runs out the provider returns status=incomplete with
# an empty answer channel and every generated token is wasted.  Measured over
# one run on a single model, capped calls failed this way 11 times (4096: 6/79,
# 8192: 1/1, 32768: 4/156) while 60 uncapped calls on the same model failed 0
# times -- and three of those legitimately produced 33841-48555 tokens, which a
# 32768 cap would have destroyed.  Omitting the field entirely lets the model
# finish.
#
# What bounds a request instead is wall-clock: DEFAULT_REQUEST_TIMEOUT is a
# total deadline on both the non-streaming call and the streaming one (see
# _StartedStream in utils/llm_client_parts/adapters.py).
#
# Pass this at call sites rather than a bare None so the intent is greppable and
# nobody reintroduces a number by accident.
NO_OUTPUT_TOKEN_CAP: None = None
