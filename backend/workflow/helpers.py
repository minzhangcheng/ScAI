"""Pure helper functions for the workflow — no side effects, no class dependencies."""
import ast
import asyncio
import inspect
import logging
import random
from typing import Dict, List, Optional, Union

from ..models.schemas import AgentRoundCache, AgentType, AnalysisState

logger = logging.getLogger(__name__)


class WorkflowTerminatedError(RuntimeError):
    """Raised when the user chooses Terminate at a HITL gate (disabled-agent or
    LLM-failure dialog). Propagates up to the workflow run loop, which aborts the
    analysis gracefully. Defined here (a dependency-free module) so low-level
    modules like retry.py can catch it without importing core.py."""
    pass


class WorkflowSupersededError(RuntimeError):
    """Raised when a snapshot restore has replaced the state this run is built on.

    A restore rewinds the session to an earlier point, so the graph position of
    the run that was in flight is stale by definition — and its checkpointer is
    still bound to the pre-restore database file, which the restore replaced by
    rename. Letting it resume would write the old position back over the
    restored checkpoint.

    Unlike :class:`WorkflowTerminatedError` this must NOT touch session state:
    the restore has already persisted the authoritative state, and overwriting it
    with a terminal status is exactly the corruption we are avoiding. The run
    loop catches this and returns the restored state untouched.
    """
    pass


async def safe_pi_brief(make_awaitable, session_id: str, label: str) -> Optional[str]:
    """Run a PI brief factory, returning a stripped brief or None.

    ``make_awaitable`` is a zero-arg callable that produces the PI brief
    coroutine; it is invoked inside the try so that even attribute/lookup errors
    on the PI agent degrade gracefully. Centralizes the brief-fallback contract
    shared by the annotation and research-round wiring: any empty/whitespace
    output or exception yields None so the caller falls back to the focused topic
    (never the old injection).
    """
    try:
        brief = await make_awaitable()
        brief = (brief or "").strip()
        if not brief:
            logger.info(f"[{session_id}] {label} brief was empty; falling back to focused topic")
            return None
        return brief
    except Exception as e:
        logger.warning(f"[{session_id}] {label} brief generation failed: {e}")
        return None


async def ack_restore_safe_point(owner) -> None:
    """Acknowledge a pending restore safe point on *owner*'s behalf.

    The single implementation of the boundary-hook idiom every acknowledgement
    site shares. Both tolerances are required, not defensive habit: the hook may
    be absent entirely (bare mocks and lightweight harnesses are bound to these
    code paths), and it may return a non-awaitable (tests bind plain callables
    for it).
    """
    await_restore = getattr(owner, "await_restore_safe_point", None)
    if callable(await_restore):
        result = await_restore()
        if inspect.isawaitable(result):
            await result


HITL_SAFE_POINT_POLL_SECONDS = 1.0


async def wait_with_skip_time(
    event: asyncio.Event,
    wait_seconds: int,
    session_id: str,
    gate_name: str,
    *,
    safe_point_hook=None,
) -> bool:
    """Unified HITL wait gate honoring user-configured `round_wait_seconds`.

    Semantics (same as round-wait):
      -1  manual  → block forever for the event
       0  no-wait → return immediately as if timed out
      >0  countdown → wait `wait_seconds + 60` seconds (60s buffer for client lag),
                     then return as timed out

    *safe_point_hook* is an optional zero-argument callable (normally the
    workflow's ``await_restore_safe_point``) invoked periodically while the gate
    is blocked. Every HITL gate in this system blocks *inside* a graph node, so
    the run task stays in flight and the graph never reaches another node entry
    — which is the only place the cooperative snapshot safe point is otherwise
    acknowledged. Without this hook a Snapshot V2 restore requested while the
    user is sitting at a gate had no acknowledger at all: the quiescent
    coordinator declines because the run task is live, and the graph is parked
    in ``event.wait()``. The restore waited out its full timeout and failed,
    while its pending request made every other snapshot operation on the session
    answer SAFE_POINT_BUSY — and the gate the user was sitting at is exactly the
    moment they reach for a restore.

    The hook returns immediately when nothing is pending, so the poll is cheap.
    Time spent frozen inside it does not consume the countdown: the deadline is
    pushed out by the frozen duration so a restore hold cannot silently trip the
    auto-skip the moment the user regains control.

    Returns:
        True  if the event was signaled (user responded)
        False if timed out / no-wait → caller should apply its default auto-skip action
    """
    if wait_seconds == 0:
        logger.info(f"[{session_id}] {gate_name}: skip time = 0, auto-skipping")
        return False

    loop = asyncio.get_running_loop()
    if wait_seconds == -1:
        logger.info(f"[{session_id}] {gate_name}: manual mode, waiting indefinitely")
        deadline = None
        timeout = None
    else:
        timeout = wait_seconds + 60
        logger.info(f"[{session_id}] {gate_name}: waiting up to {timeout}s (skip time {wait_seconds}s + 60s buffer)")
        deadline = loop.time() + timeout

    hooked = callable(safe_point_hook)
    while True:
        if event.is_set():
            return True
        # Nothing to wake up for without a hook, so the gate waits out whatever
        # budget is left in a single slice, exactly as it did before polling.
        slice_seconds = HITL_SAFE_POINT_POLL_SECONDS if hooked else None
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(f"[{session_id}] {gate_name}: timed out after {timeout}s, auto-skipping")
                return False
            slice_seconds = remaining if slice_seconds is None else min(slice_seconds, remaining)
        try:
            await asyncio.wait_for(event.wait(), timeout=slice_seconds)
            return True
        except asyncio.TimeoutError:
            pass
        if not hooked:
            continue
        frozen_from = loop.time()
        result = safe_point_hook()
        if inspect.isawaitable(result):
            await result
        frozen_for = loop.time() - frozen_from
        if deadline is not None and frozen_for > HITL_SAFE_POINT_POLL_SECONDS:
            deadline += frozen_for


def _retry_backoff_delay(attempt: int) -> float:
    """Compute exponential backoff delay with jitter (shared by @with_retry and inline retries)."""
    return min(2 ** attempt + random.uniform(0, 1), 30)


# Stage1 keyword set for filtering bioinformatics tasks.
# Stage1 = QC + clustering + annotation ONLY. Condition-level DEG (incl. bulk
# DESeq2/volcano) belongs to Stage2 — see agents.py stage1 phase_instruction
# which explicitly forbids DEG. Do NOT add deseq/bulk/volcano/vst/tmm here.
STAGE1_KEYWORDS = frozenset([
    "qc", "quality", "filter", "normaliz", "cluster", "umap", "pca",
    "dimension", "annotat", "cell type", "doublet", "batch", "harmony",
    "hvg", "preprocess", "load", "inspect", "merg", "integrat",
])


def _has_stage1_keyword(text: str) -> bool:
    """Return True if text mentions a stage1 (exploration) keyword, while
    treating 're-' prefixed forms (re-cluster, re-annotate) as NOT stage1."""
    text_lower = text.lower()
    for kw in STAGE1_KEYWORDS:
        idx = 0
        while True:
            pos = text_lower.find(kw, idx)
            if pos < 0:
                break
            preceding3 = text_lower[max(0, pos - 3):pos]
            if not preceding3.endswith("re-"):
                return True
            idx = pos + len(kw)
    return False

def decide_bioinfo_post_execution(
    planning_stage: str,
    plan_revision_requested: bool,
    result_status: str = "completed",
    figure_contract_passed: bool = True,
) -> str:
    """Decide what _bioinformatics_analysis should do after the agent returns.

    A "paused_for_revision" return is NOT a stage completion: the agent stopped
    partway through the current plan and must resume the SAME plan/stage after
    the revision is approved.

    Returns one of:
      "revision"        — agent paused mid-plan; keep current stage + current_plan,
                          do not cache the partial result.
      "incomplete"      — execution or required-figure contract is incomplete;
                          keep the stage and request a bounded plan revision.
      "stage1_complete" — stage1 genuinely finished; advance to stage2.
      "complete"        — stage2/full finished normally; cache result.
    """
    if plan_revision_requested:
        return "revision"
    if str(result_status or "").strip().lower() not in ("completed", "partial"):
        return "incomplete"
    if not figure_contract_passed:
        return "incomplete"
    if planning_stage == "stage1":
        return "stage1_complete"
    return "complete"


_DIFF_LABELS = {
    'added': '[ADDED]',
    'removed': '[REMOVED]',
    'modified': '[MODIFIED]',
    'unchanged': '',
}


def _diff_plan_steps(prev_steps: List[Dict], curr_steps: List[Dict]) -> Dict[int, str]:
    """Compare steps between two plan versions. Returns {step_id: 'added'|'removed'|'modified'|'unchanged'}."""
    prev_by_id = {s['id']: s for s in prev_steps}
    curr_by_id = {s['id']: s for s in curr_steps}
    diff = {}
    all_ids = set(prev_by_id.keys()) | set(curr_by_id.keys())
    for sid in all_ids:
        if sid in curr_by_id and sid not in prev_by_id:
            diff[sid] = 'added'
        elif sid in prev_by_id and sid not in curr_by_id:
            diff[sid] = 'removed'
        elif sid in curr_by_id and sid in prev_by_id:
            prev = prev_by_id[sid]
            curr = curr_by_id[sid]
            if (prev.get('title') != curr.get('title') or
                prev.get('method') != curr.get('method') or
                prev.get('figure_requirements', []) != curr.get('figure_requirements', []) or
                bool(prev.get('allow_text_only', False)) !=
                bool(curr.get('allow_text_only', False)) or
                prev.get('text_only_reason', '') != curr.get('text_only_reason', '')):
                diff[sid] = 'modified'
            else:
                diff[sid] = 'unchanged'
    return diff


def _get_task_text(t) -> str:
    """Extract searchable text from a task (string or dict)."""
    if isinstance(t, dict):
        return " ".join(str(v) for v in t.values()).lower()
    return str(t).lower()


def prepend_species(text: str, species: str) -> str:
    """Prefix a task/topic string with the analysis species, if both are set.

    No-op when either the species or the text is empty, so callers can apply it
    unconditionally instead of repeating the same guard at each injection site.
    """
    if species and text:
        return f"Species: {species}\n\n{text}"
    return text


def _dict_to_task_text(t: dict) -> str:
    """Render a task/topic dict into a single string, matching keys case-insensitively."""
    lower = {str(k).lower(): v for k, v in t.items()}
    parts: List[str] = []
    for key in ("task", "description", "topic", "name"):
        if key in lower:
            parts.append(str(lower[key]))
            break
    if not parts:
        parts.append(str(t))
    if lower.get("rationale"):
        parts.append(f"(Rationale: {lower['rationale']})")
    if lower.get("recommended_methods"):
        methods = lower["recommended_methods"]
        if isinstance(methods, list):
            parts.append(f"[Methods: {'; '.join(str(m) for m in methods)}]")
        else:
            parts.append(f"[Methods: {methods}]")
    if lower.get("target_readouts"):
        readouts = lower["target_readouts"]
        if isinstance(readouts, list):
            parts.append(f"[Readouts: {'; '.join(str(r) for r in readouts)}]")
        else:
            parts.append(f"[Readouts: {readouts}]")
    return " ".join(parts)


def _normalize_tasks(tasks: list) -> List[str]:
    """Convert task items to strings — LLMs sometimes return dicts (or their repr) instead of plain strings."""
    result = []
    for t in tasks:
        if isinstance(t, str):
            s = t.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    parsed = ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, dict):
                    result.append(_dict_to_task_text(parsed))
                    continue
            result.append(t)
        elif isinstance(t, dict):
            result.append(_dict_to_task_text(t))
        else:
            result.append(str(t))
    return result


# === Per-round agent result cache ===
# Lives in AnalysisState (snapshot-persisted) so a snapshot-restored round can
# short-circuit nodes whose results are already known.

_AgentKey = Union[AgentType, str]


def _key(agent: _AgentKey) -> str:
    return agent.value if isinstance(agent, AgentType) else agent


def cache_hit(analysis_state, agent: _AgentKey) -> Optional[dict]:
    """Return the cached result dict for this agent in the current round, or None on miss/stale."""
    cache = analysis_state.current_round_cache
    if cache is None or cache.round_number != analysis_state.current_round:
        return None
    return cache.results.get(_key(agent))


def cache_store(analysis_state, agent: _AgentKey, result: dict) -> None:
    """Store an agent's canonical node-return dict into the current-round cache."""
    cache = analysis_state.current_round_cache
    if cache is None or cache.round_number != analysis_state.current_round:
        cache = AgentRoundCache(round_number=analysis_state.current_round)
        analysis_state.current_round_cache = cache
    cache.results[_key(agent)] = result


def cache_invalidate(analysis_state, agent: _AgentKey) -> None:
    """Drop a single agent's entry from the current-round cache (no-op if absent)."""
    cache = analysis_state.current_round_cache
    if cache is None:
        return
    cache.results.pop(_key(agent), None)


def build_integration_gate_payload(state: AnalysisState, session_id: str) -> dict:
    """Build the `integration_gate` WebSocket message content.

    Single source for the {cycle, integration_analysis, candidates,
    detailed_report_url} shape emitted at the integration gate.
    """
    reports = getattr(state, "integration_reports", None) or []
    detailed_report_url = (
        f"/api/session/{session_id}/detailed-report/html"
        if getattr(state, "detailed_report_html_path", None) else None
    )
    integration_report_url = (
        f"/api/session/{session_id}/integration-report/html"
        if getattr(state, "integration_report_html_path", None) else None
    )
    return {
        "cycle": getattr(state, "integration_cycle", 0),
        "integration_analysis": reports[-1] if reports else "",
        "candidates": list(getattr(state, "integration_gate_candidates", None) or []),
        "detailed_report_url": detailed_report_url,
        "html_url": integration_report_url,
    }
