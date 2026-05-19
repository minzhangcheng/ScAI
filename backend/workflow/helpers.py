"""Pure helper functions for the workflow — no side effects, no class dependencies."""
import asyncio
import logging
import random
from typing import Dict, List, Optional, Union

from ..models.schemas import AgentRoundCache, AgentType

logger = logging.getLogger(__name__)


async def wait_with_skip_time(
    event: asyncio.Event,
    wait_seconds: int,
    session_id: str,
    gate_name: str,
) -> bool:
    """Unified HITL wait gate honoring user-configured `round_wait_seconds`.

    Semantics (same as round-wait):
      -1  manual  → block forever for the event
       0  no-wait → return immediately as if timed out
      >0  countdown → wait `wait_seconds + 60` seconds (60s buffer for client lag),
                     then return as timed out

    Returns:
        True  if the event was signaled (user responded)
        False if timed out / no-wait → caller should apply its default auto-skip action
    """
    if wait_seconds == 0:
        logger.info(f"[{session_id}] {gate_name}: skip time = 0, auto-skipping")
        return False
    if wait_seconds == -1:
        logger.info(f"[{session_id}] {gate_name}: manual mode, waiting indefinitely")
        await event.wait()
        return True
    timeout = wait_seconds + 60
    logger.info(f"[{session_id}] {gate_name}: waiting up to {timeout}s (skip time {wait_seconds}s + 60s buffer)")
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning(f"[{session_id}] {gate_name}: timed out after {timeout}s, auto-skipping")
        return False


def _retry_backoff_delay(attempt: int) -> float:
    """Compute exponential backoff delay with jitter (shared by @with_retry and inline retries)."""
    return min(2 ** attempt + random.uniform(0, 1), 30)


# Stage1 keyword set for filtering bioinformatics tasks
STAGE1_KEYWORDS = frozenset([
    "qc", "quality", "filter", "normaliz", "cluster", "umap", "pca",
    "dimension", "annotat", "cell type", "doublet", "batch", "harmony",
    "hvg", "preprocess", "load", "inspect", "merg", "integrat",
])

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
                prev.get('method') != curr.get('method')):
                diff[sid] = 'modified'
            else:
                diff[sid] = 'unchanged'
    return diff


def _get_task_text(t) -> str:
    """Extract searchable text from a task (string or dict)."""
    if isinstance(t, dict):
        return " ".join(str(v) for v in t.values()).lower()
    return str(t).lower()


def _normalize_tasks(tasks: list) -> List[str]:
    """Convert task items to strings — LLMs sometimes return dicts instead of plain strings."""
    result = []
    for t in tasks:
        if isinstance(t, str):
            result.append(t)
        elif isinstance(t, dict):
            parts = []
            for key in ("task", "description", "topic", "name"):
                if key in t:
                    parts.append(str(t[key]))
                    break
            if not parts:
                parts.append(str(t))
            if "rationale" in t:
                parts.append(f"(Rationale: {t['rationale']})")
            if "recommended_methods" in t and t["recommended_methods"]:
                methods = t["recommended_methods"]
                if isinstance(methods, list):
                    parts.append(f"[Methods: {'; '.join(str(m) for m in methods)}]")
                else:
                    parts.append(f"[Methods: {methods}]")
            if "target_readouts" in t and t["target_readouts"]:
                readouts = t["target_readouts"]
                if isinstance(readouts, list):
                    parts.append(f"[Readouts: {'; '.join(str(r) for r in readouts)}]")
                else:
                    parts.append(f"[Readouts: {readouts}]")
            result.append(" ".join(parts))
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
