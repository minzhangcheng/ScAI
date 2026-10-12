"""Shared estimation, compaction, and PI-visible prompt-budget observations.

The character/token conversion is deliberately approximate.  Budget failures
are observations: this module always returns a best-effort payload and never
blocks analysis or changes workflow state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.models.schemas import AnalysisState, ContextBudgetObservation
from backend.utils.formatting import truncate_for_llm

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN: float = 3.5
FRAMING_RESERVE_TOKENS: int = 512
DEFAULT_COMPLETION_RESERVE: int = 16_384
RECENT_PROTECT: int = 6
TRUNCATE_BUDGETS: Tuple[int, ...] = (64 * 1024, 32 * 1024, 16 * 1024)
MAX_DECLARATION_CHARS: int = 2_000
MAX_PI_OBSERVATION_CHARS: int = 4_000
MAX_CONTEXT_BUDGET_OBSERVATIONS: int = 20


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens using the project-wide 3.5 characters/token ratio."""

    return int(len(text or "") / CHARS_PER_TOKEN)


def estimate_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Estimate text, echoed reasoning, and tool-call argument tokens."""

    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += len(json.dumps(content, default=str))
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            total_chars += len(reasoning_content)
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                arguments = tool_call.get("function", {}).get("arguments", "")
            else:
                function = getattr(tool_call, "function", None)
                arguments = getattr(function, "arguments", "") if function else ""
            if isinstance(arguments, str):
                total_chars += len(arguments)
        total_chars += len(message.get("name", "") or "")
        total_chars += len(message.get("tool_call_id", "") or "")
    return int(total_chars / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class ContextBudget:
    """A context window with a positive best-effort prompt allowance."""

    context_window: int
    completion_reserve: int = DEFAULT_COMPLETION_RESERVE
    budget_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.context_window <= FRAMING_RESERVE_TOKENS:
            raise ValueError("context_window must exceed the framing reserve")
        if self.completion_reserve < 0:
            raise ValueError("completion_reserve must be non-negative")
        if not 0.0 < self.budget_factor <= 1.0:
            raise ValueError("budget_factor must be in (0, 1]")

    @property
    def requested_completion_reserve(self) -> int:
        return self.completion_reserve

    @property
    def effective_completion_reserve(self) -> int:
        return min(
            self.completion_reserve,
            max(0, self.context_window - FRAMING_RESERVE_TOKENS - 1),
        )

    @property
    def prompt_ceiling(self) -> int:
        available = (
            self.context_window
            - self.effective_completion_reserve
            - FRAMING_RESERVE_TOKENS
        )
        return max(1, int(available * self.budget_factor))


def resolve_ceiling(
    context_window: int,
    completion_reserve: int = DEFAULT_COMPLETION_RESERVE,
    budget_factor: float = 1.0,
) -> int:
    return ContextBudget(
        context_window=context_window,
        completion_reserve=completion_reserve,
        budget_factor=budget_factor,
    ).prompt_ceiling


@dataclass(frozen=True)
class CompactionAction:
    kind: str
    detail: str
    removed_tokens: int


@dataclass(frozen=True)
class CompactionReport:
    measured_tokens: int
    final_tokens: int
    ceiling: int
    requested_completion_reserve: int
    effective_completion_reserve: int
    actions: Tuple[CompactionAction, ...]
    fits: bool


def _report(
    *,
    measured_tokens: int,
    final_tokens: int,
    budget: ContextBudget,
    actions: Sequence[CompactionAction] = (),
) -> CompactionReport:
    return CompactionReport(
        measured_tokens=measured_tokens,
        final_tokens=final_tokens,
        ceiling=budget.prompt_ceiling,
        requested_completion_reserve=budget.requested_completion_reserve,
        effective_completion_reserve=budget.effective_completion_reserve,
        actions=tuple(actions),
        fits=final_tokens <= budget.prompt_ceiling,
    )


def compact_messages(
    messages: Sequence[Mapping[str, Any]],
    budget: ContextBudget,
    *,
    anchor_count: int = 1,
    summarize: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> Tuple[List[Dict[str, Any]], CompactionReport]:
    """Apply deterministic, tool-pair-aware degradation to a message list."""

    result = [dict(message) for message in messages]
    measured_tokens = estimate_tokens(result)
    if measured_tokens <= budget.prompt_ceiling:
        return result, _report(
            measured_tokens=measured_tokens,
            final_tokens=measured_tokens,
            budget=budget,
        )

    actions: List[CompactionAction] = []
    anchor_count = min(max(0, anchor_count), len(result))
    protect_end = max(anchor_count, len(result) - RECENT_PROTECT)

    for char_budget in TRUNCATE_BUDGETS:
        if estimate_tokens(result) <= budget.prompt_ceiling:
            break
        candidates = []
        for index in range(anchor_count, protect_end):
            message = result[index]
            content = message.get("content")
            if message.get("role") not in ("tool", "user"):
                continue
            if isinstance(content, str) and len(content) > char_budget:
                candidates.append((index, len(content)))
        for index, _length in sorted(candidates, key=lambda item: -item[1]):
            if estimate_tokens(result) <= budget.prompt_ceiling:
                break
            before = estimate_tokens(result)
            old_content = result[index]["content"]
            new_content = truncate_for_llm(old_content, max_chars=char_budget)
            if len(new_content) >= len(old_content):
                continue
            result[index] = {**result[index], "content": new_content}
            actions.append(CompactionAction(
                kind="truncate",
                detail=f"message {index} ({result[index].get('role')}) to {char_budget} chars",
                removed_tokens=max(0, before - estimate_tokens(result)),
            ))

    if estimate_tokens(result) > budget.prompt_ceiling:
        tail_start = max(anchor_count, len(result) - RECENT_PROTECT)
        cut = tail_start
        while cut > anchor_count and result[cut].get("role") == "tool":
            cut -= 1
        if cut > anchor_count:
            dropped = result[anchor_count:cut]
            summary = (
                summarize(dropped)
                if summarize is not None
                else f"{len(dropped)} earlier messages omitted."
            )
            summary_message = {
                "role": "user",
                "content": f"[Previous analysis summary (compacted)]\n{summary}",
            }
            before = estimate_tokens(result)
            result = result[:anchor_count] + [summary_message] + result[cut:]
            actions.append(CompactionAction(
                kind="drop",
                detail=f"{len(dropped)} oldest middle messages",
                removed_tokens=max(0, before - estimate_tokens(result)),
            ))

    if estimate_tokens(result) > budget.prompt_ceiling:
        fallback_start = max(anchor_count, len(result) - RECENT_PROTECT)
        for index in range(fallback_start, len(result)):
            message = result[index]
            content = message.get("content")
            if message.get("role") not in ("tool", "user"):
                continue
            if not isinstance(content, str) or len(content) <= 16 * 1024:
                continue
            before = estimate_tokens(result)
            new_content = truncate_for_llm(content, max_chars=16 * 1024)
            result[index] = {**message, "content": new_content}
            actions.append(CompactionAction(
                kind="fallback_truncate",
                detail=f"protected message {index} to 16384 chars",
                removed_tokens=max(0, before - estimate_tokens(result)),
            ))
            if estimate_tokens(result) <= budget.prompt_ceiling:
                break

    final_tokens = estimate_tokens(result)
    return result, _report(
        measured_tokens=measured_tokens,
        final_tokens=final_tokens,
        budget=budget,
        actions=actions,
    )


def withheld_declaration(report: CompactionReport) -> str:
    """Return a bounded recipient-facing declaration of reduction or overflow."""

    if not report.actions and report.fits:
        return ""
    status = "reduced" if report.fits else "reduced on a best-effort basis"
    lines = [
        f"[Host] Context was {status} (measured ~{report.measured_tokens} tokens; "
        f"final ~{report.final_tokens}; prompt ceiling {report.ceiling}).",
    ]
    if report.requested_completion_reserve != report.effective_completion_reserve:
        lines.append(
            "Completion allowance was adjusted from "
            f"{report.requested_completion_reserve} to "
            f"{report.effective_completion_reserve} tokens."
        )
    by_kind: Dict[str, Dict[str, int]] = {}
    for action in report.actions:
        stats = by_kind.setdefault(action.kind, {"count": 0, "removed": 0})
        stats["count"] += 1
        stats["removed"] += action.removed_tokens
    for kind, stats in by_kind.items():
        lines.append(
            f"- {kind}: {stats['count']} action(s), ~{stats['removed']} tokens withheld"
        )
    if not report.fits:
        lines.append(
            "The protected/recent context still exceeds this estimate; dispatch continues "
            "for the recipient to assess."
        )
    return "\n".join(lines)[:MAX_DECLARATION_CHARS]


def apply_budget(
    messages: Sequence[Mapping[str, Any]],
    *,
    context_window: Optional[int] = None,
    ceiling: Optional[int] = None,
    agent_label: str = "",
    completion_reserve: Optional[int] = None,
    budget_factor: float = 1.0,
    anchor_count: int = 1,
    summarize: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    declare: bool = True,
) -> Tuple[List[Dict[str, Any]], CompactionReport]:
    """Return dispatchable best-effort messages and an observational report."""

    if (context_window is None) == (ceiling is None):
        raise ValueError("pass exactly one of context_window or ceiling")
    reserve = (
        completion_reserve
        if isinstance(completion_reserve, int)
        else DEFAULT_COMPLETION_RESERVE
    )
    if ceiling is not None:
        if ceiling <= 0:
            raise ValueError("ceiling must be positive")
        context_window = ceiling + reserve + FRAMING_RESERVE_TOKENS
    budget = ContextBudget(
        context_window=int(context_window),
        completion_reserve=reserve,
        budget_factor=budget_factor,
    )

    measured = estimate_tokens(messages)
    if measured <= budget.prompt_ceiling:
        return [dict(message) for message in messages], _report(
            measured_tokens=measured,
            final_tokens=measured,
            budget=budget,
        )

    declaration_tokens = estimate_text_tokens("x" * MAX_DECLARATION_CHARS) + 16
    internal_ceiling = max(1, budget.prompt_ceiling - declaration_tokens)
    internal_budget = ContextBudget(
        context_window=internal_ceiling + reserve + FRAMING_RESERVE_TOKENS,
        completion_reserve=reserve,
    )
    compacted, internal_report = compact_messages(
        messages,
        internal_budget,
        anchor_count=anchor_count,
        summarize=summarize,
    )
    provisional = CompactionReport(
        measured_tokens=measured,
        final_tokens=estimate_tokens(compacted),
        ceiling=budget.prompt_ceiling,
        requested_completion_reserve=budget.requested_completion_reserve,
        effective_completion_reserve=budget.effective_completion_reserve,
        actions=internal_report.actions,
        fits=estimate_tokens(compacted) <= budget.prompt_ceiling,
    )
    declaration = withheld_declaration(provisional) if declare else ""
    if declaration:
        compacted.append({"role": "user", "content": declaration})
    final_tokens = estimate_tokens(compacted)
    report = CompactionReport(
        measured_tokens=measured,
        final_tokens=final_tokens,
        ceiling=budget.prompt_ceiling,
        requested_completion_reserve=budget.requested_completion_reserve,
        effective_completion_reserve=budget.effective_completion_reserve,
        actions=internal_report.actions,
        fits=final_tokens <= budget.prompt_ceiling,
    )
    log = logger.info if report.fits else logger.error
    log(
        "[ContextBudget] agent=%s measured=%d final=%d ceiling=%d fits=%s",
        agent_label or "unknown",
        report.measured_tokens,
        report.final_tokens,
        report.ceiling,
        report.fits,
    )
    return compacted, report


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def summarize_dropped_messages(
    dropped: Sequence[Mapping[str, Any]],
    *,
    extra_sections: Optional[
        Callable[[Sequence[Mapping[str, Any]]], List[str]]
    ] = None,
) -> str:
    """Locally summarize domain-neutral tool usage without another LLM call."""

    tool_counts: Dict[str, int] = {}
    error_count = 0
    for message in dropped:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict):
                    name = tool_call.get("function", {}).get("name", "unknown")
                else:
                    function = getattr(tool_call, "function", None)
                    name = getattr(function, "name", "unknown") if function else "unknown"
                tool_counts[name] = tool_counts.get(name, 0) + 1
        text = message_text(message)
        if message.get("role") == "tool" and "error" in text[:300].lower():
            error_count += 1
    parts = []
    if tool_counts:
        parts.append("Tools used: " + ", ".join(
            f"{name} x{count}" for name, count in tool_counts.items()
        ))
    if error_count:
        parts.append(f"Errors encountered: {error_count}")
    if extra_sections is not None:
        parts.extend(extra_sections(dropped))
    return "\n".join(parts) if parts else "Previous analysis steps completed."


def _action_summaries(report: CompactionReport) -> List[str]:
    summaries = [
        f"{action.kind}: {action.detail} (~{action.removed_tokens} tokens)"[:300]
        for action in report.actions[:7]
    ]
    if not summaries:
        summaries.append("protected context remained above the estimated ceiling")
    return summaries


def observation_from_report(
    agent: str,
    report: CompactionReport,
    *,
    actions: Optional[List[str]] = None,
) -> ContextBudgetObservation:
    return ContextBudgetObservation(
        agent=agent or "unknown",
        measured_tokens=report.measured_tokens,
        final_tokens=report.final_tokens,
        ceiling=report.ceiling,
        requested_completion_reserve=report.requested_completion_reserve,
        effective_completion_reserve=report.effective_completion_reserve,
        actions=(actions or _action_summaries(report))[:8],
        fits=report.fits,
    )


def _observation_key(observation: ContextBudgetObservation) -> tuple:
    return (
        observation.agent,
        observation.measured_tokens,
        observation.final_tokens,
        observation.ceiling,
        observation.requested_completion_reserve,
        observation.effective_completion_reserve,
        tuple(observation.actions),
        observation.fits,
    )


def append_observation(
    state: AnalysisState,
    observation: ContextBudgetObservation,
) -> ContextBudgetObservation:
    """Append or deduplicate one pending record, preferring pending retention."""

    for current in state.context_budget_observations:
        if current.pi_delivered_at is None and _observation_key(current) == _observation_key(observation):
            current.occurrence_count += 1
            return current
    state.context_budget_observations.append(observation)
    while len(state.context_budget_observations) > MAX_CONTEXT_BUDGET_OBSERVATIONS:
        delivered_index = next(
            (
                index
                for index, current in enumerate(state.context_budget_observations)
                if current.pi_delivered_at is not None
            ),
            0,
        )
        state.context_budget_observations.pop(delivered_index)
    return observation


def pending_observation_anchor(
    state: AnalysisState,
) -> Tuple[Optional[Dict[str, str]], Tuple[str, ...]]:
    """Render pending technical facts as one bounded PI dynamic anchor."""

    pending = [
        observation
        for observation in state.context_budget_observations
        if observation.pi_delivered_at is None
    ]
    if not pending:
        return None, ()
    lines = [
        "Technical context observations (advisory; analysis continued):",
    ]
    for observation in pending:
        action_text = "; ".join(observation.actions[:3])
        lines.append(
            f"- {observation.agent}: measured ~{observation.measured_tokens}, "
            f"final ~{observation.final_tokens}, ceiling {observation.ceiling}, "
            f"fits={str(observation.fits).lower()}, occurrences="
            f"{observation.occurrence_count}. {action_text}"
        )
    content = "\n".join(lines)[:MAX_PI_OBSERVATION_CHARS]
    return {"role": "user", "content": content}, tuple(
        observation.id for observation in pending
    )


def acknowledge_observations(
    state: AnalysisState,
    observation_ids: Sequence[str],
) -> None:
    delivered_at = datetime.now()
    wanted = set(observation_ids)
    for observation in state.context_budget_observations:
        if observation.id in wanted and observation.pi_delivered_at is None:
            observation.pi_delivered_at = delivered_at


async def record_observation(
    state: AnalysisState,
    observation: ContextBudgetObservation,
    *,
    session_id: Optional[str],
) -> ContextBudgetObservation:
    """Record under the session mutation lock; persistence failure is advisory."""

    if not session_id or session_id == "default":
        return append_observation(state, observation)

    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.services.session_store import _async_save_session_under_lifecycle
    from backend.state import get_session_lock, sessions, state_lock

    async with get_session_lock(session_id):
        async with get_lifecycle_lock(session_id):
            async with state_lock:
                target = sessions.get(session_id) or state
                stored = append_observation(target, observation)
                is_canonical = sessions.get(session_id) is target
            if not is_canonical:
                return stored
            try:
                await _async_save_session_under_lifecycle(
                    session_id,
                    target,
                    force=True,
                )
            except Exception:
                logger.warning(
                    "[%s] Failed to persist context-budget observation; "
                    "the live state retains it",
                    session_id,
                    exc_info=True,
                )
            return stored


async def mark_observations_delivered(
    state: AnalysisState,
    observation_ids: Sequence[str],
    *,
    session_id: Optional[str],
) -> None:
    """Acknowledge only after a successful PI dispatch."""

    if not observation_ids:
        return
    if not session_id or session_id == "default":
        acknowledge_observations(state, observation_ids)
        return

    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.services.session_store import _async_save_session_under_lifecycle
    from backend.state import get_session_lock, sessions, state_lock

    async with get_session_lock(session_id):
        async with get_lifecycle_lock(session_id):
            async with state_lock:
                target = sessions.get(session_id) or state
                acknowledge_observations(target, observation_ids)
                is_canonical = sessions.get(session_id) is target
            if not is_canonical:
                return
            try:
                await _async_save_session_under_lifecycle(
                    session_id,
                    target,
                    force=True,
                )
            except Exception:
                logger.warning(
                    "[%s] Failed to persist PI delivery acknowledgement; "
                    "the live state retains it",
                    session_id,
                    exc_info=True,
                )
