"""Recover explicit provider context overflow within the transport retry budget."""

import json
import logging

from openai import BadRequestError

from backend.utils.context_budget import apply_budget, estimate_tokens, resolve_ceiling

logger = logging.getLogger(__name__)


def context_settings(entry):
    """Only settings whose user edits invalidate an in-flight recovery."""
    return {key: entry.get(key) for key in (
        "context_window", "default_context_window", "adaptive_context_window",
    )}


def is_context_overflow(error: Exception) -> bool:
    """Recognize explicit input/context limits, never arbitrary HTTP 400 errors."""
    if not isinstance(error, BadRequestError) and getattr(error, "status_code", None) != 400:
        return False
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else {}
    code = detail.get("code") if isinstance(detail, dict) else None
    if isinstance(code, str) and code in {"context_length_exceeded", "context_window_exceeded", "prompt_too_long"}:
        return True
    text = str(error).lower()
    return (
        ("maximum context length" in text and
         any(term in text for term in ("requested", "exceed", "input tokens")))
        or "prompt is too long" in text
        or "input is too long" in text
        or "input tokens exceed" in text
    )


async def recover_context_overflow(
    *, session_id, agent_name, model, endpoint, messages, tools=None,
    max_tokens=None, expected_context=None,
) -> bool:
    """Persist a smaller session budget and mutate only this request's messages.

    The adapter rebuilds its wire payload from this list on each attempt. Full
    agent history and execution receipts are not changed by request compaction.
    """
    from backend.config import SystemConfig, resolve_agent_context_window_default
    from backend.state import sessions, get_session_lock
    from backend.services.session_lifecycle import get_lifecycle_lock
    from backend.services.session_store import _async_save_session_under_lifecycle
    from backend.utils.config_resolver import resolve_config_for_user
    from backend.utils.settings_provider import AgentSettingsProvider

    state = sessions.get(session_id)
    if state is None:
        return False
    config = await resolve_config_for_user(state.user_id) or SystemConfig()
    async with get_session_lock(session_id):
        async with get_lifecycle_lock(session_id):
            state = sessions.get(session_id)
            if state is None:
                return False
            provider = AgentSettingsProvider(state, config, global_client=None)
            entry = dict((state.agent_server_configs or {}).get(agent_name) or {})
            if expected_context is not None and context_settings(entry) != expected_context:
                return False
            current_endpoint = entry.get("llm_base_url") or config.api.llm_base_url
            if provider.get_model(agent_name) != model or current_endpoint != endpoint:
                # A concurrent user edit must not inherit the old request's limit.
                return False
            previous = provider.get_context_window(agent_name)
            reduced = max(8192, previous * 4 // 5)
            if reduced >= previous:
                return False
            entry.setdefault("default_context_window", resolve_agent_context_window_default(config, agent_name))
            entry.update(
                adaptive_context_window=reduced,
                adaptive_context_model=model,
                adaptive_context_endpoint=endpoint,
            )
            state.agent_server_configs[agent_name] = entry
            if expected_context is not None:
                expected_context.update(context_settings(entry))
            try:
                await _async_save_session_under_lifecycle(session_id, state, force=True)
            except Exception:
                logger.warning("[%s] Failed to persist reduced context budget; live state retains it",
                               session_id, exc_info=True)
            # Notify while configuration changes are still serialized. The UI
            # fetches authoritative settings rather than trusting delayed values.
            from backend.utils.websocket_delivery import send_analysis_json
            try:
                await send_analysis_json(session_id, {
                    "type": "agent_context_updated", "session_id": session_id,
                    "agent": agent_name,
                })
            except Exception:
                logger.warning("[%s] Failed to notify context budget update", session_id,
                               exc_info=True)

    measured = estimate_tokens(messages)
    schema_tokens = len(json.dumps(tools or [], ensure_ascii=False)) // 3 + 1
    # Force a smaller prompt even when the local estimate underestimated the
    # rejected request. Schemas remain intact and consume part of the budget.
    ceiling = max(1, min(measured * 4 // 5,
                        resolve_ceiling(reduced, max_tokens or 16384) - schema_tokens))
    anchor_count = 1 if messages and messages[0].get("role") == "system" else 0
    # Preserve the host's existing dynamic anchors, including observations that
    # the caller acknowledges as delivered after a successful PI response.
    while anchor_count < len(messages):
        content = messages[anchor_count].get("content")
        if not isinstance(content, str) or not content.startswith((
            "[Analysis Context]", "Technical context observations (advisory; analysis continued):",
        )):
            break
        anchor_count += 1
    compacted, _ = apply_budget(messages, ceiling=ceiling,
                                    anchor_count=anchor_count, agent_label=agent_name)
    if estimate_tokens(compacted) >= measured:
        # The ordinary compactor deliberately stops at 16 KiB for recent tool
        # results. A confirmed rejection permits a further proportional text
        # reduction, while preserving system instructions and tool identities.
        from backend.utils.formatting import truncate_for_llm
        compacted = [dict(message) for message in messages]
        for message in compacted[anchor_count:]:
            content = message.get("content")
            if (message.get("role") in {"tool", "user"}
                    and isinstance(content, str) and len(content) > 1024):
                message["content"] = truncate_for_llm(content, max_chars=len(content) * 4 // 5)
    if estimate_tokens(compacted) >= measured:
        logger.warning("[%s] Context overflow cannot be reduced further", agent_name)
        return False
    messages[:] = compacted
    logger.warning("[%s] Provider context overflow: budget %s -> %s; prompt estimate %s -> %s",
                   agent_name, previous, reduced, measured, estimate_tokens(compacted))
    return True
