"""One replay budget for a vision call, shared by the sites that own one.

A stream cut short after its first event is not a visual verdict, but every
vision caller turns a failure into a recorded non-verdict or a text-only
fallback. Replaying the identical request first converts most of those
degradations back into results; a deterministic payload or capability
rejection is not replay-safe and passes straight through.

``agent_name`` and ``model`` are passed in rather than read off the caller.
Several hosts here are mixin compositions rather than ``BaseAgent`` subclasses,
and every one of these call sites sits inside a broad ``except Exception`` that
would file an ``AttributeError`` as a transient upstream failure -- hiding it.
"""
from typing import Any, Optional

from ..utils.llm_client import retry_replay_safe_llm_call


async def replay_safe_vision_call(
    client: Any,
    *,
    agent_name: str,
    model: Any,
    on_retry=None,
    **vision_kwargs: Any,
):
    """Issue one ``chat_with_vision`` request with a bounded replay budget."""
    return await retry_replay_safe_llm_call(
        lambda: client.chat_with_vision(**vision_kwargs),
        agent_name=agent_name,
        model_name=str(model or ""),
        llm_server=str(getattr(client, "base_url", "") or ""),
        on_retry=on_retry,
    )
