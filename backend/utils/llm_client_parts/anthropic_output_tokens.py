"""Negotiate the required Anthropic output budget once per provider/model."""

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable
from weakref import WeakKeyDictionary


logger = logging.getLogger(__name__)
INITIAL_MAX_TOKENS = 65_536

# Accepted budgets live for the backend process. Locks are local to an event
# loop so clients recreated for another loop do not retain a bound asyncio lock.
_accepted: dict[tuple[str, str], int] = {}
_locks: WeakKeyDictionary = WeakKeyDictionary()


def _is_output_limit_rejection(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status not in {400, 422}:
        return False
    body = getattr(error, "body", None)
    text = (str(error) + " " + json.dumps(body, default=str)).lower()
    # Input/context overflow, thinking budgets, and unrelated parameter errors
    # must retain their existing recovery paths rather than shrink output.
    if any(marker in text for marker in (
        "context", "prompt is too long", "input tokens", "input_tokens",
        "budget_tokens", "thinking",
    )):
        return False
    return bool(re.search(
        r"max_tokens[\s\"':=]*(?:\d+\s*>\s*\d+|"
        r"(?:must be |must not |cannot |should be |is |value )*"
        r"(?:less than|at most|<=|exceed|too (?:large|high)))",
        text,
    ))


async def negotiate_output_tokens(
    provider: str,
    model: str,
    request: Callable[[int], Awaitable[Any]],
) -> Any:
    """Use real requests, cache only acceptance, and serialize the first probe."""
    key = (provider.rstrip("/"), model)
    if key in _accepted:
        return await request(_accepted[key])

    locks = _locks.setdefault(asyncio.get_running_loop(), {})
    lock = locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key not in _accepted:
            value = INITIAL_MAX_TOKENS
            while True:
                try:
                    response = await request(value)
                except Exception as error:
                    if value <= 1 or not _is_output_limit_rejection(error):
                        raise
                    value //= 2
                    logger.info(
                        "Anthropic max_tokens rejected for %s; retrying with %s",
                        model, value,
                    )
                else:
                    _accepted[key] = value
                    return response
    # A concurrent request learned the budget. Ordinary requests need not wait
    # for each other's generation once the provider has accepted the value.
    return await request(_accepted[key])
