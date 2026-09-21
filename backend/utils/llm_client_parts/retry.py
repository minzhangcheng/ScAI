# Retry policy: backoff constants, rate-limit delay, and the retry driver.
# Moved verbatim out of backend/utils/llm_client.py.
"""
LLM client utilities using OpenAI SDK for OpenAI-compatible APIs
Supports OpenRouter, OpenAI, and other compatible endpoints
"""
import base64
import html
import httpx
import inspect
import asyncio
import json
import logging
import re
import time
from uuid import uuid4
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError, InternalServerError, BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError

from ..llm_logger import get_llm_logger, LLMInteraction

logger = logging.getLogger(__name__)

from .error_signals import (
    _is_hard_quota_error,
    _is_protocol_adaptation_signal,
    is_responses_input_shape_error,
)
from .errors import StreamInterruptedError, UpstreamExhaustedError
from ..context_overflow import is_context_overflow


# Hard per-attempt deadline for one LLM request (seconds). A reasoning model
# routinely spends several minutes on a single non-streamed completion, and the
# client giving up first does NOT cancel the upstream generation — the relay
# still bills the full response. A deadline that is too tight therefore pays for
# the same tokens on every retry and receives nothing.
DEFAULT_REQUEST_TIMEOUT = 600.0

# Retry configuration
MAX_RETRIES = 4

MAX_RETRIES_UPSTREAM_5XX = 8  # Dedicated for flaky upstream (e.g., Cloudflare 524)

RETRY_DELAY_BASE = 2.0  # Base delay in seconds, will be multiplied by attempt number

# Rate limit (429) uses fixed escalating delays to handle TPM limits
RATE_LIMIT_DELAYS = [5, 10, 20, 40]  # seconds between retries

# Note: retry_with_backoff now uses a broad except Exception with protocol/parameter
# adaptation signal gating, so RETRYABLE_EXCEPTIONS is mainly used as a fallback set
# for log labeling. Non-adaptation errors of ALL types are retried, with exhaustion
# producing UpstreamExhaustedError → user-facing HITL gate.
RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,  # 500 - transient server errors
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    asyncio.TimeoutError,
)

# JSONDecodeError gets fewer retries (max 2) since repeated bad JSON likely indicates
# a persistent issue rather than a transient network problem.
MAX_JSON_RETRIES = 2

EMPTY_CONTENT_PLACEHOLDER = "[Error: LLM generated a response but content extraction failed. Please retry.]"

# A provider/proxy that replies HTTP 200 with an empty answer AND
# completion_tokens == 0 generated nothing at all — a transient upstream blip
# (silent proxy failure, dropped stream). Retry the same request with backoff
# before handing the empty result to callers, which otherwise treat "" as a
# valid answer (e.g. the supervisor report path saved a 0-byte "completed"
# report). Distinct from the reasoning-truncation guard, which fires only when
# finish_reason == "length".
EMPTY_RESPONSE_MAX_RETRIES = 3

# Explicit opt-in budget for callers that know replay cannot duplicate tools or
# external side effects.  This is intentionally separate from transport retry:
# stream completion and terminal response status are observed after the generic
# adapter retry boundary.
REPLAY_SAFE_OUTCOME_MAX_RETRIES = 3


_REPLAY_SAFE_FAILURE_KINDS = frozenset(
    {"incomplete_response", "stream_interrupted"}
)


def _is_replay_safe_outcome_error(error: Exception) -> bool:
    # The raw type is kept as defence in depth: adapters raise it before the
    # client classifies it at its dispatch exit.  No in-tree caller sees the raw
    # shape today, so do not read this branch as evidence of a live producer.
    return (
        isinstance(error, StreamInterruptedError)
        or (
            isinstance(error, UpstreamExhaustedError)
            and error.failure_kind in _REPLAY_SAFE_FAILURE_KINDS
        )
    )


async def retry_replay_safe_llm_call(
    func: Callable[[], Awaitable[Any]],
    *,
    agent_name: str,
    model_name: str,
    llm_server: str = "",
    max_retries: int = REPLAY_SAFE_OUTCOME_MAX_RETRIES,
    retry_delay_base: float = RETRY_DELAY_BASE,
    on_retry: Optional[
        Callable[[int, int, float, Exception], Awaitable[None]]
    ] = None,
):
    """Retry one explicitly replay-safe LLM call without changing its options.

    ``func`` owns the complete request, including its current protocol and
    streaming preference.  This helper only invokes that same callable again;
    it never adds or changes ``stream``/``prefer_stream`` arguments.
    """
    retry_budget = max(0, max_retries)
    attempt = 0
    while True:
        try:
            return await func()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not _is_replay_safe_outcome_error(error):
                raise
            if attempt >= retry_budget:
                # The client classifies stream interruption before the caller
                # sees it, so match the classified shape too — otherwise the
                # bare re-raise drops the attempt count and the gate reads like
                # a single-shot failure.
                if (
                    isinstance(error, StreamInterruptedError)
                    or getattr(error, "failure_kind", None) == "stream_interrupted"
                ):
                    raise UpstreamExhaustedError(
                        f"LLM stream remained incomplete after {attempt} retries: {error}",
                        agent_name=agent_name,
                        model=model_name,
                        llm_server=llm_server,
                        failure_kind="stream_interrupted",
                    ) from error
                raise

            attempt += 1
            delay = retry_delay_base * (2 ** (attempt - 1))
            logger.warning(
                "[%s] Replay-safe LLM outcome retry %s/%s in %.1fs "
                "after %s (model=%s)",
                agent_name,
                attempt,
                retry_budget,
                delay,
                type(error).__name__,
                model_name,
            )
            if on_retry is not None:
                try:
                    await on_retry(attempt, retry_budget, delay, error)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "[%s] Failed to emit replay-safe retry notification",
                        agent_name,
                        exc_info=True,
                    )
            await asyncio.sleep(delay)

def compute_rate_limit_delay(rate_limit_attempt: int, max_retries: int) -> Optional[float]:
    """Delay (seconds) for the given 0-based 429 retry attempt, or None when the
    dedicated rate-limit budget is exhausted.

    The 429 budget is tracked separately from other error types so a burst of
    5xx / timeout / JSON errors cannot shift its index or consume its budget.
    """
    budget = min(max_retries, len(RATE_LIMIT_DELAYS))
    if rate_limit_attempt < budget:
        return float(RATE_LIMIT_DELAYS[rate_limit_attempt])
    return None

async def retry_with_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    retry_delay_base: float = RETRY_DELAY_BASE,
    agent_name: str = "unknown",
    model_name: str = "",
    llm_server: str = "",
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    on_retry: Optional[Callable[[int, int, float, str, str], Awaitable[None]]] = None,
    non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
    rate_limit_retries: Optional[int] = None,
    connection_retries: Optional[int] = None,
    before_retry: Optional[Callable[[], Awaitable[None]]] = None,
    on_context_overflow: Optional[Callable[[], Awaitable[bool]]] = None,
):
    """
    Execute an async function with exponential backoff retry on timeout/connection errors

    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts
        retry_delay_base: Base delay between retries (multiplied by attempt number)
        agent_name: Agent name for logging
        request_timeout: Hard per-attempt timeout in seconds (asyncio-level deadline).
            Fires even if the HTTP client hangs during response body reading.
        connection_retries: Optional retry budget dedicated to API connection
            failures. This lets long-generation callers keep safe transport
            retries while disabling replay of requests that may have started.
        on_retry: Optional async callback invoked before each retry sleep.
            Signature: (attempt, max_retries, delay, error_type, agent_name) -> None.
            Useful for sending progress updates to WebSocket clients during
            long retry waits. agent_name is passed through because the retry
            notice is the only trace of the stall the user sees, and a notice
            attributed to the wrong agent sends them to the wrong logs.
        before_retry: Optional async gate awaited after each retry backoff and
            immediately before the next request. Exceptions, including
            ``asyncio.CancelledError``, propagate to the caller.

    Returns:
        Result from the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    caller_max = max(0, max_retries)
    effective_max = caller_max

    json_error_count = 0
    rate_limit_attempt = 0  # independent 429 counter (see compute_rate_limit_delay)
    connection_retry_attempt = 0
    connection_retry_budget = (
        None if connection_retries is None else max(0, connection_retries)
    )
    rate_limit_budget = min(
        max_retries if rate_limit_retries is None else max(0, rate_limit_retries),
        len(RATE_LIMIT_DELAYS),
    )

    async def _notify_retry(
        attempt: int,
        delay: float,
        error_type: str,
        callback_budget: Optional[int] = None,
    ):
        """Fire the on_retry callback if provided, swallowing errors."""
        if on_retry is not None:
            try:
                await on_retry(
                    attempt,
                    max_retries if callback_budget is None else callback_budget,
                    delay,
                    error_type,
                    agent_name,
                )
            except Exception:
                pass  # Never let notification failure break retry logic

    async def _await_before_retry() -> None:
        """Wait for lifecycle permission before starting another request."""
        if before_retry is not None:
            await before_retry()

    attempt = 0
    total_attempts = 0
    while attempt <= effective_max:
        try:
            total_attempts += 1
            return await asyncio.wait_for(func(), timeout=request_timeout)
        except json.JSONDecodeError as e:
            if non_retryable_predicate is not None and non_retryable_predicate(e):
                raise
            # JSON errors get fewer retries — repeated bad JSON is likely persistent.
            # MAX_JSON_RETRIES is the retry budget; <= keeps the actual retry count
            # equal to MAX_JSON_RETRIES (total invocations == MAX_JSON_RETRIES + 1).
            json_error_count += 1
            last_exception = e
            if json_error_count <= MAX_JSON_RETRIES:
                delay = retry_delay_base * json_error_count
                logger.warning(
                    f"[{agent_name}] Invalid JSON response (retry {json_error_count}/{MAX_JSON_RETRIES}): {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await _notify_retry(attempt, delay, "JSONDecodeError")
                await asyncio.sleep(delay)
                await _await_before_retry()
            else:
                logger.error(
                    f"[{agent_name}] Invalid JSON after {json_error_count} attempts: {str(e)}"
                )
                raise
        except (AuthenticationError, PermissionDeniedError) as e:
            # Non-transient: retrying with the same credentials cannot help.
            # Surface immediately; @with_retry / planning turn this into a HITL
            # gate where the user can fix the key and click Retry.
            last_exception = e
            logger.error(
                f"[{agent_name}] Auth/permission error (not retrying): "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            break
        except RateLimitError as e:
            if non_retryable_predicate is not None and non_retryable_predicate(e):
                raise
            last_exception = e
            if _is_hard_quota_error(e):
                logger.error(
                    f"[{agent_name}] Hard quota exhausted (not retrying): {str(e)[:200]}"
                )
                break
            delay = compute_rate_limit_delay(
                rate_limit_attempt, rate_limit_budget
            )
            if delay is not None:
                logger.warning(
                    f"[{agent_name}] Rate limited (429 retry {rate_limit_attempt + 1}/{rate_limit_budget}). "
                    f"Retrying in {delay}s..."
                )
                await _notify_retry(
                    rate_limit_attempt,
                    delay,
                    "RateLimitError",
                    rate_limit_budget,
                )
                await asyncio.sleep(delay)
                rate_limit_attempt += 1
                await _await_before_retry()
                continue
            else:
                logger.error(
                    f"[{agent_name}] Rate limit exceeded after "
                    f"{rate_limit_budget} retries"
                )
                break  # stop spinning; fall through to UpstreamExhaustedError
        except Exception as e:
            if is_context_overflow(e):
                last_exception = e
                effective_max = caller_max
                if attempt >= caller_max or on_context_overflow is None:
                    break
                await _await_before_retry()
                try:
                    recovered = await on_context_overflow()
                except Exception:
                    logger.warning("[%s] Context-overflow recovery failed", agent_name,
                                   exc_info=True)
                    recovered = False
                if not recovered:
                    break
                # Recovery owns no extra attempts: consume the same retry slot
                # used by transient errors and rebuild the adapter request.
                await _notify_retry(attempt, 0.0, "ContextOverflow", caller_max)
                attempt += 1
                continue
            # An already-exhausted upstream error (e.g. raised by a nested
            # retry_with_backoff / probe loop) must not be retried or re-wrapped;
            # propagate it as-is so the workflow's HITL gate sees it.
            if isinstance(e, UpstreamExhaustedError):
                raise
            # Protocol/parameter adaptation signals have dedicated upstream
            # handlers (protocol redirect, temperature fallback, tool_choice
            # fallback). They must NEVER be retried here.
            if _is_protocol_adaptation_signal(e):
                raise
            # A strict Responses gateway has identified the exact invalid
            # input item/field. Retry would resend byte-for-byte equivalent
            # JSON, so stop immediately and fall through to the standard
            # UpstreamExhaustedError/HITL path.
            if is_responses_input_shape_error(e):
                last_exception = e
                logger.error(
                    f"[{agent_name}] Deterministic Responses input schema error "
                    f"(not retrying): {str(e)[:300]}"
                )
                break
            if non_retryable_predicate is not None and non_retryable_predicate(e):
                raise
            last_exception = e
            if (
                isinstance(e, APIConnectionError)
                and not isinstance(e, APITimeoutError)
                and connection_retry_budget is not None
            ):
                # A connection error has no upstream response or partial
                # completion to preserve, so it is safe to retry even when a
                # long request disabled ordinary replay. Keep this counter
                # independent for the same reason as the 429 budget above.
                effective_max = caller_max
                if connection_retry_attempt < connection_retry_budget:
                    delay = retry_delay_base * (connection_retry_attempt + 1)
                    logger.warning(
                        f"[{agent_name}] Connection failed "
                        f"(retry {connection_retry_attempt + 1}/"
                        f"{connection_retry_budget}). Retrying in {delay:.1f}s..."
                    )
                    await _notify_retry(
                        connection_retry_attempt,
                        delay,
                        "APIConnectionError",
                        connection_retry_budget,
                    )
                    await asyncio.sleep(delay)
                    connection_retry_attempt += 1
                    await _await_before_retry()
                    continue
                logger.error(
                    f"[{agent_name}] Connection failed after "
                    f"{total_attempts} attempts: {str(e)}"
                )
                break
            is_5xx = isinstance(e, InternalServerError)
            # A 5xx episode gets its own enlarged budget.  The enlarged
            # budget must not leak into a later deterministic error, and a
            # caller that explicitly asked for zero retries never gets an
            # implicit retry.
            if is_5xx and caller_max > 0:
                effective_max = max(caller_max, MAX_RETRIES_UPSTREAM_5XX)
            elif not is_5xx:
                effective_max = caller_max
            current_max = effective_max
            if attempt < current_max:
                delay = retry_delay_base * (attempt + 1)
                error_label = "InternalServerError(5xx)" if is_5xx else type(e).__name__
                logger.warning(
                    f"[{agent_name}] Request failed (attempt {attempt + 1}/{current_max + 1}): {error_label}: {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await _notify_retry(attempt, delay, error_label, current_max)
                await asyncio.sleep(delay)
                attempt += 1
                await _await_before_retry()
                continue
            else:
                logger.error(
                    f"[{agent_name}] Request failed after {current_max + 1} attempts: {type(e).__name__}: {str(e)}"
                )
                break

    raise UpstreamExhaustedError(
        f"[{agent_name}] Upstream LLM exhausted after "
        f"{total_attempts} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}",
        agent_name=agent_name,
        model=model_name,
        llm_server=llm_server,
    ) from last_exception
