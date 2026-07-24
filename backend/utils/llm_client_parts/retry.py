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

from .error_signals import _is_hard_quota_error, _is_protocol_adaptation_signal, is_responses_input_shape_error
from .errors import UpstreamExhaustedError


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

# Floor for the one-shot retry budget when a reasoning model truncates an empty
# response — large enough to hold reasoning_content plus a short JSON answer.
_TRUNCATION_RETRY_FLOOR = 4096

# Ceiling so an already-large budget that truncates empty doesn't balloon into an
# oversized request the endpoint may reject.
_TRUNCATION_RETRY_CEILING = 32768

# A provider/proxy that replies HTTP 200 with an empty answer AND
# completion_tokens == 0 generated nothing at all — a transient upstream blip
# (silent proxy failure, dropped stream). Retry the same request with backoff
# before handing the empty result to callers, which otherwise treat "" as a
# valid answer (e.g. the supervisor report path saved a 0-byte "completed"
# report). Distinct from the reasoning-truncation guard, which fires only when
# finish_reason == "length".
EMPTY_RESPONSE_MAX_RETRIES = 3

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
    request_timeout: float = 300.0,
    on_retry: Optional[Callable[[int, int, float, str], Awaitable[None]]] = None,
    non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
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
        on_retry: Optional async callback invoked before each retry sleep.
            Signature: (attempt, max_retries, delay, error_type) -> None.
            Useful for sending progress updates to WebSocket clients during
            long retry waits.

    Returns:
        Result from the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    effective_max = max_retries

    json_error_count = 0
    rate_limit_attempt = 0  # independent 429 counter (see compute_rate_limit_delay)

    async def _notify_retry(attempt: int, delay: float, error_type: str):
        """Fire the on_retry callback if provided, swallowing errors."""
        if on_retry is not None:
            try:
                await on_retry(attempt, max_retries, delay, error_type)
            except Exception:
                pass  # Never let notification failure break retry logic

    for attempt in range(effective_max + 1):
        try:
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
            delay = compute_rate_limit_delay(rate_limit_attempt, max_retries)
            if delay is not None:
                budget = min(max_retries, len(RATE_LIMIT_DELAYS))
                logger.warning(
                    f"[{agent_name}] Rate limited (429 retry {rate_limit_attempt + 1}/{budget}). "
                    f"Retrying in {delay}s..."
                )
                await _notify_retry(attempt, delay, "RateLimitError")
                await asyncio.sleep(delay)
                rate_limit_attempt += 1
            else:
                budget = min(max_retries, len(RATE_LIMIT_DELAYS))
                logger.error(f"[{agent_name}] Rate limit exceeded after {budget} retries")
                break  # stop spinning; fall through to UpstreamExhaustedError
        except Exception as e:
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
            is_5xx = isinstance(e, InternalServerError)
            if is_5xx:
                effective_max = max(effective_max, MAX_RETRIES_UPSTREAM_5XX)
            current_max = effective_max
            if attempt < current_max:
                delay = retry_delay_base * (attempt + 1)
                error_label = "InternalServerError(5xx)" if is_5xx else type(e).__name__
                logger.warning(
                    f"[{agent_name}] Request failed (attempt {attempt + 1}/{current_max + 1}): {error_label}: {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await _notify_retry(attempt, delay, error_label)
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] Request failed after {current_max + 1} attempts: {type(e).__name__}: {str(e)}"
                )

    raise UpstreamExhaustedError(
        f"[{agent_name}] Upstream LLM exhausted after {effective_max + 1} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}",
        agent_name=agent_name,
        model=model_name,
        llm_server=llm_server,
    ) from last_exception
