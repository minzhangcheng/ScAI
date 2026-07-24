# Classification of upstream error responses into adaptation signals.
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

from .errors import UpstreamExhaustedError
from .model_traits import _WEB_SEARCH_TOOL_NAMES, _is_deep_research_model


def _inject_web_search_if_needed(model: str, tools: list) -> list:
    """Auto-inject a built-in web search tool for deep-research models that
    require one. No-op for non-DR models or when a search-tool variant is
    already present. Preserves caller-provided tools."""
    if not _is_deep_research_model(model):
        return tools
    existing = {t.get("type") for t in (tools or []) if isinstance(t, dict)}
    for name in _WEB_SEARCH_TOOL_NAMES:
        if name in existing:
            return tools
    return list(tools or []) + [{"type": _WEB_SEARCH_TOOL_NAMES[0]}]

def _is_unknown_web_search_tool_error(error: Exception) -> bool:
    """True when the error indicates an unrecognized tool type that is
    one of the _WEB_SEARCH_TOOL_NAMES variants."""
    s = str(error).lower()
    if not any(kw in s for kw in (
        "unknown tool", "unsupported tool", "not a valid tool",
        "tool not found", "unrecognized tool",
    )):
        return False
    for name in _WEB_SEARCH_TOOL_NAMES:
        if name.lower() in s:
            return True
    return False

def _swap_web_search_tool_name(tools: list, from_name: str, to_name: str) -> list:
    """Return a new list with from_name replaced by to_name in dict type fields."""
    return [
        {**t, "type": to_name} if isinstance(t, dict) and t.get("type") == from_name else t
        for t in tools
    ]

def _is_unsupported_background_error(error: Exception) -> bool:
    """True when a provider rejects the Responses `background` parameter
    (a gateway that doesn't implement background mode). String-based (like
    _is_unknown_web_search_tool_error) to stay testable without constructing
    SDK error objects: requires the word 'background' AND a rejection keyword,
    so prose that merely mentions a background does not match."""
    s = str(error).lower()
    if "background" not in s:
        return False
    return any(kw in s for kw in (
        "unknown parameter", "unsupported parameter", "unsupported",
        "not supported", "unexpected", "not allowed", "invalid parameter",
    ))

def is_temperature_error(error: Exception) -> bool:
    """Check if a BadRequestError is related to the temperature parameter."""
    if not isinstance(error, BadRequestError):
        return False

    error_str = str(error).lower()
    temperature_keywords = [
        "temperature",
        "only 1 is allowed",
        "invalid temperature",
    ]
    return any(kw in error_str for kw in temperature_keywords)

def is_tool_choice_error(error: Exception) -> bool:
    """Check if a BadRequestError is related to the tool_choice parameter.

    Some OpenAI-compatible providers reject `tool_choice="required"` (or
    a specific-function dict) with a BadRequestError mentioning the
    parameter by name. Match conservatively: only the explicit
    `tool_choice` / `tool choice` token, not the unrelated legacy
    `function_call` surface.
    """
    if not isinstance(error, BadRequestError):
        return False
    error_str = str(error).lower()
    return "tool_choice" in error_str or "tool choice" in error_str

def is_responses_endpoint_absent(error: Exception) -> bool:
    """True when the Responses endpoint itself is absent on this server.

    This is a base_url-level property. A True result may populate
    _NO_RESPONSES_ENDPOINT because every model on this base URL would fail the
    Responses route.
    """
    if isinstance(error, NotFoundError):
        return True
    s = str(error).lower()
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            s = " ".join([
                s,
                str(err.get("message", "")).lower(),
                str(err.get("code", "")).lower(),
                str(err.get("type", "")).lower(),
            ])
    if "not implemented" in s and ("responses" in s or "local:convert_request_failed" in s):
        return True
    route_markers = (
        "cannot post",
        "unknown request url",
        "no route",
        "404 page not found",
        "405 method not allowed",
    )
    return any(kw in s for kw in route_markers)

def is_responses_chat_only_model(error: Exception) -> bool:
    """True when Responses rejects this specific model and tells us to use
    chat/completions. This is model-level and must not mark the whole base_url
    as missing the Responses endpoint."""
    return "only supported in v1/chat/completions" in str(error).lower()

def is_endpoint_absent(error: Exception) -> bool:
    """Compatibility wrapper for callers/tests that only need to know whether
    Responses should fall back to chat for this request."""
    return is_responses_endpoint_absent(error) or is_responses_chat_only_model(error)

def is_responses_only_model(error: Exception) -> bool:
    """True when a chat/completions error indicates the model is only served
    via the Responses API (reactive chat -> responses fallback)."""
    return "only supported in v1/responses" in str(error).lower()

def _is_protocol_adaptation_signal(error: Exception) -> bool:
    """Errors that the protocol dispatcher adapts to (switch protocol / drop a
    parameter) rather than retrying — they must always propagate to their
    dedicated handlers and never be caught by generic retry logic."""
    return (is_responses_endpoint_absent(error)
            or is_responses_chat_only_model(error)
            or is_responses_only_model(error)
            or is_temperature_error(error)
            or is_tool_choice_error(error))

_RESPONSES_INPUT_PARAMETER_RE = re.compile(r"\binput\[\d+\]\.")

def is_responses_input_shape_error(error: Exception) -> bool:
    """Return True for deterministic Responses ``input[]`` schema errors.

    These errors identify a concrete invalid field path such as
    ``input[19].tool_calls``. Repeating the same serialized request cannot
    recover, but the caller should still receive ``UpstreamExhaustedError`` so
    the workflow enters its normal Retry/Skip/Terminate HITL gate.
    """
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None and status_code != 400:
        return False
    message = str(error).lower()
    if not _RESPONSES_INPUT_PARAMETER_RE.search(message):
        return False
    return any(marker in message for marker in (
        "unknown parameter",
        "unsupported parameter",
        "unrecognized parameter",
        "unexpected parameter",
        "extra inputs are not permitted",
    ))

def is_transient_probe_error(error: Exception) -> bool:
    """True when an error is a transient upstream blip (retry the SAME protocol),
    as opposed to a deterministic protocol/parameter adaptation signal (switch
    protocol immediately or fall back to a different parameter set).

    Uses a blacklist model: only protocol/parameter adaptation signals return
    False (non-retryable). EVERYTHING else — including unknown/unexpected error
    types and error codes — is treated as potentially transient and retried.
    This is a deliberate design choice: upstream LLM providers return error codes
    inconsistently, so we cannot maintain a reliable whitelist. Retry exhaustion
    always produces UpstreamExhaustedError, which the workflow layer turns into
    a user-facing HITL gate (retry/skip/stop).
    """
    if isinstance(error, UpstreamExhaustedError):
        return False
    if _is_protocol_adaptation_signal(error):
        return False
    if is_responses_input_shape_error(error):
        return False
    return True

# Hard-quota markers: a 429 carrying any of these never recovers by waiting
# (monthly budget / billing exhausted), so we surface it immediately.
_HARD_QUOTA_KEYWORDS = (
    "monthly max",
    "auto reload",
    "insufficient_quota",
    "exceeded your current quota",
    "billing hard limit",
)

# Transient-limiter markers: per-window rate limits DO recover by waiting, so a
# message carrying any of these is never treated as a hard quota even if it also
# contains the word "quota" (Google's wording does).
_TRANSIENT_RATE_MARKERS = (
    "per minute",
    "per second",
    "per day",
    "per region",
    "requests per",
    "tokens per",
)

def _is_hard_quota_error(error: Exception) -> bool:
    """True when a 429 is a non-recoverable quota exhaustion (not transient TPM).

    Prefers the SDK error code (OpenAI sets ``insufficient_quota`` for hard
    caps); falls back to narrowed keyword matching that explicitly excludes
    per-window limiter text such as Google's "per minute per region".
    """
    if getattr(error, "code", None) == "insufficient_quota":
        return True
    s = str(error).lower()
    if any(m in s for m in _TRANSIENT_RATE_MARKERS):
        return False
    return any(k in s for k in _HARD_QUOTA_KEYWORDS)
