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
from ..formatting import loads_llm_json


# Implementations live in ./llm_client_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.
from ..llm_client_parts.media import (  # noqa: F401
    ENGLISH_ENFORCEMENT,
    VisionPayloadError,
    _inject_english_enforcement,
    detect_image_mime,
    _detect_image_mime_type,
    normalize_image_input,
)
from ..llm_client_parts.errors import (  # noqa: F401
    UpstreamExhaustedError,
    ToolProtocolError,
    EmptyResponseError,
    StreamInterruptedError,
    StreamNotStartedError,
    ReasoningOnlyStreamError,
)
from ..llm_client_parts.responses import (  # noqa: F401
    LLMResponse,
    ToolCall,
    _tool_calls_to_raw_assistant,
    LLMToolResponse,
)
from ..llm_client_parts.model_traits import (  # noqa: F401
    _OPENAI_REASONING_PREFIXES,
    _GEMINI3_PREFIX,
    _NO_CUSTOM_TEMP_MODELS,
    _extract_model_name,
    _WEB_SEARCH_TOOL_NAMES,
    _is_deep_research_model,
    _is_o_series,
    _get_effective_temperature,
)
from ..llm_client_parts.error_signals import (  # noqa: F401
    _inject_web_search_if_needed,
    _is_unknown_web_search_tool_error,
    _swap_web_search_tool_name,
    _is_unsupported_background_error,
    is_stream_unsupported_error,
    is_stream_options_unsupported_error,
    is_responses_stream_unsupported_error,
    is_temperature_error,
    is_tool_choice_error,
    is_responses_endpoint_absent,
    is_responses_chat_only_model,
    is_endpoint_absent,
    is_responses_only_model,
    is_responses_vision_shape_error,
    is_vision_request_shape_error,
    is_vision_capability_mismatch,
    is_max_tokens_field_error,
    max_tokens_field_from_error,
    _is_protocol_adaptation_signal,
    _RESPONSES_INPUT_PARAMETER_RE,
    is_responses_input_shape_error,
    is_transient_probe_error,
    _HARD_QUOTA_KEYWORDS,
    _TRANSIENT_RATE_MARKERS,
    _is_hard_quota_error,
)
from ..llm_client_parts.retry import (  # noqa: F401
    DEFAULT_REQUEST_TIMEOUT,
    MAX_RETRIES,
    MAX_RETRIES_UPSTREAM_5XX,
    RETRY_DELAY_BASE,
    RATE_LIMIT_DELAYS,
    RETRYABLE_EXCEPTIONS,
    MAX_JSON_RETRIES,
    EMPTY_CONTENT_PLACEHOLDER,
    EMPTY_RESPONSE_MAX_RETRIES,
    REPLAY_SAFE_OUTCOME_MAX_RETRIES,
    compute_rate_limit_delay,
    retry_replay_safe_llm_call,
    retry_with_backoff,
)
from ..llm_client_parts.protocol import (  # noqa: F401
    _NO_BACKGROUND_ENDPOINT,
    _NO_RESPONSES_ENDPOINT,
    _NO_RESPONSES_VISION_STREAM,
    _NO_TEXT_STREAM,
    _NO_CHAT_STREAM_OPTIONS,
    _PROBE_CACHE,
    _PROBE_LOCKS,
    _url_path_suffix_protocol,
    _DEFAULT_PROTOCOL_ORDER,
    PROBE_RETRY_DELAYS,
    TEXT_MODALITY,
    CACHED_TEXT_MODALITY,
    TOOLS_MODALITY,
    VISION_MODALITY,
    _probe_key,
    _keyword_protocol_hints,
    _route_protocol_order,
    _route_protocol,
    _strip_path_suffix,
    _openai_sdk_base_url,
    _anthropic_sdk_base_url,
    _responses_messages_to_input,
    _responses_translate_tools,
    _responses_translate_tool_choice,
    _anthropic_messages_translate,
    _anthropic_translate_tools,
    _anthropic_translate_tool_choice,
    _parse_anthropic_dump,
    _anthropic_restore_thinking,
    _parse_responses_dump,
)
from ..llm_client_parts.adapters import (  # noqa: F401
    _ANTHROPIC_DEFAULT_MAX_TOKENS,
    _StartedStream,
    _start_stream,
    stream_with_fallback,
    _AnthropicAdapter,
    _ResponsesAdapter,
)
from ..llm_client_parts.google import (  # noqa: F401
    _DR_BACKGROUND_POLL_INTERVAL,
    _DR_BACKGROUND_MAX_WAIT,
    _DR_BACKGROUND_MAX_POLL_ERRORS,
    _DR_BACKGROUND_TERMINAL_FAILED,
    GoogleDeepResearchClient,
)


def _recover_stringified_tool_call(
    content: Optional[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Attempt to recover a tool call that an upstream gateway stringified into content.

    Some gateways (notably litellm proxying n1n.ai for Claude / Gemini) lose
    the structured tool_calls field and put the tool_use JSON into
    message.content as a string. We recognize three shapes:

      - {"tool": NAME, "parameters": {...}}                  (n1n.ai stringified)
      - {"name": NAME, "arguments": {...}}                   (OpenAI stringified)
      - {"type": "tool_use", "name": NAME, "input": {...}}   (Anthropic native)

    The payload may be wrapped in a ```json ... ``` markdown fence. Returns
    (name, arguments_dict) on success or None if content does not look like
    a stringified tool call.
    """
    if not content:
        return None
    s = content.strip()
    if not s:
        return None

    # Strip a single ```json ... ``` (or plain ``` ... ```) fence if present
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline == -1:
            return None
        s = s[first_newline + 1 :]
        if s.rstrip().endswith("```"):
            s = s[:-3]
        s = s.strip()

    if not s.startswith("{"):
        return None

    try:
        obj = loads_llm_json(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    name = obj.get("tool") or obj.get("name")
    args = obj.get("parameters")
    if args is None:
        args = obj.get("arguments")
    if args is None:
        args = obj.get("input")

    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        return None

    return name, args


def _collect_reasoning_parts(getter) -> list:
    """Collect non-empty reasoning fields from a message-like object.

    ``getter`` is a callable taking a field name and returning the value
    (e.g. ``msg_data.get`` for dicts or ``lambda k: getattr(obj, k, None)``).
    Handles str / dict / list element types robustly and concatenates all
    non-empty fields rather than returning only the first match.
    """
    parts: list = []
    for field in ("reasoning", "reasoning_content", "thinking", "thought"):
        val = getter(field)
        if not val:
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for elem in val:
                if isinstance(elem, str):
                    if elem:
                        parts.append(elem)
                elif isinstance(elem, dict):
                    text = elem.get("text") or elem.get("content") or ""
                    if text:
                        parts.append(str(text))
                else:
                    s = str(elem)
                    if s:
                        parts.append(s)
        elif isinstance(val, dict):
            text = val.get("text") or val.get("content") or ""
            if text:
                parts.append(str(text))
        else:
            s = str(val)
            if s:
                parts.append(s)
    return [p for p in parts if p]


# Reasoning fields a thinking-mode provider may attach to an assistant turn.
# These must be echoed back VERBATIM (original key + original structure, be it
# a plain string or a signature-carrying array) on tool-call turns. See:
#   - https://api-docs.deepseek.com/guides/thinking_mode
#   - https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/passing-back-reasoning_content
_REASONING_ECHO_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought")


def _reasoning_echo_allowed(model: str) -> bool:
    """Whether reasoning fields may be echoed back for this model.

    DeepSeek R1 (deepseek-reasoner) rejects multi-turn requests that include
    reasoning_content with HTTP 400, so it must never be echoed. Every other
    known reasoning provider either requires the echo (DeepSeek v4, MiMo,
    Anthropic/OpenRouter thinking) or ignores unknown fields — so default True.

    Detection uses substring matching on the model name:
    - "reasoner" catches the deepseek-reasoner endpoint name
    - "-r1" catches deepseek-r1 variants (note: could also match *-r10 etc.,
      but no such model exists today)
    - "deepseek-r1" is a redundant catch for the canonical name

    DeepSeek v4 / v4-pro do NOT contain any of these substrings, so they
    correctly return True (echo is required for their tool-call turns).
    """
    m = (model or "").lower()
    if "reasoner" in m or "-r1" in m or "deepseek-r1" in m:
        return False
    return True


def _build_assistant_echo(msg_dict: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Build the assistant message to replay into multi-turn history.

    Reasoning fields are preserved verbatim ONLY when (a) the model permits it
    and (b) the turn contains tool_calls — matching the official contract that
    scopes the echo requirement to tool-call turns, not by recency.
    """
    out: Dict[str, Any] = {
        "role": "assistant",
        "content": msg_dict.get("content"),
        "tool_calls": msg_dict.get("tool_calls"),
    }
    if out.get("tool_calls") and _reasoning_echo_allowed(model):
        for key in _REASONING_ECHO_KEYS:
            val = msg_dict.get(key)
            if val:  # truthy guard drops "" / None / [] uniformly
                out[key] = val
    return out
