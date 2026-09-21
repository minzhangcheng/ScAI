# Extraction of content/usage from upstream messages and empty-response diagnostics.
# Moved verbatim out of backend/utils/llm_client.py; composed into OpenAICompatibleClient.
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

_PRIVATE_REASONING_BLOCK_TYPES = frozenset({
    "reasoning", "reasoning_content", "thinking", "thought",
    "redacted_thinking", "reasoning_details", "analysis",
})
_PRIVATE_REASONING_BLOCK_KEYS = frozenset({
    "reasoning", "reasoning_content", "reasoning_details", "thinking", "thought",
})


def _is_private_reasoning_block(part: dict) -> bool:
    block_type = str(part.get("type", "") or "").strip().lower()
    return (
        block_type in _PRIVATE_REASONING_BLOCK_TYPES
        or any(
            marker in block_type
            for marker in ("reason", "think", "thought")
        )
        or any(key in part for key in _PRIVATE_REASONING_BLOCK_KEYS)
    )


def _extract_answer_content(value) -> str:
    """Extract only explicit public text blocks from one content payload."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return ""
    text_parts = []
    for part in value:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        block_type = str(part.get("type", "") or "").strip().lower()
        if _is_private_reasoning_block(part):
            continue
        if block_type in {"text", "output_text"}:
            text_parts.append(str(part.get("text", "") or ""))
        elif not block_type and "text" in part:
            text_parts.append(str(part.get("text", "") or ""))
    return "\n".join(part for part in text_parts if part)

# Implementations live in ./llm_client_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.

from .shared import (  # noqa: F401
    base64,
    html,
    httpx,
    inspect,
    asyncio,
    json,
    logging,
    re,
    time,
    uuid4,
    urlparse,
    urlunparse,
    List,
    Dict,
    Any,
    Optional,
    AsyncGenerator,
    Union,
    Callable,
    Awaitable,
    Tuple,
    dataclass,
    field,
    AsyncOpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    APIError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    get_llm_logger,
    LLMInteraction,
    ENGLISH_ENFORCEMENT,
    _inject_english_enforcement,
    detect_image_mime,
    _detect_image_mime_type,
    UpstreamExhaustedError,
    EmptyResponseError,
    LLMResponse,
    ToolCall,
    _tool_calls_to_raw_assistant,
    LLMToolResponse,
    _OPENAI_REASONING_PREFIXES,
    _GEMINI3_PREFIX,
    _NO_CUSTOM_TEMP_MODELS,
    _extract_model_name,
    _WEB_SEARCH_TOOL_NAMES,
    _is_deep_research_model,
    _is_o_series,
    _get_effective_temperature,
    _inject_web_search_if_needed,
    _is_unknown_web_search_tool_error,
    _swap_web_search_tool_name,
    _is_unsupported_background_error,
    is_temperature_error,
    is_tool_choice_error,
    is_responses_endpoint_absent,
    is_responses_chat_only_model,
    is_endpoint_absent,
    is_responses_only_model,
    _is_protocol_adaptation_signal,
    _RESPONSES_INPUT_PARAMETER_RE,
    is_responses_input_shape_error,
    is_transient_probe_error,
    _HARD_QUOTA_KEYWORDS,
    _TRANSIENT_RATE_MARKERS,
    _is_hard_quota_error,
    MAX_RETRIES,
    MAX_RETRIES_UPSTREAM_5XX,
    RETRY_DELAY_BASE,
    RATE_LIMIT_DELAYS,
    RETRYABLE_EXCEPTIONS,
    MAX_JSON_RETRIES,
    EMPTY_CONTENT_PLACEHOLDER,
    EMPTY_RESPONSE_MAX_RETRIES,
    compute_rate_limit_delay,
    retry_with_backoff,
    _NO_BACKGROUND_ENDPOINT,
    _NO_RESPONSES_ENDPOINT,
    _PROBE_CACHE,
    _PROBE_LOCKS,
    _url_path_suffix_protocol,
    _DEFAULT_PROTOCOL_ORDER,
    PROBE_RETRY_DELAYS,
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
    _ANTHROPIC_DEFAULT_MAX_TOKENS,
    _AnthropicAdapter,
    _ResponsesAdapter,
    _DR_BACKGROUND_POLL_INTERVAL,
    _DR_BACKGROUND_MAX_WAIT,
    _DR_BACKGROUND_MAX_POLL_ERRORS,
    _DR_BACKGROUND_TERMINAL_FAILED,
    GoogleDeepResearchClient,
    _recover_stringified_tool_call,
    _REASONING_ECHO_KEYS,
    _reasoning_echo_allowed,
    _build_assistant_echo,
)


class ClientContentMixin:
    """Extraction of content/usage from upstream messages and empty-response diagnostics."""

    async def _emit_log_response(self, **kwargs) -> None:
        """Persist one response exactly once.

        Production uses the async logger to keep file I/O off the event loop.
        Older/custom test loggers may expose only the synchronous method, which
        remains a fallback rather than a second write.
        """
        interaction = kwargs.get("interaction")
        cache_context = getattr(interaction, "cache", None)
        if isinstance(cache_context, dict):
            protocol = cache_context.get("protocol")
            modality = cache_context.get("modality", "text")
            model = getattr(interaction, "model", None)
            if protocol and model:
                interaction.cache = self._prompt_cache_log_context(
                    model, protocol, modality
                )

        async_log_response = getattr(self.llm_logger, "log_response_async", None)
        if callable(async_log_response):
            result = async_log_response(**kwargs)
            if inspect.isawaitable(result):
                await result
                return

        sync_log_response = getattr(self.llm_logger, "log_response", None)
        if callable(sync_log_response):
            sync_log_response(**kwargs)

    @staticmethod
    def _extract_usage(response) -> Dict[str, int]:
        """Extract usage dict from an LLM response."""
        if response and response.usage:
            result = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
            details = getattr(response.usage, 'prompt_tokens_details', None)
            missing = object()

            def reported_value(container, key):
                """Read only fields explicitly present in the provider payload."""
                if container is None:
                    return missing
                if isinstance(container, dict):
                    return container.get(key, missing)
                model_fields_set = getattr(container, "model_fields_set", missing)
                if model_fields_set is not missing:
                    if key not in model_fields_set:
                        return missing
                    return getattr(container, key, missing)
                model_dump = getattr(type(container), "model_dump", None)
                if callable(model_dump):
                    try:
                        payload = model_dump(container)
                    except (TypeError, ValueError):
                        payload = None
                    if isinstance(payload, dict) and key in payload:
                        return payload[key]
                attributes = getattr(container, "__dict__", {})
                return attributes.get(key, missing)

            cached = reported_value(details, "cached_tokens")
            cache_write = reported_value(details, "cache_write_tokens")
            top_level_cached = reported_value(
                response.usage, "cache_read_input_tokens"
            )
            top_level_cache_write = reported_value(
                response.usage, "cache_creation_input_tokens"
            )
            if top_level_cached is not missing and top_level_cached is not None:
                if cached is missing or cached is None:
                    cached = top_level_cached
            if cached is not missing and cached is not None:
                # Preserve an explicitly reported zero so report generation can
                # distinguish a measured 0% hit rate from unavailable metrics.
                result["cached_tokens"] = cached or 0
            if (
                top_level_cache_write is not missing
                and top_level_cache_write is not None
                and (cache_write is missing or cache_write is None)
            ):
                cache_write = top_level_cache_write
            if cache_write is not missing and cache_write is not None:
                result["cache_write_tokens"] = cache_write or 0
            return result
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _extract_content_from_message(self, message, response=None) -> str:
        """Extract text content from a chat completion message, handling multiple formats.

        Handles string content, list content (multimodal), and an SDK
        ``model_dump`` fallback. Provider reasoning/thinking fields are
        deliberately excluded: they are private scratch space, not an answer.
        """
        if not message:
            return ""

        raw_content = message.content

        # Case 1: String content (normal case) — only return if non-empty
        if isinstance(raw_content, str) and raw_content:
            return raw_content

        # Case 2: List content (multimodal response format)
        if isinstance(raw_content, list):
            public_text = _extract_answer_content(raw_content)
            if public_text:
                return public_text

        # Case 3: None or empty — try model_dump() for Pydantic SDK objects
        content = ""
        if not raw_content:
            try:
                if hasattr(message, "model_dump"):
                    msg_data = message.model_dump()
                    if "content" in msg_data and msg_data["content"]:
                        content = _extract_answer_content(msg_data["content"])
            except Exception as e:
                logger.debug(
                    "[%s] model_dump extraction failed (type=%s)",
                    self.agent_name,
                    type(e).__name__,
                )

            # Fallback: check text attribute
            if not content and hasattr(message, "text"):
                text_value = message.text
                if isinstance(text_value, str):
                    content = text_value

        # Other object shapes are not answer channels. In particular, never
        # stringify a dict that may carry provider-private reasoning fields.

        # Check for refusal
        if not content and hasattr(message, "refusal") and message.refusal:
            content = f"[Refusal]: {message.refusal}"
            logger.warning(f"[{self.agent_name}] Request was refused: {message.refusal}")

        if content:
            return content

        # Debug logging: tokens generated and final extracted content is empty.
        completion_tokens, has_tool_calls = self._log_empty_content_diagnostic(message, response)

        # If tokens were generated but content extraction failed, return an error
        # message instead of empty string so downstream agents can detect the failure.
        if not content and completion_tokens and completion_tokens > 0 and not has_tool_calls:
            logger.warning(
                f"[{self.agent_name}] Non-zero completion tokens ({completion_tokens}) "
                f"but empty extracted content — returning error placeholder"
            )
            return EMPTY_CONTENT_PLACEHOLDER

        return ""

    @staticmethod
    def _response_length_truncated(response) -> bool:
        """True when the response was cut off because it hit the token cap
        (finish_reason == 'length') — the signature of a reasoning model that
        burned its whole budget on reasoning_content."""
        try:
            return response.choices[0].finish_reason == "length"
        except (AttributeError, IndexError, TypeError):
            return False

    @staticmethod
    def _answer_content_empty(response) -> bool:
        """True when the assistant's *answer* content field is empty and no tool
        call was made. Reasoning fields are intentionally irrelevant: only the
        provider's answer channel counts as usable output."""
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return False
        if getattr(message, "tool_calls", None):
            return False
        return not _extract_answer_content(getattr(message, "content", None))

    @staticmethod
    def _response_has_private_reasoning(response) -> bool:
        """Detect a reasoning-only provider reply without exposing its text."""
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return False

        raw_content = getattr(message, "content", None)
        content_parts = raw_content if isinstance(raw_content, list) else [raw_content]
        if any(
            isinstance(part, dict) and _is_private_reasoning_block(part)
            for part in content_parts
        ):
            return True
        if any(
            bool(getattr(message, key, None))
            for key in _PRIVATE_REASONING_BLOCK_KEYS
        ):
            return True
        try:
            dumped = message.model_dump() if hasattr(message, "model_dump") else {}
        except Exception:
            return False
        return isinstance(dumped, dict) and any(
            bool(dumped.get(key)) for key in _PRIVATE_REASONING_BLOCK_KEYS
        )

    @staticmethod
    def _answer_empty_no_output(response) -> bool:
        """True when the assistant answer is empty, no tool call was made, AND
        the provider generated zero completion tokens. This is the 'produced
        nothing' failure — separate from a legitimate empty answer that still
        consumed tokens (which we do not retry, to avoid looping on a model that
        deliberately returned nothing)."""
        if not ClientContentMixin._answer_content_empty(response):
            return False
        try:
            return (response.usage.completion_tokens or 0) == 0
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _llm_response_empty_no_output(result) -> bool:
        """True when an adapter-backed LLMResponse/LLMToolResponse returned no
        text, made no tool call, and reported zero generated tokens."""
        if not result or getattr(result, "content", None):
            return False
        if getattr(result, "tool_calls", None):
            return False
        usage = getattr(result, "usage", None) or {}
        if not isinstance(usage, dict):
            return False
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        return completion == 0

    def _log_empty_content_diagnostic(self, message, response) -> tuple:
        """Log diagnostic info when content extraction yielded empty string.

        Returns (completion_tokens, has_tool_calls) so the caller can decide
        whether to return an error placeholder.
        """
        completion_tokens = 0
        has_tool_calls = False
        if not response or not response.usage:
            return completion_tokens, has_tool_calls
        completion_tokens = response.usage.completion_tokens or 0
        if completion_tokens <= 0:
            return completion_tokens, has_tool_calls

        has_tool_calls = bool(getattr(message, "tool_calls", None))
        if has_tool_calls:
            return completion_tokens, has_tool_calls

        msg_dict = {}
        for attr_name in ("content", "refusal", "role", "reasoning", "reasoning_content", "thinking", "thought"):
            if hasattr(message, attr_name):
                val = getattr(message, attr_name)
                msg_dict[attr_name] = bool(val)

        logger.warning(
            f"[{self.agent_name}] Response has {completion_tokens} completion tokens "
            f"but empty content. Message fields: {msg_dict}"
        )
        return completion_tokens, has_tool_calls
