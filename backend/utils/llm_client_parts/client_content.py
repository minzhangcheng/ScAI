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
    _TRUNCATION_RETRY_FLOOR,
    _TRUNCATION_RETRY_CEILING,
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
    _collect_reasoning_parts,
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
            cached = getattr(details, 'cached_tokens', 0) if details else 0
            if cached:
                result["cached_tokens"] = cached
            return result
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _extract_content_from_message(self, message, response=None) -> str:
        """Extract text content from a chat completion message, handling multiple formats.

        Handles: string content, list content (multimodal), None content with
        model_dump fallback, and reasoning model responses where content may be
        in the reasoning/reasoning_content field (e.g. OpenRouter deep research).
        """
        if not message:
            return ""

        raw_content = message.content

        # Case 1: String content (normal case) — only return if non-empty
        if isinstance(raw_content, str) and raw_content:
            return raw_content

        # Case 2: List content (multimodal response format)
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif "text" in part:
                        text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            if text_parts:
                return "\n".join(text_parts)

        # Case 3: None or empty — try model_dump() for Pydantic SDK objects
        content = ""
        if not raw_content:
            try:
                if hasattr(message, "model_dump"):
                    msg_data = message.model_dump()
                    if "content" in msg_data and msg_data["content"]:
                        raw = msg_data["content"]
                        if isinstance(raw, str):
                            content = raw
                        elif isinstance(raw, list):
                            content = "\n".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in raw
                            )
                    # Reasoning model fallback: content in reasoning/reasoning_content/thinking/thought
                    if not content:
                        parts = _collect_reasoning_parts(msg_data.get)
                        if parts:
                            content = "\n".join(parts)
            except Exception as e:
                logger.debug(f"[{self.agent_name}] model_dump extraction failed: {e}")

            # Fallback: check text attribute
            if not content and hasattr(message, "text"):
                content = message.text or ""
            # Fallback: check reasoning attribute directly on the object
            if not content:
                parts = _collect_reasoning_parts(lambda k: getattr(message, k, None))
                if parts:
                    content = "\n".join(parts)

        # Case 4: Other types — convert to string
        if not content and raw_content is not None:
            content = str(raw_content)

        # Check for refusal
        if not content and hasattr(message, "refusal") and message.refusal:
            content = f"[Refusal]: {message.refusal}"
            logger.warning(f"[{self.agent_name}] Request was refused: {message.refusal}")

        # Debug logging: tokens generated but content empty
        completion_tokens, has_tool_calls = self._log_empty_content_diagnostic(message, response)

        # If tokens were generated but content extraction failed, return an error
        # message instead of empty string so downstream agents can detect the failure.
        if not content and completion_tokens and completion_tokens > 0 and not has_tool_calls:
            logger.warning(
                f"[{self.agent_name}] Non-zero completion tokens ({completion_tokens}) "
                f"but empty extracted content — returning error placeholder"
            )
            return EMPTY_CONTENT_PLACEHOLDER

        return content

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
        call was made. Note this looks at the raw message.content, not the
        extracted content — the extractor may have salvaged truncated reasoning
        text, which is still not a usable answer."""
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return False
        if getattr(message, "tool_calls", None):
            return False
        return not getattr(message, "content", None)

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

        try:
            if hasattr(message, "model_dump"):
                raw_dump = message.model_dump()
                logger.info(
                    f"[{self.agent_name}] Empty content with {completion_tokens} tokens. "
                    f"Raw message keys: {list(raw_dump.keys())}, "
                    f"values preview: {str(raw_dump)[:2000]}"
                )
        except Exception as e:
            logger.debug(f"[{self.agent_name}] Failed to dump raw message for debugging: {e}")

        msg_dict = {}
        for attr_name in ("content", "refusal", "role", "reasoning", "reasoning_content", "thinking", "thought"):
            if hasattr(message, attr_name):
                val = getattr(message, attr_name)
                msg_dict[attr_name] = str(val)[:200] if val else None
        try:
            if hasattr(message, "model_dump"):
                msg_dict["_all_keys"] = list(message.model_dump().keys())
        except Exception as e:
            logger.debug(f"[{self.agent_name}] Failed to extract message keys: {e}")

        logger.warning(
            f"[{self.agent_name}] Response has {completion_tokens} completion tokens "
            f"but empty content. Message fields: {msg_dict}"
        )
        return completion_tokens, has_tool_calls
