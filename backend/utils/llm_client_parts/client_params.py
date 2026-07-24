# Per-model request parameters, temperature fallback, and retry kwargs.
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


class ClientParamsMixin:
    """Per-model request parameters, temperature fallback, and retry kwargs."""

    def _apply_model_params(self, params: Dict[str, Any], model: str, temperature: Optional[float], max_tokens: Optional[int], skip_temperature: bool = False) -> None:
        """Apply temperature and max_tokens to request params based on model type.

        Args:
            params: Request parameters dict to modify in-place
            model: Model identifier
            temperature: Requested temperature (may be None)
            max_tokens: Maximum tokens to generate
            skip_temperature: If True, skip setting temperature entirely (for fallback mode)
        """
        if skip_temperature:
            # Fallback mode: don't set temperature, but keep max_tokens
            if max_tokens is not None:
                model_name = _extract_model_name(model)
                # P3-4: honor previously discovered field override before falling
                # back to the static _is_o_series heuristic.
                override = self._max_tokens_field_overrides.get(model_name)
                if override:
                    field = override
                elif _is_o_series(model_name):
                    field = "max_completion_tokens"
                else:
                    field = "max_tokens"
                params[field] = max_tokens
            logger.info(f"[{self.agent_name}] Temperature skipped for {model} (using server default)")
            return

        effective_temp = _get_effective_temperature(model, temperature)
        if effective_temp is not None:
            params["temperature"] = effective_temp
        elif temperature is not None:
            logger.info(f"[{self.agent_name}] Temperature omitted for {model} (not supported)")

        # max_tokens: only OpenAI o-series uses max_completion_tokens
        model_name = _extract_model_name(model)
        if max_tokens is not None:
            # P3-4: honor previously discovered field override before falling
            # back to the static _is_o_series heuristic. This lets the client
            # adapt to provider routes (e.g. openrouter/openai/o3-mini) that
            # the heuristic misclassifies.
            override = self._max_tokens_field_overrides.get(model_name)
            if override:
                field = override
            elif _is_o_series(model_name):
                field = "max_completion_tokens"
            else:
                field = "max_tokens"
            params[field] = max_tokens

        if effective_temp is not None and effective_temp != temperature:
            logger.info(f"[{self.agent_name}] Temperature overridden: {temperature} -> {effective_temp} for {model}")

    async def _call_with_temp_fallback(
        self,
        make_request: Callable[[], Awaitable[Any]],
        make_request_no_temp: Callable[[], Awaitable[Any]],
        model_key: str,
        retry_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Execute a request with automatic fallback to no-temperature mode if needed.

        If the request fails with a BadRequestError related to the temperature
        parameter, the model is added to _temp_unsupported_models and the request
        is retried without the temperature parameter.
        """
        # If model is already known to reject temperature, skip directly.
        # P2-1: serialize read under the lock to avoid racing with a writer.
        self._ensure_cache_locks()
        async with self._temp_cache_lock:
            cached_no_temp = model_key in self._temp_unsupported_models
        if cached_no_temp:
            return await retry_with_backoff(make_request_no_temp, **retry_kwargs)
        try:
            return await retry_with_backoff(make_request, **retry_kwargs)
        except BadRequestError as e:
            if is_temperature_error(e):
                logger.warning(
                    f"[{self.agent_name}] Temperature rejected by {model_key}, "
                    f"retrying without temperature..."
                )
                async with self._temp_cache_lock:
                    if model_key not in self._temp_unsupported_models:
                        self._temp_unsupported_models.add(model_key)
                return await retry_with_backoff(make_request_no_temp, **retry_kwargs)
            else:
                raise

    def _build_retry_kwargs(
        self,
        model: str,
        request_timeout: Optional[float] = None,
        max_retries_override: Optional[int] = None,
        non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "agent_name": self.agent_name,
            "model_name": model,
            "llm_server": self.base_url,
        }
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        if max_retries_override is not None:
            kwargs["max_retries"] = max_retries_override
        elif request_timeout is not None and request_timeout > 300:
            kwargs["max_retries"] = 0
        if self.on_retry is not None:
            kwargs["on_retry"] = self.on_retry
        if non_retryable_predicate is not None:
            kwargs["non_retryable_predicate"] = non_retryable_predicate
        return kwargs

    def _build_request_pair(
        self,
        base_params: Dict[str, Any],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        request_timeout: Optional[float] = None,
    ) -> tuple:
        """构建请求闭包对：(标准请求, 无temperature备选请求)."""
        async def make_request():
            params = {**base_params}
            self._apply_model_params(params, model, temperature, max_tokens)
            if request_timeout is not None:
                params["timeout"] = request_timeout
            return await self.client.chat.completions.create(**params)

        async def make_request_no_temp():
            params = {**base_params}
            self._apply_model_params(params, model, temperature, max_tokens, skip_temperature=True)
            if request_timeout is not None:
                params["timeout"] = request_timeout
            return await self.client.chat.completions.create(**params)

        return make_request, make_request_no_temp
