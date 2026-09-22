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

from .retry import DEFAULT_REQUEST_TIMEOUT, llm_timeout_seconds
from .prompt_cache import PromptCacheMode
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
    is_max_tokens_field_error,
    max_tokens_field_from_error,
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
    _collect_reasoning_parts,
    _REASONING_ECHO_KEYS,
    _reasoning_echo_allowed,
    _build_assistant_echo,
)


class ClientParamsMixin:
    """Per-model request parameters, temperature fallback, and retry kwargs."""

    def _prompt_cache_log_context(
        self,
        model: str,
        protocol: str,
        modality: str = "text",
        cache_mode: PromptCacheMode = "auto",
    ) -> Optional[Dict[str, Any]]:
        controller = getattr(self, "_prompt_cache_controller", None)
        if controller is None:
            return None
        return controller.describe(model, protocol, modality, cache_mode)

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
                field = self._max_tokens_field(model_name)
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
            field = self._max_tokens_field(model_name)
            params[field] = max_tokens

        if effective_temp is not None and effective_temp != temperature:
            logger.info(f"[{self.agent_name}] Temperature overridden: {temperature} -> {effective_temp} for {model}")

    def _max_tokens_field(self, model_name: str) -> str:
        override = self._max_tokens_field_overrides.get(model_name)
        if override:
            return override
        return "max_completion_tokens" if _is_o_series(model_name) else "max_tokens"

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
        # Hold one lock for the complete first attempt and any bounded
        # adaptation probes.  This is intentionally per model: concurrent
        # calls for unrelated models remain independent, while only one call
        # can discover a provider's spelling for this model.
        self._ensure_cache_locks()
        lock = self._parameter_adaptation_locks.get(model_key)
        if lock is None:
            lock = asyncio.Lock()
            self._parameter_adaptation_locks[model_key] = lock

        async with lock:
            async with self._temp_cache_lock:
                no_temp = model_key in self._temp_unsupported_models
            max_adapted = False
            tentative_field: Optional[str] = None
            previous_field: Optional[str] = None

            async def _request() -> Any:
                return await retry_with_backoff(
                    make_request_no_temp if no_temp else make_request,
                    **retry_kwargs,
                )

            try:
                while True:
                    try:
                        return await _request()
                    except BadRequestError as error:
                        if is_temperature_error(error) and not no_temp:
                            logger.warning(
                                "[%s] Temperature rejected by %s; retrying "
                                "without temperature",
                                self.agent_name,
                                model_key,
                            )
                            async with self._temp_cache_lock:
                                self._temp_unsupported_models.add(model_key)
                            no_temp = True
                            continue

                        if not is_max_tokens_field_error(error) or max_adapted:
                            raise

                        accepted = max_tokens_field_from_error(error)
                        previous_field = self._max_tokens_field_overrides.get(model_key)
                        current = previous_field or (
                            "max_completion_tokens"
                            if _is_o_series(model_key)
                            else "max_tokens"
                        )
                        if accepted is None or accepted == current:
                            raise
                        tentative_field = accepted
                        self._max_tokens_field_overrides[model_key] = accepted
                        max_adapted = True
                        continue
            except BaseException:
                # Never erase a newer choice made by another owner.  The
                # per-model lock normally serializes this path; the identity
                # check also protects tests/fixtures that replace the cache.
                if (
                    tentative_field is not None
                    and self._max_tokens_field_overrides.get(model_key)
                    == tentative_field
                ):
                    if previous_field is None:
                        self._max_tokens_field_overrides.pop(model_key, None)
                    else:
                        self._max_tokens_field_overrides[model_key] = previous_field
                raise

    def _build_retry_kwargs(
        self,
        model: str,
        request_timeout: Optional[float] = None,
        max_retries_override: Optional[int] = None,
        non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
        messages=None,
        tools=None,
        max_tokens=None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "agent_name": self.agent_name,
            "model_name": model,
            "llm_server": self.base_url,
        }
        if messages is not None:
            from functools import partial
            from ..context_overflow import recover_context_overflow, context_settings
            from backend.state import sessions
            state = sessions.get(self.session_id)
            expected_context = context_settings(
                (state.agent_server_configs or {}).get(self.agent_name) or {}
            ) if state is not None else None
            kwargs["on_context_overflow"] = partial(
                recover_context_overflow,
                session_id=self.session_id, agent_name=self.agent_name,
                model=model, endpoint=self.base_url, messages=messages,
                tools=tools, max_tokens=max_tokens, expected_context=expected_context,
            )
        if request_timeout is not None:
            kwargs["request_timeout"] = llm_timeout_seconds(request_timeout)
        if max_retries_override is not None:
            kwargs["max_retries"] = max_retries_override
        elif request_timeout is not None and request_timeout > DEFAULT_REQUEST_TIMEOUT:
            # A long per-attempt deadline disables ordinary retries to avoid
            # paying for the same long generation twice. Connection failures
            # and transient 429s do not produce a completion, so retain their
            # independent backoff budgets.
            kwargs["max_retries"] = 0
            kwargs["connection_retries"] = MAX_RETRIES
            kwargs["rate_limit_retries"] = len(RATE_LIMIT_DELAYS)
        if self.on_retry is not None:
            kwargs["on_retry"] = self.on_retry
        before_retry = getattr(self, "before_retry", None)
        if before_retry is not None:
            kwargs["before_retry"] = before_retry
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
        modality: str = "text",
        cache_mode: PromptCacheMode = "auto",
    ) -> tuple:
        """构建请求闭包对：(标准请求, 无temperature备选请求)."""
        async def send(params: Dict[str, Any]):
            controller = getattr(self, "_prompt_cache_controller", None)
            if controller is None:
                return await self.client.chat.completions.create(**params)
            return await controller.request(
                self.client.chat.completions.create,
                params,
                model=model,
                protocol="chat",
                modality=modality,
                cache_mode=cache_mode,
            )

        async def make_request():
            params = {**base_params}
            self._apply_model_params(params, model, temperature, max_tokens)
            if request_timeout is not None:
                params["timeout"] = llm_timeout_seconds(request_timeout)
            return await send(params)

        async def make_request_no_temp():
            params = {**base_params}
            self._apply_model_params(params, model, temperature, max_tokens, skip_temperature=True)
            if request_timeout is not None:
                params["timeout"] = llm_timeout_seconds(request_timeout)
            return await send(params)

        return make_request, make_request_no_temp
