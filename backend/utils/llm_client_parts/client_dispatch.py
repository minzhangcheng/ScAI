# Protocol probing, decision caching, adapter selection, and non-streaming dispatch.
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
    is_responses_vision_shape_error,
    is_vision_request_shape_error,
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
    TEXT_MODALITY,
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


class ClientDispatchMixin:
    """Protocol probing, decision caching, adapter selection, and non-streaming dispatch."""

    def _get_responses_adapter(self) -> "_ResponsesAdapter":
        if getattr(self, "_responses_adapter", None) is None:
            self._responses_adapter = _ResponsesAdapter(
                self.client, agent_name=self.agent_name, base_url=self.base_url)
        return self._responses_adapter

    def _get_anthropic_adapter(self) -> "_AnthropicAdapter":
        if getattr(self, "_anthropic_adapter", None) is None:
            self._anthropic_adapter = _AnthropicAdapter(
                api_key=self.api_key, base_url=_anthropic_sdk_base_url(self.base_url),
                http_client=self.http_client, agent_name=self.agent_name)
        return self._anthropic_adapter

    @staticmethod
    def _adapter_temp_call_pair(
        call_factory: Callable[[bool], Awaitable[Any]]
    ) -> Tuple[Callable[[], Awaitable[Any]], Callable[[], Awaitable[Any]]]:
        return (lambda: call_factory(False), lambda: call_factory(True))

    async def _invoke_adapter(self, *, interaction_type, model, log_messages,
                              make_coro, make_coro_no_temp,
                              tools=None, tool_choice=None, temperature=None,
                              max_tokens=None, request_timeout=None,
                              non_retryable_predicate=None, max_retries_override=None,
                              _empty_retry_count: int = 0,
                              _allow_empty_retry: bool = True):
        """Wrap an adapter call (Responses/Anthropic) with the same cross-cutting
        machinery the inline chat path uses: LLM logging + retry_with_backoff
        (via _call_with_temp_fallback, which also gives on_retry, temperature
        fallback, and UpstreamExhaustedError on exhaustion).

        make_coro / make_coro_no_temp are zero-arg callables returning fresh
        coroutines (with / without temperature)."""
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=log_messages,
            interaction_type=interaction_type,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=self.base_url,
        )
        start_time = time.time()
        model_key = _extract_model_name(model)
        retry_kwargs = self._build_retry_kwargs(
            model, request_timeout=request_timeout, max_retries_override=max_retries_override)
        if non_retryable_predicate is not None:
            retry_kwargs["non_retryable_predicate"] = non_retryable_predicate
        try:
            result = await self._call_with_temp_fallback(
                make_coro, make_coro_no_temp, model_key, retry_kwargs
            )
        except Exception as e:
            await self._emit_log_response(
                interaction=interaction, error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
            raise

        if self._llm_response_empty_no_output(result):
            if not _allow_empty_retry:
                raise EmptyResponseError(
                    f"[{self.agent_name}] Adapter empty answer, zero completion "
                    f"tokens (model={model}) during protocol probe"
                )
            if _empty_retry_count < EMPTY_RESPONSE_MAX_RETRIES:
                delay = RETRY_DELAY_BASE * (_empty_retry_count + 1)
                logger.warning(
                    f"[{self.agent_name}] Adapter returned empty answer, zero "
                    f"completion tokens (model={model}); retry "
                    f"{_empty_retry_count + 1}/{EMPTY_RESPONSE_MAX_RETRIES} "
                    f"after {delay:.0f}s"
                )
                await self._emit_log_response(
                    interaction=interaction,
                    response_content=getattr(result, "content", None),
                    usage=getattr(result, "usage", {}) or {},
                    duration_ms=(time.time() - start_time) * 1000,
                )
                await asyncio.sleep(delay)
                return await self._invoke_adapter(
                    interaction_type=interaction_type,
                    model=model,
                    log_messages=log_messages,
                    make_coro=make_coro,
                    make_coro_no_temp=make_coro_no_temp,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                    non_retryable_predicate=non_retryable_predicate,
                    max_retries_override=max_retries_override,
                    _empty_retry_count=_empty_retry_count + 1,
                    _allow_empty_retry=_allow_empty_retry,
                )
            error = UpstreamExhaustedError(
                "LLM adapter returned empty content (zero completion tokens) "
                f"after {EMPTY_RESPONSE_MAX_RETRIES} retries",
                agent_name=self.agent_name,
                model=model,
                llm_server=self.base_url,
            )
            await self._emit_log_response(
                interaction=interaction,
                error=str(error),
                duration_ms=(time.time() - start_time) * 1000,
            )
            raise error

        tool_calls_log = None
        tcs = getattr(result, "tool_calls", None)
        if tcs:
            tool_calls_log = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments,
                 "parse_error": tc.parse_error}
                for tc in tcs
            ]
        await self._emit_log_response(
            interaction=interaction,
            response_content=getattr(result, "content", None),
            tool_calls=tool_calls_log,
            usage=getattr(result, "usage", {}) or {},
            duration_ms=(time.time() - start_time) * 1000,
        )
        return result

    def _cache_protocol_decision(self, model: str, protocol: str) -> None:
        _PROBE_CACHE[_probe_key(self.base_url, model)] = protocol

    def _cache_protocol_decision_for_modality(
        self, model: str, protocol: str, modality: str
    ) -> None:
        _PROBE_CACHE[_probe_key(self.base_url, model, modality)] = protocol

    def _note_probe_protocol_failure(
        self,
        model: str,
        protocol: str,
        error: Exception,
        modality: str = TEXT_MODALITY,
    ) -> Optional[str]:
        """Record side effects for a DETERMINISTIC failure of `protocol` and
        return the protocol to redirect to next, or None. Never writes a success
        cache (only success writes _PROBE_CACHE); only records the site-level
        _NO_RESPONSES_ENDPOINT fact for a truly absent responses endpoint."""
        if protocol == "responses":
            if modality == VISION_MODALITY and is_responses_vision_shape_error(error):
                logger.warning(
                    f"[{self.agent_name}] Responses vision shape rejected at "
                    f"{self.base_url}; switching this vision request to chat"
                )
                return "chat"
            if is_responses_endpoint_absent(error):
                logger.warning(f"[{self.agent_name}] Responses endpoint absent at "
                               f"{self.base_url}; switching to chat")
                _NO_RESPONSES_ENDPOINT.add(self.base_url)
                return "chat"
            if is_responses_chat_only_model(error):
                logger.warning(f"[{self.agent_name}] model {model} chat-only on Responses; "
                               f"switching to chat")
                return "chat"
        if protocol == "chat":
            if modality == VISION_MODALITY and is_vision_request_shape_error(error):
                logger.warning(
                    f"[{self.agent_name}] Chat vision shape rejected at "
                    f"{self.base_url}; switching this vision request to Responses"
                )
                return "responses"
            if is_responses_only_model(error):
                logger.warning(f"[{self.agent_name}] model {model} is responses-only; "
                               f"switching to responses")
                return "responses"
        return None

    async def _call_decided_protocol(
        self,
        protocol: str,
        call_for,
        model: str,
        modality: str = TEXT_MODALITY,
    ):
        """Call a single, already-decided protocol with FULL retry/backoff. On a
        deterministic 'wrong protocol' error, redirect once to the indicated
        protocol (also full-retry). Fixes the confirmed-responses no-retry bug."""
        factory = call_for(protocol, probe=False)
        if factory is None:
            raise ValueError(f"protocol {protocol} unsupported for this interaction")
        try:
            result = await factory()
        except Exception as e:
            redirect = self._note_probe_protocol_failure(model, protocol, e, modality)
            if redirect is None:
                raise
            key = _probe_key(self.base_url, model, modality)
            if _PROBE_CACHE.get(key) == protocol:
                _PROBE_CACHE.pop(key, None)
            rfactory = call_for(redirect, probe=False)
            if rfactory is None:
                raise
            result = await rfactory()
            self._cache_protocol_decision_for_modality(model, redirect, modality)
            return result
        self._cache_protocol_decision_for_modality(model, protocol, modality)
        return result

    async def _probe_in_order(
        self,
        order: List[str],
        call_for,
        model: str,
        modality: str = TEXT_MODALITY,
    ):
        """Try protocols in `order`. Per protocol: up to len(PROBE_RETRY_DELAYS)
        single-shot attempts on TRANSIENT errors (sleeping PROBE_RETRY_DELAYS
        between/after each), then move on. A DETERMINISTIC 'wrong protocol' error
        switches immediately (and may inject a redirect protocol at the front).
        Only a SUCCESS writes the protocol cache."""
        key = _probe_key(self.base_url, model, modality)
        lock = _PROBE_LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            recheck = (
                _route_protocol_order(self.base_url, model)
                if modality == TEXT_MODALITY
                else _route_protocol_order(self.base_url, model, modality)
            )
            if len(recheck) == 1:
                return await self._call_decided_protocol(
                    recheck[0], call_for, model, modality
                )

            pending = list(order)
            tried: set = set()
            last_error: Optional[Exception] = None
            while pending:
                # No inter-protocol sleep: PROBE_RETRY_DELAYS already back off on
                # transient errors, and protocol switches should be immediate.
                protocol = pending.pop(0)
                if protocol in tried:
                    continue
                tried.add(protocol)
                factory = call_for(protocol, probe=True)
                if factory is None:
                    continue
                for attempt in range(len(PROBE_RETRY_DELAYS)):
                    try:
                        result = await factory()
                        self._cache_protocol_decision_for_modality(
                            model, protocol, modality
                        )
                        return result
                    except Exception as e:
                        last_error = e
                        if is_transient_probe_error(e):
                            await asyncio.sleep(PROBE_RETRY_DELAYS[attempt])
                            continue
                        redirect = self._note_probe_protocol_failure(
                            model, protocol, e, modality
                        )
                        if redirect and redirect not in tried and redirect not in pending:
                            pending.insert(0, redirect)
                        break
            raise UpstreamExhaustedError(
                f"[{self.agent_name}] Protocol probe exhausted across {sorted(tried)} "
                f"for {model}. Last error: {type(last_error).__name__}: {last_error}",
                agent_name=self.agent_name, model=model, llm_server=self.base_url,
            ) from last_error

    async def _dispatch_non_streaming(
        self,
        *,
        model: str,
        interaction_type: str,
        log_messages: List[Dict[str, Any]],
        make_anthropic_adapter_call: Optional[Callable[[Any, bool], Awaitable[Any]]],
        make_responses_adapter_call: Callable[[Any, bool], Awaitable[Any]],
        make_inline_call: Callable[[], Awaitable[Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        make_probe_fallback_call: Optional[Callable[[], Awaitable[Any]]] = None,
    ):
        def invoke(adapter, adapter_call, max_retries_override=None,
                   non_retryable_predicate=None, allow_empty_retry=True):
            if non_retryable_predicate is None:
                non_retryable_predicate = lambda e: not is_transient_probe_error(e)
            make_coro, make_coro_no_temp = self._adapter_temp_call_pair(
                lambda skip_temperature: adapter_call(adapter, skip_temperature)
            )
            return self._invoke_adapter(
                interaction_type=interaction_type, model=model, log_messages=log_messages,
                make_coro=make_coro, make_coro_no_temp=make_coro_no_temp,
                tools=tools, tool_choice=tool_choice, temperature=temperature,
                max_tokens=max_tokens, request_timeout=request_timeout,
                non_retryable_predicate=non_retryable_predicate,
                max_retries_override=max_retries_override,
                _allow_empty_retry=allow_empty_retry)

        def invoke_inline(inline_call, max_retries_override=None, non_retryable_predicate=None):
            return inline_call(
                max_retries_override=max_retries_override,
                non_retryable_predicate=non_retryable_predicate,
            )

        def call_for(protocol, *, probe):
            mr = 0 if probe else None
            # In probe mode, the probe loop owns ALL retry logic; the backoff
            # layer must not intervene.  In decided mode, only deterministic
            # wrong-protocol errors are non-retryable (they trigger redirect).
            np = (lambda e: True) if probe else (lambda e: not is_transient_probe_error(e))
            if protocol == "responses":
                return lambda: invoke(self._get_responses_adapter(),
                                      make_responses_adapter_call, max_retries_override=mr,
                                      non_retryable_predicate=np, allow_empty_retry=not probe)
            if protocol == "anthropic":
                if make_anthropic_adapter_call is None:
                    return None
                return lambda: invoke(self._get_anthropic_adapter(),
                                      make_anthropic_adapter_call, max_retries_override=mr,
                                      non_retryable_predicate=np, allow_empty_retry=not probe)
            if protocol == "chat":
                target = (make_probe_fallback_call or make_inline_call) if probe else make_inline_call
                return lambda: invoke_inline(
                    target,
                    max_retries_override=mr,
                    non_retryable_predicate=np,
                )
            return None

        modality = VISION_MODALITY if interaction_type == "vision" else TEXT_MODALITY
        order = (
            _route_protocol_order(self.base_url, model)
            if modality == TEXT_MODALITY
            else _route_protocol_order(self.base_url, model, modality)
        )
        if len(order) == 1:
            return await self._call_decided_protocol(order[0], call_for, model, modality)
        return await self._probe_in_order(order, call_for, model, modality)

    def _responses_probe_decision(self, model: str) -> Optional[str]:
        key = (self.base_url, model)
        if self.base_url in _NO_RESPONSES_ENDPOINT or _PROBE_CACHE.get(key) == "chat":
            return "chat"
        if _PROBE_CACHE.get(key) == "responses":
            return "responses"
        return None

    def _record_responses_probe_failure(self, model: str, error: Exception, stream: bool = False) -> Optional[str]:
        key = (self.base_url, model)
        stream_label = " stream" if stream else ""
        if is_responses_endpoint_absent(error):
            logger.warning(f"[{self.agent_name}] Responses{stream_label} endpoint absent at {self.base_url}; falling back to chat{stream_label}")
            _NO_RESPONSES_ENDPOINT.add(self.base_url)
            _PROBE_CACHE[key] = "chat"
            return "chat"
        if is_responses_chat_only_model(error):
            logger.warning(f"[{self.agent_name}] model {model} is chat/completions-only on Responses; falling back to chat{stream_label}")
            _PROBE_CACHE[key] = "chat"
            return "chat"
        return None
