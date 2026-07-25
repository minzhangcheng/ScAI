# Plain chat and tool-calling chat entry points.
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


class ClientChatMixin:
    """Plain chat and tool-calling chat entry points."""

    async def _chat_inline(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        _truncation_retried: bool = False,
        _empty_retry_count: int = 0,
        _allow_empty_retry: bool = True,
        non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
        max_retries_override: Optional[int] = None,
    ) -> LLMResponse:
        # Create log record
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=messages,
            interaction_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=self.base_url,
        )

        start_time = time.time()
        error_msg = None

        model_key = _extract_model_name(model)
        base_params = {"model": model, "messages": messages, "stream": stream}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens, request_timeout=timeout
        )
        effective_max_retries = max_retries_override if max_retries_override is not None else max_retries
        retry_kwargs = self._build_retry_kwargs(
            model, request_timeout=timeout, max_retries_override=effective_max_retries,
            non_retryable_predicate=non_retryable_predicate,
        )

        try:
            logger.info(f"[{self.agent_name}] Sending chat request to {model}")
            response = await self._call_with_temp_fallback(
                _make_request, _make_request_no_temp, model_key, retry_kwargs
            )

            # Extract content — handles reasoning models, list content, etc.
            content = ""
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                content = self._extract_content_from_message(message, response)

            # Reasoning-model truncation guard: when a reasoning model spends the
            # entire max_tokens budget on reasoning the response comes back with an
            # empty answer field AND length-truncated. The extractor may salvage the
            # (also truncated) reasoning text into `content`, so we cannot rely on
            # the EMPTY_CONTENT_PLACEHOLDER alone — gate on the *answer* content field
            # being empty instead. Callers that don't special-case this (supervisor,
            # PI, technique...) would otherwise silently treat it as "no answer".
            # Retry once with a larger budget so the model can finish.
            if (
                not _truncation_retried
                and max_tokens is not None
                and self._response_length_truncated(response)
                and self._answer_content_empty(response)
            ):
                bumped = min(max(max_tokens * 4, _TRUNCATION_RETRY_FLOOR), _TRUNCATION_RETRY_CEILING)
                logger.warning(
                    f"[{self.agent_name}] Empty content from a length-truncated response "
                    f"(max_tokens={max_tokens}); retrying once with max_tokens={bumped}"
                )
                await self._emit_log_response(
                    interaction=interaction,
                    response_content=content,
                    usage=self._extract_usage(response),
                    duration_ms=(time.time() - start_time) * 1000,
                )
                return await self._chat_inline(
                    model, messages, temperature, bumped, stream, timeout, max_retries,
                    _truncation_retried=True,
                    non_retryable_predicate=non_retryable_predicate,
                    max_retries_override=max_retries_override,
                )

            # Empty-response guard: a 200 reply with an empty answer, no tool
            # call, and zero generated tokens is a transient upstream blip
            # (proxy hiccup, dropped stream). Unlike the truncation guard above
            # we keep max_tokens unchanged and just retry with backoff. Gated on
            # completion_tokens == 0 so a model that deliberately returned an
            # empty (but token-consuming) answer is NOT looped on. On exhaustion
            # RAISE — the extractor leaves a zero-token empty as "" (it only
            # emits EMPTY_CONTENT_PLACEHOLDER when completion_tokens > 0), so
            # returning would hand callers a bare "". Raising routes affected
            # workflow calls into existing HITL failure handling. Never
            # returns "" on this signature.
            if self._answer_empty_no_output(response):
                if not _allow_empty_retry:
                    raise EmptyResponseError(
                        f"[{self.agent_name}] Empty answer, zero completion "
                        f"tokens (model={model}) during protocol probe"
                    )
                if _empty_retry_count < EMPTY_RESPONSE_MAX_RETRIES:
                    delay = RETRY_DELAY_BASE * (_empty_retry_count + 1)
                    logger.warning(
                        f"[{self.agent_name}] Empty answer, zero completion tokens "
                        f"(model={model}); retry "
                        f"{_empty_retry_count + 1}/{EMPTY_RESPONSE_MAX_RETRIES} "
                        f"after {delay:.0f}s"
                    )
                    await self._emit_log_response(
                        interaction=interaction,
                        response_content=content,
                        usage=self._extract_usage(response),
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                    await asyncio.sleep(delay)
                    return await self._chat_inline(
                        model, messages, temperature, max_tokens, stream, timeout,
                        max_retries,
                        _truncation_retried=_truncation_retried,
                        _empty_retry_count=_empty_retry_count + 1,
                        _allow_empty_retry=_allow_empty_retry,
                        non_retryable_predicate=non_retryable_predicate,
                        max_retries_override=max_retries_override,
                    )
                raise UpstreamExhaustedError(
                    "LLM returned empty content (zero completion tokens) after "
                    f"{EMPTY_RESPONSE_MAX_RETRIES} retries",
                    agent_name=self.agent_name,
                    model=model,
                    llm_server=self.base_url,
                )

            result = LLMResponse(
                content=content,
                model=response.model if response else model,
                usage=self._extract_usage(response),
                raw_response=response
            )

            # Log response
            await self._emit_log_response(
                interaction=interaction,
                response_content=result.content,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except Exception as e:
            error_msg = str(e)
            logger.exception(f"[{self.agent_name}] Chat error")

            # Log error
            await self._emit_log_response(
                interaction=interaction,
                error=error_msg,
                duration_ms=(time.time() - start_time) * 1000
            )
            raise

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Send chat request to OpenAI-compatible API with automatic retry on timeout
        and automatic protocol dispatch (chat / responses / anthropic).

        Args:
            model: Model identifier (e.g., "google/gemini-2.5-pro-preview-05-06")
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Unsupported; use chat_stream() for streaming responses
            timeout: Per-request timeout in seconds (overrides client default)
            tools: Optional Responses API tools for non-streaming Responses calls

        Returns:
            LLMResponse object
        """
        if stream:
            raise ValueError("chat(stream=True) is not supported; use chat_stream() for streaming responses")

        enforced = _inject_english_enforcement(messages)
        probe_timeout = timeout or 300.0
        return await self._dispatch_non_streaming(
            model=model,
            interaction_type="chat",
            log_messages=enforced,
            make_anthropic_adapter_call=lambda adapter, skip_temperature: adapter.complete(
                model=model, messages=enforced, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout, skip_temperature=skip_temperature),
            make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete(
                model=model, messages=enforced, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout, skip_temperature=skip_temperature,
                tools=tools),
            make_inline_call=lambda max_retries_override=None, non_retryable_predicate=None: self._chat_inline(
                model, enforced, temperature, max_tokens, False, timeout, max_retries,
                non_retryable_predicate=non_retryable_predicate,
                max_retries_override=max_retries_override),
            make_probe_fallback_call=lambda max_retries_override=None, non_retryable_predicate=None: self._chat_inline(
                model, enforced, temperature, max_tokens, False, probe_timeout, max_retries=0,
                _allow_empty_retry=False,
                non_retryable_predicate=non_retryable_predicate,
                max_retries_override=max_retries_override),
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout,
        )

    async def _chat_with_tools_inline(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
        max_retries_override: Optional[int] = None,
    ) -> LLMToolResponse:
        # Create log record
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=messages,
            interaction_type="chat_with_tools",
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=self.base_url,
        )

        start_time = time.time()

        model_key = _extract_model_name(model)

        # Pre-emptive demotion: if this model has previously rejected non-"auto"
        # tool_choice, skip the round-trip and demote now.
        effective_tool_choice = tool_choice
        if (
            effective_tool_choice not in (None, "auto", "none")
            and model_key in self._toolchoice_unsupported_models
        ):
            logger.warning(
                f"[{self.agent_name}] tool_choice={effective_tool_choice!r} demoted to 'auto' "
                f"for {model_key} (cached as tool-choice-unsupported)"
            )
            effective_tool_choice = "auto"

        base_params = {"model": model, "messages": messages, "tools": tools, "tool_choice": effective_tool_choice}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )
        effective_max_retries = max_retries_override if max_retries_override is not None else max_retries
        retry_kwargs = self._build_retry_kwargs(
            model, max_retries_override=effective_max_retries,
            non_retryable_predicate=non_retryable_predicate,
        )

        try:
            logger.info(f"[{self.agent_name}] Sending chat_with_tools request to {model}")
            try:
                response = await self._call_with_temp_fallback(
                    _make_request, _make_request_no_temp, model_key, retry_kwargs
                )
            except BadRequestError as tc_err:
                if (
                    is_tool_choice_error(tc_err)
                    and effective_tool_choice not in (None, "auto", "none")
                ):
                    logger.warning(
                        f"[{self.agent_name}] tool_choice rejected by {model_key}, "
                        f"caching as unsupported and retrying with 'auto'"
                    )
                    self._ensure_cache_locks()
                    async with self._toolchoice_cache_lock:
                        if model_key not in self._toolchoice_unsupported_models:
                            self._toolchoice_unsupported_models.add(model_key)
                    base_params["tool_choice"] = "auto"
                    _make_request, _make_request_no_temp = self._build_request_pair(
                        base_params, model, temperature, max_tokens
                    )
                    response = await self._call_with_temp_fallback(
                        _make_request, _make_request_no_temp, model_key, retry_kwargs
                    )
                else:
                    raise

            message = None
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message

            tool_calls = []
            tool_calls_log = []
            if message and message.tool_calls:
                for tc in message.tool_calls:
                    parse_error: Optional[str] = None
                    raw_args: Optional[str] = None
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[{self.agent_name}] Tool call arguments JSON parse failed: {e}; "
                            f"function={tc.function.name}, arguments={tc.function.arguments!r}"
                        )
                        arguments = {}
                        parse_error = str(e)
                        raw_args = tc.function.arguments

                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                        parse_error=parse_error,
                        raw_arguments=raw_args,
                    ))
                    tool_calls_log.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                        "parse_error": parse_error,
                    })

            content = self._extract_content_from_message(message, response) if message else None

            raw_assistant_message = None
            if message:
                try:
                    raw_assistant_message = _build_assistant_echo(message.model_dump(), model)
                except Exception as e:
                    logger.debug(f"[{self.agent_name}] Failed to preserve raw assistant message: {e}")

            if not tool_calls and content:
                recovered = _recover_stringified_tool_call(content)
                if recovered is not None:
                    rec_name, rec_args = recovered
                    rec_id = f"recovered_{uuid4().hex[:8]}"
                    logger.warning(
                        f"[{self.agent_name}] Recovered stringified tool_call "
                        f"name={rec_name!r} from content at {self.base_url} "
                        f"(model={model}); upstream gateway lost tool_calls field"
                    )
                    tool_calls.append(ToolCall(
                        id=rec_id, name=rec_name, arguments=rec_args,
                    ))
                    tool_calls_log.append({
                        "id": rec_id, "name": rec_name,
                        "arguments": rec_args, "parse_error": None,
                        "recovered": True,
                    })
                    content = None
                    if raw_assistant_message is not None:
                        raw_assistant_message["content"] = None
                        raw_assistant_message["tool_calls"] = [{
                            "id": rec_id,
                            "function": {"name": rec_name, "arguments": json.dumps(rec_args)},
                            "type": "function",
                        }]

            result = LLMToolResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model if response else model,
                usage=self._extract_usage(response),
                raw_response=response,
                raw_assistant_message=raw_assistant_message
            )

            await self._emit_log_response(
                interaction=interaction,
                response_content=content,
                tool_calls=tool_calls_log if tool_calls_log else None,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except BadRequestError as e:
            logger.warning(
                f"[{self.agent_name}] Chat with tools BadRequestError"
                f" (may be handled upstream): {e}"
            )
            await self._emit_log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise
        except Exception as e:
            logger.exception(f"[{self.agent_name}] Chat with tools error")
            await self._emit_log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise

    async def chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMToolResponse:
        """
        Send chat request with function/tool calling support and automatic retry on timeout.
        Auto-routes to Responses or Anthropic adapter when appropriate.

        Args:
            model: Model identifier
            messages: List of messages
            tools: List of tool definitions in OpenAI format
            tool_choice: "auto", "none", "required", or specific tool
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMToolResponse with potential tool calls
        """
        enforced = _inject_english_enforcement(messages)
        return await self._dispatch_non_streaming(
            model=model,
            interaction_type="chat_with_tools",
            log_messages=enforced,
            make_anthropic_adapter_call=lambda adapter, skip_temperature: adapter.complete_with_tools(
                model=model, messages=enforced, tools=tools, tool_choice=tool_choice,
                temperature=temperature, max_tokens=max_tokens, skip_temperature=skip_temperature),
            make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete_with_tools(
                model=model, messages=enforced, tools=tools, tool_choice=tool_choice,
                temperature=temperature, max_tokens=max_tokens, skip_temperature=skip_temperature),
            make_inline_call=lambda max_retries_override=None, non_retryable_predicate=None: self._chat_with_tools_inline(
                model, enforced, tools, tool_choice, temperature, max_tokens,
                non_retryable_predicate=non_retryable_predicate,
                max_retries_override=max_retries_override),
            make_probe_fallback_call=lambda max_retries_override=None, non_retryable_predicate=None: self._chat_with_tools_inline(
                model, enforced, tools, tool_choice, temperature, max_tokens, max_retries=0,
                non_retryable_predicate=non_retryable_predicate,
                max_retries_override=max_retries_override),
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
