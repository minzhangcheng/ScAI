# Streaming and vision chat entry points.
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


class ClientStreamVisionMixin:
    """Streaming and vision chat entry points."""

    async def _chat_stream_inline(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        model_key = _extract_model_name(model)

        base_params = {"model": model, "messages": messages, "stream": True}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )

        stream_retry_kwargs = self._build_retry_kwargs(model, request_timeout=120.0)
        stream_retry_kwargs["max_retries"] = MAX_RETRIES

        stream = await self._call_with_temp_fallback(
            _make_request, _make_request_no_temp, model_key, stream_retry_kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _responses_or_fallback_stream(self, model, messages, temperature, max_tokens):
        """Try Responses streaming; endpoint-absent (raised before first chunk)
        -> cache + switch to inline chat stream. Per-(base_url, model) lock makes
        concurrent first calls probe once."""
        key = (self.base_url, model)
        lock = _PROBE_LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            if self._responses_probe_decision(model) == "chat":
                async for ch in self._chat_stream_inline(model, messages, temperature, max_tokens):
                    yield ch
                return
            adapter = self._get_responses_adapter()
            gen = adapter.stream(model=model, messages=messages,
                                 temperature=temperature, max_tokens=max_tokens)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                _PROBE_CACHE[key] = "responses"
                return
            except Exception as e:
                if self._record_responses_probe_failure(model, e, stream=True) == "chat":
                    async for ch in self._chat_stream_inline(model, messages, temperature, max_tokens):
                        yield ch
                    return
                raise
            _PROBE_CACHE[key] = "responses"
            yield first
        # Lock released here; remaining chunks are streamed without lock
        async for ch in gen:
            yield ch

    async def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response from OpenAI-compatible API with auto-routing
        to Responses or Anthropic adapter when appropriate.

        Connection establishment is retried with backoff on transient errors.
        Mid-stream failures propagate to the caller (partial content already yielded).

        Yields:
            Content chunks
        """
        logger.info(f"[{self.agent_name}] Starting stream request to {model}")

        enforced = _inject_english_enforcement(messages)
        protocol = _route_protocol(self.base_url, model)
        if protocol == "anthropic":
            async for ch in self._get_anthropic_adapter().stream(
                    model=model, messages=enforced, temperature=temperature, max_tokens=max_tokens):
                yield ch
            return
        if protocol == "responses":
            async for ch in self._responses_or_fallback_stream(model, enforced, temperature, max_tokens):
                yield ch
            return
        try:
            stream = self._chat_stream_inline(model, enforced, temperature, max_tokens)
            first = await stream.__anext__()
        except StopAsyncIteration:
            return
        except BadRequestError as e:
            if is_responses_only_model(e):
                logger.warning(f"[{self.agent_name}] model {model} is responses-only; "
                               f"switching chat stream to Responses adapter")
                _PROBE_CACHE[(self.base_url, model)] = "responses"
                adapter = self._get_responses_adapter()
                async for ch in adapter.stream(model=model, messages=enforced,
                                               temperature=temperature, max_tokens=max_tokens):
                    yield ch
                return
            raise
        yield first
        async for ch in stream:
            yield ch

    async def _chat_with_vision_inline(
        self,
        model: str,
        text_prompt: str,
        images: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": ENGLISH_ENFORCEMENT},
            {"role": "user", "content": [{"type": "text", "text": text_prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:{_detect_image_mime_type(img_b64)};base64,{img_b64}"}}
                for img_b64 in images
            ]},
        ]

        # Create log record (without the actual image data to save space)
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=[{"role": "user", "content": f"[{len(images)} images] + {text_prompt[:500]}"}],
            interaction_type="vision",
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=self.base_url,
        )

        start_time = time.time()

        model_key = _extract_model_name(model)
        base_params = {"model": model, "messages": messages}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )
        retry_kwargs = self._build_retry_kwargs(
            model, request_timeout=request_timeout, max_retries_override=max_retries,
            non_retryable_predicate=non_retryable_predicate,
        )

        try:
            logger.info(f"[{self.agent_name}] Sending vision request to {model} with {len(images)} images")
            response = await self._call_with_temp_fallback(
                _make_request, _make_request_no_temp, model_key, retry_kwargs
            )

            content = ""
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                content = self._extract_content_from_message(message, response)
            else:
                logger.warning(f"[{self.agent_name}] Vision response has no choices")

            result = LLMResponse(
                content=content,
                model=response.model if response else model,
                usage=self._extract_usage(response),
                raw_response=response
            )

            await self._emit_log_response(
                interaction=interaction,
                response_content=result.content,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except Exception as e:
            logger.exception(f"[{self.agent_name}] Vision error")

            await self._emit_log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise

    async def chat_with_vision(
        self,
        model: str,
        text_prompt: str,
        images: List[str],  # Base64 encoded images
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None
    ) -> LLMResponse:
        """
        Send chat request with images to OpenAI-compatible API with automatic retry on timeout.
        Auto-routes to Responses or Anthropic adapter when appropriate.

        Args:
            model: Model identifier
            text_prompt: Text prompt
            images: List of base64 encoded images
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse object
        """
        log_msgs = [{"role": "user", "content": f"[{len(images)} images] + {text_prompt[:500]}"}]
        probe_timeout = request_timeout or 300.0
        return await self._dispatch_non_streaming(
            model=model,
            interaction_type="vision",
            log_messages=log_msgs,
            make_anthropic_adapter_call=lambda adapter, skip_temperature: adapter.complete_vision(
                model=model, text_prompt=text_prompt, images=images,
                temperature=temperature, max_tokens=max_tokens,
                timeout=request_timeout, skip_temperature=skip_temperature),
            make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete_vision(
                model=model, text_prompt=text_prompt, images=images,
                temperature=temperature, max_tokens=max_tokens,
                timeout=request_timeout, skip_temperature=skip_temperature),
            make_inline_call=lambda max_retries_override=None, non_retryable_predicate=None: self._chat_with_vision_inline(
                model, text_prompt, images, temperature, max_tokens, request_timeout,
                non_retryable_predicate=non_retryable_predicate),
            make_probe_fallback_call=lambda max_retries_override=None, non_retryable_predicate=None: self._chat_with_vision_inline(
                model, text_prompt, images, temperature, max_tokens, probe_timeout, max_retries=0,
                non_retryable_predicate=non_retryable_predicate),
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
