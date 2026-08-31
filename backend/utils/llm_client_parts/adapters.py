# Anthropic Messages and OpenAI Responses protocol adapters.
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

from .error_signals import _inject_web_search_if_needed, _is_unknown_web_search_tool_error, _is_unsupported_background_error, _swap_web_search_tool_name
from .errors import StreamInterruptedError
from .google import _DR_BACKGROUND_MAX_POLL_ERRORS, _DR_BACKGROUND_MAX_WAIT, _DR_BACKGROUND_POLL_INTERVAL, _DR_BACKGROUND_TERMINAL_FAILED
from .media import _detect_image_mime_type, _inject_english_enforcement
from .model_traits import _WEB_SEARCH_TOOL_NAMES, _get_effective_temperature, _is_deep_research_model
from .protocol import _NO_BACKGROUND_ENDPOINT, _anthropic_messages_translate, _anthropic_restore_thinking, _anthropic_translate_tool_choice, _anthropic_translate_tools, _parse_anthropic_dump, _parse_responses_dump, _responses_messages_to_input, _responses_translate_tool_choice, _responses_translate_tools
from .responses import LLMResponse, LLMToolResponse, _tool_calls_to_raw_assistant
from .retry import DEFAULT_REQUEST_TIMEOUT


_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


async def _close_stream(stream) -> None:
    """Close an SDK stream without masking the request's primary outcome."""
    closer = getattr(stream, "close", None)
    if not callable(closer):
        closer = getattr(stream, "aclose", None)
    if not callable(closer):
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception as close_error:  # cleanup must not replace the primary error
        logger.warning("Failed to close LLM stream: %s", close_error)


@dataclass
class _StartedStream:
    """A stream whose first event was received inside the retry boundary."""

    stream: Any
    iterator: Any
    first_event: Any
    consumer: Callable[["_StartedStream"], Awaitable[Any]]
    idle_timeout: float
    label: str
    _closed: bool = False

    async def events(self):
        yield self.first_event
        while True:
            try:
                event = await asyncio.wait_for(
                    self.iterator.__anext__(), timeout=self.idle_timeout
                )
            except StopAsyncIteration:
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise StreamInterruptedError(
                    f"{self.label} interrupted after the first event: "
                    f"{type(error).__name__}: {error}"
                ) from error
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _close_stream(self.stream)

    async def finish(self):
        try:
            return await self.consumer(self)
        finally:
            await self.close()


async def _start_stream(stream, consumer, *, timeout, label) -> _StartedStream:
    """Receive the first event before returning control to the retry layer."""
    idle_timeout = timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
    iterator = stream.__aiter__()
    try:
        first_event = await asyncio.wait_for(
            iterator.__anext__(), timeout=idle_timeout
        )
    except StopAsyncIteration as error:
        await _close_stream(stream)
        raise RuntimeError(f"{label} ended before its first event") from error
    except BaseException:
        await _close_stream(stream)
        raise
    return _StartedStream(
        stream=stream,
        iterator=iterator,
        first_event=first_event,
        consumer=consumer,
        idle_timeout=idle_timeout,
        label=label,
    )

class _AnthropicAdapter:
    """Adapter for the Anthropic Messages API. anthropic SDK is imported lazily
    so the module loads even when the package is absent (only this class needs it)."""

    def __init__(self, api_key: str, base_url: str, http_client=None, agent_name: str = "unknown"):
        from anthropic import AsyncAnthropic  # lazy import
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self.client = AsyncAnthropic(**kwargs)
        self.agent_name = agent_name

    def _build_params(self, model, messages, temperature, max_tokens,
                      tools=None, tool_choice=None, skip_temperature=False) -> Dict[str, Any]:
        prepared = _anthropic_restore_thinking(messages)
        system, anth_messages = _anthropic_messages_translate(prepared)
        params: Dict[str, Any] = {
            "model": model,
            "messages": anth_messages,
            "max_tokens": max_tokens if max_tokens is not None else _ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if system is not None:
            params["system"] = system
        if not skip_temperature:
            eff_temp = _get_effective_temperature(model, temperature)
            if eff_temp is not None:
                params["temperature"] = eff_temp
        choice = _anthropic_translate_tool_choice(tool_choice) if tool_choice is not None else None
        # tool_choice 'none' -> choice is None -> omit tools entirely
        if tools and not (tool_choice == "none"):
            params["tools"] = _anthropic_translate_tools(tools)
            if choice is not None:
                params["tool_choice"] = choice
        return params

    def _to_tool_response(self, resp, model) -> LLMToolResponse:
        dump = resp.model_dump()
        text, tool_calls, thinking, usage = _parse_anthropic_dump(dump)
        raw_assistant = _tool_calls_to_raw_assistant(text, tool_calls)
        if thinking and tool_calls:
            raw_assistant["reasoning_details"] = thinking
        return LLMToolResponse(content=text or None, tool_calls=tool_calls,
                               model=getattr(resp, "model", model), usage=usage,
                               raw_response=resp, raw_assistant_message=raw_assistant,
                               **self._terminal_metadata(dump.get("stop_reason")))

    @staticmethod
    def _terminal_metadata(stop_reason: Any) -> Dict[str, Any]:
        if stop_reason == "max_tokens":
            return {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_tokens"},
            }
        return {"status": "completed"}

    async def _complete_stream_response(
        self, started: _StartedStream, model: str
    ) -> LLMResponse:
        text_parts = []
        raw_message = None
        usage_data = {}
        has_cache_telemetry = False
        model_name = model
        message_stopped = False
        stop_reason = None
        async for event in started.events():
            event_type = getattr(event, "type", "")
            if event_type == "message_start":
                raw_message = getattr(event, "message", None)
                model_name = getattr(raw_message, "model", model_name)
                usage_obj = getattr(raw_message, "usage", None)
                if usage_obj is not None:
                    usage_data["input_tokens"] = (
                        getattr(usage_obj, "input_tokens", 0) or 0
                    )
                    cache_creation = getattr(
                        usage_obj, "cache_creation_input_tokens", None
                    )
                    cache_read = getattr(
                        usage_obj, "cache_read_input_tokens", None
                    )
                    has_cache_telemetry = (
                        cache_creation is not None or cache_read is not None
                    )
                    if cache_creation is not None:
                        usage_data["cache_creation_input_tokens"] = cache_creation or 0
                    if cache_read is not None:
                        usage_data["cache_read_input_tokens"] = cache_read or 0
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "type", "") == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        text_parts.append(text)
            elif event_type == "message_delta":
                delta = getattr(event, "delta", None)
                event_stop_reason = getattr(delta, "stop_reason", None)
                if isinstance(event_stop_reason, str):
                    stop_reason = event_stop_reason
                usage_obj = getattr(event, "usage", None)
                if usage_obj is not None:
                    usage_data["output_tokens"] = (
                        getattr(usage_obj, "output_tokens", 0) or 0
                    )
            elif event_type == "message_stop":
                message_stopped = True
                break
        if not message_stopped:
            raise StreamInterruptedError(
                "Anthropic text stream ended after the first event without message_stop"
            )
        prompt_tokens = sum(
            usage_data.get(key, 0)
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        completion_tokens = usage_data.get("output_tokens", 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if has_cache_telemetry:
            usage["cached_tokens"] = usage_data.get("cache_read_input_tokens", 0)
        return LLMResponse(
            content="".join(text_parts),
            model=model_name,
            usage=usage,
            raw_response=raw_message,
            **self._terminal_metadata(stop_reason),
        )

    async def complete(self, model, messages, temperature, max_tokens, timeout=None,
                       skip_temperature=False, stream=False,
                       defer_stream=False) -> LLMResponse:
        params = self._build_params(model, messages, temperature, max_tokens, skip_temperature=skip_temperature)
        if stream:
            params["stream"] = True
        if timeout is not None:
            params["timeout"] = timeout
        resp = await self.client.messages.create(**params)
        if stream:
            started = await _start_stream(
                resp,
                lambda value: self._complete_stream_response(value, model),
                timeout=timeout,
                label="Anthropic text stream",
            )
            return started if defer_stream else await started.finish()
        dump = resp.model_dump()
        text, _tc, _think, usage = _parse_anthropic_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp,
                           **self._terminal_metadata(dump.get("stop_reason")))

    async def complete_with_tools(self, model, messages, tools, tool_choice,
                                  temperature, max_tokens, skip_temperature=False) -> LLMToolResponse:
        params = self._build_params(model, messages, temperature, max_tokens,
                                    tools=tools, tool_choice=tool_choice, skip_temperature=skip_temperature)
        resp = await self.client.messages.create(**params)
        return self._to_tool_response(resp, model)

    async def complete_vision(self, model, text_prompt, images, temperature,
                              max_tokens, timeout=None, skip_temperature=False,
                              system_prompt=None) -> LLMResponse:
        content = [{"type": "text", "text": text_prompt}]
        for img_b64 in images:
            mime = _detect_image_mime_type(img_b64)
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": mime, "data": img_b64}})
        messages = _inject_english_enforcement(
            ([{"role": "system", "content": system_prompt}]
             if system_prompt else [])
            + [{"role": "user", "content": content}]
        )
        params = self._build_params(model, messages, temperature, max_tokens, skip_temperature=skip_temperature)
        if timeout is not None:
            params["timeout"] = timeout
        resp = await self.client.messages.create(**params)
        dump = resp.model_dump()
        text, _tc, _think, usage = _parse_anthropic_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp,
                           **self._terminal_metadata(dump.get("stop_reason")))

    async def stream(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False):
        params = self._build_params(model, messages, temperature, max_tokens, skip_temperature=skip_temperature)
        params["stream"] = True
        stream = await self.client.messages.create(**params)
        try:
            async for event in stream:
                if getattr(event, "type", "") == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is not None and getattr(delta, "type", "") == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            yield text
        finally:
            await _close_stream(stream)

class _ResponsesAdapter:
    """Adapter for the OpenAI Responses API (sync form). Holds an AsyncOpenAI
    client and translates the canonical OpenAI-chat request shape to/from
    Responses. Parsing operates on model_dump() dicts (see _parse_responses_dump)."""

    def __init__(self, client: "AsyncOpenAI", agent_name: str = "unknown",
                 base_url: str = ""):
        self.client = client
        self.agent_name = agent_name
        self.base_url = base_url

    def _build_params(self, model, messages, temperature, max_tokens, tools=None,
                      tool_choice=None, stream=False, timeout=None, skip_temperature=False,
                      background_ok=False) -> Dict[str, Any]:
        instructions, input_items = _responses_messages_to_input(messages)
        params: Dict[str, Any] = {"model": model, "input": input_items}
        if instructions is not None:
            params["instructions"] = instructions
        if not skip_temperature:
            eff_temp = _get_effective_temperature(model, temperature)
            if eff_temp is not None:
                params["temperature"] = eff_temp
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        if tools:
            params["tools"] = _responses_translate_tools(_inject_web_search_if_needed(model, tools))
        elif _is_deep_research_model(model):
            params["tools"] = _responses_translate_tools(_inject_web_search_if_needed(model, []))
        if tool_choice is not None:
            params["tool_choice"] = _responses_translate_tool_choice(tool_choice)
        # Deep-research models run for minutes: submit as a background task so
        # create() returns immediately and we poll retrieve() (see
        # _ResponsesAdapter._await_completion). Only the caller that runs under a
        # long outer timeout opts in (background_ok); streaming keeps its own
        # long-lived connection so it never needs background; and an endpoint
        # known to reject `background` is skipped so we don't pay a failed
        # round-trip every call.
        if (background_ok and not stream and _is_deep_research_model(model)
                and self.base_url not in _NO_BACKGROUND_ENDPOINT):
            params["background"] = True
        if stream:
            params["stream"] = True
        if timeout is not None:
            params["timeout"] = timeout
        return params

    async def _create_responses_with_fallback(self, params: Dict[str, Any]):
        """Call responses.create with web_search tool name fallback.
        On an unknown-tool error for an auto-injected web_search tool,
        retries once with the alternate name."""
        try:
            return await self.client.responses.create(**params)
        except Exception as e:
            # Provider rejected background mode: cache the endpoint, drop the
            # flag, and retry synchronously (best we can do without background).
            if _is_unsupported_background_error(e) and params.get("background"):
                logger.warning(
                    f"[{self.agent_name}] Responses background mode rejected at "
                    f"{self.base_url}; retrying synchronously: {str(e)[:120]}"
                )
                if self.base_url:
                    _NO_BACKGROUND_ENDPOINT.add(self.base_url)
                params.pop("background", None)
                return await self.client.responses.create(**params)
            if not _is_unknown_web_search_tool_error(e):
                raise
            tools = params.get("tools")
            if not tools:
                raise
            current_names = {t["type"] for t in tools
                             if isinstance(t, dict) and t.get("type") in set(_WEB_SEARCH_TOOL_NAMES)}
            if not current_names:
                raise
            tried_name = next(iter(current_names))
            for alt in _WEB_SEARCH_TOOL_NAMES:
                if alt != tried_name:
                    logger.warning(
                        f"[{self.agent_name}] Web search tool '{tried_name}' unknown,"
                        f" retrying with '{alt}'"
                    )
                    params["tools"] = _swap_web_search_tool_name(tools, tried_name, alt)
                    return await self.client.responses.create(**params)
            raise

    async def _await_completion(self, resp):
        """Poll a background Responses task to completion.

        Deep-research models are submitted with background=True so the initial
        create() returns immediately with status queued/in_progress instead of
        holding the connection open for minutes (which trips an upstream gateway's
        idle timeout -> HTTP 524 while the task keeps running server-side). We then
        poll responses.retrieve() with short requests until a terminal status.

        Non-background responses (status completed/None, or anything not in the
        queued/in_progress set) are returned as-is, so this is a no-op for
        ordinary calls.
        """
        status = getattr(resp, "status", None)
        if status not in ("queued", "in_progress"):
            return resp
        resp_id = getattr(resp, "id", None)
        if not resp_id:
            return resp
        logger.info(
            f"[{self.agent_name}] Deep research submitted (background), "
            f"polling {resp_id}"
        )
        deadline = time.monotonic() + _DR_BACKGROUND_MAX_WAIT
        consecutive_errors = 0
        while time.monotonic() < deadline:
            await asyncio.sleep(_DR_BACKGROUND_POLL_INTERVAL)
            try:
                resp = await self.client.responses.retrieve(resp_id)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.warning(
                    f"[{self.agent_name}] Background poll transient error "
                    f"({consecutive_errors}/{_DR_BACKGROUND_MAX_POLL_ERRORS}): "
                    f"{type(e).__name__}: {str(e)[:120]}"
                )
                if consecutive_errors >= _DR_BACKGROUND_MAX_POLL_ERRORS:
                    raise
                continue
            status = getattr(resp, "status", None)
            if status == "completed":
                logger.info(
                    f"[{self.agent_name}] Deep research {resp_id} completed"
                )
                return resp
            if status in _DR_BACKGROUND_TERMINAL_FAILED:
                err = getattr(resp, "error", None)
                raise RuntimeError(
                    f"Deep research task {resp_id} {status}: {err}"
                )
        raise TimeoutError(
            f"Deep research task {resp_id} did not complete within "
            f"{_DR_BACKGROUND_MAX_WAIT}s"
        )

    @staticmethod
    def _response_metadata(resp, dump, fallback_status=None) -> Dict[str, Any]:
        status = dump.get("status") if isinstance(dump, dict) else None
        if status is None:
            raw_status = getattr(resp, "status", None)
            if isinstance(raw_status, str):
                status = raw_status
        incomplete_details = (
            dump.get("incomplete_details") if isinstance(dump, dict) else None
        )
        upstream_error = dump.get("error") if isinstance(dump, dict) else None
        return {
            "status": status or fallback_status,
            "incomplete_details": incomplete_details,
            "error": upstream_error,
        }

    async def _complete_stream_response(
        self, started: _StartedStream, model: str, *, with_tools: bool = False
    ):
        text_parts = []
        completed_response = None
        terminal_event_type = None
        terminal_event_error = None
        async for event in started.events():
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    text_parts.append(delta)
            elif event_type in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                completed_response = getattr(event, "response", None)
                terminal_event_type = event_type
                terminal_event_error = getattr(event, "error", None)
                break

        if completed_response is None and terminal_event_type is None:
            raise StreamInterruptedError(
                "Responses stream ended after the first event without a terminal response"
            )
        if completed_response is None:
            status = terminal_event_type.removeprefix("response.")
            metadata = {
                "status": status,
                "incomplete_details": None,
                "error": terminal_event_error,
            }
            if with_tools:
                return LLMToolResponse(
                    content=None,
                    tool_calls=[],
                    model=model,
                    usage={},
                    raw_response=None,
                    raw_assistant_message=_tool_calls_to_raw_assistant("", []),
                    **metadata,
                )
            return LLMResponse(
                content="".join(text_parts),
                model=model,
                usage={},
                raw_response=None,
                **metadata,
            )
        dump = completed_response.model_dump()
        final_text, tool_calls, citations, usage = _parse_responses_dump(dump)
        text = final_text or "".join(text_parts)
        metadata = self._response_metadata(
            completed_response,
            dump,
            fallback_status=(
                terminal_event_type.removeprefix("response.")
                if terminal_event_type else None
            ),
        )
        if metadata["error"] is None and terminal_event_type == "response.failed":
            metadata["error"] = terminal_event_error
        if with_tools:
            raw_assistant = _tool_calls_to_raw_assistant(text, tool_calls)
            return LLMToolResponse(
                content=text or None,
                tool_calls=tool_calls,
                model=getattr(completed_response, "model", model),
                usage=usage,
                raw_response=completed_response,
                raw_assistant_message=raw_assistant,
                **metadata,
            )
        return LLMResponse(
            content=text,
            model=getattr(completed_response, "model", model),
            usage=usage,
            raw_response=completed_response,
            citations=citations,
            **metadata,
        )

    async def complete(self, model, messages, temperature, max_tokens, timeout=None,
                       skip_temperature=False, tools=None, stream=False,
                       defer_stream=False) -> LLMResponse:
        params = self._build_params(
            model,
            messages,
            temperature,
            max_tokens,
            tools=tools,
            timeout=timeout,
            skip_temperature=skip_temperature,
            background_ok=True,
            stream=stream,
        )
        resp = await self._create_responses_with_fallback(params)
        if stream:
            started = await _start_stream(
                resp,
                lambda value: self._complete_stream_response(value, model),
                timeout=timeout,
                label="Responses text stream",
            )
            return started if defer_stream else await started.finish()
        resp = await self._await_completion(resp)
        dump = resp.model_dump()
        text, _tools, citations, usage = _parse_responses_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp, citations=citations,
                           **self._response_metadata(resp, dump))

    async def complete_with_tools(self, model, messages, tools, tool_choice,
                                  temperature, max_tokens, skip_temperature=False,
                                  stream=False, defer_stream=False) -> LLMToolResponse:
        params = self._build_params(model, messages, temperature, max_tokens,
                                    tools=tools, tool_choice=tool_choice,
                                    skip_temperature=skip_temperature, stream=stream)
        resp = await self._create_responses_with_fallback(params)
        if stream:
            started = await _start_stream(
                resp,
                lambda value: self._complete_stream_response(
                    value, model, with_tools=True
                ),
                timeout=None,
                label="Responses tool stream",
            )
            return started if defer_stream else await started.finish()
        resp = await self._await_completion(resp)
        dump = resp.model_dump()
        text, tool_calls, _citations, usage = _parse_responses_dump(dump)
        raw_assistant = _tool_calls_to_raw_assistant(text, tool_calls)
        return LLMToolResponse(content=text or None, tool_calls=tool_calls,
                               model=getattr(resp, "model", model), usage=usage,
                               raw_response=resp, raw_assistant_message=raw_assistant,
                               **self._response_metadata(resp, dump))

    async def complete_vision(self, model, text_prompt, images, temperature,
                              max_tokens, timeout=None, skip_temperature=False,
                              system_prompt=None, stream=False,
                              defer_stream=False) -> LLMResponse:
        content = [{"type": "input_text", "text": text_prompt}]
        for img_b64 in images:
            mime = _detect_image_mime_type(img_b64)
            content.append({"type": "input_image", "image_url": f"data:{mime};base64,{img_b64}"})
        messages = _inject_english_enforcement(
            ([{"role": "system", "content": system_prompt}]
             if system_prompt else [])
            + [{"role": "user", "content": content}]
        )
        params = self._build_params(
            model, messages, temperature, max_tokens, stream=stream,
            timeout=timeout, skip_temperature=skip_temperature,
        )
        resp = await self._create_responses_with_fallback(params)
        if stream:
            started = await _start_stream(
                resp,
                lambda value: self._complete_stream_response(value, model),
                timeout=timeout,
                label="Responses vision stream",
            )
            return started if defer_stream else await started.finish()

        resp = await self._await_completion(resp)
        dump = resp.model_dump()
        text, _tools, citations, usage = _parse_responses_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp, citations=citations,
                           **self._response_metadata(resp, dump))

    async def stream(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False):
        """Yield text deltas from a streaming Responses call. Connection
        failures (incl. endpoint-absent) raise before any chunk is yielded."""
        params = self._build_params(model, messages, temperature, max_tokens,
                                    stream=True, timeout=timeout, skip_temperature=skip_temperature)
        stream = await self._create_responses_with_fallback(params)
        try:
            async for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield delta
        finally:
            await _close_stream(stream)
