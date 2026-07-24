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
from .google import _DR_BACKGROUND_MAX_POLL_ERRORS, _DR_BACKGROUND_MAX_WAIT, _DR_BACKGROUND_POLL_INTERVAL, _DR_BACKGROUND_TERMINAL_FAILED
from .media import ENGLISH_ENFORCEMENT, _detect_image_mime_type
from .model_traits import _WEB_SEARCH_TOOL_NAMES, _get_effective_temperature, _is_deep_research_model
from .protocol import _NO_BACKGROUND_ENDPOINT, _anthropic_messages_translate, _anthropic_restore_thinking, _anthropic_translate_tool_choice, _anthropic_translate_tools, _parse_anthropic_dump, _parse_responses_dump, _responses_messages_to_input, _responses_translate_tool_choice, _responses_translate_tools
from .responses import LLMResponse, LLMToolResponse, _tool_calls_to_raw_assistant


_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096

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
                               raw_response=resp, raw_assistant_message=raw_assistant)

    async def complete(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False) -> LLMResponse:
        params = self._build_params(model, messages, temperature, max_tokens, skip_temperature=skip_temperature)
        resp = await self.client.messages.create(**params)
        dump = resp.model_dump()
        text, _tc, _think, usage = _parse_anthropic_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp)

    async def complete_with_tools(self, model, messages, tools, tool_choice,
                                  temperature, max_tokens, skip_temperature=False) -> LLMToolResponse:
        params = self._build_params(model, messages, temperature, max_tokens,
                                    tools=tools, tool_choice=tool_choice, skip_temperature=skip_temperature)
        resp = await self.client.messages.create(**params)
        return self._to_tool_response(resp, model)

    async def complete_vision(self, model, text_prompt, images, temperature,
                              max_tokens, timeout=None, skip_temperature=False) -> LLMResponse:
        content = [{"type": "text", "text": text_prompt}]
        for img_b64 in images:
            mime = _detect_image_mime_type(img_b64)
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": mime, "data": img_b64}})
        messages = [
            {"role": "system", "content": ENGLISH_ENFORCEMENT},
            {"role": "user", "content": content},
        ]
        params = self._build_params(model, messages, temperature, max_tokens, skip_temperature=skip_temperature)
        if timeout is not None:
            params["timeout"] = timeout
        resp = await self.client.messages.create(**params)
        dump = resp.model_dump()
        text, _tc, _think, usage = _parse_anthropic_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp)

    async def stream(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False):
        params = self._build_params(model, messages, temperature, max_tokens, skip_temperature=skip_temperature)
        params["stream"] = True
        stream = await self.client.messages.create(**params)
        async for event in stream:
            if getattr(event, "type", "") == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "type", "") == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        yield text

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

    async def complete(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False, tools=None) -> LLMResponse:
        params = self._build_params(
            model,
            messages,
            temperature,
            max_tokens,
            tools=tools,
            timeout=timeout,
            skip_temperature=skip_temperature,
            background_ok=True,
        )
        resp = await self._create_responses_with_fallback(params)
        resp = await self._await_completion(resp)
        dump = resp.model_dump()
        text, _tools, citations, usage = _parse_responses_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp, citations=citations)

    async def complete_with_tools(self, model, messages, tools, tool_choice,
                                  temperature, max_tokens, skip_temperature=False) -> LLMToolResponse:
        params = self._build_params(model, messages, temperature, max_tokens,
                                    tools=tools, tool_choice=tool_choice, skip_temperature=skip_temperature)
        resp = await self._create_responses_with_fallback(params)
        resp = await self._await_completion(resp)
        dump = resp.model_dump()
        text, tool_calls, _citations, usage = _parse_responses_dump(dump)
        raw_assistant = _tool_calls_to_raw_assistant(text, tool_calls)
        return LLMToolResponse(content=text or None, tool_calls=tool_calls,
                               model=getattr(resp, "model", model), usage=usage,
                               raw_response=resp, raw_assistant_message=raw_assistant)

    async def complete_vision(self, model, text_prompt, images, temperature,
                              max_tokens, timeout=None, skip_temperature=False) -> LLMResponse:
        content = [{"type": "input_text", "text": text_prompt}]
        for img_b64 in images:
            mime = _detect_image_mime_type(img_b64)
            content.append({"type": "input_image", "image_url": f"data:{mime};base64,{img_b64}"})
        messages = [
            {"role": "system", "content": ENGLISH_ENFORCEMENT},
            {"role": "user", "content": content},
        ]
        params = self._build_params(model, messages, temperature, max_tokens, timeout=timeout, skip_temperature=skip_temperature)
        resp = await self._create_responses_with_fallback(params)
        resp = await self._await_completion(resp)
        dump = resp.model_dump()
        text, _tools, citations, usage = _parse_responses_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp, citations=citations)

    async def stream(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False):
        """Yield text deltas from a streaming Responses call. Connection
        failures (incl. endpoint-absent) raise before any chunk is yielded."""
        params = self._build_params(model, messages, temperature, max_tokens,
                                    stream=True, timeout=timeout, skip_temperature=skip_temperature)
        stream = await self._create_responses_with_fallback(params)
        async for event in stream:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    yield delta
