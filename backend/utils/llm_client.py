"""
LLM client utilities using OpenAI SDK for OpenAI-compatible APIs
Supports OpenRouter, OpenAI, and other compatible endpoints
"""
import base64
import html
import httpx
import ipaddress
import inspect
import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError, InternalServerError, BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError

from .llm_logger import get_llm_logger, LLMInteraction

logger = logging.getLogger(__name__)


def _proxy_bypass_for_url(base_url: str) -> bool:
    """Do not send loopback/private/link-local or explicit local hosts via proxy."""
    host = (urlparse(base_url).hostname or "").strip(".").lower()
    if not host:
        return False
    if host in {"localhost", "localhost.localdomain", "ip6-localhost"}:
        return True
    if (
        host.endswith((".localhost", ".internal", ".local"))
        or "." not in host
    ):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
    )

# Implementations live in ./llm_client_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.

# Re-exported so `backend.utils.llm_client.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
from .llm_client_parts.retry import DEFAULT_REQUEST_TIMEOUT  # noqa: F401
from .llm_client_parts.shared import (  # noqa: F401
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
    ToolProtocolError,
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
    _NO_RESPONSES_VISION_STREAM,
    _NO_TEXT_STREAM,
    _NO_CHAT_STREAM_OPTIONS,
    _PROBE_CACHE,
    _PROBE_LOCKS,
    _url_path_suffix_protocol,
    _DEFAULT_PROTOCOL_ORDER,
    PROBE_RETRY_DELAYS,
    TEXT_MODALITY,
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
    _ANTHROPIC_DEFAULT_MAX_TOKENS,
    StreamInterruptedError,
    _StartedStream,
    _start_stream,
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
from .llm_client_parts import (  # noqa: F401
    ClientParamsMixin,
    ClientContentMixin,
    ClientDispatchMixin,
    ClientChatMixin,
    ClientStreamVisionMixin,
)
class OpenAICompatibleClient(
    ClientParamsMixin,
    ClientContentMixin,
    ClientDispatchMixin,
    ClientChatMixin,
    ClientStreamVisionMixin,
):
    """Client for OpenAI-compatible APIs (OpenRouter, OpenAI, etc.)"""
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
        session_id: str = "default",
        agent_name: str = "unknown",
    ):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("LLM api_key is not configured; set it in Config")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.proxy_url = proxy_url
        self.session_id = session_id
        self.agent_name = agent_name
        self.llm_logger = get_llm_logger()
        # Optional callback fired before each retry sleep, used to push
        # status updates to WebSocket so the frontend knows we're still alive.
        # Signature: async (attempt, max_retries, delay, error_type, agent_name) -> None
        self.on_retry: Optional[Callable[[int, int, float, str, str], Awaitable[None]]] = None
        # Optional lifecycle gate awaited after retry backoff and immediately
        # before another request.  Unlike on_retry, this callback is not
        # best-effort: cancellation and gate failures propagate.
        self.before_retry: Optional[Callable[[], Awaitable[None]]] = None

        # Session-level cache of models that rejected temperature parameter
        # Key is model name (extracted), value is True
        self._temp_unsupported_models: set[str] = set()

        # P3-4: Session-level override cache mapping model_name -> field name
        # ("max_tokens" or "max_completion_tokens") that the provider accepts.
        # Populated reactively when a BadRequestError indicates the wrong field
        # was sent (e.g. an o-series variant routed through OpenRouter that the
        # static _is_o_series check missed, or vice versa).
        self._max_tokens_field_overrides: dict[str, str] = {}
        self._parameter_adaptation_locks: dict[str, asyncio.Lock] = {}

        # Session-level cache of models that rejected non-"auto" tool_choice.
        # Key is the extracted model name; when present, chat_with_tools demotes
        # tool_choice to "auto" pre-emptively to avoid the round-trip.
        self._toolchoice_unsupported_models: set[str] = set()

        # P2-1: asyncio.Lock protecting check-then-add on the unsupported
        # model caches. Concurrent first-time probes for the same model could
        # otherwise both proceed past the membership check before either adds.
        self._temp_cache_lock: asyncio.Lock = asyncio.Lock()
        self._toolchoice_cache_lock: asyncio.Lock = asyncio.Lock()
        self._max_tokens_cache_lock: asyncio.Lock = asyncio.Lock()

        # Always create an explicit HTTP client with tuned timeouts and pool settings,
        # so both proxy and non-proxy scenarios get the same configuration.
        http_client_kwargs = {
            "timeout": httpx.Timeout(
                connect=10.0, read=DEFAULT_REQUEST_TIMEOUT, write=30.0, pool=10.0
            ),
        }
        if proxy_url and not _proxy_bypass_for_url(self.base_url):
            http_client_kwargs["proxy"] = proxy_url
        self.http_client = httpx.AsyncClient(**http_client_kwargs)
        logger.info(f"[{agent_name}] LLM client initialized (session={session_id}, proxy={'set' if proxy_url else 'none'})")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=_openai_sdk_base_url(self.base_url),
            http_client=self.http_client,
            timeout=DEFAULT_REQUEST_TIMEOUT,  # Safety net; deep research uses per-request timeout=1800 override
            max_retries=0,  # Disable SDK internal retries; retry_with_backoff handles all retries
            default_headers={
                "HTTP-Referer": "https://scai.local",
                "X-Title": "ScAI - Single Cell Sequencing Analysis Driven by AI"
            }
        )

        # Protocol adapters (lazily built). None until first use.
        self._responses_adapter: Optional[_ResponsesAdapter] = None
        self._anthropic_adapter: Optional[_AnthropicAdapter] = None

        # H20: Idempotent-close guard. Set only after both owned clients close;
        # a transient failure must remain retryable by workflow cleanup.
        self._closed: bool = False
        self._closing: bool = False
        self._close_lock = asyncio.Lock()
        self._active_requests = 0
        self._requests_idle = asyncio.Event()
        self._requests_idle.set()

    def __del__(self):
        if not getattr(self, '_closed', True):
            logger.warning(
                f"[{self.agent_name}] OpenAICompatibleClient leaked without close() "
                f"(session={self.session_id})"
            )

    def set_context(self, session_id: str, agent_name: str):
        """Set session and agent context for logging"""
        self.session_id = session_id
        self.agent_name = agent_name

    def _ensure_cache_locks(self) -> None:
        # Lazy-init guard for objects constructed via __new__ or fixtures that
        # bypass __init__ (e.g., unit tests). Safe to call repeatedly.
        if getattr(self, "_temp_cache_lock", None) is None:
            self._temp_cache_lock = asyncio.Lock()
        if getattr(self, "_toolchoice_cache_lock", None) is None:
            self._toolchoice_cache_lock = asyncio.Lock()
        if getattr(self, "_max_tokens_cache_lock", None) is None:
            self._max_tokens_cache_lock = asyncio.Lock()
        if getattr(self, "_parameter_adaptation_locks", None) is None:
            self._parameter_adaptation_locks = {}

    def _ensure_lifecycle_state(self) -> None:
        if not hasattr(self, "_closed"):
            self._closed = False
        if not hasattr(self, "_active_requests"):
            self._active_requests = 0
        if getattr(self, "_requests_idle", None) is None:
            self._requests_idle = asyncio.Event()
            if self._active_requests == 0:
                self._requests_idle.set()
        if not hasattr(self, "_closing"):
            self._closing = False

    def _begin_request(self) -> None:
        """Register one public client call before its first await."""
        self._ensure_lifecycle_state()
        if self._closing or self._closed:
            raise RuntimeError("LLM client is closing")
        self._active_requests += 1
        self._requests_idle.clear()

    def _end_request(self) -> None:
        """Release one public client call and wake a waiting close()."""
        self._ensure_lifecycle_state()
        if self._active_requests <= 0:
            return
        self._active_requests -= 1
        if self._active_requests == 0:
            self._requests_idle.set()

    @asynccontextmanager
    async def direct_request_scope(self):
        """Protect one logical operation that uses the owned transports directly.

        Public chat methods register themselves automatically. Callers that need
        the lower-level OpenAI SDK or shared HTTP client must keep this scope for
        the entire logical operation, including retries, polling, and downloads.
        """
        self._begin_request()
        try:
            yield self
        finally:
            self._end_request()

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
        prefer_stream: bool = False,
    ) -> LLMResponse:
        self._begin_request()
        try:
            return await ClientChatMixin.chat(
                self,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                timeout=timeout,
                max_retries=max_retries,
                tools=tools,
                prefer_stream=prefer_stream,
            )
        finally:
            self._end_request()

    async def chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        prefer_stream: bool = False,
    ) -> LLMToolResponse:
        self._begin_request()
        try:
            return await ClientChatMixin.chat_with_tools(
                self,
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                prefer_stream=prefer_stream,
            )
        finally:
            self._end_request()

    async def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield streamed chunks; callers that stop early must call ``aclose()``."""
        self._begin_request()
        try:
            async for chunk in ClientStreamVisionMixin.chat_stream(
                self,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
        finally:
            self._end_request()

    async def chat_with_vision(
        self,
        model: str,
        text_prompt: str,
        images: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        system_prompt: Optional[str] = None,
        prefer_stream: bool = False,
    ) -> LLMResponse:
        self._begin_request()
        try:
            return await ClientStreamVisionMixin.chat_with_vision(
                self,
                model=model,
                text_prompt=text_prompt,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                system_prompt=system_prompt,
                prefer_stream=prefer_stream,
            )
        finally:
            self._end_request()

    async def close(self):
        self._ensure_lifecycle_state()
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = asyncio.Lock()
            self._close_lock = close_lock
        async with close_lock:
            if self._closed:
                return
            # Prevent new calls, then let every call that already owns this
            # client finish before its transport is closed. This covers both
            # config hot-reload retirement and workflow teardown.
            self._closing = True
            await self._requests_idle.wait()

            errors: list[BaseException] = []
            try:
                await self.client.close()
            except BaseException as exc:
                errors.append(exc)
            try:
                await self.http_client.aclose()
            except BaseException as exc:
                errors.append(exc)

            if errors:
                for extra in errors[1:]:
                    logger.debug("Additional LLM client close failure: %s", extra)
                raise errors[0]
            self._closed = True


class LLMManager:
    """Manager for LLM clients (OpenAI-compatible)"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
        session_id: str = "default",
        agent_name: str = "unknown",
    ):
        self.llm = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            proxy_url=proxy_url,
            session_id=session_id,
            agent_name=agent_name,
        )

        # Alias for backward compatibility
        self.openrouter = self.llm

    def set_context(self, session_id: str, agent_name: str):
        """Set session and agent context for logging"""
        self.llm.set_context(session_id, agent_name)

    async def close(self):
        await self.llm.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
