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

from .llm_logger import get_llm_logger, LLMInteraction

logger = logging.getLogger(__name__)

# Implementations live in ./llm_client_parts/; re-exported here so existing
# importers and tests keep resolving them from this module.

# Re-exported so `backend.utils.llm_client.<name>` keeps resolving
# for existing importers and for tests that patch these by name.
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
        self.session_id = session_id
        self.agent_name = agent_name
        self.llm_logger = get_llm_logger()
        # Optional callback fired before each retry sleep, used to push
        # status updates to WebSocket so the frontend knows we're still alive.
        # Signature: async (attempt, max_retries, delay, error_type) -> None
        self.on_retry: Optional[Callable[[int, int, float, str], Awaitable[None]]] = None

        # Session-level cache of models that rejected temperature parameter
        # Key is model name (extracted), value is True
        self._temp_unsupported_models: set[str] = set()

        # P3-4: Session-level override cache mapping model_name -> field name
        # ("max_tokens" or "max_completion_tokens") that the provider accepts.
        # Populated reactively when a BadRequestError indicates the wrong field
        # was sent (e.g. an o-series variant routed through OpenRouter that the
        # static _is_o_series check missed, or vice versa).
        self._max_tokens_field_overrides: dict[str, str] = {}

        # Session-level cache of models that rejected non-"auto" tool_choice.
        # Key is the extracted model name; when present, chat_with_tools demotes
        # tool_choice to "auto" pre-emptively to avoid the round-trip.
        self._toolchoice_unsupported_models: set[str] = set()

        # P2-1: asyncio.Lock protecting check-then-add on the unsupported
        # model caches. Concurrent first-time probes for the same model could
        # otherwise both proceed past the membership check before either adds.
        self._temp_cache_lock: asyncio.Lock = asyncio.Lock()
        self._toolchoice_cache_lock: asyncio.Lock = asyncio.Lock()

        # Always create an explicit HTTP client with tuned timeouts and pool settings,
        # so both proxy and non-proxy scenarios get the same configuration.
        http_client_kwargs = {
            "timeout": httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
        }
        if proxy_url:
            http_client_kwargs["proxy"] = proxy_url
        self.http_client = httpx.AsyncClient(**http_client_kwargs)
        logger.info(f"[{agent_name}] LLM client initialized (session={session_id}, proxy={'set' if proxy_url else 'none'})")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=_openai_sdk_base_url(self.base_url),
            http_client=self.http_client,
            timeout=300.0,  # Safety net; deep research uses per-request timeout=1800 override
            max_retries=0,  # Disable SDK internal retries; retry_with_backoff handles all retries
            default_headers={
                "HTTP-Referer": "https://scai.local",
                "X-Title": "ScAI - Single Cell Sequencing Analysis Driven by AI"
            }
        )

        # Protocol adapters (lazily built). None until first use.
        self._responses_adapter: Optional[_ResponsesAdapter] = None
        self._anthropic_adapter: Optional[_AnthropicAdapter] = None

        # H20: Idempotent-close guard. Set True on first close(); prevents
        # double-free of the httpx.AsyncClient and its underlying connection pool.
        self._closed: bool = False

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
        
    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self.client.close()
        finally:
            await self.http_client.aclose()


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
