"""
LLM client utilities using OpenAI SDK for OpenAI-compatible APIs
Supports OpenRouter, OpenAI, and other compatible endpoints
"""
import base64
import httpx
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError, InternalServerError, BadRequestError

from .llm_logger import get_llm_logger, LLMInteraction

logger = logging.getLogger(__name__)

# English enforcement instruction injected into every LLM call
ENGLISH_ENFORCEMENT = (
    "CRITICAL INSTRUCTION: You MUST respond exclusively in English. "
    "All output — analysis, descriptions, findings, instructions, JSON values, "
    "and communications — must be written in English regardless of the language "
    "of input data or context."
)


def _inject_english_enforcement(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Inject English enforcement instruction into the message list.

    If a system message exists, appends the enforcement to it.
    Otherwise, prepends a new system message.
    Returns a shallow-copied list to avoid mutating the original.
    """
    messages = list(messages)  # shallow copy
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            messages[i] = {**msg, "content": msg["content"] + "\n\n" + ENGLISH_ENFORCEMENT}
            return messages
    # No system message found — prepend one
    messages.insert(0, {"role": "system", "content": ENGLISH_ENFORCEMENT})
    return messages


def _detect_image_mime_type(img_b64: str) -> str:
    """Detect image MIME type from base64-encoded data using magic bytes.

    Checks the first few decoded bytes to identify JPEG, PNG, GIF, or WebP.
    Falls back to 'image/png' if the format cannot be determined.
    """
    try:
        # 12 base64 chars → 9 bytes, enough to cover all magic byte sequences
        header = base64.b64decode(img_b64[:12] + "==")
        if header[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        if header[:4] == b'RIFF' and len(header) >= 8:
            return "image/webp"
    except Exception as e:
        logger.debug(f"Image MIME type detection failed, defaulting to image/png: {e}")
    return "image/png"


# Retry configuration
MAX_RETRIES = 4
MAX_RETRIES_UPSTREAM_5XX = 8  # Dedicated for flaky upstream (e.g., Cloudflare 524)
RETRY_DELAY_BASE = 2.0  # Base delay in seconds, will be multiplied by attempt number
# Rate limit (429) uses fixed escalating delays to handle TPM limits
RATE_LIMIT_DELAYS = [5, 10, 20, 40]  # seconds between retries
# Note: Only retry transient errors. Non-transient errors like AuthenticationError,
# BadRequestError, NotFoundError etc. should NOT be retried.
RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    InternalServerError,  # 500 - transient server errors
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    asyncio.TimeoutError,
)
# JSONDecodeError gets fewer retries (max 2) since repeated bad JSON likely indicates
# a persistent issue rather than a transient network problem.
MAX_JSON_RETRIES = 2


class UpstreamExhaustedError(RuntimeError):
    """Raised when retries are exhausted against a flaky upstream (5xx/timeouts)."""

# OpenAI reasoning models that do NOT support the temperature parameter at all.
# Matched by exact model name segment (e.g., "openai/o4-mini" -> "o4-mini").
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4")

# Gemini 3.x thinking models: temperature is accepted but Google strongly recommends
# keeping it at 1.0. Lower values cause looping or degraded output quality.
_GEMINI3_PREFIX = "gemini-3"

# Models that do NOT allow custom temperature — the parameter must be omitted entirely.
# (e.g., Moonshot kimi-k2.5: docs state "不能修改该参数")
_NO_CUSTOM_TEMP_MODELS = {"kimi-k2.5"}


def _extract_model_name(model: str) -> str:
    """Extract model name after provider prefix (e.g. 'google/gemini-3.1-pro' -> 'gemini-3.1-pro')."""
    return model.split("/")[-1] if "/" in model else model


def _is_o_series(model_name: str) -> bool:
    """Check if a model name matches OpenAI o-series reasoning models."""
    for prefix in _OPENAI_REASONING_PREFIXES:
        if model_name == prefix or model_name.startswith(prefix + "-"):
            return True
    return False


def _get_effective_temperature(model: str, temperature: float) -> Optional[float]:
    """Determine the effective temperature for a model.

    - OpenAI o-series reasoning models: returns None (parameter not supported)
    - Models in _NO_CUSTOM_TEMP_MODELS: returns None (must omit temperature)
    - Gemini 3.x thinking models: returns 1.0 (Google recommends default for thinking)
    - All other models: returns the requested temperature as-is
    """
    model_name = _extract_model_name(model)
    # OpenAI reasoning models: temperature not supported at all
    if _is_o_series(model_name):
        return None
    # Models that prohibit custom temperature: omit entirely
    if model_name in _NO_CUSTOM_TEMP_MODELS:
        return None
    # Gemini 3.x thinking models: force temperature=1.0
    if model_name.startswith(_GEMINI3_PREFIX):
        return 1.0
    return temperature


def is_temperature_error(error: Exception) -> bool:
    """Check if a BadRequestError is related to the temperature parameter."""
    if not isinstance(error, BadRequestError):
        return False

    error_str = str(error).lower()
    temperature_keywords = [
        "temperature",
        "only 1 is allowed",
        "invalid temperature",
    ]
    return any(kw in error_str for kw in temperature_keywords)


async def retry_with_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    retry_delay_base: float = RETRY_DELAY_BASE,
    agent_name: str = "unknown",
    request_timeout: float = 300.0,
    on_retry: Optional[Callable[[int, int, float, str], Awaitable[None]]] = None,
):
    """
    Execute an async function with exponential backoff retry on timeout/connection errors

    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts
        retry_delay_base: Base delay between retries (multiplied by attempt number)
        agent_name: Agent name for logging
        request_timeout: Hard per-attempt timeout in seconds (asyncio-level deadline).
            Fires even if the HTTP client hangs during response body reading.
        on_retry: Optional async callback invoked before each retry sleep.
            Signature: (attempt, max_retries, delay, error_type) -> None.
            Useful for sending progress updates to WebSocket clients during
            long retry waits.

    Returns:
        Result from the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    effective_max = max_retries

    json_error_count = 0

    async def _notify_retry(attempt: int, delay: float, error_type: str):
        """Fire the on_retry callback if provided, swallowing errors."""
        if on_retry is not None:
            try:
                await on_retry(attempt, max_retries, delay, error_type)
            except Exception:
                pass  # Never let notification failure break retry logic

    for attempt in range(effective_max + 1):
        try:
            return await asyncio.wait_for(func(), timeout=request_timeout)
        except json.JSONDecodeError as e:
            # JSON errors get fewer retries — repeated bad JSON is likely persistent
            json_error_count += 1
            last_exception = e
            if json_error_count < MAX_JSON_RETRIES:
                delay = retry_delay_base * json_error_count
                logger.warning(
                    f"[{agent_name}] Invalid JSON response (attempt {json_error_count}/{MAX_JSON_RETRIES}): {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await _notify_retry(attempt, delay, "JSONDecodeError")
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] Invalid JSON after {MAX_JSON_RETRIES} attempts: {str(e)}"
                )
                raise
        except RateLimitError as e:
            # RateLimitError needs special handling with longer backoff delays.
            last_exception = e
            rate_limit_max = min(max_retries, len(RATE_LIMIT_DELAYS))
            if attempt < rate_limit_max:
                delay = RATE_LIMIT_DELAYS[attempt]
                logger.warning(
                    f"[{agent_name}] Rate limited (attempt {attempt + 1}/{rate_limit_max + 1}). "
                    f"Retrying in {delay}s..."
                )
                await _notify_retry(attempt, delay, "RateLimitError")
                await asyncio.sleep(delay)
            else:
                logger.error(f"[{agent_name}] Rate limit exceeded after {rate_limit_max + 1} attempts")
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            is_5xx = isinstance(e, InternalServerError)
            if is_5xx:
                effective_max = max(effective_max, MAX_RETRIES_UPSTREAM_5XX)
            current_max = effective_max
            if attempt < current_max:
                delay = retry_delay_base * (attempt + 1)
                error_label = "InternalServerError(5xx)" if is_5xx else type(e).__name__
                logger.warning(
                    f"[{agent_name}] Request failed (attempt {attempt + 1}/{current_max + 1}): {error_label}: {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await _notify_retry(attempt, delay, error_label)
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] Request failed after {current_max + 1} attempts: {type(e).__name__}: {str(e)}"
                )

    raise UpstreamExhaustedError(
        f"[{agent_name}] Upstream LLM exhausted after {effective_max + 1} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    ) from last_exception


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any


@dataclass
class ToolCall:
    """Represents a tool/function call from the model"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMToolResponse:
    """Response from LLM with tool calls"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    model: str
    usage: Dict[str, int]
    raw_response: Any
    raw_assistant_message: Optional[Dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class OpenAICompatibleClient:
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
            base_url=self.base_url,
            http_client=self.http_client,
            timeout=300.0,  # Safety net; deep research uses per-request timeout=1800 override
            max_retries=0,  # Disable SDK internal retries; retry_with_backoff handles all retries
            default_headers={
                "HTTP-Referer": "https://scai.local",
                "X-Title": "ScAI - Single Cell Sequencing Analysis Driven by AI"
            }
        )
    
    def set_context(self, session_id: str, agent_name: str):
        """Set session and agent context for logging"""
        self.session_id = session_id
        self.agent_name = agent_name
        
    async def close(self):
        try:
            await self.client.close()
        finally:
            await self.http_client.aclose()

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
                if _is_o_series(model_name):
                    params["max_completion_tokens"] = max_tokens
                else:
                    params["max_tokens"] = max_tokens
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
            if _is_o_series(model_name):
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

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
        # If model is already known to reject temperature, skip directly
        if model_key in self._temp_unsupported_models:
            return await retry_with_backoff(make_request_no_temp, **retry_kwargs)
        try:
            return await retry_with_backoff(make_request, **retry_kwargs)
        except BadRequestError as e:
            if is_temperature_error(e):
                logger.warning(
                    f"[{self.agent_name}] Temperature rejected by {model_key}, "
                    f"retrying without temperature..."
                )
                self._temp_unsupported_models.add(model_key)
                return await retry_with_backoff(make_request_no_temp, **retry_kwargs)
            else:
                raise

    def _build_retry_kwargs(self, request_timeout: Optional[float] = None, max_retries_override: Optional[int] = None) -> Dict[str, Any]:
        """统一构建 retry_kwargs，保持各方法逻辑一致."""
        kwargs: Dict[str, Any] = {"agent_name": self.agent_name}
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
            if max_retries_override is not None:
                kwargs["max_retries"] = max_retries_override
            elif request_timeout > 300:
                kwargs["max_retries"] = 0  # Don't retry expensive long-running requests
        if self.on_retry is not None:
            kwargs["on_retry"] = self.on_retry
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
                    # Reasoning model fallback: content in reasoning/reasoning_content
                    if not content:
                        for key in ("reasoning", "reasoning_content"):
                            val = msg_data.get(key)
                            if val and isinstance(val, str):
                                content = val
                                break
            except Exception as e:
                logger.debug(f"[{self.agent_name}] model_dump extraction failed: {e}")

            # Fallback: check text attribute
            if not content and hasattr(message, "text"):
                content = message.text or ""
            # Fallback: check reasoning attribute directly on the object
            if not content:
                for attr in ("reasoning", "reasoning_content"):
                    val = getattr(message, attr, None)
                    if val and isinstance(val, str):
                        content = val
                        break

        # Case 4: Other types — convert to string
        if not content and raw_content is not None:
            content = str(raw_content)

        # Check for refusal
        if not content and hasattr(message, "refusal") and message.refusal:
            content = f"[Refusal]: {message.refusal}"
            logger.warning(f"[{self.agent_name}] Request was refused: {message.refusal}")

        # Debug logging: tokens generated but content empty
        completion_tokens = 0
        has_tool_calls = False
        if not content and response and response.usage:
            completion_tokens = response.usage.completion_tokens or 0
            if completion_tokens > 0:
                # Skip warning when the response is a tool call — no text content is expected
                has_tool_calls = bool(getattr(message, "tool_calls", None))
                if not has_tool_calls:
                    # Dump raw message object for debugging
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
                    for attr_name in ("content", "refusal", "role", "reasoning", "reasoning_content"):
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

        # If tokens were generated but content extraction failed, return an error
        # message instead of empty string so downstream agents can detect the failure.
        if not content and completion_tokens and completion_tokens > 0 and not has_tool_calls:
            logger.warning(
                f"[{self.agent_name}] Non-zero completion tokens ({completion_tokens}) "
                f"but empty extracted content — returning error placeholder"
            )
            return "[Error: LLM generated a response but content extraction failed. Please retry.]"

        return content

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> LLMResponse:
        """
        Send chat request to OpenAI-compatible API with automatic retry on timeout

        Args:
            model: Model identifier (e.g., "google/gemini-2.5-pro-preview-05-06")
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            timeout: Per-request timeout in seconds (overrides client default)

        Returns:
            LLMResponse object
        """
        # Enforce English output
        messages = _inject_english_enforcement(messages)

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
        retry_kwargs = self._build_retry_kwargs(request_timeout=timeout, max_retries_override=max_retries)

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

            result = LLMResponse(
                content=content,
                model=response.model if response else model,
                usage=self._extract_usage(response),
                raw_response=response
            )

            # Log response
            self.llm_logger.log_response(
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
            self.llm_logger.log_response(
                interaction=interaction,
                error=error_msg,
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
        Send chat request with function/tool calling support and automatic retry on timeout
        
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
        # Enforce English output
        messages = _inject_english_enforcement(messages)

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
        base_params = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )
        retry_kwargs = self._build_retry_kwargs()

        try:
            logger.info(f"[{self.agent_name}] Sending chat_with_tools request to {model}")
            response = await self._call_with_temp_fallback(
                _make_request, _make_request_no_temp, model_key, retry_kwargs
            )

            # Safely extract message from response
            message = None
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message

            # Parse tool calls if present
            tool_calls = []
            tool_calls_log = []
            if message and message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}

                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments
                    ))
                    tool_calls_log.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments
                    })

            # Extract content — handles reasoning models, list content, etc.
            content = self._extract_content_from_message(message, response) if message else None

            # Preserve the full raw assistant message so callers can echo it back verbatim.
            # This is required for providers like Gemini that attach extra fields (e.g.
            # thought_signature) to tool calls and require them in subsequent requests.
            raw_assistant_message = None
            if message:
                try:
                    msg_dict = message.model_dump()
                    raw_assistant_message = {
                        "role": "assistant",
                        "content": msg_dict.get("content"),
                        "tool_calls": msg_dict.get("tool_calls"),
                    }
                except Exception as e:
                    logger.debug(f"[{self.agent_name}] Failed to preserve raw assistant message: {e}")

            result = LLMToolResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model if response else model,
                usage=self._extract_usage(response),
                raw_response=response,
                raw_assistant_message=raw_assistant_message
            )

            # Log response
            self.llm_logger.log_response(
                interaction=interaction,
                response_content=content,
                tool_calls=tool_calls_log if tool_calls_log else None,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except Exception as e:
            logger.exception(f"[{self.agent_name}] Chat with tools error")

            # Log error
            self.llm_logger.log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise
    
    async def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response from OpenAI-compatible API.

        Connection establishment is retried with backoff on transient errors.
        Mid-stream failures propagate to the caller (partial content already yielded).

        Yields:
            Content chunks
        """
        logger.info(f"[{self.agent_name}] Starting stream request to {model}")

        # Enforce English output
        messages = _inject_english_enforcement(messages)

        # Extract model name for cache key
        model_key = _extract_model_name(model)

        base_params = {"model": model, "messages": messages, "stream": True}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )

        # Retry connection establishment (most common failure point)
        stream_retry_kwargs = self._build_retry_kwargs(request_timeout=120.0)
        stream_retry_kwargs["max_retries"] = MAX_RETRIES  # stream needs explicit retries

        stream = await self._call_with_temp_fallback(
            _make_request, _make_request_no_temp, model_key, stream_retry_kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
        Send chat request with images to OpenAI-compatible API with automatic retry on timeout

        Args:
            model: Model identifier
            text_prompt: Text prompt
            images: List of base64 encoded images
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse object
        """
        # Build multimodal content (text first, then images — per OpenRouter recommendation)
        content = []

        # Add text prompt first
        content.append({
            "type": "text",
            "text": text_prompt
        })

        for img_b64 in images:
            mime_type = _detect_image_mime_type(img_b64)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_b64}"
                }
            })

        messages = [
            {"role": "system", "content": ENGLISH_ENFORCEMENT},
            {"role": "user", "content": content},
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
        retry_kwargs = self._build_retry_kwargs(request_timeout=request_timeout)

        try:
            logger.info(f"[{self.agent_name}] Sending vision request to {model} with {len(images)} images")
            response = await self._call_with_temp_fallback(
                _make_request, _make_request_no_temp, model_key, retry_kwargs
            )

            # Extract content — uses shared helper for reasoning / multimodal formats
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

            # Log response
            self.llm_logger.log_response(
                interaction=interaction,
                response_content=result.content,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except Exception as e:
            logger.exception(f"[{self.agent_name}] Vision error")

            # Log error
            self.llm_logger.log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise


class GoogleDeepResearchClient:
    """Client for Google Gemini Deep Research via Interactions API"""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    DEFAULT_MODEL = "deep-research-pro-preview-12-2025"

    def __init__(self, api_key: str, proxy_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        # httpx will also read from HTTP_PROXY/HTTPS_PROXY env vars if trust_env=True (default)
        if proxy_url:
            self.client = httpx.AsyncClient(
                timeout=300.0,
                proxy=proxy_url
            )
        else:
            self.client = httpx.AsyncClient(timeout=300.0)  # Long timeout for research

    async def close(self):
        await self.client.aclose()

    async def start_research(
        self,
        query: str,
        background: bool = True
    ) -> str:
        """
        Start a Deep Research task

        Args:
            query: Research query
            background: Whether to run in background

        Returns:
            Interaction ID
        """
        url = f"{self.BASE_URL}/interactions"

        payload = {
            "input": query,
            "agent": self.model,
            "background": background
        }

        response = await self.client.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            json=payload
        )
        response.raise_for_status()

        data = response.json()
        # Handle both "id" field and "name" resource path format
        return data.get("id") or data.get("name", "").split("/")[-1]
    
    async def poll_research(
        self,
        interaction_id: str,
        poll_interval: float = 5.0,
        max_wait: float = 1800.0  # 30 minutes max
    ) -> Dict[str, Any]:
        """
        Poll for research results
        
        Args:
            interaction_id: Interaction ID from start_research
            poll_interval: Seconds between polls
            max_wait: Maximum seconds to wait
            
        Returns:
            Research results
        """
        url = f"{self.BASE_URL}/interactions/{interaction_id}"
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            response = await self.client.get(
                url,
                headers={"x-goog-api-key": self.api_key}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check if completed (handle both "status" and legacy "state", case-insensitive)
            status = data.get("status", data.get("state", "")).lower()
            if status == "completed":
                return self._extract_research_result(data)
            elif status == "failed":
                raise Exception(f"Research failed: {data.get('error', 'Unknown error')}")
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError("Research timed out")
    
    def _extract_research_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research result from Interactions API response"""
        result = {
            "report": "",
            "sources": [],
            "thinking": []
        }

        # Try new format first: outputs array (Interactions API)
        outputs = data.get("outputs", [])
        if outputs:
            for output_item in outputs:
                if isinstance(output_item, dict):
                    # Direct text field
                    if "text" in output_item:
                        result["report"] += output_item["text"]
                    # Nested parts format
                    elif "parts" in output_item:
                        for part in output_item["parts"]:
                            if isinstance(part, dict) and "text" in part:
                                result["report"] += part["text"]
                    # Extract grounding metadata from each output
                    grounding = output_item.get("groundingMetadata", {})
                    self._extract_sources(grounding, result["sources"])
        else:
            # Fallback: legacy output.parts[].text format
            output = data.get("output", {})
            if isinstance(output, dict):
                for part in output.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        result["report"] += part["text"]
                grounding = output.get("groundingMetadata", {})
                self._extract_sources(grounding, result["sources"])

        # Also check top-level groundingMetadata
        top_grounding = data.get("groundingMetadata", {})
        if top_grounding:
            self._extract_sources(top_grounding, result["sources"])

        return result

    @staticmethod
    def _extract_sources(grounding: Dict[str, Any], sources: List[Dict[str, str]]):
        """Extract source URLs from grounding metadata"""
        seen_uris = {s.get("uri") for s in sources}
        for chunk in grounding.get("groundingChunks", []):
            if "web" in chunk:
                uri = chunk["web"].get("uri", "")
                if uri and uri not in seen_uris:
                    sources.append({
                        "title": chunk["web"].get("title", ""),
                        "uri": uri
                    })
                    seen_uris.add(uri)
    
    async def research(self, query: str) -> Dict[str, Any]:
        """
        Convenience method to start research and wait for results
        
        Args:
            query: Research query
            
        Returns:
            Research results with report and sources
        """
        interaction_id = await self.start_research(query)
        return await self.poll_research(interaction_id)


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
