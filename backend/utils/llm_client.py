"""
LLM client utilities using OpenAI SDK for OpenAI-compatible APIs
Supports OpenRouter, OpenAI, and other compatible endpoints
"""
import httpx
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass, field
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError

from .llm_logger import get_llm_logger, LLMInteraction

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # Base delay in seconds, will be multiplied by attempt number
# Include JSONDecodeError for when API returns invalid JSON (truncated response, etc.)
RETRYABLE_EXCEPTIONS = (
    APITimeoutError,
    APIConnectionError,
    APIError,  # General API errors
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    asyncio.TimeoutError,
    json.JSONDecodeError  # Invalid JSON response from API
)

# Models that do not support the temperature parameter (reasoning models)
# Match by model name segments after provider prefix (e.g., "openai/o4-mini" -> "o4-mini")
_NO_TEMPERATURE_PATTERNS = ("o1", "o3", "o4")


def _model_supports_temperature(model: str) -> bool:
    """Check if a model supports the temperature parameter.

    Reasoning models (o1, o3, o4 series) do not support temperature.
    Model string may have provider prefix like 'openai/o4-mini-deep-research'.
    """
    # Extract model name after provider prefix
    model_name = model.split("/")[-1] if "/" in model else model
    # Check if model name starts with any no-temperature pattern
    for pattern in _NO_TEMPERATURE_PATTERNS:
        if model_name.startswith(pattern):
            return False
    return True


async def retry_with_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    retry_delay_base: float = RETRY_DELAY_BASE,
    agent_name: str = "unknown",
    request_timeout: float = 300.0,
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

    Returns:
        Result from the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(func(), timeout=request_timeout)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt < max_retries:
                delay = retry_delay_base * (attempt + 1)
                logger.warning(
                    f"[{agent_name}] Request failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] Request failed after {max_retries + 1} attempts: {type(e).__name__}: {str(e)}"
                )
        except RateLimitError as e:
            # For rate limit errors, wait longer
            last_exception = e
            if attempt < max_retries:
                delay = retry_delay_base * (attempt + 1) * 2  # Double the delay for rate limits
                logger.warning(
                    f"[{agent_name}] Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[{agent_name}] Rate limit exceeded after {max_retries + 1} attempts")
    
    raise last_exception


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
        agent_name: str = "unknown"
    ):
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.session_id = session_id
        self.agent_name = agent_name
        self.llm_logger = get_llm_logger()
        
        # Configure HTTP client with proxy if provided
        http_client = None
        if proxy_url:
            http_client = httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
            )
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            http_client=http_client,
            timeout=120.0,
            max_retries=0,  # Disable SDK internal retries; retry_with_backoff handles all retries
            default_headers={
                "HTTP-Referer": "https://intesc.local",
                "X-Title": "InteSc - Integrated Single Cell Analysis"
            }
        )
    
    def set_context(self, session_id: str, agent_name: str):
        """Set session and agent context for logging"""
        self.session_id = session_id
        self.agent_name = agent_name
        
    async def close(self):
        await self.client.close()
        
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
        if not content and response and response.usage:
            completion_tokens = response.usage.completion_tokens or 0
            if completion_tokens > 0:
                # Dump raw message object for debugging
                try:
                    if hasattr(message, "model_dump"):
                        raw_dump = message.model_dump()
                        logger.info(
                            f"[{self.agent_name}] Empty content with {completion_tokens} tokens. "
                            f"Raw message keys: {list(raw_dump.keys())}, "
                            f"values preview: {str(raw_dump)[:2000]}"
                        )
                except Exception:
                    pass
                msg_dict = {}
                for attr_name in ("content", "refusal", "role", "reasoning", "reasoning_content"):
                    if hasattr(message, attr_name):
                        val = getattr(message, attr_name)
                        msg_dict[attr_name] = str(val)[:200] if val else None
                try:
                    if hasattr(message, "model_dump"):
                        msg_dict["_all_keys"] = list(message.model_dump().keys())
                except Exception:
                    pass
                logger.warning(
                    f"[{self.agent_name}] Response has {completion_tokens} completion tokens "
                    f"but empty content. Message fields: {msg_dict}"
                )

        return content

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        timeout: Optional[float] = None
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
        # Create log record
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=messages,
            interaction_type="chat",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        start_time = time.time()
        error_msg = None
        
        async def _make_request():
            params: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                stream=stream,
            )
            # o-series reasoning models use max_completion_tokens, not max_tokens
            if _model_supports_temperature(model):
                params["temperature"] = temperature
                params["max_tokens"] = max_tokens
            else:
                params["max_completion_tokens"] = max_tokens
            if timeout is not None:
                params["timeout"] = timeout
            return await self.client.chat.completions.create(**params)

        try:
            logger.info(f"[{self.agent_name}] Sending chat request to {model}")
            
            # Use retry mechanism for the API call
            response = await retry_with_backoff(
                _make_request,
                agent_name=self.agent_name
            )
            
            # Extract content — handles reasoning models, list content, etc.
            content = ""
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                content = self._extract_content_from_message(message, response)
            
            result = LLMResponse(
                content=content,
                model=response.model if response else model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response and response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response and response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response and response.usage else 0
                },
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
            logger.error(f"[{self.agent_name}] Chat error: {e}")
            
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
        max_tokens: int = 4096
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
            max_tokens=max_tokens
        )
        
        start_time = time.time()
        
        async def _make_request():
            params: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            # o-series reasoning models use max_completion_tokens, not max_tokens
            if _model_supports_temperature(model):
                params["temperature"] = temperature
                params["max_tokens"] = max_tokens
            else:
                params["max_completion_tokens"] = max_tokens
            return await self.client.chat.completions.create(**params)

        try:
            logger.info(f"[{self.agent_name}] Sending chat_with_tools request to {model}")
            
            # Use retry mechanism for the API call
            response = await retry_with_backoff(
                _make_request,
                agent_name=self.agent_name
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
            
            result = LLMToolResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model if response else model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response and response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response and response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response and response.usage else 0
                },
                raw_response=response
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
            logger.error(f"[{self.agent_name}] Chat with tools error: {e}")
            
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
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response from OpenAI-compatible API
        
        Yields:
            Content chunks
        """
        logger.info(f"[{self.agent_name}] Starting stream request to {model}")
        
        params: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            stream=True,
        )
        # o-series reasoning models use max_completion_tokens, not max_tokens
        if _model_supports_temperature(model):
            params["temperature"] = temperature
            params["max_tokens"] = max_tokens
        else:
            params["max_completion_tokens"] = max_tokens
        stream = await self.client.chat.completions.create(**params)
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat_with_vision(
        self,
        model: str,
        text_prompt: str,
        images: List[str],  # Base64 encoded images
        temperature: float = 0.7,
        max_tokens: int = 4096,
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
        # Build multimodal content
        content = []
        
        # Add images first
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        # Create log record (without the actual image data to save space)
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=[{"role": "user", "content": f"[{len(images)} images] + {text_prompt[:500]}"}],
            interaction_type="vision",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        start_time = time.time()
        
        async def _make_request():
            params: Dict[str, Any] = dict(
                model=model,
                messages=messages,
            )
            # o-series reasoning models use max_completion_tokens, not max_tokens
            if _model_supports_temperature(model):
                params["temperature"] = temperature
                params["max_tokens"] = max_tokens
            else:
                params["max_completion_tokens"] = max_tokens
            return await self.client.chat.completions.create(**params)

        try:
            logger.info(f"[{self.agent_name}] Sending vision request to {model} with {len(images)} images")

            # Use retry mechanism for the API call
            retry_kwargs: Dict[str, Any] = {"agent_name": self.agent_name}
            if request_timeout is not None:
                retry_kwargs["request_timeout"] = request_timeout
            response = await retry_with_backoff(
                _make_request,
                **retry_kwargs
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
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response and response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response and response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response and response.usage else 0
                },
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
            logger.error(f"[{self.agent_name}] Vision error: {e}")
            
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
    
    def __init__(self, api_key: str, proxy_url: Optional[str] = None):
        self.api_key = api_key
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
            "agent": "deep-research-pro-preview-12-2025",
            "background": background
        }

        response = await self.client.post(
            url,
            params={"key": self.api_key},
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
                params={"key": self.api_key}
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
        agent_name: str = "unknown"
    ):
        self.llm = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            proxy_url=proxy_url,
            session_id=session_id,
            agent_name=agent_name
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
