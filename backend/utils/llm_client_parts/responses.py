# Normalized response types returned to callers regardless of upstream protocol.
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


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any
    citations: Optional[List[Dict[str, str]]] = None  # Responses API url_citation; None for chat/anthropic
    status: Optional[str] = None
    incomplete_details: Any = None
    error: Any = None

@dataclass
class ToolCall:
    """Represents a tool/function call from the model.

    parse_error / raw_arguments are populated when the JSON arguments
    string returned by the model cannot be decoded. Callers should
    check parse_error before dispatching the tool.
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    parse_error: Optional[str] = None
    raw_arguments: Optional[str] = None

def _tool_calls_to_raw_assistant(text: Optional[str], tool_calls: List[ToolCall]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
            for tc in tool_calls
        ] or None,
    }

@dataclass
class LLMToolResponse:
    """Response from LLM with tool calls"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    model: str
    usage: Dict[str, int]
    raw_response: Any
    raw_assistant_message: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    incomplete_details: Any = None
    error: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
