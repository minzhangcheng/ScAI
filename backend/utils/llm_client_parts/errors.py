# Upstream-exhaustion and empty-response error types.
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


class UpstreamExhaustedError(RuntimeError):
    """Raised when retries are exhausted against a flaky upstream (5xx/timeouts).

    Carries context so callers can present meaningful intervention dialogs
    without reverse-looking up agent configuration.
    """
    def __init__(
        self,
        message: str,
        agent_name: str = "unknown",
        model: str = "",
        llm_server: str = "",
        failure_kind: str = "upstream_exhausted",
    ):
        super().__init__(message)
        self.agent_name = agent_name
        self.model = model
        self.llm_server = llm_server
        self.failure_kind = failure_kind


class ToolProtocolError(UpstreamExhaustedError):
    """A required tool-call contract was refused after bounded retries."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, failure_kind="tool_protocol_failure", **kwargs)

class EmptyResponseError(RuntimeError):
    """A zero-token empty answer from the provider, raised on the protocol-probe
    path where the probe loop owns retry policy. Deliberately NOT a subclass of
    UpstreamExhaustedError: is_transient_probe_error() treats UpstreamExhaustedError
    as a wrong-protocol signal (switch protocol), whereas this must be classified
    transient so the probe retries the SAME protocol. Probe exhaustion then
    surfaces as UpstreamExhaustedError from _probe_in_order."""


class StreamInterruptedError(RuntimeError):
    """A stream failed after its first event, so replay is unsafe."""


class StreamNotStartedError(RuntimeError):
    """A stream ended cleanly before yielding its first event."""


class ReasoningOnlyStreamError(RuntimeError):
    """A completed Chat stream contained private reasoning but no public text."""

    def __init__(self, message: str, usage=None):
        super().__init__(message)
        self.usage = usage or {}
