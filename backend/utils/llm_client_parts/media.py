# Image MIME sniffing and the English-output enforcement preamble.
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

def detect_image_mime(header: bytes) -> str:
    """Identify image MIME type from leading magic bytes; default image/png.

    WebP lives in a RIFF container shared with WAV/AVI, so the RIFF prefix
    alone is ambiguous — the 'WEBP' fourcc at bytes 8:12 must also match.
    """
    if header[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if header[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return "image/webp"
    return "image/png"

def _detect_image_mime_type(img_b64: str) -> str:
    """Detect image MIME type from base64-encoded data using magic bytes.

    Falls back to 'image/png' if the format cannot be determined.
    """
    try:
        # 16 base64 chars → 12 bytes, enough to read the WebP fourcc at [8:12]
        header = base64.b64decode(img_b64[:16])
        return detect_image_mime(header)
    except Exception as e:
        logger.debug(f"Image MIME type detection failed, defaulting to image/png: {e}")
    return "image/png"
