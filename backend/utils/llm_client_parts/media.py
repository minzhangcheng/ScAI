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


class VisionPayloadError(ValueError):
    """The local image value cannot be serialized into a vision request."""


_DATA_IMAGE_RE = re.compile(
    r"^data:(image/[a-z0-9.+-]+);base64,(.*)$",
    re.IGNORECASE | re.DOTALL,
)


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


def normalize_image_input(value: str) -> Tuple[str, str]:
    """Return ``(raw_base64, mime_type)`` for raw base64 or a data URI.

    Report helpers intentionally produce data URIs while the LLM adapters expect
    raw base64. Normalizing at this boundary prevents a second ``data:`` prefix
    and keeps every protocol adapter on the same byte payload.
    """
    text = str(value or "").strip()
    declared_mime = ""
    match = _DATA_IMAGE_RE.match(text)
    if match:
        declared_mime = match.group(1).lower()
        text = match.group(2)
    raw = "".join(text.split())
    if not raw:
        raise VisionPayloadError("vision image payload is empty")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise VisionPayloadError("vision image payload is not valid base64") from exc
    if not decoded:
        raise VisionPayloadError("vision image payload decoded to no bytes")
    detected_mime = detect_image_mime(decoded[:12])
    mime = detected_mime or declared_mime or "image/png"
    return raw, mime
