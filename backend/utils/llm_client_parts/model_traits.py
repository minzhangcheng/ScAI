# Model-name classification: reasoning models, deep research, temperature support.
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

# Deep-research 内置检索工具候选名（不同服务商/模型命名不一致）。
_WEB_SEARCH_TOOL_NAMES = ["web_search_preview", "web_search"]

def _is_deep_research_model(model: str) -> bool:
    """True for deep-research models. Normalizes by removing hyphens so both
    'o3-deep-research' and 'o4-deepresearch-mini' match 'deepresearch'."""
    model_name = _extract_model_name(model).lower().replace("-", "")
    return "deepresearch" in model_name

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
