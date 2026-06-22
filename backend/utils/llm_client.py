"""
LLM client utilities using OpenAI SDK for OpenAI-compatible APIs
Supports OpenRouter, OpenAI, and other compatible endpoints
"""
import base64
import html
import httpx
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


def _recover_stringified_tool_call(
    content: Optional[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Attempt to recover a tool call that an upstream gateway stringified into content.

    Some gateways (notably litellm proxying n1n.ai for Claude / Gemini) lose
    the structured tool_calls field and put the tool_use JSON into
    message.content as a string. We recognize three shapes:

      - {"tool": NAME, "parameters": {...}}                  (n1n.ai stringified)
      - {"name": NAME, "arguments": {...}}                   (OpenAI stringified)
      - {"type": "tool_use", "name": NAME, "input": {...}}   (Anthropic native)

    The payload may be wrapped in a ```json ... ``` markdown fence. Returns
    (name, arguments_dict) on success or None if content does not look like
    a stringified tool call.
    """
    if not content:
        return None
    s = content.strip()
    if not s:
        return None

    # Strip a single ```json ... ``` (or plain ``` ... ```) fence if present
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline == -1:
            return None
        s = s[first_newline + 1 :]
        if s.rstrip().endswith("```"):
            s = s[:-3]
        s = s.strip()

    if not s.startswith("{"):
        return None

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    name = obj.get("tool") or obj.get("name")
    args = obj.get("parameters")
    if args is None:
        args = obj.get("arguments")
    if args is None:
        args = obj.get("input")

    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        return None

    return name, args


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
EMPTY_CONTENT_PLACEHOLDER = "[Error: LLM generated a response but content extraction failed. Please retry.]"


def compute_rate_limit_delay(rate_limit_attempt: int, max_retries: int) -> Optional[float]:
    """Delay (seconds) for the given 0-based 429 retry attempt, or None when the
    dedicated rate-limit budget is exhausted.

    The 429 budget is tracked separately from other error types so a burst of
    5xx / timeout / JSON errors cannot shift its index or consume its budget.
    """
    budget = min(max_retries, len(RATE_LIMIT_DELAYS))
    if rate_limit_attempt < budget:
        return float(RATE_LIMIT_DELAYS[rate_limit_attempt])
    return None


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
    ):
        super().__init__(message)
        self.agent_name = agent_name
        self.model = model
        self.llm_server = llm_server

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


# 两级协议探测缓存（模块级，跨 client 实例 / 跨 agent 共享，因协议只由 base_url+model 决定）。
# 第一级：整站无 Responses 端点（base_url 级，与 model 无关）。
_NO_RESPONSES_ENDPOINT: set = set()
# 第二级：模型级协议选择。value ∈ {"responses","chat","anthropic"}
_PROBE_CACHE: Dict[Tuple[str, str], str] = {}
_PROBE_LOCKS: Dict[Tuple[str, str], asyncio.Lock] = {}


def _url_path_suffix_protocol(base_url: str):
    """Explicit endpoint override by URL path tail. Returns protocol or None.
    Matches the path SUFFIX only (not arbitrary substring) to avoid false
    positives like a domain literally containing 'responses'."""
    path = urlparse(base_url).path.rstrip("/").lower()
    if path.endswith("/responses"):
        return "responses"
    if path.endswith("/messages"):
        return "anthropic"
    if path.endswith("/chat/completions"):
        return "chat"
    return None


# Canonical probe order when nothing else disambiguates. responses first
# (richer: citations + built-in web_search), then chat, then anthropic.
_DEFAULT_PROTOCOL_ORDER = ("responses", "chat", "anthropic")

# Per-protocol transient-retry waits (seconds) BEFORE switching to the next
# protocol in a probe. len() == attempts per protocol on transient errors
# (429/timeout/5xx). Deterministic "wrong protocol" errors skip these.
PROBE_RETRY_DELAYS = (2.0, 15.0, 10.0)


def _keyword_protocol_hints(base_url: str) -> List[str]:
    """Soft protocol hints from substring keywords in base_url, highest priority
    first. These only REORDER the probe sequence (never hard-pin a protocol —
    that is the job of an exact path suffix, see _url_path_suffix_protocol).
    Matching is loose on purpose: a wrong hint costs at most one failed probe
    attempt before moving on."""
    low = base_url.lower()
    hints: List[str] = []
    # anthropic is wire-incompatible with the OpenAI family, so its markers are
    # the strongest hint (covers api.anthropic.com/v1, missed by '/v1 absent').
    if "anthropic" in low or "/messages" in low:
        hints.append("anthropic")
    if "responses" in low:
        hints.append("responses")
    if "chat" in low or "completion" in low:
        hints.append("chat")
    return hints


def _route_protocol_order(base_url: str, model: str) -> List[str]:
    """Ordered list of protocols to try for (base_url, model).

    Priority:
      0. explicit endpoint URL suffix  -> single hard-pinned protocol
      1. cached probe decision         -> single decided protocol
      2. deep-research model           -> ['responses', 'chat']
      3. base order, reordered by keyword hints; '/v1' absence floats anthropic
         to the front (legacy heuristic); a site known to lack /responses drops
         'responses' from the order.
    """
    explicit = _url_path_suffix_protocol(base_url)
    if explicit is not None:
        return [explicit]
    cached = _PROBE_CACHE.get((base_url, model))
    if cached is not None:
        return [cached]
    if _is_deep_research_model(model):
        return ["responses", "chat"]

    order = list(_DEFAULT_PROTOCOL_ORDER)
    if "/v1" not in base_url:
        order = ["anthropic", "responses", "chat"]
    for proto in reversed(_keyword_protocol_hints(base_url)):
        if proto in order:
            order.remove(proto)
            order.insert(0, proto)
    if base_url in _NO_RESPONSES_ENDPOINT and "responses" in order:
        order.remove("responses")
    return order


def _route_protocol(base_url: str, model: str) -> str:
    """Primary protocol for (base_url, model) — head of the probe order. Retained
    for streaming dispatch and callers/tests that need a single value."""
    return _route_protocol_order(base_url, model)[0]


def _strip_path_suffix(base_url: str, suffixes: tuple) -> str:
    """Strip the first matching endpoint suffix from a URL's path (case-insensitive),
    preserving scheme/host/query. Used so a user-supplied full endpoint URL is not
    double-appended by the SDK."""
    p = urlparse(base_url)
    path = p.path.rstrip("/")
    low = path.lower()
    for suf in suffixes:
        if low.endswith(suf):
            path = path[: -len(suf)]
            break
    return urlunparse(p._replace(path=path))


def _openai_sdk_base_url(base_url: str) -> str:
    """Normalize an OpenAI-family base_url for AsyncOpenAI: strip a trailing
    /chat/completions or /responses so the SDK's own path append yields exactly
    the user's endpoint."""
    return _strip_path_suffix(base_url, ("/chat/completions", "/responses"))


def _anthropic_sdk_base_url(base_url: str) -> str:
    """Normalize a base_url for AsyncAnthropic: strip a trailing /v1/messages or
    /messages so the SDK's own /v1/messages append is not doubled."""
    return _strip_path_suffix(base_url, ("/v1/messages", "/messages"))


def _responses_messages_to_input(messages: List[Dict[str, Any]]):
    """Translate OpenAI chat messages to Responses (instructions, input[]).

    The first system message becomes top-level `instructions`. Chat-completions
    tool history is converted to Responses input item types because Responses
    rejects raw assistant.tool_calls and role=tool messages in input[].
    """
    instructions = None
    items: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and instructions is None:
            instructions = msg.get("content")
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content:
                items.append({"role": "assistant", "content": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                })
            continue

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id"),
                "output": msg.get("content") or "",
            })
            continue

        items.append(msg)
    return instructions, items


def _responses_translate_tools(tools: Optional[List[Dict[str, Any]]]):
    """Chat function-tool format -> Responses flat function format. Built-in
    tools (e.g. {'type':'web_search'}) pass through unchanged."""
    if not tools:
        return tools
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
            flat = {"type": "function", "name": fn.get("name")}
            if "description" in fn:
                flat["description"] = fn["description"]
            if "parameters" in fn:
                flat["parameters"] = fn["parameters"]
            out.append(flat)
        else:
            out.append(t)
    return out


def _responses_translate_tool_choice(tool_choice):
    """auto/none/required pass through; specific-function dict is flattened."""
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "function" in tool_choice:
        return {"type": "function", "name": tool_choice["function"].get("name")}
    return tool_choice


def _anthropic_messages_translate(messages: List[Dict[str, Any]]):
    """Translate canonical OpenAI-chat messages -> (system, anthropic_messages).
    - first system message -> top-level `system`
    - assistant.tool_calls -> tool_use content blocks
    - role 'tool' -> user message with a tool_result block"""
    system = None
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and system is None:
            system = msg.get("content")
            continue
        if role == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": msg.get("content"),
            }]})
            continue
        if role == "assistant" and msg.get("tool_calls"):
            blocks = []
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"):
                        blocks.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        blocks.append(block)
                    elif isinstance(block, str) and block:
                        blocks.append({"type": "text", "text": block})
            elif content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id"),
                               "name": fn.get("name"), "input": args})
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": role, "content": msg.get("content")})
    return system, out


def _anthropic_translate_tools(tools: Optional[List[Dict[str, Any]]]):
    if not tools:
        return tools
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            fn = t["function"]
            spec = {"name": fn.get("name"), "input_schema": fn.get("parameters", {"type": "object"})}
            if "description" in fn:
                spec["description"] = fn["description"]
            out.append(spec)
    return out


def _anthropic_translate_tool_choice(tool_choice):
    """auto->{type:auto}; required->{type:any}; specific function->{type:tool,name};
    none-> None (caller omits tools entirely, since Anthropic has no 'none')."""
    if tool_choice in (None, "auto"):
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "tool", "name": tool_choice["function"].get("name")}
    return {"type": "auto"}


def _parse_anthropic_dump(dump: Dict[str, Any]):
    """Parse an Anthropic Messages response (model_dump dict) into
    (text, tool_calls, thinking_blocks, usage). thinking_blocks is None when
    absent; when present it is a list of {'type':'thinking','thinking':...,
    'signature':...} ready to be echoed verbatim on the next tool-call turn."""
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    thinking_blocks: List[Dict[str, Any]] = []

    for block in dump.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif btype == "thinking":
            tb = {"type": "thinking", "thinking": block.get("thinking", "")}
            if block.get("signature"):
                tb["signature"] = block["signature"]
            thinking_blocks.append(tb)
        elif btype == "tool_use":
            tool_calls.append(ToolCall(
                id=block.get("id") or f"anthropic_{uuid4().hex[:8]}",
                name=block.get("name", ""),
                arguments=block.get("input") or {},
            ))

    u = dump.get("usage") or {}
    inp = u.get("input_tokens", 0) or 0
    outp = u.get("output_tokens", 0) or 0
    usage = {"prompt_tokens": inp, "completion_tokens": outp, "total_tokens": inp + outp}

    return "\n".join(text_parts), tool_calls, (thinking_blocks or None), usage


def _anthropic_restore_thinking(messages: List[Dict[str, Any]]):
    """Inbound translation helper: if an assistant message carries thinking
    blocks under 'reasoning_details', prepend them (with signature) back into
    its content so Anthropic accepts the multi-turn tool-call replay. Returns a
    new list; does not mutate inputs. Operates BEFORE _anthropic_messages_translate."""
    out = []
    for msg in messages:
        rd = msg.get("reasoning_details") if isinstance(msg, dict) else None
        if msg.get("role") == "assistant" and rd:
            new_msg = {k: v for k, v in msg.items() if k != "reasoning_details"}
            content = new_msg.get("content")
            blocks = list(rd)
            if isinstance(content, list):
                blocks = blocks + content
            elif content:
                blocks = blocks + [{"type": "text", "text": content}]
            new_msg["content"] = blocks
            out.append(new_msg)
        else:
            out.append(msg)
    return out


def _parse_responses_dump(dump: Dict[str, Any]):
    """Parse a Responses API response (already model_dump()'d to a dict) into
    (text, tool_calls, citations, usage). citations is None when no
    url_citation annotations are present."""
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    citations: List[Dict[str, str]] = []
    seen_uris = set()

    for item in dump.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for block in item.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("output_text", "text") and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                for ann in block.get("annotations", []) or []:
                    if isinstance(ann, dict) and ann.get("type") == "url_citation":
                        uri = ann.get("url", "")
                        if uri and uri not in seen_uris:
                            seen_uris.add(uri)
                            citations.append({"title": ann.get("title", ""), "url": uri})
        elif itype == "function_call":
            raw_args = item.get("arguments") or "{}"
            parse_error = None
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                args = {}
                parse_error = str(e)
            tool_calls.append(ToolCall(
                id=item.get("call_id") or item.get("id") or f"resp_{uuid4().hex[:8]}",
                name=item.get("name", ""),
                arguments=args,
                parse_error=parse_error,
                raw_arguments=raw_args if parse_error else None,
            ))

    u = dump.get("usage") or {}
    usage = {
        "prompt_tokens": u.get("input_tokens", 0) or 0,
        "completion_tokens": u.get("output_tokens", 0) or 0,
        "total_tokens": u.get("total_tokens", 0) or 0,
    }
    details = u.get("input_tokens_details") or {}
    if details.get("cached_tokens"):
        usage["cached_tokens"] = details["cached_tokens"]

    return "\n".join(text_parts), tool_calls, (citations or None), usage


def _inject_web_search_if_needed(model: str, tools: list) -> list:
    """Auto-inject a built-in web search tool for deep-research models that
    require one. No-op for non-DR models or when a search-tool variant is
    already present. Preserves caller-provided tools."""
    if not _is_deep_research_model(model):
        return tools
    existing = {t.get("type") for t in (tools or []) if isinstance(t, dict)}
    for name in _WEB_SEARCH_TOOL_NAMES:
        if name in existing:
            return tools
    return list(tools or []) + [{"type": _WEB_SEARCH_TOOL_NAMES[0]}]


def _is_unknown_web_search_tool_error(error: Exception) -> bool:
    """True when the error indicates an unrecognized tool type that is
    one of the _WEB_SEARCH_TOOL_NAMES variants."""
    s = str(error).lower()
    if not any(kw in s for kw in (
        "unknown tool", "unsupported tool", "not a valid tool",
        "tool not found", "unrecognized tool",
    )):
        return False
    for name in _WEB_SEARCH_TOOL_NAMES:
        if name.lower() in s:
            return True
    return False


def _swap_web_search_tool_name(tools: list, from_name: str, to_name: str) -> list:
    """Return a new list with from_name replaced by to_name in dict type fields."""
    return [
        {**t, "type": to_name} if isinstance(t, dict) and t.get("type") == from_name else t
        for t in tools
    ]


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


def is_tool_choice_error(error: Exception) -> bool:
    """Check if a BadRequestError is related to the tool_choice parameter.

    Some OpenAI-compatible providers reject `tool_choice="required"` (or
    a specific-function dict) with a BadRequestError mentioning the
    parameter by name. Match conservatively: only the explicit
    `tool_choice` / `tool choice` token, not the unrelated legacy
    `function_call` surface.
    """
    if not isinstance(error, BadRequestError):
        return False
    error_str = str(error).lower()
    return "tool_choice" in error_str or "tool choice" in error_str


def is_responses_endpoint_absent(error: Exception) -> bool:
    """True when the Responses endpoint itself is absent on this server.

    This is a base_url-level property. A True result may populate
    _NO_RESPONSES_ENDPOINT because every model on this base URL would fail the
    Responses route.
    """
    if isinstance(error, NotFoundError):
        return True
    s = str(error).lower()
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            s = " ".join([
                s,
                str(err.get("message", "")).lower(),
                str(err.get("code", "")).lower(),
                str(err.get("type", "")).lower(),
            ])
    if "not implemented" in s and ("responses" in s or "local:convert_request_failed" in s):
        return True
    route_markers = (
        "cannot post",
        "unknown request url",
        "no route",
        "404 page not found",
        "405 method not allowed",
    )
    return any(kw in s for kw in route_markers)


def is_responses_chat_only_model(error: Exception) -> bool:
    """True when Responses rejects this specific model and tells us to use
    chat/completions. This is model-level and must not mark the whole base_url
    as missing the Responses endpoint."""
    return "only supported in v1/chat/completions" in str(error).lower()


def is_endpoint_absent(error: Exception) -> bool:
    """Compatibility wrapper for callers/tests that only need to know whether
    Responses should fall back to chat for this request."""
    return is_responses_endpoint_absent(error) or is_responses_chat_only_model(error)


def is_responses_only_model(error: Exception) -> bool:
    """True when a chat/completions error indicates the model is only served
    via the Responses API (reactive chat -> responses fallback)."""
    return "only supported in v1/responses" in str(error).lower()


def is_transient_probe_error(error: Exception) -> bool:
    """True when an error is a transient upstream blip (retry the SAME protocol),
    as opposed to a deterministic 'wrong protocol / endpoint' signal (switch
    protocol immediately).

    Deterministic protocol signals (endpoint-absent, chat-only, responses-only)
    take precedence and return False even though some arrive as 5xx — those mean
    the protocol is wrong, not that the server hiccuped.
    """
    if (is_responses_endpoint_absent(error)
            or is_responses_chat_only_model(error)
            or is_responses_only_model(error)):
        return False
    return isinstance(error, (
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        asyncio.TimeoutError,
    ))


# Hard-quota markers: a 429 carrying any of these never recovers by waiting
# (monthly budget / billing exhausted), so we surface it immediately.
_HARD_QUOTA_KEYWORDS = (
    "monthly max",
    "auto reload",
    "insufficient_quota",
    "exceeded your current quota",
    "billing hard limit",
)
# Transient-limiter markers: per-window rate limits DO recover by waiting, so a
# message carrying any of these is never treated as a hard quota even if it also
# contains the word "quota" (Google's wording does).
_TRANSIENT_RATE_MARKERS = (
    "per minute",
    "per second",
    "per day",
    "per region",
    "requests per",
    "tokens per",
)


def _is_hard_quota_error(error: Exception) -> bool:
    """True when a 429 is a non-recoverable quota exhaustion (not transient TPM).

    Prefers the SDK error code (OpenAI sets ``insufficient_quota`` for hard
    caps); falls back to narrowed keyword matching that explicitly excludes
    per-window limiter text such as Google's "per minute per region".
    """
    if getattr(error, "code", None) == "insufficient_quota":
        return True
    s = str(error).lower()
    if any(m in s for m in _TRANSIENT_RATE_MARKERS):
        return False
    return any(k in s for k in _HARD_QUOTA_KEYWORDS)


async def retry_with_backoff(
    func,
    max_retries: int = MAX_RETRIES,
    retry_delay_base: float = RETRY_DELAY_BASE,
    agent_name: str = "unknown",
    model_name: str = "",
    llm_server: str = "",
    request_timeout: float = 300.0,
    on_retry: Optional[Callable[[int, int, float, str], Awaitable[None]]] = None,
    non_retryable_predicate: Optional[Callable[[Exception], bool]] = None,
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
    rate_limit_attempt = 0  # independent 429 counter (see compute_rate_limit_delay)

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
            if non_retryable_predicate is not None and non_retryable_predicate(e):
                raise
            # JSON errors get fewer retries — repeated bad JSON is likely persistent.
            # MAX_JSON_RETRIES is the retry budget; <= keeps the actual retry count
            # equal to MAX_JSON_RETRIES (total invocations == MAX_JSON_RETRIES + 1).
            json_error_count += 1
            last_exception = e
            if json_error_count <= MAX_JSON_RETRIES:
                delay = retry_delay_base * json_error_count
                logger.warning(
                    f"[{agent_name}] Invalid JSON response (retry {json_error_count}/{MAX_JSON_RETRIES}): {str(e)[:100]}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await _notify_retry(attempt, delay, "JSONDecodeError")
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{agent_name}] Invalid JSON after {json_error_count} attempts: {str(e)}"
                )
                raise
        except (AuthenticationError, PermissionDeniedError) as e:
            # Non-transient: retrying with the same credentials cannot help.
            # Surface immediately; @with_retry / planning turn this into a HITL
            # gate where the user can fix the key and click Retry.
            last_exception = e
            logger.error(
                f"[{agent_name}] Auth/permission error (not retrying): "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            break
        except RateLimitError as e:
            if non_retryable_predicate is not None and non_retryable_predicate(e):
                raise
            last_exception = e
            if _is_hard_quota_error(e):
                logger.error(
                    f"[{agent_name}] Hard quota exhausted (not retrying): {str(e)[:200]}"
                )
                break
            delay = compute_rate_limit_delay(rate_limit_attempt, max_retries)
            if delay is not None:
                budget = min(max_retries, len(RATE_LIMIT_DELAYS))
                logger.warning(
                    f"[{agent_name}] Rate limited (429 retry {rate_limit_attempt + 1}/{budget}). "
                    f"Retrying in {delay}s..."
                )
                await _notify_retry(attempt, delay, "RateLimitError")
                await asyncio.sleep(delay)
                rate_limit_attempt += 1
            else:
                budget = min(max_retries, len(RATE_LIMIT_DELAYS))
                logger.error(f"[{agent_name}] Rate limit exceeded after {budget} retries")
                break  # stop spinning; fall through to UpstreamExhaustedError
        except RETRYABLE_EXCEPTIONS as e:
            if non_retryable_predicate is not None and non_retryable_predicate(e):
                raise
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
        f"Last error: {type(last_exception).__name__}: {last_exception}",
        agent_name=agent_name,
        model=model_name,
        llm_server=llm_server,
    ) from last_exception


def _collect_reasoning_parts(getter) -> list:
    """Collect non-empty reasoning fields from a message-like object.

    ``getter`` is a callable taking a field name and returning the value
    (e.g. ``msg_data.get`` for dicts or ``lambda k: getattr(obj, k, None)``).
    Handles str / dict / list element types robustly and concatenates all
    non-empty fields rather than returning only the first match.
    """
    parts: list = []
    for field in ("reasoning", "reasoning_content", "thinking", "thought"):
        val = getter(field)
        if not val:
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for elem in val:
                if isinstance(elem, str):
                    if elem:
                        parts.append(elem)
                elif isinstance(elem, dict):
                    text = elem.get("text") or elem.get("content") or ""
                    if text:
                        parts.append(str(text))
                else:
                    s = str(elem)
                    if s:
                        parts.append(s)
        elif isinstance(val, dict):
            text = val.get("text") or val.get("content") or ""
            if text:
                parts.append(str(text))
        else:
            s = str(val)
            if s:
                parts.append(s)
    return [p for p in parts if p]


# Reasoning fields a thinking-mode provider may attach to an assistant turn.
# These must be echoed back VERBATIM (original key + original structure, be it
# a plain string or a signature-carrying array) on tool-call turns. See:
#   - https://api-docs.deepseek.com/guides/thinking_mode
#   - https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/passing-back-reasoning_content
_REASONING_ECHO_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking", "thought")


def _reasoning_echo_allowed(model: str) -> bool:
    """Whether reasoning fields may be echoed back for this model.

    DeepSeek R1 (deepseek-reasoner) rejects multi-turn requests that include
    reasoning_content with HTTP 400, so it must never be echoed. Every other
    known reasoning provider either requires the echo (DeepSeek v4, MiMo,
    Anthropic/OpenRouter thinking) or ignores unknown fields — so default True.

    Detection uses substring matching on the model name:
    - "reasoner" catches the deepseek-reasoner endpoint name
    - "-r1" catches deepseek-r1 variants (note: could also match *-r10 etc.,
      but no such model exists today)
    - "deepseek-r1" is a redundant catch for the canonical name

    DeepSeek v4 / v4-pro do NOT contain any of these substrings, so they
    correctly return True (echo is required for their tool-call turns).
    """
    m = (model or "").lower()
    if "reasoner" in m or "-r1" in m or "deepseek-r1" in m:
        return False
    return True


def _build_assistant_echo(msg_dict: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Build the assistant message to replay into multi-turn history.

    Reasoning fields are preserved verbatim ONLY when (a) the model permits it
    and (b) the turn contains tool_calls — matching the official contract that
    scopes the echo requirement to tool-call turns, not by recency.
    """
    out: Dict[str, Any] = {
        "role": "assistant",
        "content": msg_dict.get("content"),
        "tool_calls": msg_dict.get("tool_calls"),
    }
    if out.get("tool_calls") and _reasoning_echo_allowed(model):
        for key in _REASONING_ECHO_KEYS:
            val = msg_dict.get(key)
            if val:  # truthy guard drops "" / None / [] uniformly
                out[key] = val
    return out


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any
    citations: Optional[List[Dict[str, str]]] = None  # Responses API url_citation; None for chat/anthropic


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

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


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

    def __init__(self, client: "AsyncOpenAI", agent_name: str = "unknown"):
        self.client = client
        self.agent_name = agent_name

    def _build_params(self, model, messages, temperature, max_tokens, tools=None,
                      tool_choice=None, stream=False, timeout=None, skip_temperature=False) -> Dict[str, Any]:
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

    async def complete(self, model, messages, temperature, max_tokens, timeout=None, skip_temperature=False, tools=None) -> LLMResponse:
        params = self._build_params(
            model,
            messages,
            temperature,
            max_tokens,
            tools=tools,
            timeout=timeout,
            skip_temperature=skip_temperature,
        )
        resp = await self._create_responses_with_fallback(params)
        dump = resp.model_dump()
        text, _tools, citations, usage = _parse_responses_dump(dump)
        return LLMResponse(content=text, model=getattr(resp, "model", model),
                           usage=usage, raw_response=resp, citations=citations)

    async def complete_with_tools(self, model, messages, tools, tool_choice,
                                  temperature, max_tokens, skip_temperature=False) -> LLMToolResponse:
        params = self._build_params(model, messages, temperature, max_tokens,
                                    tools=tools, tool_choice=tool_choice, skip_temperature=skip_temperature)
        resp = await self._create_responses_with_fallback(params)
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
                # P3-4: honor previously discovered field override before falling
                # back to the static _is_o_series heuristic.
                override = self._max_tokens_field_overrides.get(model_name)
                if override:
                    field = override
                elif _is_o_series(model_name):
                    field = "max_completion_tokens"
                else:
                    field = "max_tokens"
                params[field] = max_tokens
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
            # P3-4: honor previously discovered field override before falling
            # back to the static _is_o_series heuristic. This lets the client
            # adapt to provider routes (e.g. openrouter/openai/o3-mini) that
            # the heuristic misclassifies.
            override = self._max_tokens_field_overrides.get(model_name)
            if override:
                field = override
            elif _is_o_series(model_name):
                field = "max_completion_tokens"
            else:
                field = "max_tokens"
            params[field] = max_tokens

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
        # If model is already known to reject temperature, skip directly.
        # P2-1: serialize read under the lock to avoid racing with a writer.
        self._ensure_cache_locks()
        async with self._temp_cache_lock:
            cached_no_temp = model_key in self._temp_unsupported_models
        if cached_no_temp:
            return await retry_with_backoff(make_request_no_temp, **retry_kwargs)
        try:
            return await retry_with_backoff(make_request, **retry_kwargs)
        except BadRequestError as e:
            if is_temperature_error(e):
                logger.warning(
                    f"[{self.agent_name}] Temperature rejected by {model_key}, "
                    f"retrying without temperature..."
                )
                async with self._temp_cache_lock:
                    if model_key not in self._temp_unsupported_models:
                        self._temp_unsupported_models.add(model_key)
                return await retry_with_backoff(make_request_no_temp, **retry_kwargs)
            else:
                raise

    def _build_retry_kwargs(self, model: str, request_timeout: Optional[float] = None, max_retries_override: Optional[int] = None) -> Dict[str, Any]:
        """统一构建 retry_kwargs，保持各方法逻辑一致."""
        kwargs: Dict[str, Any] = {
            "agent_name": self.agent_name,
            "model_name": model,
            "llm_server": self.base_url,
        }
        if request_timeout is not None:
            kwargs["request_timeout"] = request_timeout
        if max_retries_override is not None:
            kwargs["max_retries"] = max_retries_override
        elif request_timeout is not None and request_timeout > 300:
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
                    # Reasoning model fallback: content in reasoning/reasoning_content/thinking/thought
                    if not content:
                        parts = _collect_reasoning_parts(msg_data.get)
                        if parts:
                            content = "\n".join(parts)
            except Exception as e:
                logger.debug(f"[{self.agent_name}] model_dump extraction failed: {e}")

            # Fallback: check text attribute
            if not content and hasattr(message, "text"):
                content = message.text or ""
            # Fallback: check reasoning attribute directly on the object
            if not content:
                parts = _collect_reasoning_parts(lambda k: getattr(message, k, None))
                if parts:
                    content = "\n".join(parts)

        # Case 4: Other types — convert to string
        if not content and raw_content is not None:
            content = str(raw_content)

        # Check for refusal
        if not content and hasattr(message, "refusal") and message.refusal:
            content = f"[Refusal]: {message.refusal}"
            logger.warning(f"[{self.agent_name}] Request was refused: {message.refusal}")

        # Debug logging: tokens generated but content empty
        completion_tokens, has_tool_calls = self._log_empty_content_diagnostic(message, response)

        # If tokens were generated but content extraction failed, return an error
        # message instead of empty string so downstream agents can detect the failure.
        if not content and completion_tokens and completion_tokens > 0 and not has_tool_calls:
            logger.warning(
                f"[{self.agent_name}] Non-zero completion tokens ({completion_tokens}) "
                f"but empty extracted content — returning error placeholder"
            )
            return EMPTY_CONTENT_PLACEHOLDER

        return content

    def _log_empty_content_diagnostic(self, message, response) -> tuple:
        """Log diagnostic info when content extraction yielded empty string.

        Returns (completion_tokens, has_tool_calls) so the caller can decide
        whether to return an error placeholder.
        """
        completion_tokens = 0
        has_tool_calls = False
        if not response or not response.usage:
            return completion_tokens, has_tool_calls
        completion_tokens = response.usage.completion_tokens or 0
        if completion_tokens <= 0:
            return completion_tokens, has_tool_calls

        has_tool_calls = bool(getattr(message, "tool_calls", None))
        if has_tool_calls:
            return completion_tokens, has_tool_calls

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
        for attr_name in ("content", "refusal", "role", "reasoning", "reasoning_content", "thinking", "thought"):
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
        return completion_tokens, has_tool_calls

    def _get_responses_adapter(self) -> "_ResponsesAdapter":
        if getattr(self, "_responses_adapter", None) is None:
            self._responses_adapter = _ResponsesAdapter(self.client, agent_name=self.agent_name)
        return self._responses_adapter

    def _get_anthropic_adapter(self) -> "_AnthropicAdapter":
        if getattr(self, "_anthropic_adapter", None) is None:
            self._anthropic_adapter = _AnthropicAdapter(
                api_key=self.api_key, base_url=_anthropic_sdk_base_url(self.base_url),
                http_client=self.http_client, agent_name=self.agent_name)
        return self._anthropic_adapter

    @staticmethod
    def _adapter_temp_call_pair(
        call_factory: Callable[[bool], Awaitable[Any]]
    ) -> Tuple[Callable[[], Awaitable[Any]], Callable[[], Awaitable[Any]]]:
        return (lambda: call_factory(False), lambda: call_factory(True))

    async def _invoke_adapter(self, *, interaction_type, model, log_messages,
                              make_coro, make_coro_no_temp,
                              tools=None, tool_choice=None, temperature=None,
                              max_tokens=None, request_timeout=None,
                              non_retryable_predicate=None, max_retries_override=None):
        """Wrap an adapter call (Responses/Anthropic) with the same cross-cutting
        machinery the inline chat path uses: LLM logging + retry_with_backoff
        (via _call_with_temp_fallback, which also gives on_retry, temperature
        fallback, and UpstreamExhaustedError on exhaustion).

        make_coro / make_coro_no_temp are zero-arg callables returning fresh
        coroutines (with / without temperature)."""
        interaction = self.llm_logger.log_request(
            session_id=self.session_id,
            agent=self.agent_name,
            model=model,
            messages=log_messages,
            interaction_type=interaction_type,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=self.base_url,
        )
        start_time = time.time()
        model_key = _extract_model_name(model)
        retry_kwargs = self._build_retry_kwargs(
            model, request_timeout=request_timeout, max_retries_override=max_retries_override)
        if non_retryable_predicate is not None:
            retry_kwargs["non_retryable_predicate"] = non_retryable_predicate
        try:
            result = await self._call_with_temp_fallback(
                make_coro, make_coro_no_temp, model_key, retry_kwargs
            )
        except Exception as e:
            self.llm_logger.log_response(
                interaction=interaction, error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
            raise
        tool_calls_log = None
        tcs = getattr(result, "tool_calls", None)
        if tcs:
            tool_calls_log = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments,
                 "parse_error": tc.parse_error}
                for tc in tcs
            ]
        self.llm_logger.log_response(
            interaction=interaction,
            response_content=getattr(result, "content", None),
            tool_calls=tool_calls_log,
            usage=getattr(result, "usage", {}) or {},
            duration_ms=(time.time() - start_time) * 1000,
        )
        return result

    def _cache_protocol_decision(self, model: str, protocol: str) -> None:
        _PROBE_CACHE[(self.base_url, model)] = protocol

    def _note_probe_protocol_failure(self, model: str, protocol: str,
                                     error: Exception) -> Optional[str]:
        """Record side effects for a DETERMINISTIC failure of `protocol` and
        return the protocol to redirect to next, or None. Never writes a success
        cache (only success writes _PROBE_CACHE); only records the site-level
        _NO_RESPONSES_ENDPOINT fact for a truly absent responses endpoint."""
        if protocol == "responses":
            if is_responses_endpoint_absent(error):
                logger.warning(f"[{self.agent_name}] Responses endpoint absent at "
                               f"{self.base_url}; switching to chat")
                _NO_RESPONSES_ENDPOINT.add(self.base_url)
                return "chat"
            if is_responses_chat_only_model(error):
                logger.warning(f"[{self.agent_name}] model {model} chat-only on Responses; "
                               f"switching to chat")
                return "chat"
        if protocol == "chat":
            if is_responses_only_model(error):
                logger.warning(f"[{self.agent_name}] model {model} is responses-only; "
                               f"switching to responses")
                return "responses"
        return None

    async def _call_decided_protocol(self, protocol: str, call_for, model: str):
        """Call a single, already-decided protocol with FULL retry/backoff. On a
        deterministic 'wrong protocol' error, redirect once to the indicated
        protocol (also full-retry). Fixes the confirmed-responses no-retry bug."""
        factory = call_for(protocol, probe=False)
        if factory is None:
            raise ValueError(f"protocol {protocol} unsupported for this interaction")
        try:
            result = await factory()
        except Exception as e:
            redirect = self._note_probe_protocol_failure(model, protocol, e)
            if redirect is None:
                raise
            if _PROBE_CACHE.get((self.base_url, model)) == protocol:
                _PROBE_CACHE.pop((self.base_url, model), None)
            rfactory = call_for(redirect, probe=False)
            if rfactory is None:
                raise
            result = await rfactory()
            self._cache_protocol_decision(model, redirect)
            return result
        self._cache_protocol_decision(model, protocol)
        return result

    async def _probe_in_order(self, order: List[str], call_for, model: str):
        """Try protocols in `order`. Per protocol: up to len(PROBE_RETRY_DELAYS)
        single-shot attempts on TRANSIENT errors (sleeping PROBE_RETRY_DELAYS
        between/after each), then move on. A DETERMINISTIC 'wrong protocol' error
        switches immediately (and may inject a redirect protocol at the front).
        Only a SUCCESS writes the protocol cache."""
        key = (self.base_url, model)
        lock = _PROBE_LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            recheck = _route_protocol_order(self.base_url, model)
            if len(recheck) == 1:
                return await self._call_decided_protocol(recheck[0], call_for, model)

            pending = list(order)
            tried: set = set()
            last_error: Optional[Exception] = None
            while pending:
                protocol = pending.pop(0)
                if protocol in tried:
                    continue
                tried.add(protocol)
                factory = call_for(protocol, probe=True)
                if factory is None:
                    continue
                for attempt in range(len(PROBE_RETRY_DELAYS)):
                    try:
                        result = await factory()
                        self._cache_protocol_decision(model, protocol)
                        return result
                    except Exception as e:
                        last_error = e
                        if is_transient_probe_error(e):
                            await asyncio.sleep(PROBE_RETRY_DELAYS[attempt])
                            continue
                        redirect = self._note_probe_protocol_failure(model, protocol, e)
                        if redirect and redirect not in tried and redirect not in pending:
                            pending.insert(0, redirect)
                        break
            raise UpstreamExhaustedError(
                f"[{self.agent_name}] Protocol probe exhausted across {sorted(tried)} "
                f"for {model}. Last error: {type(last_error).__name__}: {last_error}",
                agent_name=self.agent_name, model=model, llm_server=self.base_url,
            ) from last_error

    async def _dispatch_non_streaming(
        self,
        *,
        model: str,
        interaction_type: str,
        log_messages: List[Dict[str, Any]],
        make_anthropic_adapter_call: Optional[Callable[[Any, bool], Awaitable[Any]]],
        make_responses_adapter_call: Callable[[Any, bool], Awaitable[Any]],
        make_inline_call: Callable[[], Awaitable[Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        make_probe_fallback_call: Optional[Callable[[], Awaitable[Any]]] = None,
    ):
        def invoke(adapter, adapter_call, max_retries_override=None,
                   non_retryable_predicate=None):
            if non_retryable_predicate is None:
                non_retryable_predicate = lambda e: not is_transient_probe_error(e)
            make_coro, make_coro_no_temp = self._adapter_temp_call_pair(
                lambda skip_temperature: adapter_call(adapter, skip_temperature)
            )
            return self._invoke_adapter(
                interaction_type=interaction_type, model=model, log_messages=log_messages,
                make_coro=make_coro, make_coro_no_temp=make_coro_no_temp,
                tools=tools, tool_choice=tool_choice, temperature=temperature,
                max_tokens=max_tokens, request_timeout=request_timeout,
                non_retryable_predicate=non_retryable_predicate,
                max_retries_override=max_retries_override)

        def call_for(protocol, *, probe):
            mr = 0 if probe else None
            # In probe mode, the probe loop owns ALL retry logic; the backoff
            # layer must not intervene.  In decided mode, only deterministic
            # wrong-protocol errors are non-retryable (they trigger redirect).
            np = (lambda e: True) if probe else (lambda e: not is_transient_probe_error(e))
            if protocol == "responses":
                return lambda: invoke(self._get_responses_adapter(),
                                      make_responses_adapter_call, max_retries_override=mr,
                                      non_retryable_predicate=np)
            if protocol == "anthropic":
                if make_anthropic_adapter_call is None:
                    return None
                return lambda: invoke(self._get_anthropic_adapter(),
                                      make_anthropic_adapter_call, max_retries_override=mr,
                                      non_retryable_predicate=np)
            if protocol == "chat":
                if probe:
                    return make_probe_fallback_call or make_inline_call
                return make_inline_call
            return None

        order = _route_protocol_order(self.base_url, model)
        if len(order) == 1:
            return await self._call_decided_protocol(order[0], call_for, model)
        return await self._probe_in_order(order, call_for, model)

    def _responses_probe_decision(self, model: str) -> Optional[str]:
        key = (self.base_url, model)
        if self.base_url in _NO_RESPONSES_ENDPOINT or _PROBE_CACHE.get(key) == "chat":
            return "chat"
        if _PROBE_CACHE.get(key) == "responses":
            return "responses"
        return None

    def _record_responses_probe_failure(self, model: str, error: Exception, stream: bool = False) -> Optional[str]:
        key = (self.base_url, model)
        stream_label = " stream" if stream else ""
        if is_responses_endpoint_absent(error):
            logger.warning(f"[{self.agent_name}] Responses{stream_label} endpoint absent at {self.base_url}; falling back to chat{stream_label}")
            _NO_RESPONSES_ENDPOINT.add(self.base_url)
            _PROBE_CACHE[key] = "chat"
            return "chat"
        if is_responses_chat_only_model(error):
            logger.warning(f"[{self.agent_name}] model {model} is chat/completions-only on Responses; falling back to chat{stream_label}")
            _PROBE_CACHE[key] = "chat"
            return "chat"
        return None

    async def _chat_inline(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> LLMResponse:
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
        retry_kwargs = self._build_retry_kwargs(model, request_timeout=timeout, max_retries_override=max_retries)

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
    ) -> LLMResponse:
        """
        Send chat request to OpenAI-compatible API with automatic retry on timeout
        and automatic protocol dispatch (chat / responses / anthropic).

        Args:
            model: Model identifier (e.g., "google/gemini-2.5-pro-preview-05-06")
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Unsupported; use chat_stream() for streaming responses
            timeout: Per-request timeout in seconds (overrides client default)
            tools: Optional Responses API tools for non-streaming Responses calls

        Returns:
            LLMResponse object
        """
        if stream:
            raise ValueError("chat(stream=True) is not supported; use chat_stream() for streaming responses")

        enforced = _inject_english_enforcement(messages)
        probe_timeout = timeout or 300.0
        return await self._dispatch_non_streaming(
            model=model,
            interaction_type="chat",
            log_messages=enforced,
            make_anthropic_adapter_call=lambda adapter, skip_temperature: adapter.complete(
                model=model, messages=enforced, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout, skip_temperature=skip_temperature),
            make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete(
                model=model, messages=enforced, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout, skip_temperature=skip_temperature,
                tools=tools),
            make_inline_call=lambda: self._chat_inline(
                model, enforced, temperature, max_tokens, False, timeout, max_retries),
            make_probe_fallback_call=lambda: self._chat_inline(
                model, enforced, temperature, max_tokens, False, probe_timeout, max_retries=0),
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout,
        )

    async def _chat_with_tools_inline(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> LLMToolResponse:
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

        # Pre-emptive demotion: if this model has previously rejected non-"auto"
        # tool_choice, skip the round-trip and demote now.
        effective_tool_choice = tool_choice
        if (
            effective_tool_choice not in (None, "auto", "none")
            and model_key in self._toolchoice_unsupported_models
        ):
            logger.warning(
                f"[{self.agent_name}] tool_choice={effective_tool_choice!r} demoted to 'auto' "
                f"for {model_key} (cached as tool-choice-unsupported)"
            )
            effective_tool_choice = "auto"

        base_params = {"model": model, "messages": messages, "tools": tools, "tool_choice": effective_tool_choice}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )
        retry_kwargs = self._build_retry_kwargs(model)
        if max_retries is not None:
            retry_kwargs["max_retries"] = max_retries

        try:
            logger.info(f"[{self.agent_name}] Sending chat_with_tools request to {model}")
            try:
                response = await self._call_with_temp_fallback(
                    _make_request, _make_request_no_temp, model_key, retry_kwargs
                )
            except BadRequestError as tc_err:
                if (
                    is_tool_choice_error(tc_err)
                    and effective_tool_choice not in (None, "auto", "none")
                ):
                    logger.warning(
                        f"[{self.agent_name}] tool_choice rejected by {model_key}, "
                        f"caching as unsupported and retrying with 'auto'"
                    )
                    self._ensure_cache_locks()
                    async with self._toolchoice_cache_lock:
                        if model_key not in self._toolchoice_unsupported_models:
                            self._toolchoice_unsupported_models.add(model_key)
                    base_params["tool_choice"] = "auto"
                    _make_request, _make_request_no_temp = self._build_request_pair(
                        base_params, model, temperature, max_tokens
                    )
                    response = await self._call_with_temp_fallback(
                        _make_request, _make_request_no_temp, model_key, retry_kwargs
                    )
                else:
                    raise

            message = None
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message

            tool_calls = []
            tool_calls_log = []
            if message and message.tool_calls:
                for tc in message.tool_calls:
                    parse_error: Optional[str] = None
                    raw_args: Optional[str] = None
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[{self.agent_name}] Tool call arguments JSON parse failed: {e}; "
                            f"function={tc.function.name}, arguments={tc.function.arguments!r}"
                        )
                        arguments = {}
                        parse_error = str(e)
                        raw_args = tc.function.arguments

                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                        parse_error=parse_error,
                        raw_arguments=raw_args,
                    ))
                    tool_calls_log.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                        "parse_error": parse_error,
                    })

            content = self._extract_content_from_message(message, response) if message else None

            raw_assistant_message = None
            if message:
                try:
                    raw_assistant_message = _build_assistant_echo(message.model_dump(), model)
                except Exception as e:
                    logger.debug(f"[{self.agent_name}] Failed to preserve raw assistant message: {e}")

            if not tool_calls and content:
                recovered = _recover_stringified_tool_call(content)
                if recovered is not None:
                    rec_name, rec_args = recovered
                    rec_id = f"recovered_{uuid4().hex[:8]}"
                    logger.warning(
                        f"[{self.agent_name}] Recovered stringified tool_call "
                        f"name={rec_name!r} from content at {self.base_url} "
                        f"(model={model}); upstream gateway lost tool_calls field"
                    )
                    tool_calls.append(ToolCall(
                        id=rec_id, name=rec_name, arguments=rec_args,
                    ))
                    tool_calls_log.append({
                        "id": rec_id, "name": rec_name,
                        "arguments": rec_args, "parse_error": None,
                        "recovered": True,
                    })
                    content = None
                    if raw_assistant_message is not None:
                        raw_assistant_message["content"] = None
                        raw_assistant_message["tool_calls"] = [{
                            "id": rec_id,
                            "function": {"name": rec_name, "arguments": json.dumps(rec_args)},
                            "type": "function",
                        }]

            result = LLMToolResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model if response else model,
                usage=self._extract_usage(response),
                raw_response=response,
                raw_assistant_message=raw_assistant_message
            )

            self.llm_logger.log_response(
                interaction=interaction,
                response_content=content,
                tool_calls=tool_calls_log if tool_calls_log else None,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except BadRequestError as e:
            logger.warning(
                f"[{self.agent_name}] Chat with tools BadRequestError"
                f" (may be handled upstream): {e}"
            )
            self.llm_logger.log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise
        except Exception as e:
            logger.exception(f"[{self.agent_name}] Chat with tools error")
            self.llm_logger.log_response(
                interaction=interaction,
                error=str(e),
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
        Send chat request with function/tool calling support and automatic retry on timeout.
        Auto-routes to Responses or Anthropic adapter when appropriate.

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
        enforced = _inject_english_enforcement(messages)
        return await self._dispatch_non_streaming(
            model=model,
            interaction_type="chat_with_tools",
            log_messages=enforced,
            make_anthropic_adapter_call=lambda adapter, skip_temperature: adapter.complete_with_tools(
                model=model, messages=enforced, tools=tools, tool_choice=tool_choice,
                temperature=temperature, max_tokens=max_tokens, skip_temperature=skip_temperature),
            make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete_with_tools(
                model=model, messages=enforced, tools=tools, tool_choice=tool_choice,
                temperature=temperature, max_tokens=max_tokens, skip_temperature=skip_temperature),
            make_inline_call=lambda: self._chat_with_tools_inline(
                model, enforced, tools, tool_choice, temperature, max_tokens),
            make_probe_fallback_call=lambda: self._chat_with_tools_inline(
                model, enforced, tools, tool_choice, temperature, max_tokens, max_retries=0),
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _chat_stream_inline(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        model_key = _extract_model_name(model)

        base_params = {"model": model, "messages": messages, "stream": True}
        _make_request, _make_request_no_temp = self._build_request_pair(
            base_params, model, temperature, max_tokens
        )

        stream_retry_kwargs = self._build_retry_kwargs(model, request_timeout=120.0)
        stream_retry_kwargs["max_retries"] = MAX_RETRIES

        stream = await self._call_with_temp_fallback(
            _make_request, _make_request_no_temp, model_key, stream_retry_kwargs
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _responses_or_fallback_stream(self, model, messages, temperature, max_tokens):
        """Try Responses streaming; endpoint-absent (raised before first chunk)
        -> cache + switch to inline chat stream. Per-(base_url, model) lock makes
        concurrent first calls probe once."""
        key = (self.base_url, model)
        lock = _PROBE_LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            if self._responses_probe_decision(model) == "chat":
                async for ch in self._chat_stream_inline(model, messages, temperature, max_tokens):
                    yield ch
                return
            adapter = self._get_responses_adapter()
            gen = adapter.stream(model=model, messages=messages,
                                 temperature=temperature, max_tokens=max_tokens)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                _PROBE_CACHE[key] = "responses"
                return
            except Exception as e:
                if self._record_responses_probe_failure(model, e, stream=True) == "chat":
                    async for ch in self._chat_stream_inline(model, messages, temperature, max_tokens):
                        yield ch
                    return
                raise
            _PROBE_CACHE[key] = "responses"
            yield first
        # Lock released here; remaining chunks are streamed without lock
        async for ch in gen:
            yield ch

    async def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat response from OpenAI-compatible API with auto-routing
        to Responses or Anthropic adapter when appropriate.

        Connection establishment is retried with backoff on transient errors.
        Mid-stream failures propagate to the caller (partial content already yielded).

        Yields:
            Content chunks
        """
        logger.info(f"[{self.agent_name}] Starting stream request to {model}")

        enforced = _inject_english_enforcement(messages)
        protocol = _route_protocol(self.base_url, model)
        if protocol == "anthropic":
            async for ch in self._get_anthropic_adapter().stream(
                    model=model, messages=enforced, temperature=temperature, max_tokens=max_tokens):
                yield ch
            return
        if protocol == "responses":
            async for ch in self._responses_or_fallback_stream(model, enforced, temperature, max_tokens):
                yield ch
            return
        try:
            stream = self._chat_stream_inline(model, enforced, temperature, max_tokens)
            first = await stream.__anext__()
        except StopAsyncIteration:
            return
        except BadRequestError as e:
            if is_responses_only_model(e):
                logger.warning(f"[{self.agent_name}] model {model} is responses-only; "
                               f"switching chat stream to Responses adapter")
                _PROBE_CACHE[(self.base_url, model)] = "responses"
                adapter = self._get_responses_adapter()
                async for ch in adapter.stream(model=model, messages=enforced,
                                               temperature=temperature, max_tokens=max_tokens):
                    yield ch
                return
            raise
        yield first
        async for ch in stream:
            yield ch

    async def _chat_with_vision_inline(
        self,
        model: str,
        text_prompt: str,
        images: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": ENGLISH_ENFORCEMENT},
            {"role": "user", "content": [{"type": "text", "text": text_prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:{_detect_image_mime_type(img_b64)};base64,{img_b64}"}}
                for img_b64 in images
            ]},
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
        retry_kwargs = self._build_retry_kwargs(model, request_timeout=request_timeout, max_retries_override=max_retries)

        try:
            logger.info(f"[{self.agent_name}] Sending vision request to {model} with {len(images)} images")
            response = await self._call_with_temp_fallback(
                _make_request, _make_request_no_temp, model_key, retry_kwargs
            )

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

            self.llm_logger.log_response(
                interaction=interaction,
                response_content=result.content,
                usage=result.usage,
                duration_ms=(time.time() - start_time) * 1000
            )

            return result

        except Exception as e:
            logger.exception(f"[{self.agent_name}] Vision error")

            self.llm_logger.log_response(
                interaction=interaction,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            raise

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
        Send chat request with images to OpenAI-compatible API with automatic retry on timeout.
        Auto-routes to Responses or Anthropic adapter when appropriate.

        Args:
            model: Model identifier
            text_prompt: Text prompt
            images: List of base64 encoded images
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            LLMResponse object
        """
        log_msgs = [{"role": "user", "content": f"[{len(images)} images] + {text_prompt[:500]}"}]
        probe_timeout = request_timeout or 300.0
        return await self._dispatch_non_streaming(
            model=model,
            interaction_type="vision",
            log_messages=log_msgs,
            make_anthropic_adapter_call=lambda adapter, skip_temperature: adapter.complete_vision(
                model=model, text_prompt=text_prompt, images=images,
                temperature=temperature, max_tokens=max_tokens,
                timeout=request_timeout, skip_temperature=skip_temperature),
            make_responses_adapter_call=lambda adapter, skip_temperature: adapter.complete_vision(
                model=model, text_prompt=text_prompt, images=images,
                temperature=temperature, max_tokens=max_tokens,
                timeout=request_timeout, skip_temperature=skip_temperature),
            make_inline_call=lambda: self._chat_with_vision_inline(
                model, text_prompt, images, temperature, max_tokens, request_timeout),
            make_probe_fallback_call=lambda: self._chat_with_vision_inline(
                model, text_prompt, images, temperature, max_tokens, probe_timeout, max_retries=0),
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )


class GoogleDeepResearchClient:
    """Client for Google Gemini Deep Research via Interactions API"""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    DEFAULT_MODEL = "deep-research-preview-04-2026"
    API_REVISION = "2026-05-20"
    GROUNDING_REDIRECT_MARKER = "grounding-api-redirect"

    def __init__(self, api_key: str, proxy_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self._default_headers = {
            "x-goog-api-key": self.api_key,
            "Api-Revision": self.API_REVISION,
        }
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

    def _check_response(self, response: httpx.Response, context: str) -> None:
        """Raise HTTPStatusError for 4xx/5xx responses with logging.

        Matches raise_for_status() behavior (raises on 4xx/5xx only, lets 3xx
        pass through) while adding structured warning logs for debugging.
        """
        if response.is_error:
            body = response.text
            logger.warning(
                f"[DeepResearch] {context} {response.status_code} from {response.url} "
                f"| response_body={body[:2000]}"
            )
            raise httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase} from {response.url}: {body[:1000]}",
                request=response.request,
                response=response,
            )

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
            "agent": self.model,
            "input": query,
            "background": background,
            "agent_config": {
                "type": "deep-research",
                "thinking_summaries": "auto",
                # "off" disables agent-generated charts/images. We only consume
                # the textual report, so suppressing inline images keeps the
                # response light and the parser simple.
                "visualization": "off",
                "collaborative_planning": False,
            },
        }

        response = await self.client.post(
            url,
            headers=self._default_headers,
            json=payload,
        )
        self._check_response(response, f"start_research model={self.model} api_revision={self.API_REVISION}")

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
            try:
                response = await self.client.get(
                    url,
                    headers=self._default_headers,
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                logger.warning(
                    f"[DeepResearch] Poll transient error, will retry: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                await asyncio.sleep(poll_interval)
                continue
            self._check_response(response, f"poll_research interaction_id={interaction_id}")
            
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
        """Extract research result from Interactions API response.

        Current schema (Api-Revision 2026-05-20, agent deep-research-preview-04-2026):
        the interaction holds a `steps[]` timeline. The final report lives in
        `model_output` steps as `content[]` blocks of `{"type": "text", "text",
        "annotations": [...]}`; planning summaries live in `thought` steps as a
        `summary[]` array. Citations are `url_citation` annotations on the text
        blocks. Legacy `output_text`/`outputs[]`/`output` paths are kept as
        defensive fallbacks for older revisions.
        """
        result = {
            "report": "",
            "sources": [],
            "thinking": []
        }
        seen_uris = set()

        def _add_source(uri: str, title: str = "") -> None:
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                result["sources"].append({"title": title, "uri": uri})

        # Current schema: steps[] timeline
        for step in data.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type", "")

            if step_type == "thought":
                # Planning / reasoning summaries: summary is a list of text parts.
                for part in step.get("summary", []) or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                        result["thinking"].append({"type": step_type, "text": part["text"]})

            elif step_type == "model_output":
                # Final report text + inline citations live in content[] text blocks.
                for block in step.get("content", []) or []:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    if isinstance(block.get("text"), str):
                        result["report"] += block["text"]
                    for ann in block.get("annotations", []) or []:
                        if isinstance(ann, dict) and ann.get("type") == "url_citation":
                            _add_source(ann.get("url", ""), ann.get("title", ""))

        # Legacy fallback: top-level output_text
        if not result["report"]:
            output_text = data.get("output_text")
            if isinstance(output_text, str) and output_text:
                result["report"] = output_text

        # Legacy fallback: steps[].text / steps[].parts[].text + groundingMetadata
        if not result["report"]:
            for step in data.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                text = step.get("text")
                if isinstance(text, str) and text:
                    result["report"] += text
                for part in step.get("parts", []) or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        result["report"] += part["text"]
                grounding = step.get("groundingMetadata") or step.get("grounding_metadata") or {}
                if grounding:
                    self._extract_sources(grounding, result["sources"])

        # Legacy fallback: outputs[]
        if not result["report"]:
            for output_item in data.get("outputs", []) or []:
                if not isinstance(output_item, dict):
                    continue
                if "text" in output_item:
                    result["report"] += output_item["text"]
                elif "parts" in output_item:
                    for part in output_item["parts"]:
                        if isinstance(part, dict) and "text" in part:
                            result["report"] += part["text"]
                grounding = output_item.get("groundingMetadata", {})
                self._extract_sources(grounding, result["sources"])

        # Legacy fallback: single output.parts[].text
        if not result["report"]:
            output = data.get("output", {})
            if isinstance(output, dict):
                for part in output.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        result["report"] += part["text"]
                grounding = output.get("groundingMetadata", {})
                self._extract_sources(grounding, result["sources"])

        # Top-level groundingMetadata (any schema)
        top_grounding = data.get("groundingMetadata") or data.get("grounding_metadata") or {}
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

    async def _resolve_redirect(self, url: str) -> str:
        """Resolve a Vertex grounding-api-redirect URL to the real article URL.

        Reads the 302 ``Location`` header without following it further. Returns
        the original ``url`` unchanged for non-grounding URLs or on any failure
        (the redirect links are time-sensitive, so a miss must not break the
        literature flow). Relative ``Location`` values are joined against the
        request URL to produce an absolute URL.
        """
        if not url or self.GROUNDING_REDIRECT_MARKER not in url:
            return url
        try:
            resp = await self.client.get(
                url, follow_redirects=False, timeout=15.0
            )
            if resp.is_redirect:
                location = resp.headers.get("location")
                if location:
                    # Handle relative Location headers (e.g. "/article/123").
                    if not location.startswith(("http://", "https://")):
                        location = str(httpx.URL(resp.url).join(location))
                    return location
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            logger.warning(f"[DeepResearch] redirect resolve failed for {url[:80]}: {e}")
        return url

    _TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

    async def _fetch_title(self, url: str) -> str:
        """Fetch ``url`` and return its HTML ``<title>`` text, or '' on failure.

        Only parses ``text/html`` responses; collapses internal whitespace and
        unescapes HTML entities. A browser-ish User-Agent avoids trivial bot
        blocks. Any error yields '' so the caller can fall back.
        """
        try:
            resp = await self.client.get(
                url,
                follow_redirects=True,
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ScAI-LiteratureBot/1.0)"},
            )
            if "html" not in resp.headers.get("content-type", "").lower():
                return ""
            match = self._TITLE_RE.search(resp.text)
            if match:
                title = html.unescape(match.group(1))
                return re.sub(r"\s+", " ", title).strip()
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            logger.warning(f"[DeepResearch] title fetch failed for {url[:80]}: {e}")
        return ""

    async def _enrich_source(self, source: Dict[str, str]) -> Dict[str, str]:
        """Resolve one source's redirect URL and backfill its title if empty."""
        real_url = await self._resolve_redirect(source.get("uri", ""))
        title = source.get("title") or ""
        if not title and real_url:
            title = await self._fetch_title(real_url)
        return {"title": title, "uri": real_url}

    async def enrich_sources(
        self, sources: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Resolve redirects + backfill titles for all sources, concurrently.

        Order is preserved so the report's inline ``[cite: N]`` markers stay
        aligned with the References list. Network failures degrade to the
        original URL / empty title per-source (handled in ``_enrich_source``).
        Concurrency is capped to avoid triggering rate limits on academic
        sites (nih.gov, researchgate.net, etc.).
        """
        if not sources:
            return sources
        sem = asyncio.Semaphore(5)

        async def _bounded_enrich(s):
            async with sem:
                return await self._enrich_source(s)

        return list(await asyncio.gather(*(_bounded_enrich(s) for s in sources)))

    async def research(self, query: str) -> Dict[str, Any]:
        """
        Convenience method to start research and wait for results
        
        Args:
            query: Research query
            
        Returns:
            Research results with report and sources
        """
        interaction_id = await self.start_research(query)
        result = await self.poll_research(interaction_id)
        result["sources"] = await self.enrich_sources(result.get("sources", []))
        return result


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
