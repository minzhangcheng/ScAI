# Endpoint protocol routing and request/response translation between the three APIs.
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

from .model_traits import _is_deep_research_model
from .prompt_cache import (
    PROMPT_CACHE_EPHEMERAL_TAIL_KEY,
    PROMPT_CACHE_STABLE_BOUNDARY_KEY,
    PromptCachePlan,
    _is_openrouter_anthropic_model,
    _is_openrouter_url,
)
from .responses import ToolCall


# Per-endpoint cache (base_url) of Responses endpoints that rejected the
# `background` parameter. Once seen, _build_params skips background for that
# endpoint so we don't pay a failed round-trip on every deep-research call.
_NO_BACKGROUND_ENDPOINT: set = set()

# Two-level protocol cache shared across client instances and agents.
# The first level records endpoints with no Responses API.
_NO_RESPONSES_ENDPOINT: set = set()

# The second level records per-model protocol choices. Default text keeps its
# legacy two-part key; vision, tools, and explicit-cache text use isolated
# modality keys so their protocol decisions cannot alter ordinary text calls.
_PROBE_CACHE: Dict[Tuple[Any, ...], str] = {}

# Deterministic Responses vision-stream rejections, keyed like the vision
# protocol cache: (base_url, model, "vision"). Transient failures never enter it.
_NO_RESPONSES_VISION_STREAM: set = set()

# Deterministic stream rejections for every (modality, protocol) pair except
# Responses vision, which predates this set and keeps its own cache above.
# Keyed by (base_url, model, modality, wire_protocol).
_NO_TEXT_STREAM: set = set()

# Chat Completions endpoints that stream successfully but reject
# stream_options={"include_usage": True}, keyed by (base_url, model).
_NO_CHAT_STREAM_OPTIONS: set = set()

_PROBE_LOCKS: Dict[Tuple[Any, ...], asyncio.Lock] = {}

TEXT_MODALITY = "text"
CACHED_TEXT_MODALITY = "cached_text"
TOOLS_MODALITY = "tools"
VISION_MODALITY = "vision"


def _probe_key(base_url: str, model: str, modality: str = TEXT_MODALITY):
    return (base_url, model) if modality == TEXT_MODALITY else (base_url, model, modality)

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

def _base_protocol_order(base_url: str) -> List[str]:
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


def _route_protocol_order(
    base_url: str,
    model: str,
    modality: str = TEXT_MODALITY,
    *,
    prefer_anthropic_cache: bool = False,
) -> List[str]:
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
    cached = _PROBE_CACHE.get(_probe_key(base_url, model, modality))
    if cached is not None:
        return [cached]
    if modality == TEXT_MODALITY and _is_deep_research_model(model):
        return ["responses", "chat"]

    order = _base_protocol_order(base_url)
    # Vision discovery may follow the already-proven text protocol, but its
    # result is cached independently under the modality key.
    if modality != TEXT_MODALITY:
        text_choice = _PROBE_CACHE.get(_probe_key(base_url, model, TEXT_MODALITY))
        if text_choice in order:
            order.remove(text_choice)
            order.insert(0, text_choice)
    if (
        prefer_anthropic_cache
        and _is_openrouter_url(base_url)
        and _is_openrouter_anthropic_model(model)
        and "anthropic" in order
    ):
        order.remove("anthropic")
        order.insert(0, "anthropic")
    return order

def _route_protocol(
    base_url: str,
    model: str,
    modality: str = TEXT_MODALITY,
    *,
    prefer_anthropic_cache: bool = False,
) -> str:
    """Primary protocol for (base_url, model) — head of the probe order. Retained
    for streaming dispatch and callers/tests that need a single value."""
    return _route_protocol_order(
        base_url,
        model,
        modality,
        prefer_anthropic_cache=prefer_anthropic_cache,
    )[0]

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
    """Normalize a base_url for AsyncAnthropic: strip a trailing /v1/messages,
    /messages, or /v1 so the SDK's own /v1/messages append is not doubled."""
    return _strip_path_suffix(base_url, ("/v1/messages", "/messages", "/v1"))

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

        # A regular Responses message accepts role/content, not the wider
        # Chat Completions message shape. Rebuild the item instead of passing
        # the source dict through: raw assistant echoes commonly carry
        # ``tool_calls=None`` plus provider-specific reasoning fields, and
        # strict Responses gateways reject even null unknown parameters (for
        # example ``input[19].tool_calls``).
        if role is not None:
            content = msg.get("content")
            # An empty assistant echo has no conversational value and some
            # gateways reject a null content item. Real tool-call turns were
            # already converted above.
            if role == "assistant" and content is None:
                continue
            items.append({"role": role, "content": content})
            continue

        # Already-native Responses input items have no role (for example a
        # caller-provided function_call item). Preserve those verbatim.
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

def _with_anthropic_cache_control(content: Any) -> Any:
    """Mark the last cacheable Anthropic block without mutating the source."""

    if isinstance(content, str):
        if not content:
            return content
        return [{
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"},
        }]
    if not isinstance(content, list):
        return content
    blocks = [dict(block) if isinstance(block, dict) else block for block in content]
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {"text", "tool_use", "tool_result"}:
            continue
        blocks[index] = {
            **block,
            "cache_control": {"type": "ephemeral"},
        }
        break
    return blocks


def _anthropic_messages_translate(
    messages: List[Dict[str, Any]],
    prompt_cache_plan: Optional[PromptCachePlan] = None,
):
    """Translate canonical OpenAI-chat messages -> (system, anthropic_messages).
    - first system message -> top-level `system`
    - assistant.tool_calls -> tool_use content blocks
    - role 'tool' -> user message with a tool_result block"""
    explicit_cache = (
        prompt_cache_plan is not None
        and prompt_cache_plan.mode == "explicit"
    )
    stable_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get(PROMPT_CACHE_STABLE_BOUNDARY_KEY) is True
    ]
    tail_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get(PROMPT_CACHE_EPHEMERAL_TAIL_KEY) is True
    ]
    stable_boundary = None
    if len(stable_indexes) == 1 and not tail_indexes:
        stable_boundary = stable_indexes[0]
    elif len(tail_indexes) == 1 and not stable_indexes:
        stable_boundary = tail_indexes[0] - 1
    elif explicit_cache:
        logger.warning(
            "Explicit prompt-cache plan expected one stable-boundary or "
            "ephemeral-tail locator; got stable=%d ephemeral=%d",
            len(stable_indexes),
            len(tail_indexes),
        )

    system = None
    out: List[Dict[str, Any]] = []
    for message_index, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system" and system is None:
            system = msg.get("content")
            if explicit_cache:
                system = _with_anthropic_cache_control(system)
            continue
        if role == "tool":
            content = [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": msg.get("content"),
            }]
            if explicit_cache and message_index == stable_boundary:
                content = _with_anthropic_cache_control(content)
            out.append({"role": "user", "content": content})
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
            if explicit_cache and message_index == stable_boundary:
                blocks = _with_anthropic_cache_control(blocks)
            out.append({"role": "assistant", "content": blocks})
            continue
        content = msg.get("content")
        if explicit_cache and message_index == stable_boundary:
            content = _with_anthropic_cache_control(content)
        out.append({"role": role, "content": content})
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
    uncached_input = u.get("input_tokens", 0) or 0
    cache_creation_input = u.get("cache_creation_input_tokens", 0) or 0
    cache_read_input = u.get("cache_read_input_tokens", 0) or 0
    inp = uncached_input + cache_creation_input + cache_read_input
    outp = u.get("output_tokens", 0) or 0
    usage = {"prompt_tokens": inp, "completion_tokens": outp, "total_tokens": inp + outp}
    has_cache_telemetry = any(
        u.get(key) is not None
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
    )
    if has_cache_telemetry:
        usage["cached_tokens"] = cache_read_input
        usage["cache_write_tokens"] = cache_creation_input

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
                elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                    # A refusal is still the provider's public answer channel;
                    # do not misclassify it as a reasoning-only empty response.
                    text_parts.append(block["refusal"])
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
    if details.get("cached_tokens") is not None:
        usage["cached_tokens"] = details["cached_tokens"] or 0
    if details.get("cache_write_tokens") is not None:
        usage["cache_write_tokens"] = details["cache_write_tokens"] or 0

    return "\n".join(text_parts), tool_calls, (citations or None), usage
