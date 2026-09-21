"""Automatic, provider-compatible prompt-cache negotiation."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
)
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_CACHE_CONTROL = {"type": "ephemeral"}
PROMPT_CACHE_EPHEMERAL_TAIL_KEY = "_scai_prompt_cache_ephemeral_tail"
PROMPT_CACHE_STABLE_BOUNDARY_KEY = "_scai_prompt_cache_stable_boundary"
_PROMPT_CACHE_METADATA_PREFIX = "_scai_prompt_cache_"
PromptCacheMode = Literal["auto", "explicit", "disabled"]
_CACHE_PARAMETER_PATTERN = r"(?:cache[\s_-]*control|prompt_cache_breakpoint|prompt_cache_options)"
_CACHE_PARAMETER_RE = re.compile(_CACHE_PARAMETER_PATTERN, re.IGNORECASE)
_CACHE_REJECTION_AFTER = (
    r"(?:unsupported|not supported|unknown|unrecognized|unexpected|invalid|"
    r"not (?:allowed|permitted)|cannot be used|"
    r"extra (?:field|input)s? (?:are|is) not permitted)"
)
_CACHE_REJECTION_RE = re.compile(
    rf"(?:"
    rf"{_CACHE_PARAMETER_PATTERN}\s*"
    rf"(?:(?:field|parameter|argument|input)\s*)?(?:is\s+|[:=]\s*)?"
    rf"{_CACHE_REJECTION_AFTER}"
    rf"|(?:unsupported|unknown|unrecognized|unexpected|invalid)\s+"
    rf"(?:(?:request\s+)?(?:field|parameter|argument|input)"
    rf"(?:\s+supplied)?\s*[:=]?\s*)?[\"']?{_CACHE_PARAMETER_PATTERN}"
    rf"|(?:field|parameter|argument|input)\s+[\"']?"
    rf"{_CACHE_PARAMETER_PATTERN}[\"']?\s+(?:is\s+)?"
    rf"{_CACHE_REJECTION_AFTER}"
    rf"|extra inputs? are not permitted\s*[:=]?\s*[\"']?"
    rf"{_CACHE_PARAMETER_PATTERN}"
    rf")",
    re.IGNORECASE,
)
_EXTRA_BODY_TYPE_ERROR_RE = re.compile(
    r"unexpected keyword argument [\"']extra_body[\"']", re.IGNORECASE
)


@dataclass(frozen=True)
class PromptCachePlan:
    """Request-scoped cache boundaries; existing plans target Anthropic."""

    mode: Literal["explicit", "disabled"] = "explicit"
    prefer_anthropic_messages: bool = True
    responses_prefix: str = ""

    def mode_for_protocol(self, protocol: str) -> PromptCacheMode:
        target = "responses" if self.responses_prefix else "anthropic"
        return self.mode if protocol == target else "disabled"


def responses_prefix_cache_plan(
    base_url: str, model: str, prefix: str,
) -> Optional[PromptCachePlan]:
    """Opt in only on documented OpenAI GPT-5.6+ Responses routes."""
    if not isinstance(base_url, str) or not isinstance(model, str):
        return None
    if not prefix:
        return None
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").rstrip(".").lower()
    if parsed.path.rstrip("/").endswith(("/messages", "/chat/completions")):
        return None
    name = model.lower()
    if _is_openrouter_url(base_url):
        if name.startswith("openrouter/"):
            name = name[len("openrouter/"):]
        if name.startswith("openai/"):
            name = name[len("openai/"):]
        else:
            return None
    elif host != "api.openai.com":
        return None
    match = re.fullmatch(r"gpt-(\d+)(?:\.(\d+))?(?:-[a-z0-9-]+)?", name)
    if not match or (int(match[1]), int(match[2] or 0)) < (5, 6):
        return None
    return PromptCachePlan(
        prefer_anthropic_messages=False, responses_prefix=prefix,
    )


def apply_responses_prefix_cache(
    params: Mapping[str, Any], prefix: str,
) -> Optional[Dict[str, Any]]:
    """Split the final user text without changing bytes, roles, or image order.

    A budget rewrite may have replaced the prefix. In that case leave the
    request unmarked rather than caching a different portion of the input.
    """
    items = list(params.get("input") or [])
    if not items or items[-1].get("role") != "user":
        return None
    content = items[-1].get("content")
    blocks = ([{"type": "input_text", "text": content}]
              if isinstance(content, str) else list(content or []))
    if not blocks or blocks[0].get("type") != "input_text":
        return None
    text = blocks[0].get("text")
    if not prefix or not isinstance(text, str) or not text.startswith(prefix):
        return None
    split = [{
        **blocks[0], "text": prefix,
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }]
    if text[len(prefix):]:
        split.append({**blocks[0], "text": text[len(prefix):]})
    items[-1] = {**items[-1], "content": split + blocks[1:]}
    return {
        **params,
        "input": items,
        "extra_body": {
            **(params.get("extra_body") or {}),
            "prompt_cache_options": {"mode": "explicit"},
        },
    }


def strip_prompt_cache_metadata(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove ScAI-only cache locators before non-Anthropic serialization."""

    return [
        {
            key: value
            for key, value in message.items()
            if not key.startswith(_PROMPT_CACHE_METADATA_PREFIX)
        }
        for message in messages
    ]


def _is_openrouter_url(base_url: str) -> bool:
    """Recognize OpenRouter by hostname without matching lookalike domains."""
    try:
        host = (urlparse(base_url).hostname or "").rstrip(".").lower()
    except (TypeError, ValueError):
        return False
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _is_anthropic_url(base_url: str) -> bool:
    """Recognize Anthropic-owned endpoints without matching lookalike domains."""
    try:
        host = (urlparse(base_url).hostname or "").rstrip(".").lower()
    except (TypeError, ValueError):
        return False
    return host == "anthropic.com" or host.endswith(".anthropic.com")


def _is_openrouter_anthropic_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    if normalized.startswith("openrouter/"):
        normalized = normalized[len("openrouter/") :]
    normalized = normalized.lstrip("~")
    return normalized.startswith("anthropic/")


def _error_text(error: Exception) -> str:
    parts = []
    message = getattr(error, "message", None)
    if message:
        parts.append(str(message))
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict) and detail.get("message"):
            parts.append(str(detail["message"]))
    elif body is not None:
        parts.append(str(body))
    if not parts:
        parts.append(str(error))
    return " ".join(parts)


def is_prompt_cache_rejection(error: Exception) -> bool:
    """Return True only for a deterministic rejection of our cache parameter."""
    text = _error_text(error)
    if isinstance(error, TypeError):
        return bool(
            _CACHE_PARAMETER_RE.search(text)
            and re.search(r"unexpected keyword argument", text, re.IGNORECASE)
        )

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status not in {400, 422}:
        return False
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict):
            parameter = str(detail.get("param") or "").lower().replace("-", "_")
            if parameter:
                return bool(re.search(
                    rf"(?:^|[.\[]){_CACHE_PARAMETER_PATTERN}(?:$|[.\]])", parameter
                ))
    if not _CACHE_PARAMETER_RE.search(text):
        return False
    return bool(_CACHE_REJECTION_RE.search(text))


class PromptCacheController:
    """Apply cache hints and remember deterministic incompatibilities.

    Automatic cache control is used only on Anthropic's wire protocol or on a
    confidently identified Anthropic model routed through OpenRouter. Other
    providers retain their native implicit-cache behavior unless a caller
    supplies an explicit prefix plan with marker-free fallback parameters.
    """

    def __init__(
        self,
        base_url: str,
        context_getter: Optional[Callable[[], Tuple[str, str]]] = None,
    ) -> None:
        self.base_url = base_url
        self._context_getter = context_getter or (lambda: ("default", "unknown"))
        self._capabilities: Dict[Tuple[str, str, str, str], str] = {}
        self._announced: set[Tuple[str, str, str, str]] = set()
        self._extra_body_unsupported_protocols: set[str] = set()

    def _capability_key(
        self,
        model: str,
        protocol: str,
        modality: str,
        cache_mode: PromptCacheMode,
    ) -> Tuple[str, str, str, str]:
        return protocol, model, modality, cache_mode

    def _uses_automatic_cache(self, model: str, protocol: str, modality: str) -> bool:
        return protocol == "anthropic" or (
            _is_openrouter_url(self.base_url)
            and _is_openrouter_anthropic_model(model)
        )

    def _extra_body_supported(self, protocol: str) -> bool:
        return protocol not in self._extra_body_unsupported_protocols

    def _session_affinity_id(self, protocol: str) -> Optional[str]:
        if (
            not self._extra_body_supported(protocol)
            or not _is_openrouter_url(self.base_url)
        ):
            return None
        session_id, _agent_name = self._context_getter()
        if not session_id:
            return None
        digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:32]
        return f"scai-{digest}"

    def describe(
        self,
        model: str,
        protocol: str,
        modality: str = "text",
        cache_mode: PromptCacheMode = "auto",
    ) -> Dict[str, Any]:
        key = self._capability_key(model, protocol, modality, cache_mode)
        automatic = self._uses_automatic_cache(model, protocol, modality)
        capability = self._capabilities.get(key)
        cache_requested = cache_mode == "explicit" or (
            cache_mode == "auto" and automatic
        )
        if (
            cache_mode == "auto"
            and automatic
            and not self._extra_body_supported(protocol)
        ):
            capability = capability or "sdk_unsupported"
        cache_control_enabled = (
            cache_requested
            and (
                cache_mode == "explicit"
                or self._extra_body_supported(protocol)
            )
            and capability != "unsupported"
        )
        return {
            "protocol": protocol,
            "modality": modality,
            "strategy": (
                cache_mode
                if cache_mode != "auto"
                else "automatic" if automatic else "provider_managed"
            ),
            "capability": (
                capability if cache_requested else "provider_managed"
            ) or "probing",
            "cache_control_enabled": cache_control_enabled,
            "session_affinity": self._session_affinity_id(protocol) is not None,
            "routing_constraint": (
                "direct_anthropic"
                if (
                    cache_control_enabled
                    and _is_openrouter_url(self.base_url)
                    and protocol == "anthropic"
                )
                else None
            ),
        }

    def _announce(
        self,
        model: str,
        protocol: str,
        modality: str,
        cache_mode: PromptCacheMode,
    ) -> None:
        key = (protocol, model, modality, cache_mode)
        if key in self._announced:
            return
        self._announced.add(key)
        state = self.describe(model, protocol, modality, cache_mode)
        _session_id, agent_name = self._context_getter()
        logger.info(
            "[%s] Prompt cache policy model=%s protocol=%s modality=%s "
            "strategy=%s session_affinity=%s",
            agent_name,
            model,
            protocol,
            modality,
            state["strategy"],
            state["session_affinity"],
        )

    def _prepare(
        self,
        params: Mapping[str, Any],
        model: str,
        protocol: str,
        modality: str,
        cache_mode: PromptCacheMode,
    ) -> Tuple[Dict[str, Any], bool]:
        prepared = dict(params)
        raw_extra = prepared.get("extra_body")
        extra_body = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}

        affinity_id = self._session_affinity_id(protocol)
        if affinity_id is not None:
            extra_body.setdefault("session_id", affinity_id)

        key = self._capability_key(model, protocol, modality, cache_mode)
        cache_enabled = (
            cache_mode != "disabled"
            and self._capabilities.get(key) != "unsupported"
            and (
                cache_mode == "explicit"
                or (
                    self._extra_body_supported(protocol)
                    and self._uses_automatic_cache(model, protocol, modality)
                )
            )
        )
        if cache_mode == "auto" and cache_enabled:
            extra_body.setdefault("cache_control", dict(_CACHE_CONTROL))
        elif cache_mode != "auto":
            extra_body.pop("cache_control", None)

        if extra_body:
            prepared["extra_body"] = extra_body
        else:
            prepared.pop("extra_body", None)
        self._announce(model, protocol, modality, cache_mode)
        return prepared, cache_enabled

    async def request(
        self,
        create: Callable[..., Awaitable[Any]],
        params: Mapping[str, Any],
        *,
        model: str,
        protocol: str,
        modality: str = "text",
        cache_mode: PromptCacheMode = "auto",
        fallback_params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        key = self._capability_key(model, protocol, modality, cache_mode)
        if cache_mode == "explicit" and fallback_params is None:
            raise ValueError("explicit prompt caching requires fallback_params")
        if (
            cache_mode == "explicit"
            and self._capabilities.get(key) == "unsupported"
        ):
            source_params = fallback_params
            effective_mode: PromptCacheMode = "disabled"
        else:
            source_params = params
            effective_mode = cache_mode
        prepared, cache_enabled = self._prepare(
            source_params, model, protocol, modality, effective_mode
        )
        try:
            result = await create(**prepared)
        except Exception as error:
            if (
                "extra_body" in prepared
                and isinstance(error, TypeError)
                and _EXTRA_BODY_TYPE_ERROR_RE.search(str(error))
            ):
                self._extra_body_unsupported_protocols.add(protocol)
                if cache_enabled:
                    self._capabilities[key] = "unsupported"
                _session_id, agent_name = self._context_getter()
                logger.warning(
                    "[%s] LLM SDK rejected extension request fields for model=%s "
                    "protocol=%s; retrying once without cache control or session "
                    "affinity: %s",
                    agent_name,
                    model,
                    protocol,
                    str(error)[:200],
                )
                fallback = dict(fallback_params if cache_mode == "explicit" else params)
                fallback.pop("extra_body", None)
                return await create(**fallback)
            if not cache_enabled or not is_prompt_cache_rejection(error):
                raise

            self._capabilities[key] = "unsupported"
            _session_id, agent_name = self._context_getter()
            logger.warning(
                "[%s] Prompt cache control rejected for model=%s protocol=%s; "
                "retrying once without cache parameters: %s: %s",
                agent_name,
                model,
                protocol,
                type(error).__name__,
                str(error)[:200],
            )
            fallback_source = (
                fallback_params if cache_mode == "explicit" else params
            )
            fallback, _ = self._prepare(
                fallback_source,
                model,
                protocol,
                modality,
                "disabled" if cache_mode == "explicit" else "auto",
            )
            return await create(**fallback)

        if cache_enabled:
            self._capabilities[key] = "supported"
        return result
