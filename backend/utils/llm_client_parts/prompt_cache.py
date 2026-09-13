"""Automatic, provider-compatible prompt-cache negotiation."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_CACHE_CONTROL = {"type": "ephemeral"}
_CACHE_PARAMETER_PATTERN = r"cache[\s_-]*control"
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


def _is_openrouter_url(base_url: str) -> bool:
    """Recognize OpenRouter by hostname without matching lookalike domains."""
    try:
        host = (urlparse(base_url).hostname or "").rstrip(".").lower()
    except (TypeError, ValueError):
        return False
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


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
                    r"(?:^|[.\[])cache_control(?:$|[.\]])", parameter
                ))
    if not _CACHE_PARAMETER_RE.search(text):
        return False
    return bool(_CACHE_REJECTION_RE.search(text))


class PromptCacheController:
    """Apply cache hints and remember deterministic incompatibilities.

    Automatic cache control is used only on Anthropic's wire protocol or on a
    confidently identified Anthropic model routed through OpenRouter. Other
    providers retain their native implicit-cache behavior.
    """

    def __init__(
        self,
        base_url: str,
        context_getter: Optional[Callable[[], Tuple[str, str]]] = None,
    ) -> None:
        self.base_url = base_url
        self._context_getter = context_getter or (lambda: ("default", "unknown"))
        self._capabilities: Dict[Tuple[str, str, str], str] = {}
        self._announced: set[Tuple[str, str, str]] = set()
        self._extra_body_unsupported_protocols: set[str] = set()

    def _key(self, model: str, protocol: str, modality: str) -> Tuple[str, str, str]:
        return protocol, model, modality

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

    def describe(self, model: str, protocol: str, modality: str = "text") -> Dict[str, Any]:
        key = self._key(model, protocol, modality)
        automatic = self._uses_automatic_cache(model, protocol, modality)
        capability = self._capabilities.get(key)
        if automatic and not self._extra_body_supported(protocol):
            capability = capability or "sdk_unsupported"
        cache_control_enabled = (
            automatic
            and self._extra_body_supported(protocol)
            and capability != "unsupported"
        )
        return {
            "protocol": protocol,
            "modality": modality,
            "strategy": "automatic" if automatic else "provider_managed",
            "capability": (
                capability if automatic else "provider_managed"
            ) or "probing",
            "cache_control_enabled": cache_control_enabled,
            "session_affinity": self._session_affinity_id(protocol) is not None,
            "routing_constraint": (
                "direct_anthropic"
                if (
                    cache_control_enabled
                    and _is_openrouter_url(self.base_url)
                )
                else None
            ),
        }

    def _announce(self, model: str, protocol: str, modality: str) -> None:
        key = self._key(model, protocol, modality)
        if key in self._announced:
            return
        self._announced.add(key)
        state = self.describe(model, protocol, modality)
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
    ) -> Tuple[Dict[str, Any], bool]:
        prepared = dict(params)
        raw_extra = prepared.get("extra_body")
        extra_body = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}

        affinity_id = self._session_affinity_id(protocol)
        if affinity_id is not None:
            extra_body.setdefault("session_id", affinity_id)

        key = self._key(model, protocol, modality)
        automatic = (
            self._extra_body_supported(protocol)
            and self._uses_automatic_cache(model, protocol, modality)
            and self._capabilities.get(key) != "unsupported"
        )
        if automatic:
            extra_body.setdefault("cache_control", dict(_CACHE_CONTROL))

        if extra_body:
            prepared["extra_body"] = extra_body
        self._announce(model, protocol, modality)
        return prepared, automatic

    async def request(
        self,
        create: Callable[..., Awaitable[Any]],
        params: Mapping[str, Any],
        *,
        model: str,
        protocol: str,
        modality: str = "text",
    ) -> Any:
        prepared, automatic = self._prepare(params, model, protocol, modality)
        key = self._key(model, protocol, modality)
        try:
            result = await create(**prepared)
        except Exception as error:
            if (
                "extra_body" in prepared
                and isinstance(error, TypeError)
                and _EXTRA_BODY_TYPE_ERROR_RE.search(str(error))
            ):
                self._extra_body_unsupported_protocols.add(protocol)
                if automatic:
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
                fallback = dict(params)
                fallback.pop("extra_body", None)
                return await create(**fallback)
            if not automatic or not is_prompt_cache_rejection(error):
                raise

            self._capabilities[key] = "unsupported"
            _session_id, agent_name = self._context_getter()
            logger.warning(
                "[%s] Prompt cache control rejected for model=%s protocol=%s; "
                "retrying once without cache_control: %s: %s",
                agent_name,
                model,
                protocol,
                type(error).__name__,
                str(error)[:200],
            )
            fallback, _ = self._prepare(params, model, protocol, modality)
            return await create(**fallback)

        if automatic:
            self._capabilities[key] = "supported"
        return result
