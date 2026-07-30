"""Secret sealing and log-safe redaction helpers.

Runtime configuration keeps plaintext credentials in memory because upstream
clients need them.  Every serialized ``AnalysisState`` seals those credentials
with the installation Fernet key, and validation opens them again after load.
Logging uses a stable preview format: first three characters, six asterisks,
and the final three characters.
"""

from __future__ import annotations

import logging
import re
import copy
from typing import Any, Mapping

from backend.utils.crypto import decrypt_value, encrypt_value


logger = logging.getLogger(__name__)

SEALED_SECRET_PREFIX = "@scai-secret:v1:"
SECRET_FIELD_NAMES = frozenset({
    "api_key",
    "deep_research_api_key",
    "google_api_key",
})


def _transform_nested_secret_fields(
    value: Any, transform, fields: frozenset[str] = SECRET_FIELD_NAMES
) -> Any:
    """Copy a small config tree while transforming recognized secret fields."""
    if isinstance(value, dict):
        return {
            key: (
                transform(item)
                if key in fields and item
                else _transform_nested_secret_fields(item, transform, fields)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_transform_nested_secret_fields(item, transform, fields) for item in value]
    return value


def mask_api_key(value: str) -> str:
    """Return the required non-reversible display preview for an API key."""
    text = str(value or "")
    if not text:
        return ""
    return f"{text[:3]}******{text[-3:]}"


def seal_secret(value: str) -> str:
    """Encrypt one plaintext secret, preserving empty/already-sealed values."""
    text = str(value or "")
    if not text or text.startswith(SEALED_SECRET_PREFIX):
        return text
    return SEALED_SECRET_PREFIX + encrypt_value(text)


def open_secret(value: str) -> str:
    """Decrypt one sealed secret, preserving plaintext legacy values.

    A secret sealed under a *different* installation key cannot be opened here.
    That is the normal case for a session imported from another instance, and
    for any instance whose ``SCAI_CONFIG_SECRET`` was rotated. Degrading to an
    empty value keeps the surrounding payload loadable: raising would abort
    ``AnalysisState`` validation, which in turn makes an imported LangGraph
    checkpoint impossible to deserialize and blocks the workflow from resuming.

    Dropping the credential is the same outcome portable export already chooses
    (:func:`strip_analysis_state_secrets` blanks these fields), and matches the
    rule stated in :func:`preserve_runtime_secrets` — credentials are
    operational configuration, not historical scientific state. The user
    reconfigures them; nothing else about the session is lost.
    """
    text = str(value or "")
    if not text.startswith(SEALED_SECRET_PREFIX):
        return text
    try:
        return decrypt_value(text[len(SEALED_SECRET_PREFIX):])
    except Exception as exc:
        # Never log the ciphertext itself — only why it could not be opened.
        logger.warning(
            "Sealed credential could not be decrypted (%s); treating it as unset. "
            "This is expected for a session imported from another instance or "
            "after SCAI_CONFIG_SECRET was rotated — reconfigure the API key.",
            type(exc).__name__,
        )
        return ""


def _transform_secret_fields(payload: Mapping[str, Any], transform) -> dict[str, Any]:
    # model_dump already owns its nested payload. Keep this helper selective so
    # multi-GB analysis state does not get deep-copied merely to transform a few
    # short credential strings.
    result = dict(payload)
    raw_server_configs = result.get("agent_server_configs")
    server_configs = dict(raw_server_configs) if isinstance(raw_server_configs, dict) else raw_server_configs
    if isinstance(server_configs, dict):
        for agent_name, raw_config in list(server_configs.items()):
            config = dict(raw_config) if isinstance(raw_config, dict) else raw_config
            if not isinstance(config, dict):
                continue
            server_configs[agent_name] = config
            for field in SECRET_FIELD_NAMES:
                if field in config and config[field]:
                    config[field] = transform(config[field])
        result["agent_server_configs"] = server_configs

    raw_literature_config = result.get("literature_config")
    literature_config = (
        dict(raw_literature_config)
        if isinstance(raw_literature_config, dict)
        else raw_literature_config
    )
    if isinstance(literature_config, dict):
        for field in SECRET_FIELD_NAMES:
            if field in literature_config and literature_config[field]:
                literature_config[field] = transform(literature_config[field])
        result["literature_config"] = literature_config

    # Imported analysis configurations may repeat default, per-agent, and
    # literature credentials in a compact nested snapshot.
    config_snapshot = result.get("config_snapshot")
    if isinstance(config_snapshot, dict):
        result["config_snapshot"] = _transform_nested_secret_fields(
            config_snapshot, transform
        )
    return result


def seal_analysis_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a selectively copied state payload with credentials encrypted."""
    return _transform_secret_fields(payload, seal_secret)


def open_analysis_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a selectively copied state payload with credentials decrypted."""
    return _transform_secret_fields(payload, open_secret)


# Endpoint fields are not credentials, but they are just as installation-bound:
# an archive that keeps them points the destination's agents at a host only the
# source could reach, and — since the paired api_key is stripped — invites the
# destination to send its own credential there.
ENDPOINT_FIELD_NAMES = frozenset({
    "llm_base_url",
    "base_url",
    "deep_research_llm_base_url",
    "proxy_url",
})


_ENDPOINT_BEARING_SECTIONS = (
    "agent_server_configs",
    "literature_config",
    "config_snapshot",
)


def strip_analysis_state_secrets(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a portable payload with credentials and endpoints removed entirely."""
    result = _transform_secret_fields(payload, lambda _value: "")

    for key in _ENDPOINT_BEARING_SECTIONS:
        if key in result:
            result[key] = _transform_nested_secret_fields(
                result[key], lambda _value: "", ENDPOINT_FIELD_NAMES
            )

    return result


def seal_system_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Encrypt global API credentials before writing configuration metadata."""
    result = copy.deepcopy(dict(payload))
    api = result.get("api")
    if isinstance(api, dict):
        for field in ("api_key", "proxy_url"):
            if api.get(field):
                api[field] = seal_secret(api[field])
    return result


def open_system_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Decrypt global API credentials, accepting plaintext legacy metadata."""
    result = copy.deepcopy(dict(payload))
    api = result.get("api")
    if isinstance(api, dict):
        for field in ("api_key", "proxy_url"):
            if api.get(field):
                api[field] = open_secret(api[field])
    return result


def preserve_runtime_secrets(source: Any, target: Any) -> None:
    """Copy current credentials into a credential-free restored state.

    Credentials are operational configuration, not historical scientific state.
    Snapshots intentionally omit them, so restore keeps the current credentials
    instead of rolling secrets backward or importing them from an archive.
    """
    if source is None or target is None:
        return
    source_servers = getattr(source, "agent_server_configs", {}) or {}
    target_servers = getattr(target, "agent_server_configs", {}) or {}
    for agent_name, source_config in source_servers.items():
        if not isinstance(source_config, dict):
            continue
        target_config = target_servers.setdefault(agent_name, {})
        if not isinstance(target_config, dict):
            continue
        for field in SECRET_FIELD_NAMES:
            if source_config.get(field):
                target_config[field] = source_config[field]
    target.agent_server_configs = target_servers

    source_literature = getattr(source, "literature_config", {}) or {}
    target_literature = getattr(target, "literature_config", {}) or {}
    for field in SECRET_FIELD_NAMES:
        if source_literature.get(field):
            target_literature[field] = source_literature[field]
    target.literature_config = target_literature


_NAMED_SECRET_RE = re.compile(
    r"(?P<prefix>(?:api_key|deep_research_api_key|google_api_key)"
    r"(?:\\?['\"])?\s*(?:=|:)\s*(?:\\?['\"])?)(?P<secret>[^\s,}'\"\\]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?P<prefix>\bBearer\s+)(?P<secret>[^\s,}'\"\\]+)", re.IGNORECASE)
_COMMON_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_.-]{6,}")


def redact_secrets(text: str) -> str:
    """Redact recognizable credentials in already-formatted log text."""
    rendered = str(text)

    def _named(match: re.Match[str]) -> str:
        return match.group("prefix") + mask_api_key(match.group("secret"))

    rendered = _NAMED_SECRET_RE.sub(_named, rendered)
    rendered = _BEARER_RE.sub(_named, rendered)
    rendered = _COMMON_KEY_RE.sub(lambda match: mask_api_key(match.group(0)), rendered)
    return rendered


class SecretRedactionFilter(logging.Filter):
    """Format a record once, then replace any credential with a safe preview."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
            record.msg = redact_secrets(rendered)
            record.args = ()
        except Exception:
            # Logging must never become a new application failure path.
            pass
        return True
