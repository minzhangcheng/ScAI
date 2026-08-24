"""Unified settings reader for agent hot-reload.

Agents read model, temperature, system_prompt, and LLM client from this
provider on each call. The provider resolves the current AnalysisState
from the global sessions dict (when session_id is set), ensuring
freshness even after LangGraph checkpoint-deserialization replaces the
object. Changes made via session_config API routes are immediately
visible on the next get_* call.
"""
import asyncio
import hashlib
import ipaddress
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

from backend.config._config import APIConfig, SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.llm_client import OpenAICompatibleClient
from backend.state import sessions

logger = logging.getLogger(__name__)

# API contract: must stay in sync with MASKED_SECRET_VALUE in
# frontend/src/utils/literatureConfig.ts — secrets masked with this sentinel on
# the way out are detected and preserved (not overwritten) on the way back in.
MASKED_SECRET_VALUE = "********"
_SECRET_FIELD_NAMES = {
    "api_key",
    "deep_research_api_key",
    "google_api_key",
    "proxy_url",
}


def agent_endpoint_lacks_credential(
    agent_url: str,
    agent_key: str,
    global_url: str,
    global_key: str,
) -> bool:
    """Report a custom endpoint whose key is absent or installation-owned."""
    return bool(agent_url) and agent_url != global_url and (
        not agent_key or agent_key == global_key
    )


_UNROUTABLE_HOST_SUFFIXES = (".local", ".localhost", ".internal")


def endpoint_bypasses_global_proxy(url: str) -> bool:
    """Return whether a per-agent endpoint must remain directly reachable."""
    try:
        parsed = urlparse(url if "://" in url else f"//{url}")
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(_UNROUTABLE_HOST_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
    )


class AgentSettingsProvider:
    """Per-session settings reader for agents with LLM client caching.

    Every ``get_*`` method resolves the effective value by merging
    session-level overrides (``AnalysisState``) with global defaults
    (``SystemConfig``). The provider reads the current state from
    ``sessions[session_id]`` when available, falling back to the
    direct reference passed at construction time (for tests).
    """

    def __init__(
        self,
        state: AnalysisState,
        config: SystemConfig,
        global_client: Optional[OpenAICompatibleClient],
        session_id: Optional[str] = None,
        retry_callback: Optional[Callable] = None,
    ):
        self._state = state
        self._session_id = session_id
        self._config = config
        self._global_client = global_client
        self._global_client_signature = _api_transport_signature(config.api)
        self._retry_callback = retry_callback
        self._client_cache: Dict[Tuple[str, ...], OpenAICompatibleClient] = {}
        self._retired_clients: set[OpenAICompatibleClient] = set()
        self._client_close_tasks: set[asyncio.Task[None]] = set()
        self._client_close_task_ids: set[int] = set()
        self._client_use_counts: Dict[int, int] = {}
        self._cleanup_lock = asyncio.Lock()
        self._closing = False
        self._cleanup_complete = False

    def set_retry_callback(self, callback: Optional[Callable]) -> None:
        """Set or update the retry callback for newly created LLM clients."""
        self._retry_callback = callback
        if self._global_client is not None:
            self._global_client.on_retry = callback
        for client in self._client_cache.values():
            client.on_retry = callback

    def apply_retry_callback(self, client: Optional[OpenAICompatibleClient]) -> None:
        """Apply the current retry callback to an externally-created client."""
        if client is not None and self._retry_callback:
            client.on_retry = self._retry_callback

    def _get_state(self) -> AnalysisState:
        """Return the freshest available AnalysisState.

        When session_id is set, prefer the global sessions dict (which
        is updated by LangGraph's state_callback on every node
        transition). Falls back to the direct reference (for tests).
        """
        if self._session_id:
            current = sessions.get(self._session_id)
            if current is not None:
                return current
        return self._state

    def get_model(self, agent_name: str) -> str:
        # Explicit override (including empty / whitespace -> disabled) wins over default.
        custom_models = self._get_state().custom_models or {}
        if agent_name in custom_models:
            return (custom_models[agent_name] or "").strip()
        return getattr(self._config.models, f"{agent_name}_model", "")

    def get_temperature(self, agent_name: str) -> float:
        custom = (self._get_state().custom_temperatures or {}).get(agent_name)
        if custom is not None:
            return custom
        return getattr(self._config.temperatures, agent_name, 0.5)

    def get_system_prompt(self, agent_name: str) -> str:
        custom_prompts = self._get_state().custom_prompts or {}
        if agent_name in custom_prompts:
            return (custom_prompts[agent_name] or "").strip()
        return getattr(self._config.prompts, f"{agent_name}_prompt", "")

    def get_vision_setting(self, agent_name: str) -> bool:
        return (self._get_state().agent_vision_settings or {}).get(agent_name, True)

    def _live_api_config(self) -> APIConfig:
        return self._config.api

    def refresh_resolved_config(
        self,
        config: Optional[SystemConfig],
    ) -> Optional[OpenAICompatibleClient]:
        """Publish a resolved replacement before retiring the old transport."""
        if self._closing:
            return self._global_client

        resolved = config or self._config.model_copy(update={"api": APIConfig()})
        new_signature = _api_transport_signature(resolved.api)
        old_client = self._global_client
        old_signature = self._global_client_signature
        self._config = resolved

        if not resolved.api.api_key:
            self._global_client = None
            self._global_client_signature = new_signature
            self.clear_client_cache()
            if old_client is not None:
                self._retire_client(old_client)
            return None
        if new_signature == old_signature and _client_is_usable(old_client):
            return old_client

        try:
            replacement = OpenAICompatibleClient(
                api_key=resolved.api.api_key,
                base_url=resolved.api.llm_base_url,
                proxy_url=resolved.api.proxy_url,
                session_id=self._session_id,
                agent_name="workflow",
            )
            replacement.on_retry = self._retry_callback
        except Exception:
            self._global_client = None
            self._global_client_signature = new_signature
            self.clear_client_cache()
            if old_client is not None:
                self._retire_client(old_client)
            raise

        self._global_client = replacement
        self._global_client_signature = new_signature
        self.clear_client_cache()
        if old_client is not None and old_client is not replacement:
            self._retire_client(old_client)
        return replacement

    def get_llm_client(self, agent_name: str) -> Optional[OpenAICompatibleClient]:
        if self._closing:
            raise RuntimeError("settings provider is closing")
        agent_cfg = (self._get_state().agent_server_configs or {}).get(agent_name, {})
        agent_url = agent_cfg.get("llm_base_url", "")
        agent_key = agent_cfg.get("api_key", "")

        live_api = self._live_api_config()
        if (not agent_url or agent_url == live_api.llm_base_url) and \
           (not agent_key or agent_key == live_api.api_key):
            return self._usable_global_client()

        if agent_endpoint_lacks_credential(
            agent_url,
            agent_key,
            live_api.llm_base_url,
            live_api.api_key,
        ):
            # A custom endpoint whose credential is gone — the state of every
            # session that arrives through export/import, because export blanks
            # credentials and keeps everything else. Falling through would build
            # a client that sends *this* installation's key to an endpoint the
            # archive chose, which fails every call and hands the credential to
            # a host the operator never configured. Use the global client until
            # the endpoint is reconfigured with its own key.
            logger.warning(
                "[%s] agent %s has a custom endpoint but no credential of its own; "
                "using the global LLM client instead of pairing it with the "
                "installation key.",
                self._session_id,
                agent_name,
            )
            return self._usable_global_client()

        effective_url = agent_url or live_api.llm_base_url
        effective_key = agent_key or live_api.api_key
        effective_proxy = self._effective_proxy(effective_url)
        cache_key = (
            agent_name,
            effective_url,
            effective_key,
            effective_proxy or "",
        )
        if cache_key not in self._client_cache:
            client = OpenAICompatibleClient(
                api_key=cache_key[2],
                base_url=cache_key[1],
                proxy_url=effective_proxy,
                session_id=self._session_id,
                agent_name=agent_name,
            )
            if self._retry_callback:
                client.on_retry = self._retry_callback
            self._client_cache[cache_key] = client
        return self._client_cache[cache_key]

    def _usable_global_client(self) -> Optional[OpenAICompatibleClient]:
        client = self._global_client
        if client is None:
            raise ValueError("LLM api_key is not configured; set it in Config")
        if not _client_is_usable(client):
            raise RuntimeError("resolved global LLM client is closing")
        return client

    def _effective_proxy(self, effective_url: str) -> Optional[str]:
        live_api = self._live_api_config()
        proxy = live_api.proxy_url
        if not proxy:
            return None
        if effective_url == live_api.llm_base_url:
            return proxy
        return None if endpoint_bypasses_global_proxy(effective_url) else proxy

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def register_client(self, key: Tuple[str, str], client: OpenAICompatibleClient) -> None:
        """Register an externally-created client (e.g. deep research) for cleanup.

        Legacy callers pass a two-part key. Store it under a namespaced key so it
        cannot collide with normal per-agent server clients.
        """
        if self._cleanup_complete:
            # A completed provider can no longer honour an ownership transfer.
            # Reject before retaining the client so the caller remains its owner.
            raise RuntimeError("settings provider is already closed")
        if self._closing:
            # Ownership transfers at this API boundary even when registration
            # races with teardown; retain and close the rejected client.
            self._retire_client(client)
            raise RuntimeError("settings provider is closing")
        cache_key = ("registered", key[0], key[1])
        previous = self._client_cache.get(cache_key)
        if previous is not None and previous is not client:
            self._retire_client(previous)
        self._client_cache[cache_key] = client

    def cached_clients(self):
        """Iterate over all cached LLM clients (for retry-callback wiring)."""
        return self._client_cache.values()

    def lease_client(self, client: OpenAICompatibleClient) -> None:
        """Prevent a cached client from closing while one request uses it."""
        if self._closing:
            raise RuntimeError("settings provider is closing")
        client_id = id(client)
        self._client_use_counts[client_id] = (
            self._client_use_counts.get(client_id, 0) + 1
        )

    def release_client(self, client: OpenAICompatibleClient) -> None:
        """Release a request lease and close a retired client when now idle."""
        client_id = id(client)
        remaining = self._client_use_counts.get(client_id, 0) - 1
        if remaining > 0:
            self._client_use_counts[client_id] = remaining
            return
        self._client_use_counts.pop(client_id, None)
        if client in self._retired_clients:
            self._retire_client(client)

    def clear_agent_client_cache(self, agent_name: str) -> None:
        """Retire cached clients for one agent and close them when possible.

        This method stays synchronous because agents call it from their runtime
        refresh hook. Ownership nevertheless remains with the provider: under a
        running event loop each retired client receives a tracked close task;
        otherwise it remains queued for :meth:`cleanup`. A removed client must
        never become unreachable before one of those paths closes it.
        """
        for key in list(self._client_cache):
            if key[0] == agent_name:
                client = self._client_cache.pop(key, None)
                if client is not None:
                    self._retire_client(client)

    def clear_client_cache(self) -> None:
        """Retire every cached client without dropping provider ownership."""
        for key in list(self._client_cache):
            client = self._client_cache.pop(key, None)
            if client is not None:
                self._retire_client(client)

    def _retire_client(self, client: OpenAICompatibleClient) -> None:
        self._retired_clients.add(client)
        client_id = id(client)
        if self._client_use_counts.get(client_id, 0) > 0:
            return
        if client_id in self._client_close_task_ids:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._close_retired_client(client))
        self._client_close_tasks.add(task)
        self._client_close_task_ids.add(client_id)

        def _discard(done_task):
            self._client_close_tasks.discard(done_task)
            self._client_close_task_ids.discard(client_id)
            try:
                outcome = done_task.result()
            except (Exception, asyncio.CancelledError):
                return
            if (
                outcome == "deferred"
                and client in self._retired_clients
                and self._client_use_counts.get(client_id, 0) == 0
            ):
                self._retire_client(client)

        task.add_done_callback(_discard)

    async def _close_retired_client(
        self, client: OpenAICompatibleClient,
    ) -> str:
        # A lease can be acquired after retirement but before this scheduled
        # coroutine starts. Recheck at the actual close boundary.
        if self._client_use_counts.get(id(client), 0) > 0:
            return "deferred"
        try:
            await client.close()
        except Exception as exc:
            # Keep ownership in _retired_clients so cleanup can retry.
            logger.debug("Failed to close retired LLM client: %s", exc)
            return "failed"
        else:
            self._retired_clients.discard(client)
            return "closed"

    # ------------------------------------------------------------------
    # Literature config
    # ------------------------------------------------------------------

    def get_literature_config(self) -> Dict[str, Any]:
        return self._get_state().literature_config or {}

    @staticmethod
    def mask_literature_config(config: Dict[str, Any]) -> Dict[str, Any]:
        masked = dict(config or {})
        for key in _SECRET_FIELD_NAMES:
            if masked.get(key):
                masked[key] = MASKED_SECRET_VALUE
        return masked

    @staticmethod
    def resolve_literature_config_update(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        existing = existing or {}
        incoming = incoming or {}
        resolved = dict(incoming)
        for key in _SECRET_FIELD_NAMES:
            if resolved.get(key) == MASKED_SECRET_VALUE:
                resolved[key] = existing.get(key, "")
        return resolved

    # ------------------------------------------------------------------
    # Config revision tracking
    # ------------------------------------------------------------------

    def _agent_extra_signature(self, agent_name: str, state: AnalysisState) -> Dict[str, Any]:
        """Return agent-specific config values that affect runtime behavior."""
        if agent_name == "literature":
            return {"literature_config": state.literature_config or {}}
        return {}

    def get_config_signature(self, agent_name: str) -> tuple:
        """Return all config values that affect an agent's next behavior."""
        state = self._get_state()
        signature = {
            "model": self.get_model(agent_name),
            "temperature": self.get_temperature(agent_name),
            "system_prompt": self.get_system_prompt(agent_name),
            "vision_enabled": self.get_vision_setting(agent_name),
            "server": (state.agent_server_configs or {}).get(agent_name, {}),
            "resolved_api": {
                "llm_base_url": self._live_api_config().llm_base_url,
                "api_key": self._live_api_config().api_key,
                "proxy_url": self._live_api_config().proxy_url,
            },
            "proxy_url": self._effective_proxy(
                (state.agent_server_configs or {}).get(agent_name, {}).get(
                    "llm_base_url"
                )
                or self._live_api_config().llm_base_url
            ),
        }
        signature.update(self._agent_extra_signature(agent_name, state))
        return tuple(sorted(_secret_safe_signature(signature).items()))

    def get_config_revision(self, agent_name: str) -> int:
        """Stable integer revision for an agent's effective runtime config."""
        raw = json.dumps(
            self.get_config_signature(agent_name),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return int(digest, 16)

    async def cleanup(self):
        async with self._cleanup_lock:
            if self._cleanup_complete:
                return
            # This flag is set before the first await. Event-loop callers can
            # no longer create a client while cleanup is draining ownership.
            self._closing = True

            while True:
                self._retired_clients.update(self._client_cache.values())
                self._client_cache.clear()

                pending_tasks = list(self._client_close_tasks)
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    self._client_close_tasks.difference_update(pending_tasks)

                failures: list[Exception] = []
                in_use = 0
                for client in list(self._retired_clients):
                    client_id = id(client)
                    if self._client_use_counts.get(client_id, 0) > 0:
                        in_use += 1
                        continue
                    # register_client() can retire a late client while the
                    # task snapshot above is being awaited. Its tracked task
                    # remains the single closer; do not close it concurrently.
                    if client_id in self._client_close_task_ids:
                        continue
                    try:
                        await client.close()
                    except Exception as exc:
                        failures.append(exc)
                        logger.debug("Failed to close cached LLM client: %s", exc)
                    else:
                        self._retired_clients.discard(client)

                if failures:
                    # Keep failed clients strongly owned for the next
                    # workflow.close()/cleanup() retry.
                    raise RuntimeError(
                        f"failed to close {len(failures)} settings-provider client(s)"
                    ) from failures[0]
                if in_use:
                    raise RuntimeError(
                        f"{in_use} settings-provider client(s) still in use"
                    )

                if self._client_close_tasks:
                    continue
                if self._retired_clients:
                    await asyncio.sleep(0)
                    continue

                # clear_agent_client_cache() may have retired an existing
                # client while this coroutine was awaiting a close. Loop until
                # every provider-owned source is empty.
                if (
                    not self._client_cache
                    and not self._retired_clients
                    and not self._client_close_tasks
                ):
                    self._cleanup_complete = True
                    return


def _secret_safe_signature(value: Any, key_name: str = "") -> Any:
    if isinstance(value, dict):
        return {str(k): _secret_safe_signature(v, str(k)) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_secret_safe_signature(v, key_name) for v in value]
    if key_name in _SECRET_FIELD_NAMES and value:
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return {"sha256": digest}
    return value


def _api_transport_signature(api: APIConfig) -> Tuple[str, str, str]:
    """Return a transport comparison key without retaining raw credentials."""

    def _digest(value: Any) -> str:
        if not value:
            return ""
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    return (
        str(api.llm_base_url or ""),
        _digest(api.api_key),
        _digest(api.proxy_url),
    )


def _client_is_usable(client: Optional[OpenAICompatibleClient]) -> bool:
    return bool(
        client is not None
        and getattr(client, "_closing", False) is not True
        and getattr(client, "_closed", False) is not True
    )


async def refresh_active_providers(
    user_id: Optional[str] = None,
) -> int:
    """Refresh resolved transports for active analysis/report workflows."""
    from backend.state import report_workflows, workflows
    from backend.utils.config_resolver import resolve_config_for_user

    targets: list[tuple[Any, AgentSettingsProvider, str]] = []
    seen: set[int] = set()
    for workflow in list(workflows.values()) + list(report_workflows.values()):
        provider = getattr(workflow, "_settings_provider", None)
        if not isinstance(provider, AgentSettingsProvider) or id(provider) in seen:
            continue
        seen.add(id(provider))
        try:
            owner_id = str(getattr(provider._get_state(), "user_id", "") or "")
        except Exception:
            continue
        if user_id is not None and owner_id != user_id:
            continue
        targets.append((workflow, provider, owner_id))

    resolved_by_user: dict[str, Optional[SystemConfig]] = {}
    for owner_id in {owner for _, _, owner in targets}:
        try:
            resolved_by_user[owner_id] = await resolve_config_for_user(owner_id)
        except Exception:
            logger.warning(
                "Could not resolve refreshed LLM configuration for an active workflow"
            )
            resolved_by_user[owner_id] = None

    refreshed = 0
    for workflow, provider, owner_id in targets:
        try:
            replacement = provider.refresh_resolved_config(
                resolved_by_user.get(owner_id)
            )
        except Exception:
            logger.warning(
                "Could not construct a replacement LLM client for an active workflow"
            )
            continue

        manager = getattr(workflow, "llm_manager", None)
        if manager is not None and replacement is not None:
            manager.llm = replacement
            manager.openrouter = replacement
        refreshed += 1
    return refreshed
