"""Unified settings reader for agent hot-reload.

Agents read model, temperature, system_prompt, and LLM client from this
provider on each call. The provider resolves the current AnalysisState
from the global sessions dict (when session_id is set), ensuring
freshness even after LangGraph checkpoint-deserialization replaces the
object. Changes made via session_config API routes are immediately
visible on the next get_* call.
"""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from backend.config._config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.llm_client import OpenAICompatibleClient
from backend.state import sessions

logger = logging.getLogger(__name__)


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
        self._retry_callback = retry_callback
        self._client_cache: Dict[Tuple[str, str], OpenAICompatibleClient] = {}

    def set_retry_callback(self, callback: Optional[Callable]) -> None:
        """Set or update the retry callback for newly created LLM clients."""
        self._retry_callback = callback

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

    def get_llm_client(self, agent_name: str) -> Optional[OpenAICompatibleClient]:
        agent_cfg = (self._get_state().agent_server_configs or {}).get(agent_name, {})
        agent_url = agent_cfg.get("llm_base_url", "")
        agent_key = agent_cfg.get("api_key", "")

        if (not agent_url or agent_url == self._config.api.llm_base_url) and \
           (not agent_key or agent_key == self._config.api.api_key):
            return self._global_client

        cache_key = (agent_url or self._config.api.llm_base_url,
                     agent_key or self._config.api.api_key)
        if cache_key not in self._client_cache:
            client = OpenAICompatibleClient(
                api_key=cache_key[1],
                base_url=cache_key[0],
                proxy_url=self._config.api.proxy_url,
                session_id=self._session_id,
                agent_name=agent_name,
            )
            if self._retry_callback:
                client.on_retry = self._retry_callback
            self._client_cache[cache_key] = client
        return self._client_cache[cache_key]

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def register_client(self, key: Tuple[str, str], client: OpenAICompatibleClient) -> None:
        """Register an externally-created client (e.g. deep research) for cleanup."""
        self._client_cache[key] = client

    def cached_clients(self):
        """Iterate over all cached LLM clients (for retry-callback wiring)."""
        return self._client_cache.values()

    # ------------------------------------------------------------------
    # Literature config
    # ------------------------------------------------------------------

    def get_literature_config(self) -> Dict[str, Any]:
        return self._get_state().literature_config or {}

    async def cleanup(self):
        for client in self._client_cache.values():
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"Failed to close cached LLM client: {e}")
        self._client_cache.clear()
