"""Config-backed properties, LLM client wiring, and sandbox/notification plumbing.

Moved verbatim out of backend/agents/supervisor_agent.py; composed into SupervisorAgent.
"""
import logging


logger = logging.getLogger(__name__)


from ...utils.llm_client import OpenAICompatibleClient


# ---------------------------------------------------------------------------
# Rich HTML report CSS (extracted to utils/report_css.py for separation of concerns)
# ---------------------------------------------------------------------------
from ...utils.report_css import REPORT_CSS as _REPORT_CSS  # noqa: E402

from .shared import (  # noqa: F401
    _SPECIFIC_SECTION_IDS,
    _SECTION_ID_TO_DEFAULT_TITLE,
    _PLACEHOLDER_SECTION_TITLES,
    SupervisorRecord,
    SupervisorEvalResult,
    _CRITICAL_OUTPUT_PATTERNS,
    _PERCENTAGE_PATTERN,
    _INF_NAN_PATTERN,
    _SUPERVISOR_SYSTEM_PROMPT,
    _serialize_record,
    persist_graphic_abstract,
)


class RuntimeMixin:
    """Config-backed properties, LLM client wiring, and sandbox/notification plumbing."""

    @property
    def has_vision(self) -> bool:
        if self._vision_disabled_runtime:
            return False
        return self._provider.get_vision_setting("supervisor")

    @has_vision.setter
    def has_vision(self, value: bool):
        self._vision_disabled_runtime = not value

    @property
    def llm_client(self) -> OpenAICompatibleClient:
        return self._provider.get_llm_client("supervisor")

    @property
    def model(self) -> str:
        return self._provider.get_model("supervisor")

    @property
    def temperature(self) -> float:
        return self._provider.get_temperature("supervisor")

    @property
    def system_prompt(self) -> str:
        return self._provider.get_system_prompt("supervisor")

    def refresh_runtime_config_if_needed(self) -> None:
        """Reset cached supervisor client when effective config changed.

        Mirror of BaseAgent.refresh_runtime_config_if_needed for the
        SupervisorAgent which does not inherit BaseAgent.
        """
        current_revision = self._provider.get_config_revision("supervisor")
        if current_revision == self._last_config_revision:
            return

        self._last_config_revision = current_revision
        self._provider.clear_agent_client_cache("supervisor")

    def _set_llm_context(self):
        """Set LLM client context before each call (since client is shared)"""
        self.llm_client.set_context(self.session_id, "supervisor")

    def set_sandbox(self, sandbox):
        """Set the sandbox reference for Technique Agent doc lookups."""
        self._sandbox = sandbox

    async def _notify(self, msg_type: str, content: str):
        """Send notification via callback."""
        if self.message_callback:
            try:
                await self.message_callback(msg_type, content)
            except Exception as e:
                logger.warning(f"[{self.session_id}] Supervisor notify failed: {e}")
