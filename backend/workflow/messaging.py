"""WebSocket messaging and state-sync helpers."""
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from ..agents.supervisor_agent import SupervisorEvalResult
from ..models.schemas import AgentType, AnalysisStep, AnalysisState, WebSocketMessage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MessagingMixin:
    """Mixin providing messaging and state-sync methods.

    These methods are mixed into SingleCellAnalysisWorkflow and rely on
    self.message_callback, self.state_callback, self.session_id (declared in core).
    """

    async def _send_message(self, msg_type: str, content: Any, agent: Optional[AgentType] = None):
        """Send message through callback"""
        if self.message_callback:
            msg = WebSocketMessage(
                type=msg_type,
                agent=agent,
                content=content
            )
            try:
                await self.message_callback(msg)
            except Exception as e:
                logger.exception("[Workflow] Failed to send message")

    async def _send_state_sync(self, analysis_state: AnalysisState, change_type: str):
        """Send unified state synchronization message after state changes"""
        # Increment version for synchronization
        await analysis_state.increment_version()

        # Save state via callback first
        if self.state_callback:
            try:
                await self.state_callback(analysis_state)
            except Exception as e:
                logger.exception(f"[{self.session_id}] State callback failed")

        # Then send sync message to frontend
        if self.message_callback:
            sync_msg = WebSocketMessage(
                type="state_sync",
                agent=None,
                content={
                    "change_type": change_type,
                    "version": analysis_state.state_version,
                    "status": analysis_state.status.value,
                    "current_round": analysis_state.current_round,
                    "analysis_phase": analysis_state.analysis_phase,
                    "timestamp": analysis_state.last_sync_timestamp.isoformat(),
                    "unresolved_errors": len(analysis_state.get_unresolved_errors()),
                    "wait_input_reason": analysis_state.wait_input_reason,
                    "llm_failure_info": analysis_state.llm_failure_info,
                    "agent_disabled_info": analysis_state.agent_disabled_info,
                    "species_input_info": analysis_state.species_input_info,
                }
            )
            try:
                await self.message_callback(sync_msg)
            except Exception as e:
                logger.exception(f"[{self.session_id}] Failed to send state sync")

    async def _agent_message_callback(self, msg_type: str, content: str):
        """Callback for agent messages"""
        await self._send_message(msg_type, content, AgentType.BIOINFORMATICS)

    async def _supervisor_message_callback(self, msg_type: str, content: str):
        """Callback for supervisor agent messages"""
        await self._send_message(msg_type, content, AgentType.SUPERVISOR)

    async def _technique_message_callback(self, msg_type: str, content: str):
        """Callback for technique agent messages"""
        await self._send_message(msg_type, content, AgentType.TECHNIQUE)

    async def _supervisor_callback(self, step: AnalysisStep, figures: Optional[List[str]] = None) -> SupervisorEvalResult:
        """Bridge bioinformatics agent's supervisor_callback to SupervisorAgent."""
        if not self.ensure_agent_ready("supervisor"):
            return SupervisorEvalResult()
        try:
            return await self.supervisor_agent.evaluate_step(step, figures or [])
        except Exception as e:
            logger.warning(f"[{self.session_id}] Supervisor callback failed: {e}")
            return SupervisorEvalResult()
