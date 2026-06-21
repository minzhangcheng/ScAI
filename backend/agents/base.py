"""
Base Agent class for the multi-agent system
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import json
import logging
import re

logger = logging.getLogger(__name__)

from ..models.schemas import (
    AgentType, AgentMessage, MessageRole, AnalysisState
)
from ..utils.llm_client import OpenAICompatibleClient, LLMResponse
from ..utils.formatting import safe_parse_json as _safe_parse_json_impl
from ..utils.settings_provider import AgentSettingsProvider


@dataclass
class AgentConfig:
    """Runtime configuration for an agent."""
    temperature: float = 0.5
    max_tokens: int = 32768


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        agent_type: AgentType,
        settings_provider: AgentSettingsProvider,
        session_id: str = "default",
    ):
        self.agent_type = agent_type
        self._provider = settings_provider
        self.message_history: List[Dict[str, Any]] = []
        self.session_id = session_id
        self._vision_disabled_runtime: bool = False
        self._last_config_revision: Optional[int] = None

    @property
    def model(self) -> str:
        return self._provider.get_model(self.agent_type.value)

    @property
    def system_prompt(self) -> str:
        return self._provider.get_system_prompt(self.agent_type.value)

    @property
    def llm_client(self) -> OpenAICompatibleClient:
        return self._provider.get_llm_client(self.agent_type.value)

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(temperature=self._provider.get_temperature(self.agent_type.value))

    @property
    def has_vision(self) -> bool:
        if self._vision_disabled_runtime:
            return False
        return self._provider.get_vision_setting(self.agent_type.value)

    @has_vision.setter
    def has_vision(self, value: bool):
        self._vision_disabled_runtime = not value
    
    def _safe_parse_json(self, response: str, default: Dict[str, Any] = None, agent_label: str = None) -> Dict[str, Any]:
        """Safely parse JSON from LLM response. Delegates to shared implementation.

        When agent_label is provided, a warning is logged on parse failure
        (replaces former per-agent _parse_json_response wrappers).
        """
        if default is None:
            default = {"raw_response": "", "error": "Empty response from LLM"}

        if not response:
            return default

        result = _safe_parse_json_impl(response, default=None)
        if result is not None:
            return result

        if agent_label:
            logger.warning(
                f"[{agent_label}] JSON parse failed, raw response: {response[:200]}..."
            )
        return {"raw_response": response, "parse_error": "Failed to extract valid JSON from response"}
    
    def _set_llm_context(self):
        """Set LLM client context before each call (since client is shared)"""
        self.llm_client.set_context(self.session_id, self.agent_type.value)
    
    def set_session(self, session_id: str):
        """Update session ID for logging"""
        self.session_id = session_id

    def refresh_runtime_config_if_needed(self) -> None:
        """Reset per-agent runtime state when effective config changed.

        Most config is read dynamically through AgentSettingsProvider. This hook
        handles cached runtime objects, such as LLM clients and agent-specific
        clients, before the next agent use.
        """
        current_revision = self._provider.get_config_revision(self.agent_type.value)
        if current_revision == self._last_config_revision:
            return

        previous_revision = self._last_config_revision
        self._last_config_revision = current_revision
        self._provider.clear_agent_client_cache(self.agent_type.value)
        self.on_runtime_config_changed(previous_revision, current_revision)

    def on_runtime_config_changed(
        self,
        previous_revision: Optional[int],
        current_revision: int,
    ) -> None:
        """Subclass hook for runtime objects derived from config."""
        return None
        
    def reset_history(self):
        """Reset message history"""
        self.message_history = []

    def rollback_last_turn(self) -> bool:
        """Remove the last user+assistant turn from history.

        Safely verifies that the last two messages are an assistant reply
        followed by the user prompt that triggered it (in history order:
        [..., user, assistant]) before removing them.

        Returns:
            True if a turn was removed, False if history was too short
            or the tail didn't match the expected pattern.
        """
        if (len(self.message_history) >= 2
                and self.message_history[-1].get("role") == "assistant"
                and self.message_history[-2].get("role") == "user"):
            self.message_history.pop()  # assistant
            self.message_history.pop()  # user
            return True
        return False

    def add_to_history(self, role: str, content: str):
        """Add a message to history."""
        self.message_history.append({"role": role, "content": content})
    
    def get_messages_for_llm(self, user_message: str) -> List[Dict[str, Any]]:
        """Prepare messages for LLM call"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        # Limit history to the most recent 60 exchange pairs (models with 1M+ token context can handle more)
        _MAX_HISTORY_PAIRS = 60
        history = self.message_history
        if len(history) > _MAX_HISTORY_PAIRS * 2:
            dropped = len(history) - _MAX_HISTORY_PAIRS * 2
            logger.warning(
                f"[{self.agent_type}] Message history truncated: dropped {dropped} oldest messages "
                f"(keeping last {_MAX_HISTORY_PAIRS} pairs)"
            )
            history = history[-(_MAX_HISTORY_PAIRS * 2):]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages
    
    async def chat(self, message: str, temperature: Optional[float] = None) -> str:
        """
        Send a message and get a response

        Args:
            message: User message
            temperature: Sampling temperature (uses config default if None)

        Returns:
            Assistant response
        """
        # Set context before each call (client is shared between agents)
        self.refresh_runtime_config_if_needed()
        self._set_llm_context()

        messages = self.get_messages_for_llm(message)

        # Use provided temperature or fall back to config default
        actual_temp = temperature if temperature is not None else self.config.temperature

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=actual_temp
        )

        # Handle empty response - return empty string instead of None
        content = response.content if response and response.content else ""

        if not content:
            logger.warning(f"[{self.agent_type.value}] LLM returned empty response")

        # Add to history
        self.add_to_history("user", message)
        self.add_to_history("assistant", content)

        return content

    async def chat_without_history(self, message: str, temperature: Optional[float] = None, system_prompt: Optional[str] = None, timeout: Optional[float] = None, max_retries: Optional[int] = None) -> str:
        """Send a message and get a response without using or modifying message history.

        Useful for isolated Q&A interactions that should not be influenced by
        or contaminate the accumulated conversation context.

        Args:
            message: User message
            temperature: Sampling temperature (uses config default if None)
            system_prompt: Override system prompt (uses default if None)
            timeout: Per-request timeout in seconds (uses client default if None)
            max_retries: Override retry count (uses client default if None)
        """
        self.refresh_runtime_config_if_needed()
        self._set_llm_context()

        prompt = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ]

        actual_temp = temperature if temperature is not None else self.config.temperature

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": actual_temp,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries

        response = await self.llm_client.chat(**kwargs)

        content = response.content if response and response.content else ""
        if not content:
            logger.warning(f"[{self.agent_type.value}] LLM returned empty response (standalone)")

        return content

    async def chat_stream(
        self,
        message: str,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response

        Args:
            message: User message
            temperature: Sampling temperature (uses config default if None)

        Yields:
            Response chunks
        """
        # Set context before each call (client is shared between agents)
        self.refresh_runtime_config_if_needed()
        self._set_llm_context()

        messages = self.get_messages_for_llm(message)

        # Use provided temperature or fall back to config default
        actual_temp = temperature if temperature is not None else self.config.temperature

        full_response = ""
        async for chunk in self.llm_client.chat_stream(
            model=self.model,
            messages=messages,
            temperature=actual_temp
        ):
            full_response += chunk
            yield chunk
        
        # Add to history
        self.add_to_history("user", message)
        self.add_to_history("assistant", full_response)
    
    @abstractmethod
    async def process(
        self,
        state: AnalysisState,
        instruction: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an instruction and return results
        
        Args:
            state: Current analysis state
            instruction: Instruction to process
            **kwargs: Additional arguments
            
        Returns:
            Dict with results
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def create_message(
        self,
        role: MessageRole,
        content: str,
        **metadata
    ) -> AgentMessage:
        """Create an AgentMessage"""
        return AgentMessage(
            agent=self.agent_type,
            role=role,
            content=content,
            metadata=metadata
        )
