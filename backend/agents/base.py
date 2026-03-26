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


@dataclass
class AgentConfig:
    """Runtime configuration for an agent - allows centralized temperature management"""
    temperature: float = 0.5
    max_tokens: int = 32768

    @classmethod
    def from_temperatures(cls, agent_type: str, temperatures: Dict[str, float]) -> 'AgentConfig':
        """Create config from a temperatures dict (e.g., from AgentTemperatures)"""
        return cls(
            temperature=temperatures.get(agent_type, 0.5)
        )


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(
        self,
        agent_type: AgentType,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        session_id: str = "default",
        config: Optional[AgentConfig] = None
    ):
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = system_prompt
        self.message_history: List[Dict[str, Any]] = []
        self.session_id = session_id
        self.config = config or AgentConfig()
        self.has_vision: bool = True
    
    def _safe_parse_json(self, response: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Safely parse JSON from LLM response using balanced bracket matching.

        Args:
            response: LLM response string (may be None)
            default: Default value to return if parsing fails

        Returns:
            Parsed dict or default value
        """
        if default is None:
            default = {"raw_response": "", "error": "Empty response from LLM"}

        # Handle None or empty response
        if not response:
            return default

        try:
            # Find balanced JSON object using bracket counting (avoids greedy regex)
            start = response.find('{')
            if start == -1:
                return {"raw_response": response}

            depth = 0
            in_string = False
            escape_next = False
            end = start

            for i, c in enumerate(response[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\' and in_string:
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if depth == 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            return {"raw_response": response, "parse_error": str(e)}

        return {"raw_response": response}
    
    def _set_llm_context(self):
        """Set LLM client context before each call (since client is shared)"""
        self.llm_client.set_context(self.session_id, self.agent_type.value)
    
    def set_session(self, session_id: str):
        """Update session ID for logging"""
        self.session_id = session_id
        
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
        """Add a message to history"""
        self.message_history.append({
            "role": role,
            "content": content
        })
    
    def get_messages_for_llm(self, user_message: str) -> List[Dict[str, Any]]:
        """Prepare messages for LLM call"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        # Limit history to the most recent 20 exchange pairs to prevent unbounded growth
        _MAX_HISTORY_PAIRS = 20
        history = self.message_history
        if len(history) > _MAX_HISTORY_PAIRS * 2:
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

    async def chat_without_history(self, message: str, temperature: Optional[float] = None, system_prompt: Optional[str] = None) -> str:
        """Send a message and get a response without using or modifying message history.

        Useful for isolated Q&A interactions that should not be influenced by
        or contaminate the accumulated conversation context.
        """
        self._set_llm_context()

        prompt = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ]

        actual_temp = temperature if temperature is not None else self.config.temperature

        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=actual_temp
        )

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
        pass
    
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
