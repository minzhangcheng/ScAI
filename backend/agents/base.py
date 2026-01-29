"""
Base Agent class for the multi-agent system
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import json
import re

from ..models.schemas import (
    AgentType, AgentMessage, MessageRole, AnalysisState
)
from ..utils.llm_client import OpenAICompatibleClient, LLMResponse


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        agent_type: AgentType,
        llm_client: OpenAICompatibleClient,
        model: str,
        system_prompt: str,
        session_id: str = "default"
    ):
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = system_prompt
        self.message_history: List[Dict[str, Any]] = []
        self.session_id = session_id
    
    def _safe_parse_json(self, response: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Safely parse JSON from LLM response
        
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
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
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
        messages.extend(self.message_history)
        messages.append({"role": "user", "content": user_message})
        return messages
    
    async def chat(self, message: str, temperature: float = 0.7) -> str:
        """
        Send a message and get a response
        
        Args:
            message: User message
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        # Set context before each call (client is shared between agents)
        self._set_llm_context()
        
        messages = self.get_messages_for_llm(message)
        
        response = await self.llm_client.chat(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        
        # Add to history
        self.add_to_history("user", message)
        self.add_to_history("assistant", response.content)
        
        return response.content
    
    async def chat_stream(
        self,
        message: str,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response
        
        Args:
            message: User message
            temperature: Sampling temperature
            
        Yields:
            Response chunks
        """
        # Set context before each call (client is shared between agents)
        self._set_llm_context()
        
        messages = self.get_messages_for_llm(message)
        
        full_response = ""
        async for chunk in self.llm_client.chat_stream(
            model=self.model,
            messages=messages,
            temperature=temperature
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
