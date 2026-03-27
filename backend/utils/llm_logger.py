"""
LLM Interaction Logger

Records all LLM prompts, responses, and tool calls to files for debugging and analysis.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading

# Configure logger
logger = logging.getLogger(__name__)

# Log directory - use centralized data directory
from ..config import LOG_DIR, get_session_logs
LLM_LOG_DIR = LOG_DIR / "llm"
LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMInteraction:
    """Record of a single LLM interaction"""
    timestamp: str
    session_id: str
    agent: str
    model: str
    interaction_type: str  # "chat", "chat_with_tools", "vision"
    
    # Request
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    temperature: float = 0.7
    max_tokens: int = 32768
    
    # Response
    response_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    
    # Metadata
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class LLMLogger:
    """
    Singleton logger for all LLM interactions.

    Logs are written to:
    - {DATA_BASE_DIR}/logs/llm/YYYY-MM-DD.jsonl (all interactions, one per line)
    - {DATA_BASE_DIR}/sessions/<session_id>/logs/llm.jsonl (per-session logs)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.enabled = True
        self.log_to_console = True
        
        # Legacy session log directory (no longer used for new writes)
        self.session_log_dir = LLM_LOG_DIR / "sessions"
        
        logger.info(f"LLM Logger initialized. Logs at: {LLM_LOG_DIR.absolute()}")
    
    def _get_daily_log_path(self) -> Path:
        """Get path for today's log file"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return LLM_LOG_DIR / f"{date_str}.jsonl"
    
    def _get_session_log_path(self, session_id: str) -> Path:
        """Get path for session-specific log file (inside session dir)"""
        session_log_dir = get_session_logs(session_id)
        session_log_dir.mkdir(parents=True, exist_ok=True)
        return session_log_dir / "llm.jsonl"
    
    def log_interaction(self, interaction: LLMInteraction):
        """Log an LLM interaction"""
        if not self.enabled:
            return
        
        try:
            # Convert to dict
            data = asdict(interaction)

            # Truncate long messages for the log (keep full in separate field)
            log_data = data.copy()

            LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)

            # Write to daily log
            daily_path = self._get_daily_log_path()
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
            # Write to session log
            session_path = self._get_session_log_path(interaction.session_id)
            with open(session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
            # Console logging (summarized)
            if self.log_to_console:
                self._log_to_console(interaction)
                
        except Exception as e:
            logger.exception("Failed to log LLM interaction")
    
    def _log_to_console(self, interaction: LLMInteraction):
        """Log a summarized version to console"""
        # Get last user message
        last_user_msg = ""
        for msg in reversed(interaction.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_msg = content[:200]
                elif isinstance(content, list):
                    # Multi-modal content
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            last_user_msg = part.get("text", "")[:200]
                            break
                break
        
        # Summarize response
        response_summary = ""
        if interaction.response_content:
            response_summary = interaction.response_content[:200]
        if interaction.tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in interaction.tool_calls]
            response_summary = f"Tool calls: {tool_names}"
        
        # Log
        duration_str = f"{interaction.duration_ms:.0f}ms" if interaction.duration_ms is not None else "N/A"
        logger.info(
            f"\n{'='*60}\n"
            f"[LLM] [{interaction.agent}] - {interaction.model}\n"
            f"{'='*60}\n"
            f"> PROMPT: {last_user_msg}{'...' if len(last_user_msg) >= 200 else ''}\n"
            f"< RESPONSE: {response_summary}{'...' if len(response_summary) >= 200 else ''}\n"
            f"  Duration: {duration_str} | Tokens: {interaction.usage}\n"
            f"{'='*60}"
        )
    
    def log_request(
        self,
        session_id: str,
        agent: str,
        model: str,
        messages: List[Dict[str, Any]],
        interaction_type: str = "chat",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: int = 32768
    ) -> LLMInteraction:
        """
        Create an interaction record for a request.
        Returns the interaction object to be updated with response later.
        """
        return LLMInteraction(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            agent=agent,
            model=model,
            interaction_type=interaction_type,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def log_response(
        self,
        interaction: LLMInteraction,
        response_content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, int]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ):
        """Update interaction with response and log it"""
        interaction.response_content = response_content
        interaction.tool_calls = tool_calls
        interaction.usage = usage
        interaction.error = error
        interaction.duration_ms = duration_ms
        
        self.log_interaction(interaction)


# Global instance
llm_logger = LLMLogger()


def get_llm_logger() -> LLMLogger:
    """Get the global LLM logger instance"""
    return llm_logger
