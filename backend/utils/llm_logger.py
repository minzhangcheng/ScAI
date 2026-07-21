"""
LLM Interaction Logger

Records bounded LLM diagnostics without duplicating complete growing histories.
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading

_MAX_LOG_MESSAGES = 12
_MAX_LOG_TEXT_CHARS = 20_000
_MAX_LOG_RESPONSE_CHARS = 40_000
_MAX_LOG_TOOL_CALL_CHARS = 8_000


def _truncate_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _compact_content(value: Any) -> Any:
    """Bound text and remove inline image bytes from one logged message."""
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return f"[inline image omitted: {len(value)} chars]"
        return _truncate_text(value, _MAX_LOG_TEXT_CHARS)
    if isinstance(value, list):
        compact = []
        for item in value[:20]:
            if not isinstance(item, dict):
                compact.append(_truncate_text(item, _MAX_LOG_TEXT_CHARS))
                continue
            cleaned = dict(item)
            for key in ("text", "content"):
                if key in cleaned:
                    cleaned[key] = _compact_content(cleaned[key])
            image_url = cleaned.get("image_url")
            if isinstance(image_url, dict):
                image_url = dict(image_url)
                url = image_url.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    image_url["url"] = f"[inline image omitted: {len(url)} chars]"
                cleaned["image_url"] = image_url
            compact.append(cleaned)
        return compact
    return value


def _compact_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the system contract plus the recent conversational window."""
    if len(messages) <= _MAX_LOG_MESSAGES:
        selected = messages
    else:
        first_system = next((m for m in messages if m.get("role") == "system"), None)
        recent_count = _MAX_LOG_MESSAGES - (1 if first_system is not None else 0)
        selected = ([first_system] if first_system is not None else []) + messages[-recent_count:]
    return [
        {**message, "content": _compact_content(message.get("content"))}
        for message in selected
    ]


def _compact_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return tools
    compact = []
    for tool in tools[:30]:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        compact.append({
            "type": tool.get("type", "function") if isinstance(tool, dict) else "unknown",
            "function": {
                "name": function.get("name"),
                "description": _truncate_text(function.get("description"), 1_000),
            },
        })
    return compact


def _tool_name(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    return function.get("name") if isinstance(function, dict) else None


def _compact_tool_calls(calls: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not calls:
        return calls
    compact = []
    for call in calls[:30]:
        item = dict(call)
        function = item.get("function")
        if isinstance(function, dict):
            function = dict(function)
            function["arguments"] = _truncate_text(
                function.get("arguments"), _MAX_LOG_TOOL_CALL_CHARS
            )
            item["function"] = function
        compact.append(item)
    return compact

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
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None  # LLM API endpoint

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
        
        # Async I/O lock for offloading blocking file writes to a thread pool
        self._async_lock = asyncio.Lock()
        
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
            # The per-session record is the canonical diagnostic copy. Histories
            # grow on every agent turn, so storing every prior message verbatim
            # would make JSONL size quadratic in conversation length.
            data = asdict(interaction)
            session_data = {
                **data,
                "messages": _compact_messages(data["messages"]),
                "tools": _compact_tools(data.get("tools")),
                "response_content": _truncate_text(
                    data.get("response_content"), _MAX_LOG_RESPONSE_CHARS
                ),
                "tool_calls": _compact_tool_calls(data.get("tool_calls")),
                "message_count": len(data["messages"]),
            }
            # Daily logs are an operational index, not a second full copy.
            daily_data = {
                key: data.get(key)
                for key in (
                    "timestamp", "session_id", "agent", "model",
                    "interaction_type", "temperature", "max_tokens", "base_url",
                    "usage", "error", "duration_ms",
                )
            }
            daily_data.update({
                "message_count": len(data["messages"]),
                "response_chars": len(data.get("response_content") or ""),
                "tool_names": [
                    _tool_name(tool)
                    for tool in (data.get("tools") or [])
                    if _tool_name(tool)
                ],
            })

            LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)

            # Write to daily log
            daily_path = self._get_daily_log_path()
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(daily_data, ensure_ascii=False) + "\n")
            
            # Write to session log
            session_path = self._get_session_log_path(interaction.session_id)
            with open(session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_data, ensure_ascii=False) + "\n")
            
            # Console logging (summarized)
            if self.log_to_console:
                self._log_to_console(interaction)
                
        except Exception as e:
            logger.exception("Failed to log LLM interaction")
    
    async def log_interaction_async(self, interaction: LLMInteraction):
        """Async version of log_interaction — offloads blocking file I/O
        to a thread pool, serialized under an asyncio.Lock."""
        if not self.enabled:
            return
        async with self._async_lock:
            await asyncio.to_thread(self.log_interaction, interaction)
    
    async def log_response_async(
        self,
        interaction: LLMInteraction,
        response_content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, int]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ):
        """Async version of log_response — updates interaction and logs via
        log_interaction_async."""
        interaction.response_content = response_content
        interaction.tool_calls = tool_calls
        interaction.usage = usage
        interaction.error = error
        interaction.duration_ms = duration_ms
        
        await self.log_interaction_async(interaction)
    
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
        # Build token summary with cache info
        if interaction.usage and interaction.usage.get("cached_tokens", 0) > 0:
            cached = interaction.usage["cached_tokens"]
            prompt = interaction.usage.get("prompt_tokens", 0)
            cache_pct = f"{cached / prompt * 100:.0f}%" if prompt else "N/A"
            token_summary = f"{interaction.usage} (cache hit: {cache_pct})"
        else:
            token_summary = str(interaction.usage)
        logger.info(
            f"\n{'='*60}\n"
            f"[LLM] [{interaction.agent}] - {interaction.model}\n"
            f"{'='*60}\n"
            f"> PROMPT: {last_user_msg}{'...' if len(last_user_msg) >= 200 else ''}\n"
            f"< RESPONSE: {response_summary}{'...' if len(response_summary) >= 200 else ''}\n"
            f"  Duration: {duration_str} | Tokens: {token_summary}\n"
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
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
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
            max_tokens=max_tokens,
            base_url=base_url,
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
