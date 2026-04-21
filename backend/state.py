"""Global state for ScAI backend sessions, workflows, and tasks.

Extracted from main.py to break circular imports.
"""
from typing import Dict, Any

# In-memory session store: session_id -> AnalysisState
sessions: Dict[str, Any] = {}
workflows: Dict[str, Any] = {}

# Background run tasks: session_id -> asyncio.Task
_run_tasks: Dict[str, Any] = {}
