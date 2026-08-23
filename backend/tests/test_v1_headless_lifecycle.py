"""Focused V1 proof for headless execution and durable state continuity."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, TaskStatus, WebSocketMessage
from backend.state import (
    _run_tasks,
    detailed_report_tasks,
    initializing_sessions,
    initializing_session_owners,
    report_workflows,
    sessions,
    workflows,
)


class _Request:
    def __init__(self, user_id: str, payload: dict | None = None):
        self.state = SimpleNamespace(user_id=user_id)
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_v1_headless_run_reaches_node_and_survives_process_reload(
    tmp_path: Path, monkeypatch
):
    from backend.config import _config as config_module
    from backend.routes import v1
    from backend.services import session_store, workflow_runner
    from backend.services import session_lifecycle
    import backend.workflow as workflow_module

    session_root = tmp_path / "sessions"
    session_root.mkdir()
    monkeypatch.setattr(config_module, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_root)

    config = {
        "schema_version": "1.0",
        "analysis": {
            "name": "headless lifecycle",
            "initial_instruction": "exercise the first graph node",
            "round_wait_seconds": 0,
        },
        "api_config": {
            "default_llm": {
                "llm_base_url": "",
                "api_key": "",
                "model": "",
            },
            "literature": {"mode": "disabled"},
        },
        "data": {"groups": [], "files": [], "indicators": []},
        "agents": {},
    }
    request = _Request("u1", config)
    created = await v1.create_analysis(request)
    session_id = created["job_id"]
    first_node = asyncio.Event()

    async def _resolve_config(_user_id):
        return SimpleNamespace(max_rounds=3)

    class _FirstNodeWorkflow:
        def __init__(self, config, session_id, message_callback=None, state_callback=None):
            self.message_callback = message_callback
            self.state_callback = state_callback

        async def run(self, initial_state, thread_id):
            first_node.set()
            await self.message_callback(
                WebSocketMessage(
                    type="status",
                    content={"round": 1, "binary": b"not durable"},
                )
            )
            graph_state = initial_state.model_copy(deep=True)
            graph_state.current_round = 2
            graph_state.status = TaskStatus.WAITING_INPUT
            assert await self.state_callback(graph_state) is True
            return graph_state

        async def resume(self, thread_id, analysis_state):
            return await self.run(analysis_state, thread_id)

        async def close(self):
            return None

    monkeypatch.setattr(v1, "resolve_config_for_user", _resolve_config)
    monkeypatch.setattr(workflow_runner, "resolve_config_for_user", _resolve_config)
    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", _FirstNodeWorkflow)

    started = await v1.start_analysis(session_id, _Request("u1"))
    runner_task = _run_tasks.get(session_id)
    try:
        assert started == {"job_id": session_id, "status": "running"}
        assert runner_task is not None

        await asyncio.wait_for(first_node.wait(), timeout=2)
        await asyncio.wait_for(runner_task, timeout=2)

        live_status = await v1.get_analysis(session_id, _Request("u1"))
        live_state = sessions[session_id]
        live_history = [
            message.model_dump(mode="json")
            for message in live_state.realtime_messages
        ]
        assert live_status["status"] == TaskStatus.WAITING_INPUT.value
        assert live_status["current_round"] == 2
        assert len(live_history) == 1
        assert live_history[0]["content"]["binary"] == "[image]"

        sessions.clear()
        workflows.clear()
        report_workflows.clear()
        initializing_sessions.clear()
        initializing_session_owners.clear()
        _run_tasks.clear()
        detailed_report_tasks.clear()

        reloaded_status = await v1.get_analysis(session_id, _Request("u1"))
        reloaded = sessions[session_id]
        reloaded_history = [
            message.model_dump(mode="json") for message in reloaded.realtime_messages
        ]
        assert reloaded_status["status"] == live_status["status"]
        assert reloaded_status["current_round"] == live_status["current_round"]
        assert reloaded_history == live_history
    finally:
        if runner_task is not None and not runner_task.done():
            runner_task.cancel()
        if runner_task is not None:
            await asyncio.gather(runner_task, return_exceptions=True)
        sessions.clear()
        workflows.clear()
        report_workflows.clear()
        initializing_sessions.clear()
        initializing_session_owners.clear()
        _run_tasks.clear()
        detailed_report_tasks.clear()
