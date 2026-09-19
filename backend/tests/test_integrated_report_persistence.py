"""Strict report publication bypasses throttling through both workflow owners."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.messaging import MessagingMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["headless", "websocket"])
async def test_running_report_publication_forces_persistence(monkeypatch, owner):
    from backend import state as state_mod
    from backend import workflow as workflow_module
    from backend.routes import websockets as websocket_routes
    from backend.services import session_lifecycle, session_store, workflow_runner

    session_id = str(uuid4())
    state = AnalysisState(
        id=session_id, user_id="report-owner", status=TaskStatus.RUNNING,
    )
    monkeypatch.setitem(state_mod.sessions, session_id, state)
    write = MagicMock()
    monkeypatch.setattr(session_store, "_externalize_and_write", write)
    monkeypatch.setattr(
        session_lifecycle, "assert_generation_writable", lambda *_args: None,
    )
    emissions = []

    class Workflow(MessagingMixin):
        def __init__(self, config, session_id, message_callback, state_callback):
            self.session_id = session_id
            self.state_callback = state_callback

            async def record_message(message):
                emissions.append((message.content["change_type"], write.call_count))

            self.message_callback = record_message

        async def run(self, current, _thread_id):
            # LLM intervention resumes as RUNNING, shortly after its last save.
            monkeypatch.setitem(state_mod.last_save_times, session_id, time.monotonic())
            await self._send_state_sync(current, "ordinary_update")
            current.integrated_report = "Persisted report"
            await self._send_state_sync(
                current, "integrated_report_published", raise_on_save_error=True,
            )
            return current

        async def close(self, **_kwargs):
            return True

    monkeypatch.setattr(workflow_module, "SingleCellAnalysisWorkflow", Workflow)
    config = SimpleNamespace(max_rounds=3)
    if owner == "headless":
        monkeypatch.setattr(
            workflow_runner, "resolve_config_for_user", AsyncMock(return_value=config),
        )
        workflow, current = await workflow_runner._build_workflow_for_session(session_id)
        await workflow.run(current, session_id)
    else:
        class WebSocket:
            async def receive_text(self):
                await asyncio.Event().wait()

            async def close(self, **_kwargs):
                return None

        monkeypatch.setattr(
            websocket_routes, "ws_authenticate_and_load",
            AsyncMock(return_value=(state.user_id, state)),
        )
        monkeypatch.setattr(
            websocket_routes, "resolve_config_for_user", AsyncMock(return_value=config),
        )
        monkeypatch.setattr(
            websocket_routes, "register_analysis_websocket",
            AsyncMock(return_value=object()),
        )
        monkeypatch.setattr(
            websocket_routes, "send_connection_json", AsyncMock(return_value=True),
        )
        monkeypatch.setattr(websocket_routes, "unregister_websocket", AsyncMock())
        monkeypatch.setattr(
            websocket_routes, "assert_generation_writable", lambda *_args: None,
        )
        monkeypatch.setattr(
            session_store, "assert_recovery_clear", lambda *_args, **_kwargs: None,
        )
        try:
            await websocket_routes.websocket_endpoint(WebSocket(), session_id)
        finally:
            for registry in (
                state_mod.workflows, state_mod._run_tasks, state_mod._monitor_tasks,
            ):
                registry.pop(session_id, None)
            state_mod.release_initializing_session(session_id)

    assert emissions == [
        ("ordinary_update", 0),
        ("integrated_report_published", 1),
    ]
    assert write.call_args_list[0].args[1]["integrated_report"] == "Persisted report"
