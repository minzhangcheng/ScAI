"""Lifecycle coverage for the standalone detailed-report background task."""

import asyncio
import threading
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.routes.reports_parts import html_views


_CREATED_SESSION_IDS: set[str] = set()


def _report_state(session_id: str):
    state = MagicMock()
    state.id = session_id
    state.user_id = "user-1"
    state.lifecycle_generation = 1
    state.get_files_by_section.return_value = []
    state.status = MagicMock()
    state.status.value = "completed"
    state.round_results = []
    state.chat_messages = []
    state.restore_branch_id = None
    html_views.sessions[session_id] = state
    _CREATED_SESSION_IDS.add(session_id)
    return state


@pytest_asyncio.fixture(autouse=True)
async def _isolate_report_producers(monkeypatch):
    monkeypatch.setattr(
        html_views, "assert_generation_writable", lambda *_args: None,
    )
    yield
    tasks = [
        task
        for session_id, task in list(html_views.detailed_report_tasks.items())
        if session_id in _CREATED_SESSION_IDS
    ]
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for session_id in list(_CREATED_SESSION_IDS):
        if html_views.detailed_report_tasks.get(session_id) in tasks:
            html_views.detailed_report_tasks.pop(session_id, None)
        html_views.sessions.pop(session_id, None)
        html_views.clear_generation_pending(session_id)
    _CREATED_SESSION_IDS.clear()


def _patch_report_prelude(monkeypatch, tmp_path, *, config):
    monkeypatch.setattr(
        html_views,
        "resolve_config_for_user",
        AsyncMock(return_value=config),
    )
    monkeypatch.setattr(
        html_views,
        "resolve_report_target_lineage",
        lambda branch_id=None: "main",
    )
    monkeypatch.setattr(
        html_views,
        "resolve_report_output_dir",
        lambda session_id, branch_id=None: tmp_path,
    )
    monkeypatch.setattr(
        html_views,
        "build_lineage_report_state",
        lambda state, lineage: state,
    )
    monkeypatch.setattr(html_views, "send_report_json", AsyncMock())


def _request():
    request = MagicMock()
    request.state.user_id = "user-1"
    return request


async def _wait_thread_event(event: threading.Event, timeout: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set():
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.005)
    return True


@pytest.mark.asyncio
async def test_route_transfers_pending_marker_only_after_background_handoff(
    monkeypatch,
):
    session_id = "successful-handoff"
    html_views.clear_generation_pending(session_id)
    state = _report_state(session_id)
    state.status.value = "completed"
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    background_tasks = MagicMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_report(_session_id, _state):
        started.set()
        await release.wait()

    monkeypatch.setattr(
        html_views, "_run_generate_report_task", _blocked_report,
    )

    result = await html_views.generate_detailed_report(
        session_id,
        background_tasks,
        _request(),
    )

    assert result["status"] == "started"
    await started.wait()
    assert session_id in html_views._report_generation_pending
    task = html_views.detailed_report_tasks[session_id]
    assert task.get_name() == f"detailed-report:{session_id}"
    background_tasks.add_task.assert_not_called()

    release.set()
    await task
    await asyncio.sleep(0)
    assert session_id not in html_views.detailed_report_tasks
    assert session_id not in html_views._report_generation_pending


@pytest.mark.asyncio
async def test_report_setup_exception_closes_provider_and_global_client(
    monkeypatch,
    tmp_path,
):
    _patch_report_prelude(monkeypatch, tmp_path, config=MagicMock())
    session_id = "setup-failure"
    token = await html_views._try_begin_generation(session_id)
    assert isinstance(token, str)

    llm_client = MagicMock()
    llm_client.close = AsyncMock()
    provider = MagicMock()
    provider.cleanup = AsyncMock()
    monkeypatch.setattr(
        html_views,
        "OpenAICompatibleClient",
        MagicMock(return_value=llm_client),
    )
    monkeypatch.setattr(
        html_views,
        "AgentSettingsProvider",
        MagicMock(return_value=provider),
    )
    monkeypatch.setattr(
        "backend.config.get_session_workspace",
        lambda session_id: tmp_path,
    )
    monkeypatch.setattr(
        "backend.agents.supervisor_agent.SupervisorAgent",
        MagicMock(side_effect=RuntimeError("supervisor setup failed")),
    )

    with pytest.raises(RuntimeError, match="supervisor setup failed"):
        await html_views._generate_report_task(
            session_id, _report_state(session_id), token,
        )

    assert session_id not in html_views._report_generation_pending
    provider.cleanup.assert_awaited_once_with()
    llm_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_owned_thread_drains_after_repeated_cancellation_without_spinning():
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _successful_owned_work():
        entered.set()
        assert release.wait(timeout=2)
        finished.set()
        return "done"

    owner = asyncio.create_task(
        html_views._await_owned_thread_call(_successful_owned_work)
    )
    try:
        assert await _wait_thread_event(entered, 2)
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner, timeout=2)
        assert finished.is_set()
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_old_detailed_report_callback_cannot_clear_replacement_owner(
    monkeypatch,
):
    session_id = f"detailed-replacement-{uuid.uuid4().hex}"
    state = _report_state(session_id)
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    release_new = asyncio.Event()

    async def _old_report(_session_id, _state):
        old_started.set()
        await release_old.wait()

    monkeypatch.setattr(
        html_views, "_run_generate_report_task", _old_report,
    )
    await html_views.generate_detailed_report(
        session_id, MagicMock(), _request(),
    )
    await old_started.wait()
    old_task = html_views.detailed_report_tasks[session_id]
    old_token = html_views._report_generation_pending[session_id]
    replacement = asyncio.create_task(release_new.wait())
    html_views.detailed_report_tasks[session_id] = replacement
    html_views.clear_generation_pending(session_id)
    replacement_token = await html_views._try_begin_generation(session_id)
    assert isinstance(replacement_token, str)
    assert replacement_token != old_token
    try:
        release_old.set()
        await old_task
        await asyncio.sleep(0)

        assert html_views.detailed_report_tasks.get(session_id) is replacement
        assert session_id in html_views._report_generation_pending
    finally:
        release_old.set()
        release_new.set()
        await asyncio.gather(old_task, replacement, return_exceptions=True)
        if html_views.detailed_report_tasks.get(session_id) is replacement:
            html_views.detailed_report_tasks.pop(session_id, None)
        html_views.clear_generation_pending(session_id, replacement_token)
