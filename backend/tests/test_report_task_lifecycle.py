"""Lifecycle coverage for the standalone detailed-report background task."""

import asyncio
import logging
import threading
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException

from backend.routes.reports_parts import html_views
from backend.utils.llm_client import UpstreamExhaustedError


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
async def test_route_ownership_failure_releases_acquired_pending_marker(
    monkeypatch,
):
    session_id = "ownership-failure"
    html_views.clear_generation_pending(session_id)
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(side_effect=HTTPException(status_code=403, detail="denied")),
    )
    background_tasks = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await html_views.generate_detailed_report(
            session_id,
            background_tasks,
            _request(),
        )

    assert exc_info.value.status_code == 403
    assert session_id not in html_views._report_generation_pending
    background_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_route_cancellation_releases_acquired_pending_marker(monkeypatch):
    session_id = "ownership-cancelled"
    html_views.clear_generation_pending(session_id)
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await html_views.generate_detailed_report(
            session_id,
            MagicMock(),
            _request(),
        )

    assert session_id not in html_views._report_generation_pending


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
async def test_route_handoff_failure_releases_pending_marker(monkeypatch):
    session_id = "handoff-failure"
    html_views.clear_generation_pending(session_id)
    state = _report_state(session_id)
    state.status.value = "completed"
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    background_tasks = MagicMock()

    def _fail_create_task(coro, **_kwargs):
        coro.close()
        raise RuntimeError("task create failed")

    monkeypatch.setattr(html_views.asyncio, "create_task", _fail_create_task)

    with pytest.raises(RuntimeError, match="task create failed"):
        await html_views.generate_detailed_report(
            session_id,
            background_tasks,
            _request(),
        )

    assert session_id not in html_views._report_generation_pending
    assert session_id not in html_views.detailed_report_tasks
    background_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("safe_point_state", ["pending", "acquired"])
async def test_route_refuses_generation_while_snapshot_safe_point_is_reserved(
    monkeypatch,
    safe_point_state,
):
    from backend.services.snapshot_v2.safe_point import (
        get_snapshot_safe_point_manager,
    )

    session_id = f"snapshot-reserved-{uuid.uuid4().hex}"
    state = _report_state(session_id)
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    safe_point = get_snapshot_safe_point_manager()
    owner_token = f"test:{uuid.uuid4().hex}"
    if safe_point_state == "pending":
        assert await safe_point.request_safe_point(session_id, owner_token)
    else:
        assert await safe_point.acquire(
            session_id,
            {"status": "capture"},
            owner_token=owner_token,
        )
    try:
        with pytest.raises(HTTPException) as exc_info:
            await html_views.generate_detailed_report(
                session_id,
                MagicMock(),
                _request(),
            )

        assert exc_info.value.status_code == 409
        assert "Snapshot capture or restore" in exc_info.value.detail
        assert session_id not in html_views.detailed_report_tasks
        assert session_id not in html_views._report_generation_pending
    finally:
        if safe_point_state == "pending":
            await safe_point.cancel_safe_point_request(session_id, owner_token)
        else:
            await safe_point.release(session_id, owner_token)


@pytest.mark.asyncio
async def test_missing_config_releases_report_pending_marker(monkeypatch, tmp_path):
    _patch_report_prelude(monkeypatch, tmp_path, config=None)
    session_id = "missing-config"
    token = await html_views._try_begin_generation(session_id)
    assert isinstance(token, str)

    await html_views._generate_report_task(
        session_id, _report_state(session_id), token,
    )

    assert session_id not in html_views._report_generation_pending
    error_message = html_views.send_report_json.await_args.args[1]
    assert error_message["type"] == "error"


@pytest.mark.asyncio
async def test_config_resolution_exception_releases_report_pending_marker(
    monkeypatch,
):
    session_id = "config-error"
    token = await html_views._try_begin_generation(session_id)
    assert isinstance(token, str)
    monkeypatch.setattr(
        html_views,
        "resolve_config_for_user",
        AsyncMock(side_effect=RuntimeError("config lookup failed")),
    )

    with pytest.raises(RuntimeError, match="config lookup failed"):
        await html_views._generate_report_task(
            session_id, _report_state(session_id), token,
        )

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
async def test_report_passes_matching_pi_client_to_supervisor(monkeypatch, tmp_path):
    config = MagicMock()
    config.api.api_key = "global-key"
    config.api.llm_base_url = "https://global.example/v1"
    config.api.proxy_url = ""
    _patch_report_prelude(monkeypatch, tmp_path, config=config)
    monkeypatch.setattr(
        "backend.config.get_session_workspace",
        lambda session_id: tmp_path,
    )

    global_client = MagicMock()
    global_client.close = AsyncMock()
    report_client = MagicMock(name="pi_report_client")
    provider = MagicMock()
    provider.get_llm_client.return_value = report_client
    provider.get_model.return_value = "pi-model"
    provider.cleanup = AsyncMock()
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(return_value={"ok": True})

    client_factory = MagicMock(return_value=global_client)
    monkeypatch.setattr(html_views, "OpenAICompatibleClient", client_factory)
    monkeypatch.setattr(
        html_views,
        "AgentSettingsProvider",
        MagicMock(return_value=provider),
    )
    monkeypatch.setattr(
        "backend.agents.supervisor_agent.SupervisorAgent",
        MagicMock(return_value=supervisor),
    )

    session_id = "matching-pi-client"
    token = await html_views._try_begin_generation(session_id)
    assert isinstance(token, str)
    await html_views._generate_report_task(
        session_id, _report_state(session_id), token,
    )

    client_factory.assert_called_once_with(
        api_key="global-key",
        base_url="https://global.example/v1",
        proxy_url="",
        session_id=session_id,
        agent_name=html_views.PI_AGENT_NAME,
    )
    kwargs = supervisor.generate_detailed_report.await_args.kwargs
    assert kwargs["report_model"] == "pi-model"
    assert kwargs["report_llm_client"] is report_client
    provider.cleanup.assert_awaited_once_with()
    global_client.close.assert_awaited_once_with()
    assert session_id not in html_views._report_generation_pending


@pytest.mark.asyncio
async def test_report_retry_refreshes_pi_model_and_client_as_one_pair(
    monkeypatch,
    tmp_path,
):
    config = MagicMock()
    config.api.api_key = "global-key"
    config.api.llm_base_url = "https://global.example/v1"
    config.api.proxy_url = ""
    _patch_report_prelude(monkeypatch, tmp_path, config=config)
    monkeypatch.setattr(
        "backend.config.get_session_workspace",
        lambda session_id: tmp_path,
    )
    monkeypatch.setattr(html_views.asyncio, "sleep", AsyncMock())

    global_client = MagicMock()
    global_client.close = AsyncMock()
    old_client = MagicMock(name="old_pi_client")
    new_client = MagicMock(name="new_pi_client")
    provider = MagicMock()
    provider.get_llm_client.side_effect = [old_client, new_client]
    provider.get_model.side_effect = ["old-model", "new-model"]
    provider.cleanup = AsyncMock()
    supervisor = MagicMock()
    supervisor.generate_detailed_report = AsyncMock(
        side_effect=[
            UpstreamExhaustedError(
                "upstream exhausted",
                agent_name="pi",
                model="old-model",
                llm_server="https://old.example/v1",
            ),
            {"ok": True},
        ]
    )

    monkeypatch.setattr(
        html_views,
        "OpenAICompatibleClient",
        MagicMock(return_value=global_client),
    )
    monkeypatch.setattr(
        html_views,
        "AgentSettingsProvider",
        MagicMock(return_value=provider),
    )
    monkeypatch.setattr(
        "backend.agents.supervisor_agent.SupervisorAgent",
        MagicMock(return_value=supervisor),
    )

    await html_views._generate_report_task(
        "report-retry", _report_state("report-retry"),
    )

    calls = supervisor.generate_detailed_report.await_args_list
    assert calls[0].kwargs["report_model"] == "old-model"
    assert calls[0].kwargs["report_llm_client"] is old_client
    assert calls[1].kwargs["report_model"] == "new-model"
    assert calls[1].kwargs["report_llm_client"] is new_client
    provider.clear_client_cache.assert_called_once_with()
    provider.cleanup.assert_awaited_once_with()
    global_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_report_resource_cleanup_retries_each_owner_independently():
    provider = MagicMock()
    provider.cleanup = AsyncMock(
        side_effect=[RuntimeError("provider close failed"), None]
    )
    llm_client = MagicMock()
    llm_client.close = AsyncMock(
        side_effect=[RuntimeError("global close failed"), None]
    )

    await html_views._cleanup_report_llm_resources(provider, llm_client)

    assert provider.cleanup.await_count == 2
    assert llm_client.close.await_count == 2


@pytest.mark.asyncio
async def test_report_cleanup_preserves_cancellation_after_draining_resources(
    caplog,
):
    provider = MagicMock()
    provider.cleanup = AsyncMock(side_effect=[asyncio.CancelledError(), None])
    llm_client = MagicMock()
    llm_client.close = AsyncMock()

    with caplog.at_level(logging.WARNING), pytest.raises(asyncio.CancelledError):
        await html_views._cleanup_report_llm_resources(provider, llm_client)

    assert provider.cleanup.await_count == 2
    llm_client.close.assert_awaited_once_with()
    assert "retrying owned-resource cleanup" in caplog.text


@pytest.mark.asyncio
async def test_owned_thread_failure_cannot_replace_pending_cancellation():
    entered = threading.Event()
    release = threading.Event()

    def _failing_owned_work():
        entered.set()
        assert release.wait(timeout=2)
        raise RuntimeError("archive write failed")

    owner = asyncio.create_task(
        html_views._await_owned_thread_call(_failing_owned_work)
    )
    try:
        assert await _wait_thread_event(entered, 2)
        owner.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(owner, timeout=2)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ["analysis", "standalone_report"])
async def test_route_refuses_other_live_workflow_owner(monkeypatch, owner_kind):
    from backend.services import workflow_runner
    from backend.state import workflows

    session_id = f"detailed-owner-conflict-{owner_kind}-{uuid.uuid4().hex}"
    state = _report_state(session_id)
    monkeypatch.setattr(
        html_views,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    release = asyncio.Event()
    owner_task = None
    if owner_kind == "analysis":
        workflows[session_id] = object()
    else:
        owner_task = asyncio.create_task(release.wait())
        workflow_runner._report_tasks[session_id] = owner_task
    try:
        with pytest.raises(HTTPException) as exc_info:
            await html_views.generate_detailed_report(
                session_id, MagicMock(), _request(),
            )
        assert exc_info.value.status_code == 409
        assert session_id not in html_views.detailed_report_tasks
        assert session_id not in html_views._report_generation_pending
    finally:
        workflows.pop(session_id, None)
        release.set()
        if owner_task is not None:
            await owner_task
            if workflow_runner._report_tasks.get(session_id) is owner_task:
                workflow_runner._report_tasks.pop(session_id, None)
