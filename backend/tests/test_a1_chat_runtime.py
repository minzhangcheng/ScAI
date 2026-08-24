"""A1 chat/figure runtime ownership and per-turn transport tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import backend.state as app_state
from backend.routes.chat_parts import endpoints, figures


@pytest.fixture(autouse=True)
def _clear_runtime_registries():
    endpoints.chat_tasks.clear()
    endpoints.chat_sandboxes.clear()
    endpoints.workflows.clear()
    app_state.report_workflows.clear()
    app_state.initializing_sessions.clear()
    app_state.initializing_session_owners.clear()
    yield
    endpoints.chat_tasks.clear()
    endpoints.chat_sandboxes.clear()
    endpoints.workflows.clear()
    app_state.report_workflows.clear()
    app_state.initializing_sessions.clear()
    app_state.initializing_session_owners.clear()


@pytest.mark.asyncio
async def test_concurrent_chat_runtime_has_one_exact_owner(monkeypatch):
    session_id = "a1-runtime"
    state = SimpleNamespace(chat_messages=[], user_id="u")
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_delegation(*_args):
        started.set()
        await release.wait()

    monkeypatch.setattr(endpoints, "_run_chat_delegation", run_delegation)
    monkeypatch.setattr(endpoints, "_async_save_session_under_lifecycle", AsyncMock())
    monkeypatch.setattr(endpoints, "assert_generation_writable", lambda *_args: None)

    owner = await endpoints._publish_chat_delegation(
        session_id, state, {"plan": "test"}, "user", [], []
    )
    await started.wait()
    with pytest.raises(HTTPException) as exc:
        await endpoints._publish_chat_delegation(
            session_id, state, {"plan": "second"}, "user", [], []
        )
    assert exc.value.status_code == 409
    assert endpoints.chat_tasks[session_id] is owner
    release.set()
    await owner

    # Workflow and standalone-report owners block the shared runtime, while a
    # delegation task may reserve the runtime on behalf of its own figure or
    # bioinformatics phase without replacing its outer task identity.
    for registry in (endpoints.workflows, app_state.report_workflows):
        registry[session_id] = object()
        assert await figures._reserve_chat_runtime(session_id) is None
        registry.pop(session_id, None)

    current = asyncio.current_task()
    endpoints.chat_tasks[session_id] = current
    reservation = await figures._reserve_chat_runtime(session_id)
    assert reservation is not None
    assert endpoints.chat_tasks[session_id] is current
    await figures._release_chat_runtime(session_id, reservation)
    assert endpoints.chat_tasks[session_id] is current
    endpoints.chat_tasks.pop(session_id, None)

    reserved = asyncio.Event()
    release_runtime = asyncio.Event()

    async def hold_runtime():
        owned = await figures._reserve_chat_runtime(session_id)
        assert owned is not None
        reserved.set()
        await release_runtime.wait()
        await figures._release_chat_runtime(session_id, owned)

    first_runtime = asyncio.create_task(hold_runtime())
    await reserved.wait()
    assert await figures._reserve_chat_runtime(session_id) is None
    release_runtime.set()
    await first_runtime
    assert session_id not in endpoints.chat_tasks
    assert session_id not in app_state.initializing_sessions


@pytest.mark.asyncio
async def test_cancelled_chat_runtime_releases_only_its_reservation(monkeypatch):
    session_id = "a1-cancel"
    state = SimpleNamespace(chat_messages=[], user_id="u")
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_delegation(*_args):
        started.set()
        await release.wait()

    monkeypatch.setattr(endpoints, "_run_chat_delegation", run_delegation)
    monkeypatch.setattr(endpoints, "_async_save_session_under_lifecycle", AsyncMock())
    monkeypatch.setattr(endpoints, "assert_generation_writable", lambda *_args: None)

    owner = await endpoints._publish_chat_delegation(
        session_id, state, {"plan": "test"}, "user", [], []
    )
    await started.wait()
    replacement = asyncio.create_task(asyncio.sleep(60))
    endpoints.chat_tasks[session_id] = replacement
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert endpoints.chat_tasks[session_id] is replacement
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement


@pytest.mark.asyncio
async def test_failed_sandbox_start_preserves_existing_notebook(monkeypatch):
    session_id = "a1-figure-failure"

    class _BrokenSandbox:
        km = None

        stop_calls = []

        def __init__(self, **_kwargs):
            pass

        async def start(self):
            raise RuntimeError("start failed")

        async def stop(self, **_kwargs):
            self.stop_calls.append(_kwargs)
            return None

    monkeypatch.setattr("backend.utils.sandbox.LocalJupyterSandbox", _BrokenSandbox)
    monkeypatch.setattr(figures, "resolve_config_for_user", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(figures, "get_session_states", lambda _sid: "/tmp/states")
    monkeypatch.setattr(figures, "get_session_workspace", lambda _sid: "/tmp/workspace")

    await figures._generate_figures_in_chat(session_id, SimpleNamespace(user_id="u"), ["plot"])

    assert session_id not in figures.chat_sandboxes
    assert session_id not in app_state.initializing_sessions
    # A failed start must not save the unloaded replacement notebook.
    # The instance is not retained, so inspect the class-level call record.
    assert _BrokenSandbox.stop_calls == [{"save": False}]


@pytest.mark.asyncio
async def test_per_turn_clients_close_on_success_and_failure(monkeypatch, tmp_path):
    class _Sandbox:
        km = None

        def __init__(self, **_kwargs):
            pass

        async def start(self):
            return {"session_restored": True}

        async def execute(self, _code, **_kwargs):
            return SimpleNamespace(success=True, figures=[], output="", error=None)

        async def stop(self, **_kwargs):
            return None

    clients = []

    def make_client(*_args, **_kwargs):
        client = MagicMock()
        if not clients:
            client.chat = AsyncMock(return_value=SimpleNamespace(content="```python\npass\n```"))
        else:
            client.chat = AsyncMock(side_effect=RuntimeError("chat failure"))
        client.close = AsyncMock()
        clients.append(client)
        return client

    monkeypatch.setattr("backend.utils.sandbox.LocalJupyterSandbox", _Sandbox)
    monkeypatch.setattr(figures, "resolve_config_for_user", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(figures, "get_session_states", lambda _sid: str(tmp_path / "states"))
    monkeypatch.setattr(figures, "get_session_workspace", lambda _sid: tmp_path)
    monkeypatch.setattr(
        figures,
        "AgentSettingsProvider",
        lambda *_args, **_kwargs: SimpleNamespace(get_model=lambda _name: "model"),
    )
    monkeypatch.setattr("backend.utils.agent_llm.get_agent_llm_client", make_client)

    await figures._generate_figures_in_chat("a1-success", SimpleNamespace(user_id="u"), ["plot"])
    await figures._generate_figures_in_chat("a1-failure", SimpleNamespace(user_id="u"), ["plot"])

    assert len(clients) == 2
    for client in clients:
        client.close.assert_awaited_once()
