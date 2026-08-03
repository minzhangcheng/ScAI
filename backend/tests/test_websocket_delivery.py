import asyncio
import threading

import pytest

from backend.state import report_websockets, websockets
from backend.utils.websocket_delivery import (
    close_analysis_websocket,
    register_analysis_websocket,
    register_report_websocket,
    reset_websocket_delivery_for_tests,
    send_analysis_json,
    send_connection_json,
    send_report_json,
    unregister_websocket,
)


class _Socket:
    def __init__(self, *, fail=False, block=False):
        self.fail = fail
        self.block = block
        self.sent = []
        self.closed = []
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send_json(self, payload):
        self.send_started.set()
        if self.block:
            await self.release_send.wait()
        if self.fail:
            raise RuntimeError("websocket.send after websocket.close")
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed.append(code)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_websocket_delivery_for_tests()
    websockets.clear()
    report_websockets.clear()
    yield
    reset_websocket_delivery_for_tests()
    websockets.clear()
    report_websockets.clear()


@pytest.mark.asyncio
async def test_old_handler_cleanup_never_removes_new_generation():
    old_socket = _Socket()
    new_socket = _Socket()
    old = await register_analysis_websocket("session", old_socket)
    new = await register_analysis_websocket("session", new_socket)

    await unregister_websocket(old, close=True, code=4000)
    assert old_socket.closed == [1000]
    assert websockets["session"] is new_socket
    assert await send_analysis_json("session", {"generation": "new"}) is True
    assert new_socket.sent == [{"generation": "new"}]

    await unregister_websocket(new)
    assert "session" not in websockets


@pytest.mark.asyncio
async def test_send_failure_does_not_mutate_caller_durable_buffer():
    socket = _Socket(fail=True)
    await register_analysis_websocket("session", socket)
    durable_buffer = [{"type": "status", "content": "persisted first"}]

    assert await send_analysis_json("session", durable_buffer[-1]) is False
    assert durable_buffer == [{"type": "status", "content": "persisted first"}]
    assert "session" not in websockets


@pytest.mark.asyncio
async def test_report_channel_uses_its_own_generation_registry():
    analysis_socket = _Socket()
    report_socket = _Socket()
    await register_analysis_websocket("session", analysis_socket)
    await register_report_websocket("session", report_socket)

    assert await send_report_json("session", {"type": "progress"}) is True
    assert report_socket.sent == [{"type": "progress"}]
    assert analysis_socket.sent == []


@pytest.mark.asyncio
async def test_send_and_close_are_serialized_for_one_connection():
    socket = _Socket(block=True)
    connection = await register_analysis_websocket("session", socket)

    send_task = asyncio.create_task(
        send_connection_json(connection, {"type": "progress"})
    )
    await socket.send_started.wait()
    close_task = asyncio.create_task(
        close_analysis_websocket("session", code=4000)
    )
    await asyncio.sleep(0)
    assert socket.closed == []

    socket.release_send.set()
    assert await send_task is True
    assert await close_task is True
    assert socket.sent == [{"type": "progress"}]
    assert socket.closed == [4000]
    assert "session" not in websockets


@pytest.mark.asyncio
async def test_cancelled_send_releases_current_registry_slot():
    socket = _Socket(block=True)
    connection = await register_analysis_websocket("session", socket)
    send_task = asyncio.create_task(
        send_connection_json(connection, {"type": "progress"})
    )
    await socket.send_started.wait()

    send_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await send_task

    assert "session" not in websockets


@pytest.mark.asyncio
async def test_replacement_publishes_without_waiting_for_inflight_old_send():
    old_socket = _Socket(block=True)
    new_socket = _Socket()
    old = await register_analysis_websocket("session", old_socket)

    old_send = asyncio.create_task(
        send_connection_json(old, {"generation": "old-before-replace"})
    )
    await old_socket.send_started.wait()
    replacement = asyncio.create_task(
        register_analysis_websocket("session", new_socket)
    )
    new = await asyncio.wait_for(replacement, timeout=0.2)
    assert websockets["session"] is new_socket

    new_send = asyncio.create_task(
        send_connection_json(new, {"generation": "new"})
    )
    await asyncio.sleep(0)
    assert not new_send.done(), "new delivery overtook the old generation"

    old_socket.release_send.set()
    assert await old_send is False
    assert await new_send is True
    assert old_socket.closed == [1000]
    assert websockets["session"] is new_socket

    assert await send_connection_json(old, {"generation": "stale"}) is False
    assert old_socket.sent == [{"generation": "old-before-replace"}]
    assert new_socket.sent == [{"generation": "new"}]


@pytest.mark.asyncio
async def test_send_deadline_releases_replacement_delivery(monkeypatch):
    from backend.utils import websocket_delivery

    monkeypatch.setattr(
        websocket_delivery, "_WEBSOCKET_SEND_TIMEOUT_SECONDS", 0.01,
    )
    old_socket = _Socket(block=True)
    old = await register_analysis_websocket("send-timeout", old_socket)
    old_send = asyncio.create_task(
        send_connection_json(old, {"generation": "old"})
    )
    await old_socket.send_started.wait()

    new_socket = _Socket()
    new = await asyncio.wait_for(
        register_analysis_websocket("send-timeout", new_socket),
        timeout=0.2,
    )
    new_send = asyncio.create_task(
        send_connection_json(new, {"generation": "new"})
    )

    assert await asyncio.wait_for(old_send, timeout=0.2) is False
    assert await asyncio.wait_for(new_send, timeout=0.2) is True
    assert old_socket.sent == []
    assert old_socket.closed == [1011]
    assert new_socket.sent == [{"generation": "new"}]


@pytest.mark.asyncio
async def test_replacement_close_deadline_does_not_block_registration(monkeypatch):
    from backend.utils import websocket_delivery

    monkeypatch.setattr(
        websocket_delivery, "_WEBSOCKET_CLOSE_TIMEOUT_SECONDS", 0.01,
    )
    close_started = asyncio.Event()

    class BlockingCloseSocket(_Socket):
        async def close(self, code=1000):
            close_started.set()
            await asyncio.Event().wait()

    old = await register_analysis_websocket(
        "close-timeout", BlockingCloseSocket(),
    )
    new_socket = _Socket()
    new = await asyncio.wait_for(
        register_analysis_websocket("close-timeout", new_socket),
        timeout=0.2,
    )

    await asyncio.wait_for(close_started.wait(), timeout=0.2)
    for _ in range(50):
        if old.socket_closed:
            break
        await asyncio.sleep(0.001)

    assert old.socket_closed is True
    assert websockets["close-timeout"] is new_socket
    assert await send_connection_json(new, {"generation": "new"}) is True


@pytest.mark.asyncio
async def test_delayed_old_handler_cleanup_preserves_new_runtime_generation():
    from backend.routes.websockets import _cleanup_owned_workflow_handler
    from backend.state import (
        _monitor_tasks,
        _run_tasks,
        initializing_sessions,
        workflows,
    )

    class _Workflow:
        def __init__(self):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    async def _wait_forever():
        await asyncio.Event().wait()

    session_id = "runtime-generation"
    old_workflow = _Workflow()
    new_workflow = _Workflow()
    old_run = asyncio.create_task(asyncio.sleep(0))
    await old_run
    old_monitor = asyncio.create_task(_wait_forever())
    new_run = asyncio.create_task(_wait_forever())
    new_monitor = asyncio.create_task(_wait_forever())
    old_connection = await register_analysis_websocket(session_id, _Socket())
    new_socket = _Socket()
    await register_analysis_websocket(session_id, new_socket)

    workflows[session_id] = new_workflow
    _run_tasks[session_id] = new_run
    _monitor_tasks[session_id] = new_monitor
    initializing_sessions.add(session_id)
    try:
        await _cleanup_owned_workflow_handler(
            session_id,
            workflow=old_workflow,
            run_task=old_run,
            monitor_task=old_monitor,
            connection_ref=old_connection,
        )

        assert workflows[session_id] is new_workflow
        assert _run_tasks[session_id] is new_run
        assert _monitor_tasks[session_id] is new_monitor
        assert session_id in initializing_sessions
        assert old_workflow.close_calls == 1
        assert new_workflow.close_calls == 0
        assert not new_run.cancelled()
        assert not new_monitor.cancelled()
        assert await send_analysis_json(session_id, {"generation": "new"}) is True
        assert new_socket.sent == [{"generation": "new"}]
    finally:
        workflows.pop(session_id, None)
        _run_tasks.pop(session_id, None)
        _monitor_tasks.pop(session_id, None)
        initializing_sessions.discard(session_id)
        new_run.cancel()
        new_monitor.cancel()
        await asyncio.gather(new_run, new_monitor, return_exceptions=True)


@pytest.mark.asyncio
async def test_activity_classifier_observes_unfinished_analysis_task():
    from backend.routes.websockets import _session_workflow_activity
    from backend.state import _run_tasks

    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    _run_tasks["run-task-only"] = task
    try:
        assert _session_workflow_activity("run-task-only") == (
            True,
            False,
            False,
            False,
        )
    finally:
        release.set()
        await task
        if _run_tasks.get("run-task-only") is task:
            _run_tasks.pop("run-task-only", None)


def test_registry_contention_is_safe_across_separate_event_loops():
    async def exercise(label: str):
        socket = _Socket(block=True)
        connection = await register_analysis_websocket("cross-loop", socket)

        first_send = asyncio.create_task(
            send_connection_json(connection, {"loop": label, "send": 1})
        )
        await socket.send_started.wait()
        second_send = asyncio.create_task(
            send_connection_json(connection, {"loop": label, "send": 2})
        )
        await asyncio.sleep(0)
        assert not second_send.done(), "the connection lock did not contend"

        socket.release_send.set()
        assert await first_send is True
        assert await second_send is True
        assert socket.sent == [
            {"loop": label, "send": 1},
            {"loop": label, "send": 2},
        ]
        return connection, socket

    # Do not reset between loops: the first connection and its genuinely bound
    # lock stay alive while the second loop replaces the same session ID.
    first_connection, first_socket = asyncio.run(exercise("first-loop"))
    second_connection, second_socket = asyncio.run(exercise("second-loop"))

    assert first_connection.generation < second_connection.generation
    assert first_socket.closed == []
    assert websockets["cross-loop"] is second_socket

    durable_buffer = [{"loop": "foreign-cleanup"}]
    assert asyncio.run(
        send_connection_json(first_connection, durable_buffer[-1])
    ) is False
    asyncio.run(unregister_websocket(first_connection, close=True, code=4000))
    assert durable_buffer == [{"loop": "foreign-cleanup"}]
    assert websockets["cross-loop"] is second_socket

    reset_websocket_delivery_for_tests()


@pytest.mark.parametrize("old_fails", [False, True])
def test_cross_loop_replacement_orders_delivery_and_releases_lease(old_fails):
    started = threading.Event()
    release = threading.Event()
    replacement_registered = threading.Event()
    events: list[str] = []
    results: dict[str, object] = {}

    class OldSocket:
        async def send_json(self, _payload):
            events.append("old-start")
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            if old_fails:
                events.append("old-error")
                raise RuntimeError("old socket disconnected")
            events.append("old-finish")

        async def close(self, code=1000):
            events.append(f"old-close-{code}")

    class NewSocket:
        async def send_json(self, _payload):
            events.append("new-send")

        async def close(self, code=1000):
            events.append(f"new-close-{code}")

    def old_loop():
        async def run():
            connection = await register_analysis_websocket("cross-loop", OldSocket())
            results["old_send"] = await send_connection_json(
                connection, {"generation": "old"}
            )

        asyncio.run(run())

    def new_loop():
        async def run():
            connection = await register_analysis_websocket("cross-loop", NewSocket())
            replacement_registered.set()
            results["new_send"] = await send_connection_json(
                connection, {"generation": "new"}
            )

        asyncio.run(run())

    old_thread = threading.Thread(target=old_loop)
    old_thread.start()
    assert started.wait(3)
    new_thread = threading.Thread(target=new_loop)
    new_thread.start()

    # Registration itself is non-blocking; only delivery waits behind the
    # in-flight old generation.
    assert replacement_registered.wait(1)
    assert events == ["old-start"]
    release.set()
    old_thread.join(3)
    new_thread.join(3)

    assert not old_thread.is_alive()
    assert not new_thread.is_alive(), (events, results)
    assert results == {"old_send": False, "new_send": True}
    expected_old_terminal = "old-error" if old_fails else "old-finish"
    assert events.index(expected_old_terminal) < events.index("new-send")


def test_old_loop_workflow_send_does_not_route_to_new_connection_owner_loop():
    old_registered = threading.Event()
    replacement_registered = threading.Event()
    old_finished = threading.Event()
    results: dict[str, object] = {}

    class OldSocket:
        def __init__(self):
            self.closed = []

        async def send_json(self, _payload):
            raise AssertionError("stale socket received a post-replacement payload")

        async def close(self, code=1000):
            self.closed.append(code)

    class NewSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

        async def close(self, code=1000):
            pass

    old_socket = OldSocket()
    new_socket = NewSocket()

    async def wait_for(signal: threading.Event):
        while not signal.is_set():
            await asyncio.sleep(0.001)

    def old_loop():
        async def run():
            old = await register_analysis_websocket("owner-route", old_socket)
            old_registered.set()
            await wait_for(replacement_registered)
            results["sent"] = await send_connection_json(
                old, {"generation": "replacement"}
            )
            old_finished.set()

        asyncio.run(run())

    def new_loop():
        async def run():
            await register_analysis_websocket("owner-route", new_socket)
            replacement_registered.set()
            await wait_for(old_finished)

        asyncio.run(run())

    old_thread = threading.Thread(target=old_loop)
    old_thread.start()
    assert old_registered.wait(3)
    new_thread = threading.Thread(target=new_loop)
    new_thread.start()
    old_thread.join(3)
    new_thread.join(3)

    assert not old_thread.is_alive()
    assert not new_thread.is_alive()
    assert results == {"sent": False}
    assert old_socket.closed == [1000]
    assert new_socket.sent == []
