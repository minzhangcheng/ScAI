import asyncio
import copy
import math
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pytest

from backend.models.schemas import WebSocketMessage
from backend.routes import websockets as websocket_routes
from backend.state import report_websockets, websockets
from backend.constants import STRIPPED_FIGURE_PLACEHOLDER
from backend.utils.websocket_delivery import (
    close_analysis_websocket,
    register_analysis_websocket,
    reset_websocket_delivery_for_tests,
    serialize_realtime_message,
    send_analysis_json,
    send_connection_json,
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
async def test_durable_history_preserves_long_scientific_text_in_media_context():
    image = "iVBORw0KGgo" + ("A" * 2048)
    scientific_text = "marker " + ("gene-expression evidence; " * 80)
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    message = WebSocketMessage(
        type="graphic_abstract",
        content={
            "graphic_abstract": {
                "caption": scientific_text,
                "base64_summary": scientific_text,
                "summary": cycle,
                "image_base64": image,
            }
        },
    )
    original_content = copy.deepcopy(message.content)

    durable = serialize_realtime_message(message, strip_binary=True)
    live = serialize_realtime_message(message, strip_binary=False)
    history = websocket_routes._serialize_realtime_messages([message])[0]

    for payload in (durable, history):
        graphic = payload["content"]["graphic_abstract"]
        assert graphic["caption"] == scientific_text
        assert graphic["base64_summary"] == scientific_text
        assert graphic["image_base64"] == STRIPPED_FIGURE_PLACEHOLDER
        assert graphic["summary"] == {"self": "[cycle]"}
    graphic_live = live["content"]["graphic_abstract"]
    assert graphic_live["caption"] == scientific_text
    assert graphic_live["base64_summary"] == scientific_text
    assert graphic_live["image_base64"] == image
    assert graphic_live["summary"] == {"self": "[cycle]"}
    assert message.content["graphic_abstract"]["summary"] is cycle
    assert message.content["graphic_abstract"]["caption"] == original_content[
        "graphic_abstract"
    ]["caption"]
    assert message.content["graphic_abstract"]["base64_summary"] == original_content[
        "graphic_abstract"
    ]["base64_summary"]
    assert message.content["graphic_abstract"]["image_base64"] == original_content[
        "graphic_abstract"
    ]["image_base64"]


@pytest.mark.asyncio
async def test_live_delivery_normalizes_non_json_values_without_mutating_payload():
    class Marker(Enum):
        READY = "ready"

    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    payload = {
        "timestamp": datetime(2026, 8, 23, 1, 2, 3, tzinfo=timezone.utc),
        "marker": Marker.READY,
        "path": Path("nested/file.txt"),
        "set": {"b", "a"},
        "nan": math.nan,
        "raw": memoryview(b"raw"),
        "cycle": cycle,
    }
    original = {
        key: value
        for key, value in payload.items()
        if key != "cycle"
    }
    original["cycle"] = cycle
    socket = _Socket()
    connection = await register_analysis_websocket("json-values", socket)

    assert await send_connection_json(connection, payload) is True

    normalized = socket.sent[0]
    assert normalized["timestamp"] == "2026-08-23T01:02:03+00:00"
    assert normalized["marker"] == "ready"
    assert normalized["path"] == "nested/file.txt"
    assert normalized["set"] == ["a", "b"]
    assert normalized["nan"] is None
    assert normalized["raw"] == STRIPPED_FIGURE_PLACEHOLDER
    assert "cycle" in normalized
    assert payload.keys() == original.keys()
    assert payload["timestamp"] == original["timestamp"]
    assert payload["raw"].tobytes() == original["raw"].tobytes()
    assert payload["cycle"] is cycle


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
async def test_send_deadline_is_hard_when_transport_suppresses_cancellation(
    monkeypatch,
):
    from backend.utils import websocket_delivery

    monkeypatch.setattr(
        websocket_delivery, "_WEBSOCKET_SEND_TIMEOUT_SECONDS", 0.01,
    )

    class CancellationResistantSocket(_Socket):
        async def send_json(self, payload):
            self.send_started.set()
            try:
                await self.release_send.wait()
            except asyncio.CancelledError:
                await self.release_send.wait()
            self.sent.append(payload)

    old_socket = CancellationResistantSocket()
    old = await register_analysis_websocket("hard-send-timeout", old_socket)
    old_send = asyncio.create_task(
        send_connection_json(old, {"generation": "old"})
    )
    await old_socket.send_started.wait()
    new_socket = _Socket()
    new = await register_analysis_websocket("hard-send-timeout", new_socket)

    assert await asyncio.wait_for(old_send, timeout=0.2) is False
    assert await asyncio.wait_for(
        send_connection_json(new, {"generation": "new"}), timeout=0.2
    ) is True
    assert new_socket.sent == [{"generation": "new"}]
    assert websocket_delivery.abandoned_transport_task_count() == 1

    old_socket.release_send.set()
    for _ in range(50):
        if websocket_delivery.abandoned_transport_task_count() == 0:
            break
        await asyncio.sleep(0.001)
    assert websocket_delivery.abandoned_transport_task_count() == 0


@pytest.mark.asyncio
async def test_cancelled_unregister_removes_registry_and_observes_late_close(
    monkeypatch,
):
    from backend.utils import websocket_delivery

    monkeypatch.setattr(
        websocket_delivery, "_WEBSOCKET_CLOSE_TIMEOUT_SECONDS", 0.01,
    )
    close_started = asyncio.Event()
    release = asyncio.Event()

    class CancellationResistantCloseSocket(_Socket):
        async def close(self, code=1000):
            self.closed.append(code)
            close_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

    socket = CancellationResistantCloseSocket()
    connection = await register_analysis_websocket("cancel-close", socket)
    unregister = asyncio.create_task(
        unregister_websocket(connection, close=True, code=4001)
    )
    await close_started.wait()
    unregister.cancel()

    with pytest.raises(asyncio.CancelledError):
        await unregister

    assert connection.closed is True
    assert "cancel-close" not in websockets
    assert await send_connection_json(connection, {"late": True}) is False
    assert websocket_delivery.abandoned_transport_task_count() == 1

    release.set()
    for _ in range(50):
        if connection.socket_closed:
            break
        await asyncio.sleep(0.001)
    assert connection.socket_closed is True
    assert websocket_delivery.abandoned_transport_task_count() == 0


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
    old_closed = threading.Event()
    old_finished = threading.Event()
    results: dict[str, object] = {}

    class OldSocket:
        def __init__(self):
            self.closed = []

        async def send_json(self, _payload):
            raise AssertionError("stale socket received a post-replacement payload")

        async def close(self, code=1000):
            self.closed.append(code)
            old_closed.set()

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
            await wait_for(old_closed)
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
