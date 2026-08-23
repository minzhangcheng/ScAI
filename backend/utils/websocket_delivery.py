"""Generation-aware delivery for analysis and report WebSocket channels."""

from __future__ import annotations

import asyncio
import copy
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, replace
from datetime import date, datetime, time
from enum import Enum
import logging
import math
from pathlib import Path
import re
import threading
from typing import Any, Literal
import weakref
from collections.abc import Mapping

from fastapi import WebSocketDisconnect

from backend.constants import STRIPPED_FIGURE_PLACEHOLDER
from backend.state import report_websockets, websockets
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    consume_current_cancellation_request,
)


logger = logging.getLogger(__name__)
Channel = Literal["analysis", "report"]

# Network backpressure must not hold a connection lock forever. Registration
# publishes a replacement without waiting for this deadline, while delivery
# ordering lets the replacement wait only for this bounded old-generation send.
_WEBSOCKET_SEND_TIMEOUT_SECONDS = 10.0
_WEBSOCKET_CLOSE_TIMEOUT_SECONDS = 2.0
_WEBSOCKET_RETIRE_LOCK_TIMEOUT_SECONDS = 12.0


@dataclass(eq=False)
class WebSocketConnection:
    channel: Channel
    session_id: str
    generation: int
    websocket: Any
    owner_loop: weakref.ReferenceType[asyncio.AbstractEventLoop]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False
    socket_closed: bool = False
    transport_close_started: bool = False
    transport_close_abandoned: bool = False
    transport_close_task: asyncio.Task[Any] | None = None
    replacement_close_code: int | None = None


_LOOP_SLOTS_ATTR = "_scai_websocket_delivery_slots"
_tracked_loops: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, bool
] = weakref.WeakKeyDictionary()
_generations: dict[tuple[Channel, str], int] = {}
_current_generations: dict[tuple[Channel, str], int] = {}
_current_connections: dict[tuple[Channel, str], WebSocketConnection] = {}
_delivery_tails: dict[tuple[Channel, str], Future[None]] = {}
_registry_lock_guard = threading.Lock()
_registry_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()
_abandoned_transport_tasks: set[asyncio.Task[Any]] = set()
_abandoned_transport_labels: dict[asyncio.Task[Any], str] = {}


_PAYLOAD_MAX_DEPTH = 64
_KNOWN_ENCODED_PREFIXES = (
    "iVBORw0KGgo",  # PNG
    "/9j/",         # JPEG
    "R0lGOD",       # GIF
    "UklGR",        # WEBP
    "JVBERi0",      # PDF
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_EXACT_BASE64_KEYS = {
    "base64",
    "base64data",
    "figurebase64",
    "gifbase64",
    "graphicbase64",
    "imagebase64",
    "jpegbase64",
    "jpgbase64",
    "pdfbase64",
    "plotbase64",
    "pngbase64",
    "previewbase64",
    "svgbase64",
    "thumbnailbase64",
    "webpbase64",
}
_DURABLE_REALTIME_MESSAGE_LIMIT = 1000


def _normalized_payload_key(key: Any) -> str:
    return _NON_ALNUM_RE.sub("", str(key).casefold())


def _is_media_value_key(key: Any) -> bool:
    normalized = _normalized_payload_key(key)
    return normalized in {
        "image",
        "imagepng",
        "imagejpeg",
        "imagejpg",
        "imagewebp",
        "png",
        "jpeg",
        "jpg",
        "webp",
        "thumbnail",
        "preview",
        "figuredata",
        "plotdata",
        "graphicdata",
    }


def _is_base64_key(key: Any) -> bool:
    return _normalized_payload_key(key) in _EXACT_BASE64_KEYS


def _is_encoded_media(value: str, *, key: Any = None, media_value: bool = False) -> bool:
    lowered = value.casefold()
    if lowered.startswith("data:image/") or lowered.startswith("data:application/pdf"):
        return True
    if key is not None and _is_base64_key(key):
        return bool(value)
    if media_value and value.startswith(_KNOWN_ENCODED_PREFIXES):
        return True
    return value.startswith(_KNOWN_ENCODED_PREFIXES)


def _json_key(key: Any) -> Any:
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return str(key)


def _cycle_placeholder(json_mode: bool) -> Any:
    return "[cycle]"


def _copy_payload(
    value: Any,
    *,
    seen: set[int],
    depth: int,
    strip_binary: bool,
    json_mode: bool,
    key: Any = None,
    media_value: bool = False,
) -> Any:
    """Copy a payload without allowing binary output to cross a boundary.

    Durable copies preserve ordinary Python container types so LangGraph and
    session models can continue to deserialize them. The live JSON mode is
    deliberately stricter: every value is converted to a JSON-compatible
    representation and the original object graph is never mutated.
    """
    if depth > _PAYLOAD_MAX_DEPTH:
        return _cycle_placeholder(json_mode)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return STRIPPED_FIGURE_PLACEHOLDER
    if value is None or isinstance(value, (str, int, bool)):
        if isinstance(value, str) and strip_binary and _is_encoded_media(
            value, key=key, media_value=media_value
        ):
            return STRIPPED_FIGURE_PLACEHOLDER
        return value
    if isinstance(value, float):
        if json_mode and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Enum):
        enum_value = value.value
        if json_mode:
            return _copy_payload(
                enum_value,
                seen=seen,
                depth=depth + 1,
                strip_binary=strip_binary,
                json_mode=True,
                key=key,
                media_value=media_value,
            )
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat() if json_mode else value
    if isinstance(value, Path):
        return str(value) if json_mode else value

    object_id = id(value)
    if object_id in seen:
        return _cycle_placeholder(json_mode)
    seen.add(object_id)
    try:
        if isinstance(value, Mapping):
            items = []
            for raw_key, raw_value in value.items():
                copied_value = _copy_payload(
                    raw_value,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    key=raw_key,
                    media_value=_is_media_value_key(raw_key),
                )
                items.append((_json_key(raw_key) if json_mode else raw_key, copied_value))
            if json_mode:
                return dict(items)
            try:
                return type(value)(items)
            except (TypeError, ValueError):
                return dict(items)

        if isinstance(value, list):
            return [
                _copy_payload(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    media_value=media_value,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            copied = tuple(
                _copy_payload(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    media_value=media_value,
                )
                for item in value
            )
            return copied
        if isinstance(value, (set, frozenset)):
            copied_items = [
                _copy_payload(
                    item,
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    media_value=media_value,
                )
                for item in value
            ]
            if json_mode:
                return sorted(copied_items, key=repr)
            return type(value)(copied_items)

        if hasattr(value, "model_copy") and hasattr(value, "model_fields"):
            updates = {
                name: _copy_payload(
                    getattr(value, name),
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    key=name,
                    media_value=_is_media_value_key(name),
                )
                for name in value.model_fields
                if hasattr(value, name)
            }
            if json_mode:
                return {
                    _json_key(name): copied
                    for name, copied in updates.items()
                }
            return value.model_copy(update=updates, deep=False)

        if is_dataclass(value) and not isinstance(value, type):
            updates = {
                field.name: _copy_payload(
                    getattr(value, field.name),
                    seen=seen,
                    depth=depth + 1,
                    strip_binary=strip_binary,
                    json_mode=json_mode,
                    key=field.name,
                    media_value=_is_media_value_key(field.name),
                )
                for field in dataclass_fields(value)
            }
            if json_mode:
                return updates
            return replace(value, **updates)

        if json_mode:
            return str(value)
        return value
    finally:
        seen.discard(object_id)


def strip_nested_binary_payload(value: Any) -> Any:
    """Return a copy suitable for durable state or reconnect history."""
    return _copy_payload(
        value,
        seen=set(),
        depth=0,
        strip_binary=True,
        json_mode=False,
    )


def append_durable_realtime_message(target: Any, message: Any) -> None:
    """Append a detached binary-free message under the shared history cap."""
    durable_content = strip_nested_binary_payload(message.content)
    if hasattr(message, "model_copy"):
        durable_message = message.model_copy(
            deep=False,
            update={"content": durable_content},
        )
    else:
        durable_message = copy.copy(message)
        durable_message.content = durable_content
    target.realtime_messages.append(durable_message)
    if len(target.realtime_messages) > _DURABLE_REALTIME_MESSAGE_LIMIT:
        target.realtime_messages = target.realtime_messages[
            -_DURABLE_REALTIME_MESSAGE_LIMIT:
        ]


def json_safe_payload(value: Any, *, strip_binary: bool = True) -> Any:
    """Return a detached JSON-safe copy for the final WebSocket boundary."""
    return _copy_payload(
        value,
        seen=set(),
        depth=0,
        strip_binary=strip_binary,
        json_mode=True,
    )


def serialize_realtime_message(
    message: Any, *, strip_binary: bool = True
) -> dict[str, Any]:
    """Serialize one realtime message without mutating its Pydantic model."""
    if hasattr(message, "model_copy") and hasattr(message, "model_fields"):
        safe = _copy_payload(
            message,
            seen=set(),
            depth=0,
            strip_binary=strip_binary,
            json_mode=True,
        )
    elif hasattr(message, "model_dump"):
        safe = json_safe_payload(
            message.model_dump(mode="python"),
            strip_binary=strip_binary,
        )
    elif isinstance(message, Mapping):
        safe = json_safe_payload(message, strip_binary=strip_binary)
    else:
        safe = json_safe_payload({
            "type": getattr(message, "type", None),
            "agent": getattr(message, "agent", None),
            "content": getattr(message, "content", None),
            "timestamp": getattr(message, "timestamp", None),
            "id": getattr(message, "id", None),
        }, strip_binary=strip_binary)
    if isinstance(safe, dict):
        return safe
    return {"content": safe}


def _log_background_failure(label: str, exc: BaseException) -> None:
    logger.warning(
        "Background WebSocket task failed operation=%s type=%s",
        label,
        type(exc).__name__,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def _observe_abandoned_transport_task(task: asyncio.Task[Any]) -> None:
    with _registry_lock_guard:
        label = _abandoned_transport_labels.pop(task, "transport")
        _abandoned_transport_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        _log_background_failure(label, exc)


def _track_abandoned_transport_task(
    task: asyncio.Task[Any], *, operation: str
) -> None:
    with _registry_lock_guard:
        if task in _abandoned_transport_tasks:
            return
        _abandoned_transport_tasks.add(task)
        _abandoned_transport_labels[task] = operation
    task.add_done_callback(_observe_abandoned_transport_task)


async def _await_task_with_hard_timeout(
    task: asyncio.Task[Any], timeout: float, *, operation: str
) -> Any:
    """Wait for one transport task without trusting cancellation cooperation."""
    try:
        done, _pending = await asyncio.wait((task,), timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        _track_abandoned_transport_task(task, operation=operation)
        raise
    if task not in done:
        task.cancel()
        _track_abandoned_transport_task(task, operation=operation)
        raise asyncio.TimeoutError
    return task.result()


async def _await_with_hard_timeout(
    awaitable: Any, timeout: float, *, operation: str
) -> Any:
    """Enforce a deadline even when the underlying coroutine ignores cancel."""
    task = asyncio.create_task(awaitable, name=f"websocket-{operation}")
    return await _await_task_with_hard_timeout(
        task, timeout, operation=operation
    )


def _slots_for_running_loop() -> dict[
    tuple[Channel, str], WebSocketConnection
]:
    loop = asyncio.get_running_loop()
    slots = getattr(loop, _LOOP_SLOTS_ATTR, None)
    if slots is None:
        slots = {}
        setattr(loop, _LOOP_SLOTS_ATTR, slots)
    with _registry_lock_guard:
        _tracked_loops[loop] = True
    return slots


def _registry_lock_for_running_loop() -> asyncio.Lock:
    """Return a live lock owned by the caller's event loop.

    A module-global ``asyncio.Lock`` becomes permanently bound to the first
    loop that contends on it. ASGI tests and process lifecycle tooling can run
    this module through multiple sequential loops, so retain both the loop key
    and its idle registry lock weakly.
    """
    loop = asyncio.get_running_loop()
    with _registry_lock_guard:
        loop_locks = _registry_locks.get(loop)
        if loop_locks is None:
            loop_locks = weakref.WeakValueDictionary()
            _registry_locks[loop] = loop_locks
        lock = loop_locks.get("registry")
        if lock is None:
            lock = asyncio.Lock()
            loop_locks["registry"] = lock
        return lock


def _install_current_connection(
    key: tuple[Channel, str], websocket: Any,
    owner_loop: asyncio.AbstractEventLoop,
) -> tuple[WebSocketConnection, WebSocketConnection | None]:
    """Atomically publish one process-global current connection."""
    with _registry_lock_guard:
        outgoing = _current_connections.get(key)
        generation = _generations.get(key, 0) + 1
        _generations[key] = generation
        _current_generations[key] = generation
        connection = WebSocketConnection(
            channel=key[0],
            session_id=key[1],
            generation=generation,
            websocket=websocket,
            owner_loop=weakref.ref(owner_loop),
        )
        _current_connections[key] = connection
        if outgoing is not None:
            outgoing.closed = True
            outgoing.replacement_close_code = 1000
        _legacy_registry(key[0])[key[1]] = websocket
        return connection, outgoing


def _globally_current_connection(
    key: tuple[Channel, str],
) -> WebSocketConnection | None:
    with _registry_lock_guard:
        return _current_connections.get(key)


def _is_globally_current(connection: WebSocketConnection) -> bool:
    key = (connection.channel, connection.session_id)
    with _registry_lock_guard:
        return _current_generations.get(key) == connection.generation


def _legacy_registry(channel: Channel) -> dict[str, Any]:
    return websockets if channel == "analysis" else report_websockets


def _finish_delivery_turn(
    key: tuple[Channel, str], turn: Future[None]
) -> None:
    if not turn.done():
        turn.set_result(None)
    with _registry_lock_guard:
        if _delivery_tails.get(key) is turn:
            _delivery_tails.pop(key, None)


@asynccontextmanager
async def _delivery_turn(key: tuple[Channel, str]):
    """Serialize one session channel across threads and event loops.

    ``asyncio.Lock`` instances are loop-bound and therefore cannot arbitrate
    two ASGI/test loops. A short thread-safe reservation chains
    ``concurrent.futures.Future`` leases; each caller awaits its predecessor in
    its own loop and always releases (or delegates release after cancellation)
    without holding a blocking thread lock across an await.
    """
    with _registry_lock_guard:
        previous = _delivery_tails.get(key)
        turn: Future[None] = Future()
        _delivery_tails[key] = turn

    predecessor_finished = previous is None
    try:
        if previous is not None:
            # Polling a thread-safe Future avoids relying on cross-loop callback
            # delivery during loop shutdown/replacement. The wait remains
            # asynchronous and bounded to one scheduler tick.
            while not previous.done():
                await asyncio.sleep(0.001)
            previous.result()
            predecessor_finished = True
        yield
    finally:
        if predecessor_finished or previous is None or previous.done():
            _finish_delivery_turn(key, turn)
        else:
            # Cancellation while waiting must not let a successor overtake the
            # still-running predecessor. Complete this abandoned lease only
            # when the predecessor actually releases its own turn.
            previous.add_done_callback(
                lambda _completed: _finish_delivery_turn(key, turn)
            )


async def register_websocket(
    channel: Channel, session_id: str, websocket: Any
) -> WebSocketConnection:
    """Publish a new generation without waiting on old network I/O.

    ``_install_current_connection`` marks the displaced generation closed in
    the same atomic registry section that publishes the replacement. Sends
    which have not acquired the old connection lock therefore fail their
    current-generation check. A send already in flight may finish only on its
    own old socket; the channel-wide delivery turn prevents the new socket from
    overtaking it, and the send deadline bounds that handoff. Retirement is
    best-effort on the displaced connection's owner loop and is never awaited
    by a registrar that may hold a session lifecycle lock.
    """
    key = (channel, session_id)
    slots = _slots_for_running_loop()
    owner_loop = asyncio.get_running_loop()
    async with _registry_lock_for_running_loop():
        connection, displaced = _install_current_connection(
            key, websocket, owner_loop
        )
        slots[key] = connection

    if displaced is not None:
        displaced_loop = displaced.owner_loop()
        if displaced_loop is owner_loop:
            retire_task = asyncio.create_task(
                _retire_connection_on_owner(displaced),
                name=(
                    f"websocket-retire:{channel}:{session_id}:"
                    f"{displaced.generation}"
                ),
            )
            retire_task.add_done_callback(_consume_thread_future)
        elif displaced_loop is not None and displaced_loop.is_running():
            coroutine = _retire_connection_on_owner(displaced)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    coroutine, displaced_loop
                )
            except RuntimeError:
                coroutine.close()
            else:
                future.add_done_callback(_consume_thread_future)
    return connection


def _consume_thread_future(future: Future[Any]) -> None:
    try:
        future.result()
    except (asyncio.CancelledError, FutureCancelledError):
        return
    except BaseException as exc:
        _log_background_failure("connection-retirement", exc)


async def _wait_thread_future(future: Future[Any]) -> Any:
    """Await a concurrent Future without loop-bound callback handoff."""
    while not future.done():
        await asyncio.sleep(0.001)
    return future.result()


def _finalize_transport_close(
    connection: WebSocketConnection, task: asyncio.Task[Any]
) -> None:
    """Publish the eventual outcome of a close that outlived its caller."""
    if connection.transport_close_task is task:
        connection.transport_close_task = None
    try:
        task.result()
    except asyncio.CancelledError:
        connection.transport_close_abandoned = True
    except (WebSocketDisconnect, RuntimeError):
        connection.socket_closed = True
    except BaseException:
        connection.transport_close_abandoned = True
    else:
        connection.socket_closed = True


async def _close_socket_locked(
    connection: WebSocketConnection, *, code: int
) -> None:
    if connection.socket_closed:
        return
    existing = connection.transport_close_task
    if existing is not None and not existing.done():
        return
    if existing is not None:
        connection.transport_close_task = None

    connection.transport_close_started = True
    close_task = asyncio.create_task(
        connection.websocket.close(code=code),
        name=(
            f"websocket-close:{connection.channel}:{connection.session_id}:"
            f"{connection.generation}"
        ),
    )
    connection.transport_close_task = close_task
    close_task.add_done_callback(
        lambda completed: _finalize_transport_close(connection, completed)
    )
    close_completed = False
    try:
        await _await_task_with_hard_timeout(
            close_task,
            _WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
            operation=(
                f"close:{connection.channel}:{connection.session_id}:"
                f"{connection.generation}"
            ),
        )
        close_completed = True
    except asyncio.TimeoutError:
        connection.transport_close_abandoned = True
        logger.warning(
            "WebSocket close timed out channel=%s session=%s generation=%s",
            connection.channel,
            connection.session_id,
            connection.generation,
        )
    except asyncio.CancelledError:
        connection.transport_close_abandoned = True
        raise
    except (WebSocketDisconnect, RuntimeError):
        close_completed = True
    except Exception as exc:
        connection.transport_close_abandoned = True
        logger.debug(
            "WebSocket close failed channel=%s session=%s generation=%s type=%s",
            connection.channel,
            connection.session_id,
            connection.generation,
            type(exc).__name__,
        )
    finally:
        if close_task.done() and connection.transport_close_task is close_task:
            connection.transport_close_task = None
    if close_completed:
        connection.socket_closed = True


async def _retire_connection_on_owner(connection: WebSocketConnection) -> None:
    """Close a displaced socket on the event loop that owns its lock."""
    key = (connection.channel, connection.session_id)
    connection.closed = True
    acquired = False
    try:
        try:
            await asyncio.wait_for(
                connection.lock.acquire(),
                timeout=_WEBSOCKET_RETIRE_LOCK_TIMEOUT_SECONDS,
            )
            acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket retirement timed out channel=%s session=%s "
                "generation=%s",
                connection.channel,
                connection.session_id,
                connection.generation,
            )
            # The send holding the lock may be cancellation-resistant.  The
            # displaced generation is already closed logically, so perform a
            # bounded best-effort transport close without the lock before its
            # owner-loop slot becomes unreachable.
            await _close_socket_locked(connection, code=1000)
            return
        await _close_socket_locked(connection, code=1000)
    finally:
        if acquired:
            connection.lock.release()
        owner_loop = connection.owner_loop()
        if owner_loop is asyncio.get_running_loop():
            slots = getattr(owner_loop, _LOOP_SLOTS_ATTR, None)
            if isinstance(slots, dict) and slots.get(key) is connection:
                slots.pop(key, None)


async def _remove_if_current(connection: WebSocketConnection) -> None:
    key = (connection.channel, connection.session_id)
    owner_loop = connection.owner_loop()
    running_loop = asyncio.get_running_loop()
    if owner_loop is running_loop:
        slots = _slots_for_running_loop()
        async with _registry_lock_for_running_loop():
            if slots.get(key) is connection:
                slots.pop(key, None)
    with _registry_lock_guard:
        if _current_generations.get(key) == connection.generation:
            _current_generations.pop(key, None)
            _current_connections.pop(key, None)
            registry = _legacy_registry(connection.channel)
            if registry.get(connection.session_id) is connection.websocket:
                registry.pop(connection.session_id, None)


async def _close_and_remove_on_owner(
    connection: WebSocketConnection,
    *,
    code: int,
) -> None:
    """Own transport close independently of cancellation of its caller."""
    connection.closed = True
    acquired = False
    try:
        try:
            await asyncio.wait_for(
                connection.lock.acquire(),
                timeout=_WEBSOCKET_RETIRE_LOCK_TIMEOUT_SECONDS,
            )
            acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket close lock timed out channel=%s session=%s "
                "generation=%s",
                connection.channel,
                connection.session_id,
                connection.generation,
            )
        await _close_socket_locked(connection, code=code)
    finally:
        if acquired:
            connection.lock.release()
        await _remove_if_current(connection)


async def unregister_websocket(
    connection: WebSocketConnection | None,
    *,
    close: bool = False,
    code: int = 1000,
) -> None:
    """Close/remove only this generation; never remove a replacement socket."""
    if connection is None:
        return
    if connection.owner_loop() is not asyncio.get_running_loop():
        # A connection lock that genuinely contended is owned by its original
        # loop. A stale handler running elsewhere must not touch that lock; the
        # global generation check below is sufficient to protect a replacement.
        owner_loop = connection.owner_loop()
        if owner_loop is not None and owner_loop.is_running():
            coroutine = unregister_websocket(
                connection, close=close, code=code
            )
            try:
                concurrent = asyncio.run_coroutine_threadsafe(
                    coroutine, owner_loop
                )
            except RuntimeError:
                coroutine.close()
            else:
                await _wait_thread_future(concurrent)
                return
        await _remove_if_current(connection)
        return
    if close:
        # Fence new sends before handing transport ownership to the independent
        # cleanup task. The task may not receive its first scheduler turn before
        # this caller is cancelled.
        connection.closed = True
        cleanup = asyncio.create_task(
            _close_and_remove_on_owner(
                connection,
                code=connection.replacement_close_code or code,
            ),
            name=(
                f"websocket-close-remove:{connection.channel}:"
                f"{connection.session_id}:{connection.generation}"
            ),
        )
        await await_with_deferred_cancellation(cleanup)
        return

    cancellation: asyncio.CancelledError | None = None
    try:
        async with connection.lock:
            connection.closed = True
    except asyncio.CancelledError as exc:
        # Cancellation must not strand a logically current registry entry. The
        # underlying transport task remains managed independently if it did not
        # cooperate with cancellation.
        cancellation = exc
        consume_current_cancellation_request()
        connection.closed = True

    removal = asyncio.create_task(
        _remove_if_current(connection),
        name=(
            f"websocket-remove:{connection.channel}:{connection.session_id}:"
            f"{connection.generation}"
        ),
    )
    try:
        while not removal.done():
            try:
                await asyncio.wait((removal,), timeout=0.05)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                consume_current_cancellation_request()
        removal.result()
    except BaseException:
        if not removal.done():
            removal.cancel()
        raise
    if cancellation is not None:
        raise cancellation


async def send_connection_json(
    connection: WebSocketConnection | None, payload: dict[str, Any]
) -> bool:
    """Send through one exact generation, serialized with its close operation."""
    if connection is None:
        return False
    if connection.closed or not _is_globally_current(connection):
        return False
    owner_loop = connection.owner_loop()
    if owner_loop is None or owner_loop.is_closed():
        return False
    if owner_loop is not asyncio.get_running_loop():
        if not owner_loop.is_running():
            return False
        coroutine = _send_current_connection_json(connection, payload)
        try:
            concurrent = asyncio.run_coroutine_threadsafe(coroutine, owner_loop)
        except RuntimeError:
            coroutine.close()
            return False
        try:
            return await _wait_thread_future(concurrent)
        except asyncio.CancelledError:
            # The owner-loop send remains live and releases its own delivery
            # lease. Cancellation of a foreign workflow must not strand or drop
            # a payload already handed to the current connection.
            concurrent.add_done_callback(_consume_thread_future)
            raise
    return await _send_current_connection_json(connection, payload)


async def _send_current_connection_json(
    connection: WebSocketConnection, payload: dict[str, Any]
) -> bool:
    """Send on the connection owner loop after a global-current recheck."""
    key = (connection.channel, connection.session_id)
    # Normalize on the owner loop before taking the connection lock. The copy
    # is detached from workflow state, and the existing generation/ordering
    # fences below remain the authority for transport ownership.
    safe_payload = json_safe_payload(payload, strip_binary=False)
    cancellation: asyncio.CancelledError | None = None
    async with connection.lock:
        async with _delivery_turn(key):
            if connection.closed or not _is_globally_current(connection):
                return False
            try:
                await _await_with_hard_timeout(
                    connection.websocket.send_json(safe_payload),
                    _WEBSOCKET_SEND_TIMEOUT_SECONDS,
                    operation=(
                        f"send:{connection.channel}:{connection.session_id}:"
                        f"{connection.generation}"
                    ),
                )
                # A cross-loop replacement can become current while the socket
                # send is suspended. The lease keeps every new-generation send
                # behind this one; report False because this generation no
                # longer owns the channel when delivery completes.
                delivered_current = _is_globally_current(connection)
                if not delivered_current:
                    connection.closed = True
                    await _close_socket_locked(connection, code=1000)
                return delivered_current
            except asyncio.TimeoutError:
                connection.closed = True
                logger.warning(
                    "WebSocket send timed out channel=%s session=%s generation=%s",
                    connection.channel,
                    connection.session_id,
                    connection.generation,
                )
                await _close_socket_locked(connection, code=1011)
            except asyncio.CancelledError as exc:
                connection.closed = True
                cancellation = exc
                consume_current_cancellation_request()
            except (WebSocketDisconnect, RuntimeError) as exc:
                connection.closed = True
                logger.debug(
                    "WebSocket disconnected during send channel=%s session=%s "
                    "generation=%s type=%s",
                    connection.channel,
                    connection.session_id,
                    connection.generation,
                    type(exc).__name__,
                )
            except Exception as exc:
                connection.closed = True
                logger.warning(
                    "WebSocket send failed channel=%s session=%s generation=%s type=%s",
                    connection.channel,
                    connection.session_id,
                    connection.generation,
                    type(exc).__name__,
                )
    await _remove_if_current(connection)
    if cancellation is not None:
        raise cancellation
    return False


async def send_websocket_json(
    channel: Channel, session_id: str, payload: dict[str, Any]
) -> bool:
    """Send to the current generation, or leave the durable caller buffer intact."""
    connection = _globally_current_connection((channel, session_id))
    return await send_connection_json(connection, payload)


async def close_websocket(
    channel: Channel, session_id: str, *, code: int = 1000
) -> bool:
    """Close the generation that is current at lookup time."""
    connection = _globally_current_connection((channel, session_id))
    if connection is None:
        return False
    await unregister_websocket(connection, close=True, code=code)
    return True


async def register_analysis_websocket(
    session_id: str, websocket: Any
) -> WebSocketConnection:
    return await register_websocket("analysis", session_id, websocket)


async def register_report_websocket(
    session_id: str, websocket: Any
) -> WebSocketConnection:
    return await register_websocket("report", session_id, websocket)


async def send_analysis_json(session_id: str, payload: dict[str, Any]) -> bool:
    return await send_websocket_json("analysis", session_id, payload)


async def send_report_json(session_id: str, payload: dict[str, Any]) -> bool:
    return await send_websocket_json("report", session_id, payload)


async def close_analysis_websocket(session_id: str, *, code: int = 1000) -> bool:
    return await close_websocket("analysis", session_id, code=code)


def abandoned_transport_task_count() -> int:
    """Return the number of timed-out transport operations still running."""
    with _registry_lock_guard:
        return sum(not task.done() for task in _abandoned_transport_tasks)


async def drain_abandoned_transport_tasks(*, timeout: float = 2.0) -> int:
    """Cancel and boundedly drain transport operations owned by this loop.

    A transport coroutine can suppress ``CancelledError`` indefinitely, so
    shutdown cannot promise completion. It does retain and observe every such
    task, retries cancellation, and reports the number still pending after the
    deadline.
    """
    loop = asyncio.get_running_loop()
    with _registry_lock_guard:
        tasks = {
            task
            for task in _abandoned_transport_tasks
            if task.get_loop() is loop and not task.done()
        }
    if not tasks:
        return 0
    for task in tasks:
        task.cancel()
    _done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout))
    await asyncio.sleep(0)
    if pending:
        logger.warning(
            "WebSocket shutdown left %d cancellation-resistant transport "
            "task(s) pending",
            len(pending),
        )
    return len(pending)


def reset_websocket_delivery_for_tests() -> None:
    """Clear process-local delivery metadata; tests also clear legacy registries."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    with _registry_lock_guard:
        abandoned = list(_abandoned_transport_tasks)
        _abandoned_transport_tasks.clear()
        _abandoned_transport_labels.clear()
    for task in abandoned:
        if task.get_loop() is running_loop and not task.done():
            task.cancel()
    with _registry_lock_guard:
        pending_turns = list(_delivery_tails.values())
        for loop in list(_tracked_loops.keys()):
            slots = getattr(loop, _LOOP_SLOTS_ATTR, None)
            if isinstance(slots, dict):
                slots.clear()
            try:
                delattr(loop, _LOOP_SLOTS_ATTR)
            except AttributeError:
                pass
        _tracked_loops.clear()
        _generations.clear()
        _current_generations.clear()
        _current_connections.clear()
        _delivery_tails.clear()
        _registry_locks.clear()
    # Future callbacks may themselves take the registry guard, so resolve
    # abandoned test leases only after releasing that guard.
    for turn in pending_turns:
        if not turn.done():
            turn.set_result(None)
