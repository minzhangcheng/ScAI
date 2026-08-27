"""WebSocket delivery queue owners."""
from __future__ import annotations

from contextlib import asynccontextmanager

_OWNED_NAMES = frozenset({'_install_current_connection', '_observe_abandoned_transport_task', '_is_globally_current', '_await_task_with_hard_timeout', '_registry_lock_for_running_loop', '_log_background_failure', '_slots_for_running_loop', '_track_abandoned_transport_task', '_legacy_registry', '_delivery_turn', '_await_with_hard_timeout', '_globally_current_connection', '_finish_delivery_turn'})

def bind_facade(facade) -> None:
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


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
