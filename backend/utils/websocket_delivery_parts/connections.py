"""WebSocket delivery connections owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'_consume_thread_future', 'close_websocket', 'register_analysis_websocket', 'register_websocket', '_close_and_remove_on_owner', 'abandoned_transport_task_count', '_close_socket_locked', '_send_current_connection_json', '_wait_thread_future', 'drain_abandoned_transport_tasks', '_remove_if_current', 'close_analysis_websocket', '_retire_connection_on_owner', 'send_connection_json', 'reset_websocket_delivery_for_tests', '_finalize_transport_close', 'unregister_websocket', 'send_analysis_json', 'send_report_json', 'send_websocket_json', 'register_report_websocket'})

bindings = None


def bind_facade(facade) -> None:
    global bindings
    bindings = facade
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


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
            bindings._WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
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
                timeout=bindings._WEBSOCKET_RETIRE_LOCK_TIMEOUT_SECONDS,
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
                timeout=bindings._WEBSOCKET_RETIRE_LOCK_TIMEOUT_SECONDS,
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
                    bindings._WEBSOCKET_SEND_TIMEOUT_SECONDS,
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
    return await bindings.close_websocket("analysis", session_id, code=code)

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
