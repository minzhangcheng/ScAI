"""Cancellation-safe helpers for draining independently owned asyncio tasks."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


_OWNED_THREAD_MAX_WORKERS = 8
_OWNED_BULK_THREAD_MAX_WORKERS = 4
_OWNED_THREAD_DRAIN_POLL_SECONDS = 0.05
_owned_thread_executor: ThreadPoolExecutor | None = None
_owned_thread_executor_pid = 0
_owned_bulk_thread_executor: ThreadPoolExecutor | None = None
_owned_bulk_thread_executor_pid = 0
_owned_thread_executor_lock = threading.Lock()


def _get_owned_thread_executor() -> ThreadPoolExecutor:
    """Return the bounded, process-local executor for owned blocking calls."""
    global _owned_thread_executor, _owned_thread_executor_pid

    pid = os.getpid()
    with _owned_thread_executor_lock:
        if (
            _owned_thread_executor is None
            or _owned_thread_executor_pid != pid
        ):
            # Never reuse a pool inherited across fork: its worker threads do
            # not exist in the child even though the Python object does.
            _owned_thread_executor = ThreadPoolExecutor(
                max_workers=_OWNED_THREAD_MAX_WORKERS,
                thread_name_prefix="scai-owned",
            )
            _owned_thread_executor_pid = pid
        return _owned_thread_executor


def _get_owned_bulk_thread_executor() -> ThreadPoolExecutor:
    """Return the pool reserved for long report/archive filesystem work."""
    global _owned_bulk_thread_executor, _owned_bulk_thread_executor_pid

    pid = os.getpid()
    with _owned_thread_executor_lock:
        if (
            _owned_bulk_thread_executor is None
            or _owned_bulk_thread_executor_pid != pid
        ):
            _owned_bulk_thread_executor = ThreadPoolExecutor(
                max_workers=_OWNED_BULK_THREAD_MAX_WORKERS,
                thread_name_prefix="scai-owned-bulk",
            )
            _owned_bulk_thread_executor_pid = pid
        return _owned_bulk_thread_executor


def consume_current_cancellation_request() -> bool:
    """Consume one delivered caller-cancellation request for deferred cleanup."""
    current = asyncio.current_task()
    if current is None or current.cancelling() <= 0:
        return False
    current.uncancel()
    return True


async def await_with_deferred_cancellation(task: asyncio.Future[T]) -> T:
    """Await an owned task to completion, then preserve caller cancellation.

    ``asyncio.wait`` wakes immediately on ordinary completion, while its timeout
    remains an independent backstop for a CPython 3.13 edge where an executor
    future can finish without waking a direct waiter. Repeated caller
    cancellation is absorbed until the owned operation is terminal.
    """
    cancellation: asyncio.CancelledError | None = None
    current = asyncio.current_task()

    while not task.done():
        try:
            await asyncio.wait(
                (task,),
                timeout=_OWNED_THREAD_DRAIN_POLL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError as exc:
            if current is None or current.cancelling() <= 0:
                raise
            cancellation = cancellation or exc

    try:
        result = task.result()
    except BaseException as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


async def run_in_owned_thread(
    func: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run blocking work in a bounded reusable executor and drain the call.

    ``asyncio.to_thread`` uses the loop's shared default executor.  On CPython
    3.13, cancelling a task that shields such work can leave the default
    executor's completion waiter unwoken during ``asyncio.run()`` shutdown even
    after the worker returned. Critical filesystem/kernel operations use a
    separate bounded pool and always drain their own submitted future before
    propagating cancellation. The pool is process-local and reused, so upload
    chunk loops do not create and join one OS thread per MiB.
    """
    return await _run_in_owned_executor(
        _get_owned_thread_executor(), func, *args, **kwargs
    )


async def run_in_owned_bulk_thread(
    func: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run long bulk I/O without occupying latency-sensitive owned workers."""
    return await _run_in_owned_executor(
        _get_owned_bulk_thread_executor(), func, *args, **kwargs
    )


async def _run_in_owned_executor(
    executor: ThreadPoolExecutor,
    func: Callable[P, T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = functools.partial(context.run, func, *args, **kwargs)
    future = loop.run_in_executor(executor, call)
    return await await_with_deferred_cancellation(future)
