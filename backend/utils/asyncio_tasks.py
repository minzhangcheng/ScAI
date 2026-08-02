"""Cancellation-safe helpers for draining independently owned asyncio tasks."""

from __future__ import annotations

import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


def consume_current_cancellation_request() -> bool:
    """Consume one delivered caller-cancellation request for deferred cleanup."""
    current = asyncio.current_task()
    if current is None or current.cancelling() <= 0:
        return False
    current.uncancel()
    return True


async def await_with_deferred_cancellation(task: asyncio.Future[T]) -> T:
    """Await an owned task to completion, then preserve caller cancellation.

    Drain by cooperative polling from the start instead of relying on a direct
    cross-thread completion wakeup.  This avoids a CPython 3.13 edge where an
    executor or worker-thread future can finish without waking its direct
    asyncio waiter.  The timer is an independent wakeup source and also absorbs
    later caller cancellation requests before the first one is re-raised.
    """
    cancellation: asyncio.CancelledError | None = None
    current = asyncio.current_task()

    while not task.done():
        try:
            await asyncio.sleep(0.01)
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
    """Run blocking work in a one-call executor that is always joined.

    ``asyncio.to_thread`` uses the loop's shared default executor.  On CPython
    3.13, cancelling a task that shields such work can leave the default
    executor's completion waiter unwoken during ``asyncio.run()`` shutdown even
    after the worker returned.  Critical filesystem/kernel operations need a
    stronger ownership boundary: submit to a dedicated executor, drain the
    future, then join that executor before propagating cancellation.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = functools.partial(context.run, func, *args, **kwargs)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scai-owned")
    future = loop.run_in_executor(executor, call)
    try:
        return await await_with_deferred_cancellation(future)
    finally:
        # Polling above observes a terminal asyncio wrapper before every normal
        # or deferred-cancellation exit.  Joining here also proves the native
        # worker cannot outlive the caller's lifecycle record.
        executor.shutdown(wait=True, cancel_futures=False)
