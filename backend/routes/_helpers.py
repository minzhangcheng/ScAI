"""Shared helpers for route modules."""
import asyncio
import logging

from backend.state import workflows, _run_tasks, _monitor_tasks, state_lock

logger = logging.getLogger(__name__)


async def teardown_running_workflow(
    session_id: str,
    *,
    discard_runtime: bool = False,
) -> bool:
    """Stop active workflow, cancel run/monitor tasks, and delete checkpoint.

    Used before operations that must start from a clean slate (snapshot
    restore, re-analysis) to avoid stale checkpoints leaking state.
    """
    async with state_lock:
        workflow = workflows.get(session_id)

    if workflow:
        try:
            if discard_runtime:
                close_result = await workflow.close(save_runtime=False)
            else:
                close_result = await workflow.close()
            if close_result is False:
                logger.warning(
                    "[%s] Workflow teardown is incomplete; retaining owner and checkpoint",
                    session_id,
                )
                return False
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[{session_id}] Error closing workflow during teardown")
            return False
        async with state_lock:
            if workflows.get(session_id) is workflow:
                workflows.pop(session_id, None)

    task = _run_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    _run_tasks.pop(session_id, None)

    monitor = _monitor_tasks.pop(session_id, None)
    if monitor and not monitor.done():
        monitor.cancel()

    from backend.utils.snapshot import SnapshotManager
    await SnapshotManager(session_id).delete_checkpoint()
    return True
