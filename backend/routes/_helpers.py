"""Shared helpers for route modules."""
import asyncio
import logging

from backend.state import workflows, _run_tasks, _monitor_tasks, state_lock

logger = logging.getLogger(__name__)


async def teardown_running_workflow(
    session_id: str,
    *,
    discard_runtime: bool = False,
    retire_runtime: bool = False,
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

    async with state_lock:
        task = _run_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Run task ignored cancellation; retaining its owner handle",
                session_id,
            )
            return False
        except asyncio.CancelledError:
            # A task that accepted our cancellation is terminal. If it is not,
            # this was cancellation of the teardown caller and must propagate.
            if not task.done():
                raise
        except Exception:
            # The producer failed while draining, which is still a terminal
            # state. Keep going only after proving it cannot write again.
            if not task.done():
                logger.exception(
                    "[%s] Run task drain failed before reaching a terminal state",
                    session_id,
                )
                return False
    if task is not None and not task.done():
        return False
    async with state_lock:
        current_task = _run_tasks.get(session_id)
        if current_task is not task:
            logger.warning(
                "[%s] A replacement run task appeared during teardown",
                session_id,
            )
            return False
        _run_tasks.pop(session_id, None)

    async with state_lock:
        monitor = _monitor_tasks.pop(session_id, None)
    if monitor and not monitor.done():
        monitor.cancel()

    if retire_runtime:
        from backend.services.session_lifecycle import (
            get_lifecycle_lock, lifecycle_file_lock, assert_session_generation_active,
        )
        from backend.services.workflow_runner import session_has_any_workflow_owner
        from backend.utils.runtime_apply import retire_live_runtime

        async with get_lifecycle_lock(session_id):
            if session_has_any_workflow_owner(session_id):
                return False
            with lifecycle_file_lock(session_id):
                assert_session_generation_active(session_id)
                retire_live_runtime(session_id, include_checkpoint=True)
        return True

    from backend.utils.snapshot import SnapshotManager
    await SnapshotManager(session_id).delete_checkpoint()
    return True
