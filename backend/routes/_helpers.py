"""Shared helpers for route modules."""
import asyncio
import logging
from datetime import datetime

from backend.models.schemas import TaskStatus
from backend.state import workflows, _run_tasks, _monitor_tasks, state_lock
from backend.utils.websocket_delivery import send_analysis_json

logger = logging.getLogger(__name__)


async def mark_gate_action_received(state) -> None:
    """Leave a user-facing wait state once its action has been accepted."""
    state.status = TaskStatus.RUNNING
    state.waiting_input_started_at = None
    await state.increment_version()


async def send_gate_action_received(
    session_id: str,
    state,
    change_type: str,
) -> None:
    """Publish the authoritative post-acceptance state without workflow delay."""
    await send_analysis_json(session_id, {
        "type": "state_sync",
        "agent": None,
        "content": {
            "change_type": change_type,
            "version": state.state_version,
            "status": state.status.value,
            "current_round": state.current_round,
            "analysis_phase": state.analysis_phase,
            "timestamp": state.last_sync_timestamp.isoformat(),
        },
        "timestamp": datetime.now().isoformat(),
    })


async def teardown_running_workflow(
    session_id: str,
    *,
    discard_runtime: bool = False,
    retire_runtime: bool = False,
    retain_restore_point: bool = False,
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

    if retain_restore_point:
        # Closing the owner is all that was asked for. Both branches below
        # destroy part of a restore point -- retirement takes the generation,
        # and the default takes the graph position -- and the caller is about
        # to put that position back from the generation's captured pair.
        return True

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

    from backend.config import get_session_states
    from backend.utils.checkpoint_paths import (
        checkpoint_sqlite_path,
        sqlite_companion_paths,
    )

    # Delete the LangGraph checkpoint so the next resume() starts fresh. This
    # was SnapshotManager.delete_checkpoint, whose only surviving caller is
    # this one; the paths come from the same helper workflow/core.py uses.
    db_path = checkpoint_sqlite_path(get_session_states(session_id))
    deleted = []
    for path in (db_path, *sqlite_companion_paths(db_path)):
        if path.exists():
            deleted.append(path.name)
        path.unlink(missing_ok=True)
    if deleted:
        logger.info(
            "[%s] Deleted checkpoint files: %s", session_id, ", ".join(deleted),
        )
    return True
