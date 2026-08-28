"""Lifecycle snapshot failures action owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'_snapshot_failure_proceed_locked', '_snapshot_failure_skip_locked', 'snapshot_failure_retry', 'snapshot_failure_proceed', 'snapshot_failure_skip', 'snapshot_failure_terminate', '_snapshot_failure_terminate_locked', '_snapshot_failure_retry_locked'})

bindings = None


def bind_facade(facade) -> None:
    global bindings
    bindings = facade
    globals().update(
        {
            name: value
            for name, value in vars(facade).items()
            if not name.startswith("__") and name not in _OWNED_NAMES
        }
    )


async def snapshot_failure_retry(session_id: str, request: Request):
    """Retry a snapshot failure by re-invoking the exact failed capture producer.

    Validates ``SnapshotRetryContext`` from the failure gate. Missing or
    invalid context returns 409 and leaves the gate intact. Never defaults
    a missing producer to workflow-boundary.
    """
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Serialize the three snapshot-failure actions. Retry is the longest-running
    # mutator in the system - it re-runs the failed capture while the gate stays
    # armed - and all three actions read-modify-persist the *same* cached
    # AnalysisState. Without this, a Terminate issued mid-capture wins its
    # is_snapshot_failure_blocked() check, tears the workflow down and persists
    # STOPPED, and is then silently overwritten when this handler persists its
    # own outcome below.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_retry_locked(session_id, user_id)

async def _snapshot_failure_retry_locked(session_id: str, user_id: str):
    from pydantic import ValidationError

    from backend.services.snapshot_v2.failure_gate import (
        default_snapshot_retry_dependencies,
        retry_snapshot_producer,
    )
    from backend.services.snapshot_v2.models import SnapshotRetryContext

    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    if not bindings.is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    failure = getattr(state, "snapshot_failure", None) or {}
    raw_context = failure.get("retry_context")
    if not isinstance(raw_context, dict) or not raw_context:
        raise HTTPException(
            status_code=409,
            detail=(
                "Snapshot failure retry context is missing; "
                "gate left intact"
            ),
        )
    try:
        context = SnapshotRetryContext.model_validate(raw_context)
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Snapshot failure retry context is invalid; "
                "gate left intact"
            ),
        ) from exc

    outcome = await retry_snapshot_producer(
        session_id,
        state,
        context,
        default_snapshot_retry_dependencies(),
    )

    await bindings._update_session_cache(session_id, state)
    # force=True for the same reason as Skip below: the statuses this write
    # installs (RUNNING / PAUSED) do not bypass the 5s throttle, and the gate
    # write it supersedes was force-saved moments earlier.
    await bindings.async_save_session(session_id, state, force=True)

    if outcome.success and context.producer == "workflow_boundary":
        await bindings._close_hitl_socket_for_redrive(session_id, "snapshot_retry_resume")

    data = {
        "status": outcome.status
        if outcome.status is not None
        else getattr(state.status, "value", str(state.status)),
        "wait_input_reason": (
            outcome.wait_input_reason
            if outcome.wait_input_reason is not None
            or outcome.still_blocked
            else getattr(state, "wait_input_reason", None)
        ),
        "snapshot_failure": (
            outcome.snapshot_failure
            if outcome.still_blocked
            else getattr(state, "snapshot_failure", None)
        ),
    }
    if outcome.still_blocked:
        data["still_blocked"] = True
    if outcome.transaction_id is not None:
        data["transaction_id"] = outcome.transaction_id
    if outcome.snapshot_transaction is not None:
        data["snapshot_transaction"] = outcome.snapshot_transaction

    return APIResponse(
        success=outcome.success,
        message=outcome.message,
        data=data,
    )

async def snapshot_failure_proceed(session_id: str, request: Request):
    """Capture anyway after the memory preflight refused, accepting the OOM risk.

    Only valid on a gate that offered ``proceed`` — i.e. one raised by the
    runtime memory preflight. Arms a one-shot override and then runs the
    ordinary retry dispatch, so the capture re-runs through the same producer
    with the preflight lowered for that attempt only.
    """
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Same gate and same cached AnalysisState as Retry/Skip/Terminate; this
    # handler additionally mutates the override flag, so it must not interleave.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_proceed_locked(session_id, user_id)

async def _snapshot_failure_proceed_locked(session_id: str, user_id: str):
    from backend.services.snapshot_v2.failure_gate import (
        arm_memory_preflight_override,
        clear_memory_preflight_override,
    )

    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    if not bindings.is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    failure = getattr(state, "snapshot_failure", None) or {}
    if "proceed" not in (failure.get("allowed_actions") or []):
        raise HTTPException(
            status_code=409,
            detail=(
                "This snapshot failure cannot be overridden. Proceed is offered "
                "only when the memory preflight refused the capture."
            ),
        )

    arm_memory_preflight_override(state)
    # Reuse retry verbatim: the producer dispatch, context validation and gate
    # bookkeeping are identical — the override flag is the only difference, and
    # the capture consumes it on its own.
    #
    # The approval covers exactly the capture this dispatch leads to. Retry can
    # refuse before reaching one at all — 409 for a missing or invalid retry
    # context, an unknown producer — and the override was already force-written
    # to session.json by the time that happens, so without this the preflight
    # stayed disabled for the session and some later capture on a different
    # object skipped it. Disarm and persist on that path; a dispatch that does
    # run re-persists through _snapshot_failure_retry_locked's own save, and a
    # capture that fails clears the approval as it re-enters the gate.
    try:
        return await _snapshot_failure_retry_locked(session_id, user_id)
    except Exception:
        if clear_memory_preflight_override(state):
            await bindings._update_session_cache(session_id, state)
            await bindings.async_save_session(session_id, state, force=True)
        raise

async def snapshot_failure_skip(session_id: str, request: Request):
    """Skip snapshot creation and continue the analysis.

    Clears the failure gate WITHOUT capturing a snapshot and resumes the
    workflow, forfeiting only the restore point at that boundary. Producer-aware:
    workflow_boundary / post_report / chat are skippable; pre_restore is not
    (skipping would proceed with a restore that overwrites live state without a
    safety point) and returns 409.
    """
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Same gate, same cached state object as Retry/Terminate - must not interleave.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_skip_locked(session_id, user_id)

async def _snapshot_failure_skip_locked(session_id: str, user_id: str):
    from pydantic import ValidationError

    from backend.services.snapshot_v2.failure_gate import skip_snapshot_producer
    from backend.services.snapshot_v2.models import SnapshotRetryContext

    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    if not bindings.is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    failure = getattr(state, "snapshot_failure", None) or {}
    raw_context = failure.get("retry_context")
    if not isinstance(raw_context, dict) or not raw_context:
        raise HTTPException(
            status_code=409,
            detail="Snapshot failure retry context is missing; gate left intact",
        )
    try:
        context = SnapshotRetryContext.model_validate(raw_context)
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail="Snapshot failure retry context is invalid; gate left intact",
        ) from exc

    outcome = skip_snapshot_producer(state, context)
    if not outcome.success:
        # Skip unsupported for this producer (e.g. pre_restore): gate left intact.
        raise HTTPException(status_code=409, detail=outcome.message)

    # Clear gate + arm the one-shot skip marker + set RUNNING, then persist. The
    # frontend reconnects the WebSocket, which resumes the workflow from the graph
    # checkpoint; the armed marker makes the next required capture at that phase a
    # no-op so the analysis proceeds past the boundary that previously failed.
    # (Mirrors the Retry flow, which likewise defers graph re-drive to reconnect.)
    #
    # force=True: RUNNING is not one of the statuses that bypass the 5s save
    # throttle, and the gate write this request is undoing was force-saved
    # moments earlier — so the throttle window is entered from a guaranteed
    # preceding write and the cleared gate could be dropped silently.
    await bindings._update_session_cache(session_id, state)
    await bindings.async_save_session(session_id, state, force=True)

    # Without this close the session sat at status=RUNNING with no workflow, no
    # run task and no dialog, making zero progress until a full page reload.
    if context.producer == "workflow_boundary":
        await bindings._close_hitl_socket_for_redrive(session_id, "snapshot_skip_resume")

    return APIResponse(
        success=True,
        message=outcome.message,
        data={
            "status": outcome.status
            if outcome.status is not None
            else getattr(state.status, "value", str(state.status)),
            "wait_input_reason": None,
            "snapshot_failure": None,
        },
    )

async def snapshot_failure_terminate(session_id: str, request: Request):
    """Terminate the session on snapshot failure.

    Delegates to existing workflow teardown — does NOT create a second
    kernel teardown. Clears the failure gate and stops the workflow.
    """
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # Teardown pops the workflow, closes the kernel and deletes the graph
    # checkpoint. Running that under an in-flight Retry would yank the sandbox
    # out from under a live capture and then have the STOPPED write overwritten,
    # so wait for any Retry/Skip on this gate to finish first.
    async with get_snapshot_action_lock(session_id):
        return await _snapshot_failure_terminate_locked(session_id, user_id)

async def _snapshot_failure_terminate_locked(session_id: str, user_id: str):
    from backend.services.snapshot_v2.failure_gate import (
        terminate_snapshot_failure,
    )

    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    if not bindings.is_snapshot_failure_blocked(state):
        raise HTTPException(
            status_code=409,
            detail="Session is not blocked on a snapshot failure",
        )

    terminate_snapshot_failure(state)

    # Persist STOPPED *before* the irreversible teardown, and force the write.
    #
    # teardown_running_workflow closes the kernel and unlinks the graph
    # checkpoint (plus its -wal/-shm). Running it first meant a crash anywhere
    # inside it left session.json still saying WAITING_INPUT with the gate armed
    # while the checkpoint was already gone — a session the user had terminated,
    # shown to them as blocked, whose Retry would then resume against a
    # checkpoint that no longer exists. STOPPED is also not in
    # _FORCE_SAVE_STATUSES, so the unforced write could be dropped by the 5s
    # throttle on top of that, entered from the force-saved gate write.
    await bindings._update_session_cache(session_id, state)
    await bindings.async_save_session(session_id, state, force=True)

    # After STOPPED + termination_reason: tear down workflow/kernel resources.
    # Do NOT pause — pause would clobber STOPPED → PAUSED on the same state.
    cleanup_complete = await bindings.teardown_running_workflow(
        session_id, discard_runtime=True,
    )
    if cleanup_complete is False:
        return APIResponse(
            success=False,
            message=(
                "Session termination was committed, but resource cleanup "
                "is incomplete"
            ),
            data={
                "status": getattr(state.status, "value", str(state.status)),
                "cleanup_incomplete": True,
                "termination_committed": True,
            },
        )

    return APIResponse(
        success=True,
        message="Session terminated due to snapshot failure",
        data={
            "status": getattr(state.status, "value", str(state.status)),
            "termination_committed": True,
        },
    )
