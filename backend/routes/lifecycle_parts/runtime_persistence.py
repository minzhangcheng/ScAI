"""Lifecycle runtime persistence action owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'runtime_persistence_terminate', 'runtime_persistence_retry', '_raise_runtime_persistence_action_409', 'runtime_persistence_recompute', '_runtime_persistence_action_locked', '_runtime_persistence_action'})

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


def _raise_runtime_persistence_action_409(result) -> None:
    """Expose stable, machine-readable arbitration outcomes to clients."""
    from backend.services.runtime_persistence_gate import (
        RuntimePersistenceActionOutcome,
    )

    contracts = {
        RuntimePersistenceActionOutcome.STALE: (
            "RUNTIME_PERSISTENCE_ACTION_STALE",
            "The runtime-persistence action targets a stale gate token or generation.",
        ),
        RuntimePersistenceActionOutcome.DUPLICATE_OR_CONFLICT: (
            "RUNTIME_PERSISTENCE_ACTION_DUPLICATE_OR_CONFLICT",
            "A runtime-persistence action is already pending for this gate.",
        ),
        RuntimePersistenceActionOutcome.INACTIVE: (
            "RUNTIME_PERSISTENCE_ACTION_INACTIVE",
            "No active runtime-persistence action consumer is available.",
        ),
    }
    code, message = contracts.get(
        result.outcome,
        (
            "RUNTIME_PERSISTENCE_ACTION_CONFLICT",
            "The runtime-persistence action could not be accepted.",
        ),
    )
    raise HTTPException(
        status_code=409,
        detail={
            "code": code,
            "message": message,
            "action_result": result.outcome.value,
            "gate_token": result.gate_token,
            "gate_generation": result.generation,
        },
    )

async def _runtime_persistence_action_locked(
    session_id: str,
    user_id: str,
    action: str,
    *,
    gate_token: str | None = None,
    generation: int | None = None,
):
    from backend.services.runtime_persistence_gate import (
        RuntimePersistenceActionOutcome,
        RuntimePersistenceActionResult,
        clear_runtime_persistence_gate,
        is_runtime_persistence_blocked,
        prepare_runtime_recompute,
        runtime_persistence_gate_identity,
        validate_runtime_persistence_action,
    )

    state = await bindings.require_session_ownership(
        session_id=session_id,
        user_id=user_id,
    )
    if not is_runtime_persistence_blocked(state):
        token, current_generation = runtime_persistence_gate_identity(state)
        _raise_runtime_persistence_action_409(
            RuntimePersistenceActionResult(
                RuntimePersistenceActionOutcome.INACTIVE,
                action,
                token,
                current_generation,
            )
        )

    workflow = bindings.workflows.get(session_id)
    if workflow is not None:
        delivery = workflow.signal_runtime_persistence_action(
            action,
            gate_token=gate_token,
            generation=generation,
        )
        if delivery.outcome is RuntimePersistenceActionOutcome.ACCEPTED:
            await bindings._update_session_cache(session_id, state)
            await bindings.async_save_session(session_id, state, force=True)
            return APIResponse(
                success=True,
                message=f"Runtime persistence action '{action}' accepted",
                data={
                    "status": state.status.value,
                    "action_pending": True,
                    "still_blocked": True,
                    "action_result": delivery.outcome.value,
                    "gate_token": delivery.gate_token,
                    "gate_generation": delivery.generation,
                },
            )
        _raise_runtime_persistence_action_409(delivery)

    if action == "retry":
        # The process no longer owns the unsaved kernel namespace. Pretending a
        # retry succeeded here would silently bless the older committed
        # generation. Keep the durable gate and direct the user to recompute.
        retry_claim = validate_runtime_persistence_action(
            state,
            action,
            gate_token=gate_token,
            generation=generation,
            active=False,
        )
        _raise_runtime_persistence_action_409(retry_claim)

    fallback_claim = validate_runtime_persistence_action(
        state,
        action,
        gate_token=gate_token,
        generation=generation,
        active=True,
    )
    if fallback_claim.outcome is not RuntimePersistenceActionOutcome.ACCEPTED:
        _raise_runtime_persistence_action_409(fallback_claim)

    if action == "manage":
        prepare_runtime_recompute(state)
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state, force=True)
        await bindings._close_hitl_socket_for_redrive(
            session_id,
            "runtime_persistence_recompute",
        )
        return APIResponse(
            success=True,
            message="Runtime recompute/reduction authorized",
            data={
                "status": state.status.value,
                "still_blocked": False,
                "snapshot_failure": None,
                "action_result": fallback_claim.outcome.value,
            },
        )

    if action == "terminate":
        clear_runtime_persistence_gate(state)
        state.status = TaskStatus.STOPPED
        state.termination_reason = "runtime_persistence_failure"
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state, force=True)
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
                    "status": state.status.value,
                    "still_blocked": False,
                    "action_result": fallback_claim.outcome.value,
                    "cleanup_incomplete": True,
                    "termination_committed": True,
                },
            )
        return APIResponse(
            success=True,
            message="Session terminated after runtime persistence failure",
            data={
                "status": state.status.value,
                "still_blocked": False,
                "action_result": fallback_claim.outcome.value,
            },
        )

    raise HTTPException(status_code=400, detail="Unknown runtime persistence action")

async def _runtime_persistence_action(
    session_id: str,
    request: Request,
    action: str,
):
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    gate_token = body.get("gate_token")
    raw_generation = body.get("gate_generation", body.get("generation"))
    try:
        generation = int(raw_generation) if raw_generation is not None else None
    except (TypeError, ValueError):
        generation = None
    async with get_session_lock(session_id):
        return await _runtime_persistence_action_locked(
            session_id,
            user_id,
            action,
            gate_token=str(gate_token) if gate_token is not None else None,
            generation=generation,
        )

async def runtime_persistence_retry(session_id: str, request: Request):
    return await _runtime_persistence_action(session_id, request, "retry")

async def runtime_persistence_recompute(session_id: str, request: Request):
    return await _runtime_persistence_action(session_id, request, "manage")

async def runtime_persistence_terminate(session_id: str, request: Request):
    return await _runtime_persistence_action(session_id, request, "terminate")
