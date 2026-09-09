"""Lifecycle run resume action owners."""
from __future__ import annotations

_OWNED_NAMES = frozenset({'resume_analysis', '_pause_analysis_locked', '_resume_analysis_locked', 'run_analysis', '_run_analysis_locked', 'stop_analysis', 'pause_analysis', '_stop_analysis_locked'})

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


async def run_analysis(session_id: str, request: Request):
    """Start running analysis (use WebSocket for real-time updates)"""
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    async with get_session_lock(session_id):
        return await _run_analysis_locked(session_id, user_id)


async def _run_analysis_locked(session_id: str, user_id: str):
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)
    from backend.utils.runtime_pairing import refresh_runtime_availability
    refresh_runtime_availability(state)

    from backend.services.legacy_state_sanitizer import sanitize_legacy_state

    sanitize_legacy_state(
        state,
        workflow_active=session_id in bindings.workflows,
    )

    if state.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis already running")

    try:
        bindings.assert_recovery_clear(state, session_id=session_id)
    except RecoveryBlockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Idle-restored session: signal explicit_start so the graph node
    # preserves the restored round and plan rather than auto-incrementing.
    #
    # The same signal is what makes an *imported* session runnable at all. A
    # session that has already run (current_round > 0) is dispatched to
    # workflow.resume() by the WebSocket handler, and resume() refuses to
    # "invent a fresh initialize_round run" when it finds no checkpoint --
    # which an import never carries unless include_runtime was set. Without an
    # explicit start intent it logs, flips the status back to PAUSED and
    # executes nothing, so Start/Re-run looked like a silent no-op and the
    # session had no way forward at all.
    from backend.config import get_session_states
    from backend.utils.runtime_apply import _live_pkl

    # Decided first, because retiring the runtime changes the answer to every
    # question below it. "Re-run from your data" promises an empty namespace,
    # and the agent prompt says so; leaving the dump on disk let the new kernel
    # rehydrate the old variables, and retiring it *after* the checkpoint test
    # left the graph resuming an old node against a namespace that was gone.
    if is_forward_resume_blocked(
        getattr(state, "runtime_restore_status", "complete"),
        coherence=state_runtime_coherence(state),
    ):
        # Spec B2-2. `stale` here says the state and the two positions moved
        # past the committed generation -- not that the generation is bad. By
        # §3.1 that is true of any run interrupted anywhere except the instant
        # a save completed, so retiring on it made "re-run from the input
        # files" the ordinary outcome of every crash. If B2-1 captured this
        # generation's pair, put the positions back and continue from the step
        # the dump was taken at. Retirement stays the fallback for a generation
        # with no usable pair, and for `unavailable` and `inconsistent`.
        from backend.utils.runtime_apply import restore_generation_pair
        from backend.services.generation_pair_state import read_generation_pair_state
        from backend.services.session_store import refresh_runtime_persistence_projection

        committed = dict(
            getattr(state, "runtime_persistence", None) or {}
        ).get("committed_revision")
        restored = False
        if committed is not None:
            closed = await bindings.teardown_running_workflow(
                session_id, discard_runtime=True, retain_restore_point=True,
            )
            if closed is False:
                raise HTTPException(
                    status_code=409,
                    detail="Previous workflow cleanup is incomplete",
                )
            paired_state = await read_generation_pair_state(state, revision=int(committed))
            if paired_state is not None:
                # Progress must reach session.json before the notebook record
                # can declare the pair current. Both WebSocket routing and
                # resume's aupdate_state then see the paired plan and results.
                # Keep the existing object identity for cached route references.
                for name in type(state).model_fields:
                    setattr(state, name, getattr(paired_state, name))
                await bindings.async_save_session(session_id, state, force=True)
                restored = restore_generation_pair(session_id, revision=int(committed))
                if restored:
                    refresh_runtime_persistence_projection(session_id, state)

        # An unusable paired analysis state takes the same recompute path as a
        # missing pair; it cannot safely drive the graph over the old namespace.
        if not restored:
            closed = await bindings.teardown_running_workflow(
                session_id, discard_runtime=True, retire_runtime=True,
            )
            if closed is False:
                raise HTTPException(status_code=409, detail="Previous workflow cleanup is incomplete")
            state.runtime_recompute_pending = True
        state.restore_resume_kind = None
        state.restore_resume_workflow_state = None

    # A checkpoint is only resumable together with the kernel it was written
    # against: it names the graph node to continue from, and the pkl holds the
    # values that node expects to find. Testing the file alone let an archive
    # exported with include_runtime but imported without trust=true — the
    # checkpoint travels unconditionally, the pkl does not — look resumable,
    # which withheld the explicit_start intent and sent Re-run into an exact
    # continuation on an empty kernel instead of a fresh round.
    has_resumable_checkpoint = (
        checkpoint_sqlite_path(get_session_states(session_id)).exists()
        and _live_pkl(session_id).exists()
    )
    if getattr(state, "restore_resume_kind", None) is None and (
        getattr(state, "restore_branch_id", None)
        or (not has_resumable_checkpoint and state.current_round > 0)
    ):
        state.restore_start_intent = "explicit_start"

    # A Skip armed at a boundary the graph never reached again stays on the
    # session forever — consume_snapshot_skip only clears it at a *matching*
    # boundary. Starting a run is a fresh intent, so an unconsumed marker from a
    # previous run must not silently suppress this run's first required capture
    # (the wildcard form, armed when the failing phase is unknown, would
    # suppress whichever capture happens to come first).
    if getattr(state, "skip_required_snapshot_phase", None) is not None:
        state.skip_required_snapshot_phase = None

    # The recompute decision and its retirement happened above, before anything
    # read the files it retires.
    state.runtime_restore_status = "complete"
    state.runtime_restore_warning = None
    state.status = TaskStatus.RUNNING
    # Forced: this write carries the cleared runtime block and the explicit
    # start intent, and RUNNING is not one of the statuses that bypasses the
    # 5s save throttle. Losing it to the throttle would leave a restarted
    # backend still refusing to start the session the user just started.
    await bindings.async_save_session(session_id, state, force=True)

    return APIResponse(
        success=True,
        message="Analysis started. Connect to WebSocket for updates.",
        data={"websocket_url": f"/ws/{session_id}"}
    )

async def pause_analysis(session_id: str, request: Request):
    """Pause running analysis"""
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent resume/stop/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _pause_analysis_locked(session_id, user_id)

async def _pause_analysis_locked(session_id: str, user_id: str):
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    # Idempotent: already paused → no-op success (rapid double-click safe).
    if state.status == TaskStatus.PAUSED:
        return APIResponse(success=True, message="Analysis already paused")

    # Accept RUNNING and any WAITING_* state. Pausing while waiting for HITL
    # input still snapshots sandbox and clears the cooperative gate so when
    # the user later resumes from the gate, the agent loop also honors pause.
    _pausable = {TaskStatus.RUNNING} | WAITING_HITL_STATUSES
    if state.status not in _pausable:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause from status '{state.status}'"
        )

    if state.status in WAITING_HITL_STATUSES:
        state.pre_pause_status = state.status
    else:
        state.pre_pause_status = None
    # Grab workflow reference under lock to avoid race with cleanup_stale_sessions
    async with _state_lock:
        workflow = bindings.workflows.get(session_id)
    if workflow:
        result = await workflow.pause(session_id, analysis_state=state)
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state, force=not result.get("success", True))
        if not result.get("success", True):
            return APIResponse(
                success=False,
                message=(
                    "Analysis execution is held, but its live runtime could not "
                    "be persisted. Choose Retry persistence, Recompute or reduce, "
                    "or Terminate."
                ),
                data=result,
            )
    else:
        state.status = TaskStatus.PAUSED
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state)

    return APIResponse(
        success=True,
        message="Analysis paused",
        data={"status": state.status.value, "still_blocked": False},
    )

async def resume_analysis(session_id: str, request: Request):
    """Resume paused or completed analysis (for re-analysis)"""
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/stop/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _resume_analysis_locked(session_id, user_id)

def _why_the_runtime_cannot_be_continued(coherence: str, state) -> str:
    """Say what happened, in the words of the thing that actually happened.

    A stale verdict has two causes and this used to report one of them for
    both. On a real crash it told the operator "the committed runtime is stale
    after a failed save" when no save had failed: the kernel had simply run
    past its last committed generation, which is what a crash between saves
    looks like and is a different thing to fix.
    """
    if coherence == "stale":
        persistence = getattr(state, "runtime_persistence", None) or {}
        if not isinstance(persistence, dict):
            persistence = {}
        committed = int(persistence.get("committed_revision") or 0)
        reached = max(
            int(persistence.get("current_revision") or 0),
            int(persistence.get("degraded_from_revision") or 0),
        )
        if reached > committed:
            return (
                "The analysis ran past its last saved runtime, so the kernel "
                "state on disk is older than the work it records. This is what "
                "an interrupted session looks like: the executor kept going "
                "after the last save and stopped before the next one."
            )
        return (
            "A runtime save failed, so the kernel state on disk is older than "
            "the work the session records."
        )
    if coherence == "inconsistent":
        return (
            "The saved kernel state belongs to a different point in this "
            "analysis than the state that was persisted with it."
        )
    return "This session has no kernel state that can be continued."


def _remedies_for_an_unusable_runtime(session_id: str, state=None) -> str:
    """What the operator can actually do, in the order they would rather do it.

    The first remedy is usually the cheapest and this message used not to
    mention it at all. ``_run_analysis_locked`` rolls a blocked session back to
    its committed generation -- the dump, the notebook record, the graph
    checkpoint and the executor position together -- and continues from the
    step that dump was taken at. What the user is being asked to give up is the
    work since the last save, not the run. Saying "re-run the analysis from
    your data" described the fallback as though it were the only outcome, on a
    screen where a five-hour analysis is at stake.
    """
    rollback = ""
    persistence = dict(getattr(state, "runtime_persistence", None) or {})
    committed = persistence.get("committed_revision")
    try:
        from backend.utils.runtime_apply import generation_pair_is_restorable

        if generation_pair_is_restorable(session_id, committed):
            reached = max(
                int(persistence.get("current_revision") or 0),
                int(persistence.get("degraded_from_revision") or 0),
            )
            lost = max(0, reached - int(committed or 0))
            rollback = (
                "Starting the analysis again rolls it back to the last saved "
                f"runtime and continues from there, giving up the {lost} "
                "step(s) run after that save. "
                if lost
                else "Starting the analysis again continues from the last "
                "saved runtime. "
            )
    except Exception:  # noqa: BLE001 - an offer, never a reason to fail a refusal
        logger.warning(
            "Could not tell whether %s can roll back to its committed "
            "generation", session_id, exc_info=True,
        )

    offer = ""
    try:
        from backend.config import snapshots_enabled

        if snapshots_enabled():
            from backend.services.round_snapshots.listing import (
                read_session_listing,
            )

            count = len(read_session_listing(session_id)["snapshots"])
            if count == 1:
                offer = (
                    "This session has one decision snapshot; restoring it "
                    "opens a new session from just before that decision, with "
                    "a runtime that verifies. "
                )
            elif count > 1:
                offer = (
                    f"This session has {count} decision snapshots; restoring "
                    "one opens a new session from just before that decision, "
                    "with a runtime that verifies. "
                )
    except Exception:  # noqa: BLE001 - an offer, never a reason to fail a refusal
        logger.warning(
            "Could not list decision snapshots for %s while explaining a "
            "blocked resume", session_id, exc_info=True,
        )
    if rollback:
        return (
            f"{rollback}{offer}Otherwise re-run the analysis from your data to "
            "recompute the kernel state, or view the session read-only."
        )
    if offer:
        return (
            f"{offer}Otherwise re-run the analysis from your data to recompute "
            "the kernel state, or view the session read-only."
        )
    return (
        "Re-run the analysis from your data to recompute the kernel state, or "
        "view the session read-only."
    )


async def _resume_analysis_locked(session_id: str, user_id: str):
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)
    from backend.utils.runtime_pairing import refresh_runtime_availability

    # Check the live owner before the cold-file probe. A paused workflow whose
    # save failed still owns an intact kernel, and a live HITL gate is itself
    # the continuation boundary; absence of a committed pkl must not turn
    # either case into a false runtime-unavailable refusal.
    live_continuation = is_live_runtime_continuation_safe(session_id, state)
    if not live_continuation:
        refresh_runtime_availability(state)

    from backend.services.legacy_state_sanitizer import sanitize_legacy_state

    sanitize_legacy_state(
        state,
        workflow_active=session_id in bindings.workflows,
    )

    try:
        bindings.assert_recovery_clear(state, session_id=session_id)
    except RecoveryBlockedError as exc:
        return APIResponse(
            success=False,
            message=str(exc),
            data={"needs_reconnect": False, "recovery_blocked": True},
        )

    # A completed session first has to retire its old workflow. Do not consume
    # the saved round limit before that teardown succeeds, otherwise a refused
    # re-analysis request still mutates the durable analysis contract.
    if (
        state.status != TaskStatus.COMPLETED
        and state.original_max_rounds is not None
    ):
        state.max_rounds = state.original_max_rounds
        state.original_max_rounds = None

    coherence = state_runtime_coherence(state)
    if (
        is_forward_resume_blocked(
            state.runtime_restore_status,
            coherence=coherence,
        )
        and not live_continuation
        and state.status != TaskStatus.COMPLETED
    ):
        warning = getattr(state, "runtime_restore_warning", None)
        if not warning:
            warning = _why_the_runtime_cannot_be_continued(coherence, state)
        remedies = _remedies_for_an_unusable_runtime(session_id, state)
        state.runtime_restore_status = "unavailable"
        state.runtime_restore_warning = f"{warning} {remedies}"
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state, force=True)
        return APIResponse(
            success=True,
            message=(
                "This session cannot continue on the runtime it has. "
                f"{remedies}"
            ),
            data={
                "needs_reconnect": False,
                "runtime_unavailable": True,
                "runtime_restore_status": state.runtime_restore_status,
                "runtime_restore_warning": state.runtime_restore_warning,
                "status": state.status.value,
            },
        )

    if state.status == TaskStatus.COMPLETED:
        # Re-analysis: tear down the terminated checkpoint so the WS
        # reconnect triggers a fresh run (core.py "no checkpoint" path)
        # instead of resuming a graph already at END (infinite loop).
        from backend.workflow.provenance import consume_restore_provenance

        recompute = is_forward_resume_blocked(
            state.runtime_restore_status, coherence=state_runtime_coherence(state),
        )
        cleanup_options = {"discard_runtime": True}
        if recompute:
            cleanup_options["retire_runtime"] = True
        cleanup_complete = await bindings.teardown_running_workflow(
            session_id, **cleanup_options,
        )
        if cleanup_complete is False:
            return APIResponse(
                success=False,
                message=(
                    "Re-analysis was not started because the previous "
                    "workflow cleanup is incomplete"
                ),
                data={
                    "status": state.status.value,
                    "needs_reconnect": False,
                    "cleanup_incomplete": True,
                },
            )
        if state.original_max_rounds is not None:
            state.max_rounds = state.original_max_rounds
            state.original_max_rounds = None
        state.status = TaskStatus.RUNNING
        state.max_rounds = state.current_round + 1
        # Clear both durable restore fields so the next run cannot mis-tag
        # plan/RoundResult from a prior restore (terminal/idle leave provenance).
        consume_restore_provenance(state)
        # Same contract as POST /run: a re-analysis that is clearing an
        # unavailable runtime has to tell the agent its kernel is empty, or it
        # carries on as though adata were still loaded from the original run.
        if recompute:
            state.runtime_recompute_pending = True
            state.restore_resume_kind = None
            state.restore_resume_workflow_state = None
            state.restore_start_intent = "explicit_start"
        state.runtime_restore_status = "complete"  # re-run recomputes kernel state
        state.runtime_restore_warning = None  # fresh rerun clears any restore warning
        await bindings._update_session_cache(session_id, state)
        # Forced for the same reason as POST /run: the cleared runtime block and
        # the recompute flag must survive a restart, and RUNNING does not bypass
        # the save throttle on its own.
        await bindings.async_save_session(session_id, state, force=True)
        return APIResponse(
            success=True,
            message="Re-analysis started",
            data={"needs_reconnect": True}
        )
    elif state.status == TaskStatus.WAITING_INPUT:
        # P3-5: directionless-INNOVATION default lives in workflow's
        # _initialize_round (workflow-state invariant, not a transport concern).
        # Continue from round-end wait (user sent input or skipped)
        # Signal the workflow to continue from _wait_for_user_input
        # Grab workflow reference under lock to avoid race with cleanup_stale_sessions
        async with _state_lock:
            workflow = bindings.workflows.get(session_id)
        if workflow:
            state.status = TaskStatus.RUNNING
            await bindings._update_session_cache(session_id, state)
            await bindings.async_save_session(session_id, state)
            workflow.signal_continue()
            return APIResponse(
                success=True,
                message="Continuing analysis"
            )
        else:
            if (
                state.restore_resume_kind == "semantic_gate"
                and state.restore_resume_gate in SEMANTIC_WAIT_GATES
            ):
                from backend.workflow.restore_targets import (
                    prepare_restored_semantic_action,
                )

                target = prepare_restored_semantic_action(state)
                if target is None:
                    raise HTTPException(
                        status_code=409,
                        detail="Restored semantic gate has no exact post-action target",
                    )
                state.status = TaskStatus.RUNNING
                await bindings._update_session_cache(session_id, state)
                await bindings.async_save_session(session_id, state)
                return APIResponse(
                    success=True,
                    message="Continuing restored analysis",
                    data={
                        "needs_reconnect": True,
                        "restored_wait_continuation": True,
                        "next_node": target,
                    },
                )
            await bindings._pause_and_restart_ws(session_id, state)
            return APIResponse(
                success=True,
                message="Continuing analysis",
                data={"needs_reconnect": True},
            )
    elif state.status == TaskStatus.WAITING_ANNOTATION:
        # Continue from annotation confirmation wait
        state.status = TaskStatus.RUNNING
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state)
        async with _state_lock:
            workflow = bindings.workflows.get(session_id)
        if workflow:
            workflow.signal_annotation_confirmed()
            return APIResponse(
                success=True,
                message="Annotation confirmed, continuing analysis"
            )
        else:
            await bindings._pause_and_restart_ws(session_id, state)
            return APIResponse(
                success=True,
                message="Continuing analysis",
                data={"needs_reconnect": True},
            )
    elif state.status == TaskStatus.WAITING_PLAN_REVIEW:
        # Plan review requires an explicit approve/skip/modify action via /plan-review.
        # Generic resume must not silently convert the gate into "skip".
        return APIResponse(
            success=True,
            message="Session is waiting for plan review. Submit /plan-review to approve, skip, or modify the plan.",
            data={"needs_reconnect": False},
        )
    elif state.status == TaskStatus.WAITING_SPECIES_INPUT:
        # Continue from species-input wait by sending an empty signal
        # (workflow treats empty input as "use suggestion / auto-skip").
        # An explicit submission goes through /interaction/species-input instead.
        state.status = TaskStatus.RUNNING
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state)
        async with _state_lock:
            workflow = bindings.workflows.get(session_id)
        if workflow:
            workflow.signal_species_input("")
            return APIResponse(
                success=True,
                message="Continuing analysis (species input auto-skipped)"
            )
        else:
            await bindings._pause_and_restart_ws(session_id, state)
            return APIResponse(
                success=True,
                message="Continuing analysis",
                data={"needs_reconnect": True},
            )
    elif state.status == TaskStatus.PAUSED:
        if state.pre_pause_status in WAITING_HITL_STATUSES:
            restored_status = state.pre_pause_status
            state.status = restored_status
            state.pre_pause_status = None
        else:
            state.status = TaskStatus.RUNNING
            state.pre_pause_status = None
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state)
        async with _state_lock:
            workflow = bindings.workflows.get(session_id)
        if workflow:
            workflow.signal_resume()
            # P0-5: do NOT auto-signal HITL gates on generic resume. The user
            # asked to unpause; they must still submit an explicit action via
            # /choose-direction, /confirm-annotation, or /plan-review to
            # advance past the restored WAITING_* gate.
            if state.status in WAITING_HITL_STATUSES:
                message = (
                    "Analysis resumed; awaiting explicit user input via the "
                    "appropriate gate endpoint"
                )
            else:
                message = "Analysis resumed"
            return APIResponse(
                success=True,
                message=message,
                data={"needs_reconnect": False, "status": state.status.value},
            )
        return APIResponse(
            success=True,
            message="Analysis resumed",
            data={"needs_reconnect": True}
        )
    elif state.status in (TaskStatus.GENERATING_REPORT, TaskStatus.FAILED):
        # Interrupted during report generation or failed — allow re-run
        state.status = TaskStatus.RUNNING
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state)
        return APIResponse(
            success=True,
            message="Resuming analysis",
            data={"needs_reconnect": True}
        )
    else:
        raise HTTPException(status_code=400, detail="Analysis not paused, waiting, or completed")

async def stop_analysis(session_id: str, request: Request, req: StopAnalysisRequest):
    """Stop a running analysis - either immediately or with summary report"""
    bindings.validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    # P1-9: serialize with concurrent pause/resume/llm-failure mutators.
    sess_lock = get_session_lock(session_id)
    async with sess_lock:
        return await _stop_analysis_locked(session_id, user_id, req)

async def _stop_analysis_locked(session_id: str, user_id: str, req: StopAnalysisRequest):
    state = await bindings.require_session_ownership(session_id=session_id, user_id=user_id)

    if state.status not in ({TaskStatus.RUNNING, TaskStatus.GENERATING_REPORT} | WAITING_HITL_STATUSES):
        if state.status == TaskStatus.PAUSED:
            return APIResponse(success=True, message="Analysis is already paused")
        return APIResponse(success=True, message=f"Analysis is {state.status.value}")

    # Handle GENERATING_REPORT state
    if state.status == TaskStatus.GENERATING_REPORT:
        if req.action == "immediate":
            state.status = TaskStatus.PAUSED
            await bindings._update_session_cache(session_id, state)
            await bindings.async_save_session(session_id, state)
            task = _run_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
            return APIResponse(success=True, message="Report generation cancelled")
        else:  # summarize
            return APIResponse(success=True, message="Report generation already in progress")

    if req.action == "immediate":
        state.status = TaskStatus.PAUSED
        await bindings._update_session_cache(session_id, state)
        await bindings.async_save_session(session_id, state)
        task = _run_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
        # Notify via WebSocket
        await send_analysis_json(session_id, {
                    "type": "status",
                    "agent": None,
                    "content": "Analysis terminated. State saved.",
                    "timestamp": datetime.now().isoformat()
        })
        return APIResponse(success=True, message="Analysis terminated")

    elif req.action == "summarize":
        # Snapshot-failure gate: stop-and-summarize must not skip or advance.
        if bindings.is_snapshot_failure_blocked(state):
            return APIResponse(
                success=False,
                message=(
                    "Session is blocked on a required snapshot failure. "
                    "Stop-and-summarize is refused until Retry, Manage, or Terminate."
                ),
                data={
                    "snapshot_failure": getattr(state, "snapshot_failure", None),
                    "wait_input_reason": getattr(state, "wait_input_reason", None),
                },
            )
        if state.status == TaskStatus.WAITING_INPUT:
            skip_target = resolve_hitl_skip_target(state)
            if skip_target is not None:
                node_name, generation = skip_target
                if node_name == "species_input":
                    # Species-input gate — released via signal_species_input
                    if state.original_max_rounds is None:
                        state.original_max_rounds = state.max_rounds
                    state.max_rounds = state.current_round
                    state.status = TaskStatus.RUNNING
                    await _resume_workflow(session_id, state,
                        signal_fn=lambda w: w.signal_species_input(""))
                    return APIResponse(success=True, message="Skipping species input, generating report...")
                # LLM-failure OR disabled-agent gate — both wait on
                # _llm_failure_events[node] and are released by
                # signal_llm_failure_action. Using signal_continue here would
                # leave the gate blocked forever (deadlock).
                if state.original_max_rounds is None:
                    state.original_max_rounds = state.max_rounds
                state.max_rounds = state.current_round
                state.status = TaskStatus.RUNNING
                await _resume_workflow(session_id, state,
                    signal_fn=lambda w, nn=node_name, g=generation: w.signal_llm_failure_action(nn, "skip", g))
                return APIResponse(success=True, message="Skipping blocked step, generating report...")
            else:
                # Round completion wait — existing behavior
                if state.original_max_rounds is None:
                    state.original_max_rounds = state.max_rounds
                state.max_rounds = state.current_round
                state.status = TaskStatus.RUNNING
                await _resume_workflow(session_id, state,
                    signal_fn=lambda w: w.signal_continue())
                return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_ANNOTATION:
            # Workflow is blocked at _annotation_event.wait() inside callback
            # Unblock it and let the round finish, then route to report gen
            if state.original_max_rounds is None:
                state.original_max_rounds = state.max_rounds
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _resume_workflow(session_id, state,
                signal_fn=lambda w: w.signal_annotation_confirmed())
            return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_PLAN_REVIEW:
            # Workflow is blocked at _plan_review_event.wait()
            # Unblock it by skipping the review and finishing the round
            if state.original_max_rounds is None:
                state.original_max_rounds = state.max_rounds
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _resume_workflow(session_id, state,
                signal_fn=lambda w: w.signal_plan_review("skip"))
            return APIResponse(success=True, message="Generating report...")
        elif state.status == TaskStatus.WAITING_SPECIES_INPUT:
            # Workflow is blocked at _species_event.wait() inside planning or
            # gene validation callback. Release via empty signal (auto-skip).
            if state.original_max_rounds is None:
                state.original_max_rounds = state.max_rounds
            state.max_rounds = state.current_round
            state.status = TaskStatus.RUNNING
            await _resume_workflow(session_id, state,
                signal_fn=lambda w: w.signal_species_input(""))
            return APIResponse(success=True, message="Generating report...")
        else:
            # RUNNING - mid-agent execution: cancel task, start report generation
            state.status = TaskStatus.GENERATING_REPORT
            await bindings._update_session_cache(session_id, state)
            await bindings.async_save_session(session_id, state)
            task = _run_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
            return APIResponse(success=True, message="Generating report...")
