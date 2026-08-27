"""SingleCellAnalysisWorkflow shutdown method owners."""
from __future__ import annotations


CHECKPOINTS_TO_KEEP_PER_NAMESPACE = 5


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )


async def _register_close_persistence_gate(
    self,
    analysis_state: AnalysisState,
) -> tuple[bool, bool]:
    """Make a final-save gate reachable without replacing a newer owner.

    WebSocket and headless workflows are already registered. Standalone
    report workflows are not, so a final-save failure temporarily publishes
    their workflow object while the live kernel is awaiting an action.
    """
    from backend.state import sessions, state_lock, workflows

    async with state_lock:
        current_state = sessions.get(self.session_id)
        current_workflow = workflows.get(self.session_id)
        # A missing session means a tombstone/delete already detached this
        # generation. Never resurrect an old workflow into the public
        # registry merely to host a gate no user can reach.
        if current_state is not analysis_state:
            return False, False
        if current_workflow is None:
            workflows[self.session_id] = self
            self._close_gate_registered_here = True
            return True, True
        return current_workflow is self, False

async def _release_close_persistence_gate_registration(self) -> None:
    """Compare-and-remove this closed workflow's registry ownership."""
    from backend.state import report_workflows, state_lock, workflows

    async with state_lock:
        if workflows.get(self.session_id) is self:
            workflows.pop(self.session_id, None)
        if report_workflows.get(self.session_id) is self:
            report_workflows.pop(self.session_id, None)
    self._close_gate_registered_here = False

def _ensure_close_retry_task(self) -> None:
    """Retry incomplete owned-resource cleanup with bounded backoff.

    A literal ``False`` from close is intentionally non-terminal: at least
    one live handle remains. The workflow stays in the public registry and
    this owned task makes that retention actionable instead of relying on a
    future WebSocket/request to happen to call close again.
    """
    existing = getattr(self, "_close_retry_task", None)
    if existing is not None and not existing.done():
        return

    async def _retry() -> None:
        delay = 0.5
        try:
            while not getattr(self, "_close_complete", False):
                await asyncio.sleep(delay)
                try:
                    result = await self.close(
                        save_runtime=not bool(
                            getattr(self, "_discard_runtime_on_close", False)
                            or getattr(self, "_restore_superseded", False)
                        )
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "[%s] Automatic workflow cleanup retry failed",
                        self.session_id,
                    )
                    result = False
                if result is not False:
                    return
                delay = min(delay * 2.0, 30.0)
        finally:
            self._close_retry_task = None

    self._close_retry_task = asyncio.create_task(
        _retry(),
        name=f"workflow-close-retry:{self.session_id}",
    )

async def close(self, *, save_runtime: bool = True) -> bool:
    """Run close as an owned task so caller cancellation cannot orphan a gate.

    ``save_runtime=False`` is reserved for an already-authorized discard
    such as snapshot supersession, session deletion, or explicit restart
    from a clean analysis generation.
    """
    if not save_runtime:
        self._discard_runtime_on_close = True
        discard_event = getattr(self, "_close_discard_event", None)
        if discard_event is None:
            discard_event = asyncio.Event()
            self._close_discard_event = discard_event
        discard_event.set()
    discard_requested = bool(
        getattr(self, "_discard_runtime_on_close", False)
        or getattr(self, "_restore_superseded", False)
    )
    if discard_requested and getattr(
        self, "_runtime_persistence_gate_active", False
    ):
        # Wake an owned close task that is already awaiting a user action.
        # Restore/delete authority supersedes that action without cancelling
        # the task and leaving a durable gate with no consumer.
        self._runtime_persistence_action_pending = True
        self._runtime_persistence_action = "discard"
        self._runtime_persistence_event.set()
    if getattr(self, "_close_complete", False):
        return True

    task = getattr(self, "_close_task", None)
    if task is None or task.done():
        task = asyncio.create_task(
            self._close_owned_resources(),
            name=f"workflow-close:{self.session_id}",
        )
        self._close_task = task
    # The workflow owns this cleanup. A WebSocket/request task being
    # cancelled must not cancel the only consumer of an armed persistence
    # gate; callers still receive their own cancellation immediately.
    return await asyncio.shield(task)

async def _close_secondary_resources(self, close_status, close_resource) -> None:
    """Close database and provider resources after the sandbox is settled."""
    db_conn = getattr(self, "_db_conn", None)
    if db_conn:
        compacted = await close_resource(
            "db_compaction",
            "compacting sqlite checkpoints",
            self._compact_checkpoints,
        )
        db_closed = await close_resource(
            "db", "closing sqlite connection", db_conn.close
        )
        if db_closed:
            self._db_conn = None
            # Compaction is maintenance, not a separately owned resource. If
            # it failed but the connection closed, no live handle remains.
            if not compacted:
                close_status["db_compaction"] = True
    else:
        close_status["db"] = True
        close_status["db_compaction"] = True

    llm_manager = getattr(self, "llm_manager", None)
    if llm_manager:
        manager_closed = await close_resource(
            "llm_manager", "closing LLM manager", llm_manager.close
        )
        if manager_closed:
            logger.info(f"[{self.session_id}] LLM manager closed")
    else:
        close_status["llm_manager"] = True

    # Close literature-owned runtime clients before provider cleanup.
    literature_agent = getattr(self, "literature_agent", None)
    if literature_agent:
        close_runtime_clients = getattr(
            literature_agent, "close_runtime_clients", None
        )
        if close_runtime_clients:
            await close_resource(
                "literature_clients",
                "closing literature runtime clients",
                close_runtime_clients,
            )
        else:
            close_status["literature_clients"] = True
    else:
        close_status["literature_clients"] = True

    # Close per-agent LLM clients managed by the settings provider.
    settings_provider = getattr(self, "_settings_provider", None)
    if settings_provider:
        await close_resource(
            "settings_provider",
            "closing settings-provider clients",
            settings_provider.cleanup,
        )
    else:
        close_status["settings_provider"] = True


async def _close_owned_resources(self) -> bool:
    """Close owned resources, preserving a kernel until its final save lands.

    Returns ``True`` only when every owned resource is closed. A caller that
    receives ``False`` must retain the workflow so a later cleanup retry can
    reach any resource that is still live.
    """
    close_lock = getattr(self, "_close_lock", None)
    if close_lock is None:
        close_lock = asyncio.Lock()
        self._close_lock = close_lock
    async with close_lock:
        if getattr(self, "_close_complete", False):
            return True
        logger.info(f"[{self.session_id}] Closing workflow and cleaning up resources")
        cancellation: Optional[asyncio.CancelledError] = None
        close_status = getattr(self, "_close_status", None)
        if close_status is None:
            close_status = {}
            self._close_status = close_status

        async def _close_resource(key: str, label: str, closer) -> bool:
            nonlocal cancellation
            if close_status.get(key, False):
                return True
            try:
                await closer()
                close_status[key] = True
                return True
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                consume_current_cancellation_request()
                logger.warning(
                    "[%s] Cancellation received while %s; continuing owned-resource cleanup",
                    self.session_id,
                    label,
                )
                return False
            except Exception:
                logger.exception("[%s] Error %s", self.session_id, label)
                return False

        async def _await_close_critical(awaitable, *, label: str):
            """Defer caller cancellation until one gate-critical await ends.

            ``close()`` shields the owned close task from ordinary request
            cancellation, but shutdown code and tests may cancel that task
            directly. Once a durable final-save gate is armed, abandoning
            its only consumer would strand a reconnect-visible gate with no
            live kernel. Keep the child operation owned, remember the first
            cancellation, and re-raise only after cleanup reaches a safe
            terminal boundary.
            """
            nonlocal cancellation
            owned = asyncio.ensure_future(awaitable)
            while not owned.done():
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                    consume_current_cancellation_request()
                    logger.warning(
                        "[%s] Cancellation received while %s; deferring "
                        "it until the persistence gate reaches a durable "
                        "terminal state",
                        self.session_id,
                        label,
                    )
                except BaseException:
                    break
            return owned.result()

        # sandbox.stop() ends with save_session() + a notebook save. After a
        # restore superseded this run those writes would land on top of the pkl
        # and notebook apply() just installed and _verify_live_restore verified —
        # the same hazard the restore path avoids by tearing down before apply.
        # Shut the kernel down, but do not let it persist itself.
        from ...services.snapshot_v2.safe_point import (
            get_snapshot_safe_point_manager as _sp_manager,
        )

        def _is_superseded() -> bool:
            return (
                bool(getattr(self, "_discard_runtime_on_close", False))
                or _sp_manager().is_superseded(self.session_id)
                or bool(getattr(self, "_restore_superseded", False))
            )

        sandbox = getattr(self, "sandbox", None)
        if sandbox and not close_status.get("sandbox", False):
            from ...services.runtime_persistence_gate import (
                RuntimePersistenceShutdownBlocked,
            )

            prior_status: Optional[TaskStatus] = None
            persistence_failures = 0
            while not close_status.get("sandbox", False):
                try:
                    if _is_superseded():
                        await sandbox.stop(save=False)
                    else:
                        await sandbox.stop(save=True)
                    close_status["sandbox"] = True
                    logger.info(f"[{self.session_id}] Sandbox stopped")
                    break
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                    consume_current_cancellation_request()
                    logger.warning(
                        "[%s] Cancellation received while stopping sandbox; "
                        "the live kernel remains registered for cleanup retry",
                        self.session_id,
                    )
                    break
                except RuntimePersistenceShutdownBlocked:
                    # A restore/delete may have superseded the workflow while
                    # the final save was in flight. Its installed generation is
                    # authoritative, so retry as an explicit no-save shutdown.
                    if _is_superseded():
                        continue

                    analysis_state = getattr(
                        self, "_current_analysis_state", None
                    )
                    if analysis_state is None:
                        from backend.state import sessions

                        analysis_state = sessions.get(self.session_id)
                    if analysis_state is None:
                        logger.exception(
                            "[%s] Final runtime save failed but no analysis "
                            "state exists to host the intervention gate",
                            self.session_id,
                        )
                        break

                    reachable, _registered_here = (
                        await self._register_close_persistence_gate(
                            analysis_state
                        )
                    )
                    if not reachable:
                        # A replacement state/workflow owns the session. This
                        # object is stale and must not write over it.
                        self._restore_superseded = True
                        continue

                    if prior_status is None:
                        prior_status = getattr(
                            analysis_state, "status", TaskStatus.RUNNING
                        )
                    persistence_failures += 1
                    try:
                        action = await _await_close_critical(
                            self._wait_for_runtime_persistence_intervention(
                                analysis_state,
                                successful_status=prior_status,
                                persist_transitions=True,
                                rotate_generation=persistence_failures > 1,
                            ),
                            label="waiting for final-save intervention",
                        )
                    except WorkflowSupersededError:
                        # The safe-point hook can raise from inside this
                        # persistence-handler block. Convert it into the
                        # same explicit no-save shutdown used by the normal
                        # supersession check, then continue cleanup.
                        self._restore_superseded = True
                        continue
                    except asyncio.CancelledError as exc:
                        cancellation = cancellation or exc
                        consume_current_cancellation_request()
                        logger.warning(
                            "[%s] Runtime-persistence close gate was cancelled; "
                            "preserving the live kernel for cleanup retry",
                            self.session_id,
                        )
                        break
                    except Exception:
                        logger.exception(
                            "[%s] Runtime-persistence close gate failed; "
                            "preserving the live kernel for cleanup retry",
                            self.session_id,
                        )
                        break
                    if action == "retry":
                        # Retry already committed the namespace. Re-enter
                        # stop() so it verifies/reuses that generation and
                        # saves notebook metadata before shutdown.
                        continue

                    if action == "terminate":
                        self._set_status(analysis_state, TaskStatus.STOPPED)
                        analysis_state.termination_reason = (
                            "runtime_persistence_failure"
                        )
                        delay = 0.1
                        while not _is_superseded():
                            persisted = await _await_close_critical(
                                self._publish_runtime_persistence_transition(
                                    analysis_state,
                                    persist_state=True,
                                    abort_on_close_discard=True,
                                ),
                                label=(
                                    "persisting final-save termination"
                                ),
                            )
                            if persisted:
                                break
                            await _await_close_critical(
                                asyncio.sleep(delay),
                                label=(
                                    "retrying final-save termination "
                                    "persistence"
                                ),
                            )
                            delay = min(delay * 2.0, 1.0)
                    # Recompute/manage and terminate are both explicit
                    # authorization to discard this unsaved live generation.
                    await _await_close_critical(
                        sandbox.stop(save=False),
                        label="discarding an explicitly released runtime",
                    )
                    close_status["sandbox"] = True
                    logger.info(
                        "[%s] Sandbox stopped after explicit runtime "
                        "persistence action %s",
                        self.session_id,
                        action,
                    )
                    break
                except Exception:
                    logger.exception(
                        "[%s] Error stopping sandbox", self.session_id
                    )
                    break
        else:
            if not sandbox:
                close_status["sandbox"] = True

        await _close_secondary_resources(self, close_status, _close_resource)

        self._close_complete = all(
            close_status.get(key, False)
            for key in (
                "sandbox",
                "db",
                "llm_manager",
                "literature_clients",
                "settings_provider",
            )
        )
        if not self._close_complete:
            analysis_state = getattr(self, "_current_analysis_state", None)
            if analysis_state is None:
                from backend.state import sessions

                analysis_state = sessions.get(self.session_id)
            reachable = False
            if analysis_state is not None:
                reachable, _registered_here = (
                    await self._register_close_persistence_gate(
                        analysis_state
                    )
                )
            if not reachable:
                # A replacement state/workflow or tombstone owns this
                # session now. The stale object still owns resources that
                # must be retried, but it must never save its runtime over
                # the replacement generation.
                self._restore_superseded = True
            self._ensure_close_retry_task()
        if self._close_complete:
            await self._release_close_persistence_gate_registration()
        if cancellation is not None:
            raise cancellation
        return self._close_complete

async def _compact_checkpoints(
    self,
    keep: int = CHECKPOINTS_TO_KEEP_PER_NAMESPACE,
    *,
    vacuum: bool = True,
) -> None:
    """Retain latest graph checkpoints; optionally reclaim the main DB."""
    if self._db_conn is None or keep < 1:
        return
    await compact_checkpoint_history(
        self._db_conn,
        keep=keep,
        vacuum=vacuum,
    )
    logger.info(
        "[%s] Compacted LangGraph checkpoints to %d per namespace",
        self.session_id,
        keep,
    )

def get_notebook_path(self) -> Optional[str]:
    """Get path to the saved Jupyter notebook"""
    if self.sandbox:
        return self.sandbox.get_notebook_path()
    return None
