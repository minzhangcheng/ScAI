"""Non-tool-dispatch runtime setup for the autonomous executor."""
from __future__ import annotations


async def start_sandbox(agent) -> dict:
    if agent.sandbox.km is not None:
        return {}
    await agent._notify("status", "Starting analysis environment...")
    return await agent.sandbox.start()


def _runtime_marked_unavailable(state, session_id: str) -> bool:
    """Whether either view of this session says no runtime could be restored.

    ``_mark_runtime_unavailable_if_work_exists`` runs inside ``sandbox.start()``
    and can only reach the cached ``AnalysisState`` in ``backend.state.sessions``.
    The graph carries its own object -- ``_restore_entry`` deep-copies one -- and
    the two coincide only from the next state sync onward. Reading the graph's
    object alone therefore lets a verdict recorded moments earlier, by the very
    call this guard follows, go unseen: the run would continue over a namespace
    that does not exist, which is exactly what the mark exists to prevent.
    """
    from backend.state import sessions as live_sessions

    for candidate in (state, live_sessions.get(session_id)):
        if candidate is None:
            continue
        if getattr(candidate, "runtime_restore_status", "complete") == "unavailable":
            return True
    return False


async def prepare_sandbox(agent, state, *, bindings) -> None:
    logger = getattr(bindings, "logger")
    CodeExecution = getattr(bindings, "CodeExecution")
    # Start sandbox if needed
    if agent.sandbox.km is None:
        start_result = await start_sandbox(agent)
        logger.info("[Bioinformatics] Sandbox started")
        if start_result.get("restore_failed"):
            logger.warning("[Bioinformatics] Previous session restore failed, starting with fresh environment")
            await agent._notify("status", "Note: Previous session could not be restored, starting fresh...")
        elif start_result.get("session_restored") and not agent.code_history:
            # Populate code_history from restored notebook cells so _build_notebook_context()
            # shows previous analysis, preventing the agent from restarting from scratch
            restored_cells = agent.sandbox.get_code_history()
            for cell in restored_cells:
                outputs = cell.get('outputs', [])
                output_text = '\n'.join(
                    o.get('text', '') for o in outputs if o.get('type') == 'stream'
                )
                error_text = '\n'.join(
                    o.get('evalue', '') for o in outputs if o.get('type') == 'error'
                )
                figures = [
                    o.get('data', {}).get('image/png', '')
                    for o in outputs
                    if o.get('type') == 'display_data' and 'image/png' in o.get('data', {})
                ]
                agent.code_history.append(CodeExecution(
                    code=cell.get('code', ''),
                    output=output_text if output_text else None,
                    error=error_text if error_text else None,
                    figures=figures,
                    execution_time=None,
                ))
            logger.info(f"[Bioinformatics] Restored {len(agent.code_history)} cells into code_history from notebook state")

    # No committed generation at all is the one condition that still stops
    # forward execution, and it is answered by an explicit re-run rather than a
    # dialog: there is nothing to retry, because nothing was ever saved.
    persistence = agent.sandbox.sync_runtime_persistence_state(state)
    if (
        _runtime_marked_unavailable(state, agent.sandbox.session_id)
        and not getattr(state, "runtime_recompute_pending", False)
    ):
        from backend.services.runtime_recompute import RuntimeUnavailable

        raise RuntimeUnavailable("No runtime survived; explicit re-run is required")
    if getattr(state, "runtime_recompute_pending", False) and persistence.get(
        "requires_recompute"
    ):
        from backend.services.runtime_recompute import (
            prepare_runtime_recompute,
        )

        # A process restart leaves the notebook metadata as the authoritative
        # failed record while the user's explicit re-run authorization lives in
        # AnalysisState. Apply it to the newly created sandbox so the executor
        # loop reads the authorization rather than the stale record.
        prepare_runtime_recompute(state, agent.sandbox)
        agent.sandbox.sync_runtime_persistence_state(state)
    # A failed or degraded newest save no longer holds the run. The previous
    # committed generation is what continues from, coherence reports it as
    # stale, and the UI carries a persistent non-blocking warning (spec 5.1).
