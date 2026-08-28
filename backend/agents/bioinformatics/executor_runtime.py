"""Non-tool-dispatch runtime setup for the autonomous executor."""
from __future__ import annotations


async def start_sandbox(agent) -> dict:
    if agent.sandbox.km is not None:
        return {}
    await agent._notify("status", "Starting analysis environment...")
    return await agent.sandbox.start()


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

    # A verified older generation may be the only runtime that survived a
    # newer failed save. Do not silently call that fully current: stop at a
    # durable HITL gate unless the user already chose recompute/reduce.
    persistence = agent.sandbox.sync_runtime_persistence_state(state)
    if getattr(state, "runtime_recompute_pending", False) and persistence.get(
        "requires_recompute"
    ):
        from backend.services.runtime_persistence_gate import (
            prepare_runtime_recompute,
        )

        # A process restart leaves the notebook metadata as the
        # authoritative failed/degraded record while the user's explicit
        # Recompute action lives in AnalysisState. Apply that authorization
        # to the newly-created sandbox before the executor loop checks the
        # latch, otherwise it immediately re-arms the same gate.
        prepare_runtime_recompute(state, agent.sandbox)
        persistence = agent.sandbox.sync_runtime_persistence_state(state)
    if (
        (
            persistence.get("degraded_recovery")
            or persistence.get("status") == "failed"
            or persistence.get("requires_recompute")
        )
        and not getattr(state, "runtime_recompute_pending", False)
    ):
        from backend.services.runtime_persistence_gate import (
            RuntimePersistenceBlocked,
            enter_runtime_persistence_gate,
        )

        enter_runtime_persistence_gate(
            state,
            persistence,
            degraded_recovery=bool(persistence.get("degraded_recovery")),
        )
        raise RuntimePersistenceBlocked(
            "Runtime persistence requires an explicit recovery action"
        )
