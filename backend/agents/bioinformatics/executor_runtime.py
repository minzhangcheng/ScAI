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
    from backend.services.adata_checkpoints.reconcile import (
        reconcile_live_runtime_state,
    )

    if await reconcile_live_runtime_state(agent.sandbox, state):
        pending_rebase = dict(
            getattr(agent.sandbox, "_adata_lineage_rebase_pending", None) or {}
        )
        if pending_rebase:
            try:
                await agent.sandbox.save_session()
                agent.sandbox.sync_runtime_persistence_state(state)
                from backend.utils.runtime_manifest import (
                    read_runtime_manifest,
                    runtime_manifest_path,
                )

                manifest = read_runtime_manifest(
                    runtime_manifest_path(str(state.id))
                )
                pairing = manifest.get("pairing") or {}
                preparation_ids = set(
                    pending_rebase.get("preparation_ids") or []
                )
                binding_ids = set(pending_rebase.get("binding_ids") or [])
                for preparation in (
                    getattr(state, "adata_source_preparations", None) or []
                ):
                    if preparation.preparation_id not in preparation_ids:
                        continue
                    preparation.status = "durable"
                    preparation.reconcile_from_session_lineage_id = None
                    preparation.lifecycle_generation = int(
                        getattr(state, "lifecycle_generation", 0) or 0
                    )
                    preparation.runtime_instance_id = getattr(
                        agent.sandbox, "runtime_instance_id", None
                    )
                    preparation.runtime_revision = int(
                        getattr(agent.sandbox, "runtime_revision", 0) or 0
                    )
                    preparation.runtime_manifest_revision = int(
                        manifest.get("revision") or 0
                    )
                    preparation.runtime_pkl_sha256 = str(
                        manifest.get("pkl_sha256") or ""
                    ) or None
                    preparation.runtime_pairing_state_version = pairing.get(
                        "state_version"
                    )
                    preparation.runtime_pairing_checkpoint = pairing.get(
                        "checkpoint"
                    )
                    preparation.registry_saved_name_ok = True
                    preparation.binding_saved_names_ok = True
                for binding in getattr(state, "adata_live_bindings", None) or []:
                    if binding.binding_id not in binding_ids:
                        continue
                    binding.status = "registered"
                    binding.reconcile_from_session_lineage_id = None
                agent.sandbox._adata_lineage_rebase_pending = None
            except Exception:
                logger.warning(
                    "Could not durably publish child AnnData registry ownership",
                    exc_info=True,
                )
        try:
            await state.increment_version()
            from backend.services.session_store import async_save_session

            await async_save_session(str(state.id), state, force=True)
        except Exception:
            logger.warning(
                "Could not persist live AnnData source reconciliation",
                exc_info=True,
            )
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
