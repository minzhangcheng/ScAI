"""Agent selection of a missing canonical binding before checkpoint capture."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def select_checkpoint_adata(agent, sandbox, state, *, boundary, summary):
    """Return inspected identity metadata; never execute model-generated code.

    The kernel lock covers inspection only, not the LLM call. The caller must
    revalidate and bind under the same lock as checkpoint capture.
    """
    if sandbox is None or getattr(sandbox, "km", None) is None:
        return None
    inspect = getattr(sandbox, "inspect_checkpoint_adata_locked", None)
    if not callable(inspect):
        return None
    try:
        lock = getattr(sandbox, "_exec_lock", None)
        if lock is None:
            inventory = await inspect()
            revision = getattr(sandbox, "runtime_revision", None)
        else:
            async with lock:
                inventory = await inspect()
                revision = getattr(sandbox, "runtime_revision", None)
        if inventory.get("error") or inventory.get("canonical_ready"):
            return None
        candidates = inventory.get("candidates") or []
        if not candidates or agent is None:
            return None
        plan = getattr(state, "current_plan", None)
        context = {
            "boundary": boundary,
            "completed_analysis_summary": str(summary or "")[:12000],
            "approved_plan": plan.model_dump(mode="json") if plan is not None else None,
            "source_preparations": [
                item.model_dump(mode="json")
                for item in (getattr(state, "adata_source_preparations", None) or [])[-5:]
            ],
            "inventory": inventory,
        }
        agent.refresh_runtime_config_if_needed()
        response = await agent.chat_without_history(
            "Before saving an AnnData checkpoint, check the live object inventory. "
            "The canonical adata is absent or technically unsuitable (see inventory). Select the completed primary analysis "
            "AnnData to expose as adata, using the approved plan, completion summary "
            "and metadata. This creates an alias to the same object, not a copy or "
            "new analysis. Do not select a temporary subset or an independent "
            "checkpoint_copy working source as the canonical object. Do not choose "
            "by name or size alone. Check technical readiness (consistent axes, not "
            "a view or backed object). Scientific limitations do not prevent backup. "
            "If the primary object cannot be identified or is not technically ready, "
            "return null and explain. Return JSON only: "
            '{"binding_name": "exact inventory name", "reason": "explanation"}; '
            'use JSON null for binding_name when no candidate is suitable.\n'
            + json.dumps(context, ensure_ascii=False, default=str),
        )
        choice = agent._safe_parse_json(response, default={})
        name = choice.get("binding_name") if isinstance(choice, dict) else None
        selected = next((item for item in candidates if item["name"] == name), None)
        logger.info("AnnData checkpoint preflight (%s): %s", boundary, choice)
        return dict(
            selected, runtime_revision=revision,
            canonical_identity=inventory.get("canonical_identity"),
        ) if selected else None
    except Exception:
        logger.warning("AnnData checkpoint preflight failed; capture will disclose availability", exc_info=True)
        return None


async def bind_selected_checkpoint_adata(sandbox, selected):
    """Revalidate and apply a selection; caller owns the kernel execution lock."""
    if selected is None:
        return None
    if selected["runtime_revision"] != getattr(sandbox, "runtime_revision", None):
        logger.warning("AnnData checkpoint selection expired after a runtime change")
        return None
    try:
        result = await sandbox.bind_checkpoint_adata_locked(
            name=selected["name"], identity=selected["identity"],
            canonical_identity=selected.get("canonical_identity"),
        )
        logger.info("AnnData checkpoint canonical binding: %s", result)
        return result
    except Exception:
        logger.warning("Could not establish canonical AnnData binding", exc_info=True)
        return None


async def ensure_checkpoint_adata(agent, sandbox, state, *, boundary, summary):
    """Prepare the main namespace before a consumer pins its runtime generation."""
    selected = await select_checkpoint_adata(
        agent, sandbox, state, boundary=boundary, summary=summary,
    )
    if selected is None:
        return None
    lock = getattr(sandbox, "_exec_lock", None)
    if lock is None:
        return await bind_selected_checkpoint_adata(sandbox, selected)
    async with lock:
        return await bind_selected_checkpoint_adata(sandbox, selected)
