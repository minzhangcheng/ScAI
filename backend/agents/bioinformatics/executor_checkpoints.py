"""Read-only Agent lookup over the shared AnnData checkpoint projection."""
from __future__ import annotations

import json

from .executor_dispatch import ToolAction
from .shared import AnalysisStep


async def handle_list_adata_checkpoints(context, tool_call) -> ToolAction:
    args = tool_call.arguments or {}
    try:
        from ...services.adata_checkpoints import build_checkpoint_projection

        payload = build_checkpoint_projection(
            context.state,
            page=args.get("page", 1),
            page_size=args.get("page_size", 20),
            checkpoint_id=args.get("checkpoint_id"),
            live_binding_id=args.get("live_binding_id"),
        )
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        content = json.dumps(
            {
                "error": "checkpoint_lookup_failed",
                "detail": type(exc).__name__,
            },
            ensure_ascii=False,
        )
    context.messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": content,
    })
    context.analysis_log.append(AnalysisStep(
        step_number=context.step_count,
        step_type="list_adata_checkpoints",
        agent_reasoning=context.agent_reasoning,
        code_description="Read-only AnnData checkpoint lookup",
    ))
    return ToolAction.CONTINUE
