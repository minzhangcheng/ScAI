# Shared router/logger and the chat context builder.
# Moved verbatim out of backend/routes/chat.py.
"""Chat routes for PI Agent free-form chat with analysis sessions."""
import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from backend.config import get_session_workspace, get_session_uploads, get_session_states, PI_AGENT_NAME
from backend.models.schemas import WebSocketMessage, AgentType, DataFile, AnalysisState
from backend.services.session_store import ensure_session_loaded, async_save_session, update_session_cache
from backend.state import (
    chat_tasks, chat_task_lock, chat_sandboxes,
    websockets, workflows, state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.files import safe_filename, validate_session_id
from backend.utils.upload import stream_save_file
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL
from backend.services.report.paths import resolve_published_report_dir


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["chat"])

def _build_chat_context(
    state: Any,
    direction_context: str = "",
) -> tuple:
    """Build analysis context summary for PI Agent chat.

    Returns:
        (context_str, figure_inventory) where figure_inventory is a list of
        {"name": str, "path": str} dicts for PNG files found on disk.
    """
    parts = []

    if state.initial_instruction:
        parts.append(f"Research Goal: {state.initial_instruction}")

    parts.append(f"Current Round: {state.current_round}")
    parts.append(f"Status: {state.status.value}")

    # Experimental design summary
    design_summary = state.build_experimental_design_summary()
    if design_summary:
        parts.append(design_summary)

    # Biochemistry indicators summary
    if state.biochemistry_indicators:
        indicators = [f"{r.indicator}={r.value}{r.unit}" for r in state.biochemistry_indicators]
        parts.append(f"Biochemistry: {', '.join(indicators)}")

    # Round results (last 3)
    for rr in state.round_results[-3:]:
        round_parts = [f"\n--- Round {rr.round_number} Summary ---"]
        if rr.pathology_findings:
            round_parts.append(f"Pathology: {'; '.join(rr.pathology_findings[:3])}")
        if rr.bioinformatics_findings:
            round_parts.append(f"Bioinformatics: {'; '.join(rr.bioinformatics_findings[:3])}")
        if rr.biochemistry_findings:
            round_parts.append(f"Biochemistry: {'; '.join(rr.biochemistry_findings[:3])}")
        if rr.literature_findings:
            round_parts.append(f"Literature: {'; '.join(rr.literature_findings[:3])}")
        if rr.pi_summary:
            round_parts.append(f"PI Summary: {rr.pi_summary[:500]}")
        parts.extend(round_parts)

    # Final summary
    if state.final_summary:
        parts.append(f"\nFinal Summary: {state.final_summary[:1000]}")

    # A later Chat request starts a new executor transcript. Give the PI the
    # durable prior review and receipt identities needed to continue useful work.
    current_branch = getattr(state, "restore_branch_id", None) or "main"
    recent_reviews = [
        review for review in (getattr(state, "chat_analysis_reviews", None) or [])
        if review.get("branch_id", "main") == current_branch
    ][-3:]
    for review in recent_reviews:
        execution_ids = [
            execution_id
            for cycle in review.get("cycles", [])
            for execution_id in cycle.get("execution_ids", [])
        ]
        parts.append(
            "\nPrevious Chat analysis: "
            f"review_id={review.get('id')}; status={review.get('status')}; "
            f"execution_incomplete={bool(review.get('execution_incomplete'))}; "
            f"request={str(review.get('request') or '')[:500]}; "
            f"execution_ids={execution_ids[-12:]}; "
            f"unexecuted_revision={str(review.get('unexecuted_revision') or '')[:500]}. "
            "Use these as prior evidence, not as proof that unfinished work ran."
        )

    # A report follow-up may refer to a direction from an older round.  The
    # caller resolves that context against the owned session before any chat
    # mutation; keep the optional block separate from the ordinary recency
    # summary so it cannot be supplied by the client.
    if isinstance(direction_context, str) and direction_context.strip():
        parts.append(
            "\nServer-resolved selected-direction context:\n"
            + direction_context.strip()
        )

    # Literature reports
    if state.literature_reports:
        parts.append(f"\nLiterature Reports: {len(state.literature_reports)} available")
        for lr in state.literature_reports[-2:]:
            parts.append(f"- Topic: {lr.topic}")

    # Scan disk for actual figure files
    figure_inventory: List[Dict[str, str]] = []
    seen_names: set = set()
    from backend.services.report.artifacts import eligible_artifacts, verified_artifact_bytes

    publication_by_path = {
        str(Path(figure.png_path).resolve()): figure
        for figure in eligible_artifacts(
            getattr(state, "publication_figures", None), state=state,
        )
        if verified_artifact_bytes(figure.png_path, figure, kind="image") is not None
    }

    def _add_pngs(directory: Path):
        if not directory.exists():
            return
        # Scan JPG first (new format), then PNG for backward compatibility
        for img in sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png")):
            if img.stem not in seen_names:
                seen_names.add(img.stem)
                entry = {"name": img.stem, "path": str(img),
                         "figure_role": "quick_display"}
                artifact = publication_by_path.get(str(img.resolve()))
                if artifact is not None:
                    entry.update({
                        "figure_role": "publication",
                        "artifact_id": artifact.id,
                        "review_status": artifact.review_status.value,
                    })
                figure_inventory.append(entry)

    _add_pngs(get_session_workspace(state.id))
    _add_pngs(get_session_workspace(state.id) / "figures")
    _add_pngs(
        resolve_published_report_dir(
            state.id,
            getattr(state, "restore_branch_id", None),
        ) / "figures"
    )

    if figure_inventory:
        parts.append(f"\nAvailable Figures ({len(figure_inventory)} on disk):")
        for fig in figure_inventory:
            parts.append(f"  - {fig['name']} ({fig['figure_role']})")

    return "\n".join(parts), figure_inventory
