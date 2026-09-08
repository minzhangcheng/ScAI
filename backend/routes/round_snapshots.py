"""Decision-snapshot routes. Part B spec 4.8.

There is deliberately no create endpoint. Capture is automatic at the four
decision points and nowhere else, so a route that made one on request would be
a second producer with different semantics -- one that could run while the
kernel was mid-cell, against a generation nothing had committed.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from backend.services.round_snapshots import read_session_listing
from backend.utils.asyncio_tasks import run_in_owned_thread
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["snapshots"])


@router.get("/{session_id}/snapshots")
async def list_session_snapshots(session_id: str, request: Request):
    """Every decision snapshot this session has, and every capture that failed.

    ``enabled`` is what the frontend renders the panel on: with no snapshot
    directory configured there is nothing to show and no action to offer, and a
    panel that appeared empty would read as "this analysis produced none"
    rather than "this installation does not take them".

    Rows carry ``deletion_order`` and ``retention_rank`` already computed, so
    the browser, the CLI and the space-pressure prune cannot disagree about
    which snapshot to give up first.
    """
    from backend.config import snapshots_enabled

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    if not snapshots_enabled():
        return {"enabled": False, "snapshots": [], "failed": []}

    listing = await run_in_owned_thread(read_session_listing, session_id)
    return {"enabled": True, **listing}
