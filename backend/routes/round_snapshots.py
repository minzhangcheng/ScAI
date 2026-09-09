"""Decision-snapshot routes. Part B spec 4.8.

There is deliberately no create endpoint. Capture is automatic at the four
decision points and nowhere else, so a route that made one on request would be
a second producer with different semantics -- one that could run while the
kernel was mid-cell, against a generation nothing had committed.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from backend.services.round_snapshots import read_session_listing
from backend.state import import_jobs as _import_jobs, import_jobs_lock as _import_jobs_lock
from backend.utils.asyncio_tasks import run_in_owned_thread
from backend.utils.auth import require_session_ownership
from backend.utils.files import validate_session_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["snapshots"])


async def _owned_listing(session_id: str, request: Request) -> dict:
    from backend.config import snapshots_enabled

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    if not snapshots_enabled():
        return {"enabled": False, "snapshots": [], "failed": []}
    listing = await run_in_owned_thread(read_session_listing, session_id)
    return {"enabled": True, **listing}


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
    listing = await _owned_listing(session_id, request)
    if listing["enabled"]:
        listing["restored_sessions"] = await run_in_owned_thread(
            _restored_from, session_id,
        )
    return listing


def _restored_from(session_id: str) -> dict[str, list[str]]:
    """Which sessions each of this one's snapshots was forked into.

    Derived from the sessions rather than recorded on the snapshot: a snapshot
    is immutable once published, and a restore must not have to write to the
    source session -- which may be running, and which this feature promises not
    to modify.
    """
    from backend.services.session_store import list_saved_sessions

    forks: dict[str, list[str]] = {}
    for summary in list_saved_sessions():
        origin = str(summary.get("last_restored_snapshot_id") or "")
        prefix = f"{session_id}/"
        if origin.startswith(prefix):
            forks.setdefault(origin[len(prefix):], []).append(str(summary["id"]))
    return forks


@router.post("/{session_id}/snapshots/{snapshot_id}/restore")
async def restore_session_snapshot(session_id: str, snapshot_id: str, request: Request):
    """Fork this session as it was before one decision. The source is untouched.

    Reported through the import-job registry, because a restore is an import
    from a local bundle and the frontend already renders that progress; the
    terminal record carries ``new_session_id``.
    """
    from backend.config import snapshots_enabled
    from backend.routes.export_import_parts.core import import_job_record
    from backend.services.round_snapshots.restore import (
        SnapshotRestoreError,
        restore_snapshot_into_new_session,
    )

    from backend.services.round_snapshots.layout import (
        parse_snapshot_id,
        snapshot_dir,
    )

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    if not snapshots_enabled():
        raise HTTPException(status_code=409, detail="Snapshots are not enabled")
    # Answered here rather than by the job. A caller that has to poll to learn
    # the id does not exist is being told asynchronously what the server knew
    # before it started. Everything the job *cannot* know synchronously -- a
    # tampered manifest, a digest that has drifted -- is still its to report.
    try:
        parse_snapshot_id(snapshot_id)
        exists = snapshot_dir(session_id, snapshot_id).is_dir()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not exists:
        raise HTTPException(status_code=404, detail="No such snapshot")

    job_id = str(uuid.uuid4())
    async with _import_jobs_lock:
        _import_jobs[job_id] = import_job_record(
            "restoring",
            user_id=user_id,
            created_at=time.time(),
            kind="snapshot_restore",
            source_session_id=session_id,
            snapshot_id=snapshot_id,
            bytes_processed=0,
            cancel_requested=False,
        )

    def _progress(written: int) -> None:
        job = _import_jobs.get(job_id)
        if job is not None:
            job["bytes_processed"] = int(job.get("bytes_processed") or 0) + int(written)

    async def _run() -> None:
        created_at = _import_jobs.get(job_id, {}).get("created_at", time.time())
        try:
            new_id = await restore_snapshot_into_new_session(
                source_session_id=session_id,
                snapshot=snapshot_id,
                user_id=user_id,
                progress=_progress,
            )
        except SnapshotRestoreError as refused:
            logger.warning(
                "Snapshot restore %s/%s refused: %s", session_id, snapshot_id, refused,
            )
            async with _import_jobs_lock:
                _import_jobs[job_id] = import_job_record(
                    "failed", user_id=user_id, created_at=created_at,
                    error=refused.detail, kind="snapshot_restore",
                    code=refused.reason,
                )
        except Exception as exc:
            logger.exception("Snapshot restore %s/%s failed", session_id, snapshot_id)
            async with _import_jobs_lock:
                _import_jobs[job_id] = import_job_record(
                    "failed", user_id=user_id, created_at=created_at,
                    error=str(exc), kind="snapshot_restore",
                )
        else:
            async with _import_jobs_lock:
                _import_jobs[job_id] = import_job_record(
                    "completed", user_id=user_id, created_at=created_at,
                    kind="snapshot_restore", new_session_id=new_id,
                    source_session_id=session_id, snapshot_id=snapshot_id,
                )

    asyncio.create_task(_run(), name=f"snapshot-restore:{session_id}:{snapshot_id}")
    return {"status": "restoring", "job_id": job_id, "import_id": job_id}


@router.delete("/{session_id}/snapshots/{snapshot_id}")
async def delete_session_snapshot(session_id: str, snapshot_id: str, request: Request):
    """Give up one restore point. Nothing else is touched.

    Renamed aside before removal, so a caller never sees a directory being
    emptied under it. A dump shared with a neighbouring snapshot survives:
    the filesystem reference counts it, which is the whole reason sharing is a
    hardlink rather than a record.
    """
    from backend.config import snapshots_enabled
    from backend.services.round_snapshots.layout import parse_snapshot_id, snapshot_dir

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)
    if not snapshots_enabled():
        raise HTTPException(status_code=409, detail="Snapshots are not enabled")
    try:
        parse_snapshot_id(snapshot_id)
        bundle = snapshot_dir(session_id, snapshot_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not bundle.is_dir():
        raise HTTPException(status_code=404, detail="No such snapshot")

    def _remove() -> None:
        import shutil

        doomed = bundle.with_name(f".deleting-{bundle.name}")
        shutil.rmtree(doomed, ignore_errors=True)
        bundle.rename(doomed)
        shutil.rmtree(doomed, ignore_errors=True)

    await run_in_owned_thread(_remove)
    return {"status": "deleted", "snapshot_id": snapshot_id}
