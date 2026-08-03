# Publication figures/tables, graphic abstract, integrated report, code history.
# Moved verbatim out of backend/routes/reports.py.
"""Report routes: exploration-report, annotation-report, detailed-report, report/*, 
publication-figures, publication-tables, figure/{id}/pdf, figure/{id}/png, integrated-report.

Extracted from backend/main.py to decouple report generation and serving route handlers.
"""
import logging
import os
import asyncio
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from datetime import datetime
from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from backend.config import WORKSPACE_DIR, UPLOAD_DIR, get_session_workspace
from backend.models.schemas import (
    AnalysisState,
    APIResponse,
    LegacyArtifactRecord,
    TaskStatus,
)

from backend.utils.files import validate_path_within, validate_session_id
from backend.utils.auth import require_session_ownership
from ..serialization import serialize_round_results
from backend.services.session_store import _async_save_session_under_lifecycle
from backend.services.session_lifecycle import (
    SessionGoneError,
    SessionGenerationError,
    assert_session_generation_active,
    assert_generation_writable,
    get_lifecycle_lock,
    lifecycle_file_lock,
)
from backend.state import sessions, state_lock, touch_access_time
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_bulk_thread,
    run_in_owned_thread,
)
from backend.services.report.artifacts import (
    archival_render_available,
    gallery_artifact_by_id,
    gallery_artifacts,
    has_blocking_layout_defects,
    resolve_verified_archival_artifact,
    resolve_verified_legacy_artifact,
)
from backend.services.report.figure_discovery import ensure_legacy_artifact_inventory

logger = logging.getLogger(__name__)

_LEGACY_INVENTORY_CACHE_MAX = 64
_legacy_inventory_cache: OrderedDict[
    tuple[str, int, str], tuple[int, tuple[dict, ...]]
] = OrderedDict()
_legacy_inventory_cache_guard = threading.Lock()


from .core import router


async def _ensure_frozen_artifact_inventory(
    session_id: str,
    state: AnalysisState,
    workspace: Path,
) -> None:
    """Apply the one-time compatibility scan to the request's loaded state.

    ``ensure_session_loaded`` performs and durably persists this migration under
    the session lifecycle lock before an authenticated route receives the state.
    Keeping the idempotent call here covers direct/internal callers without
    opening a second, route-local read-modify-write race.
    """
    if getattr(state, "artifact_registry_version", None) is not None:
        return

    generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    cache_key = (
        str(session_id),
        generation,
        os.path.abspath(os.fspath(workspace)),
    )
    async with get_lifecycle_lock(session_id):
        if getattr(state, "artifact_registry_version", None) is not None:
            return
        with _legacy_inventory_cache_guard:
            cached = _legacy_inventory_cache.get(cache_key)
            if cached is not None:
                _legacy_inventory_cache.move_to_end(cache_key)
        if cached is not None:
            await run_in_owned_thread(
                _apply_cached_legacy_inventory,
                session_id,
                generation,
                state,
                cached,
            )
            return

        await run_in_owned_bulk_thread(
            _scan_and_freeze_legacy_inventory,
            session_id,
            generation,
            state,
            workspace,
            cache_key,
        )


def _with_generation_file_barrier(
    session_id: str,
    generation: int,
    operation: Callable[[], Any],
) -> Any:
    """Run one bounded filesystem operation against an exact generation."""
    if generation <= 0:
        return operation()
    with lifecycle_file_lock(session_id):
        assert_session_generation_active(
            session_id, expected_generation=generation,
        )
        result = operation()
        assert_session_generation_active(
            session_id, expected_generation=generation,
        )
        return result


def _inventory_payload(state: AnalysisState) -> tuple[int, tuple[dict, ...]]:
    version = int(getattr(state, "artifact_registry_version", 0) or 0)
    records = tuple(
        item.model_dump(mode="json")
        for item in (getattr(state, "legacy_artifacts", None) or [])
    )
    return version, records


def _remember_legacy_inventory(
    cache_key: tuple[str, int, str],
    frozen: tuple[int, tuple[dict, ...]],
) -> None:
    with _legacy_inventory_cache_guard:
        # A lifecycle generation owns one compatibility scan. Remove older
        # generations for the same session before retaining the bounded result
        # used by concurrent/reloaded route state objects.
        for key in list(_legacy_inventory_cache):
            if key[0] == cache_key[0] and key != cache_key:
                _legacy_inventory_cache.pop(key, None)
        _legacy_inventory_cache[cache_key] = frozen
        _legacy_inventory_cache.move_to_end(cache_key)
        while len(_legacy_inventory_cache) > _LEGACY_INVENTORY_CACHE_MAX:
            _legacy_inventory_cache.popitem(last=False)


def _scan_and_freeze_legacy_inventory(
    session_id: str,
    generation: int,
    state: AnalysisState,
    workspace: Path,
    cache_key: tuple[str, int, str],
) -> tuple[int, tuple[dict, ...]]:
    def scan():
        ensure_legacy_artifact_inventory(state, workspace)
        return _inventory_payload(state)

    frozen = _with_generation_file_barrier(session_id, generation, scan)
    # Store from the owned worker before cancellation is re-delivered to the
    # awaiting route. A completed migration must remain single-flight even if
    # the client that initiated it disconnected during the scan.
    _remember_legacy_inventory(cache_key, frozen)
    return frozen


def _apply_cached_legacy_inventory(
    session_id: str,
    generation: int,
    state: AnalysisState,
    cached: tuple[int, tuple[dict, ...]],
) -> None:
    def apply() -> None:
        version, raw_records = cached
        existing = {
            str(getattr(item, "id", "") or ""): item
            for item in (getattr(state, "legacy_artifacts", None) or [])
        }
        for raw in raw_records:
            record_id = str(raw.get("id") or "")
            if record_id and record_id not in existing:
                record = LegacyArtifactRecord.model_validate(raw)
                state.legacy_artifacts.append(record)
                existing[record_id] = record
        state.artifact_registry_version = version

    _with_generation_file_barrier(session_id, generation, apply)


def _legacy_artifact_by_id(
    state: AnalysisState,
    artifact_id: str,
    lineage_id: str,
):
    return next(
        (
            item for item in (state.legacy_artifacts or [])
            if str(getattr(item, "id", "")) == str(artifact_id)
            and str(getattr(item, "lineage_id", None) or "main") == lineage_id
        ),
        None,
    )


def _legacy_artifact_by_integrated_report_id(
    state: AnalysisState,
    artifact_id: str,
    workspace: Path,
    lineage_id: str,
    *,
    kind: str,
):
    """Resolve a pre-registry message ID through the frozen inventory.

    Old integrated-report messages used an ephemeral workspace figure ID. The
    message supplies only an alias: bytes are served exclusively through the
    path and digest already frozen in ``LegacyArtifactRecord``.
    """
    workspace = workspace.resolve()
    legacy_records = list(state.legacy_artifacts or [])
    for message in state.realtime_messages or []:
        message_type = (
            getattr(message, "type", None)
            if not isinstance(message, dict)
            else message.get("type")
        )
        content = (
            getattr(message, "content", None)
            if not isinstance(message, dict)
            else message.get("content")
        )
        if message_type != "integrated_report" or not isinstance(content, dict):
            continue
        for raw in content.get("figures") or []:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("id") or "") != str(artifact_id):
                continue
            # 0.38.9 added this discriminator; older workspace entries had no
            # flag.  An explicit True is rejected, while a missing flag still
            # has to resolve to an unregistered, frozen workspace inventory
            # record below.
            publication_flag = raw.get("is_publication")
            if publication_flag is not None and publication_flag is not False:
                continue
            raw_path = str(raw.get("png_path") or "")
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = workspace / candidate
            try:
                resolved = validate_path_within(candidate, workspace)
                relative = resolved.relative_to(workspace).as_posix()
            except Exception:
                continue
            legacy = next(
                (
                    item for item in legacy_records
                    if str(getattr(item, "relative_path", "") or "") == relative
                    and str(getattr(item, "lineage_id", None) or "main")
                    == lineage_id
                ),
                None,
            )
            if legacy is None:
                continue
            message_render_digest = str(raw.get("render_sha256") or "").strip()
            if (
                message_render_digest
                and message_render_digest
                != str(getattr(legacy, "render_sha256", "") or "")
            ):
                continue
            if kind == "pdf":
                message_pdf_digest = str(raw.get("pdf_sha256") or "").strip()
                if (
                    message_pdf_digest
                    and message_pdf_digest
                    != str(getattr(legacy, "pdf_sha256", "") or "")
                ):
                    continue
            return legacy
    return None


def _immutable_download_response(
    payload: bytes,
    source_path: Path,
    *,
    media_type: str,
    filename: str,
) -> Response:
    """Build a delayed-response-safe download from already verified bytes."""
    safe_filename = os.path.basename(str(filename)).replace('"', "_")
    response = Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"'
        },
    )
    # Preserve the introspection attribute that FileResponse exposed to direct
    # route callers while ensuring ASGI delivery never reopens this path.
    response.path = os.fspath(source_path)
    return response


def _prepare_publication_download_sync(
    session_id: str,
    state: AnalysisState,
    figure_id: str,
    workspace: Path,
    target_lineage: str,
    *,
    kind: str,
) -> tuple[bytes, Path, str]:
    """Resolve and verify one immutable response body off the event loop."""
    from backend.config import REPORTS_DIR as _REPORTS_DIR, SESSION_DIR as _SESSION_DIR

    allowed_roots = tuple(dict.fromkeys(
        root.resolve()
        for root in (
            Path(workspace),
            _SESSION_DIR / session_id,
            Path(WORKSPACE_DIR) / session_id,
            Path(UPLOAD_DIR) / session_id,
            _REPORTS_DIR / session_id,
        )
    ))

    def path_is_session_owned(candidate: Path) -> bool:
        try:
            resolved_candidate = Path(candidate).resolve()
        except (OSError, RuntimeError):
            return False
        return any(
            resolved_candidate == root or resolved_candidate.is_relative_to(root)
            for root in allowed_roots
        )

    figure = gallery_artifact_by_id(
        state.publication_figures,
        figure_id,
        lineage_id=target_lineage,
    )
    registered_id = any(
        str(getattr(item, "id", "")) == str(figure_id)
        for item in state.publication_figures
    )

    if figure is not None:
        fig_name = figure.name
        resolved_artifact = resolve_verified_archival_artifact(
            figure,
            workspace,
            kind=kind,
            path_authorizer=path_is_session_owned,
        )
        if resolved_artifact is None:
            label = "PDF" if kind == "pdf" else "Image"
            raise HTTPException(
                status_code=404,
                detail=f"{label} not available for this figure",
            )
        payload, artifact_path = resolved_artifact
    else:
        if registered_id:
            raise HTTPException(
                status_code=404,
                detail="Figure is not a current gallery artifact",
            )
        legacy = _legacy_artifact_by_id(state, figure_id, target_lineage)
        if legacy is None:
            legacy = _legacy_artifact_by_integrated_report_id(
                state,
                figure_id,
                workspace,
                target_lineage,
                kind=kind,
            )
        if legacy is None:
            raise HTTPException(status_code=404, detail="Figure not found")
        fig_name = legacy.name
        resolved_artifact = resolve_verified_legacy_artifact(
            legacy, workspace, kind=kind,
        )
        if resolved_artifact is None:
            label = "PDF" if kind == "pdf" else "Image"
            raise HTTPException(
                status_code=404,
                detail=(
                    f"{label} file is missing or failed frozen-inventory "
                    "validation"
                ),
            )
        payload, artifact_path = resolved_artifact

    if not path_is_session_owned(artifact_path):
        raise HTTPException(
            status_code=403,
            detail="Access denied: artifact does not belong to this session",
        )

    return payload, artifact_path, fig_name


def _prepare_generation_bound_download_sync(
    session_id: str,
    generation: int,
    state: AnalysisState,
    figure_id: str,
    workspace: Path,
    target_lineage: str,
    kind: str,
) -> tuple[bytes, Path, str]:
    # Validate the generation on both sides of the expensive immutable-byte
    # read.  The lifecycle flock is held only for each check, not for image
    # decoding/hashing; a concurrent delete or replacement makes the second
    # check fail and the already-read bytes are discarded.
    _with_generation_file_barrier(session_id, generation, lambda: None)
    result = _prepare_publication_download_sync(
        session_id,
        state,
        figure_id,
        workspace,
        target_lineage,
        kind=kind,
    )
    _with_generation_file_barrier(session_id, generation, lambda: None)
    return result


async def _prepare_generation_bound_download(
    session_id: str,
    state: AnalysisState,
    figure_id: str,
    workspace: Path,
    target_lineage: str,
    *,
    kind: str,
) -> tuple[bytes, Path, str]:
    generation = int(getattr(state, "lifecycle_generation", 0) or 0)
    try:
        return await run_in_owned_bulk_thread(
            _prepare_generation_bound_download_sync,
            session_id,
            generation,
            state,
            figure_id,
            workspace,
            target_lineage,
            kind,
        )
    except SessionGoneError as exc:
        raise HTTPException(status_code=404, detail="Session no longer exists") from exc
    except SessionGenerationError as exc:
        raise HTTPException(status_code=409, detail="Session generation changed") from exc


@router.get("/{session_id}/graphic-abstract")
async def get_graphic_abstract(session_id: str, request: Request):
    """Get the graphic abstract for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.graphic_abstract:
        return {
            "success": True,
            "graphic_abstract": {
                "id": state.graphic_abstract.id,
                "image_base64": state.graphic_abstract.image_base64,
                "prompt": state.graphic_abstract.prompt,
                "created_at": state.graphic_abstract.created_at.isoformat()
            }
        }

    return {
        "success": False,
        "error": "Graphic abstract not generated yet",
        "graphic_abstract": None
    }

@router.get("/{session_id}/publication-figures")
async def get_publication_figures(session_id: str, request: Request):
    """Get all publication figures for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    figures = []
    approved = 0
    target_lineage = getattr(state, "restore_branch_id", None) or "main"
    for fig in gallery_artifacts(state.publication_figures):
        if not getattr(fig, "is_current", True) or getattr(fig, "invalidated_at", None):
            continue
        if (getattr(fig, "lineage_id", None) or "main") != target_lineage:
            continue
        if not archival_render_available(fig):
            continue
        review_status = fig.review_status.value
        if review_status == "approved":
            approved += 1
        figures.append({
            "id": fig.id,
            "name": fig.name,
            "description": fig.description,
            "figure_type": fig.figure_type,
            "png_path": fig.png_path,
            "pdf_path": fig.pdf_path,
            "png_base64": fig.png_base64,
            "review_status": review_status,
            "pi_feedback": fig.pi_feedback,
            "revision_count": fig.revision_count,
            "plan_step_id": fig.plan_step_id,
            "plan_id": getattr(fig, "plan_id", None),
            "purpose": fig.purpose,
            "logical_figure_id": getattr(fig, "logical_figure_id", None),
            "render_attempt_id": getattr(fig, "render_attempt_id", None),
            "producer_status": getattr(fig, "producer_status", "published"),
            "revision_of_id": getattr(fig, "revision_of_id", None),
            "supersedes_id": getattr(fig, "supersedes_id", None),
            "is_published_head": bool(getattr(fig, "is_published_head", False)),
            "is_publication": True,
            "round": fig.round_number or None,
            "layout_blocking": has_blocking_layout_defects(fig),
            "diagnostic": bool(fig.diagnostic),
            "created_at": fig.created_at.isoformat()
        })

    return {
        "success": True,
        "figures": figures,
        "total": len(figures),
        "approved": approved
    }

@router.get("/{session_id}/publication-tables")
async def get_publication_tables(session_id: str, request: Request):
    """Get all publication tables for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    tables = [
        {
            "id": tbl.id,
            "name": tbl.name,
            "description": tbl.description,
            "csv_path": tbl.csv_path,
            "columns": tbl.columns,
            "preview_data": tbl.preview_data,
            "review_status": tbl.review_status.value,
            "pi_feedback": tbl.pi_feedback,
            "created_at": tbl.created_at.isoformat()
        }
        for tbl in state.publication_tables
    ]

    return {
        "success": True,
        "tables": tables,
        "total": len(tables)
    }

@router.get("/{session_id}/figure/{figure_id}/pdf")
async def download_publication_figure_pdf(session_id: str, figure_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download a specific publication figure as PDF"""
    workspace = get_session_workspace(session_id)
    await _ensure_frozen_artifact_inventory(session_id, state, workspace)
    target_lineage = getattr(state, "restore_branch_id", None) or "main"
    payload, pdf_path, fig_name = await _prepare_generation_bound_download(
        session_id,
        state,
        figure_id,
        workspace,
        target_lineage,
        kind="pdf",
    )
    return _immutable_download_response(
        payload,
        pdf_path,
        media_type="application/pdf",
        filename=f"{fig_name}.pdf",
    )

@router.get("/{session_id}/figure/{figure_id}/png")
async def download_publication_figure_png(session_id: str, figure_id: str, state: AnalysisState = Depends(require_session_ownership)):
    """Download a specific publication figure as PNG/JPG"""
    workspace = get_session_workspace(session_id)
    await _ensure_frozen_artifact_inventory(session_id, state, workspace)
    target_lineage = getattr(state, "restore_branch_id", None) or "main"
    payload, img_path, fig_name = await _prepare_generation_bound_download(
        session_id,
        state,
        figure_id,
        workspace,
        target_lineage,
        kind="image",
    )

    # Determine media type from extension
    ext = img_path.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    dl_ext = ext if ext in (".jpg", ".jpeg") else ".png"
    return _immutable_download_response(
        payload,
        img_path,
        media_type=media_type,
        filename=f"{fig_name}{dl_ext}",
    )

@router.get("/{session_id}/integrated-report")
async def get_integrated_report(session_id: str, request: Request):
    """Get the integrated report for a session"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    if state.integrated_report:
        return {
            "success": True,
            "report": state.integrated_report,
            "figures_count": len(state.publication_figures),
            "tables_count": len(state.publication_tables)
        }

    return {
        "success": False,
        "error": "Integrated report not generated yet",
        "report": None
    }

@router.get("/{session_id}/code")
async def get_code_history(session_id: str, request: Request):
    """Get all executed code for reproducibility"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    code_blocks = []
    for i, ce in enumerate(state.code_history):
        code_blocks.append({
            "step": i + 1,
            "code": ce.code,
            "output": ce.output[:500] if ce.output else None,
            "error": ce.error,
            "execution_time": ce.execution_time
        })

    return {"code_history": code_blocks}

@router.get("/{session_id}/report")
async def get_report(session_id: str, request: Request):
    """Get final analysis report"""
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {
        "session_id": session_id,
        "name": state.name,
        "status": state.status.value,
        "round_results": serialize_round_results(state.round_results),
        "final_summary": state.final_summary
    }

@router.post("/{session_id}/report/partial/approve")
async def approve_partial_report(session_id: str, request: Request):
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    from backend.services.workflow_runner import (
        SessionWorkflowConflictError,
        start_report_generation,
    )

    async def _prepare_partial_report() -> None:
        async def _persist_candidate_and_swap() -> None:
            async with get_lifecycle_lock(session_id):
                assert_generation_writable(session_id, state)
                async with state_lock:
                    if sessions.get(session_id) is not state:
                        raise SessionGenerationError(
                            "session state changed before partial-report approval"
                        )
                    candidate = state.model_copy(deep=True)
                    candidate.allow_partial_report = True
                    candidate.partial_report_requested_at = datetime.now()
                    candidate.status = TaskStatus.RUNNING

                # Publish durable state before exposing the candidate in cache.
                # A failed save leaves the original object completely untouched.
                await _async_save_session_under_lifecycle(
                    session_id, candidate, force=True,
                )
                async with state_lock:
                    if sessions.get(session_id) is not state:
                        raise SessionGenerationError(
                            "session state changed during partial-report approval"
                        )
                    sessions[session_id] = candidate
                    touch_access_time(session_id)

        transaction = asyncio.create_task(_persist_candidate_and_swap())
        await await_with_deferred_cancellation(transaction)

    try:
        await start_report_generation(
            session_id,
            expected_state=state,
            prepare=_prepare_partial_report,
        )
    except SessionWorkflowConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Another workflow, snapshot, or restore operation is active "
                "for this session"
            ),
        ) from exc
    return APIResponse(
        success=True,
        message="Partial report approved",
        data={"needs_reconnect": True},
    )
