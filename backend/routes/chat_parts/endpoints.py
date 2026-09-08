# Chat HTTP endpoints.
# Moved verbatim out of backend/routes/chat.py.
"""Chat routes for PI Agent free-form chat with analysis sessions."""
import asyncio
import base64
import io
import json
import logging
import os
import re
import stat
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from backend.config import get_session_workspace, get_session_reports, get_session_uploads, get_session_states, PI_AGENT_NAME
from backend.models.schemas import WebSocketMessage, AgentType, DataFile, AnalysisState
from backend.services.session_lifecycle import (
    assert_generation_writable,
    get_lifecycle_lock,
)
from backend.services.session_store import (
    _async_save_session_under_lifecycle,
    ensure_session_loaded,
    async_save_session,
    update_session_cache,
)
from backend.state import (
    chat_tasks, chat_task_lock, chat_sandboxes,
    websockets, workflows, state_lock,
)
from backend.utils.config_resolver import resolve_config_for_user
from backend.utils.auth import require_session_ownership
from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    run_in_owned_thread,
)
from backend.utils.files import (
    safe_filename,
    validate_path_within,
    validate_session_id,
)
from backend.routes.upload import (
    _cleanup_upload_transaction_safely,
    _publish_upload_staging,
    _stream_generic_upload,
    _upload_staging_path,
)
from backend.utils.settings_provider import AgentSettingsProvider
from backend.utils.sandbox import _TPL_NOOP_INSTALL
from backend.utils.direction_payload import (
    build_direction_projection_block,
    resolve_followup_reference,
)

from .core import _build_chat_context, router

from .delegation import (
    _persist_chat_terminal,
    _run_chat_delegation,
    _try_parse_delegation,
)
from .figures import _generate_figures_in_chat


import sys
from . import endpoint_flow


logger = logging.getLogger(__name__)

_CHAT_UPLOAD_MAX_FILES = 10
_CHAT_UPLOAD_MAX_FILE_BYTES = 20 * 1024 * 1024
_CHAT_UPLOAD_MAX_REQUEST_BYTES = 50 * 1024 * 1024
_CHAT_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".svg",
})
_CHAT_TEXT_PREVIEW_EXTENSIONS = frozenset({".csv", ".tsv", ".txt", ".json"})
_CHAT_TEXT_PREVIEW_BYTES = 20 * 1024
_CHAT_TEXT_PREVIEW_CHARS = 5000
_CHAT_MESSAGE_ID_NAMESPACE = uuid.UUID("13d6ca86-81cc-4e56-8e03-98de44d90a04")


def _new_chat_message_id(candidate: str = "") -> str:
    """Return a canonical client UUID or allocate a server-owned message ID."""
    try:
        candidate = (candidate or "").strip()
        uuid.UUID(candidate)
        return candidate
    except (AttributeError, TypeError, ValueError):
        return str(uuid.uuid4())


def _chat_history_messages(
    session_id: str,
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Serialize chat history with stable IDs without mutating legacy records."""
    serialized: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        item = dict(message)
        if not item.get("id"):
            legacy_key = "|".join((
                session_id,
                str(index),
                str(item.get("timestamp", "")),
                str(item.get("role", "")),
                str(item.get("content", "")),
            ))
            item["id"] = str(uuid.uuid5(_CHAT_MESSAGE_ID_NAMESPACE, legacy_key))
        serialized.append(item)
    return serialized


def _chat_upload_error(detail: str) -> HTTPException:
    return HTTPException(status_code=413, detail=detail)


def _preflight_chat_upload_limits(files: List[UploadFile]) -> None:
    if len(files) > _CHAT_UPLOAD_MAX_FILES:
        raise _chat_upload_error(
            f"A chat message may attach at most {_CHAT_UPLOAD_MAX_FILES} files"
        )
    declared_total = 0
    for upload_file in files:
        declared_size = getattr(upload_file, "size", None)
        if not isinstance(declared_size, int) or declared_size < 0:
            continue
        if declared_size > _CHAT_UPLOAD_MAX_FILE_BYTES:
            raise _chat_upload_error(
                "Chat attachment exceeds the per-file size limit "
                f"({_CHAT_UPLOAD_MAX_FILE_BYTES // (1024 * 1024)}MB)"
            )
        declared_total += declared_size
        if declared_total > _CHAT_UPLOAD_MAX_REQUEST_BYTES:
            raise _chat_upload_error(
                "Chat attachments exceed the request size limit "
                f"({_CHAT_UPLOAD_MAX_REQUEST_BYTES // (1024 * 1024)}MB)"
            )


def _next_chat_upload_path(
    upload_dir: Path,
    filename: str,
    reserved_names: set[str],
) -> Path:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for counter in range(10_001):
        candidate_name = filename if counter == 0 else f"{stem}_{counter}{suffix}"
        if candidate_name in reserved_names:
            continue
        candidate = upload_dir / candidate_name
        if os.path.lexists(candidate):
            continue
        validate_path_within(candidate, upload_dir)
        reserved_names.add(candidate_name)
        return candidate
    raise HTTPException(status_code=409, detail="Could not allocate an upload filename")


def _open_chat_attachment(path: Path) -> tuple[int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(fd)
        raise OSError(f"chat attachment is not a regular file: {path}")
    return fd, int(metadata.st_size)


def _encode_chat_image(path: Path) -> str:
    """Encode one bounded image without allocating a second full raw copy."""
    fd, size = _open_chat_attachment(path)
    if size > _CHAT_UPLOAD_MAX_FILE_BYTES:
        os.close(fd)
        raise _chat_upload_error("Chat image exceeds the configured size limit")
    output = io.StringIO()
    carry = b""
    try:
        with os.fdopen(fd, "rb") as handle:
            while True:
                chunk = handle.read(192 * 1024)
                if not chunk:
                    break
                chunk = carry + chunk
                complete = len(chunk) - (len(chunk) % 3)
                if complete:
                    output.write(
                        base64.b64encode(chunk[:complete]).decode("ascii")
                    )
                carry = chunk[complete:]
            if carry:
                output.write(base64.b64encode(carry).decode("ascii"))
        return output.getvalue()
    finally:
        output.close()


def _read_chat_text_preview(path: Path) -> str:
    fd, _size = _open_chat_attachment(path)
    with os.fdopen(fd, "rb") as handle:
        payload = handle.read(_CHAT_TEXT_PREVIEW_BYTES)
    return payload.decode("utf-8", errors="replace")[:_CHAT_TEXT_PREVIEW_CHARS]


def _remove_identity(items: list, target: Any) -> None:
    for index, item in enumerate(items):
        if item is target:
            items.pop(index)
            return


async def _commit_chat_uploads(
    session_id: str,
    state: AnalysisState,
    files: List[UploadFile],
    *,
    per_file_agents: list,
    per_file_groups: list,
    per_file_samples: list,
    per_file_types: list,
    message: str,
    message_id: str | None = None,
    followup_id: str = "",
    branch_id: str = "main",
) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Atomically publish one generation-bound group of chat attachments."""
    _preflight_chat_upload_limits(files)
    saved_files: List[Dict[str, Any]] = []
    image_b64_list: List[str] = []
    data_file_contents: List[str] = []
    data_file_records: List[DataFile] = []
    transactions: list[dict[str, Any]] = []
    user_message_record: dict[str, Any] | None = None
    save_committed = False

    async with get_lifecycle_lock(session_id):
        assert_generation_writable(session_id, state)
        session_uploads = get_session_uploads(session_id)
        chat_upload_dir = session_uploads / "chat"
        validate_path_within(chat_upload_dir, session_uploads)
        chat_upload_dir.mkdir(parents=True, exist_ok=True)
        reserved_names: set[str] = set()
        total_size = 0

        try:
            for idx, upload_file in enumerate(files):
                filename = safe_filename(upload_file.filename or "file")
                if not filename:
                    raise HTTPException(status_code=400, detail="Invalid filename")
                final_path = _next_chat_upload_path(
                    chat_upload_dir, filename, reserved_names,
                )
                staging_path = _upload_staging_path(final_path)
                transaction = {
                    "staging": staging_path,
                    "final": final_path,
                    "published": False,
                }
                transactions.append(transaction)

                request_remaining = _CHAT_UPLOAD_MAX_REQUEST_BYTES - total_size
                if request_remaining < 0:
                    raise _chat_upload_error(
                        "Chat attachments exceed the request size limit"
                    )
                stream_limit = min(
                    _CHAT_UPLOAD_MAX_FILE_BYTES,
                    request_remaining,
                )
                try:
                    await _stream_generic_upload(
                        upload_file,
                        staging_path,
                        max_bytes=stream_limit,
                    )
                except HTTPException as exc:
                    if exc.status_code != 413:
                        raise
                    if request_remaining < _CHAT_UPLOAD_MAX_FILE_BYTES:
                        raise _chat_upload_error(
                            "Chat attachments exceed the request size limit "
                            f"({_CHAT_UPLOAD_MAX_REQUEST_BYTES // (1024 * 1024)}MB)"
                        ) from exc
                    raise _chat_upload_error(
                        "Chat attachment exceeds the per-file size limit "
                        f"({_CHAT_UPLOAD_MAX_FILE_BYTES // (1024 * 1024)}MB)"
                    ) from exc

                file_size = staging_path.stat().st_size
                total_size += file_size
                await _publish_upload_staging(staging_path, final_path)
                transaction["published"] = True

                extension = final_path.suffix.lower()
                is_image = extension in _CHAT_IMAGE_EXTENSIONS
                file_type = "image" if is_image else "file"
                file_agent = (
                    per_file_agents[idx]
                    if idx < len(per_file_agents)
                    else PI_AGENT_NAME
                )
                file_group = (
                    per_file_groups[idx]
                    if idx < len(per_file_groups)
                    else None
                )
                file_sample = (
                    per_file_samples[idx]
                    if idx < len(per_file_samples)
                    else None
                )
                file_dtype = (
                    per_file_types[idx]
                    if idx < len(per_file_types)
                    else None
                )
                saved_file = {
                    "filename": final_path.name,
                    "type": file_dtype or file_type,
                    "path": str(final_path),
                    "target_agent": file_agent,
                    "group_id": file_group,
                    "sample_number": file_sample,
                }
                saved_files.append(saved_file)

                if is_image:
                    image_b64_list.append(
                        await run_in_owned_thread(
                            _encode_chat_image, staging_path,
                        )
                    )
                elif extension in _CHAT_TEXT_PREVIEW_EXTENSIONS:
                    try:
                        preview = await run_in_owned_thread(
                            _read_chat_text_preview, staging_path,
                        )
                        data_file_contents.append(
                            f"--- {final_path.name} ---\n{preview}"
                        )
                    except Exception:
                        logger.warning(
                            "Failed to read chat attachment preview for %s",
                            final_path,
                            exc_info=True,
                        )

                section = {
                    "pathology": "pathology",
                    "bioinformatics": "sequencing",
                    "biochemistry": "biochemistry",
                }.get(file_agent)
                if section:
                    data_file_records.append(DataFile(
                        filename=final_path.name,
                        file_path=str(final_path),
                        section=section,
                        data_type=file_dtype or "uploaded_via_chat",
                        group_id=file_group,
                        sample_number=file_sample,
                        description="Uploaded via chat",
                    ))

            state.data_files.extend(data_file_records)
            user_message_record = {
                "id": _new_chat_message_id(message_id or ""),
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "files": saved_files,
            }
            if followup_id:
                user_message_record.update({
                    "followup_id": followup_id,
                    "branch_id": branch_id,
                })
            state.chat_messages.append(user_message_record)

            save_task = asyncio.create_task(
                _async_save_session_under_lifecycle(
                    session_id, state, force=True,
                ),
                name=f"chat-upload-state-save:{session_id}",
            )
            try:
                await await_with_deferred_cancellation(save_task)
                save_committed = True
            except asyncio.CancelledError:
                if save_task.done() and not save_task.cancelled():
                    try:
                        save_task.result()
                    except BaseException:
                        pass
                    else:
                        save_committed = True
                raise

            for transaction in transactions:
                await _cleanup_upload_transaction_safely(
                    transaction["staging"]
                )
        except BaseException:
            if save_committed:
                for transaction in transactions:
                    await _cleanup_upload_transaction_safely(
                        transaction["staging"]
                    )
            else:
                if user_message_record is not None:
                    _remove_identity(state.chat_messages, user_message_record)
                for data_file in data_file_records:
                    _remove_identity(state.data_files, data_file)
                for transaction in reversed(transactions):
                    await _cleanup_upload_transaction_safely(
                        transaction["staging"],
                        transaction["final"]
                        if transaction["published"] else None,
                    )
            raise

    return saved_files, image_b64_list, data_file_contents


async def _run_owned_chat_delegation(
    session_id: str,
    start_event: asyncio.Event,
    state: AnalysisState,
    delegation: Dict[str, Any],
    user_content: str,
    image_b64_list: List[str],
    saved_files: List[Dict[str, Any]],
    followup_id: str,
    branch_id: str,
    source_context: str,
) -> None:
    """Run only after durable publication and retire only this exact owner."""
    owner = asyncio.current_task()
    try:
        await start_event.wait()
        delegation_kwargs = {}
        if followup_id:
            delegation_kwargs = {
                "followup_id": followup_id,
                "branch_id": branch_id,
                "source_context": source_context,
            }
        await _run_chat_delegation(
            session_id,
            state,
            delegation,
            user_content,
            image_b64_list,
            saved_files,
            **delegation_kwargs,
        )
    finally:
        async with chat_task_lock:
            if chat_tasks.get(session_id) is owner:
                chat_tasks.pop(session_id, None)


async def _publish_chat_delegation(
    session_id: str,
    state: AnalysisState,
    delegation: Dict[str, Any],
    user_content: str,
    image_b64_list: List[str],
    saved_files: List[Dict[str, Any]],
    user_message_id: str = "",
    *,
    followup_id: str = "",
    branch_id: str = "main",
    source_context: str = "",
) -> asyncio.Task:
    """Atomically reserve, persist, and release one generation-bound task."""
    start_event = asyncio.Event()
    placeholder = {
        "id": _new_chat_message_id(),
        "role": "assistant",
        "content": (
            "Working on it: "
            f"{delegation.get('plan', 'Coordinating agents...')}"
        ),
        "timestamp": datetime.now().isoformat(),
        "delegation_active": True,
        "reply_to": user_message_id,
    }
    if followup_id:
        placeholder.update({
            "followup_id": followup_id,
            "branch_id": branch_id,
        })
    task: asyncio.Task | None = None
    publication_committed = False

    try:
        async with get_lifecycle_lock(session_id):
            assert_generation_writable(session_id, state)
            async with chat_task_lock:
                existing = chat_tasks.get(session_id)
                if existing is not None and not existing.done():
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Chat analysis already in progress for this session"
                        ),
                    )
                if existing is not None:
                    chat_tasks.pop(session_id, None)

                task = asyncio.create_task(
                    _run_owned_chat_delegation(
                        session_id,
                        start_event,
                        state,
                    delegation,
                    user_content,
                    image_b64_list,
                    saved_files,
                    followup_id,
                    branch_id,
                    source_context,
                ),
                    name=f"chat-delegation:{session_id}",
                )
                chat_tasks[session_id] = task
                state.chat_messages.append(placeholder)

                save_task = asyncio.create_task(
                    _async_save_session_under_lifecycle(
                        session_id, state, force=True,
                    ),
                    name=f"chat-delegation-state-save:{session_id}",
                )
                try:
                    await await_with_deferred_cancellation(save_task)
                    publication_committed = True
                except asyncio.CancelledError:
                    if save_task.done() and not save_task.cancelled():
                        try:
                            save_task.result()
                        except BaseException:
                            pass
                        else:
                            publication_committed = True
                    if publication_committed:
                        start_event.set()
                    else:
                        _remove_identity(state.chat_messages, placeholder)
                        if chat_tasks.get(session_id) is task:
                            chat_tasks.pop(session_id, None)
                        task.cancel()
                    raise
                except BaseException:
                    _remove_identity(state.chat_messages, placeholder)
                    if chat_tasks.get(session_id) is task:
                        chat_tasks.pop(session_id, None)
                    task.cancel()
                    raise

                start_event.set()
    except BaseException:
        if task is not None and not publication_committed:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise

    assert task is not None
    return task


@router.post("/{session_id}/chat")
async def chat_with_pi(
    request: Request,
    session_id: str,
    message: str = Form(...),
    target_agents: str = Form(default='[]'),
    file_groups: str = Form(default='[]'),
    file_samples: str = Form(default='[]'),
    file_types: str = Form(default='[]'),
    message_id: str = Form(default=''),
    files: List[UploadFile] = File(default=[]),
    followup_id: str = Form(default=''),
    branch_id: str = Form(default=''),
):
    """Chat with PI Agent using analysis context. Supports file uploads with per-file agent targeting."""
    endpoint_flow.bind_facade(sys.modules[__name__])
    return await endpoint_flow.chat_with_pi(
        request,
        session_id,
        message,
        target_agents,
        file_groups,
        file_samples,
        file_types,
        message_id,
        files,
        followup_id=followup_id,
        branch_id=branch_id,
    )

@router.post("/{session_id}/chat/cancel")
async def cancel_chat(request: Request, session_id: str):
    """Cancel an in-progress chat request (delegation or simple)."""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    await require_session_ownership(session_id=session_id, user_id=user_id)

    cancelled = False

    # 1. Cancel delegation task if running
    async with chat_task_lock:
        task = chat_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            cancelled = True

    # 2. Interrupt sandbox kernel if running
    sandbox = chat_sandboxes.get(session_id)
    if sandbox:
        try:
            if sandbox.km:
                sandbox.km.interrupt_kernel()
                logger.info(f"Chat cancel: interrupted sandbox kernel for {session_id}")
        except Exception as e:
            logger.warning(f"Chat cancel: sandbox interrupt failed for {session_id}: {e}")

    return {"status": "cancelled", "had_delegation": cancelled}

@router.get("/{session_id}/chat")
async def get_chat_history(request: Request, session_id: str):
    """Get chat history for a session"""
    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)

    return {"messages": _chat_history_messages(session_id, state.chat_messages)}
