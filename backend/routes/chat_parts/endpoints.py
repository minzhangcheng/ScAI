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

from .core import _build_chat_context, router

from .delegation import (
    _persist_chat_terminal,
    _run_chat_delegation,
    _try_parse_delegation,
)
from .figures import _generate_figures_in_chat
from .snapshots import _create_chat_snapshot


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
) -> None:
    """Run only after durable publication and retire only this exact owner."""
    owner = asyncio.current_task()
    try:
        await start_event.wait()
        await _run_chat_delegation(
            session_id,
            state,
            delegation,
            user_content,
            image_b64_list,
            saved_files,
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
    target_agents: str = Form(default="[]"),
    file_groups: str = Form(default="[]"),
    file_samples: str = Form(default="[]"),
    file_types: str = Form(default="[]"),
    message_id: str = Form(default=""),
    files: List[UploadFile] = File(default=[]),
):
    """Chat with PI Agent using analysis context. Supports file uploads with per-file agent targeting."""
    from backend.utils.agent_llm import get_agent_llm_client
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

    validate_session_id(session_id)
    user_id = getattr(request.state, "user_id", "")
    state = await require_session_ownership(session_id=session_id, user_id=user_id)
    config = await resolve_config_for_user(user_id)

    if config is None:
        raise HTTPException(status_code=400, detail="No API key configured. Please configure your LLM API key in Settings before sending a message.")

    # Reject chat while a snapshot/restore safe point is active
    from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
    _sp = get_snapshot_safe_point_manager()
    if _sp.is_paused(session_id) or _sp.has_pending_request(session_id):
        raise HTTPException(
            status_code=409,
            detail="Cannot chat while a snapshot or restore operation is in progress",
        )

    # Parse per-file metadata arrays
    def _parse_json_list(raw: str) -> list:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    per_file_agents = _parse_json_list(target_agents)
    per_file_groups = _parse_json_list(file_groups)
    per_file_samples = _parse_json_list(file_samples)
    per_file_types = _parse_json_list(file_types)
    user_message_id = _new_chat_message_id(message_id)

    # --- Commit uploaded files and their state projection as one generation ---
    if files:
        saved_files, image_b64_list, data_file_contents = (
            await _commit_chat_uploads(
                session_id,
                state,
                files,
                per_file_agents=per_file_agents,
                per_file_groups=per_file_groups,
                per_file_samples=per_file_samples,
                per_file_types=per_file_types,
                message=message,
                message_id=user_message_id,
            )
        )
    else:
        saved_files = []
        image_b64_list = []
        data_file_contents = []
        state.chat_messages.append({
            "id": user_message_id,
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "files": [],
        })

    # --- Build context & prompt ---
    context, figure_inventory = _build_chat_context(state)
    _custom_models = state.custom_models or {}
    _custom_temps = state.custom_temperatures or {}
    pi_model = _custom_models.get(PI_AGENT_NAME) or getattr(config.models, f"{PI_AGENT_NAME}_model", "")
    pi_temp = _custom_temps.get(PI_AGENT_NAME) if _custom_temps.get(PI_AGENT_NAME) is not None else getattr(config.temperatures, PI_AGENT_NAME, 0.5)

    system_prompt = (
        "You are the PI (Principal Investigator) Agent of a biomedical research analysis system. "
        "You have access to the current analysis state and can help the user understand results, "
        "answer questions about the data, and provide scientific insights.\n\n"
        "Current Analysis Context:\n" + context + "\n\n"
        "Instructions:\n"
        "- Answer the user's questions based on available analysis results\n"
        "- If relevant data exists in the analysis, reference it specifically\n"
        "- Be concise but thorough in scientific explanations\n"
        "- If you don't have enough information, say so clearly\n"
        "- You can suggest next steps or additional analyses if appropriate\n"
        "\n"
        "FIGURE DISPLAY RULES (CRITICAL — you MUST follow these):\n"
        "You have a built-in figure rendering system. You MUST use it. "
        "NEVER provide Python/R code for the user to run. NEVER tell the user to 'run this code'. "
        "The user CANNOT execute code — only you can display figures through these commands:\n\n"
        "1. To show an EXISTING figure from the analysis, write on its own line:\n"
        "   [[FIGURE:figure_name]]\n"
        "   Use exact names from the 'Available Figures' list in the context above.\n"
        "   Example: [[FIGURE:umap_cell_types]]\n\n"
        "2. To CREATE a NEW figure that doesn't exist, write on its own line:\n"
        "   [[GENERATE_FIGURE:detailed description]]\n"
        "   The system will automatically generate and display the figure.\n"
        "   Include specific details: what plot type, which genes/clusters, colors, labels, etc.\n"
        "   Example: [[GENERATE_FIGURE:UMAP plot colored by cell_type annotation with legend on the right margin, "
        "using tab20 palette, title 'Cell Type Clusters', frameon=False]]\n\n"
        "When the user asks to see, show, display, generate, or create any figure/plot/chart/image, "
        "you MUST use [[FIGURE:...]] or [[GENERATE_FIGURE:...]] — NEVER give code.\n"
        "\n"
        "- Always respond in English\n"
        "\n"
        "DELEGATION CAPABILITY:\n"
        "When the user's request requires running new computational analysis, processing new data, "
        "or using specialized agent capabilities beyond your own knowledge, you can delegate tasks "
        "to other agents. Available agents:\n"
        "- bioinformatics: Code execution, new plots, scRNA-seq analysis, clustering, DE analysis, trajectory\n"
        "- pathology: Pathology/histology image analysis or re-examination\n"
        "- biochemistry: Biochemistry data interpretation (Western blot, ELISA, etc.)\n"
        "- literature: Literature search, evidence gathering, Deep Research\n\n"
        "When delegation is needed, respond with ONLY a JSON block (no other text):\n"
        '{"delegation": true, "plan": "Brief description of what you will coordinate", "delegates": [\n'
        '  {"agent": "bioinformatics", "instruction": "Detailed task instruction with data paths and specifics", "priority": 1}\n'
        '], "synthesis_instruction": "How to summarize results for user"}\n\n'
        "Delegation rules:\n"
        "- Same priority number = parallel execution; lower number executes first\n"
        "- For simple questions answerable from existing results, respond normally (NO delegation)\n"
        "- For showing existing figures, use [[FIGURE:...]] (NO delegation)\n"
        "- Only delegate when NEW analysis, computation, or specialized processing is genuinely needed\n"
        "- Include specific data file paths and analysis requirements in the instruction field\n"
        "- The bioinformatics agent's notebook ALREADY has all previous analysis results loaded (adata with "
        "cell_type, clustering, PCA, UMAP, DE results, etc.). Do NOT include prerequisite steps like "
        "'first ensure cell annotations are available' or 'run clustering first' in the instruction. "
        "Focus ONLY on the NEW analysis the user is asking for. The agent will check existing results automatically.\n"
    )

    # Build messages: system + history + new message
    chat_messages_list = [{"role": "system", "content": system_prompt}]
    for msg in state.chat_messages[-21:-1]:
        chat_messages_list.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Build user content with file info
    user_content = message
    if saved_files:
        # Group files by target agent for readable description
        agent_files: Dict[str, List[str]] = {}
        for sf in saved_files:
            agent_files.setdefault(sf["target_agent"], []).append(sf["filename"])
        file_parts = []
        for agent, fnames in agent_files.items():
            label = agent.upper() if agent != PI_AGENT_NAME else "PI"
            file_parts.append(f"{label} Agent: {', '.join(fnames)}")
        user_content += f"\n\n[Attached {len(saved_files)} file(s) — {'; '.join(file_parts)}]"
    if data_file_contents:
        user_content += "\n\n" + "\n\n".join(data_file_contents)

    chat_messages_list.append({"role": "user", "content": user_content})

    # --- Call LLM ---
    try:
        llm_client = get_agent_llm_client(state, config, PI_AGENT_NAME, session_id)

        if image_b64_list:
            # Vision API for images
            full_prompt = system_prompt + "\n\nChat History:\n"
            for msg in state.chat_messages[-21:-1]:  # exclude the just-added user msg
                full_prompt += f"\n{msg.get('role', 'user')}: {msg.get('content', '')}"
            full_prompt += f"\n\nuser: {user_content}"

            response = await llm_client.chat_with_vision(
                model=pi_model,
                text_prompt=full_prompt,
                images=image_b64_list,
                temperature=pi_temp,
                max_tokens=32768
            )
        else:
            response = await llm_client.chat(
                model=pi_model,
                messages=chat_messages_list,
                temperature=pi_temp,
                max_tokens=32768
            )

        assistant_content = response.content if response.content else "I'm sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        logger.exception("Chat LLM error")
        assistant_content = f"Error communicating with LLM: {str(e)}"

    # --- Check for delegation ---
    delegation = _try_parse_delegation(assistant_content)
    if delegation:
        logger.info(f"Chat delegation detected for {session_id}: {delegation.get('plan', '')}")

        await _publish_chat_delegation(
            session_id,
            state,
            delegation,
            user_content,
            image_b64_list,
            saved_files,
            user_message_id,
        )

        return {"status": "processing", "plan": delegation.get("plan", "")}

    # --- Post-process: extract figure references ---
    figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
    referenced_names = figure_pattern.findall(assistant_content)
    logger.info(f"Chat post-process: found {len(referenced_names)} FIGURE refs, {len(figure_inventory)} in inventory")

    response_figures: List[Dict[str, str]] = []
    if referenced_names and figure_inventory:
        fig_lookup = {f["name"]: f for f in figure_inventory}
        for name in dict.fromkeys(referenced_names):  # deduplicate, preserve order
            fig = fig_lookup.get(name)
            if fig:
                entry = {"name": fig["name"], "path": fig["path"]}
                # Check for corresponding PDF
                pdf_path = Path(fig["path"]).with_suffix(".pdf")
                if pdf_path.exists():
                    entry["pdfPath"] = str(pdf_path)
                response_figures.append(entry)
        logger.info(f"Chat post-process: matched {len(response_figures)} existing figures")

    # --- Post-process: extract GENERATE_FIGURE requests ---
    generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
    generate_requests = generate_pattern.findall(assistant_content)
    logger.info(f"Chat post-process: found {len(generate_requests)} GENERATE_FIGURE requests")

    # Clean markers from text
    cleaned_content = figure_pattern.sub('', assistant_content)
    cleaned_content = generate_pattern.sub('', cleaned_content).strip()
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

    # Save text response IMMEDIATELY (before figure generation which can take minutes)
    assistant_msg: Dict[str, Any] = {
        "id": _new_chat_message_id(),
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
        "reply_to": user_message_id,
        "pending_figures": bool(generate_requests),
    }
    if response_figures:
        assistant_msg["figures"] = list(response_figures)
    state.chat_messages.append(assistant_msg)
    await async_save_session(session_id, state)
    logger.info(f"Chat response: text saved ({len(cleaned_content)} chars, {len(response_figures)} existing figures)")

    # Now try to generate new figures (may take long, but text is already saved)
    if generate_requests:
        try:
            generated = await _generate_figures_in_chat(session_id, state, generate_requests)
            logger.info(f"Chat post-process: generated {len(generated)} figures")
            if generated:
                existing = assistant_msg.get("figures", [])
                assistant_msg["figures"] = existing + generated
        except Exception as e:
            logger.exception("Chat figure generation failed")
        finally:
            assistant_msg["pending_figures"] = False
            if not await _persist_chat_terminal(session_id, state):
                logger.info(
                    "[%s] Skipping stale final figure completion save",
                    session_id,
                )

    result: Dict[str, Any] = dict(assistant_msg)

    # Auto-snapshot after chat exchange (fail closed → coherent HITL)
    from backend.models.schemas import TaskStatus as _TaskStatus

    pre_snapshot_status = getattr(state, "status", _TaskStatus.COMPLETED)
    if not isinstance(pre_snapshot_status, _TaskStatus):
        try:
            pre_snapshot_status = _TaskStatus(str(pre_snapshot_status))
        except Exception:
            pre_snapshot_status = _TaskStatus.COMPLETED
    try:
        await _create_chat_snapshot(session_id, state)
        await async_save_session(session_id, state)
    except RequiredSnapshotBlocked as e:
        from backend.services.snapshot_v2.failure_gate import (
            enter_chat_snapshot_failure_gate,
        )

        logger.warning(f"[{session_id}] Post-chat snapshot blocked: {e}")
        # Always set chat retry_context (service may have written a bare failure).
        enter_chat_snapshot_failure_gate(state, e, pre_snapshot_status)
        if not await _persist_chat_terminal(session_id, state):
            raise HTTPException(
                status_code=409,
                detail="Chat session generation changed before terminal save",
            )
        result["snapshot_failure"] = state.snapshot_failure
        result["wait_input_reason"] = "snapshot_failure"
        result["status"] = "snapshot_failure"
        return result

    if not await _persist_chat_terminal(session_id, state):
        raise HTTPException(
            status_code=409,
            detail="Chat session generation changed before terminal save",
        )
    return result

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
