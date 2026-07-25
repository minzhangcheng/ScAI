# Chat HTTP endpoints.
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

from backend.config import get_session_workspace, get_session_reports, get_session_uploads, get_session_states, PI_AGENT_NAME
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

from .core import _build_chat_context, router

from .delegation import _run_chat_delegation, _try_parse_delegation
from .figures import _generate_figures_in_chat
from .snapshots import _create_chat_snapshot


logger = logging.getLogger(__name__)


@router.post("/{session_id}/chat")
async def chat_with_pi(
    request: Request,
    session_id: str,
    message: str = Form(...),
    target_agents: str = Form(default="[]"),
    file_groups: str = Form(default="[]"),
    file_samples: str = Form(default="[]"),
    file_types: str = Form(default="[]"),
    files: List[UploadFile] = File(default=[]),
):
    """Chat with PI Agent using analysis context. Supports file uploads with per-file agent targeting."""
    from backend.utils.agent_llm import get_agent_llm_client
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked
    import base64

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

    # --- Save uploaded files ---
    saved_files: List[Dict[str, str]] = []
    image_b64_list: List[str] = []
    data_file_contents: List[str] = []
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}

    chat_upload_dir = get_session_uploads(session_id) / "chat"
    chat_upload_dir.mkdir(parents=True, exist_ok=True)

    for idx, upload_file in enumerate(files):
        fname = safe_filename(upload_file.filename or "file")
        if not fname:
            continue
        dest = chat_upload_dir / fname
        # Deduplicate filenames
        counter = 1
        while dest.exists():
            stem = Path(fname).stem
            suffix = Path(fname).suffix
            dest = chat_upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        await stream_save_file(upload_file, dest)
        ext = dest.suffix.lower()
        is_image = ext in IMAGE_EXTS
        file_type = "image" if is_image else "file"
        file_agent = per_file_agents[idx] if idx < len(per_file_agents) else PI_AGENT_NAME

        file_group = per_file_groups[idx] if idx < len(per_file_groups) else None
        file_sample = per_file_samples[idx] if idx < len(per_file_samples) else None
        file_dtype = per_file_types[idx] if idx < len(per_file_types) else None

        saved_files.append({
            "filename": dest.name,
            "type": file_dtype or file_type,
            "path": str(dest),
            "target_agent": file_agent,
            "group_id": file_group,
            "sample_number": file_sample,
        })

        # Load image as base64 for vision API
        if is_image:
            with open(dest, "rb") as f:
                image_b64_list.append(base64.b64encode(f.read()).decode("utf-8"))

        # Read data file content (first 5000 chars) for text inclusion
        if not is_image and ext in {'.csv', '.tsv', '.txt', '.json'}:
            try:
                with open(dest, "r", encoding="utf-8", errors="replace") as f:
                    content_preview = f.read(5000)
                data_file_contents.append(f"--- {dest.name} ---\n{content_preview}")
            except Exception as e:
                logger.warning(f"Failed to read file preview for {dest}: {e}", exc_info=True)

    # --- Register files to target agent state (per-file) ---
    _agent_section_map = {"pathology": "pathology", "bioinformatics": "sequencing", "biochemistry": "biochemistry"}
    for sf in saved_files:
        agent = sf["target_agent"]
        section = _agent_section_map.get(agent)
        if section:
            state.data_files.append(DataFile(
                filename=sf["filename"],
                file_path=sf["path"],
                section=section,
                data_type=sf.get("type") or "uploaded_via_chat",
                group_id=sf.get("group_id"),
                sample_number=sf.get("sample_number"),
                description="Uploaded via chat",
            ))

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
    for msg in state.chat_messages[-20:]:
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

    # Store user message
    state.chat_messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat(),
        "files": saved_files,
    })

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

        # Concurrency guard
        async with chat_task_lock:
            if session_id in chat_tasks and not chat_tasks[session_id].done():
                raise HTTPException(status_code=409, detail="Chat analysis already in progress for this session")

        # Add placeholder assistant message
        state.chat_messages.append({
            "role": "assistant",
            "content": f"Working on it: {delegation.get('plan', 'Coordinating agents...')}",
            "timestamp": datetime.now().isoformat(),
            "delegation_active": True,
        })
        await async_save_session(session_id, state)

        # Launch background task
        task = asyncio.create_task(
            _run_chat_delegation(
                session_id, state, delegation, user_content,
                image_b64_list, saved_files
            )
        )
        async with chat_task_lock:
            chat_tasks[session_id] = task

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
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
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
                response_figures.extend(generated)
                await async_save_session(session_id, state)
        except Exception as e:
            logger.exception("Chat figure generation failed")

    result: Dict[str, Any] = {"role": "assistant", "content": cleaned_content}
    if assistant_msg.get("figures"):
        result["figures"] = assistant_msg["figures"]

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
        await async_save_session(session_id, state)
        result["snapshot_failure"] = state.snapshot_failure
        result["wait_input_reason"] = "snapshot_failure"
        result["status"] = "snapshot_failure"
        return result

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

    return {"messages": state.chat_messages}
