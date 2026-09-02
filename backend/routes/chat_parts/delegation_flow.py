"""Phases for _run_chat_delegation."""
from __future__ import annotations

from backend.constants import NO_OUTPUT_TOKEN_CAP

_OWNED_NAMES = frozenset({"_run_chat_delegation"})

def bind_facade(facade) -> None:
    globals().update({name: value for name, value in vars(facade).items() if not name.startswith("__") and name not in _OWNED_NAMES})


_DELEGATION_STOP = object()


async def _execute_delegation_success(
    session_id, state, delegation, user_message, saved_files,
    llm_manager, config, start_time, extra_clients,
):
    import traceback
    from backend.utils.agent_llm import get_agent_llm_client

    delegates = delegation.get("delegates", [])

    # 3. Create message callback that sends to WebSocket + persists
    async def chat_msg_callback(msg: WebSocketMessage):
        # Persist before delivery so cancellation cannot lose history.
        if hasattr(state, 'realtime_messages'):
            append_durable_realtime_message(state, msg)
        await send_analysis_json(session_id, {
                    "type": msg.type,
                    "agent": msg.agent.value if msg.agent else None,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "id": msg.id,
        })

    # 4. Check for sandbox conflicts with running workflow
    workflow_running = session_id in workflows
    if workflow_running:
        # Filter out bioinformatics to avoid sandbox conflict
        delegates = [d for d in delegates if d.get("agent") != "bioinformatics"]
        if not delegates:
            identity = _active_delegation_identity(state)
            terminal = {
                "role": "assistant",
                "content": (
                    "Cannot run bioinformatics analysis while main workflow is active. "
                    "Please wait for the workflow to complete."
                ),
                "pending_figures": False,
                "timestamp": datetime.now().isoformat(),
                **identity,
            }
            _replace_latest_chat_terminal(state, terminal)
            if not await _persist_chat_terminal(session_id, state):
                return _DELEGATION_STOP
            await _send_chat_ws(session_id, "chat_error", {
                "error": "Cannot run bioinformatics analysis while main workflow is active. "
                "Please wait for the workflow to complete.",
                **terminal,
            })
            return _DELEGATION_STOP

    # 5. Group by priority and execute
    priority_groups: Dict[int, List[Dict]] = {}
    for d in delegates:
        p = d.get("priority", 1)
        priority_groups.setdefault(p, []).append(d)

    agent_results: Dict[str, Dict[str, Any]] = {}
    all_figures: List[Dict[str, str]] = []

    for priority in sorted(priority_groups.keys()):
        group = priority_groups[priority]

        if len(group) == 1:
            # Single agent — execute directly
            d = group[0]
            agent_name = d.get("agent", "unknown")
            try:
                agent_client = get_agent_llm_client(state, config, agent_name, session_id)
                if agent_client is not llm_manager.openrouter:
                    extra_clients.append(agent_client)
                result, extra = await _execute_chat_agent(
                    session_id, state, agent_name, d.get("instruction", ""),
                    agent_client, chat_msg_callback, saved_files
                )
                extra_clients.extend(extra)
                agent_results[agent_name] = result
                if "figures" in result:
                    all_figures.extend(result["figures"])
                await _send_chat_ws(session_id, "chat_delegation", {
                    "status": "agent_complete", "agent": agent_name,
                })
            except Exception as e:
                logger.error(f"Chat delegation agent {agent_name} failed: {traceback.format_exc()}")
                agent_results[agent_name] = {"error": str(e), "findings": []}
                await _send_chat_ws(session_id, "chat_delegation", {
                    "status": "agent_error", "agent": agent_name, "error": str(e),
                })
        else:
            # Multiple agents at same priority — execute in parallel
            tasks = []
            for d in group:
                a_name = d.get("agent", "unknown")
                a_client = get_agent_llm_client(state, config, a_name, session_id)
                if a_client is not llm_manager.openrouter:
                    extra_clients.append(a_client)
                tasks.append(_execute_chat_agent(
                    session_id, state, a_name, d.get("instruction", ""),
                    a_client, chat_msg_callback, saved_files
                ))
            # Global timeout to prevent indefinite hang if an agent's LLM call stalls
            from backend.config import CHAT_DELEGATION_TIMEOUT
            try:
                raw_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=CHAT_DELEGATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"Chat delegation timed out after {CHAT_DELEGATION_TIMEOUT}s for session {session_id}")
                # Cancel all tasks and wait for them to finish cleanup
                for task in tasks:
                    if not task.done():
                        task.cancel()
                for task in tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                raw_results = [TimeoutError(f"Delegation timed out after {CHAT_DELEGATION_TIMEOUT}s")] * len(tasks)
            for d, raw_result in zip(group, raw_results):
                agent_name = d.get("agent", "unknown")
                if isinstance(raw_result, Exception):
                    logger.error(f"Chat delegation agent {agent_name} failed: {raw_result}")
                    agent_results[agent_name] = {"error": str(raw_result), "findings": []}
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_error", "agent": agent_name, "error": str(raw_result),
                    })
                else:
                    result, extra = raw_result
                    extra_clients.extend(extra)
                    agent_results[agent_name] = result
                    if "figures" in result:
                        all_figures.extend(result["figures"])
                    await _send_chat_ws(session_id, "chat_delegation", {
                        "status": "agent_complete", "agent": agent_name,
                    })

    # 6. PI Synthesis
    await _send_chat_ws(session_id, "chat_delegation", {
        "status": "synthesizing", "message": "PI synthesizing results...",
    })

    synthesis_prompt = _build_synthesis_prompt(user_message, delegation, agent_results)
    delegation_provider = AgentSettingsProvider(state, config, global_client=llm_manager.openrouter, session_id=session_id)
    pi_model = delegation_provider.get_model(PI_AGENT_NAME)
    pi_temp = delegation_provider.get_temperature(PI_AGENT_NAME)

    dispatch = await delegation_provider.budget_text_dispatch(
        PI_AGENT_NAME,
        [{"role": "user", "content": synthesis_prompt}],
        max_tokens=NO_OUTPUT_TOKEN_CAP,
    )
    pi_response = await llm_manager.openrouter.chat(
        model=pi_model,
        messages=dispatch.messages,
        temperature=pi_temp,
        max_tokens=dispatch.max_tokens,
        prefer_stream=True,
    )
    await delegation_provider.acknowledge_text_dispatch(PI_AGENT_NAME, dispatch)
    final_content = pi_response.content or "Analysis completed but synthesis failed."

    # 7. Post-process: FIGURE and GENERATE_FIGURE markers (reuse existing patterns)
    context, figure_inventory = _build_chat_context(state)
    figure_pattern = re.compile(r'\[\[FIGURE:([\w.-]+)\]\]')
    referenced_names = figure_pattern.findall(final_content)
    synthesis_figures: List[Dict[str, str]] = []

    if referenced_names and figure_inventory:
        fig_lookup = {f["name"]: f for f in figure_inventory}
        for name in dict.fromkeys(referenced_names):
            fig = fig_lookup.get(name)
            if fig:
                entry = {"name": fig["name"], "path": fig["path"]}
                pdf_path = Path(fig["path"]).with_suffix(".pdf")
                if pdf_path.exists():
                    entry["pdfPath"] = str(pdf_path)
                synthesis_figures.append(entry)

    generate_pattern = re.compile(r'\[\[GENERATE_FIGURE:(.*?)\]\]', re.DOTALL)
    generate_requests = generate_pattern.findall(final_content)
    if generate_requests:
        try:
            generated = await _generate_figures_in_chat(session_id, state, generate_requests)
            synthesis_figures.extend(generated)
        except Exception as e:
            logger.exception("Chat delegation figure generation failed")

    # Merge: all_figures from agents + synthesis_figures
    # Deduplicate by name
    seen_names = set()
    merged_figures = []
    for fig in all_figures + synthesis_figures:
        if fig["name"] not in seen_names:
            seen_names.add(fig["name"])
            merged_figures.append(fig)

    # Clean markers from text
    cleaned_content = figure_pattern.sub('', final_content)
    cleaned_content = generate_pattern.sub('', cleaned_content).strip()
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

    # 8. Build final chat message
    assistant_msg: Dict[str, Any] = {
        "role": "assistant",
        "content": cleaned_content,
        "timestamp": datetime.now().isoformat(),
        "pending_figures": False,
    }
    if merged_figures:
        assistant_msg["figures"] = merged_figures

    # Replace placeholder message in chat_messages
    replaced = _replace_latest_delegation_message(state, assistant_msg)
    if not replaced:
        assistant_msg["id"] = str(uuid.uuid4())
        state.chat_messages.append(assistant_msg)

    # Persist the identity-preserving terminal replacement before delivery.
    if not await _persist_chat_terminal(session_id, state):
        return _DELEGATION_STOP

    # 9. Send chat_result via WebSocket
    await _send_chat_ws(session_id, "chat_result", {
        "id": assistant_msg.get("id"),
        "content": cleaned_content,
        "figures": merged_figures,
        "timestamp": datetime.now().isoformat(),
        "reply_to": assistant_msg.get("reply_to"),
        "pending_figures": False,
    })

    # Auto-snapshot after chat delegation completes
    from backend.models.schemas import TaskStatus as _TaskStatus

    pre_snapshot_status = getattr(state, "status", _TaskStatus.COMPLETED)
    if not isinstance(pre_snapshot_status, _TaskStatus):
        try:
            pre_snapshot_status = _TaskStatus(str(pre_snapshot_status))
        except Exception:
            pre_snapshot_status = _TaskStatus.COMPLETED
    await _create_chat_snapshot(session_id, state)

    duration = time.time() - start_time
    logger.info(f"Chat delegation completed for {session_id} in {duration:.1f}s, "
                 f"{len(agent_results)} agents, {len(merged_figures)} figures")
    return pre_snapshot_status


async def _run_chat_delegation(
    session_id: str,
    state: Any,
    delegation: Dict[str, Any],
    user_message: str,
    image_b64_list: List[str],
    saved_files: List[Dict[str, Any]],
):
    """Background task: orchestrate delegated agent work, then PI synthesizes results."""
    import traceback
    from backend.utils.llm_client import LLMManager
    from backend.services.snapshot_v2.errors import RequiredSnapshotBlocked

    user_id = getattr(state, "user_id", "") if state else ""
    config = await resolve_config_for_user(user_id)
    llm_manager = None
    start_time = time.time()
    _delegation_extra_clients: List[Any] = []
    pre_snapshot_status = None
    try:
        # 1. Send started notification
        delegates = delegation.get("delegates", [])
        agent_names = [d.get("agent", "unknown") for d in delegates]
        await _send_chat_ws(session_id, "chat_delegation", {
            "status": "started",
            "plan": delegation.get("plan", ""),
            "agents": agent_names,
        })

        # 2. Create shared LLM client (default, used as fallback)
        llm_manager = LLMManager(
            api_key=config.api.api_key,
            base_url=config.api.llm_base_url,
            proxy_url=config.api.proxy_url,
            session_id=session_id,
            agent_name="chat_delegation",
        )
        outcome = await _execute_delegation_success(
            session_id, state, delegation, user_message, saved_files,
            llm_manager, config, start_time, _delegation_extra_clients,
        )
        if outcome is _DELEGATION_STOP:
            return
        pre_snapshot_status = outcome

    except asyncio.CancelledError:
        logger.info(f"Chat delegation cancelled by user for {session_id}")
        identity = _active_delegation_identity(state)
        _replace_latest_chat_terminal(state, {
            "role": "assistant",
            "content": "Cancelled",
            "timestamp": datetime.now().isoformat(),
            "pending_figures": False,
        })
        if await _persist_chat_terminal(session_id, state):
            try:
                await _send_chat_ws(session_id, "chat_cancelled", {
                    "message": "Cancelled by user",
                    "content": "Cancelled",
                    "pending_figures": False,
                    **identity,
                })
            except Exception:
                logger.warning(
                    f"[{session_id}] Failed to send cancellation terminal",
                    exc_info=True,
                )
        else:
            logger.info(
                "[%s] Skipping stale cancellation terminal delivery", session_id,
            )
        raise

    except RequiredSnapshotBlocked as e:
        # Coherent HITL: structured snapshot_failure, not generic chat_error.
        from backend.models.schemas import TaskStatus
        from backend.services.snapshot_v2.failure_gate import (
            enter_chat_snapshot_failure_gate,
        )

        logger.warning(
            f"[{session_id}] Post-chat snapshot blocked after delegation: {e}"
        )
        previous_status = pre_snapshot_status
        if previous_status is None:
            previous_status = getattr(state, "status", TaskStatus.COMPLETED)
        if not isinstance(previous_status, TaskStatus):
            try:
                previous_status = TaskStatus(str(previous_status))
            except Exception:
                previous_status = TaskStatus.COMPLETED
        # Always set chat retry_context (service may have written a bare failure).
        enter_chat_snapshot_failure_gate(state, e, previous_status)
        identity = _active_delegation_identity(state)
        terminal_message = (
            "Chat completed, but a required snapshot failed. "
            "Use Retry, Manage, or Terminate."
        )
        _replace_latest_chat_terminal(state, {
            "role": "assistant",
            "content": terminal_message,
            "timestamp": datetime.now().isoformat(),
            "snapshot_failure": state.snapshot_failure,
            "pending_figures": False,
        })
        if await _persist_chat_terminal(session_id, state):
            await _send_chat_ws(session_id, "state_sync", {
                "change_type": "snapshot_failure",
                "status": getattr(state.status, "value", state.status),
                "wait_input_reason": "snapshot_failure",
                "snapshot_failure": state.snapshot_failure,
                "chat_message": {
                    "id": identity.get("id"),
                    "reply_to": identity.get("reply_to"),
                    "role": "assistant",
                    "content": terminal_message,
                    "pending_figures": False,
                },
            })

    except Exception as e:
        logger.error(f"Chat delegation error for {session_id}: {traceback.format_exc()}")
        error_msg = f"Analysis failed: {str(e)}"
        identity = _active_delegation_identity(state)
        # Update placeholder message
        _replace_latest_chat_terminal(state, {
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now().isoformat(),
            "pending_figures": False,
        })
        if await _persist_chat_terminal(session_id, state):
            await _send_chat_ws(session_id, "chat_error", {
                "error": error_msg,
                "content": error_msg,
                "pending_figures": False,
                **identity,
            })

    finally:
        active_sandbox = chat_sandboxes.get(session_id)
        if active_sandbox is not None and getattr(active_sandbox, "km", None) is None:
            chat_sandboxes.pop(session_id, None)
        if llm_manager:
            try:
                await llm_manager.close()
            except Exception:
                logger.warning("Failed to close LLM manager", exc_info=True)
        for extra_client in _delegation_extra_clients:
            try:
                await extra_client.close()
            except Exception:
                logger.debug("Failed to close extra LLM client", exc_info=True)
